package plugins

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
	"time"

	"github.com/hashicorp/go-plugin"

	tauproto "github.com/taucentral/tau/internal/proto"
)

// PluginState reports where in its lifecycle a HostClient's subprocess is.
// The Manager uses this to decide whether to restart, fail the call, or
// proceed.
type PluginState int

const (
	// PluginStateNew means the HostClient has been constructed but Spawn
	// has not been called (or the previous subprocess was killed).
	PluginStateNew PluginState = iota
	// PluginStateRunning means the subprocess is alive and the gRPC
	// channel is established. Handshake may or may not have completed.
	PluginStateRunning
	// PluginStateCrashed means the subprocess exited unexpectedly. The
	// Manager's restart policy decides whether Spawn is called again.
	PluginStateCrashed
	// PluginStateShutdown means Shutdown was called and the subprocess
	// was killed cleanly. The HostClient should not be restarted.
	PluginStateShutdown
)

// String returns a lowercase identifier matching the diagnostic vocabulary.
func (s PluginState) String() string {
	switch s {
	case PluginStateNew:
		return "new"
	case PluginStateRunning:
		return "running"
	case PluginStateCrashed:
		return "crashed"
	case PluginStateShutdown:
		return "shutdown"
	}
	return "unknown"
}

// HostConfig carries the host-side inputs to a HostClient. It is captured
// at construction time and reused across restarts so a recovered plugin
// comes back with the same environment.
type HostConfig struct {
	// Name is the plugin's short identifier (e.g. "git"). Used in
	// diagnostics, tool namespacing ("<name>.<localName>"), and the
	// PluginMap key.
	Name string
	// Path is the absolute path to the plugin executable.
	Path string
	// HostServer is the host-side Host service implementation the plugin
	// will call back to for Log/Notify/GetConfig. Required: plugins
	// legitimately fail when their Host callbacks return "not
	// implemented".
	HostServer tauproto.HostServer
	// Stderr captures the plugin's raw stderr stream. Plugin logs (via
	// the host's Host.Log RPC) and go-plugin's own diagnostics land
	// here. Defaults to io.Discard.
	Stderr io.Writer
	// Restart is the policy applied when the subprocess exits
	// unexpectedly. See RestartPolicy docs for semantics.
	Restart RestartPolicy
	// StartTimeout caps how long HostClient.Spawn waits for go-plugin's
	// handshake (magic cookie negotiation) before giving up. Defaults
	// to SpawnDefaultTimeout.
	StartTimeout time.Duration
}

// SpawnDefaultTimeout is the default HostConfig.StartTimeout. Generous
// because plugin startup may include BPE encoding table loads or schema
// reflection.
const SpawnDefaultTimeout = 15 * time.Second

// RestartPolicy governs how the Manager reacts when a plugin's
// subprocess exits unexpectedly. Matches the spec values exactly.
type RestartPolicy string

const (
	// RestartNever marks the plugin's tools as permanently unavailable
	// after a crash. Subsequent Execute calls fail with ErrNotRunning.
	RestartNever RestartPolicy = "never"
	// RestartOnNextCall leaves the plugin down until a tool of its is
	// invoked; the Manager re-spawns on the next Execute call.
	RestartOnNextCall RestartPolicy = "on-next-call"
	// RestartAlways triggers an immediate respawn when the subprocess
	// exits, capped at RestartAlwaysMaxAttempts in RestartAlwaysWindow.
	RestartAlways RestartPolicy = "always"
)

// RestartAlwaysMaxAttempts caps the respawn attempts under
// RestartAlways within RestartAlwaysWindow. Prevents an infinite loop
// on a panic-at-startup plugin.
const (
	RestartAlwaysMaxAttempts = 5
	RestartAlwaysWindow      = 30 * time.Second
)

// HostClient owns the lifecycle of one plugin subprocess. It is the
// boundary between the Manager (which coordinates discovery, restart,
// and tool registration) and go-plugin's per-process primitives.
//
// HostClient is goroutine-safe: concurrent Execute and ListTools calls
// are serialized by an internal mutex, and crash detection runs in its
// own goroutine.
type HostClient struct {
	cfg HostConfig

	mu      sync.Mutex
	state   PluginState
	client  *plugin.Client
	grpc    tauproto.PluginClient
	schemas []*tauproto.ToolSchema

	// waiter is the lifetime stop channel for the crash-detection
	// goroutine. It is created in NewHostClient and never reassigned;
	// only Shutdown closes it. Keeping it immutable avoids the race
	// between Spawn reassigning it and pumpCrashDetection reading it.
	waiter chan struct{}

	// crashObservers are notified once when the subprocess exits
	// unexpectedly. The Manager adds itself as an observer so restart
	// decisions can happen out-of-band of any in-flight Execute call.
	crashObservers []chan struct{}
}

// NewHostClient constructs a HostClient in PluginStateNew. Spawn must be
// called separately to launch the subprocess.
func NewHostClient(cfg HostConfig) (*HostClient, error) {
	if cfg.Name == "" {
		return nil, errors.New("plugins: HostConfig.Name is required")
	}
	if cfg.Path == "" {
		return nil, errors.New("plugins: HostConfig.Path is required")
	}
	if cfg.HostServer == nil {
		return nil, errors.New("plugins: HostConfig.HostServer is required (use NoopHostServer for tests)")
	}
	if cfg.StartTimeout == 0 {
		cfg.StartTimeout = SpawnDefaultTimeout
	}
	if cfg.Stderr == nil {
		cfg.Stderr = nopWriter{}
	}
	h := &HostClient{
		cfg:    cfg,
		state:  PluginStateNew,
		waiter: make(chan struct{}),
	}
	// Start the crash-detection pump once for the lifetime of this
	// HostClient. It polls state under the mutex; a nil h.client is a
	// no-op. Shutdown closes h.waiter to terminate the goroutine.
	go h.pumpCrashDetection()
	return h, nil
}

// nopWriter is the default Stderr sink when HostConfig.Stderr is nil.
type nopWriter struct{}

func (nopWriter) Write(p []byte) (int, error) { return len(p), nil }

// State returns the current lifecycle state. Safe to call from any
// goroutine; the value may change asynchronously as the subprocess runs.
func (h *HostClient) State() PluginState {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.state
}

// Name returns the plugin's short identifier.
func (h *HostClient) Name() string { return h.cfg.Name }

// Path returns the plugin executable path.
func (h *HostClient) Path() string { return h.cfg.Path }

// Spawn launches the subprocess and establishes the gRPC channel. It is
// idempotent for PluginStateNew and PluginStateCrashed; on
// PluginStateRunning it is a no-op; on PluginStateShutdown it returns
// ErrAlreadyShutdown.
func (h *HostClient) Spawn(ctx context.Context) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	switch h.state {
	case PluginStateRunning:
		return nil
	case PluginStateShutdown:
		return ErrAlreadyShutdown{Name: h.cfg.Name}
	case PluginStateCrashed, PluginStateNew:
		// fallthrough: re-spawn
	}

	// exec.Command (not CommandContext): hashicorp/go-plugin owns the
	// subprocess lifecycle and would be killed unexpectedly if ctx-cancellation
	// SIGKILLed the child mid-RPC. Plugin shutdown is explicit via Shutdown().
	//
	//nolint:noctx // go-plugin owns the subprocess lifecycle, see above
	cmd := exec.Command(h.cfg.Path)
	cmd.Env = append(commandEnv(), fmt.Sprintf("%s=%s", MagicCookieKey, MagicCookieValue))
	client := plugin.NewClient(&plugin.ClientConfig{
		HandshakeConfig:  HandshakeConfig,
		Plugins:          PluginMap,
		Cmd:              cmd,
		AllowedProtocols: []plugin.Protocol{plugin.ProtocolGRPC},
		Stderr:           h.cfg.Stderr,
		StartTimeout:     h.cfg.StartTimeout,
	})

	// spawnOutcome carries the dispensed gRPC client (on success) or the
	// error (on failure) from the goroutine that performs the blocking
	// Client()/Dispense() calls. The parent mutates state under its own
	// lock after receiving the outcome — this avoids the race where the
	// goroutine sets state=Running after the parent has already returned.
	type spawnOutcome struct {
		grpc tauproto.PluginClient
		err  error
	}
	outcomeCh := make(chan spawnOutcome, 1)
	go func() {
		protocol, err := client.Client()
		if err != nil {
			outcomeCh <- spawnOutcome{err: err}
			return
		}
		raw, err := protocol.Dispense(PluginName)
		if err != nil {
			outcomeCh <- spawnOutcome{err: err}
			return
		}
		grpcClient, ok := raw.(tauproto.PluginClient)
		if !ok {
			outcomeCh <- spawnOutcome{err: fmt.Errorf("plugins: %s dispensed %T, want proto.PluginClient", h.cfg.Name, raw)}
			return
		}
		outcomeCh <- spawnOutcome{grpc: grpcClient}
	}()
	select {
	case o := <-outcomeCh:
		if o.err != nil {
			client.Kill()
			return fmt.Errorf("plugins: %s spawn failed: %w", h.cfg.Name, o.err)
		}
		// State mutation happens here under the parent's lock — no race
		// with a goroutine that mutates after sending the signal.
		h.client = client
		h.grpc = o.grpc
		h.state = PluginStateRunning
	case <-ctx.Done():
		client.Kill()
		return ctx.Err()
	}

	// The crash-detection pump was started in NewHostClient; it watches
	// h.client (now set) under the mutex. Nothing else to do here.
	return nil
}

// pumpCrashDetection runs in its own goroutine, polling plugin.Client
// .Exited() at a low cadence. When Exited() flips true unexpectedly,
// transition state and notify observers. Returns when Shutdown is called
// (close of h.waiter).
func (h *HostClient) pumpCrashDetection() {
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-h.waiter:
			return
		case <-ticker.C:
			h.mu.Lock()
			client := h.client
			running := h.state == PluginStateRunning
			h.mu.Unlock()
			if !running || client == nil {
				continue
			}
			if !client.Exited() {
				continue
			}
			// Subprocess died unexpectedly. Notify the Manager and mark
			// the HostClient as crashed.
			h.mu.Lock()
			if h.state != PluginStateRunning {
				h.mu.Unlock()
				continue
			}
			h.state = PluginStateCrashed
			h.client = nil
			h.grpc = nil
			observers := h.crashObservers
			h.mu.Unlock()
			for _, ch := range observers {
				select {
				case ch <- struct{}{}:
				default:
					// Observer hasn't drained the previous signal; drop.
					// The Manager only needs edges, not counts.
				}
			}
		}
	}
}

// CrashSignals returns a channel that receives one signal per unexpected
// subprocess exit. The Manager uses this to apply the restart policy.
// The channel is buffered(1) and never closes; observers should select
// non-blocking or drain in a loop.
func (h *HostClient) CrashSignals() <-chan struct{} {
	ch := make(chan struct{}, 1)
	h.mu.Lock()
	h.crashObservers = append(h.crashObservers, ch)
	h.mu.Unlock()
	return ch
}

// Handshake verifies the plugin's announced protocol version. Must be
// called after Spawn; calling before Spawn returns ErrNotRunning.
func (h *HostClient) Handshake(ctx context.Context) error {
	h.mu.Lock()
	grpc := h.grpc
	state := h.state
	h.mu.Unlock()
	if state != PluginStateRunning || grpc == nil {
		return ErrNotRunning{Name: h.cfg.Name, State: state.String()}
	}
	return Handshake(ctx, h.cfg.Name, grpc)
}

// ListTools calls Plugin.ListTools and caches the result. Subsequent
// calls return the cache without an RPC. The cache is cleared on
// re-spawn so schema drift is detected on restart.
//
// Returns PluginTool instances whose clientProvider points back at this
// HostClient's live gRPC client.
func (h *HostClient) ListTools(ctx context.Context) ([]*PluginTool, error) {
	h.mu.Lock()
	grpc := h.grpc
	state := h.state
	cached := h.schemas
	h.mu.Unlock()
	if state != PluginStateRunning || grpc == nil {
		return nil, ErrNotRunning{Name: h.cfg.Name, State: state.String()}
	}
	if cached == nil {
		stream, err := grpc.ListTools(ctx, &tauproto.Empty{})
		if err != nil {
			return nil, fmt.Errorf("plugins: %s ListTools RPC failed: %w", h.cfg.Name, err)
		}
		var collected []*tauproto.ToolSchema
		for {
			schema, err := stream.Recv()
			if err != nil {
				if err == io.EOF {
					break
				}
				return nil, fmt.Errorf("plugins: %s ListTools stream failed: %w", h.cfg.Name, err)
			}
			collected = append(collected, schema)
		}
		h.mu.Lock()
		h.schemas = collected
		h.mu.Unlock()
		cached = collected
	}
	tools := make([]*PluginTool, 0, len(cached))
	for _, s := range cached {
		tools = append(tools, NewPluginTool(
			h.cfg.Name,
			s.GetName(),
			s.GetDescription(),
			s.GetJsonSchema(),
			h.liveClientOrError,
		))
	}
	return tools, nil
}

// liveClientOrError is the clientProvider closure passed to PluginTool.
// It reads the current gRPC client; if the plugin has crashed or shut
// down, it returns a descriptive error so PluginTool.Execute can surface
// "plugin unavailable" to the model.
func (h *HostClient) liveClientOrError() (tauproto.PluginClient, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.state != PluginStateRunning || h.grpc == nil {
		return nil, ErrNotRunning{Name: h.cfg.Name, State: h.state.String()}
	}
	return h.grpc, nil
}

// Shutdown calls Plugin.Shutdown, waits up to grace for a clean exit,
// then SIGKILLs the subprocess if still alive and transitions to
// PluginStateShutdown. Idempotent; second and later calls are no-ops.
func (h *HostClient) Shutdown(ctx context.Context) error {
	h.mu.Lock()
	if h.state == PluginStateShutdown {
		h.mu.Unlock()
		return nil
	}
	grpc := h.grpc
	client := h.client
	waiter := h.waiter
	h.mu.Unlock()

	// Stop the crash-detection pump so it doesn't observe the
	// intentional exit and fire a misleading crash signal.
	if waiter != nil {
		close(waiter)
	}

	if grpc != nil {
		shutdownCtx, cancel := context.WithTimeout(ctx, shutdownGrace)
		defer cancel()
		_, _ = grpc.Shutdown(shutdownCtx, &tauproto.Empty{})
	}
	if client != nil {
		client.Kill()
	}
	h.mu.Lock()
	h.state = PluginStateShutdown
	h.client = nil
	h.grpc = nil
	h.schemas = nil
	h.mu.Unlock()
	return nil
}

// shutdownGrace caps the time Shutdown waits for a clean subprocess exit
// before forcing Kill. The spec mandates 5 seconds.
const shutdownGrace = 5 * time.Second

// commandEnv returns the current process environment. The plugin
// inherits the host's PATH, HOME, etc.; the magic cookie is appended
// separately in Spawn.
func commandEnv() []string {
	// We deliberately do not surface env to HostConfig: env leakage
	// (e.g. LD_PRELOAD across plugin toolchains) is a footgun. Plugins
	// that need a specific env can shell out via their own Hosts.
	return append([]string{}, os.Environ()...)
}

// ErrAlreadyShutdown is returned by Spawn (and other stateful methods)
// after Shutdown has been called on a HostClient. The client is dead.
type ErrAlreadyShutdown struct{ Name string }

func (e ErrAlreadyShutdown) Error() string {
	return fmt.Sprintf("plugins: %s already shut down", e.Name)
}

// ErrNotRunning is returned when a method requires PluginStateRunning
// but the subprocess is not currently alive. The Manager may interpret
// this as a signal to apply the on-next-call restart policy.
type ErrNotRunning struct {
	Name  string
	State string
}

func (e ErrNotRunning) Error() string {
	return fmt.Sprintf("plugins: %s not running (state=%s)", e.Name, e.State)
}
