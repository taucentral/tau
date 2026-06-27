package plugins

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	tauproto "github.com/coevin/tau/internal/proto"
	"github.com/coevin/tau/internal/tools"
)

// PluginPrefix is the executable-name prefix the spec requires for
// discovery. Files in a plugins directory that do not start with this
// prefix are ignored.
const PluginPrefix = "tau-plugin-"

// PluginFileExt is the Windows executable extension. On non-Windows it
// is empty; on Windows it is appended to PluginPrefix when matching.
var PluginFileExt = windowsExt()

func windowsExt() string {
	if runtime.GOOS == "windows" {
		return ".exe"
	}
	return ""
}

// Manager owns the set of HostClients for a session. It is the public
// boundary the agent runtime and CLI use: callers Discover() once at
// startup, then call Tools() to get a flat list of namespaced tools to
// register with the agent's tool registry, then call Execute() for each
// model-issued tool call.
//
// Manager also coordinates restart policies, collision detection, and
// clean shutdown across all plugins.
type Manager struct {
	// ProjectPluginsDir is the absolute path to <cwd>/plugins/. May be
	// empty if the cwd has no plugins/ dir; Manager still scans
	// GlobalPluginsDir.
	ProjectPluginsDir string
	// GlobalPluginsDir is the absolute path to ~/.config/tau/plugins/.
	// May be empty.
	GlobalPluginsDir string

	// HostServer is the host-side Host service implementation passed
	// to every HostClient. Required.
	HostServer tauproto.HostServer

	// HostServerLogWriter is the writer HostServer uses for plugin
	// log lines. May be nil (defaults to io.Discard via NewHostServer).
	HostServerLogWriter io.Writer

	// DefaultRestartPolicy is applied to plugins that don't override
	// it. Defaults to RestartOnNextCall — the least-surprising policy
	// for a long-lived agent.
	DefaultRestartPolicy RestartPolicy

	// Diagnostics receives one message per significant lifecycle event
	// (collision, shadow, version mismatch, restart, crash). May be nil;
	// the Manager silently drops diagnostics then. Each message is a
	// single line ending in \n.
	Diagnostics chan<- string

	mu      sync.Mutex
	plugins map[string]*managedPlugin // by short name (e.g. "git")
}

// managedPlugin bundles a HostClient with the policy + restart history.
type managedPlugin struct {
	host       *HostClient
	shortName  string                 // "git", "git-status" — what's after PluginPrefix
	namespaced map[string]*PluginTool // namespaced tool name → tool
	policy     RestartPolicy

	// restartWindowStart timestamps the first restart attempt in the
	// current always-policy window. Reset after the window elapses.
	restartWindowStart time.Time
	restartAttempts    int
}

// NewManager constructs a Manager in New state. Discover must be called
// separately. Returns an error if required fields are missing.
func NewManager(projectDir, globalDir string, hostServer tauproto.HostServer) (*Manager, error) {
	if hostServer == nil {
		return nil, errors.New("plugins: NewManager requires a non-nil HostServer")
	}
	return &Manager{
		ProjectPluginsDir:    projectDir,
		GlobalPluginsDir:     globalDir,
		HostServer:           hostServer,
		DefaultRestartPolicy: RestartOnNextCall,
		plugins:              map[string]*managedPlugin{},
	}, nil
}

// DiscoveredPath is one entry returned by Discover's enumeration. The
// Manager uses these to spawn HostClients after applying the
// project-shadows-global rule.
type DiscoveredPath struct {
	ShortName string // e.g. "git"; what follows PluginPrefix
	AbsPath   string // absolute path to the executable
	Source    PluginSource
}

// PluginSource identifies which scan directory a plugin came from.
type PluginSource int

const (
	// PluginSourceGlobal means the plugin came from ~/.config/tau/plugins/.
	PluginSourceGlobal PluginSource = iota
	// PluginSourceProject means the plugin came from <cwd>/plugins/.
	PluginSourceProject
)

// String returns "global" or "project" for diagnostics.
func (s PluginSource) String() string {
	if s == PluginSourceProject {
		return "project"
	}
	return "global"
}

// Discover scans ProjectPluginsDir and GlobalPluginsDir for executable
// files whose names start with PluginPrefix. Project-local shadows
// global: when both directories have a same-named plugin, the project
// one wins and a diagnostic is emitted. Non-matching and non-executable
// files are silently skipped.
//
// Discover is the canonical "is the on-disk layout what the spec
// requires" check. Callers should run it once at startup; the result is
// cached by SpawnAll.
func (m *Manager) Discover() ([]DiscoveredPath, error) {
	seen := map[string]DiscoveredPath{}
	for _, dir := range []struct {
		path   string
		source PluginSource
	}{
		{m.GlobalPluginsDir, PluginSourceGlobal},
		{m.ProjectPluginsDir, PluginSourceProject},
	} {
		if dir.path == "" {
			continue
		}
		entries, err := os.ReadDir(dir.path)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			return nil, fmt.Errorf("plugins: discover %s: %w", dir.path, err)
		}
		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}
			name := entry.Name()
			if !strings.HasPrefix(name, PluginPrefix) {
				continue
			}
			if PluginFileExt != "" && !strings.HasSuffix(name, PluginFileExt) {
				continue
			}
			short := strings.TrimPrefix(name, PluginPrefix)
			short = strings.TrimSuffix(short, PluginFileExt)
			abs := filepath.Join(dir.path, name)
			if !isExecutable(abs) {
				continue
			}
			entry := DiscoveredPath{ShortName: short, AbsPath: abs, Source: dir.source}
			if existing, ok := seen[short]; ok {
				if existing.Source == PluginSourceGlobal && dir.source == PluginSourceProject {
					m.diagnostic("plugin %q shadowed by project-local copy at %s (global was at %s)",
						short, abs, existing.AbsPath)
					seen[short] = entry
				}
				continue
			}
			seen[short] = entry
		}
	}
	out := make([]DiscoveredPath, 0, len(seen))
	for _, d := range seen {
		out = append(out, d)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ShortName < out[j].ShortName })
	return out, nil
}

// SpawnAll discovers plugins and spawns a HostClient for each. Tools
// are loaded (via HostClient.ListTools) so Tools() can return them
// without further RPCs. Returns the number of plugins spawned and the
// first error encountered (subsequent plugins are still attempted — a
// single bad plugin should not poison the rest).
func (m *Manager) SpawnAll(ctx context.Context) (spawned int, firstErr error) {
	paths, err := m.Discover()
	if err != nil {
		return 0, err
	}
	for _, p := range paths {
		if err := m.spawnOne(ctx, p); err != nil {
			if firstErr == nil {
				firstErr = err
			}
			m.diagnostic("plugin %q failed to spawn: %v", p.ShortName, err)
			continue
		}
		spawned++
	}
	return spawned, firstErr
}

// spawnOne creates and registers a HostClient for the discovered plugin.
// It runs Spawn, Handshake, and ListTools; a failure at any step returns
// an error and the partially-constructed HostClient is dropped.
func (m *Manager) spawnOne(ctx context.Context, p DiscoveredPath) error {
	host, err := NewHostClient(HostConfig{
		Name:       p.ShortName,
		Path:       p.AbsPath,
		HostServer: m.HostServer,
		Stderr:     m.HostServerLogWriter,
		Restart:    m.DefaultRestartPolicy,
	})
	if err != nil {
		return err
	}
	if err := host.Spawn(ctx); err != nil {
		return err
	}
	if err := host.Handshake(ctx); err != nil {
		_ = host.Shutdown(ctx)
		return err
	}
	pluginTools, err := host.ListTools(ctx)
	if err != nil {
		_ = host.Shutdown(ctx)
		return err
	}
	namespaced := make(map[string]*PluginTool, len(pluginTools))
	for _, t := range pluginTools {
		namespaced[t.Name()] = t
	}
	mp := &managedPlugin{
		host:       host,
		shortName:  p.ShortName,
		namespaced: namespaced,
		policy:     m.DefaultRestartPolicy,
	}
	m.mu.Lock()
	m.plugins[p.ShortName] = mp
	m.mu.Unlock()

	// Spawn a restart-policy goroutine for this plugin.
	go m.watchPlugin(mp)
	return nil
}

// watchPlugin subscribes to the plugin's crash channel and applies the
// restart policy when a crash signal arrives. Runs forever; the Manager
// terminates by closing the HostClient (Shutdown signals the watcher to
// exit via the HostClient's pumpCrashDetection close).
func (m *Manager) watchPlugin(mp *managedPlugin) {
	ch := mp.host.CrashSignals()
	for range ch {
		m.mu.Lock()
		policy := mp.policy
		state := mp.host.State()
		m.mu.Unlock()
		if state == PluginStateShutdown {
			return
		}
		switch policy {
		case RestartNever:
			m.diagnostic("plugin %q crashed and will not be restarted (policy: never)", mp.shortName)
			return
		case RestartAlways:
			if !m.maybeRestartAlways(mp) {
				m.diagnostic("plugin %q exceeded restart budget; giving up", mp.shortName)
				return
			}
		case RestartOnNextCall:
			// On-next-call intentionally does nothing here. The next
			// Execute will call ensureRunning, which re-spawns.
			m.diagnostic("plugin %q crashed; will restart on next tool call", mp.shortName)
		}
	}
}

// maybeRestartAlways attempts an immediate respawn under the always
// policy. Returns true if the restart succeeded (or the budget window
// rolled over and a fresh attempt is allowed). Returns false if the
// plugin has burned through RestartAlwaysMaxAttempts in the window.
func (m *Manager) maybeRestartAlways(mp *managedPlugin) bool {
	now := time.Now()
	if now.Sub(mp.restartWindowStart) > RestartAlwaysWindow {
		mp.restartWindowStart = now
		mp.restartAttempts = 0
	}
	if mp.restartAttempts >= RestartAlwaysMaxAttempts {
		return false
	}
	mp.restartAttempts++
	ctx, cancel := context.WithTimeout(context.Background(), SpawnDefaultTimeout)
	defer cancel()
	if err := mp.host.Spawn(ctx); err != nil {
		m.diagnostic("plugin %q always-restart failed: %v", mp.shortName, err)
		return false
	}
	if err := mp.host.Handshake(ctx); err != nil {
		_ = mp.host.Shutdown(ctx)
		m.diagnostic("plugin %q always-restart handshake failed: %v", mp.shortName, err)
		return false
	}
	return true
}

// ensureRunning re-spawns the plugin if it has crashed and the policy
// allows. Called from Execute.
func (m *Manager) ensureRunning(ctx context.Context, mp *managedPlugin) error {
	state := mp.host.State()
	if state == PluginStateRunning {
		return nil
	}
	if state == PluginStateShutdown {
		return ErrAlreadyShutdown{Name: mp.shortName}
	}
	if mp.policy == RestartNever {
		return ErrNotRunning{Name: mp.shortName, State: state.String()}
	}
	// RestartOnNextCall or RestartAlways with a crashed subprocess.
	if err := mp.host.Spawn(ctx); err != nil {
		return err
	}
	if err := mp.host.Handshake(ctx); err != nil {
		_ = mp.host.Shutdown(ctx)
		return err
	}
	// Re-fetch the tool list; schema drift surfaces here.
	pluginTools, err := mp.host.ListTools(ctx)
	if err != nil {
		_ = mp.host.Shutdown(ctx)
		return err
	}
	namespaced := make(map[string]*PluginTool, len(pluginTools))
	for _, t := range pluginTools {
		namespaced[t.Name()] = t
	}
	m.mu.Lock()
	mp.namespaced = namespaced
	m.mu.Unlock()
	return nil
}

// Tools returns the namespaced tools across all plugins, sorted by
// name. Tools from the same plugin retain their natural order. Collisions
// across plugins are resolved first-wins; subsequent registrants are
// dropped and a diagnostic is emitted.
//
// The return type is []tools.HeadlessTool so the runtime can merge
// plugin tools onto a HeadlessTool-capable registry without conversion.
// Plugin-dispatched tools are gRPC proxies and never implement the TUI
// rendering hooks, so HeadlessTool is the correct contract.
func (m *Manager) Tools() []tools.HeadlessTool {
	m.mu.Lock()
	defer m.mu.Unlock()
	seen := map[string]bool{}
	var out []tools.HeadlessTool
	// Sort plugins by name for deterministic collision resolution.
	keys := make([]string, 0, len(m.plugins))
	for k := range m.plugins {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, pluginName := range keys {
		mp := m.plugins[pluginName]
		// Tool names within one plugin are unique; sort for determinism.
		toolNames := make([]string, 0, len(mp.namespaced))
		for n := range mp.namespaced {
			toolNames = append(toolNames, n)
		}
		sort.Strings(toolNames)
		for _, toolName := range toolNames {
			if seen[toolName] {
				m.diagnostic("tool %q from plugin %q skipped: name already registered by another plugin",
					toolName, pluginName)
				continue
			}
			seen[toolName] = true
			out = append(out, mp.namespaced[toolName])
		}
	}
	return out
}

// Execute routes a tool call to the plugin that owns its name. Namespaced
// names ("git.status") are decomposed into plugin name + local name; the
// plugin receives the local name (PluginTool.Execute strips the prefix
// before sending the RPC).
//
// When the full name was advertised via ListTools, the cached PluginTool
// is used. When the full name was not advertised but the prefix matches a
// known plugin, Execute still routes the call to that plugin — the plugin
// is responsible for rejecting unknown tool names. This matches the spec
// invariant that the host treats plugin tools uniformly whether or not
// they appear in ListTools (ListTools is informational, not a gate).
// Returns ErrUnknownPluginTool only when no plugin's short name matches
// the prefix.
func (m *Manager) Execute(ctx context.Context, call tools.ToolCall) (tools.ToolResult, error) {
	pluginName, _, ok := splitNamespaced(call.Name)
	if !ok {
		return tools.ToolResult{}, ErrUnknownPluginTool{Name: call.Name}
	}
	m.mu.Lock()
	mp, ok := m.plugins[pluginName]
	m.mu.Unlock()
	if !ok {
		return tools.ToolResult{}, ErrUnknownPluginTool{Name: call.Name}
	}
	if err := m.ensureRunning(ctx, mp); err != nil {
		return tools.NewErrorResult(fmt.Sprintf("%s: plugin unavailable: %v", call.Name, err)), nil
	}
	m.mu.Lock()
	tool, ok := mp.namespaced[call.Name]
	provider := mp.host.liveClientOrError
	m.mu.Unlock()
	if !ok {
		// Name was not advertised but the prefix matches this plugin.
		// Route anyway so plugins can expose tools that ListTools omits
		// (e.g. debug or diagnostic endpoints). The plugin decides if
		// the local name is invalid.
		tool = NewPluginTool(mp.shortName, call.Name, "", "{}", provider)
	}
	return tool.Execute(ctx, call)
}

// splitNamespaced splits "git.status" into ("git", "status", true).
// Returns ok=false when there is no "." or the prefix is empty.
func splitNamespaced(name string) (plugin, local string, ok bool) {
	idx := strings.IndexByte(name, '.')
	if idx <= 0 || idx == len(name)-1 {
		return "", "", false
	}
	return name[:idx], name[idx+1:], true
}

// ErrUnknownPluginTool is returned by Execute when no plugin owns the
// requested tool name. The agent loop surfaces this to the model as a
// tool-call failure.
type ErrUnknownPluginTool struct{ Name string }

func (e ErrUnknownPluginTool) Error() string {
	return fmt.Sprintf("plugins: no plugin owns tool %q", e.Name)
}

// Shutdown terminates every plugin subprocess. Caller passes a context
// whose deadline bounds total shutdown (per-plugin shutdown grace is
// shutdownGrace=5s). Returns the first error encountered; remaining
// plugins are still attempted.
func (m *Manager) Shutdown(ctx context.Context) error {
	m.mu.Lock()
	plugins := make([]*managedPlugin, 0, len(m.plugins))
	for _, mp := range m.plugins {
		plugins = append(plugins, mp)
	}
	m.mu.Unlock()

	var firstErr error
	for _, mp := range plugins {
		if err := mp.host.Shutdown(ctx); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// diagnostic emits a single line to the Diagnostics channel, if one is
// set. Non-blocking; drops on a full channel.
func (m *Manager) diagnostic(format string, args ...any) {
	if m.Diagnostics == nil {
		return
	}
	select {
	case m.Diagnostics <- fmt.Sprintf(format+"\n", args...):
	default:
	}
}

// isExecutable reports whether the path is regular and executable by
// the current user. On Windows it just checks that the file exists and
// is not a directory (the executable bit is not meaningful).
func isExecutable(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	if info.IsDir() {
		return false
	}
	if runtime.GOOS == "windows" {
		return true
	}
	return info.Mode().Perm()&0o111 != 0
}
