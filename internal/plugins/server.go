package plugins

import (
	"context"
	"fmt"
	"io"
	"log"
	"sync"

	"github.com/hashicorp/go-plugin"
	"google.golang.org/grpc"

	tauproto "github.com/coevin/tau/internal/proto"
)

// HostServer implements the proto Host service that the host exposes to
// plugins. Plugins call Host.Log, Host.Notify, and Host.GetConfig to
// surface diagnostics, request user notifications, and read host config.
//
// HostServer is registered on the same gRPC server that go-plugin sets
// up for plugin→host callbacks via the broker. The plugin side dials it
// through the broker-supplied connection.
type HostServer struct {
	tauproto.UnimplementedHostServer

	// LogWriter is the destination for Host.Log calls. Defaults to
	// io.Discard; tests inject io.Discard or a buffer to assert.
	LogWriter io.Writer
	// Logger is structured logger used when LogWriter is nil but
	// Logger is non-nil. If both are nil, Host.Log is silently
	// discarded.
	Logger *log.Logger
	// ConfigSource resolves dotted config keys for Host.GetConfig.
	// Plugins should not read secrets through this RPC; the
	// ConfigSource is responsible for redacting sensitive keys.
	ConfigSource ConfigSource
	// NotifyHandler is invoked per Host.Notify call. NotifyHandler
	// should be non-blocking; long work should be queued off-thread.
	NotifyHandler func(severity, message string)

	mu sync.Mutex
}

// Compile-time interface check.
var _ tauproto.HostServer = (*HostServer)(nil)

// ConfigSource resolves a dotted-path configuration key (e.g.
// "compaction.reserveTokens") to a string value. Returns found=false if
// the key is absent. Implementations are responsible for redacting
// sensitive keys (e.g. API keys, OAuth tokens) — they should return
// found=false rather than leak the value.
type ConfigSource interface {
	// GetConfig returns the value for the dotted key, or found=false
	// if not present.
	GetConfig(key string) (value string, found bool)
}

// ConfigSourceFunc is the func adapter for ConfigSource.
type ConfigSourceFunc func(key string) (string, bool)

// GetConfig delegates to the underlying func.
func (f ConfigSourceFunc) GetConfig(key string) (string, bool) { return f(key) }

// NoopConfigSource returns found=false for every key. Used in tests and
// as a safe default when no real ConfigSource is available.
func NoopConfigSource() ConfigSource {
	return ConfigSourceFunc(func(string) (string, bool) { return "", false })
}

// NewHostServer constructs a HostServer. nil ConfigSource becomes
// NoopConfigSource; nil NotifyHandler is a no-op.
func NewHostServer(logWriter io.Writer, cfg ConfigSource, notify func(string, string)) *HostServer {
	if cfg == nil {
		cfg = NoopConfigSource()
	}
	if notify == nil {
		notify = func(string, string) {}
	}
	if logWriter == nil {
		logWriter = io.Discard
	}
	return &HostServer{
		LogWriter:     logWriter,
		ConfigSource:  cfg,
		NotifyHandler: notify,
	}
}

// Log implements Host.Log. Plugins call this for diagnostic output.
// The level controls prefix only; the host does not rate-limit.
func (s *HostServer) Log(_ context.Context, req *tauproto.LogRequest) (*tauproto.Empty, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.LogWriter != nil {
		fmt.Fprintf(s.LogWriter, "[plugin %s] %s\n", req.GetLevel().String(), req.GetMessage())
	}
	return &tauproto.Empty{}, nil
}

// Notify implements Host.Notify. Routes a user-facing notification to
// the host's event bus via the configured handler.
func (s *HostServer) Notify(_ context.Context, req *tauproto.NotifyRequest) (*tauproto.Empty, error) {
	s.mu.Lock()
	handler := s.NotifyHandler
	s.mu.Unlock()
	if handler != nil {
		handler(req.GetSeverity(), req.GetMessage())
	}
	return &tauproto.Empty{}, nil
}

// GetConfig implements Host.GetConfig. Returns the value or found=false
// if the key is absent (or redacted by the ConfigSource).
func (s *HostServer) GetConfig(_ context.Context, req *tauproto.GetConfigRequest) (*tauproto.GetConfigResponse, error) {
	s.mu.Lock()
	src := s.ConfigSource
	s.mu.Unlock()
	v, ok := src.GetConfig(req.GetKey())
	return &tauproto.GetConfigResponse{Value: v, Found: ok}, nil
}

// SetLogWriter swaps the log destination. Safe to call after the host
// has started; concurrent Log calls serialize on the internal mutex.
func (s *HostServer) SetLogWriter(w io.Writer) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.LogWriter = w
}

// PluginAdapter is the plugin-side glue between a Server implementation
// of proto.PluginServer and go-plugin's plugin.Serve entrypoint. Plugin
// authors construct one with their Server and pass it to plugin.Serve.
type PluginAdapter struct {
	// Server is the plugin's proto.PluginServer implementation.
	Server tauproto.PluginServer
	// HostCallbacks is dialed at startup via the go-plugin broker; the
	// plugin uses it to call Host.Log/Notify/GetConfig. May be nil;
	// plugin code should nil-check before calling.
	HostCallbacks tauproto.HostClient
}

// ServerFactory is the name go-plugin dispenses (we registered a single
// "tau" plugin in PluginMap on the host side; the plugin's PluginMap
// uses the same name so the wire handshake matches).
func (a *PluginAdapter) ServerFactory() *GRPCPlugin {
	return &GRPCPlugin{ServerImpl: a.Server}
}

// PluginMap returns the plugin-side PluginMap for plugin.Serve. Exactly
// one entry keyed by PluginName; its GRPCPlugin carries the Server.
func (a *PluginAdapter) PluginMap() map[string]plugin.Plugin {
	return map[string]plugin.Plugin{
		PluginName: &GRPCPlugin{ServerImpl: a.Server},
	}
}

// ServeConfig assembles the plugin.Serve config. Plugins call:
//
//	adapter := &plugins.PluginAdapter{Server: myServer}
//	plugin.Serve(&adapter.ServeConfig())
//
// or use the Serve helper below.
func (a *PluginAdapter) ServeConfig() plugin.ServeConfig {
	return plugin.ServeConfig{
		HandshakeConfig: HandshakeConfig,
		Plugins:         a.PluginMap(),
		GRPCServer:      plugin.DefaultGRPCServer,
	}
}

// Serve is the convenience entrypoint. It calls plugin.Serve with the
// adapter's config and never returns (plugin.Serve exits the process).
// Authors who need a different exit behavior should use ServeConfig()
// directly.
func (a *PluginAdapter) Serve() {
	cfg := a.ServeConfig()
	plugin.Serve(&cfg)
}

// HostGRPCRegistrar is the hook go-plugin's GRPCServer uses to register
// the host-side Host service on the same gRPC server as the broker.
// Without this, plugin→host RPCs fail with "unknown service Host".
//
// The Manager sets this up by wrapping plugin.DefaultGRPCServer with a
// registrar that calls tauproto.RegisterHostServer.
func HostGRPCRegistrar(host tauproto.HostServer) func(opts []grpc.ServerOption) *grpc.Server {
	return func(opts []grpc.ServerOption) *grpc.Server {
		srv := grpc.NewServer(opts...)
		tauproto.RegisterHostServer(srv, host)
		return srv
	}
}
