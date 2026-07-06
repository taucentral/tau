package plugins

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"

	tauproto "github.com/taucentral/tau/internal/proto"
	"github.com/taucentral/tau/internal/tools"
)

// buildMinimalPlugin builds the testdata minimal plugin binary to a
// temp path and returns the path. Fails the test if `go build` fails.
func buildMinimalPlugin(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	out := filepath.Join(dir, "tau-plugin-minimal")
	if PluginFileExt != "" {
		out += PluginFileExt
	}
	cmd := exec.Command("go", "build", "-o="+out, "./testdata/minimalplugin")
	cmd.Dir = "."
	var buf strings.Builder
	cmd.Stderr = &buf
	if err := cmd.Run(); err != nil {
		t.Fatalf("go build minimal plugin: %v\nstderr: %s", err, buf.String())
	}
	if !isExecutable(out) {
		t.Fatalf("built plugin not executable: %s", out)
	}
	_ = os.Chmod(out, 0o755)
	return out
}

// newTestHostServer returns a HostServer with discard sinks; tests can
// swap fields after construction.
func newTestHostServer() *HostServer {
	return NewHostServer(io.Discard, NoopConfigSource(), func(string, string) {})
}

// installPluginBin relocates the built binary into a plugins/
// subdirectory of projDir under the canonical plugin name ("tau-minimal").
// Returns the plugins directory path so callers don't need to recompute it.
func installPluginBin(t *testing.T, src, projDir string) string {
	t.Helper()
	const shortName = "minimal"
	pluginsDir := filepath.Join(projDir, "plugins")
	if err := os.MkdirAll(pluginsDir, 0o755); err != nil {
		t.Fatalf("mkdir plugins: %v", err)
	}
	dst := filepath.Join(pluginsDir, PluginPrefix+shortName+PluginFileExt)
	if err := os.Rename(src, dst); err != nil {
		t.Fatalf("rename plugin binary: %v", err)
	}
	return pluginsDir
}

// TestManager_SpawnListExecuteShutdown is the happy-path end-to-end
// integration test for the plugin layer. It exercises Discover,
// SpawnAll, Tools(), Execute, and Shutdown.
func TestManager_SpawnListExecuteShutdown(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping plugin integration test in -short mode")
	}
	bin := buildMinimalPlugin(t)
	projDir := t.TempDir()
	pluginsDir := installPluginBin(t, bin, projDir)

	mgr, err := NewManager(pluginsDir, "", newTestHostServer())
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	spawned, firstErr := mgr.SpawnAll(ctx)
	if firstErr != nil {
		t.Fatalf("SpawnAll: %v", firstErr)
	}
	if spawned != 1 {
		t.Fatalf("expected 1 plugin spawned, got %d", spawned)
	}
	defer func() {
		shutdownCtx, sc := context.WithTimeout(context.Background(), 10*time.Second)
		defer sc()
		if err := mgr.Shutdown(shutdownCtx); err != nil {
			t.Errorf("Shutdown: %v", err)
		}
	}()

	toolsList := mgr.Tools()
	if len(toolsList) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(toolsList))
	}
	names := []string{toolsList[0].Name(), toolsList[1].Name()}
	if !containsStr(names, "minimal.echo") || !containsStr(names, "minimal.fail") {
		t.Fatalf("expected minimal.echo and minimal.fail, got %v", names)
	}

	// Execute echo.
	res, err := mgr.Execute(ctx, tools.ToolCall{
		ID:   "call-1",
		Name: "minimal.echo",
		Args: json.RawMessage(`{"text":"hello"}`),
		Cwd:  projDir,
	})
	if err != nil {
		t.Fatalf("Execute echo: %v", err)
	}
	if res.IsError {
		t.Fatalf("echo returned error: %+v", res)
	}
	if len(res.Content) != 1 {
		t.Fatalf("echo returned %d blocks, want 1", len(res.Content))
	}

	// Execute fail.
	res2, err := mgr.Execute(ctx, tools.ToolCall{
		ID:   "call-2",
		Name: "minimal.fail",
		Args: json.RawMessage(`{"text":"nope"}`),
		Cwd:  projDir,
	})
	if err != nil {
		t.Fatalf("Execute fail: %v", err)
	}
	if !res2.IsError {
		t.Fatalf("fail should set IsError")
	}
}

// TestManager_PluginPanicReturnsIsError covers the crash mid-Execute
// scenario: the plugin process panics; the host sees the gRPC stream
// close unexpectedly and converts the failure into a ToolResult with
// IsError=true.
func TestManager_PluginPanicReturnsIsError(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping plugin integration test in -short mode")
	}
	bin := buildMinimalPlugin(t)
	projDir := t.TempDir()
	pluginsDir := installPluginBin(t, bin, projDir)

	mgr, err := NewManager(pluginsDir, "", newTestHostServer())
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if _, err := mgr.SpawnAll(ctx); err != nil {
		t.Fatalf("SpawnAll: %v", err)
	}
	defer func() {
		shutdownCtx, sc := context.WithTimeout(context.Background(), 10*time.Second)
		defer sc()
		_ = mgr.Shutdown(shutdownCtx)
	}()

	// "minimal.panic" is not in ListTools (the plugin's ListTools omits
	// it); Execute routes by name regardless. The minimal plugin's
	// Execute dispatches by call name and triggers a panic.
	res, err := mgr.Execute(ctx, tools.ToolCall{
		ID:   "panic-call",
		Name: "minimal.panic",
		Args: json.RawMessage(`{}`),
		Cwd:  projDir,
	})
	if err != nil {
		// A returned error is acceptable; the host did NOT crash.
		return
	}
	if !res.IsError {
		t.Fatalf("panic call should surface as IsError=true; got %+v", res)
	}
}

// TestManager_OnNextCallRecoverAfterCrash covers the on-next-call
// restart policy: crash the plugin deliberately, observe state becomes
// crashed, then call Execute and verify the plugin re-spawns and the
// call succeeds.
func TestManager_OnNextCallRecoverAfterCrash(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping plugin integration test in -short mode")
	}
	bin := buildMinimalPlugin(t)
	projDir := t.TempDir()
	pluginsDir := installPluginBin(t, bin, projDir)

	mgr, err := NewManager(pluginsDir, "", newTestHostServer())
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	mgr.DefaultRestartPolicy = RestartOnNextCall

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if _, err := mgr.SpawnAll(ctx); err != nil {
		t.Fatalf("SpawnAll: %v", err)
	}
	defer func() {
		shutdownCtx, sc := context.WithTimeout(context.Background(), 10*time.Second)
		defer sc()
		_ = mgr.Shutdown(shutdownCtx)
	}()

	// Verify the plugin is alive first.
	res, err := mgr.Execute(ctx, tools.ToolCall{
		ID: "before", Name: "minimal.echo", Args: json.RawMessage(`{"text":"pre"}`), Cwd: projDir,
	})
	if err != nil || res.IsError {
		t.Fatalf("pre-crash echo failed: err=%v res=%+v", err, res)
	}

	// Trigger a panic that kills the subprocess.
	_, _ = mgr.Execute(ctx, tools.ToolCall{
		ID: "panic", Name: "minimal.panic", Args: json.RawMessage(`{}`), Cwd: projDir,
	})

	// Wait for crash detection (200ms tick).
	if !waitForState(t, mgr, "minimal", PluginStateCrashed, 3*time.Second) {
		t.Fatalf("plugin did not reach Crashed state")
	}

	// On-next-call: Execute should re-spawn and succeed.
	res2, err := mgr.Execute(ctx, tools.ToolCall{
		ID: "after", Name: "minimal.echo", Args: json.RawMessage(`{"text":"post"}`), Cwd: projDir,
	})
	if err != nil {
		t.Fatalf("post-crash Execute: err=%v (expected auto-restart)", err)
	}
	if res2.IsError {
		t.Fatalf("post-crash echo should succeed: %+v", res2)
	}
}

// TestManager_NeverPolicyRefusesAfterCrash covers the never policy:
// after a crash, subsequent Execute calls fail with IsError=true rather
// than re-spawning.
func TestManager_NeverPolicyRefusesAfterCrash(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping plugin integration test in -short mode")
	}
	bin := buildMinimalPlugin(t)
	projDir := t.TempDir()
	pluginsDir := installPluginBin(t, bin, projDir)

	mgr, err := NewManager(pluginsDir, "", newTestHostServer())
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	mgr.DefaultRestartPolicy = RestartNever

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if _, err := mgr.SpawnAll(ctx); err != nil {
		t.Fatalf("SpawnAll: %v", err)
	}
	defer func() {
		shutdownCtx, sc := context.WithTimeout(context.Background(), 10*time.Second)
		defer sc()
		_ = mgr.Shutdown(shutdownCtx)
	}()

	// Crash the plugin.
	_, _ = mgr.Execute(ctx, tools.ToolCall{
		ID: "panic", Name: "minimal.panic", Args: json.RawMessage(`{}`), Cwd: projDir,
	})

	if !waitForState(t, mgr, "minimal", PluginStateCrashed, 3*time.Second) {
		t.Fatalf("plugin did not reach Crashed state")
	}

	// Subsequent Execute should return an IsError result (plugin unavailable).
	res, err := mgr.Execute(ctx, tools.ToolCall{
		ID: "after", Name: "minimal.echo", Args: json.RawMessage(`{"text":"no"}`), Cwd: projDir,
	})
	if err != nil {
		t.Fatalf("Execute returned error (expected IsError result): %v", err)
	}
	if !res.IsError {
		t.Fatalf("RestartNever should make post-crash call return IsError; got %+v", res)
	}
}

// TestManager_ProjectShadowsGlobal verifies the discovery rule: when
// both the project and global plugin dirs have a same-named plugin,
// the project one wins.
func TestManager_ProjectShadowsGlobal(t *testing.T) {
	projPlugins := t.TempDir()
	globalPlugins := t.TempDir()
	mgr, err := NewManager(projPlugins, globalPlugins, newTestHostServer())
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	for _, dir := range []string{projPlugins, globalPlugins} {
		p := filepath.Join(dir, PluginPrefix+"shadow"+PluginFileExt)
		if err := os.WriteFile(p, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
			t.Fatalf("write plugin: %v", err)
		}
	}
	paths, err := mgr.Discover()
	if err != nil {
		t.Fatalf("Discover: %v", err)
	}
	if len(paths) != 1 {
		t.Fatalf("expected 1 discovery (project shadows global), got %d: %+v", len(paths), paths)
	}
	if paths[0].Source != PluginSourceProject {
		t.Errorf("expected project source, got %v", paths[0].Source)
	}
}

// TestManager_NonMatchingFilesIgnored verifies discovery skips files
// that don't match the tau-plugin-* prefix.
func TestManager_NonMatchingFilesIgnored(t *testing.T) {
	projPlugins := t.TempDir()
	for _, name := range []string{"notes.txt", "random-binary", "tau-plugin-real"} {
		p := filepath.Join(projPlugins, name)
		if err := os.WriteFile(p, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
			t.Fatalf("write: %v", err)
		}
	}
	mgr, err := NewManager(projPlugins, "", newTestHostServer())
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	paths, err := mgr.Discover()
	if err != nil {
		t.Fatalf("Discover: %v", err)
	}
	if len(paths) != 1 {
		t.Fatalf("expected 1 discovery, got %d: %+v", len(paths), paths)
	}
	if paths[0].ShortName != "real" {
		t.Errorf("expected short name 'real', got %q", paths[0].ShortName)
	}
}

// TestManager_CollisionDeduplicatesToolNames covers Phase 6.10: two
// plugins expose the same tool name; first registration wins, the
// second is dropped, a diagnostic is emitted.
func TestManager_CollisionDeduplicatesToolNames(t *testing.T) {
	mgr, err := NewManager("", "", newTestHostServer())
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	diag := make(chan string, 16)
	mgr.Diagnostics = diag

	fakeProv := func() (tauproto.PluginClient, error) {
		return nil, errors.New("fake provider; not wired to a real plugin")
	}

	mgr.mu.Lock()
	mgr.plugins["alpha"] = &managedPlugin{
		shortName: "alpha",
		namespaced: map[string]*PluginTool{
			"shared.dup": NewPluginTool("alpha", "shared.dup", "alpha desc", "{}", fakeProv),
			"alpha.only": NewPluginTool("alpha", "alpha.only", "alpha-only", "{}", fakeProv),
		},
		policy: RestartOnNextCall,
	}
	mgr.plugins["beta"] = &managedPlugin{
		shortName: "beta",
		namespaced: map[string]*PluginTool{
			"shared.dup": NewPluginTool("beta", "shared.dup", "beta desc", "{}", fakeProv),
			"beta.only":  NewPluginTool("beta", "beta.only", "beta-only", "{}", fakeProv),
		},
		policy: RestartOnNextCall,
	}
	mgr.mu.Unlock()

	toolsList := mgr.Tools()
	if len(toolsList) != 3 {
		t.Fatalf("expected 3 tools (alpha.only, beta.only, shared.dup once), got %d: %+v", len(toolsList), toolsList)
	}
	var sawSharedDup, sawAlphaOnly, sawBetaOnly bool
	for _, tool := range toolsList {
		switch tool.Name() {
		case "shared.dup":
			sawSharedDup = true
			if tool.Description() != "alpha desc" {
				t.Errorf("shared.dup should be alpha's; got desc=%q", tool.Description())
			}
		case "alpha.only":
			sawAlphaOnly = true
		case "beta.only":
			sawBetaOnly = true
		}
	}
	if !sawSharedDup || !sawAlphaOnly || !sawBetaOnly {
		t.Errorf("missing expected tools: dup=%v alpha=%v beta=%v",
			sawSharedDup, sawAlphaOnly, sawBetaOnly)
	}
	select {
	case msg := <-diag:
		if !strings.Contains(msg, "shared.dup") || !strings.Contains(msg, "skipped") {
			t.Errorf("expected collision diagnostic, got: %q", msg)
		}
	case <-time.After(100 * time.Millisecond):
		t.Errorf("expected collision diagnostic, none emitted")
	}
}

// TestHostServer_LogNotifyGetConfig exercises the Host service handlers.
// Each method is probed directly so the integration test does not need
// a plugin process for this coverage.
func TestHostServer_LogNotifyGetConfig(t *testing.T) {
	var logBuf strings.Builder
	notifyBuf := make(chan [2]string, 4)
	cfg := ConfigSourceFunc(func(k string) (string, bool) {
		if k == "known.key" {
			return "value", true
		}
		return "", false
	})
	srv := NewHostServer(&logBuf, cfg, func(severity, message string) {
		notifyBuf <- [2]string{severity, message}
	})

	ctx := context.Background()
	if _, err := srv.Log(ctx, &tauproto.LogRequest{
		Level:   tauproto.LogRequest_LEVEL_INFO,
		Message: "hello from plugin",
	}); err != nil {
		t.Fatalf("Log: %v", err)
	}
	if !strings.Contains(logBuf.String(), "hello from plugin") {
		t.Errorf("Log writer missing message: %q", logBuf.String())
	}

	if _, err := srv.Notify(ctx, &tauproto.NotifyRequest{
		Severity: "warning", Message: "high water",
	}); err != nil {
		t.Fatalf("Notify: %v", err)
	}
	select {
	case n := <-notifyBuf:
		if n[0] != "warning" || n[1] != "high water" {
			t.Errorf("notify payload wrong: %v", n)
		}
	default:
		t.Errorf("Notify did not invoke handler")
	}

	resp, err := srv.GetConfig(ctx, &tauproto.GetConfigRequest{Key: "known.key"})
	if err != nil {
		t.Fatalf("GetConfig: %v", err)
	}
	if !resp.GetFound() || resp.GetValue() != "value" {
		t.Errorf("GetConfig known.key = found=%v value=%q", resp.GetFound(), resp.GetValue())
	}
	resp, _ = srv.GetConfig(ctx, &tauproto.GetConfigRequest{Key: "missing.key"})
	if resp.GetFound() {
		t.Errorf("GetConfig missing.key should return found=false")
	}
}

// TestHandshake_ProtocolMismatch covers the version-drift path using a
// real gRPC server running in-process via bufconn. The fake server
// always reports a mismatched protocol version; the host should return
// ErrProtocolMismatch.
func TestHandshake_ProtocolMismatch(t *testing.T) {
	server := fakePluginServer{reportedVersion: 999}
	listener := bufconn.Listen(1024 * 1024)
	srv := grpc.NewServer()
	tauproto.RegisterPluginServer(srv, server)
	go func() { _ = srv.Serve(listener) }()
	defer srv.Stop()

	// grpc.NewClient (the non-deprecated dialer in grpc-go v1.81+) parses
	// the target as a URI; without a scheme, "bufnet" is resolved via DNS,
	// which fails with "produced zero addresses". The passthrough scheme
	// delegates to WithContextDialer and skips the resolver.
	conn, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(_ context.Context, _ string) (net.Conn, error) {
			return listener.Dial()
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer conn.Close()
	client := tauproto.NewPluginClient(conn)

	err = Handshake(context.Background(), "fake", client)
	if err == nil {
		t.Fatalf("expected ErrProtocolMismatch, got nil")
	}
	var mismatch *ErrProtocolMismatch
	if !errors.As(err, &mismatch) {
		t.Fatalf("error should be ErrProtocolMismatch, got %T: %v", err, err)
	}
	if mismatch.PluginVer != 999 {
		t.Errorf("PluginVer = %d, want 999", mismatch.PluginVer)
	}
	if mismatch.HostVer != int32(ProtocolVersion) {
		t.Errorf("HostVer = %d, want %d", mismatch.HostVer, ProtocolVersion)
	}
}

// fakePluginServer implements tauproto.PluginServer for in-process gRPC
// tests. Only Handshake is exercised; the other methods inherit the
// unimplemented-status-code behavior from the embedded
// UnimplementedPluginServer.
type fakePluginServer struct {
	tauproto.UnimplementedPluginServer
	reportedVersion int32
}

func (s fakePluginServer) Handshake(_ context.Context, _ *tauproto.Empty) (*tauproto.ProtocolVersion, error) {
	return &tauproto.ProtocolVersion{Version: s.reportedVersion}, nil
}

// waitForState polls the plugin's HostClient state until it reaches
// want or the timeout elapses. Returns true on success.
func waitForState(t *testing.T, mgr *Manager, pluginShort string, want PluginState, timeout time.Duration) bool {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		mgr.mu.Lock()
		mp := mgr.plugins[pluginShort]
		mgr.mu.Unlock()
		if mp != nil && mp.host.State() == want {
			return true
		}
		time.Sleep(100 * time.Millisecond)
	}
	return false
}

// containsStr reports whether s is in list.
func containsStr(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}
