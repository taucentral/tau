// Package e2e — plugin_lifecycle_test.go covers the plugin lifecycle through
// the full agent runtime. It builds the testdata minimal plugin, installs it,
// wires a Manager into SessionOptions, and verifies that plugin-provided
// tools reach the agent loop (and that Host.Log callbacks reach the HostServer).
package e2e

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/taucentral/tau/internal/agent"
	"github.com/taucentral/tau/internal/config"
	"github.com/taucentral/tau/internal/llm"
	"github.com/taucentral/tau/internal/plugins"
	"github.com/taucentral/tau/internal/state"
	"github.com/taucentral/tau/internal/tools"
)

// buildMinimalPluginForE2E builds the testdata minimal plugin binary to a
// temp path under the tau module root. The e2e package cannot import the
// internal/plugins test helpers, so this is a local copy of the pattern.
func buildMinimalPluginForE2E(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	out := filepath.Join(dir, "tau-plugin-minimal")
	if runtime.GOOS == "windows" {
		out += ".exe"
	}
	// Resolve the testdata path relative to the tau module root.
	cmd := exec.Command("go", "build", "-o="+out,
		"github.com/taucentral/tau/internal/plugins/testdata/minimalplugin")
	var stderr strings.Builder
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("go build minimal plugin: %v\nstderr: %s", err, stderr.String())
	}
	_ = os.Chmod(out, 0o755)
	return out
}

// installMinimalForE2E places the built binary into a plugins/ subdirectory
// of projDir and returns the plugins directory path.
func installMinimalForE2E(t *testing.T, src, projDir string) string {
	t.Helper()
	pluginsDir := filepath.Join(projDir, "plugins")
	if err := os.MkdirAll(pluginsDir, 0o755); err != nil {
		t.Fatalf("mkdir plugins: %v", err)
	}
	dst := filepath.Join(pluginsDir, "tau-plugin-minimal")
	if runtime.GOOS == "windows" {
		dst += ".exe"
	}
	if err := os.Rename(src, dst); err != nil {
		t.Fatalf("rename plugin binary: %v", err)
	}
	return pluginsDir
}

// newPluginSession is like newSession but also wires a *plugins.Manager
// into SessionOptions.Plugins. The Manager is returned so tests can
// inspect plugin state after Run. The test is responsible for shutting
// down the Manager via t.Cleanup.
func newPluginSession(t *testing.T, provider llm.LLMClient, pluginMgr *plugins.Manager) (*agent.AgentSession, *agent.AgentSessionRuntime) {
	t.Helper()
	opts := agent.SessionOptions{
		Model:     "test-model",
		Settings:  config.DefaultSettings(),
		LLMClient: provider,
		Tools:     []tools.HeadlessTool{tools.NewReadTool(tools.OSReadOperations{})},
		ConfigDir: t.TempDir(),
		Plugins:   pluginMgr,
	}
	rt, err := agent.CreateAgentSessionRuntime(context.Background(), t.TempDir(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSessionRuntime: %v", err)
	}
	return agent.NewAgentSession(rt), rt
}

// TestPluginLifecycle_ToolsReachAgent verifies that plugin-provided tools
// are registered by the runtime and dispatched when the model calls them.
// The faux provider emits a single `minimal.echo` tool call; the test
// asserts the echo result appears in the state tree.
func TestPluginLifecycle_ToolsReachAgent(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping plugin lifecycle test in -short mode")
	}

	binPath := buildMinimalPluginForE2E(t)
	pluginsDir := installMinimalForE2E(t, binPath, t.TempDir())

	hostSrv := plugins.NoopHostServer()
	mgr, err := plugins.NewManager(pluginsDir, "", hostSrv)
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	if _, err := mgr.SpawnAll(context.Background()); err != nil {
		t.Fatalf("SpawnAll: %v", err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = mgr.Shutdown(ctx)
	})

	// Verify the plugin registered its tools.
	pluginTools := mgr.Tools()
	if len(pluginTools) < 2 {
		t.Fatalf("expected at least 2 plugin tools, got %d", len(pluginTools))
	}
	var hasEcho bool
	for _, tl := range pluginTools {
		if tl.Name() == "minimal.echo" {
			hasEcho = true
		}
	}
	if !hasEcho {
		t.Fatal("minimal.echo tool not registered")
	}

	provider := NewFauxProvider(
		// Iteration 1: model calls minimal.echo.
		[]llm.Delta{
			llm.ToolCallDelta{ContentIndex: 0, ID: "tu_1", Name: "minimal.echo", PartialInput: `{"text":"hello"}`},
			llm.Final{StopReason: llm.StopReasonToolUse},
		},
		// Iteration 2: model finishes.
		textThenFinal("done"),
	)

	sess, rt := newPluginSession(t, provider, mgr)

	if err := sess.Run(context.Background(), "call echo"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	tree, err := rt.State.Tree()
	if err != nil {
		t.Fatalf("Tree: %v", err)
	}
	sess.Shutdown(context.Background())

	// The state tree should contain a tool-result with the echo output.
	var foundEchoResult bool
	for _, id := range tree.IDs() {
		e, _ := tree.Get(id)
		if e.Kind != state.KindMessage {
			continue
		}
		mp, ok := e.Payload.(state.MessagePayload)
		if !ok || mp.Role != llm.RoleUser {
			continue
		}
		for _, b := range mp.Content {
			if tr, ok := b.(llm.ToolResult); ok {
				for _, cb := range tr.Content {
					if tc, ok := cb.(llm.TextContent); ok && strings.Contains(tc.Text, "hello") {
						foundEchoResult = true
					}
				}
			}
		}
	}
	if !foundEchoResult {
		t.Error("expected minimal.echo result in state tree")
	}
}

// TestPluginLifecycle_HostLogReachesHostServer verifies that plugin→host
// RPCs reach the HostServer. The minimal plugin's `minimal.log` tool
// forwards its text argument to Host.Log; the test captures the log in
// a buffer and asserts the message arrived.
func TestPluginLifecycle_HostLogReachesHostServer(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping plugin lifecycle test in -short mode")
	}

	binPath := buildMinimalPluginForE2E(t)
	pluginsDir := installMinimalForE2E(t, binPath, t.TempDir())

	var logBuf bytes.Buffer
	hostSrv := plugins.NewHostServer(&logBuf, plugins.NoopConfigSource(), nil)
	mgr, err := plugins.NewManager(pluginsDir, "", hostSrv)
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	if _, err := mgr.SpawnAll(context.Background()); err != nil {
		t.Fatalf("SpawnAll: %v", err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = mgr.Shutdown(ctx)
	})

	provider := NewFauxProvider(
		// Iteration 1: model calls minimal.log with a known string.
		[]llm.Delta{
			llm.ToolCallDelta{ContentIndex: 0, ID: "tu_1", Name: "minimal.log", PartialInput: `{"text":"host-log-marker"}`},
			llm.Final{StopReason: llm.StopReasonToolUse},
		},
		// Iteration 2: model finishes.
		textThenFinal("done"),
	)

	sess, rt := newPluginSession(t, provider, mgr)

	if err := sess.Run(context.Background(), "call log"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	sess.Shutdown(context.Background())
	_ = rt // silence unused

	// The Host.Log callback should have written the marker to logBuf.
	// Give the subprocess a moment to flush (the RPC is synchronous from
	// the plugin's perspective, but the buffer write happens in the host
	// goroutine that serves the Log RPC).
	if !strings.Contains(logBuf.String(), "host-log-marker") {
		t.Errorf("Host.Log did not capture the marker; got:\n%s", logBuf.String())
	}
}
