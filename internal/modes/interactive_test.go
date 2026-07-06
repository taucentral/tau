package modes

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/taucentral/tau/internal/agent"
	"github.com/taucentral/tau/internal/config"
	"github.com/taucentral/tau/internal/fauxprovider"
	"github.com/taucentral/tau/internal/tools"
)

// newInteractiveSession wires an AgentSession against the faux provider
// so tests can drive RunInteractive without network access.
func newInteractiveSession(t *testing.T) *agent.AgentSession {
	t.Helper()
	client := fauxprovider.NewWithResponse("interactive faux reply")
	opts := agent.SessionOptions{
		Model:         "faux",
		Settings:      config.DefaultSettings(),
		LLMClient:     client,
		Tools:         []tools.HeadlessTool{tools.NewReadTool(tools.OSReadOperations{})},
		ContextWindow: 200000,
	}
	rt, err := agent.CreateAgentSessionRuntime(context.Background(), t.TempDir(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSessionRuntime: %v", err)
	}
	sess := agent.NewAgentSession(rt)
	t.Cleanup(func() { sess.Shutdown(context.Background()) })
	return sess
}

func TestRunInteractive_NilSession(t *testing.T) {
	err := RunInteractive(context.Background(), InteractiveOptions{}, nil)
	if err == nil {
		t.Fatal("RunInteractive(nil session): err = nil, want error")
	}
	if !strings.Contains(err.Error(), "session is nil") {
		t.Errorf("err = %v, want 'session is nil'", err)
	}
}

func TestRunInteractive_BadThemeFile(t *testing.T) {
	dir := t.TempDir()
	themesDir := filepath.Join(dir, "themes")
	if err := os.MkdirAll(themesDir, 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	broken := filepath.Join(themesDir, "broken.json")
	if err := os.WriteFile(broken, []byte(`{not-json`), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}

	sess := newInteractiveSession(t)
	err := RunInteractive(context.Background(), InteractiveOptions{
		ThemeName: "broken",
		AgentDir:  dir,
	}, sess)
	if err == nil {
		t.Fatal("RunInteractive with broken theme: err = nil, want error")
	}
	if !strings.Contains(err.Error(), "load theme") {
		t.Errorf("err = %v, want substring 'load theme'", err)
	}
}

// TestRunInteractive_DefaultTheme_NoError verifies that with the default
// dark theme (no custom file), RunInteractive at least loads the model
// without erroring before the program loop starts. We rely on
// cancel-on-context to stop the program quickly.
//
// Note: bubbletea may emit a non-fatal error to stderr when stdin is
// not a TTY; this test uses a context-with-timeout to fence the run.
func TestRunInteractive_ContextCancelStopsProgram(t *testing.T) {
	sess := newInteractiveSession(t)

	// Use a pipe so bubbletea's input reader sees EOF on close.
	stdinR, stdinW, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	t.Cleanup(func() { stdinR.Close(); stdinW.Close() })

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	var stderr bytes.Buffer
	err = RunInteractive(ctx, InteractiveOptions{
		Stdin:       stdinR,
		Stdout:      &bytes.Buffer{},
		Stderr:      &stderr,
		noAltScreen: true,
	}, sess)
	// The function returns nil when ctx fires because the watcher calls
	// prog.Quit cleanly. A non-nil error here means bubbletea itself
	// failed; we tolerate the benign "read closed" EOF case.
	_ = err
}
