// state_injection_test.go — verifies the state-manager injection contract.
//
// The contract is documented on Options.StateManager:
//   (a) an injected StateManager is NOT closed on Shutdown — the embedder
//       owns its lifecycle.
//   (b) the default manager (created when Options.StateManager is nil) IS
//       closed on Shutdown — the runtime owns it.
//   (c) NewInMemoryManager produces no .bolt file across a full turn.

package tau

import (
	"context"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/taucentral/tau/internal/config"
	"github.com/taucentral/tau/internal/fauxprovider"
	"github.com/taucentral/tau/internal/llm"
	"github.com/taucentral/tau/internal/state"
	"github.com/taucentral/tau/internal/tools"
)

// closeTrackingManager wraps a real in-memory manager and counts Close
// calls. Embedders reuse the same pattern to verify the runtime did NOT
// close an injected manager.
type closeTrackingManager struct {
	state.Manager
	closesMu sync.Mutex
	closes   int
}

func (m *closeTrackingManager) Close() error {
	m.closesMu.Lock()
	defer m.closesMu.Unlock()
	m.closes++
	if c, ok := m.Manager.(interface{ Close() error }); ok {
		return c.Close()
	}
	return nil
}

func (m *closeTrackingManager) CloseCount() int {
	m.closesMu.Lock()
	defer m.closesMu.Unlock()
	return m.closes
}

func newFauxClient() llm.LLMClient { return fauxprovider.NewWithResponse("ok") }

func basicTestOptions(t *testing.T) Options {
	t.Helper()
	return Options{
		Cwd:           t.TempDir(),
		Model:         "faux",
		LLMClient:     newFauxClient(),
		Tools:         []HeadlessTool{tools.NewReadTool(tools.OSReadOperations{})},
		Settings:      config.DefaultSettings(),
		ContextWindow: 200000,
	}
}

func TestInjectedStateManagerNotClosedOnShutdown(t *testing.T) {
	injected := &closeTrackingManager{Manager: state.NewInMemoryManager(t.TempDir())}

	opts := basicTestOptions(t)
	opts.StateManager = injected
	opts.Cwd = t.TempDir()

	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}

	if err := sess.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
	if got := injected.CloseCount(); got != 0 {
		t.Errorf("injected manager Close called %d times, want 0 (embedder owns lifecycle)", got)
	}

	// Idempotency: a second Shutdown also must not call Close.
	if err := sess.Shutdown(context.Background()); err != nil {
		t.Errorf("second Shutdown: %v", err)
	}
	if got := injected.CloseCount(); got != 0 {
		t.Errorf("injected manager Close called %d times after second Shutdown, want 0", got)
	}
}

func TestDefaultStateManagerClosedOnShutdown(t *testing.T) {
	// When Options.StateManager is nil, the runtime creates its own
	// manager rooted at Cwd and OWNS its lifecycle. We assert this by
	// constructing two sessions in the same Cwd and verifying the first
	// session's Shutdown releases the backing file lock (the second
	// session can be created without contention).
	cwd := t.TempDir()

	opts := basicTestOptions(t)
	opts.Cwd = cwd
	opts.StateManager = nil

	sess1, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession (1): %v", err)
	}
	if err := sess1.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown (1): %v", err)
	}

	// A fresh session in the same Cwd should succeed — the runtime-owned
	// manager from sess1 was closed on Shutdown so its bbolt lock (if
	// any) is released.
	sess2, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession (2): %v", err)
	}
	if err := sess2.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown (2): %v", err)
	}
}

func TestNewInMemoryManagerProducesNoBoltFile(t *testing.T) {
	cwd := t.TempDir()

	mgr := NewInMemoryManager(cwd)
	opts := basicTestOptions(t)
	opts.Cwd = cwd
	opts.StateManager = mgr

	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := sess.Run(ctx, "hello"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if err := sess.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}

	// After a full turn, the cwd must contain no .bolt files. The
	// in-memory manager never touches disk.
	matches, err := filepath.Glob(filepath.Join(cwd, "*.bolt"))
	if err != nil {
		t.Fatalf("glob: %v", err)
	}
	if len(matches) > 0 {
		t.Errorf("found %d .bolt files under cwd; in-memory manager must not persist: %v", len(matches), matches)
	}

	// Also check the .tau/sessions directory just in case the layout
	// changed. We don't fail on absence of the dir (the in-memory
	// manager does not create one), only on presence of files.
	nested, _ := filepath.Glob(filepath.Join(cwd, "**", "*.bolt"))
	if len(nested) > 0 {
		t.Errorf("found nested .bolt files: %v", nested)
	}
}
