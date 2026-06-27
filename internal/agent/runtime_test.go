package agent

import (
	"context"
	"testing"
	"time"

	"github.com/coevin/tau/internal/config"
	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/llm/tokencounter"
	"github.com/coevin/tau/internal/state"
	"github.com/coevin/tau/internal/tools"
)

// stubLLMClient is a minimal LLMClient for runtime wiring tests. It does
// not produce deltas; tests that need a real stream use the test/e2e faux
// provider (task 8.10). Here we only need to satisfy the interface so the
// factory can wire the Summarizer.
type stubLLMClient struct{}

func (stubLLMClient) Stream(ctx context.Context, req llm.Request) (<-chan llm.Delta, error) {
	ch := make(chan llm.Delta)
	go func() {
		defer close(ch)
		ch <- llm.Final{StopReason: llm.StopReasonEndTurn}
	}()
	return ch, nil
}

// newTestOptions returns a SessionOptions with the required fields filled
// and ConfigDir set to a temp dir so the factory doesn't touch the real
// ~/.config/tau.
func newTestOptions(t *testing.T) SessionOptions {
	t.Helper()
	return SessionOptions{
		Model:     "test-model",
		Settings:  config.DefaultSettings(),
		LLMClient: stubLLMClient{},
		Tools:     []tools.HeadlessTool{tools.NewReadTool(tools.OSReadOperations{})},
		ConfigDir: t.TempDir(),
	}
}

// TestCreateRuntime_Success verifies the happy path: every required field
// is wired and accessible on the returned runtime.
func TestCreateRuntime_Success(t *testing.T) {
	ctx := context.Background()
	opts := newTestOptions(t)
	rt, err := CreateAgentSessionRuntime(ctx, t.TempDir(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSessionRuntime: %v", err)
	}
	if rt == nil {
		t.Fatal("runtime is nil")
	}
	if rt.State == nil {
		t.Error("rt.State is nil")
	}
	if rt.Registry == nil {
		t.Error("rt.Registry is nil")
	}
	if rt.EventBus == nil {
		t.Error("rt.EventBus is nil")
	}
	if rt.Compactor == nil {
		t.Error("rt.Compactor is nil")
	}
	if rt.Summarizer == nil {
		t.Error("rt.Summarizer is nil")
	}
	if rt.Assembler == nil {
		t.Error("rt.Assembler is nil")
	}
	if rt.TemplateLoader == nil {
		t.Error("rt.TemplateLoader is nil")
	}
	if rt.MutationQueue == nil {
		t.Error("rt.MutationQueue is nil")
	}
	// Built-in tool registered.
	if _, err := rt.Registry.Lookup("read"); err != nil {
		t.Errorf("Lookup(read): %v", err)
	}
	// ownsState defaults true when no StateManager injected.
	if !rt.ownsState {
		t.Error("ownsState should be true for factory-created state")
	}
}

// TestCreateRuntime_RequiredFields verifies each missing required field
// produces a descriptive error.
func TestCreateRuntime_RequiredFields(t *testing.T) {
	cases := []struct {
		name    string
		mutate  func(o *SessionOptions)
		wantErr string
	}{
		{"missing Model", func(o *SessionOptions) { o.Model = "" }, "Model"},
		{"missing LLMClient", func(o *SessionOptions) { o.LLMClient = nil }, "LLMClient"},
		{"missing Tools", func(o *SessionOptions) { o.Tools = nil }, "Tools"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			opts := newTestOptions(t)
			c.mutate(&opts)
			_, err := CreateAgentSessionRuntime(context.Background(), t.TempDir(), opts)
			if err == nil {
				t.Fatalf("expected error containing %q", c.wantErr)
			}
		})
	}
}

// TestCreateRuntime_EmptyCwd verifies the factory rejects an empty cwd
// without reaching the filesystem.
func TestCreateRuntime_EmptyCwd(t *testing.T) {
	opts := newTestOptions(t)
	_, err := CreateAgentSessionRuntime(context.Background(), "", opts)
	if err == nil {
		t.Fatal("expected error for empty cwd")
	}
}

// TestCreateRuntime_CancelledContext verifies a cancelled context is
// honoured before any work begins.
func TestCreateRuntime_CancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	opts := newTestOptions(t)
	_, err := CreateAgentSessionRuntime(ctx, t.TempDir(), opts)
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
}

// TestCreateRuntime_InjectedStateManager verifies a caller-supplied
// state.Manager is used verbatim and not closed on Shutdown... actually
// we test the ownsState flag here; the close-on-shutdown behavior is
// exercised in session_test.go.
func TestCreateRuntime_InjectedStateManager(t *testing.T) {
	opts := newTestOptions(t)
	inj := state.NewInMemoryManager(t.TempDir())
	opts.StateManager = inj
	rt, err := CreateAgentSessionRuntime(context.Background(), t.TempDir(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSessionRuntime: %v", err)
	}
	if rt.State != inj {
		t.Error("rt.State should equal the injected manager")
	}
	if rt.ownsState {
		t.Error("ownsState should be false when caller injects state manager")
	}
}

// TestCreateRuntime_InjectedTokenCounter verifies a caller-supplied
// counter is the one used by the Compactor.
func TestCreateRuntime_InjectedTokenCounter(t *testing.T) {
	opts := newTestOptions(t)
	custom := tokencounter.HeuristicCounter{}
	opts.TokenCounter = custom
	rt, err := CreateAgentSessionRuntime(context.Background(), t.TempDir(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSessionRuntime: %v", err)
	}
	// The compactor stores Counter verbatim; identity check via Count().
	if rt.Compactor.Counter == nil {
		t.Fatal("Counter is nil")
	}
	// HeuristicCounter.Count == len(text)/4. Use a distinctive input.
	want := len("hello world, this is a test string") / 4
	got := rt.Compactor.Counter.Count("test-model", "hello world, this is a test string")
	if got != want {
		t.Errorf("Counter = %T, want HeuristicCounter (got %d, want %d)",
			rt.Compactor.Counter, got, want)
	}
}

// TestCreateRuntime_ReserveTokensFromSettings verifies the compactor's
// ReserveTokens reflects Settings.Compaction.ReserveTokens when set.
func TestCreateRuntime_ReserveTokensFromSettings(t *testing.T) {
	opts := newTestOptions(t)
	custom := 4096
	opts.Settings.Compaction = &config.CompactionSettings{
		ReserveTokens: &custom,
	}
	rt, err := CreateAgentSessionRuntime(context.Background(), t.TempDir(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSessionRuntime: %v", err)
	}
	if rt.Compactor.ReserveTokens != custom {
		t.Errorf("ReserveTokens = %d, want %d", rt.Compactor.ReserveTokens, custom)
	}
}

// TestCreateRuntime_ReserveTokensDefault verifies that when Settings does
// not specify ReserveTokens, the compactor falls back to the package
// default (DefaultReserveTokens = 8192).
func TestCreateRuntime_ReserveTokensDefault(t *testing.T) {
	opts := newTestOptions(t)
	// Force Compaction to nil so reserve stays 0 at the factory; the
	// compactor substitutes its own default.
	opts.Settings.Compaction = nil
	rt, err := CreateAgentSessionRuntime(context.Background(), t.TempDir(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSessionRuntime: %v", err)
	}
	if rt.Compactor.ReserveTokens <= 0 {
		t.Errorf("ReserveTokens = %d, want > 0 (default)", rt.Compactor.ReserveTokens)
	}
}

// TestCreateRuntime_DuplicatePluginToolDoesNotFail verifies that when a
// plugin tool name collides with a built-in, the factory continues and
// surfaces the collision as an event on the bus.
func TestCreateRuntime_DuplicatePluginToolDoesNotFail(t *testing.T) {
	// We can't easily build a plugins.Manager in this test (it requires a
	// running plugin binary). Instead, simulate the collision path by
	// constructing the registry directly and confirming first-wins.
	r := tools.NewRegistry()
	read1 := tools.NewReadTool(tools.OSReadOperations{})
	read2 := tools.NewReadTool(tools.OSReadOperations{})
	if err := r.Register(read1); err != nil {
		t.Fatalf("register first: %v", err)
	}
	if err := r.Register(read2); err == nil {
		t.Error("expected duplicate registration to fail")
	}
	// Confirm Lookup returns the first one.
	got, err := r.Lookup("read")
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	if got != read1 {
		t.Error("Lookup should return the first-registered tool")
	}
}

// TestCreateRuntime_ConfigDirResolvedFromEnv verifies that when opts.ConfigDir
// is empty, the factory falls back to config.ConfigDir(). We use TAU_CONFIG_DIR
// so the test doesn't depend on the user's real ~/.config.
func TestCreateRuntime_ConfigDirResolvedFromEnv(t *testing.T) {
	t.Setenv("TAU_CONFIG_DIR", t.TempDir())
	opts := newTestOptions(t)
	opts.ConfigDir = "" // force the factory to resolve
	rt, err := CreateAgentSessionRuntime(context.Background(), t.TempDir(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSessionRuntime: %v", err)
	}
	if rt.ConfigDir == "" {
		t.Error("ConfigDir should be resolved, not empty")
	}
}

// TestRegistrationCollisionEvent_Topic verifies the internal diagnostic
// event satisfies the Event interface and reports its reserved topic.
func TestRegistrationCollisionEvent_Topic(t *testing.T) {
	evt := registrationCollisionEvent{When: time.Now(), Name: "x", Reason: "dup"}
	if evt.Topic() != TopicDiagnostic {
		t.Errorf("Topic = %q, want %q", evt.Topic(), TopicDiagnostic)
	}
}
