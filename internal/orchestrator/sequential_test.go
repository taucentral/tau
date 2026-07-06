// sequential_test.go — internal tests for SequentialOrchestrator.
//
// Covers:
//   - Two-phase sequential run with MergePolicyAppend.
//   - Concurrent execution: three phases with no DependsOn edges run
//     concurrently within a dependency level (task 5.1(b)).
//   - Dependency-cycle rejection.
//   - Context cancellation mid-phase (child.Abort via ctx-watcher).
//   - Replay-merge conflict with Phase attribution and LineRange (D8/D9).
//   - dependencyLevels unit tests (level grouping, cycle, duplicates).
//   - Run-error surfacing via Orchestrator.Err() (D7).
//   - ParentEventBus forwarding (D4).
//   - ErrOrchestratorClosed on closed child state (D5).
//   - Shared-store branch handoff (D1/D3): phase B's child sees phase A's
//     merged entries via the parent's store.
//
// Tests reuse the agent package's test patterns (scriptedClient,
// recordingTool) but re-declare minimal versions because the agent
// helpers live in _test.go and are not importable.

package orchestrator

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/invopop/jsonschema"

	"github.com/taucentral/tau/internal/agent"
	"github.com/taucentral/tau/internal/config"
	"github.com/taucentral/tau/internal/llm"
	"github.com/taucentral/tau/internal/state"
	"github.com/taucentral/tau/internal/tools"
)

// scriptedClient emits a scripted stream of deltas per Stream call.
type scriptedClient struct {
	mu     sync.Mutex
	script []llm.Delta
	calls  int32
}

func (s *scriptedClient) Stream(ctx context.Context, req llm.Request) (<-chan llm.Delta, error) {
	atomic.AddInt32(&s.calls, 1)
	out := make(chan llm.Delta, 8)
	go func() {
		defer close(out)
		s.mu.Lock()
		script := s.script
		s.mu.Unlock()
		for _, d := range script {
			select {
			case out <- d:
			case <-ctx.Done():
				return
			}
		}
	}()
	return out, nil
}

// textThenFinal builds a delta script: one assistant text delta followed by a Final.
func textThenFinal(text string) []llm.Delta {
	return []llm.Delta{
		llm.TextDelta{Text: text},
		llm.Final{StopReason: llm.StopReasonEndTurn},
	}
}

// concurrencyClient tracks how many Stream calls are in-flight
// concurrently. Used by the concurrency test to assert that independent
// phases within a dependency level actually overlap.
type concurrencyClient struct {
	cur atomic.Int32
	max atomic.Int32
}

func (c *concurrencyClient) Stream(ctx context.Context, req llm.Request) (<-chan llm.Delta, error) {
	cur := c.cur.Add(1)
	// Track max concurrency via CAS loop.
	for {
		m := c.max.Load()
		if cur <= m || c.max.CompareAndSwap(m, cur) {
			break
		}
	}
	// Sleep so concurrent callers overlap. 100ms is long enough that
	// three concurrent goroutines all enter Stream before any exits,
	// but short enough to keep the test fast.
	select {
	case <-time.After(100 * time.Millisecond):
	case <-ctx.Done():
		c.cur.Add(-1)
		return nil, ctx.Err()
	}
	c.cur.Add(-1)
	out := make(chan llm.Delta, 2)
	out <- llm.TextDelta{Text: "ok"}
	out <- llm.Final{StopReason: llm.StopReasonEndTurn}
	close(out)
	return out, nil
}

// errBoom is the sentinel error returned by erroringClient.
var errBoom = errors.New("boom")

// erroringClient returns an error on every Stream call. Used by the
// run-error surfacing test to assert orch.Err() wraps the phase error.
type erroringClient struct{}

func (c *erroringClient) Stream(ctx context.Context, req llm.Request) (<-chan llm.Delta, error) {
	return nil, errBoom
}

// recordingTool is a minimal HeadlessTool that records its invocations.
type recordingTool struct {
	name  string
	calls atomic.Int32
}

func (t *recordingTool) Name() string                  { return t.name }
func (t *recordingTool) Description() string           { return "recording tool for tests" }
func (t *recordingTool) Parameters() jsonschema.Schema { return jsonschema.Schema{} }
func (t *recordingTool) Execute(ctx context.Context, call tools.ToolCall) (tools.ToolResult, error) {
	t.calls.Add(1)
	return tools.NewTextResult("ok"), nil
}

// newTestSession constructs an agent.AgentSession wired with a scripted
// client. The Orchestrator gate is satisfied by passing a non-nil
// placeholder so Spawn works; the test then constructs the real
// SequentialOrchestrator explicitly.
func newTestSession(t *testing.T, client llm.LLMClient, builtinTools []tools.HeadlessTool) *agent.AgentSession {
	t.Helper()
	opts := agent.SessionOptions{
		Model:        "test-model",
		Settings:     config.DefaultSettings(),
		LLMClient:    client,
		Tools:        builtinTools,
		ConfigDir:    t.TempDir(),
		Orchestrator: &stubOrchestrator{}, // non-nil so Spawn is allowed
	}
	rt, err := agent.CreateAgentSessionRuntime(context.Background(), t.TempDir(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSessionRuntime: %v", err)
	}
	return agent.NewAgentSession(rt)
}

// stubOrchestrator is a non-nil placeholder that satisfies the
// Orchestrator gate on SessionOptions. Its Run is never invoked by
// these tests (they call the real SequentialOrchestrator directly).
type stubOrchestrator struct{}

func (*stubOrchestrator) Run(ctx context.Context, spec OrchestrationSpec) (<-chan agent.Event, error) {
	return nil, nil
}
func (*stubOrchestrator) Err() error { return nil }

// TestSequentialOrchestrator_TwoPhaseAppendMerge exercises the happy
// path: two phases run in order (B DependsOn A), MergePolicyAppend
// grows the parent state tree, the multiplexed channel receives events
// and closes.
func TestSequentialOrchestrator_TwoPhaseAppendMerge(t *testing.T) {
	client := &scriptedClient{script: textThenFinal("phase-output")}

	parent := newTestSession(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}})
	defer parent.Shutdown(context.Background())

	orch := NewSequentialOrchestrator(parent)

	spec := OrchestrationSpec{
		Phases: []PhaseSpec{
			{Name: "a", Prompt: "phase A"},
			{Name: "b", Prompt: "phase B", DependsOn: []string{"a"}},
		},
		MergePolicy: MergePolicyAppend,
	}

	ch, err := orch.Run(context.Background(), spec)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	eventCount := 0
	for range ch {
		eventCount++
	}
	if eventCount == 0 {
		t.Errorf("expected at least one event from two-phase run; got %d", eventCount)
	}

	tree, terr := parent.Runtime().State.Tree()
	if terr != nil {
		t.Errorf("parent Tree: %v", terr)
	}
	if got := tree.Len(); got < 2 {
		t.Errorf("expected parent state to grow after append-merge; got len=%d", got)
	}

	// Two phases ran cleanly; Err should be nil.
	if e := orch.Err(); e != nil {
		t.Errorf("expected nil Err after clean run; got %v", e)
	}
}

// TestSequentialOrchestrator_ConcurrentWithinLevel asserts that three
// phases with no DependsOn edges actually run CONCURRENTLY (task
// 5.1(b)). The concurrencyClient sleeps 100ms inside Stream; if the
// three phases ran sequentially the total wall time would be ~300ms
// and max concurrency would be 1. If they run concurrently, max
// concurrency is >= 2 (ideally 3).
func TestSequentialOrchestrator_ConcurrentWithinLevel(t *testing.T) {
	client := &concurrencyClient{}

	parent := newTestSession(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}})
	defer parent.Shutdown(context.Background())

	orch := NewSequentialOrchestrator(parent)

	spec := OrchestrationSpec{
		Phases: []PhaseSpec{
			{Name: "first", Prompt: "1"},
			{Name: "second", Prompt: "2"},
			{Name: "third", Prompt: "3"},
		},
		MergePolicy: MergePolicyNone,
	}

	start := time.Now()
	ch, err := orch.Run(context.Background(), spec)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	for range ch {
	}
	elapsed := time.Since(start)

	max := client.max.Load()
	if max < 2 {
		t.Errorf("expected at least 2 concurrent Stream calls; got max=%d (elapsed=%v)", max, elapsed)
	}
}

// TestSequentialOrchestrator_DependencyCycleRejected constructs a spec
// with a cycle in DependsOn and asserts Run returns
// ErrOrchestrationAborted naming the cycle.
func TestSequentialOrchestrator_DependencyCycleRejected(t *testing.T) {
	client := &scriptedClient{script: textThenFinal("ok")}
	parent := newTestSession(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}})
	defer parent.Shutdown(context.Background())

	orch := NewSequentialOrchestrator(parent)

	spec := OrchestrationSpec{
		Phases: []PhaseSpec{
			{Name: "a", DependsOn: []string{"b"}},
			{Name: "b", DependsOn: []string{"a"}},
		},
	}

	_, err := orch.Run(context.Background(), spec)
	if !errors.Is(err, agent.ErrOrchestrationAborted) {
		t.Errorf("expected ErrOrchestrationAborted, got %v", err)
	}
	if err == nil || !strings.Contains(err.Error(), "cycle") {
		t.Errorf("expected error message to name the cycle; got %q", err)
	}
}

// TestSequentialOrchestrator_ContextCancellation verifies that
// cancelling the context mid-phase causes the channel to close in
// finite time. The orchestrator's ctx-watcher calls child.Abort on
// every in-flight child (D6); the blockingClient blocks on ctx, so
// cancellation must propagate to unwind the child's Stream goroutine.
func TestSequentialOrchestrator_ContextCancellation(t *testing.T) {
	client := &blockingClient{}
	parent := newTestSession(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}})
	defer parent.Shutdown(context.Background())

	orch := NewSequentialOrchestrator(parent)

	ctx, cancel := context.WithCancel(context.Background())
	spec := OrchestrationSpec{
		Phases: []PhaseSpec{
			{Name: "a", Prompt: "block-forever"},
		},
	}

	ch, err := orch.Run(ctx, spec)
	if err != nil {
		t.Fatalf("Run returned immediate err: %v", err)
	}

	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	drained := make(chan struct{})
	go func() {
		for range ch {
		}
		close(drained)
	}()

	select {
	case <-drained:
		// OK — channel closed after cancel, proving the ctx-watcher's
		// child.Abort unwound the in-flight child.
	case <-time.After(5 * time.Second):
		t.Fatal("channel did not close within 5s after cancel")
	}
}

// blockingClient is a scriptedClient whose Stream never returns until
// ctx is cancelled.
type blockingClient struct{}

func (c *blockingClient) Stream(ctx context.Context, req llm.Request) (<-chan llm.Delta, error) {
	out := make(chan llm.Delta)
	go func() {
		defer close(out)
		<-ctx.Done()
	}()
	return out, nil
}

// TestReplayMergeConflict exercises the replay-merge conflict path via
// a direct MergeState call. Parent has a tool_call touching foo.txt;
// child also touches foo.txt; MergePolicyReplay should return
// ErrOrchestrationConflict with a populated *ConflictReportShell whose
// Phase and LineRange are non-zero (D8/D9).
func TestReplayMergeConflict(t *testing.T) {
	client := &scriptedClient{script: textThenFinal("ok")}
	parent := newTestSession(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}})
	defer parent.Shutdown(context.Background())

	// Write foo.txt to the parent's Cwd so the edit line lookup succeeds.
	fooPath := filepath.Join(parent.Runtime().Cwd, "foo.txt")
	fileContent := "line1\nline2\nline3\n"
	if err := os.WriteFile(fooPath, []byte(fileContent), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	// Pre-populate the parent's state with a tool-call entry touching foo.txt.
	toolUse := llm.ToolUse{
		ID:    "tu_parent",
		Name:  "edit",
		Input: json.RawMessage(`{"file_path":"foo.txt","old_string":"line2","new_string":"line2-edited"}`),
	}
	msgPayload := state.MessagePayload{
		Role:    llm.RoleAssistant,
		Content: []llm.ContentBlock{toolUse},
	}
	if _, err := parent.Runtime().State.Append(state.NewEntry("", msgPayload)); err != nil {
		t.Fatalf("parent Append: %v", err)
	}

	// Build a child with the same tool-call shape.
	child := newTestSession(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}})
	defer child.Shutdown(context.Background())
	childToolUse := llm.ToolUse{
		ID:    "tu_child",
		Name:  "edit",
		Input: json.RawMessage(`{"file_path":"foo.txt","old_string":"line2","new_string":"line2-child"}`),
	}
	childMsg := state.MessagePayload{
		Role:    llm.RoleAssistant,
		Content: []llm.ContentBlock{childToolUse},
	}
	if _, err := child.Runtime().State.Append(state.NewEntry("", childMsg)); err != nil {
		t.Fatalf("child Append: %v", err)
	}

	err := parent.MergeState(context.Background(), child, agent.MergeSpecShell{
		Policy: agent.MergePolicyReplayShell,
		Phase:  "review",
	})
	if !errors.Is(err, agent.ErrOrchestrationConflict) {
		t.Errorf("expected ErrOrchestrationConflict, got %v", err)
	}
	if err == nil {
		t.Fatal("expected non-nil error")
	}
	var report *agent.ConflictReportShell
	if !errors.As(err, &report) {
		t.Fatalf("expected errors.As to populate *ConflictReportShell; err=%v", err)
	}
	if report.File != "foo.txt" {
		t.Errorf("report.File = %q; want foo.txt", report.File)
	}
	// D8: Phase must be populated from MergeSpecShell.Phase.
	if report.Phase != "review" {
		t.Errorf("report.Phase = %q; want \"review\"", report.Phase)
	}
	// D9: LineRange must be populated. "line2" is on line 2 of the file.
	if report.LineRange[0] != 2 {
		t.Errorf("report.LineRange[0] = %d; want 2", report.LineRange[0])
	}
	if report.LineRange[1] < report.LineRange[0] {
		t.Errorf("report.LineRange[1] = %d; want >= LineRange[0] (%d)", report.LineRange[1], report.LineRange[0])
	}
}

// TestSequentialRunErrorSurfaced asserts that when a child's Run
// returns a non-cancel error, the error is stored and surfaced via
// Orchestrator.Err() after the channel closes (D7, task 4.2).
func TestSequentialRunErrorSurfaced(t *testing.T) {
	client := &erroringClient{}
	parent := newTestSession(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}})
	defer parent.Shutdown(context.Background())

	orch := NewSequentialOrchestrator(parent)

	spec := OrchestrationSpec{
		Phases: []PhaseSpec{
			{Name: "a", Prompt: "will-fail"},
		},
		MergePolicy: MergePolicyNone,
	}

	ch, err := orch.Run(context.Background(), spec)
	if err != nil {
		t.Fatalf("Run upfront err: %v", err)
	}
	for range ch {
	}

	got := orch.Err()
	if got == nil {
		t.Fatal("expected non-nil Err after phase failure; got nil")
	}
	if !errors.Is(got, errBoom) {
		t.Errorf("expected Err to wrap errBoom; got %v", got)
	}
}

// TestParentEventBusForwarding asserts that when OrchestrationSpec.
// ParentEventBus is true, child events are forwarded to the parent
// session's EventBus in addition to the Run channel (D4).
func TestParentEventBusForwarding(t *testing.T) {
	client := &scriptedClient{script: textThenFinal("ok")}
	parent := newTestSession(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}})
	defer parent.Shutdown(context.Background())

	// Subscribe to the parent's bus BEFORE running the orchestration.
	sub := parent.Runtime().EventBus.Subscribe()

	orch := NewSequentialOrchestrator(parent)
	spec := OrchestrationSpec{
		Phases: []PhaseSpec{
			{Name: "a", Prompt: "phase A"},
		},
		MergePolicy:    MergePolicyNone,
		ParentEventBus: true,
	}

	ch, err := orch.Run(context.Background(), spec)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Reader goroutine for the parent bus subscriber.
	readCtx, readCancel := context.WithCancel(context.Background())
	defer readCancel()
	var busEvents atomic.Int32
	go func() {
		for {
			select {
			case <-readCtx.Done():
				return
			case _, ok := <-sub:
				if !ok {
					return
				}
				busEvents.Add(1)
			}
		}
	}()

	// Drain the orchestrator channel.
	for range ch {
	}

	// Give the bus reader a brief moment to drain buffered events.
	time.Sleep(100 * time.Millisecond)
	readCancel()

	got := int(busEvents.Load())
	if got == 0 {
		t.Errorf("expected at least one event forwarded to parent bus with ParentEventBus=true; got %d", got)
	}
}

// TestParentEventBusNotForwardedByDefault asserts that when
// ParentEventBus is false (default), child events are NOT forwarded to
// the parent's bus — the Run channel is the sole sink.
func TestParentEventBusNotForwardedByDefault(t *testing.T) {
	client := &scriptedClient{script: textThenFinal("ok")}
	parent := newTestSession(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}})
	defer parent.Shutdown(context.Background())

	sub := parent.Runtime().EventBus.Subscribe()

	orch := NewSequentialOrchestrator(parent)
	spec := OrchestrationSpec{
		Phases: []PhaseSpec{
			{Name: "a", Prompt: "phase A"},
		},
		MergePolicy:    MergePolicyNone,
		ParentEventBus: false, // default
	}

	ch, err := orch.Run(context.Background(), spec)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	readCtx, readCancel := context.WithCancel(context.Background())
	defer readCancel()
	var busEvents atomic.Int32
	go func() {
		for {
			select {
			case <-readCtx.Done():
				return
			case _, ok := <-sub:
				if !ok {
					return
				}
				busEvents.Add(1)
			}
		}
	}()

	for range ch {
	}
	time.Sleep(100 * time.Millisecond)
	readCancel()

	// The parent's bus should NOT see child events (only the parent's
	// own SessionStart/TurnStart events from its construction).
	// We can't assert zero because the parent session itself publishes
	// a SessionStart on its own bus at construction. Instead assert
	// that no TURN events from the child arrived: the child's
	// turn-specific events (MessageStart, MessageUpdate) would only
	// appear if forwarding were active. We check conservatively: the
	// parent bus should have fewer events than a forwarded run.
	got := int(busEvents.Load())
	// Without forwarding, the parent bus sees only the parent's own
	// events (SessionStart at most). If forwarding were active, we'd
	// see the child's MessageStart + MessageUpdate + MessageEnd +
	// TurnStart + TurnEnd as well (5+ more events). Assert we got
	// fewer than that.
	if got > 2 {
		t.Errorf("expected at most 2 events on parent bus without forwarding; got %d (forwarding leaked?)", got)
	}
}

// TestMergeStateClosedChildReturnsErrOrchestratorClosed asserts that
// calling MergeState on a child whose state manager has been closed
// returns ErrOrchestratorClosed (D5, task 3.2).
func TestMergeStateClosedChildReturnsErrOrchestratorClosed(t *testing.T) {
	client := &scriptedClient{script: textThenFinal("ok")}
	parent := newTestSession(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}})
	defer parent.Shutdown(context.Background())

	child := newTestSession(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}})

	// Close the child's state manager explicitly.
	if err := child.Runtime().State.Close(); err != nil {
		t.Fatalf("child state Close: %v", err)
	}

	err := parent.MergeState(context.Background(), child, agent.MergeSpecShell{
		Policy: agent.MergePolicyAppendShell,
	})
	if !errors.Is(err, agent.ErrOrchestratorClosed) {
		t.Errorf("expected ErrOrchestratorClosed; got %v", err)
	}
}

// TestSharedStoreBranchHandoff verifies that after phase A's shadow is
// merged into the parent (Append), a subsequently spawned child B sees
// phase A's entries in its own state tree via the shared-store branch
// model (D1/D3). This proves sequential phase handoff works: later
// phases inherit earlier phases' state.
func TestSharedStoreBranchHandoff(t *testing.T) {
	client := &scriptedClient{script: textThenFinal("phase-output")}
	parent := newTestSession(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}})
	defer parent.Shutdown(context.Background())

	parentOpts := parent.Runtime().Options

	// Phase A: spawn child, run, merge (Append).
	childA, err := parent.Spawn(context.Background(), agent.SessionOptions{
		Tools:    parentOpts.BuiltinTools,
		Settings: parentOpts.Settings,
	})
	if err != nil {
		t.Fatalf("Spawn childA: %v", err)
	}
	if err := childA.Run(context.Background(), "phase A prompt"); err != nil {
		t.Fatalf("childA Run: %v", err)
	}
	if err := parent.MergeState(context.Background(), childA, agent.MergeSpecShell{
		Policy: agent.MergePolicyAppendShell,
		Phase:  "A",
	}); err != nil {
		t.Fatalf("MergeState childA: %v", err)
	}
	parentTreeAfterA, _ := parent.Runtime().State.Tree()
	parentLenAfterA := parentTreeAfterA.Len()
	if parentLenAfterA < 2 {
		t.Fatalf("parent tree too small after phase A merge: len=%d", parentLenAfterA)
	}
	_ = childA.Shutdown(context.Background())

	// Phase B: spawn child B. Its branchManager shares the parent's
	// store, so it should see phase A's merged entries.
	childB, err := parent.Spawn(context.Background(), agent.SessionOptions{
		Tools:    parentOpts.BuiltinTools,
		Settings: parentOpts.Settings,
	})
	if err != nil {
		t.Fatalf("Spawn childB: %v", err)
	}
	defer childB.Shutdown(context.Background())

	childBTree, err := childB.Runtime().State.Tree()
	if err != nil {
		t.Fatalf("childB Tree: %v", err)
	}
	// childB's tree should include at least as many entries as the
	// parent had after phase A's merge (root + phase A entries).
	if childBTree.Len() < parentLenAfterA {
		t.Errorf("childB tree len = %d; want >= %d (parent entries after phase A)",
			childBTree.Len(), parentLenAfterA)
	}
}

// TestDependencyLevels is a focused unit test for the level-grouping
// logic that drives concurrent execution. Phases with no edges form a
// single level; phases with edges arrive after their dependencies;
// cycles return an error.
func TestDependencyLevels(t *testing.T) {
	t.Run("no edges forms single level", func(t *testing.T) {
		in := []PhaseSpec{{Name: "a"}, {Name: "b"}, {Name: "c"}}
		levels, err := dependencyLevels(in)
		if err != nil {
			t.Fatalf("dependencyLevels: %v", err)
		}
		if len(levels) != 1 {
			t.Fatalf("got %d levels; want 1", len(levels))
		}
		if len(levels[0]) != 3 {
			t.Fatalf("level 0 has %d phases; want 3", len(levels[0]))
		}
	})

	t.Run("edges create separate levels", func(t *testing.T) {
		in := []PhaseSpec{
			{Name: "b", DependsOn: []string{"a"}},
			{Name: "a"},
			{Name: "c", DependsOn: []string{"b"}},
		}
		levels, err := dependencyLevels(in)
		if err != nil {
			t.Fatalf("dependencyLevels: %v", err)
		}
		if len(levels) != 3 {
			t.Fatalf("got %d levels; want 3 (a → b → c)", len(levels))
		}
		// Level 0: a. Level 1: b. Level 2: c.
		if levels[0][0].Name != "a" {
			t.Errorf("level 0 = %q; want a", levels[0][0].Name)
		}
		if levels[1][0].Name != "b" {
			t.Errorf("level 1 = %q; want b", levels[1][0].Name)
		}
		if levels[2][0].Name != "c" {
			t.Errorf("level 2 = %q; want c", levels[2][0].Name)
		}
	})

	t.Run("diamond dependency", func(t *testing.T) {
		//   a
		//  / \
		// b   c
		//  \ /
		//   d
		in := []PhaseSpec{
			{Name: "a"},
			{Name: "b", DependsOn: []string{"a"}},
			{Name: "c", DependsOn: []string{"a"}},
			{Name: "d", DependsOn: []string{"b", "c"}},
		}
		levels, err := dependencyLevels(in)
		if err != nil {
			t.Fatalf("dependencyLevels: %v", err)
		}
		if len(levels) != 3 {
			t.Fatalf("got %d levels; want 3 (a | b,c | d)", len(levels))
		}
		// Level 0: a alone.
		if len(levels[0]) != 1 || levels[0][0].Name != "a" {
			t.Errorf("level 0 = %v; want [a]", names(levels[0]))
		}
		// Level 1: b and c (both depend only on a).
		if len(levels[1]) != 2 {
			t.Errorf("level 1 has %d phases; want 2 (b,c)", len(levels[1]))
		}
		// Level 2: d.
		if len(levels[2]) != 1 || levels[2][0].Name != "d" {
			t.Errorf("level 2 = %v; want [d]", names(levels[2]))
		}
	})

	t.Run("cycle rejected", func(t *testing.T) {
		in := []PhaseSpec{
			{Name: "a", DependsOn: []string{"b"}},
			{Name: "b", DependsOn: []string{"a"}},
		}
		_, err := dependencyLevels(in)
		if err == nil {
			t.Fatal("expected error for cycle; got nil")
		}
		if !strings.Contains(err.Error(), "cycle") {
			t.Errorf("expected 'cycle' in error; got %q", err)
		}
	})

	t.Run("duplicate names rejected", func(t *testing.T) {
		in := []PhaseSpec{{Name: "a"}, {Name: "a"}}
		_, err := dependencyLevels(in)
		if err == nil || !strings.Contains(err.Error(), "duplicate") {
			t.Errorf("expected duplicate-name error; got %v", err)
		}
	})

	t.Run("unknown dependency rejected", func(t *testing.T) {
		in := []PhaseSpec{{Name: "a", DependsOn: []string{"nonexistent"}}}
		_, err := dependencyLevels(in)
		if err == nil || !strings.Contains(err.Error(), "unknown phase") {
			t.Errorf("expected unknown-phase error; got %v", err)
		}
	})
}

// names extracts the Name field from a slice of PhaseSpecs for
// assertion-friendly output.
func names(phases []PhaseSpec) []string {
	out := make([]string, len(phases))
	for i, p := range phases {
		out[i] = p.Name
	}
	return out
}
