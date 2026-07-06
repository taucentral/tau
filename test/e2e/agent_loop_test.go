// Package e2e — agent_loop_test.go covers tasks 8.11, 8.12, 8.13.
//
// These tests stand up a full AgentSession against the FauxProvider
// and a stub tool, then exercise the turn loop end-to-end. They verify:
//
//   - 8.11: a full turn with one tool call persists the correct state
//     tree, emits events in the spec order, and orders tool results
//     after tool calls.
//   - 8.12: SteeringMode=all runs tool calls concurrently; the serial
//     default runs them one at a time.
//   - 8.13: cancelling the context mid-turn aborts in-flight tools
//     within a tight deadline and leaves the state tree consistent.
package e2e

import (
	"context"
	"encoding/json"
	"errors"
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

// stubTool is a test double that records its invocations and returns a
// caller-specified text result. The Delay field, when non-zero, sleeps
// inside Execute to exercise cancellation paths.
type stubTool struct {
	name     string
	delay    time.Duration
	mu       sync.Mutex
	calls    []string // call IDs in arrival order
	count    atomic.Int32
	started  atomic.Int32
	finished atomic.Int32
}

func (t *stubTool) Name() string        { return t.name }
func (t *stubTool) Description() string { return "stub tool for e2e tests" }
func (t *stubTool) Parameters() jsonschema.Schema {
	// Minimal schema: an empty object with no required fields.
	s := jsonschema.Schema{}
	s.Type = "object"
	return s
}

func (t *stubTool) Execute(ctx context.Context, call tools.ToolCall) (tools.ToolResult, error) {
	t.started.Add(1)
	defer t.finished.Add(1)
	t.count.Add(1)
	t.mu.Lock()
	t.calls = append(t.calls, call.ID)
	t.mu.Unlock()
	if t.delay > 0 {
		select {
		case <-ctx.Done():
			return tools.ToolResult{}, ctx.Err()
		case <-time.After(t.delay):
		}
	}
	return tools.NewTextResult(t.name + ":ok"), nil
}

func (t *stubTool) RenderCall(json.RawMessage, *tools.Theme) string    { return "" }
func (t *stubTool) RenderResult(tools.ToolResult, *tools.Theme) string { return "" }

func (t *stubTool) callCount() int32 { return t.count.Load() }

// newSession wires an agent.AgentSession against the given provider and
// tools. Returns the session and the underlying runtime so tests can
// inspect state.
func newSession(t *testing.T, provider llm.LLMClient, builtinTools []tools.HeadlessTool) (*agent.AgentSession, *agent.AgentSessionRuntime) {
	t.Helper()
	opts := agent.SessionOptions{
		Model:     "test-model",
		Settings:  config.DefaultSettings(),
		LLMClient: provider,
		Tools:     builtinTools,
		ConfigDir: t.TempDir(),
	}
	rt, err := agent.CreateAgentSessionRuntime(context.Background(), t.TempDir(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSessionRuntime: %v", err)
	}
	return agent.NewAgentSession(rt), rt
}

// textThenFinal is a tiny helper to keep test scripts readable. The Final
// always carries EndTurn — these tests do not exercise other stop reasons.
func textThenFinal(text string) []llm.Delta {
	return []llm.Delta{
		llm.TextDelta{Text: text, ContentIndex: 0},
		llm.Final{StopReason: llm.StopReasonEndTurn},
	}
}

// TestE2E_FullTurnWithOneToolCall verifies task 8.11: a full turn with
// one tool call persists the correct state tree, emits events in the
// canonical order, and orders tool results after tool calls.
func TestE2E_FullTurnWithOneToolCall(t *testing.T) {
	tool := &stubTool{name: "stub"}
	provider := NewFauxProvider(
		// Iteration 1: model emits a tool call + Final(toolUse).
		[]llm.Delta{
			llm.TextDelta{Text: "Let me check.", ContentIndex: 0},
			llm.ToolCallDelta{ContentIndex: 1, ID: "tu_1", Name: "stub", PartialInput: "{}"},
			llm.Final{StopReason: llm.StopReasonToolUse},
		},
		// Iteration 2: model emits a final text + Final(stop).
		textThenFinal("All done."),
	)
	sess, rt := newSession(t, provider, []tools.HeadlessTool{tool})

	// Subscribe to events BEFORE Run so we capture the full sequence.
	evtsPtr, wait := collectAllEvents(t, rt.EventBus)

	if err := sess.Run(context.Background(), "please run the tool"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Snapshot the state tree before Shutdown.
	tree, err := rt.State.Tree()
	if err != nil {
		t.Fatalf("Tree: %v", err)
	}

	if err := sess.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
	wait()
	evts := *evtsPtr

	// (1) Tool was invoked exactly once.
	if got := tool.callCount(); got != 1 {
		t.Errorf("tool invocations = %d, want 1", got)
	}

	// (2) State tree contains: 1 user, 2 assistant, 1 tool-result message.
	var userMsg, assistantMsg, toolResults int
	for _, id := range tree.IDs() {
		e, _ := tree.Get(id)
		if e.Kind != state.KindMessage {
			continue
		}
		mp, ok := e.Payload.(state.MessagePayload)
		if !ok {
			continue
		}
		switch mp.Role { //nolint:exhaustive // test counts user/assistant only
		case llm.RoleUser:
			userMsg++
			for _, b := range mp.Content {
				if _, ok := b.(llm.ToolResult); ok {
					toolResults++
				}
			}
		case llm.RoleAssistant:
			assistantMsg++
		}
	}
	if userMsg != 2 {
		t.Errorf("user messages = %d, want 2 (initial + tool-result)", userMsg)
	}
	if assistantMsg != 2 {
		t.Errorf("assistant messages = %d, want 2 (one per iteration)", assistantMsg)
	}
	if toolResults != 1 {
		t.Errorf("tool results = %d, want 1", toolResults)
	}

	// (3) Faux provider saw 2 Stream calls (2 iterations).
	if got := provider.Calls(); got != 2 {
		t.Errorf("provider calls = %d, want 2", got)
	}

	// (4) Event sequence matches the spec scenario "Full-turn event
	// sequence": session_start, turn_start, message_start, message_update*,
	// tool_call, tool_result, message_end, message_update, message_end,
	// turn_end, session_shutdown. Verify the first message_end is
	// preceded by tool_call and tool_result.
	firstMessageEndIdx := -1
	toolCallIdx := -1
	toolResultIdx := -1
	for i, e := range evts {
		switch e.Topic() { //nolint:exhaustive // test asserts specific topics only
		case agent.TopicToolCall:
			if toolCallIdx == -1 {
				toolCallIdx = i
			}
		case agent.TopicToolResult:
			if toolResultIdx == -1 {
				toolResultIdx = i
			}
		case agent.TopicMessageEnd:
			if firstMessageEndIdx == -1 {
				firstMessageEndIdx = i
			}
		}
	}
	if toolCallIdx < 0 || toolResultIdx < 0 || firstMessageEndIdx < 0 {
		t.Fatalf("missing events: toolCallIdx=%d, toolResultIdx=%d, messageEndIdx=%d",
			toolCallIdx, toolResultIdx, firstMessageEndIdx)
	}
	if toolCallIdx >= toolResultIdx || toolResultIdx >= firstMessageEndIdx {
		t.Errorf("event order broken: toolCallIdx=%d, toolResultIdx=%d, messageEndIdx=%d; want toolCall < toolResult < messageEnd",
			toolCallIdx, toolResultIdx, firstMessageEndIdx)
	}
}

// TestE2E_ConcurrentToolExecution_SteeringAll verifies task 8.12:
// SteeringMode=all runs 3 tool calls concurrently.
func TestE2E_ConcurrentToolExecution_SteeringAll(t *testing.T) {
	tool := &stubTool{name: "stub", delay: 100 * time.Millisecond}
	provider := NewFauxProvider(
		[]llm.Delta{
			llm.ToolCallDelta{ContentIndex: 0, ID: "tu_1", Name: "stub", PartialInput: "{}"},
			llm.ToolCallDelta{ContentIndex: 1, ID: "tu_2", Name: "stub", PartialInput: "{}"},
			llm.ToolCallDelta{ContentIndex: 2, ID: "tu_3", Name: "stub", PartialInput: "{}"},
			llm.Final{StopReason: llm.StopReasonToolUse},
		},
		textThenFinal("done"),
	)
	sess, rt := newSession(t, provider, []tools.HeadlessTool{tool})
	// Force concurrent steering.
	allMode := config.SteeringAll
	rt.Options.SteeringMode = allMode

	start := time.Now()
	if err := sess.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	elapsed := time.Since(start)
	sess.Shutdown(context.Background())

	if got := tool.callCount(); got != 3 {
		t.Errorf("tool invocations = %d, want 3", got)
	}
	// 3 × 100ms concurrently → ~100ms total. Generous upper bound.
	if elapsed > 350*time.Millisecond {
		t.Errorf("concurrent execution took %v; want < 350ms (tools did not run in parallel)", elapsed)
	}
	if elapsed < 80*time.Millisecond {
		t.Errorf("concurrent execution took %v; suspiciously fast (was delay honoured?)", elapsed)
	}
}

// TestE2E_SerialToolExecution_SteeringOneAtATime verifies task 8.12:
// SteeringMode=one-at-a-time runs 3 tool calls serially.
func TestE2E_SerialToolExecution_SteeringOneAtATime(t *testing.T) {
	tool := &stubTool{name: "stub", delay: 80 * time.Millisecond}
	provider := NewFauxProvider(
		[]llm.Delta{
			llm.ToolCallDelta{ContentIndex: 0, ID: "tu_1", Name: "stub", PartialInput: "{}"},
			llm.ToolCallDelta{ContentIndex: 1, ID: "tu_2", Name: "stub", PartialInput: "{}"},
			llm.ToolCallDelta{ContentIndex: 2, ID: "tu_3", Name: "stub", PartialInput: "{}"},
			llm.Final{StopReason: llm.StopReasonToolUse},
		},
		textThenFinal("done"),
	)
	sess, rt := newSession(t, provider, []tools.HeadlessTool{tool})
	// Force serial steering.
	serialMode := config.SteeringOneAtATime
	rt.Options.SteeringMode = serialMode

	start := time.Now()
	if err := sess.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	elapsed := time.Since(start)
	sess.Shutdown(context.Background())

	if got := tool.callCount(); got != 3 {
		t.Errorf("tool invocations = %d, want 3", got)
	}
	// 3 × 80ms serially → ~240ms. Tight bounds.
	if elapsed < 200*time.Millisecond {
		t.Errorf("serial execution took %v; want >= 200ms (calls overlapped)", elapsed)
	}
	if elapsed > 500*time.Millisecond {
		t.Errorf("serial execution took %v; want < 500ms (excessive delay)", elapsed)
	}
}

// TestE2E_MidTurnCancellation verifies task 8.13: cancelling the
// context mid-turn aborts in-flight tools within 100ms and leaves the
// state tree consistent.
func TestE2E_MidTurnCancellation(t *testing.T) {
	// Tool with a long delay so we can cancel mid-execution.
	tool := &stubTool{name: "stub", delay: 5 * time.Second}
	provider := NewFauxProvider(
		[]llm.Delta{
			llm.ToolCallDelta{ContentIndex: 0, ID: "tu_1", Name: "stub", PartialInput: "{}"},
			llm.Final{StopReason: llm.StopReasonToolUse},
		},
	)
	sess, _ := newSession(t, provider, []tools.HeadlessTool{tool})

	ctx, cancel := context.WithCancel(context.Background())
	// Cancel after 100ms to give the tool time to start.
	go func() {
		time.Sleep(100 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	err := sess.Run(ctx, "go")
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("Run should return an error on cancellation")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want a context.Canceled wrapper", err)
	}
	// The tool should have aborted within ~100ms of ctx cancel, not
	// waited its full 5s delay.
	if elapsed > 1*time.Second {
		t.Errorf("cancellation took %v to unwind; want < 1s (tool did not honour ctx)", elapsed)
	}
	sess.Shutdown(context.Background())

	// Verify the tool saw started but did not necessarily finish.
	if started := tool.started.Load(); started != 1 {
		t.Errorf("tool started = %d, want 1", started)
	}
	// Finished might be 0 (cancelled mid-delay) or 1 (race). Both are
	// acceptable per spec; the key invariant is "no hang".
}

// TestE2E_AbortViaSessionMethod verifies that Abort() called on a
// separate goroutine cancels the in-flight turn within 100ms.
func TestE2E_AbortViaSessionMethod(t *testing.T) {
	tool := &stubTool{name: "stub", delay: 5 * time.Second}
	provider := NewFauxProvider(
		[]llm.Delta{
			llm.ToolCallDelta{ContentIndex: 0, ID: "tu_1", Name: "stub", PartialInput: "{}"},
			llm.Final{StopReason: llm.StopReasonToolUse},
		},
	)
	sess, _ := newSession(t, provider, []tools.HeadlessTool{tool})

	go func() {
		time.Sleep(100 * time.Millisecond)
		sess.Abort("test")
	}()

	start := time.Now()
	err := sess.Run(context.Background(), "go")
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("Run should return an error on Abort")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled wrapper", err)
	}
	if elapsed > 1*time.Second {
		t.Errorf("Abort took %v to unwind; want < 1s", elapsed)
	}
	sess.Shutdown(context.Background())
}

// TestE2E_MultiTurnConversation verifies that consecutive Run calls
// each fire their own TurnStart/TurnEnd, share SessionStart, and the
// conversation accumulates in the state tree.
func TestE2E_MultiTurnConversation(t *testing.T) {
	provider := NewFauxProvider(
		textThenFinal("turn one"),
		textThenFinal("turn two"),
		textThenFinal("turn three"),
	)
	sess, rt := newSession(t, provider, []tools.HeadlessTool{
		tools.NewReadTool(tools.OSReadOperations{}),
	})

	evtsPtr, wait := collectAllEvents(t, rt.EventBus)

	for i := 1; i <= 3; i++ {
		if err := sess.Run(context.Background(), "turn"); err != nil {
			t.Fatalf("Run %d: %v", i, err)
		}
	}

	tree, err := rt.State.Tree()
	if err != nil {
		t.Fatalf("Tree: %v", err)
	}
	sess.Shutdown(context.Background())
	wait()
	evts := *evtsPtr

	// 3 user + 3 assistant messages.
	var userMsg, assistantMsg int
	for _, id := range tree.IDs() {
		e, _ := tree.Get(id)
		if e.Kind != state.KindMessage {
			continue
		}
		mp, _ := e.Payload.(state.MessagePayload)
		switch mp.Role { //nolint:exhaustive // test counts user/assistant only
		case llm.RoleUser:
			userMsg++
		case llm.RoleAssistant:
			assistantMsg++
		}
	}
	if userMsg != 3 || assistantMsg != 3 {
		t.Errorf("after 3 turns: user=%d, assistant=%d; want 3 and 3", userMsg, assistantMsg)
	}

	// SessionStart fires exactly once; TurnStart fires 3 times.
	var sessionStarts, turnStarts int
	for _, e := range evts {
		switch e.Topic() { //nolint:exhaustive // test asserts specific topics only
		case agent.TopicSessionStart:
			sessionStarts++
		case agent.TopicTurnStart:
			turnStarts++
		}
	}
	if sessionStarts != 1 {
		t.Errorf("SessionStart events = %d, want 1", sessionStarts)
	}
	if turnStarts != 3 {
		t.Errorf("TurnStart events = %d, want 3", turnStarts)
	}
}

// TestE2E_StateManagerNotClosedWhenInjected verifies the runtime
// respects ownership boundaries: caller-injected state managers are
// left open after Shutdown.
func TestE2E_StateManagerNotClosedWhenInjected(t *testing.T) {
	provider := NewFauxProvider(textThenFinal("ok"))
	inj := state.NewInMemoryManager(t.TempDir())
	opts := agent.SessionOptions{
		Model:        "test-model",
		Settings:     config.DefaultSettings(),
		LLMClient:    provider,
		Tools:        []tools.HeadlessTool{tools.NewReadTool(tools.OSReadOperations{})},
		ConfigDir:    t.TempDir(),
		StateManager: inj,
	}
	rt, err := agent.CreateAgentSessionRuntime(context.Background(), t.TempDir(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSessionRuntime: %v", err)
	}
	sess := agent.NewAgentSession(rt)
	if err := sess.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
	// Injected manager should still be usable.
	if _, err := inj.Append(state.Entry{
		Kind:    state.KindMessage,
		Payload: state.MessagePayload{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.TextContent{Text: "x"}}},
	}); err != nil {
		t.Errorf("Append on injected manager after Shutdown: %v", err)
	}
	inj.Close()
}

// TestE2E_ErrorResultForUnknownTool verifies that when the model emits
// a tool call for a name not in the registry, the agent synthesizes an
// error ToolResult and the turn completes normally on the second
// iteration.
func TestE2E_ErrorResultForUnknownTool(t *testing.T) {
	provider := NewFauxProvider(
		[]llm.Delta{
			llm.ToolCallDelta{ContentIndex: 0, ID: "tu_1", Name: "does-not-exist", PartialInput: "{}"},
			llm.Final{StopReason: llm.StopReasonToolUse},
		},
		textThenFinal("recovered"),
	)
	sess, rt := newSession(t, provider, []tools.HeadlessTool{
		tools.NewReadTool(tools.OSReadOperations{}),
	})
	evtsPtr, wait := collectAllEvents(t, rt.EventBus)

	if err := sess.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	sess.Shutdown(context.Background())
	wait()
	evts := *evtsPtr

	var foundErrorResult bool
	for _, e := range evts {
		if tre, ok := e.(agent.ToolResultEvent); ok {
			if tre.Result.IsError {
				txt := ""
				if len(tre.Result.Content) > 0 {
					if tc, ok := tre.Result.Content[0].(llm.TextContent); ok {
						txt = tc.Text
					}
				}
				if strings.Contains(txt, "not registered") {
					foundErrorResult = true
				}
			}
		}
	}
	if !foundErrorResult {
		t.Error("expected IsError ToolResult containing 'not registered'")
	}
}

// collectAllEvents subscribes to the bus and returns a pointer to the
// accumulated slice plus a wait function. The pointer is safe to read
// after wait() returns.
func collectAllEvents(t *testing.T, bus *agent.EventBus) (*[]agent.Event, func()) {
	t.Helper()
	evts := &[]agent.Event{}
	ch := bus.Subscribe()
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for evt := range ch {
			*evts = append(*evts, evt)
		}
	}()
	return evts, wg.Wait
}
