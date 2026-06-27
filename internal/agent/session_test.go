package agent

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

	"github.com/coevin/tau/internal/config"
	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/state"
	"github.com/coevin/tau/internal/tools"
)

// scriptedClient emits a pre-set sequence of deltas per call to Stream.
// Each call advances the script index; tests set up the full script for
// a multi-iteration turn.
type scriptedClient struct {
	mu      sync.Mutex
	scripts [][]llm.Delta
	idx     int
}

func (s *scriptedClient) Stream(ctx context.Context, req llm.Request) (<-chan llm.Delta, error) {
	s.mu.Lock()
	if s.idx >= len(s.scripts) {
		s.mu.Unlock()
		return nil, errors.New("scriptedClient: no script left")
	}
	script := s.scripts[s.idx]
	s.idx++
	s.mu.Unlock()
	ch := make(chan llm.Delta, len(script))
	go func() {
		defer close(ch)
		for _, d := range script {
			select {
			case <-ctx.Done():
				return
			case ch <- d:
			}
		}
	}()
	return ch, nil
}

// textThenFinal returns a script that emits one text delta and a Final whose
// StopReason is always EndTurn (the only stop reason these scripts care about).
func textThenFinal(text string) []llm.Delta {
	return []llm.Delta{
		llm.TextDelta{Text: text, ContentIndex: 0},
		llm.Final{StopReason: llm.StopReasonEndTurn},
	}
}

// recordingTool records every Execute call. Returns a fixed result text.
type recordingTool struct {
	name    string
	calls   atomic.Int32
	mu      sync.Mutex
	callIDs []string
	delay   time.Duration // optional, to test cancellation timing
}

func (t *recordingTool) Name() string                  { return t.name }
func (t *recordingTool) Description() string           { return "recording tool for tests" }
func (t *recordingTool) Parameters() jsonschema.Schema { return jsonschema.Schema{} }
func (t *recordingTool) Execute(ctx context.Context, call tools.ToolCall) (tools.ToolResult, error) {
	t.calls.Add(1)
	t.mu.Lock()
	t.callIDs = append(t.callIDs, call.ID)
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
func (t *recordingTool) RenderCall(json.RawMessage, *tools.Theme) string    { return "" }
func (t *recordingTool) RenderResult(tools.ToolResult, *tools.Theme) string { return "" }

func (t *recordingTool) callCount() int32 { return t.calls.Load() }

// newSessionForTest wires a runtime + session against the given client
// and tool set. Returns the session and the underlying runtime.
func newSessionForTest(t *testing.T, client llm.LLMClient, builtinTools []tools.HeadlessTool) (*AgentSession, *AgentSessionRuntime) {
	t.Helper()
	opts := SessionOptions{
		Model:     "test-model",
		Settings:  config.DefaultSettings(),
		LLMClient: client,
		Tools:     builtinTools,
		ConfigDir: t.TempDir(),
	}
	rt, err := CreateAgentSessionRuntime(context.Background(), t.TempDir(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSessionRuntime: %v", err)
	}
	return NewAgentSession(rt), rt
}

// collectEvents subscribes to bus and starts a goroutine that drains every
// event into a slice. Returns the slice pointer and a wait function. The
// caller must call wait() AFTER triggering Shutdown (which closes the
// subscriber channel) and BEFORE reading the returned slice.
func collectEvents(t *testing.T, bus *EventBus) (*[]Event, func()) {
	t.Helper()
	evts := &[]Event{}
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

// TestRun_SingleTurnNoToolCalls verifies the R1 "single-turn" scenario:
// user input → model emits text deltas + Final(stop) → turn_end.
// Verifies the event sequence and that the assistant message persists.
func TestRun_SingleTurnNoToolCalls(t *testing.T) {
	client := &scriptedClient{
		scripts: [][]llm.Delta{
			textThenFinal("hello world"),
		},
	}
	sess, rt := newSessionForTest(t, client, []tools.HeadlessTool{
		tools.NewReadTool(tools.OSReadOperations{}),
	})

	evtsPtr, wait := collectEvents(t, rt.EventBus)

	if err := sess.Run(context.Background(), "hi"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Snapshot the state tree BEFORE Shutdown closes the store.
	tree, err := rt.State.Tree()
	if err != nil {
		t.Fatalf("Tree: %v", err)
	}

	// Shutdown so the event goroutine exits.
	if err := sess.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
	wait()
	evts := *evtsPtr

	// Verify event sequence.
	var topics []Topic
	for _, e := range evts {
		topics = append(topics, e.Topic())
	}
	// Expect: session_start (first turn), turn_start, message_start,
	// message_update (text), message_update (final), message_end,
	// turn_end, session_shutdown.
	wantPrefix := []Topic{
		TopicSessionStart, TopicTurnStart, TopicMessageStart, TopicMessageUpdate, TopicMessageUpdate,
		TopicMessageEnd, TopicTurnEnd, TopicSessionShutdown,
	}
	if len(topics) < len(wantPrefix) {
		t.Fatalf("event topics = %v, want at least %v", topics, wantPrefix)
	}
	for i, w := range wantPrefix {
		if topics[i] != w {
			t.Errorf("topics[%d] = %q, want %q (full: %v)", i, topics[i], w, topics)
		}
	}

	// Verify state tree has user + assistant messages (tree snapshotted above).
	var userMsg, assistantMsg int
	for _, id := range tree.IDs() {
		e, _ := tree.Get(id)
		if e.Kind == state.KindMessage {
			mp, ok := e.Payload.(state.MessagePayload)
			if !ok {
				continue
			}
			if mp.Role == llm.RoleUser {
				userMsg++
			}
			if mp.Role == llm.RoleAssistant {
				assistantMsg++
			}
		}
	}
	if userMsg != 1 {
		t.Errorf("user messages = %d, want 1", userMsg)
	}
	if assistantMsg != 1 {
		t.Errorf("assistant messages = %d, want 1", assistantMsg)
	}
}

// TestRun_SingleToolCallIteration verifies R1 "multi-turn tool execution":
// model emits ToolCallDelta, agent executes the tool, appends the result,
// re-dispatches, and the model returns Final(stop).
func TestRun_SingleToolCallIteration(t *testing.T) {
	tool := &recordingTool{name: "recorder"}
	client := &scriptedClient{
		scripts: [][]llm.Delta{
			// Iteration 1: a tool call + Final(toolUse).
			{
				llm.ToolCallDelta{ContentIndex: 0, ID: "tu_1", Name: "recorder", PartialInput: "{}"},
				llm.Final{StopReason: llm.StopReasonToolUse},
			},
			// Iteration 2: text + Final(stop).
			textThenFinal("done"),
		},
	}
	sess, rt := newSessionForTest(t, client, []tools.HeadlessTool{tool})

	evtsPtr, wait := collectEvents(t, rt.EventBus)

	if err := sess.Run(context.Background(), "use the tool"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Snapshot the state tree BEFORE Shutdown closes the store.
	tree, err := rt.State.Tree()
	if err != nil {
		t.Fatalf("Tree: %v", err)
	}

	sess.Shutdown(context.Background())
	wait()
	evts := *evtsPtr

	if got := tool.callCount(); got != 1 {
		t.Errorf("tool call count = %d, want 1", got)
	}

	// Verify the event sequence includes ToolCall + ToolResult BEFORE MessageEnd.
	var sawToolCall, sawToolResult, sawMessageEnd bool
	sawMessageEndIdx := -1
	for i, e := range evts {
		if e.Topic() == TopicToolCall && !sawToolCall {
			sawToolCall = true
		}
		if e.Topic() == TopicToolResult && !sawToolResult {
			sawToolResult = true
		}
		if e.Topic() == TopicMessageEnd {
			sawMessageEnd = true
			sawMessageEndIdx = i
			break
		}
	}
	if !sawToolCall {
		t.Error("did not see TopicToolCall before first MessageEnd")
	}
	if !sawToolResult {
		t.Error("did not see TopicToolResult before first MessageEnd")
	}
	if !sawMessageEnd {
		t.Fatal("did not see TopicMessageEnd at all")
	}
	_ = sawMessageEndIdx

	// Verify state has the tool-result message between iterations
	// (tree snapshotted above, before Shutdown closed the store).
	var toolResults int
	for _, id := range tree.IDs() {
		e, _ := tree.Get(id)
		if e.Kind == state.KindMessage {
			mp, ok := e.Payload.(state.MessagePayload)
			if !ok {
				continue
			}
			if mp.Role == llm.RoleUser {
				// The user message and the tool-result message are both
				// RoleUser per Anthropic convention.
				for _, b := range mp.Content {
					if _, ok := b.(llm.ToolResult); ok {
						toolResults++
					}
				}
			}
		}
	}
	if toolResults != 1 {
		t.Errorf("tool results in state = %d, want 1", toolResults)
	}
}

// TestRun_AbortMidTurn verifies R1 "cancellation mid-turn": cancelling
// the context returns a wrapped context.Canceled and the state tree
// remains consistent.
func TestRun_AbortMidTurn(t *testing.T) {
	// Use a tool with a long delay so we can cancel mid-execution.
	tool := &recordingTool{name: "slow", delay: 5 * time.Second}
	client := &scriptedClient{
		scripts: [][]llm.Delta{
			{
				llm.ToolCallDelta{ContentIndex: 0, ID: "tu_1", Name: "slow", PartialInput: "{}"},
				llm.Final{StopReason: llm.StopReasonToolUse},
			},
		},
	}
	sess, _ := newSessionForTest(t, client, []tools.HeadlessTool{tool})

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(100 * time.Millisecond)
		cancel()
	}()
	err := sess.Run(ctx, "go")
	if err == nil {
		t.Fatal("Run should fail with cancellation")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want a context.Canceled wrapper", err)
	}
	sess.Shutdown(context.Background())
}

// TestRun_AbortViaAbortMethod verifies that calling Abort() from another
// goroutine cancels the in-flight turn.
func TestRun_AbortViaAbortMethod(t *testing.T) {
	tool := &recordingTool{name: "slow", delay: 5 * time.Second}
	client := &scriptedClient{
		scripts: [][]llm.Delta{
			{
				llm.ToolCallDelta{ContentIndex: 0, ID: "tu_1", Name: "slow", PartialInput: "{}"},
				llm.Final{StopReason: llm.StopReasonToolUse},
			},
		},
	}
	sess, _ := newSessionForTest(t, client, []tools.HeadlessTool{tool})

	go func() {
		time.Sleep(100 * time.Millisecond)
		sess.Abort("test")
	}()
	err := sess.Run(context.Background(), "go")
	if err == nil {
		t.Fatal("Run should fail with cancellation")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
	sess.Shutdown(context.Background())
}

// TestRun_ConcurrentToolExecution verifies R3 "parallel execution
// default": with SteeringMode = all, multiple tool calls execute
// concurrently.
func TestRun_ConcurrentToolExecution(t *testing.T) {
	tool := &recordingTool{name: "slow", delay: 100 * time.Millisecond}
	client := &scriptedClient{
		scripts: [][]llm.Delta{
			{
				llm.ToolCallDelta{ContentIndex: 0, ID: "tu_1", Name: "slow", PartialInput: "{}"},
				llm.ToolCallDelta{ContentIndex: 1, ID: "tu_2", Name: "slow", PartialInput: "{}"},
				llm.ToolCallDelta{ContentIndex: 2, ID: "tu_3", Name: "slow", PartialInput: "{}"},
				llm.Final{StopReason: llm.StopReasonToolUse},
			},
			textThenFinal("done"),
		},
	}
	sess, rt := newSessionForTest(t, client, []tools.HeadlessTool{tool})
	// Enable concurrent steering.
	allMode := config.SteeringAll
	rt.Options.SteeringMode = allMode

	start := time.Now()
	if err := sess.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	elapsed := time.Since(start)
	sess.Shutdown(context.Background())

	if got := tool.callCount(); got != 3 {
		t.Errorf("tool call count = %d, want 3", got)
	}
	// 3 × 100ms concurrently → ~100ms total. Allow generous slack.
	if elapsed > 250*time.Millisecond {
		t.Errorf("concurrent execution took %v; want < 250ms (concurrency broken)", elapsed)
	}
}

// TestRun_SerialToolExecution verifies R3 serial mode: tool calls run
// one at a time in order.
func TestRun_SerialToolExecution(t *testing.T) {
	tool := &recordingTool{name: "slow", delay: 80 * time.Millisecond}
	client := &scriptedClient{
		scripts: [][]llm.Delta{
			{
				llm.ToolCallDelta{ContentIndex: 0, ID: "tu_1", Name: "slow", PartialInput: "{}"},
				llm.ToolCallDelta{ContentIndex: 1, ID: "tu_2", Name: "slow", PartialInput: "{}"},
				llm.ToolCallDelta{ContentIndex: 2, ID: "tu_3", Name: "slow", PartialInput: "{}"},
				llm.Final{StopReason: llm.StopReasonToolUse},
			},
			textThenFinal("done"),
		},
	}
	sess, rt := newSessionForTest(t, client, []tools.HeadlessTool{tool})
	// Enable serial steering explicitly.
	serialMode := config.SteeringOneAtATime
	rt.Options.SteeringMode = serialMode

	start := time.Now()
	if err := sess.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	elapsed := time.Since(start)
	sess.Shutdown(context.Background())

	if got := tool.callCount(); got != 3 {
		t.Errorf("tool call count = %d, want 3", got)
	}
	// 3 × 80ms serially → ~240ms. Allow slack for scheduling.
	if elapsed < 200*time.Millisecond {
		t.Errorf("serial execution took %v; want >= 200ms (serialized incorrectly)", elapsed)
	}
}

// TestRun_SessionStartFiresOnce verifies that SessionStart is emitted
// only on the first turn, not subsequent turns.
func TestRun_SessionStartFiresOnce(t *testing.T) {
	client := &scriptedClient{
		scripts: [][]llm.Delta{
			textThenFinal("one"),
			textThenFinal("two"),
			textThenFinal("three"),
		},
	}
	sess, rt := newSessionForTest(t, client, []tools.HeadlessTool{
		tools.NewReadTool(tools.OSReadOperations{}),
	})

	evtsPtr, wait := collectEvents(t, rt.EventBus)

	if err := sess.Run(context.Background(), "first"); err != nil {
		t.Fatalf("Run 1: %v", err)
	}
	if err := sess.Run(context.Background(), "second"); err != nil {
		t.Fatalf("Run 2: %v", err)
	}
	sess.Shutdown(context.Background())
	wait()
	evts := *evtsPtr

	count := 0
	for _, e := range evts {
		if e.Topic() == TopicSessionStart {
			count++
		}
	}
	if count != 1 {
		t.Errorf("SessionStart fired %d times, want 1", count)
	}
}

// TestRun_AfterShutdownReturnsErr verifies the runtime is one-shot:
// after Shutdown, Run returns ErrRuntimeShutdown.
func TestRun_AfterShutdownReturnsErr(t *testing.T) {
	client := &scriptedClient{
		scripts: [][]llm.Delta{textThenFinal("ok")},
	}
	sess, _ := newSessionForTest(t, client, []tools.HeadlessTool{
		tools.NewReadTool(tools.OSReadOperations{}),
	})
	if err := sess.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
	err := sess.Run(context.Background(), "anything")
	if !errors.Is(err, ErrRuntimeShutdown) {
		t.Errorf("Run after Shutdown: err = %v, want ErrRuntimeShutdown", err)
	}
}

// TestShutdown_Idempotent verifies Shutdown can be called multiple
// times without error.
func TestShutdown_Idempotent(t *testing.T) {
	client := &scriptedClient{
		scripts: [][]llm.Delta{textThenFinal("ok")},
	}
	sess, _ := newSessionForTest(t, client, []tools.HeadlessTool{
		tools.NewReadTool(tools.OSReadOperations{}),
	})
	if err := sess.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown 1: %v", err)
	}
	if err := sess.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown 2: %v", err)
	}
	if err := sess.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown 3: %v", err)
	}
}

// TestRun_EmptyUserInput verifies Run rejects empty input without
// touching the state tree.
func TestRun_EmptyUserInput(t *testing.T) {
	client := &scriptedClient{}
	sess, rt := newSessionForTest(t, client, []tools.HeadlessTool{
		tools.NewReadTool(tools.OSReadOperations{}),
	})
	if err := sess.Run(context.Background(), ""); err == nil {
		t.Fatal("Run with empty input should fail")
	}
	// Run with empty input must NOT have appended anything. The lazy
	// manager starts with no entries; querying Tree() would return
	// "lazy manager has no entries yet" which is not a meaningful
	// assertion. Instead, check LeafID: a fresh session has LeafID == ""
	// (no SessionHeader is materialized until the first Append).
	if leaf := rt.State.LeafID(); leaf != "" {
		t.Errorf("LeafID = %q after rejected empty-input Run; want empty", leaf)
	}
	sess.Shutdown(context.Background())
}

// TestRun_UnknownToolProducesErrorResult verifies that a model-emitted
// tool call for an unknown tool name does NOT crash the turn — instead
// the agent synthesizes an error ToolResult so the model can react.
func TestRun_UnknownToolProducesErrorResult(t *testing.T) {
	client := &scriptedClient{
		scripts: [][]llm.Delta{
			{
				llm.ToolCallDelta{ContentIndex: 0, ID: "tu_1", Name: "nonexistent", PartialInput: "{}"},
				llm.Final{StopReason: llm.StopReasonToolUse},
			},
			textThenFinal("recovered"),
		},
	}
	sess, rt := newSessionForTest(t, client, []tools.HeadlessTool{
		tools.NewReadTool(tools.OSReadOperations{}),
	})
	evtsPtr, wait := collectEvents(t, rt.EventBus)

	if err := sess.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	sess.Shutdown(context.Background())
	wait()
	evts := *evtsPtr

	// Find the ToolResult event and verify IsError=true.
	var found bool
	for _, e := range evts {
		if tre, ok := e.(ToolResultEvent); ok {
			if tre.Result.IsError && strings.Contains(textFromContent(tre.Result.Content), "not registered") {
				found = true
			}
		}
	}
	if !found {
		t.Error("expected an error ToolResult for the unknown tool, not found")
	}
}

// textFromContent extracts the text from a single-block TextContent.
func textFromContent(blocks []llm.ContentBlock) string {
	if len(blocks) == 0 {
		return ""
	}
	if tc, ok := blocks[0].(llm.TextContent); ok {
		return tc.Text
	}
	return ""
}

// TestCollectToolCalls_OrderPreserved verifies tool_use blocks are
// extracted in content order (not type-switch order).
func TestCollectToolCalls_OrderPreserved(t *testing.T) {
	m := llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
		llm.TextContent{Text: "first"},
		llm.ToolUse{ID: "a", Name: "x"},
		llm.TextContent{Text: "mid"},
		llm.ToolUse{ID: "b", Name: "y"},
	}}
	calls := collectToolCalls(m)
	if len(calls) != 2 {
		t.Fatalf("got %d calls, want 2", len(calls))
	}
	if calls[0].ID != "a" || calls[1].ID != "b" {
		t.Errorf("order = [%s, %s], want [a, b]", calls[0].ID, calls[1].ID)
	}
}

// TestRun_StateManagerIsClosedOnShutdown verifies that when the runtime
// owns the state manager, Shutdown closes it.
func TestRun_StateManagerIsClosedOnShutdown(t *testing.T) {
	client := &scriptedClient{
		scripts: [][]llm.Delta{textThenFinal("ok")},
	}
	sess, rt := newSessionForTest(t, client, []tools.HeadlessTool{
		tools.NewReadTool(tools.OSReadOperations{}),
	})
	if !rt.ownsState {
		t.Fatal("test setup: runtime should own state by default")
	}
	if err := sess.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
	// Subsequent Append should fail.
	_, err := rt.State.Append(state.Entry{
		Kind:    state.KindMessage,
		Payload: state.MessagePayload{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.TextContent{Text: "x"}}},
	})
	if err == nil {
		t.Error("Append after Close should fail")
	}
}

// TestRun_InjectedStateNotClosedOnShutdown verifies that when the caller
// injects a state manager, Shutdown leaves it open.
func TestRun_InjectedStateNotClosedOnShutdown(t *testing.T) {
	client := &scriptedClient{
		scripts: [][]llm.Delta{textThenFinal("ok")},
	}
	inj := state.NewInMemoryManager(t.TempDir())
	opts := SessionOptions{
		Model:        "test-model",
		Settings:     config.DefaultSettings(),
		LLMClient:    client,
		Tools:        []tools.HeadlessTool{tools.NewReadTool(tools.OSReadOperations{})},
		ConfigDir:    t.TempDir(),
		StateManager: inj,
	}
	rt, err := CreateAgentSessionRuntime(context.Background(), t.TempDir(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSessionRuntime: %v", err)
	}
	sess := NewAgentSession(rt)
	if err := sess.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
	// Injected manager should still be usable.
	_, err = inj.Append(state.Entry{
		Kind:    state.KindMessage,
		Payload: state.MessagePayload{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.TextContent{Text: "x"}}},
	})
	if err != nil {
		t.Errorf("Append on injected manager after Shutdown: %v (manager should still be open)", err)
	}
	inj.Close()
}
