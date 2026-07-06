// topics_test.go — verifies SubscribeTopics topic filtering.
//
// Three scenarios (per task 3.4):
//   (a) a subscriber to TopicToolCall + TopicToolResult receives EXACTLY
//       those events over a full turn and nothing else.
//   (b) Subscribe (no args) receives every event type.
//   (c) SubscribeTopics with a bogus topic receives zero events and the
//       turn still completes.

package tau

import (
	"context"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/taucentral/tau/internal/config"
	"github.com/taucentral/tau/internal/fauxprovider"
	"github.com/taucentral/tau/internal/llm"
	"github.com/taucentral/tau/internal/tools"
)

func topicOptions(t *testing.T) Options {
	t.Helper()
	return Options{
		Cwd:           t.TempDir(),
		Model:         "faux",
		LLMClient:     fauxprovider.NewWithResponse("hello"),
		Tools:         []HeadlessTool{tools.NewReadTool(tools.OSReadOperations{})},
		Settings:      config.DefaultSettings(),
		ContextWindow: 200000,
	}
}

// toolInvokingFaux is a faux provider that emits a tool call on the first
// stream, then a final text on the second. This lets us assert both
// tool_call and tool_result events fire.
type toolInvokingFaux struct {
	mu      sync.Mutex
	calls   int
	useText string
}

func (f *toolInvokingFaux) Stream(ctx context.Context, _ llm.Request) (<-chan llm.Delta, error) {
	f.mu.Lock()
	f.calls++
	idx := f.calls
	f.mu.Unlock()
	ch := make(chan llm.Delta, 4)
	go func() {
		defer close(ch)
		if idx == 1 {
			// First stream: emit a ToolUse then a Final with stop_reason tool_use.
			ch <- llm.ToolCallDelta{ID: "tu1", Name: "read", PartialInput: `"x"`}
			ch <- llm.Final{StopReason: llm.StopReasonToolUse}
			return
		}
		// Second stream: plain text + end_turn.
		ch <- llm.TextDelta{Text: f.useText}
		ch <- llm.Final{StopReason: llm.StopReasonEndTurn}
	}()
	return ch, nil
}

func drainEvents(t *testing.T, ch <-chan SessionEvent, timeout time.Duration) []string {
	t.Helper()
	var got []string
	deadline := time.After(timeout)
	for {
		select {
		case evt, ok := <-ch:
			if !ok {
				return got
			}
			got = append(got, string(evt.Topic()))
		case <-deadline:
			return got
		}
	}
}

func TestSubscribeTopicsReceivesOnlyMatching(t *testing.T) {
	opts := topicOptions(t)
	opts.LLMClient = &toolInvokingFaux{useText: "after tool"}

	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	// Subscribe BEFORE Run so we don't miss the tool_call.
	filtered := sess.SubscribeTopics(TopicToolCall, TopicToolResult)
	all := sess.Subscribe()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := sess.Run(ctx, "use the read tool"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	gotFiltered := drainEvents(t, filtered, 2*time.Second)
	gotAll := drainEvents(t, all, 2*time.Second)

	// Filtered: only tool_call + tool_result (in some order — the tool
	// result fires after the call, but concurrent tool execution could
	// reorder in principle).
	for _, topic := range gotFiltered {
		if topic != string(TopicToolCall) && topic != string(TopicToolResult) {
			t.Errorf("filtered subscriber got topic %q, want only tool_call/tool_result", topic)
		}
	}
	if len(gotFiltered) == 0 {
		t.Error("filtered subscriber received zero events; want at least one tool_call + tool_result")
	}
	if !contains(gotFiltered, string(TopicToolCall)) {
		t.Error("filtered subscriber missed tool_call")
	}
	if !contains(gotFiltered, string(TopicToolResult)) {
		t.Error("filtered subscriber missed tool_result")
	}

	// Sanity: the all-topics subscriber got at least the lifecycle.
	if !contains(gotAll, string(TopicTurnEnd)) {
		t.Error("all-topics subscriber missed turn_end")
	}
}

func TestSubscribeReceivesEveryTopic(t *testing.T) {
	opts := topicOptions(t)
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	all := sess.Subscribe()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := sess.Run(ctx, "hello"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	got := drainEvents(t, all, 2*time.Second)
	unique := dedupe(got)
	// Expect at least: session_start, turn_start, message_start, message_update,
	// message_end, turn_end. session_shutdown only on Shutdown, which we
	// haven't called yet.
	want := []string{
		string(TopicSessionStart),
		string(TopicTurnStart),
		string(TopicMessageStart),
		string(TopicMessageUpdate),
		string(TopicMessageEnd),
		string(TopicTurnEnd),
	}
	for _, w := range want {
		if !contains(unique, w) {
			t.Errorf("all-topics subscriber missed expected topic %q; got %v", w, unique)
		}
	}
}

func TestSubscribeTopicsBogusTopicIsNoOp(t *testing.T) {
	opts := topicOptions(t)
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	bogus := sess.SubscribeTopics(Topic("bogus-not-real"))

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := sess.Run(ctx, "hello"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	got := drainEvents(t, bogus, 500*time.Millisecond)
	if len(got) != 0 {
		t.Errorf("bogus-topic subscriber got %d events, want 0: %v", len(got), got)
	}
}

func contains(haystack []string, needle string) bool {
	for _, h := range haystack {
		if h == needle {
			return true
		}
	}
	return false
}

func dedupe(in []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(in))
	for _, v := range in {
		if !seen[v] {
			seen[v] = true
			out = append(out, v)
		}
	}
	sort.Strings(out)
	return out
}
