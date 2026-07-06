// lazy_integration_test.go — end-to-end smoke test for lazy tool
// registration through the full agent session stack.
//
// Task 8.4: register 20 stub tools (10 eager, 10 lazy with distinct
// intents); run a turn whose user message contains one lazy tool's
// intent keyword; confirm only that lazy tool plus all eager tools
// render in Request.Tools; confirm hidden lazy tools hydrate on
// first-call miss.

package agent

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/taucentral/tau/internal/llm"
	"github.com/taucentral/tau/internal/tools"
	"github.com/invopop/jsonschema"
)

// ---- Stubs ----

// agentEagerStub is an eager HeadlessTool for end-to-end tests.
type agentEagerStub struct {
	name string
}

func (e *agentEagerStub) Name() string                                           { return e.name }
func (e *agentEagerStub) Description() string                                    { return "eager stub: " + e.name }
func (e *agentEagerStub) Parameters() jsonschema.Schema                          { return jsonschema.Schema{Type: "object"} }
func (e *agentEagerStub) Execute(ctx context.Context, call tools.ToolCall) (tools.ToolResult, error) {
	return tools.NewTextResult("ok:" + e.name), nil
}

// agentLazyStub is a lazy HeadlessTool + LazyHeadlessTool for
// end-to-end tests. The hydrates counter tracks Hydrate calls.
type agentLazyStub struct {
	name     string
	intent   string
	hydrates int32 // atomic
}

func (l *agentLazyStub) Name() string                                           { return l.name }
func (l *agentLazyStub) Description() string                                    { return "lazy stub: " + l.name }
func (l *agentLazyStub) Parameters() jsonschema.Schema                          { return jsonschema.Schema{Type: "object"} }
func (l *agentLazyStub) Execute(ctx context.Context, call tools.ToolCall) (tools.ToolResult, error) {
	return tools.NewTextResult("ok:" + l.name), nil
}
func (l *agentLazyStub) Tag() tools.ToolTag {
	return tools.ToolTag{Intent: l.intent}
}
func (l *agentLazyStub) Hydrate(ctx context.Context) (jsonschema.Schema, error) {
	atomic.AddInt32(&l.hydrates, 1)
	return jsonschema.Schema{Type: "object"}, nil
}

func (l *agentLazyStub) hydrateCount() int { return int(atomic.LoadInt32(&l.hydrates)) }

// recordingScriptedClient is a scriptedClient variant that records
// every llm.Request it receives. Used to assert on Request.Tools.
type recordingScriptedClient struct {
	mu       sync.Mutex
	scripts  [][]llm.Delta
	idx      int
	requests []llm.Request
}

func (c *recordingScriptedClient) Stream(ctx context.Context, req llm.Request) (<-chan llm.Delta, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.requests = append(c.requests, req)
	if c.idx >= len(c.scripts) {
		return nil, errors.New("recordingScriptedClient: no more scripts")
	}
	script := c.scripts[c.idx]
	c.idx++
	ch := make(chan llm.Delta, len(script))
	for _, d := range script {
		ch <- d
	}
	close(ch)
	return ch, nil
}

func (c *recordingScriptedClient) recordedRequests() []llm.Request {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]llm.Request, len(c.requests))
	copy(out, c.requests)
	return out
}

// registerMixedTools registers nLazy lazy stubs with distinct intents
// and nEager-1 additional eager stubs (eager-00 is passed as the seed
// to newSessionForTest to satisfy the non-empty Tools requirement).
// Returns the lazy stubs so tests can assert on hydrate counts.
// The intent for lazy stub i is "intent-<i>".
func registerMixedTools(t *testing.T, sess *AgentSession, nEager, nLazy int) []*agentLazyStub {
	t.Helper()
	lazyStubs := make([]*agentLazyStub, nLazy)
	for i := 0; i < nLazy; i++ {
		lazyStubs[i] = &agentLazyStub{
			name:   fmt.Sprintf("lazy-%02d", i),
			intent: fmt.Sprintf("intent-%02d", i),
		}
		if err := sess.rt.Registry.Register(lazyStubs[i]); err != nil {
			t.Fatalf("register lazy-%02d: %v", i, err)
		}
	}
	// eager-00 is the seed; register eager-01 through eager-(nEager-1).
	for i := 1; i < nEager; i++ {
		if err := sess.rt.Registry.Register(&agentEagerStub{
			name: fmt.Sprintf("eager-%02d", i),
		}); err != nil {
			t.Fatalf("register eager-%02d: %v", i, err)
		}
	}
	return lazyStubs
}

// ---- Tests ----

// TestLazyEndToEnd_HeuristicFiltering verifies that the agent's
// buildRequest filters lazy tools per the heuristic. With 10 eager +
// 10 lazy tools and a user message matching one lazy tool's intent,
// Request.Tools should contain exactly 11 entries (10 eager + 1 lazy).
func TestLazyEndToEnd_HeuristicFiltering(t *testing.T) {
	client := &recordingScriptedClient{
		scripts: [][]llm.Delta{
			textThenFinal("done"),
		},
	}
	sess, _ := newSessionForTest(t, client, []tools.HeadlessTool{&agentEagerStub{name: "eager-00"}})
	defer sess.Shutdown(context.Background())

	lazyStubs := registerMixedTools(t, sess, 10, 10)

	// User message contains intent-03's keyword.
	if err := sess.Run(context.Background(), "please run intent-03 for me"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	reqs := client.recordedRequests()
	if len(reqs) == 0 {
		t.Fatal("no requests recorded")
	}
	toolsSent := reqs[0].Tools

	// Expect 10 eager + 1 matching lazy = 11.
	if len(toolsSent) != 11 {
		t.Fatalf("Request.Tools len = %d, want 11", len(toolsSent))
	}

	// Build a set of tool names sent.
	sent := make(map[string]bool, len(toolsSent))
	for _, ts := range toolsSent {
		sent[ts.Name] = true
	}

	// All 10 eager tools should be present.
	for i := 0; i < 10; i++ {
		name := fmt.Sprintf("eager-%02d", i)
		if !sent[name] {
			t.Errorf("eager tool %q missing from Request.Tools", name)
		}
	}

	// Only lazy-03 should be present.
	if !sent["lazy-03"] {
		t.Errorf("lazy-03 (matched intent) missing from Request.Tools")
	}
	for i := 0; i < 10; i++ {
		if i == 3 {
			continue
		}
		name := fmt.Sprintf("lazy-%02d", i)
		if sent[name] {
			t.Errorf("lazy tool %q should be hidden but was sent", name)
		}
	}

	// The matched lazy tool should have been hydrated (Schemas called Hydrate).
	if lazyStubs[3].hydrateCount() == 0 {
		t.Errorf("lazy-03 was not hydrated (hydrateCount=0)")
	}

	// Hidden lazy tools should NOT have been hydrated by Schemas.
	for i := 0; i < 10; i++ {
		if i == 3 {
			continue
		}
		if lazyStubs[i].hydrateCount() != 0 {
			t.Errorf("lazy-%02d was hydrated but should be hidden (hydrateCount=%d)", i, lazyStubs[i].hydrateCount())
		}
	}
}

// TestLazyEndToEnd_FirstCallMiss verifies the first-call miss
// fallback: when the model calls a hidden lazy tool, the runtime
// hydrates it, executes it, and emits HydrationMissEvent.
func TestLazyEndToEnd_FirstCallMiss(t *testing.T) {
	// Script 1: model calls lazy-05 (which is hidden — no intent match).
	// Script 2: text + final (end the turn).
	client := &recordingScriptedClient{
		scripts: [][]llm.Delta{
			{
				llm.ToolCallDelta{ContentIndex: 0, ID: "tu_miss_1", Name: "lazy-05", PartialInput: "{}"},
				llm.Final{StopReason: llm.StopReasonToolUse},
			},
			textThenFinal("done"),
		},
	}
	sess, rt := newSessionForTest(t, client, []tools.HeadlessTool{&agentEagerStub{name: "eager-00"}})

	lazyStubs := registerMixedTools(t, sess, 2, 10)

	// Collect events to assert on HydrationMissEvent.
	evtsPtr, wait := collectEvents(t, rt.EventBus)

	// User message does NOT match any lazy intent.
	if err := sess.Run(context.Background(), "unrelated message"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	sess.Shutdown(context.Background())
	wait()
	evts := *evtsPtr

	// Assert HydrationMissEvent was published for lazy-05.
	var missSeen bool
	for _, e := range evts {
		if hm, ok := e.(HydrationMissEvent); ok {
			if hm.ToolName == "lazy-05" {
				missSeen = true
			}
		}
	}
	if !missSeen {
		t.Errorf("HydrationMissEvent for lazy-05 not published; events: %d", len(evts))
	}

	// lazy-05 was hydrated via the miss path.
	if lazyStubs[5].hydrateCount() == 0 {
		t.Errorf("lazy-05 was not hydrated on miss (hydrateCount=0)")
	}

	// lazy-05 was still executed (ToolResultEvent with its result).
	var resultSeen bool
	for _, e := range evts {
		if tre, ok := e.(ToolResultEvent); ok {
			if tre.ToolID == "tu_miss_1" {
				resultSeen = true
				if tre.Result.IsError {
					t.Errorf("lazy-05 result is error: %+v", tre.Result)
				}
			}
		}
	}
	if !resultSeen {
		t.Errorf("ToolResultEvent for lazy-05 not seen")
	}

	// Other hidden lazy tools were NOT hydrated.
	for i := 0; i < 10; i++ {
		if i == 5 {
			continue
		}
		if lazyStubs[i].hydrateCount() != 0 {
			t.Errorf("lazy-%02d was hydrated but should not be (hydrateCount=%d)", i, lazyStubs[i].hydrateCount())
		}
	}
}

// TestLazyEndToEnd_RecordToolUseAfterExecute verifies that
// RecordToolUse is called after successful tool execution, so the
// registry's recency map reflects the call.
func TestLazyEndToEnd_RecordToolUseAfterExecute(t *testing.T) {
	// Script 1: call an eager tool, stop with tool_use.
	// Script 2: text + final.
	client := &recordingScriptedClient{
		scripts: [][]llm.Delta{
			{
				llm.ToolCallDelta{ContentIndex: 0, ID: "tu_1", Name: "eager-tool", PartialInput: "{}"},
				llm.Final{StopReason: llm.StopReasonToolUse},
			},
			textThenFinal("done"),
		},
	}
	sess, rt := newSessionForTest(t, client, []tools.HeadlessTool{&agentEagerStub{name: "eager-tool"}})
	defer sess.Shutdown(context.Background())

	if err := sess.Run(context.Background(), "call the tool"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	// The recency map should now contain eager-tool.
	m := rt.Registry.RecentUseMap()
	if _, ok := m["eager-tool"]; !ok {
		t.Errorf("eager-tool not in RecentUseMap after execution: %v", m)
	}
}
