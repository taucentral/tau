// short_circuit_test.go — runtime-level tests for the
// RequestShortCircuiter middleware seam.
//
// Coverage:
//
//   - Task 4.1 (nine cases): hit, miss, error, composition
//     (first-wins, second-wins-on-first-miss), mutator-abort-blocks-sc,
//     mutator-mutates-then-hit, cached-tool-calls, no-sc fast path.
//   - Task 4.2: assistant.Source is "cache" on hit, "llm" on Stream.
//   - Task 4.3: cached tool-call response fires ToolInterceptor +
//     Execute + AfterToolCall.
//   - Task 2.3: benchmark comparing per-turn latency with and without
//     the slice populated.
//
// Stubs use the runtime-level interfaces (runtimeRequestShortCircuiter,
// runtimeRequestMutator, runtimeResponseObserver, runtimeToolInterceptor)
// directly so the runtime invariants are pinned independently of the
// SDK adapter layer in pkg/tau.

package agent

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/taucentral/tau/internal/config"
	"github.com/taucentral/tau/internal/llm"
	"github.com/taucentral/tau/internal/tools"
)

// rtShortCircuiter is a runtime-level RequestShortCircuiter stub.
// hitMsg is returned on hit; if nil, the stub misses. err is returned
// if non-nil. The calls counter tracks invocations.
type rtShortCircuiter struct {
	calls  atomic.Int32
	hitMsg *llm.Message
	err    error
	// requestSeen captures the last *llm.Request the stub observed,
	// letting tests assert post-mutator state is visible.
	requestSeen atomic.Pointer[llm.Request]
}

func (s *rtShortCircuiter) ShortCircuit(ctx context.Context, req *llm.Request) (*llm.Message, error) {
	s.calls.Add(1)
	// Snapshot req so the test can read it after Run returns. The
	// pointer is read-only from our side; copy the value to avoid
	// aliasing later mutation.
	cp := *req
	s.requestSeen.Store(&cp)
	if s.err != nil {
		return nil, s.err
	}
	return s.hitMsg, nil
}

// rtMutatorWithCapture is a runtime-level RequestMutator that
// optionally mutates the request (recording that it ran) and can
// return a configured error to test the abort path.
type rtMutatorWithCapture struct {
	calls atomic.Int32
	err   error
	mut   func(req *llm.Request) // optional; applied when err is nil
}

func (m *rtMutatorWithCapture) MutateRequest(ctx context.Context, req *llm.Request) error {
	m.calls.Add(1)
	if m.err != nil {
		return m.err
	}
	if m.mut != nil {
		m.mut(req)
	}
	return nil
}

// cachedTextMsg builds a stub assistant message with a single
// TextContent block and the supplied StopReason.
func cachedTextMsg(text string, stop llm.StopReason) *llm.Message {
	return &llm.Message{
		Role:       llm.RoleAssistant,
		Content:    []llm.ContentBlock{llm.TextContent{Text: text}},
		StopReason: stop,
		Timestamp:  time.Now(),
	}
}

// cachedToolUseMsg builds a stub assistant message carrying one
// ToolUse block. Used to test that cached tool-call responses route
// through executeTools + ToolInterceptor.
func cachedToolUseMsg(toolName, toolUseID string) *llm.Message {
	return &llm.Message{
		Role: llm.RoleAssistant,
		Content: []llm.ContentBlock{
			llm.ToolUse{ID: toolUseID, Name: toolName, Input: []byte("{}")},
		},
		StopReason: llm.StopReasonToolUse,
		Timestamp:  time.Now(),
	}
}

// newShortCircuitSession constructs a session with the given client,
// builtin tools, and middleware set. Returns the session + runtime.
func newShortCircuitSession(t *testing.T, client llm.LLMClient, builtinTools []tools.HeadlessTool, mw MiddlewareSet) (*AgentSession, *AgentSessionRuntime) {
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
	rt.Options.Middleware = mw
	return NewAgentSession(rt), rt
}

// ---- Task 4.1: nine trigger cases ----

// Case 1: single short-circuiter, cache hit. Stream never called;
// MessageStart/MessageEnd fire; ResponseObserver sees the cached
// message.
func TestShortCircuit_HitSkipsStream(t *testing.T) {
	client := &scriptedClient{scripts: [][]llm.Delta{textThenFinal("should-not-reach")}}
	obs := &rtObserver{}
	sc := &rtShortCircuiter{hitMsg: cachedTextMsg("cached!", llm.StopReasonEndTurn)}
	sess, _ := newShortCircuitSession(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}}, MiddlewareSet{
		RequestShortCircuiters: []runtimeRequestShortCircuiter{sc},
		ResponseObservers:      []runtimeResponseObserver{obs},
	})
	defer sess.Shutdown(context.Background())

	if err := sess.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if sc.calls.Load() != 1 {
		t.Errorf("short-circuiter calls = %d, want 1", sc.calls.Load())
	}
	// Stream should not have been called: the script is untouched.
	if client.idx != 0 {
		t.Errorf("Stream called (idx = %d, want 0)", client.idx)
	}
	if obs.calls.Load() != 1 {
		t.Errorf("observer calls = %d, want 1", obs.calls.Load())
	}
	if len(obs.resps) != 1 {
		t.Fatalf("observer responses = %d, want 1", len(obs.resps))
	}
	if obs.resps[0].Source != "cache" {
		t.Errorf("observed Source = %q, want \"cache\"", obs.resps[0].Source)
	}
	if obs.resps[0].StopReason != llm.StopReasonEndTurn {
		t.Errorf("observed StopReason = %q, want %q", obs.resps[0].StopReason, llm.StopReasonEndTurn)
	}
}

// Case 2: single short-circuiter, miss ((nil, nil)). Stream called
// as usual.
func TestShortCircuit_MissFallsThroughToStream(t *testing.T) {
	client := &scriptedClient{scripts: [][]llm.Delta{textThenFinal("from-llm")}}
	obs := &rtObserver{}
	sc := &rtShortCircuiter{hitMsg: nil}
	sess, _ := newShortCircuitSession(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}}, MiddlewareSet{
		RequestShortCircuiters: []runtimeRequestShortCircuiter{sc},
		ResponseObservers:      []runtimeResponseObserver{obs},
	})
	defer sess.Shutdown(context.Background())

	if err := sess.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if sc.calls.Load() != 1 {
		t.Errorf("short-circuiter calls = %d, want 1", sc.calls.Load())
	}
	if client.idx != 1 {
		t.Errorf("Stream not called (idx = %d, want 1)", client.idx)
	}
	if obs.calls.Load() != 1 {
		t.Errorf("observer calls = %d, want 1", obs.calls.Load())
	}
	if obs.resps[0].Source != "llm" {
		t.Errorf("observed Source = %q, want \"llm\"", obs.resps[0].Source)
	}
}

// Case 3: single short-circuiter, error ((nil, err)). Turn aborts
// with the wrapped error.
func TestShortCircuit_ErrorAbortsTurn(t *testing.T) {
	client := &scriptedClient{scripts: [][]llm.Delta{textThenFinal("should-not-reach")}}
	sentinel := errors.New("cache explosion")
	sc := &rtShortCircuiter{err: sentinel}
	obs := &rtObserver{}
	sess, _ := newShortCircuitSession(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}}, MiddlewareSet{
		RequestShortCircuiters: []runtimeRequestShortCircuiter{sc},
		ResponseObservers:      []runtimeResponseObserver{obs},
	})
	defer sess.Shutdown(context.Background())

	err := sess.Run(context.Background(), "go")
	if err == nil {
		t.Fatalf("Run returned nil error, want non-nil")
	}
	if !errors.Is(err, sentinel) {
		t.Errorf("err does not wrap sentinel: %v", err)
	}
	if !contains(err.Error(), "agent: short-circuit:") {
		t.Errorf("err message = %q, want prefix \"agent: short-circuit:\"", err.Error())
	}
	if client.idx != 0 {
		t.Errorf("Stream called on short-circuit error (idx = %d, want 0)", client.idx)
	}
	// Task 6.4: ResponseObserver SHALL fire exactly once with the
	// empty message on the short-circuit error path, matching the
	// Stream-error path's behavior (session.go publishes &llm.Message{}
	// so audit/telemetry middleware see the failure).
	if obs.calls.Load() != 1 {
		t.Errorf("observer calls = %d, want 1 (observer SHALL fire on error path)", obs.calls.Load())
	}
	if got := obs.resps[0]; got == nil {
		t.Fatal("observer response is nil, want non-nil *llm.Message")
	} else if len(got.Content) != 0 {
		t.Errorf("observer response Content len = %d, want 0 (empty message)", len(got.Content))
	} else if got.Source != "" {
		t.Errorf("observer response Source = %q, want \"\" (zero-value Message, not \"llm\" or \"cache\")", got.Source)
	}
}

// contains reports whether s contains substr. Avoids pulling strings
// just for one test.
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || indexOf(s, substr) >= 0)
}

func indexOf(s, substr string) int {
	for i := 0; i+len(substr) <= len(s); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}

// Case 4: two short-circuiters, first hits. Second never runs.
func TestShortCircuit_FirstHitWins(t *testing.T) {
	client := &scriptedClient{scripts: [][]llm.Delta{textThenFinal("should-not-reach")}}
	first := &rtShortCircuiter{hitMsg: cachedTextMsg("first", llm.StopReasonEndTurn)}
	second := &rtShortCircuiter{hitMsg: cachedTextMsg("second", llm.StopReasonEndTurn)}
	sess, _ := newShortCircuitSession(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}}, MiddlewareSet{
		RequestShortCircuiters: []runtimeRequestShortCircuiter{first, second},
	})
	defer sess.Shutdown(context.Background())

	if err := sess.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if first.calls.Load() != 1 {
		t.Errorf("first short-circuiter calls = %d, want 1", first.calls.Load())
	}
	if second.calls.Load() != 0 {
		t.Errorf("second short-circuiter calls = %d, want 0 (first should win)", second.calls.Load())
	}
}

// Case 5: two short-circuiters, first misses, second hits. Stream
// never called.
func TestShortCircuit_SecondHitAfterFirstMiss(t *testing.T) {
	client := &scriptedClient{scripts: [][]llm.Delta{textThenFinal("should-not-reach")}}
	first := &rtShortCircuiter{hitMsg: nil}
	second := &rtShortCircuiter{hitMsg: cachedTextMsg("second", llm.StopReasonEndTurn)}
	sess, _ := newShortCircuitSession(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}}, MiddlewareSet{
		RequestShortCircuiters: []runtimeRequestShortCircuiter{first, second},
	})
	defer sess.Shutdown(context.Background())

	if err := sess.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if first.calls.Load() != 1 {
		t.Errorf("first calls = %d, want 1", first.calls.Load())
	}
	if second.calls.Load() != 1 {
		t.Errorf("second calls = %d, want 1", second.calls.Load())
	}
	if client.idx != 0 {
		t.Errorf("Stream called (idx = %d, want 0)", client.idx)
	}
}

// Case 6: short-circuiter + mutator, mutator aborts. Short-circuiter
// never runs.
func TestShortCircuit_MutatorAbortBlocksShortCircuiter(t *testing.T) {
	client := &scriptedClient{scripts: [][]llm.Delta{textThenFinal("should-not-reach")}}
	mutSentinel := errors.New("mutator blocked")
	mut := &rtMutatorWithCapture{err: mutSentinel}
	sc := &rtShortCircuiter{hitMsg: cachedTextMsg("should-not-reach", llm.StopReasonEndTurn)}
	sess, _ := newShortCircuitSession(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}}, MiddlewareSet{
		RequestMutators:        []runtimeRequestMutator{mut},
		RequestShortCircuiters: []runtimeRequestShortCircuiter{sc},
	})
	defer sess.Shutdown(context.Background())

	err := sess.Run(context.Background(), "go")
	if err == nil || !errors.Is(err, mutSentinel) {
		t.Fatalf("err = %v, want wrap of %v", err, mutSentinel)
	}
	if mut.calls.Load() != 1 {
		t.Errorf("mutator calls = %d, want 1", mut.calls.Load())
	}
	if sc.calls.Load() != 0 {
		t.Errorf("short-circuiter ran %d times, want 0 (mutator aborts first)", sc.calls.Load())
	}
}

// Case 7: short-circuiter + mutator, mutator mutates, hit. The
// short-circuiter observes the post-mutator request.
func TestShortCircuit_ShortCircuiterSeesPostMutatorRequest(t *testing.T) {
	client := &scriptedClient{scripts: [][]llm.Delta{textThenFinal("should-not-reach")}}
	// Mutator tags the request with a custom system field the cache
	// key can latch onto.
	mut := &rtMutatorWithCapture{
		mut: func(req *llm.Request) {
			if req.Headers == nil {
				req.Headers = map[string]string{}
			}
			req.Headers["x-cache-key"] = "mutator-applied"
		},
	}
	sc := &rtShortCircuiter{hitMsg: cachedTextMsg("cached", llm.StopReasonEndTurn)}
	sess, _ := newShortCircuitSession(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}}, MiddlewareSet{
		RequestMutators:        []runtimeRequestMutator{mut},
		RequestShortCircuiters: []runtimeRequestShortCircuiter{sc},
	})
	defer sess.Shutdown(context.Background())

	if err := sess.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if sc.calls.Load() != 1 {
		t.Fatalf("short-circuiter calls = %d, want 1", sc.calls.Load())
	}
	got := sc.requestSeen.Load()
	if got == nil {
		t.Fatal("requestSeen is nil")
	}
	if got.Headers["x-cache-key"] != "mutator-applied" {
		t.Errorf("short-circuiter saw Headers[x-cache-key] = %q, want \"mutator-applied\"", got.Headers["x-cache-key"])
	}
}

// Case 8: short-circuiter returns tool calls. executeTools runs.
func TestShortCircuit_CachedToolCallsExecute(t *testing.T) {
	tool := &recordingTool{name: "rt-counter"}
	client := &scriptedClient{scripts: [][]llm.Delta{textThenFinal("should-not-reach")}}
	// The cache stub returns a tool-use response first, then a final
	// text response once the tool result is fed back. Without the
	// second phase the turn would loop forever re-emitting the tool
	// call.
	stateful := &statefulShortCircuiter{
		first: cachedToolUseMsg("rt-counter", "tu_cached_1"),
		then:  cachedTextMsg("done after tool", llm.StopReasonEndTurn),
	}
	sess, _ := newShortCircuitSession(t, client, []tools.HeadlessTool{tool}, MiddlewareSet{
		RequestShortCircuiters: []runtimeRequestShortCircuiter{stateful},
	})
	defer sess.Shutdown(context.Background())

	if err := sess.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if tool.calls.Load() != 1 {
		t.Errorf("tool Execute calls = %d, want 1", tool.calls.Load())
	}
	if stateful.firstCalls.Load() != 1 {
		t.Errorf("first-phase cache calls = %d, want 1", stateful.firstCalls.Load())
	}
	if stateful.thenCalls.Load() != 1 {
		t.Errorf("second-phase cache calls = %d, want 1", stateful.thenCalls.Load())
	}
}

// statefulShortCircuiter returns first on the first call, then on
// subsequent calls. Used to model a realistic cache that returns a
// tool-use response first, then a final text response once the tool
// result is fed back.
type statefulShortCircuiter struct {
	first      *llm.Message
	then       *llm.Message
	firstCalls atomic.Int32
	thenCalls  atomic.Int32
}

func (s *statefulShortCircuiter) ShortCircuit(ctx context.Context, req *llm.Request) (*llm.Message, error) {
	if s.firstCalls.Load() == 0 {
		s.firstCalls.Add(1)
		return s.first, nil
	}
	s.thenCalls.Add(1)
	return s.then, nil
}

// Case 9: no short-circuiters registered. Behavior matches today
// exactly (fast path).
func TestShortCircuit_NoShortCircuitersFastPath(t *testing.T) {
	client := &scriptedClient{scripts: [][]llm.Delta{textThenFinal("from-llm")}}
	obs := &rtObserver{}
	sess, _ := newShortCircuitSession(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}}, MiddlewareSet{
		ResponseObservers: []runtimeResponseObserver{obs},
	})
	defer sess.Shutdown(context.Background())

	if err := sess.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if client.idx != 1 {
		t.Errorf("Stream not called (idx = %d, want 1)", client.idx)
	}
	if obs.calls.Load() != 1 {
		t.Errorf("observer calls = %d, want 1", obs.calls.Load())
	}
	if obs.resps[0].Source != "llm" {
		t.Errorf("Source = %q, want \"llm\"", obs.resps[0].Source)
	}
}

// ---- Task 4.2: Source value pinning ----

// TestShortCircuit_SourceFieldOnHitAndMiss pins the Source value
// on both the cache-hit path ("cache") and the Stream path ("llm").
func TestShortCircuit_SourceFieldOnHitAndMiss(t *testing.T) {
	t.Run("hit_marks_cache", func(t *testing.T) {
		client := &scriptedClient{scripts: [][]llm.Delta{textThenFinal("should-not-reach")}}
		obs := &rtObserver{}
		sc := &rtShortCircuiter{hitMsg: cachedTextMsg("cached", llm.StopReasonEndTurn)}
		sess, _ := newShortCircuitSession(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}}, MiddlewareSet{
			RequestShortCircuiters: []runtimeRequestShortCircuiter{sc},
			ResponseObservers:      []runtimeResponseObserver{obs},
		})
		defer sess.Shutdown(context.Background())
		if err := sess.Run(context.Background(), "go"); err != nil {
			t.Fatalf("Run: %v", err)
		}
		if obs.resps[0].Source != "cache" {
			t.Errorf("Source = %q, want \"cache\"", obs.resps[0].Source)
		}
	})
	t.Run("stream_marks_llm", func(t *testing.T) {
		client := &scriptedClient{scripts: [][]llm.Delta{textThenFinal("from-llm")}}
		obs := &rtObserver{}
		sc := &rtShortCircuiter{hitMsg: nil} // miss
		sess, _ := newShortCircuitSession(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}}, MiddlewareSet{
			RequestShortCircuiters: []runtimeRequestShortCircuiter{sc},
			ResponseObservers:      []runtimeResponseObserver{obs},
		})
		defer sess.Shutdown(context.Background())
		if err := sess.Run(context.Background(), "go"); err != nil {
			t.Fatalf("Run: %v", err)
		}
		if obs.resps[0].Source != "llm" {
			t.Errorf("Source = %q, want \"llm\"", obs.resps[0].Source)
		}
	})
}

// ---- Task 4.3: cached tool-call interceptor end-to-end ----

// TestShortCircuit_CachedToolCallFiresInterceptors asserts that when
// a short-circuiter returns a tool-call response, the runtime
// dispatches ToolInterceptor.BeforeToolCall, the tool's Execute, and
// ToolInterceptor.AfterToolCall exactly as it would for a fresh LLM
// tool-call response.
func TestShortCircuit_CachedToolCallFiresInterceptors(t *testing.T) {
	tool := &recordingTool{name: "rt-counter"}
	interceptor := &rtInterceptor{}
	client := &scriptedClient{scripts: [][]llm.Delta{textThenFinal("should-not-reach")}}
	stateful := &statefulShortCircuiter{
		first: cachedToolUseMsg("rt-counter", "tu_cached_1"),
		then:  cachedTextMsg("done after tool", llm.StopReasonEndTurn),
	}
	sess, _ := newShortCircuitSession(t, client, []tools.HeadlessTool{tool}, MiddlewareSet{
		RequestShortCircuiters: []runtimeRequestShortCircuiter{stateful},
		ToolInterceptors:       []runtimeToolInterceptor{interceptor},
	})
	defer sess.Shutdown(context.Background())

	if err := sess.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if tool.calls.Load() != 1 {
		t.Errorf("tool Execute calls = %d, want 1", tool.calls.Load())
	}
	if interceptor.beforeCalls.Load() != 1 {
		t.Errorf("BeforeToolCall calls = %d, want 1", interceptor.beforeCalls.Load())
	}
	if interceptor.afterCalls.Load() != 1 {
		t.Errorf("AfterToolCall calls = %d, want 1", interceptor.afterCalls.Load())
	}
}

// ---- Task 2.3: benchmark ----

// BenchmarkShortCircuit_NoSlice measures the fast path: when the
// RequestShortCircuiters slice is empty, the new short-circuit block
// costs one len() comparison per turn. Compare against
// BenchmarkShortCircuit_WithSlice to confirm the overhead target
// (< 50ns delta on commodity hardware).
func BenchmarkShortCircuit_NoSlice(b *testing.B) {
	client := &scriptedClient{scripts: nil}
	opts := SessionOptions{
		Model:     "test-model",
		Settings:  config.DefaultSettings(),
		LLMClient: client,
		Tools:     []tools.HeadlessTool{&recordingTool{name: "noop"}},
		ConfigDir: b.TempDir(),
	}
	rt, err := CreateAgentSessionRuntime(context.Background(), b.TempDir(), opts)
	if err != nil {
		b.Fatalf("CreateAgentSessionRuntime: %v", err)
	}
	// Replenish scripts each iteration: one text+final per turn.
	noopMsg := textThenFinal("ok")
	sess := NewAgentSession(rt)
	defer sess.Shutdown(context.Background())

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		// Refill the script for this iteration. The benchmark measures
		// per-turn overhead; the underlying turn does real work
		// (request build, mutator fast path, stream consume, observer
		// fast path, state append). The DELTA we care about is the
		// difference between this benchmark and the WithSlice variant
		// below — both pay the same baseline cost, only the slice
		// check differs.
		client.scripts = [][]llm.Delta{noopMsg}
		client.idx = 0
		if err := sess.Run(context.Background(), "go"); err != nil {
			b.Fatalf("Run[%d]: %v", i, err)
		}
	}
}

// BenchmarkShortCircuit_WithSlice measures the cost when one
// short-circuiter is registered but always misses ((nil, nil)). The
// per-turn overhead vs the NoSlice benchmark is the cost of one
// interface dispatch on top of the len() check.
func BenchmarkShortCircuit_WithSlice(b *testing.B) {
	client := &scriptedClient{scripts: nil}
	opts := SessionOptions{
		Model:     "test-model",
		Settings:  config.DefaultSettings(),
		LLMClient: client,
		Tools:     []tools.HeadlessTool{&recordingTool{name: "noop"}},
		ConfigDir: b.TempDir(),
	}
	rt, err := CreateAgentSessionRuntime(context.Background(), b.TempDir(), opts)
	if err != nil {
		b.Fatalf("CreateAgentSessionRuntime: %v", err)
	}
	rt.Options.Middleware = MiddlewareSet{
		RequestShortCircuiters: []runtimeRequestShortCircuiter{
			&rtShortCircuiter{hitMsg: nil}, // always misses
		},
	}
	noopMsg := textThenFinal("ok")
	sess := NewAgentSession(rt)
	defer sess.Shutdown(context.Background())

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		client.scripts = [][]llm.Delta{noopMsg}
		client.idx = 0
		if err := sess.Run(context.Background(), "go"); err != nil {
			b.Fatalf("Run[%d]: %v", i, err)
		}
	}
}

// BenchmarkShortCircuit_Helper_NoSlice isolates the cost of the
// shortCircuit helper itself when the slice is empty: the runtime
// checks `len(...) > 0` once and skips the loop entirely. This is
// the per-turn overhead the change adds on the no-middleware path.
//
// Target per task 2.3: under 50ns on commodity hardware.
func BenchmarkShortCircuit_Helper_NoSlice(b *testing.B) {
	mw := MiddlewareSet{}
	req := &llm.Request{Model: "test"}
	ctx := context.Background()
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		// One len() check + zero iterations of the loop.
		if _, err := shortCircuit(ctx, mw, req); err != nil {
			b.Fatalf("shortCircuit: %v", err)
		}
	}
}

// BenchmarkShortCircuit_Helper_WithSlice measures the cost of the
// helper when one short-circuiter is registered and misses. The
// delta vs Helper_NoSlice is the per-turn cost of one interface
// dispatch on the registered short-circuiter.
func BenchmarkShortCircuit_Helper_WithSlice(b *testing.B) {
	sc := &rtShortCircuiter{hitMsg: nil}
	mw := MiddlewareSet{
		RequestShortCircuiters: []runtimeRequestShortCircuiter{sc},
	}
	req := &llm.Request{Model: "test"}
	ctx := context.Background()
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := shortCircuit(ctx, mw, req); err != nil {
			b.Fatalf("shortCircuit: %v", err)
		}
	}
}
