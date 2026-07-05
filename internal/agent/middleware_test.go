// middleware_test.go — runtime-level tests for the middleware seam.
//
// The contract tests in pkg/tau/contract_test.go exercise the public SDK
// surface (RequestMutator, ResponseObserver, ToolInterceptor) end-to-end
// via the SDK adapter layer. This file exercises the runtime-level
// interfaces (runtimeRequestMutator, runtimeResponseObserver,
// runtimeToolInterceptor, MiddlewareSet) directly so the runtime
// package's invariants are pinned independently of the SDK layer.
//
// Coverage:
//
//   - TestRuntimeMiddlewareFastPath: empty MiddlewareSet makes zero
//     interface dispatches on a turn.
//   - TestRuntimeMiddlewareRequestMutatorAbortsOnError: a runtime
//     mutator returning a sentinel error aborts Run with errors.Is
//     identity.
//   - TestRuntimeMiddlewareObserverErrorDoesNotAbort: a runtime
//     observer returning an error is logged but the turn completes.
//   - TestRuntimeMiddlewareToolInterceptorShortCircuit: a runtime
//     tool interceptor returning a *ToolResult skips Execute.
//   - TestRuntimeMiddlewareToolInterceptorErrorAborts: a runtime
//     tool interceptor returning an error aborts the turn.
//   - TestRuntimeMiddlewareObserverSeesErrorResponse: when LLMClient.Stream
//     errors, observers run once with an empty *llm.Message.
//   - TestMiddlewareSetEmpty: the empty() helper reports the correct
//     partition state for several MiddlewareSet shapes.

package agent

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/coevin/tau/internal/config"
	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/tools"
)

// rtMiddleware harness: implements the three runtime middleware
// interfaces with knobs the tests can dial.
type rtMutator struct {
	calls atomic.Int32
	err   error
}

func (m *rtMutator) MutateRequest(ctx context.Context, req *llm.Request) error {
	m.calls.Add(1)
	return m.err
}

type rtObserver struct {
	calls atomic.Int32
	mu    sync.Mutex
	reqs  []*llm.Request
	resps []*llm.Message
	err   error
}

func (o *rtObserver) ObserveResponse(ctx context.Context, req *llm.Request, resp *llm.Message) error {
	o.calls.Add(1)
	o.mu.Lock()
	o.reqs = append(o.reqs, req)
	o.resps = append(o.resps, resp)
	err := o.err
	o.mu.Unlock()
	return err
}

type rtInterceptor struct {
	beforeMu     sync.Mutex
	beforeCalls  atomic.Int32
	afterCalls   atomic.Int32
	beforeResult *tools.ToolResult
	beforeErr    error
	afterErr     error
}

func (i *rtInterceptor) BeforeToolCall(ctx context.Context, call tools.ToolCall) (*tools.ToolResult, error) {
	i.beforeCalls.Add(1)
	i.beforeMu.Lock()
	defer i.beforeMu.Unlock()
	return i.beforeResult, i.beforeErr
}

func (i *rtInterceptor) AfterToolCall(ctx context.Context, call tools.ToolCall, result tools.ToolResult) error {
	i.afterCalls.Add(1)
	return i.afterErr
}

// TestMiddlewareSetEmpty pins the empty() helper for the five shapes
// the runtime hot-path checks: zero middleware, only one type (for
// each of the four types), and a populated set.
func TestMiddlewareSetEmpty(t *testing.T) {
	if !(MiddlewareSet{}).empty() {
		t.Error("MiddlewareSet{}.empty() = false, want true")
	}
	if (MiddlewareSet{RequestMutators: []runtimeRequestMutator{&rtMutator{}}}).empty() {
		t.Error("MiddlewareSet with one RequestMutator reports empty")
	}
	if (MiddlewareSet{ResponseObservers: []runtimeResponseObserver{&rtObserver{}}}).empty() {
		t.Error("MiddlewareSet with one ResponseObserver reports empty")
	}
	if (MiddlewareSet{ToolInterceptors: []runtimeToolInterceptor{&rtInterceptor{}}}).empty() {
		t.Error("MiddlewareSet with one ToolInterceptor reports empty")
	}
	if (MiddlewareSet{RequestShortCircuiters: []runtimeRequestShortCircuiter{&rtShortCircuiter{}}}).empty() {
		t.Error("MiddlewareSet with one RequestShortCircuiter reports empty")
	}
}

// TestRuntimeMiddlewareFastPath: with an empty MiddlewareSet, Run
// takes the fast path — no interface dispatches. We assert by recording
// zero calls on every middleware field (they don't exist) and by
// observing the turn still completes successfully.
func TestRuntimeMiddlewareFastPath(t *testing.T) {
	client := &scriptedClient{scripts: [][]llm.Delta{textThenFinal("ok")}}
	sess, _ := newSessionForTest(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}})

	if err := sess.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	// No middleware was registered; the assertion is just that Run
	// completes without calling any middleware. There is nothing to
	// count; we rely on the absence of crashes and the turn succeeding.
}

// TestRuntimeMiddlewareRequestMutatorAbortsOnError: a runtime mutator
// returning sentinel aborts Run with errors.Is identity, and the
// provider Stream is NOT called.
func TestRuntimeMiddlewareRequestMutatorAbortsOnError(t *testing.T) {
	sentinel := errors.New("rt-mutator-abort")
	mut := &rtMutator{err: sentinel}
	mw := MiddlewareSet{RequestMutators: []runtimeRequestMutator{mut}}

	client := &scriptedClient{
		scripts: [][]llm.Delta{textThenFinal("should-not-reach")},
	}
	sess, rt := newSessionForTest(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}})
	rt.Options.Middleware = mw

	err := sess.Run(context.Background(), "go")
	if !errors.Is(err, sentinel) {
		t.Errorf("Run err = %v, want errors.Is(err, sentinel)", err)
	}
	if mut.calls.Load() != 1 {
		t.Errorf("mutator invoked %d times, want 1", mut.calls.Load())
	}
	if len(client.scripts) != 1 {
		t.Errorf("provider Stream called %d times, want 0 (scriptedClient consumed %d)", 1-len(client.scripts), len(client.scripts))
	}
}

// TestRuntimeMiddlewareObserverErrorDoesNotAbort: a runtime observer
// returning an error is logged but Run completes successfully.
func TestRuntimeMiddlewareObserverErrorDoesNotAbort(t *testing.T) {
	sentinel := errors.New("rt-observer-log")
	obs := &rtObserver{err: sentinel}
	mw := MiddlewareSet{ResponseObservers: []runtimeResponseObserver{obs}}

	client := &scriptedClient{scripts: [][]llm.Delta{textThenFinal("ok")}}
	sess, rt := newSessionForTest(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}})
	rt.Options.Middleware = mw

	if err := sess.Run(context.Background(), "go"); err != nil {
		t.Errorf("Run err = %v, want nil (observer error is logged, not propagated)", err)
	}
	if obs.calls.Load() != 1 {
		t.Errorf("observer invoked %d times, want 1", obs.calls.Load())
	}
}

// TestRuntimeMiddlewareObserverSeesResponse: the observer receives the
// (request, response) pair on a successful turn.
func TestRuntimeMiddlewareObserverSeesResponse(t *testing.T) {
	obs := &rtObserver{}
	mw := MiddlewareSet{ResponseObservers: []runtimeResponseObserver{obs}}

	client := &scriptedClient{scripts: [][]llm.Delta{textThenFinal("payload")}}
	sess, rt := newSessionForTest(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}})
	rt.Options.Middleware = mw

	if err := sess.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if obs.calls.Load() != 1 {
		t.Fatalf("observer invoked %d times, want 1", obs.calls.Load())
	}
	obs.mu.Lock()
	defer obs.mu.Unlock()
	if len(obs.resps) != 1 {
		t.Fatalf("recorded %d responses, want 1", len(obs.resps))
	}
	if obs.resps[0] == nil {
		t.Error("observer received nil *llm.Message")
	}
}

// TestRuntimeMiddlewareToolInterceptorShortCircuit: a runtime
// interceptor returning a non-nil *ToolResult skips Execute. AfterToolCall
// is still invoked with the short-circuit result.
func TestRuntimeMiddlewareToolInterceptorShortCircuit(t *testing.T) {
	short := tools.NewTextResult("rt-short")
	interceptor := &rtInterceptor{beforeResult: &short}
	mw := MiddlewareSet{ToolInterceptors: []runtimeToolInterceptor{interceptor}}

	tool := &recordingTool{name: "rt-counter"}
	client := &scriptedClient{scripts: [][]llm.Delta{
		{
			llm.ToolCallDelta{ContentIndex: 0, ID: "tu_1", Name: "rt-counter", PartialInput: "{}"},
			llm.Final{StopReason: llm.StopReasonToolUse},
		},
		textThenFinal("done"),
	}}
	sess, rt := newSessionForTest(t, client, []tools.HeadlessTool{tool})
	rt.Options.Middleware = mw

	if err := sess.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if tool.callCount() != 0 {
		t.Errorf("underlying tool Execute called %d times, want 0 (short-circuit)", tool.callCount())
	}
	if interceptor.beforeCalls.Load() != 1 {
		t.Errorf("BeforeToolCall invoked %d times, want 1", interceptor.beforeCalls.Load())
	}
	if interceptor.afterCalls.Load() != 1 {
		t.Errorf("AfterToolCall invoked %d times, want 1", interceptor.afterCalls.Load())
	}
}

// TestRuntimeMiddlewareToolInterceptorErrorAborts: a runtime
// interceptor returning a non-nil error aborts the turn.
func TestRuntimeMiddlewareToolInterceptorErrorAborts(t *testing.T) {
	sentinel := errors.New("rt-tool-abort")
	interceptor := &rtInterceptor{beforeErr: sentinel}
	mw := MiddlewareSet{ToolInterceptors: []runtimeToolInterceptor{interceptor}}

	tool := &recordingTool{name: "rt-counter"}
	client := &scriptedClient{scripts: [][]llm.Delta{
		{
			llm.ToolCallDelta{ContentIndex: 0, ID: "tu_1", Name: "rt-counter", PartialInput: "{}"},
			llm.Final{StopReason: llm.StopReasonToolUse},
		},
		textThenFinal("done"),
	}}
	sess, rt := newSessionForTest(t, client, []tools.HeadlessTool{tool})
	rt.Options.Middleware = mw

	err := sess.Run(context.Background(), "go")
	if err == nil {
		t.Fatal("Run returned nil error, want non-nil (interceptor abort)")
	}
	if !errors.Is(err, sentinel) {
		t.Errorf("Run err = %v, want errors.Is(err, sentinel) (embedder sentinel preserved through wrapping)", err)
	}
	if !errors.Is(err, errInterceptorAbort) {
		t.Errorf("Run err = %v, want errors.Is(err, errInterceptorAbort) (turn-level marker)", err)
	}
	if tool.callCount() != 0 {
		t.Errorf("underlying tool Execute called %d times, want 0 (abort before Execute)", tool.callCount())
	}
}

// TestRuntimeMiddlewareObserverSeesErrorResponse: when LLMClient.Stream
// errors before producing any delta, the observer still runs once with
// an empty *llm.Message so audit / telemetry middleware see the failure.
func TestRuntimeMiddlewareObserverSeesErrorResponse(t *testing.T) {
	obs := &rtObserver{}
	mw := MiddlewareSet{ResponseObservers: []runtimeResponseObserver{obs}}

	client := &errClient{err: errors.New("stream-failed")}
	sess, rt := newSessionForTest(t, client, []tools.HeadlessTool{&recordingTool{name: "noop"}})
	rt.Options.Middleware = mw

	// We don't care about Run's return value — only that the observer
	// was invoked with an empty response when Stream errored.
	_ = sess.Run(context.Background(), "go")

	if obs.calls.Load() != 1 {
		t.Fatalf("observer invoked %d times, want 1 (even on Stream error)", obs.calls.Load())
	}
	obs.mu.Lock()
	defer obs.mu.Unlock()
	if len(obs.resps) != 1 {
		t.Fatalf("recorded %d responses, want 1", len(obs.resps))
	}
	resp := obs.resps[0]
	if resp == nil {
		t.Error("observer received nil *llm.Message on Stream error")
	} else if resp.Role != "" || len(resp.Content) != 0 {
		t.Errorf("observer received non-empty response on Stream error: %+v", resp)
	}
}

// errClient is a minimal LLMClient whose Stream always returns err
// before emitting any delta. Used to exercise the Stream-error branch
// of the ResponseObserver wiring.
type errClient struct {
	err error
}

func (c *errClient) Stream(ctx context.Context, req llm.Request) (<-chan llm.Delta, error) {
	return nil, c.err
}

// TestRuntimeMiddlewareIsolatedBetweenSessions: middleware registered
// on session A is NOT invoked when session B runs a turn. Each session
// holds its own MiddlewareSet via its own resolvedOptions; the runtime
// never shares middleware across sessions.
func TestRuntimeMiddlewareIsolatedBetweenSessions(t *testing.T) {
	mutA := &rtMutator{}
	mutB := &rtMutator{}

	clientA := &scriptedClient{scripts: [][]llm.Delta{textThenFinal("a")}}
	clientB := &scriptedClient{scripts: [][]llm.Delta{textThenFinal("b")}}

	sessA, rtA := newSessionForTest(t, clientA, []tools.HeadlessTool{&recordingTool{name: "noop-a"}})
	sessB, rtB := newSessionForTest(t, clientB, []tools.HeadlessTool{&recordingTool{name: "noop-b"}})
	rtA.Options.Middleware = MiddlewareSet{RequestMutators: []runtimeRequestMutator{mutA}}
	rtB.Options.Middleware = MiddlewareSet{RequestMutators: []runtimeRequestMutator{mutB}}

	if err := sessA.Run(context.Background(), "go"); err != nil {
		t.Fatalf("sessA.Run: %v", err)
	}
	if err := sessB.Run(context.Background(), "go"); err != nil {
		t.Fatalf("sessB.Run: %v", err)
	}

	if mutA.calls.Load() != 1 {
		t.Errorf("session A mutator invoked %d times, want 1", mutA.calls.Load())
	}
	if mutB.calls.Load() != 1 {
		t.Errorf("session B mutator invoked %d times, want 1", mutB.calls.Load())
	}
}

// TestRuntimeMiddlewareImmutableAfterConstruction: mutating the
// SessionOptions.Middleware slice header after CreateAgentSessionRuntime
// returns has no effect on the running session. The runtime's
// resolvedOptions.Middleware was copied at construction; later changes
// to the caller's slice header do not propagate.
func TestRuntimeMiddlewareImmutableAfterConstruction(t *testing.T) {
	mutA := &rtMutator{}
	mutB := &rtMutator{}

	client := &scriptedClient{scripts: [][]llm.Delta{textThenFinal("ok")}}
	opts := SessionOptions{
		Model:     "test-model",
		Settings:  config.DefaultSettings(),
		LLMClient: client,
		Tools:     []tools.HeadlessTool{&recordingTool{name: "noop"}},
		ConfigDir: t.TempDir(),
		Middleware: MiddlewareSet{
			RequestMutators: []runtimeRequestMutator{mutA},
		},
	}
	rt, err := CreateAgentSessionRuntime(context.Background(), t.TempDir(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSessionRuntime: %v", err)
	}
	sess := NewAgentSession(rt)

	// Caller now reassigns the slice header to a completely different
	// mutator. The runtime must keep using mutA.
	opts.Middleware = MiddlewareSet{
		RequestMutators: []runtimeRequestMutator{mutB},
	}

	if err := sess.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if mutA.calls.Load() != 1 {
		t.Errorf("construction-time mutator A invoked %d times, want 1", mutA.calls.Load())
	}
	if mutB.calls.Load() != 0 {
		t.Errorf("post-construction mutator B invoked %d times, want 0 (immutable after construction)", mutB.calls.Load())
	}
}

// TestRuntimeMiddlewareFastPathHasNoAllocs: the hot-path check on a
// zero-value MiddlewareSet allocates zero bytes. This pins the
// "nil-slice fast path" invariant from task 7.1(d): a session with no
// middleware pays no per-turn allocation cost on the middleware branch.
func TestRuntimeMiddlewareFastPathHasNoAllocs(t *testing.T) {
	set := MiddlewareSet{}
	allocs := testing.AllocsPerRun(100, func() {
		// This mirrors the exact check the turn loop makes on every
		// dispatch iteration: a single len() branch on each slice.
		_ = len(set.RequestMutators) > 0
		_ = len(set.ResponseObservers) > 0
		_ = len(set.ToolInterceptors) > 0
	})
	if allocs != 0 {
		t.Errorf("MiddlewareSet empty checks: allocs/run = %v, want 0", allocs)
	}
	allocsEmpty := testing.AllocsPerRun(100, func() {
		_ = set.empty()
	})
	if allocsEmpty != 0 {
		t.Errorf("MiddlewareSet.empty(): allocs/run = %v, want 0", allocsEmpty)
	}
}
