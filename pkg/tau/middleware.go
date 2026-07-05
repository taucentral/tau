// middleware.go — public SDK surface for the in-process middleware seam.
//
// The middleware seam is the first of three extension points defined in
// docs/input/context/plugin-support/whitepaper.md (§3.2). It lets an
// embedder hook three points on the agent turn loop WITHOUT spawning a
// subprocess or marshallng the request over gRPC:
//
//   - RequestMutator     — rewrite the outgoing *Request in place before
//                          it reaches the LLM provider.
//   - ResponseObserver   — inspect the completed *Response (and the
//                          *Request that produced it) without mutating.
//   - ToolInterceptor    — gate, audit, or short-circuit each tool call
//                          within a turn.
//
// Middleware is registered at session construction via Options.Middleware
// (a []any slice). CreateAgentSession type-checks each element against
// the three interfaces and partitions it into three typed slices that
// the runtime invokes in registration order, within each type.
//
// Error propagation is asymmetric by design:
//   - RequestMutator and ToolInterceptor.BeforeToolCall errors ABORT the
//     turn (they are gating hooks).
//   - ResponseObserver and ToolInterceptor.AfterToolCall errors are
//     LOGGED but do NOT abort (they are observing hooks; a buggy audit
//     logger must not break the agent loop).
//
// See the request-middleware capability spec for the full contract.

package tau

import (
	"context"
	"errors"
)

// RequestMutator modifies an outgoing LLM request in place before it
// reaches the provider. Use cases: tool schema pruning, system prompt
// augmentation, memory injection, cache-control header injection.
//
// The runtime invokes every registered RequestMutator exactly once per
// LLM round-trip, after assembling the request and before calling
// LLMClient.Stream. Mutators run in registration order; a mutator sees
// the mutations of every earlier mutator.
//
// A non-nil return aborts the turn immediately: remaining mutators are
// NOT invoked, the LLM provider is NOT called, and Run returns the
// error to its caller.
type RequestMutator interface {
	MutateRequest(ctx context.Context, req *Request) error
}

// ResponseObserver inspects a completed LLM response without modifying
// it. Use cases: audit logging, token accounting, telemetry, cost
// tracking.
//
// The runtime invokes every registered ResponseObserver exactly once
// per LLM round-trip, after LLMClient.Stream returns — whether the
// stream completed successfully or returned an error. Observers see
// the (request, response, err) triple; the request is the post-mutator
// request that was sent; the response is whatever the provider
// returned (which may be a zero-value Response when Stream errored
// before producing any output); err is the error from Stream (nil on
// success). When err is non-nil the response is guaranteed to be
// zero-value, so observers can reliably use err to detect the
// failure path and record the real error string.
//
// A non-nil return is logged but does NOT abort the turn; remaining
// observers still run. This lets audit/telemetry observers fail
// without breaking the agent loop.
type ResponseObserver interface {
	ObserveResponse(ctx context.Context, req *Request, resp *Response, err error) error
}

// ToolInterceptor wraps each tool execution within a turn. The two
// methods form a before/after pair:
//
//   - BeforeToolCall runs immediately before the tool's Execute. It
//     may short-circuit by returning a non-nil *ToolResult — the
//     runtime skips Execute and uses the returned result for the rest
//     of the turn (after still invoking AfterToolCall with the
//     short-circuit result). A non-nil error aborts the turn.
//
//   - AfterToolCall runs immediately after Execute returns (or after a
//     short-circuit result is produced). It observes the result; a
//     non-nil error is logged but does NOT abort the turn.
//
// Use cases: permission gating (BeforeToolCall short-circuits with a
// "denied" ToolResult), argument validation (same), audit logging
// (AfterToolCall), rate limiting (BeforeToolCall).
type ToolInterceptor interface {
	// BeforeToolCall gates the upcoming tool call. Returning a
	// non-nil *ToolResult short-circuits Execute; the runtime uses
	// the returned result as the tool's output. Returning a non-nil
	// error aborts the turn. Returning (nil, nil) permits Execute
	// to run.
	BeforeToolCall(ctx context.Context, call ToolCall) (*ToolResult, error)

	// AfterToolCall observes the call's result. The runtime invokes
	// this after Execute returns (or after a short-circuit result is
	// produced). result is the value the runtime will use for the
	// rest of the turn; the interceptor cannot mutate it. A non-nil
	// error is logged but does NOT abort the turn.
	AfterToolCall(ctx context.Context, call ToolCall, result ToolResult) error
}

// RequestShortCircuiter may short-circuit the LLM round-trip by
// returning a complete *Message without calling LLMClient.Stream.
//
// The runtime invokes every registered RequestShortCircuiter exactly
// once per LLM round-trip, in registration order, AFTER every
// RequestMutator has completed and AFTER MessageStartEvent has been
// published, but BEFORE LLMClient.Stream is called. The first
// short-circuiter to return a non-nil *Message wins; subsequent
// short-circuiters are NOT invoked for the current turn.
//
// ShortCircuit returns (*Message, error) with three defined shapes:
//
//   - (non-nil, nil): short-circuit hit. The runtime skips
//     LLMClient.Stream and feeds the returned *Message through
//     ResponseObserver and the tool-execution path exactly as if the
//     provider had returned it. If the message has StopReason ==
//     ToolUse, tool calls execute normally (ToolInterceptor,
//     Execute, AfterToolCall all fire).
//
//   - (nil, nil): miss. The runtime proceeds to the next
//     short-circuiter, or to LLMClient.Stream if none remain.
//
//   - (nil, err): error. The runtime aborts the turn immediately and
//     returns the error wrapped as "agent: short-circuit: <err>" to
//     the caller of Run. LLMClient.Stream is NOT called.
//
// The runtime sets Message.Source = "cache" on the returned message
// before invoking ResponseObserver so billing/telemetry observers can
// distinguish cache hits from real LLM calls.
//
// Composition rule: mutators are policy; short-circuiters are
// performance. A RequestMutator that returns a non-nil error prevents
// any short-circuiter from running. To block even cached responses,
// register as a mutator.
//
// Cache plugins returning tool-call responses (StopReason == ToolUse)
// SHALL ensure the tool calls are idempotent — the runtime
// re-executes them against the live system. A future SafeToReplay
// flag is a follow-up.
//
// The short-circuiter sees the request AFTER mutator normalization.
// Plugins wanting cross-session cache keys SHALL strip session-specific
// fields inside their ShortCircuit implementation.
type RequestShortCircuiter interface {
	ShortCircuit(ctx context.Context, req *Request) (*Message, error)
}

// ErrUnknownMiddlewareType is returned by CreateAgentSession when an
// element of Options.Middleware satisfies none of RequestMutator,
// ResponseObserver, ToolInterceptor, or RequestShortCircuiter. The
// error is compatible with errors.Is so embedders can distinguish it
// from other constructor errors. The error message names the offending
// value's Go type via fmt.Sprintf("%T", v).
//
// Verify with: errors.Is(err, tau.ErrUnknownMiddlewareType).
var ErrUnknownMiddlewareType = errors.New("tau: unknown middleware type")

// Compile-time check: the sentinel satisfies the standard errors.Is
// identity contract.
var _ error = ErrUnknownMiddlewareType
