// middleware.go — runtime-side interfaces for the in-process middleware seam.
//
// The public surface lives in pkg/tau/middleware.go; this file defines the
// internal interfaces the agent loop dispatches against. The SDK layer
// type-checks embedder-supplied values against the public interfaces and
// wraps each one in a concrete adapter (sdkRequestMutator,
// sdkResponseObserver, sdkToolInterceptor) so the runtime does not import
// pkg/tau and pkg/tau does not leak its adapter types back into the runtime.
//
// The runtime trusts the SDK's pre-validation: it never type-checks a
// MiddlewareSet element; it only checks the slice lengths to take the
// no-middleware fast path on the hot loop.

package agent

import (
	"context"
	"errors"
	"fmt"
	"log"

	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/tools"
)

// runtimeRequestMutator is the runtime-side interface for an
// in-process request mutator. Adapters wrap the SDK's RequestMutator
// to satisfy this signature.
type runtimeRequestMutator interface {
	MutateRequest(ctx context.Context, req *llm.Request) error
}

// runtimeResponseObserver is the runtime-side interface for an
// in-process response observer. Adapters wrap the SDK's
// ResponseObserver.
//
// streamErr is the error from LLMClient.Stream (nil on success). It
// lets observers record the actual failure cause rather than inferring
// one from an empty response.
type runtimeResponseObserver interface {
	ObserveResponse(ctx context.Context, req *llm.Request, resp *llm.Message, streamErr error) error
}

// runtimeToolInterceptor is the runtime-side interface for an
// in-process tool interceptor. Adapters wrap the SDK's
// ToolInterceptor.
type runtimeToolInterceptor interface {
	BeforeToolCall(ctx context.Context, call tools.ToolCall) (*tools.ToolResult, error)
	AfterToolCall(ctx context.Context, call tools.ToolCall, result tools.ToolResult) error
}

// runtimeRequestShortCircuiter is the runtime-side interface for an
// in-process request short-circuiter. Adapters wrap the SDK's
// RequestShortCircuiter. The three-return-shape contract (hit, miss,
// error) is enforced by the turn loop in session.go.
type runtimeRequestShortCircuiter interface {
	ShortCircuit(ctx context.Context, req *llm.Request) (*llm.Message, error)
}

// MiddlewareSet is the partitioned, type-checked middleware bundle the
// runtime consumes. CreateAgentSession in pkg/tau builds one via
// partitionMiddleware; an empty MiddlewareSet is the no-middleware
// default and the runtime takes a nil-slice fast path.
type MiddlewareSet struct {
	// RequestMutators runs before each LLM round-trip, in registration
	// order. A non-nil error aborts the turn.
	RequestMutators []runtimeRequestMutator

	// ResponseObservers runs after each LLM round-trip, in registration
	// order. Errors are logged but do not abort.
	ResponseObservers []runtimeResponseObserver

	// ToolInterceptors runs around each tool execution, in registration
	// order. BeforeToolCall may short-circuit (non-nil *ToolResult) or
	// abort (non-nil error); AfterToolCall errors are logged but do not
	// abort.
	ToolInterceptors []runtimeToolInterceptor

	// RequestShortCircuiters runs after RequestMutators and after
	// MessageStartEvent, before LLMClient.Stream. The first to return
	// a non-nil *llm.Message wins; subsequent short-circuiters are NOT
	// invoked for the current turn. A non-nil error aborts the turn.
	RequestShortCircuiters []runtimeRequestShortCircuiter
}

// empty reports whether the set has zero middleware of any type. Used by
// the runtime's hot-loop fast path.
func (m MiddlewareSet) empty() bool {
	return len(m.RequestMutators) == 0 && len(m.ResponseObservers) == 0 && len(m.ToolInterceptors) == 0 && len(m.RequestShortCircuiters) == 0
}

// errInterceptorAbort is the sentinel wrapper that marks a
// ToolInterceptor.BeforeToolCall error as a turn-level abort. The turn
// loop in session.go checks errors.Is(err, errInterceptorAbort) to
// distinguish an interceptor abort from a tool-execution error (which
// is captured in the ToolResult as IsError=true and the turn continues).
//
// The embedder's original error is preserved via %w wrapping so callers
// can errors.Is against their own sentinel as well.
var errInterceptorAbort = errors.New("agent: tool interceptor abort")

// wrapInterceptorAbort marks err as a turn-level interceptor abort. If
// err is nil, returns nil.
func wrapInterceptorAbort(err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("%w: %w", errInterceptorAbort, err)
}

// observeResponse invokes every registered ResponseObserver in
// registration order with the (request, response, streamErr) triple.
// Used by the turn loop after LLMClient.Stream returns (whether
// successfully or with error). streamErr is nil on success; when
// non-nil, resp is a zero-value *Message and streamErr carries the
// real failure cause so observers can record it without inferring
// from an empty response.
//
// Observer errors are logged via the standard log package and do NOT
// short-circuit the remaining observers or the turn. This honours the
// asymmetric error-propagation contract: observing hooks may fail
// without breaking the agent loop.
func observeResponse(ctx context.Context, mw MiddlewareSet, req *llm.Request, resp *llm.Message, streamErr error) {
	for _, o := range mw.ResponseObservers {
		if err := o.ObserveResponse(ctx, req, resp, streamErr); err != nil {
			log.Printf("agent: response observer %T: %v", o, err)
		}
	}
}

// interceptBefore invokes every registered ToolInterceptor's
// BeforeToolCall in registration order. Returns:
//
//   - (shortCircuit, nil) when any interceptor returned a non-nil
//     *ToolResult. Subsequent interceptors are NOT invoked; the caller
//     skips the underlying tool's Execute.
//   - (nil, err) when any interceptor returned a non-nil error. The
//     turn aborts; the caller surfaces err to Run's caller.
//   - (nil, nil) when every interceptor returned (nil, nil). The
//     caller proceeds to invoke the tool's Execute.
func interceptBefore(ctx context.Context, mw MiddlewareSet, call tools.ToolCall) (*tools.ToolResult, error) {
	for _, i := range mw.ToolInterceptors {
		r, err := i.BeforeToolCall(ctx, call)
		if err != nil {
			return nil, wrapInterceptorAbort(err)
		}
		if r != nil {
			return r, nil
		}
	}
	return nil, nil
}

// interceptAfter invokes every registered ToolInterceptor's
// AfterToolCall in registration order with the call and its result.
// Errors are logged via the standard log package and do NOT
// short-circuit the remaining interceptors or the turn. result is the
// value the runtime already committed to; AfterToolCall cannot mutate
// it.
func interceptAfter(ctx context.Context, mw MiddlewareSet, call tools.ToolCall, result tools.ToolResult) {
	for _, i := range mw.ToolInterceptors {
		if err := i.AfterToolCall(ctx, call, result); err != nil {
			log.Printf("agent: tool interceptor after %T: %v", i, err)
		}
	}
}

// shortCircuit invokes every registered RequestShortCircuiter in
// registration order. Returns:
//
//   - (non-nil, nil) when any short-circuiter returned a non-nil
//     *llm.Message. Subsequent short-circuiters are NOT invoked; the
//     caller skips LLMClient.Stream and feeds the returned message
//     through observeResponse + executeTools.
//   - (nil, err) when any short-circuiter returned a non-nil error.
//     The turn aborts; the caller surfaces err to Run's caller.
//   - (nil, nil) when every short-circuiter returned (nil, nil). The
//     caller proceeds to LLMClient.Stream.
func shortCircuit(ctx context.Context, mw MiddlewareSet, req *llm.Request) (*llm.Message, error) {
	for _, sc := range mw.RequestShortCircuiters {
		msg, err := sc.ShortCircuit(ctx, req)
		if err != nil {
			return nil, fmt.Errorf("agent: short-circuit: %w", err)
		}
		if msg != nil {
			return msg, nil
		}
	}
	return nil, nil
}
