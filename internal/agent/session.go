// session.go — the agentic turn loop.
//
// AgentSession wraps an AgentSessionRuntime with turn-tracking state:
// the current turn counter, the in-flight context.CancelFunc for Abort,
// and the SessionStart flag. The runtime holds the wired components; the
// session drives them.
//
// Turn loop (per the agent-loop spec "Agentic turn loop"):
//
//  1. Append the user message to the state tree.
//  2. MaybeCompact (honouring Settings.Compaction.Enabled).
//  3. Outer "dispatch" loop:
//     a. Build context from state (BuildContext → messages + system).
//     b. Assemble system prompt via prompts.Assembler.
//     c. Emit MessageStart.
//     d. Issue llmClient.Stream(ctx, req).
//     e. Consume deltas: emit MessageUpdate for each, accumulate into
//        an assistant Message.
//     f. On Final: extract any ToolUse blocks.
//     g. For each ToolUse: emit ToolCall, execute (concurrent or serial
//        per Settings.SteeringMode), emit ToolResult. Aggregate results
//        into a single tool-result message appended to state.
//     h. Emit MessageEnd with StopReason.
//     i. Persist the assistant message and (if any) tool-result message.
//     j. If StopReason == ToolUse and not aborted: continue the outer
//        loop with the updated context.
//     k. Else: emit TurnEnd and return.
//
// Cancellation (Abort): the active context.CancelFunc is invoked; the
// stream consumer and tool goroutines see ctx.Done() and unwind. The
// state tree remains consistent: already-persisted entries stay; the
// in-flight assistant message is dropped (the user can issue a new turn).
//
// Shutdown: emits SessionShutdown, closes the state manager (if owned),
// marks the session as terminal. Subsequent Run calls return
// ErrRuntimeShutdown.

package agent

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/state"
	"github.com/coevin/tau/internal/tools"
)

// AgentSession pairs a wired AgentSessionRuntime with turn-tracking
// state. Construct via NewAgentSession. A session is NOT safe to Run
// concurrently from multiple goroutines — the model is single-threaded
// per session. Abort and Shutdown MAY be called concurrently with Run.
//
//nolint:revive // exported name is part of the public SDK API surface (pkg/tau).
type AgentSession struct {
	rt *AgentSessionRuntime

	// turnCounter is 0 before the first turn, 1+ once Run has been
	// called. Read via atomic so Abort/Shutdown can observe the
	// current turn without taking the session lock.
	turnCounter atomic.Int32

	// sessionStarted is set true after the first SessionStartEvent is
	// emitted. Subsequent turns do not re-emit SessionStart.
	sessionStarted atomic.Bool

	// cancelMu guards cancelFn. Run stores its cancelFn here so Abort
	// can invoke it. Cleared on turn exit.
	cancelMu sync.Mutex
	cancelFn context.CancelFunc

	// shutdownMu (on rt) guards the one-shot shutdown fence.
}

// NewAgentSession wraps rt in an AgentSession. Returns nil if rt is nil
// so callers can chain without a separate nil check.
func NewAgentSession(rt *AgentSessionRuntime) *AgentSession {
	if rt == nil {
		return nil
	}
	return &AgentSession{rt: rt}
}

// Runtime returns the wired runtime the session drives. Exposed so the
// CLI / TUI / SDK can subscribe to the EventBus, inspect the State
// manager, or read the Registry without reaching into private fields.
func (s *AgentSession) Runtime() *AgentSessionRuntime { return s.rt }

// Run executes one user turn against the agent session. It blocks until
// the model returns a final response with no pending tool calls, the
// context is cancelled (Abort or external cancel), or an unrecoverable
// error occurs.
//
// Run must NOT be called concurrently with itself on the same session.
// Abort and Shutdown MAY be called concurrently with Run.
//
// Returns nil on normal completion. Returns context.Canceled (wrapped)
// when the context is cancelled mid-turn — the state tree remains
// consistent and a subsequent Run may be issued.
func (s *AgentSession) Run(ctx context.Context, userInput string) error {
	if s == nil || s.rt == nil {
		return errors.New("agent: session has no runtime")
	}
	// Reject Run on a shut-down runtime. The shutdown flag lives on rt.
	s.rt.shutdownMu.Lock()
	shutdown := s.rt.shutdownDone
	s.rt.shutdownMu.Unlock()
	if shutdown {
		return ErrRuntimeShutdown
	}
	if userInput == "" {
		return errors.New("agent: Run requires non-empty user input")
	}

	// Per-turn cancellable context. Stored so Abort can trigger it.
	turnCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	s.setCancelFn(cancel)
	defer s.setCancelFn(nil)

	turnNum := int(s.turnCounter.Add(1))
	now := time.Now()

	// SessionStart fires once per session before the first TurnStart.
	if !s.sessionStarted.Swap(true) {
		s.rt.EventBus.Publish(SessionStartEvent{
			When:      now,
			SessionID: s.rt.SessionID,
			Model:     s.rt.Options.Model,
			Cwd:       s.rt.Cwd,
		})
	}
	s.rt.EventBus.Publish(TurnStartEvent{
		When: now,
		Turn: turnNum,
	})

	// Step 1: append the user message. Persisted even if the turn later
	// aborts — the user expressed intent, and a follow-up turn will
	// need this entry to make sense of the conversation.
	userMsg := llm.Message{
		Role:    llm.RoleUser,
		Content: []llm.ContentBlock{llm.TextContent{Text: userInput}},
	}
	if _, err := s.rt.State.Append(state.Entry{
		Kind:    state.KindMessage,
		Payload: state.MessagePayload{Role: userMsg.Role, Content: userMsg.Content},
	}); err != nil {
		return fmt.Errorf("agent: append user message: %w", err)
	}

	// Step 2: compaction. Skipped when Settings.Compaction.Enabled is
	// explicitly false, or when ContextWindow is unknown (the compactor
	// would compute a nonsensical negative threshold). Errors are
	// surfaced but non-fatal: the turn continues with the un-compacted
	// context.
	if s.compactionEnabled() && s.rt.Options.ContextWindow > 0 {
		if _, err := s.rt.Compactor.MaybeCompact(turnCtx, s.rt.State, s.rt.Options.Model, s.rt.Options.ContextWindow); err != nil {
			s.rt.EventBus.Publish(registrationCollisionEvent{
				When:   time.Now(),
				Name:   "compaction",
				Reason: err.Error(),
			})
		}
	}

	// Step 3: outer dispatch loop. Each iteration issues one llm.Stream
	// call and processes its result.
	for {
		if err := turnCtx.Err(); err != nil {
			return s.abortTurn(err)
		}
		// Build the request from current state.
		req, err := s.buildRequest(turnCtx)
		if err != nil {
			return fmt.Errorf("agent: build request: %w", err)
		}

		// Middleware: RequestMutator. Runs in registration order before
		// the request reaches the provider. A nil slice is the fast path
		// — no interface dispatch, no allocation. A non-nil error aborts
		// the turn immediately (gating hook).
		if mw := s.rt.Options.Middleware.RequestMutators; len(mw) > 0 {
			for _, m := range mw {
				if err := m.MutateRequest(turnCtx, &req); err != nil {
					return fmt.Errorf("agent: request mutator: %w", err)
				}
			}
		}

		// Emit MessageStart before consuming any deltas.
		s.rt.EventBus.Publish(MessageStartEvent{When: time.Now()})

		// Issue the stream call.
		ch, err := s.rt.Options.LLMClient.Stream(turnCtx, req)
		if err != nil {
			// Even on failure, emit MessageEnd so subscribers know the
			// message slot is closed.
			s.rt.EventBus.Publish(MessageEndEvent{When: time.Now(), StopReason: llm.StopReasonError})
			// Middleware: ResponseObserver. Even on Stream error, observers
			// run once with the (request, empty-response) pair so audit /
			// telemetry middleware see the failure. Errors are logged but
			// do NOT abort the turn.
			observeResponse(turnCtx, s.rt.Options.Middleware, &req, &llm.Message{})
			if llm.IsAbort(err) {
				return s.abortTurn(err)
			}
			return fmt.Errorf("agent: stream: %w", err)
		}

		// Consume the delta stream, emitting MessageUpdate for each.
		assistant, finalErr := s.consumeStream(turnCtx, ch)

		// If consumeStream saw a context cancellation, unwind now.
		if finalErr != nil && llm.IsAbort(finalErr) {
			s.rt.EventBus.Publish(MessageEndEvent{When: time.Now(), StopReason: llm.StopReasonAborted})
			return s.abortTurn(finalErr)
		}
		// Other Final.Err values are surfaced but the turn continues:
		// the model's response (partial or complete) is still usable
		// for tool calls if it has any.

		stopReason := assistant.StopReason
		if finalErr != nil && stopReason == "" {
			stopReason = llm.StopReasonError
		}

		// Middleware: ResponseObserver. Runs in registration order after
		// the stream completes. The observer sees the (request, response)
		// pair — the response is the accumulated assistant Message. Errors
		// are logged but do NOT abort the turn (observing hook).
		observeResponse(turnCtx, s.rt.Options.Middleware, &req, &assistant)

		// Step 4: execute any tool calls. Per spec, tool_call/tool_result
		// events fire BEFORE message_end.
		var toolResults []llm.ToolResult
		if stopReason == llm.StopReasonToolUse {
			results, toolErr := s.executeTools(turnCtx, assistant)
			if toolErr != nil {
				if llm.IsAbort(toolErr) {
					s.rt.EventBus.Publish(MessageEndEvent{When: time.Now(), StopReason: llm.StopReasonAborted})
					return s.abortTurn(toolErr)
				}
				// ToolInterceptor.BeforeToolCall abort: surface to caller
				// per spec. Unlike a tool-execution error (which is
				// captured in the ToolResult as an IsError result and the
				// turn continues), an interceptor abort is a turn-level
				// signal the embedder raised explicitly.
				if errors.Is(toolErr, errInterceptorAbort) {
					s.rt.EventBus.Publish(MessageEndEvent{When: time.Now(), StopReason: llm.StopReasonError})
					return fmt.Errorf("agent: tool interceptor: %w", toolErr)
				}
				// Non-abort tool error: surface but keep going. The
				// error is captured in the relevant ToolResult already.
			}
			toolResults = results
		}

		// Emit MessageEnd now that tools have run (or there were none).
		s.rt.EventBus.Publish(MessageEndEvent{When: time.Now(), StopReason: stopReason})

		// Persist the assistant message to the state tree.
		if _, err := s.rt.State.Append(state.Entry{
			Kind:    state.KindMessage,
			Payload: state.MessagePayload{Role: assistant.Role, Content: assistant.Content},
		}); err != nil {
			return fmt.Errorf("agent: append assistant message: %w", err)
		}

		// If there are tool results, append them as a single user-role
		// message containing ToolResult blocks (Anthropic convention).
		if len(toolResults) > 0 {
			content := make([]llm.ContentBlock, 0, len(toolResults))
			for _, tr := range toolResults {
				content = append(content, tr)
			}
			if _, err := s.rt.State.Append(state.Entry{
				Kind:    state.KindMessage,
				Payload: state.MessagePayload{Role: llm.RoleUser, Content: content},
			}); err != nil {
				return fmt.Errorf("agent: append tool results: %w", err)
			}
		}

		// If the model stopped for any reason other than tool_use, the
		// turn is done.
		if stopReason != llm.StopReasonToolUse {
			break
		}
		// Otherwise loop: re-dispatch with the updated context.
	}

	s.rt.EventBus.Publish(TurnEndEvent{
		When:     time.Now(),
		Turn:     turnNum,
		Finished: true,
	})
	return nil
}

// abortTurn is the cancellation unwind path. Emits a TurnEnd with
// Finished=false and returns the wrapped cancellation error.
func (s *AgentSession) abortTurn(err error) error {
	s.rt.EventBus.Publish(TurnEndEvent{
		When:     time.Now(),
		Turn:     int(s.turnCounter.Load()),
		Finished: false,
	})
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return fmt.Errorf("agent: turn aborted: %w", err)
	}
	return err
}

// consumeStream drains the delta channel, emitting a MessageUpdate event
// for each delta and accumulating into an assistant Message. Returns the
// assembled message and the Final.Err (or a ctx error). The Message is
// usable even when finalErr is non-nil (it reflects what was collected).
func (s *AgentSession) consumeStream(ctx context.Context, ch <-chan llm.Delta) (llm.Message, error) {
	acc := llm.NewAccumulator(s.rt.Options.Model, "")
	for {
		select {
		case <-ctx.Done():
			return acc.Message(), ctx.Err()
		case d, ok := <-ch:
			if !ok {
				// Channel closed without Final.
				if acc.Err() == nil {
					return acc.Message(), llm.ErrFinalMissing
				}
				return acc.Message(), acc.Err()
			}
			// Emit a MessageUpdate for every delta (including Final —
			// subscribers that care to inspect Final type-switch).
			s.rt.EventBus.Publish(MessageUpdateEvent{
				When:  time.Now(),
				Delta: d,
			})
			if err := acc.Accumulate(d); err != nil {
				return acc.Message(), err
			}
		}
	}
}

// buildRequest assembles an llm.Request from the current state: system
// blocks come from prompts.Assembler, messages from state.BuildContext,
// tools from the runtime's Registry.
func (s *AgentSession) buildRequest(ctx context.Context) (llm.Request, error) {
	built, err := s.rt.State.BuildContext(ctx)
	if err != nil {
		return llm.Request{}, fmt.Errorf("build context: %w", err)
	}
	system, err := s.rt.Assembler.Assemble(ctx)
	if err != nil {
		return llm.Request{}, fmt.Errorf("assemble system: %w", err)
	}
	// If BuildContext returned system blocks (e.g., future synthetic
	// system messages), append ours after them so project guidance wins.
	systemBlocks := append([]llm.ContentBlock{}, built.System...)
	systemBlocks = append(systemBlocks, system...)

	return llm.Request{
		Model:     s.rt.Options.Model,
		System:    systemBlocks,
		Messages:  built.Messages,
		Tools:     s.rt.Registry.Schemas(),
		Transport: llm.Transport(s.rt.Options.Transport),
	}, nil
}

// executeTools runs every ToolUse in the assistant message, returning
// the slice of ToolResults (one per call). Concurrency is governed by
// Settings.SteeringMode: SteeringAll runs concurrently, anything else
// runs serially in the order the model emitted them.
//
// An error return indicates a ToolInterceptor.BeforeToolCall aborted the
// turn. The caller surfaces the error to Run's caller; partial results
// are still included in the returned slice so the event-bus trace stays
// consistent (ToolCall events fire before the abort; the aborting
// interceptor did NOT emit a ToolResult).
func (s *AgentSession) executeTools(ctx context.Context, assistant llm.Message) ([]llm.ToolResult, error) {
	calls := collectToolCalls(assistant)
	if len(calls) == 0 {
		return nil, nil
	}
	results := make([]llm.ToolResult, len(calls))
	concurrent := s.rt.Options.SteeringMode == steeringAllIdentifier
	if concurrent {
		var wg sync.WaitGroup
		errs := make([]error, len(calls))
		for i, c := range calls {
			wg.Add(1)
			go func(i int, c llm.ToolUse) {
				defer wg.Done()
				res, err := s.runOneTool(ctx, c)
				results[i] = res
				errs[i] = err
			}(i, c)
		}
		wg.Wait()
		// Aggregate errors — return the first non-nil for caller.
		for _, e := range errs {
			if e != nil {
				return results, e
			}
		}
	} else {
		for i, c := range calls {
			res, err := s.runOneTool(ctx, c)
			results[i] = res
			if err != nil {
				return results, err
			}
		}
	}
	return results, nil
}

// runOneTool emits ToolCall, executes the tool (subject to middleware),
// emits ToolResult, and returns the llm.ToolResult to be aggregated into
// the next request.
//
// Tool errors (file-not-found, non-zero exit) become IsError=true
// results; only infrastructure errors (panic, ctx cancellation) and
// ToolInterceptor.BeforeToolCall aborts return a non-nil error.
//
// Middleware wiring (when the runtime was constructed with middleware):
//
//   - Before the registry Lookup, ToolInterceptor.BeforeToolCall runs in
//     registration order. A short-circuit result (non-nil *ToolResult)
//     skips the underlying tool's Execute but still emits ToolCall and
//     ToolResult events so subscribers see consistent events. A non-nil
//     error short-circuits AND is returned to abort the turn.
//   - After Execute (or a short-circuit result), ToolInterceptor.AfterToolCall
//     runs in registration order. AfterToolCall errors are logged via the
//     standard log package and do NOT abort the turn.
func (s *AgentSession) runOneTool(ctx context.Context, use llm.ToolUse) (llm.ToolResult, error) {
	call := tools.ToolCall{
		ID:   use.ID,
		Name: use.Name,
		Args: use.Input,
		Cwd:  s.rt.Cwd,
	}
	s.rt.EventBus.Publish(ToolCallEvent{When: time.Now(), Call: call})

	// Middleware: ToolInterceptor.BeforeToolCall. Runs before Lookup so
	// the interceptor can deny calls to tools that aren't even registered
	// (e.g., a permission gate that whitelists by name). A short-circuit
	// result skips Execute AND Lookup; AfterToolCall still runs.
	mw := s.rt.Options.Middleware
	if len(mw.ToolInterceptors) > 0 {
		short, err := interceptBefore(ctx, mw, call)
		if err != nil {
			return llm.ToolResult{}, err
		}
		if short != nil {
			result := *short
			afterLLM := result.AsLLMResult(use.ID)
			s.rt.EventBus.Publish(ToolResultEvent{
				When:   time.Now(),
				Result: result,
				ToolID: use.ID,
			})
			interceptAfter(ctx, mw, call, result)
			return afterLLM, nil
		}
	}

	tool, err := s.rt.Registry.Lookup(use.Name)
	var result tools.ToolResult
	if err != nil {
		// Unknown tool: synthesize a failure so the model can react.
		result = tools.NewErrorResult(fmt.Sprintf("tool %q is not registered", use.Name))
	} else {
		r, execErr := tool.Execute(ctx, call)
		if execErr != nil {
			if llm.IsAbort(execErr) {
				// Re-cast as an error result but also signal up.
				result = tools.NewErrorResult(fmt.Sprintf("tool %q aborted: %v", use.Name, execErr))
				// Return via ToolResult but mark for upstream.
				// Keep going to emit ToolResult; the abort is surfaced
				// separately via ctx in the caller.
			} else {
				result = tools.NewErrorResult(fmt.Sprintf("tool %q infrastructure error: %v", use.Name, execErr))
			}
		} else {
			result = r
		}
	}

	// Middleware: ToolInterceptor.AfterToolCall. Runs after Execute (or
	// after an unknown-tool synthesized result). Errors are logged but
	// do NOT abort the turn.
	if len(mw.ToolInterceptors) > 0 {
		interceptAfter(ctx, mw, call, result)
	}

	llmResult := result.AsLLMResult(use.ID)
	s.rt.EventBus.Publish(ToolResultEvent{
		When:   time.Now(),
		Result: result,
		ToolID: use.ID,
	})
	return llmResult, nil
}

// Abort cancels the in-flight turn's context. Subscribers (tools,
// streaming consumer) see ctx.Done() and unwind. Abort is a no-op when
// no turn is in flight. Safe to call concurrently with Run.
func (s *AgentSession) Abort(reason string) {
	if s == nil {
		return
	}
	s.cancelMu.Lock()
	fn := s.cancelFn
	s.cancelMu.Unlock()
	if fn != nil {
		fn()
	}
}

// Shutdown emits SessionShutdown, closes the state manager (when the
// runtime owns it), and marks the runtime as terminal. The plugin
// manager is NOT shut down here — per the runtime factory contract, the
// caller coordinates plugin subprocess lifetime.
//
// Store asymmetry: the runtime NEVER calls Close on SessionOptions.Store.
// Unlike StateManager (which has a runtime-created default that Shutdown
// closes when ownsState is true), Store has no default — nil means "no
// store" — so there is nothing for the runtime to close. The embedder
// owns the injected store's lifecycle.
//
// Subsequent Run calls return ErrRuntimeShutdown. Shutdown is idempotent
// and safe to call concurrently with Run or Abort.
func (s *AgentSession) Shutdown(ctx context.Context) error {
	if s == nil || s.rt == nil {
		return nil
	}
	// Cancel any in-flight turn first so the streaming consumer and
	// tool goroutines unwind before we start tearing down resources.
	s.Abort("shutdown")

	s.rt.shutdownMu.Lock()
	if s.rt.shutdownDone {
		s.rt.shutdownMu.Unlock()
		return nil
	}
	s.rt.shutdownDone = true
	s.rt.shutdownReason = "user"
	// Snapshot whether we own the state manager before releasing the
	// lock so Close runs at most once.
	ownsState := s.rt.ownsState
	stateMgr := s.rt.State
	s.rt.shutdownMu.Unlock()

	// Close the state manager if we own it. Caller-injected managers
	// are the caller's responsibility.
	var firstErr error
	if ownsState && stateMgr != nil {
		if err := stateMgr.Close(); err != nil {
			firstErr = err
		}
	}

	// Emit shutdown via the bus; subscribers' channels close as a side
	// effect.
	s.rt.EventBus.Shutdown("user")
	return firstErr
}

// compactionEnabled reports whether compaction should run this turn.
// Returns false only when Settings.Compaction.Enabled is explicitly
// false; missing config (nil) defaults to enabled.
func (s *AgentSession) compactionEnabled() bool {
	c := s.rt.Options.Settings.Compaction
	if c == nil || c.Enabled == nil {
		return true
	}
	return *c.Enabled
}

// setCancelFn stores the turn's cancel function so Abort can invoke it.
// Pass nil to clear at turn exit.
func (s *AgentSession) setCancelFn(fn context.CancelFunc) {
	s.cancelMu.Lock()
	s.cancelFn = fn
	s.cancelMu.Unlock()
}

// collectToolCalls extracts ToolUse blocks from the assistant message
// in content order. Non-ToolUse blocks are ignored.
func collectToolCalls(m llm.Message) []llm.ToolUse {
	var out []llm.ToolUse
	for _, b := range m.Content {
		if tu, ok := b.(llm.ToolUse); ok {
			out = append(out, tu)
		}
	}
	return out
}

// steeringAllIdentifier is the config.SteeringMode value that selects
// concurrent tool execution. Kept as a local constant so the session
// package doesn't reach into config for one string.
const steeringAllIdentifier = "all"
