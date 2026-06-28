// Package tau is the public Go SDK for embedding the tau agentic coding
// agent in a Go program. The entry point is CreateAgentSession; an
// AgentSession runs agentic turns against an LLMClient, dispatches tool
// calls to a registry of HeadlessTools, and publishes lifecycle events
// on a bus that callers subscribe to.
//
// Everything in this package is implemented by re-exporting internal
// types as aliases — there is exactly one concrete implementation of
// each abstraction (the one used by the tau CLI). Callers can mix
// values from pkg/tau with their own helpers freely.
//
// # Minimal example
//
// Construct a session, subscribe to the full event stream, run one
// turn, drain the events, then shut down. The faux provider makes the
// example runnable without network I/O.
//
//	package main
//
//	import (
//		"context"
//		"fmt"
//		"log"
//
//		"github.com/coevin/tau/pkg/tau"
//	)
//
//	func main() {
//		ctx := context.Background()
//		sess, err := tau.CreateAgentSession(ctx, tau.Options{
//			Cwd:           ".",
//			Model:         "faux",
//			LLMClient:     tau.NewFauxProvider("hello from the model"),
//			Tools:         tau.BuiltinTools(),
//			Settings:      tau.DefaultSettings(),
//			StateManager:  tau.NewInMemoryManager("."),
//			ContextWindow: 200000,
//		})
//		if err != nil {
//			log.Fatalf("create session: %v", err)
//		}
//		defer sess.Shutdown(ctx)
//
//		go func() {
//			for evt := range sess.Subscribe() {
//				fmt.Printf("event %T\n", evt)
//			}
//		}()
//
//		if err := sess.Run(ctx, "Say hello."); err != nil {
//			log.Fatalf("run: %v", err)
//		}
//	}
//
// Swap tau.NewFauxProvider for tau.NewAnthropicClient or
// tau.NewOpenAIClient to hit a real model; everything else is
// unchanged.
//
// # Lifecycle
//
// A session moves through: Construct (CreateAgentSession) → zero or
// more Run turns → Shutdown. Run blocks until the turn completes; the
// caller drives multiple turns by calling Run in sequence. Shutdown is
// idempotent and closes the event bus; once it returns, any subsequent
// Run returns ErrRuntimeShutdown.
//
// Sessions publish nine TopicXxx event classes (session_start,
// turn_start, message_start, message_update, tool_call, tool_result,
// message_end, turn_end, session_shutdown). Subscribe returns every
// event; SubscribeTopics filters to the topics an embedder cares
// about (for example, only tool_call + tool_result to drive a CI bot).
//
// # Concurrency
//
// A session is single-threaded for Run: at most one turn may be in
// flight at a time. Calling Run while a turn is already running
// panics. Abort and Shutdown are safe to call concurrently with Run.
// Events arrive on subscriber channels in emission order; an embedder
// that needs deterministic ordering should drain from a single
// goroutine per channel.
//
// # Errors
//
// The SDK surfaces typed sentinel errors usable with errors.Is:
//
//   - ErrRuntimeShutdown        — Run called after Shutdown.
//   - ErrProviderNotFound       — LookupProvider or typed client
//     constructor against an unregistered provider name.
//   - ErrProviderAlreadyRegistered — duplicate RegisterProvider call.
//   - ErrUnknownTool            — tool dispatch against an unregistered
//     name (re-exported from internal/tools).
//   - ErrToolAlreadyRegistered  — duplicate tool registration
//     (re-exported from internal/tools).
//   - ErrUnknownCommand         — slash dispatch against an
//     unregistered command (re-exported from internal/slash).
//   - ErrUnknownMiddlewareType  — CreateAgentSession rejected an
//     Options.Middleware element that satisfied none of RequestMutator,
//     ResponseObserver, or ToolInterceptor.
//
// Provider Stream errors arrive as Final.Err on the event bus, not as
// Run return values: the agent loop converts them into an
// assistant-side error message and terminates the turn.
//
// # Middleware
//
// Three extension hooks on the turn loop, registered via Options.Middleware:
//
//   - RequestMutator     — mutate the outgoing *Request in place before
//                          it reaches the provider. Gating: a non-nil
//                          error aborts the turn.
//   - ResponseObserver   — observe the (request, response) pair after
//                          the stream completes. Non-aborting: errors
//                          are logged via the standard log package.
//   - ToolInterceptor    — gate each tool call (BeforeToolCall may
//                          short-circuit with a *ToolResult or abort
//                          with an error) and observe each result
//                          (AfterToolCall is non-aborting).
//
// Middleware is in-process only; there is no gRPC adapter. The runtime
// invokes each hook in registration order within its type. An empty
// (or nil) Options.Middleware slice is the fast path: zero interface
// dispatches on the turn loop.
//
// See docs/input/context/plugin-support/whitepaper.md §3.2 for the
// design rationale and docs/sdk/cookbook.md for runnable patterns.
//
// # Versioning
//
// The SDK follows the Go module version drawn from go.mod. The v1.0
// surface is frozen by pkg/tau/contract_test.go — that test is the
// copyable contract pattern (precedent: hashicorp/go-plugin) that any
// embedder can drop into their own package to pin the API surface they
// target. The policy is: every PR that touches pkg/tau/ must update
// pkg/tau/CHANGES.md.
//
// See docs/sdk/cookbook.md for runnable patterns (custom tool, custom
// LLM provider via the registry, headless batch mode, multi-session
// fan-out, in-memory state for unit tests, custom slash command
// registration).
package tau
