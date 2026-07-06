// sdk-embed is a minimal program that embeds the tau agent via the public
// SDK. It wires the built-in faux provider (no network access), constructs
// an AgentSession with the built-in tool set, prints every event the
// session emits, and runs a single turn.
//
// Run from the tau module directory:
//
//	go run ./examples/sdk-embed
//
// To talk to a real model, replace tau.NewFauxProvider(...) with
// tau.NewAnthropicClient(tau.AnthropicOptions{APIKey: os.Getenv("ANTHROPIC_API_KEY")})
// or tau.NewOpenAIClient(tau.OpenAIOptions{APIKey: os.Getenv("OPENAI_API_KEY")}).
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/taucentral/tau/pkg/tau"
)

func main() {
	// Surface Ctrl+C as a clean shutdown so the faux session's state
	// manager closes gracefully instead of being torn down mid-flush.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// tau.NewFauxProvider returns a deterministic LLMClient for offline
	// runs. Swap for NewAnthropicClient / NewOpenAIClient to hit a real
	// model; everything else in the program is unchanged.
	client := tau.NewFauxProvider("hello from the faux model")

	sess, err := tau.CreateAgentSession(ctx, tau.Options{
		Cwd:           ".",
		Model:         "faux",
		LLMClient:     client,
		Tools:         tau.BuiltinTools(),
		Settings:      tau.DefaultSettings(),
		StateManager:  tau.NewInMemoryManager("."), // do not persist across runs
		ContextWindow: 200000,
	})
	if err != nil {
		log.Fatalf("create session: %v", err)
	}
	defer sess.Shutdown(ctx)

	// Subscribe BEFORE launching the drain goroutine so Run can't race
	// ahead and drop early events. The faux provider streams one
	// TextDelta then a Final, so the expected sequence is:
	// session_start, turn_start, message_start, message_update (text),
	// message_end, turn_end.
	events := sess.Subscribe()
	eventsDone := make(chan struct{})
	go func() {
		defer close(eventsDone)
		for evt := range events {
			describe(evt)
		}
	}()

	if err := sess.Run(ctx, "Say hello."); err != nil {
		log.Fatalf("run: %v", err)
	}

	// Shutdown closes the bus, which closes subscriber channels, which
	// lets the drain goroutine exit.
	if err := sess.Shutdown(ctx); err != nil {
		log.Printf("shutdown: %v", err)
	}
	<-eventsDone
	fmt.Println("done")
}

// describe prints a one-line summary per event. The type-switch covers
// the canonical lifecycle; events that don't match are logged generically.
func describe(evt tau.SessionEvent) {
	switch e := evt.(type) {
	case tau.SessionStartEvent:
		fmt.Printf("session_start model=%s cwd=%s\n", e.Model, e.Cwd)
	case tau.TurnStartEvent:
		fmt.Printf("turn_start turn=%d\n", e.Turn)
	case tau.MessageStartEvent:
		fmt.Println("message_start")
	case tau.MessageUpdateEvent:
		fmt.Printf("message_update delta=%T\n", e.Delta)
	case tau.ToolCallEvent:
		fmt.Printf("tool_call name=%s\n", e.Call.Name)
	case tau.ToolResultEvent:
		fmt.Printf("tool_result id=%s\n", e.ToolID)
	case tau.MessageEndEvent:
		fmt.Printf("message_end stop=%s\n", e.StopReason)
	case tau.TurnEndEvent:
		fmt.Printf("turn_end turn=%d finished=%t\n", e.Turn, e.Finished)
	case tau.SessionShutdownEvent:
		fmt.Printf("session_shutdown reason=%s\n", e.Reason)
	default:
		fmt.Printf("event %T\n", evt)
	}
}
