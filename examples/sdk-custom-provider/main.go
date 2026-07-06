// sdk-custom-provider demonstrates registering a custom LLM provider with
// the tau SDK and running a session against it. The provider is a fake
// "Acme internal gateway" that emits a canned response without any
// network I/O — the same pattern an embedder would use to wire the agent
// loop to an in-house model serving infrastructure.
//
// Run from the tau module directory:
//
//	go run ./examples/sdk-custom-provider
//
// The example does NOT import any internal/ package; everything is wired
// through the public SDK surface at github.com/taucentral/tau/pkg/tau.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/taucentral/tau/pkg/tau"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// Register a custom provider named "acme" with the SDK's provider
	// registry. factories are invoked once per NewAnthropicClient /
	// LookupProvider call, so cheap construction is fine.
	if err := tau.RegisterProvider("acme", func(opts tau.ProviderOptions) (tau.LLMClient, error) {
		if opts.APIKey == "" {
			return nil, errors.New("acme: APIKey is required")
		}
		return &acmeClient{apiKey: opts.APIKey, response: "hello from the acme gateway"}, nil
	}); err != nil {
		log.Fatalf("register acme: %v", err)
	}

	// Resolve the provider through the registry. An embedder would
	// typically pass the resulting LLMClient straight to
	// CreateAgentSession.
	factory, err := tau.LookupProvider("acme")
	if err != nil {
		log.Fatalf("lookup acme: %v", err)
	}
	client, err := factory(tau.ProviderOptions{APIKey: "demo-key"})
	if err != nil {
		log.Fatalf("construct acme client: %v", err)
	}

	sess, err := tau.CreateAgentSession(ctx, tau.Options{
		Cwd:           ".",
		Model:         "acme-1",
		LLMClient:     client,
		Tools:         tau.BuiltinTools(),
		Settings:      tau.DefaultSettings(),
		StateManager:  tau.NewInMemoryManager("."),
		ContextWindow: 200000,
	})
	if err != nil {
		log.Fatalf("create session: %v", err)
	}
	defer sess.Shutdown(ctx)

	// Drain every lifecycle + content event the session emits. Subscribe
	// before Run so the channel does not miss the first event.
	events := sess.Subscribe()
	eventsDone := make(chan struct{})
	go func() {
		defer close(eventsDone)
		for evt := range events {
			fmt.Printf("event %T %v\n", evt, evt.Topic())
		}
	}()

	if err := sess.Run(ctx, "Say hello."); err != nil {
		log.Fatalf("run: %v", err)
	}
	if err := sess.Shutdown(ctx); err != nil {
		log.Printf("shutdown: %v", err)
	}
	<-eventsDone
	fmt.Println("done")
}

// acmeClient is a fake LLMClient backed by an "internal Acme gateway".
// In a real embedder this would hold an *http.Client, base URL, and
// auth material. Here it returns one canned TextDelta + Final.
type acmeClient struct {
	apiKey   string
	response string
}

// Stream implements tau.LLMClient (= llm.LLMClient). It emits a single
// TextDelta carrying the canned response, then a Final marker with
// StopReasonEndTurn. The agent loop assembles these into an assistant
// message. ctx cancellation is respected: if the context is already
// done we return its error; if it cancels mid-stream we exit early.
func (c *acmeClient) Stream(ctx context.Context, req tau.Request) (<-chan tau.Delta, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	ch := make(chan tau.Delta, 2)
	go func() {
		defer close(ch)
		select {
		case <-ctx.Done():
			return
		case ch <- tau.TextDelta{Text: c.response}:
		}
		select {
		case <-ctx.Done():
			return
		case ch <- tau.Final{StopReason: tau.StopReasonEndTurn}:
		}
	}()
	return ch, nil
}
