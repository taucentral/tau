# sdk-embed

A minimal program that embeds the tau agent via the public SDK
(`github.com/coevin/tau/pkg/tau`). It wires the built-in faux provider
(no network access), constructs an `AgentSession` with the built-in tool
set, prints every event the session emits, and runs a single turn.

## Run (offline, faux provider)

From the tau module directory:

```sh
go run ./examples/sdk-embed
```

Expected output (abridged):

```
session_start model=faux cwd=.
turn_start turn=1
message_start
message_update delta=*llm.TextDelta
message_end stop=end_turn
turn_end turn=1 finished=true
done
```

No network I/O occurs. The faux provider streams a fixed response
(`"hello from the faux model"`) and the session shuts down cleanly.

## Run against a real model

Edit `examples/sdk-embed/main.go` and replace the faux client:

```go
client, err := tau.NewAnthropicClient(tau.AnthropicOptions{
    APIKey: os.Getenv("ANTHROPIC_API_KEY"),
})
if err != nil {
    log.Fatalf("anthropic client: %v", err)
}
```

or, for OpenAI:

```go
client, err := tau.NewOpenAIClient(tau.OpenAIOptions{
    APIKey: os.Getenv("OPENAI_API_KEY"),
})
```

Then run with the key exported:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
go run ./examples/sdk-embed
```

Everything else in the program — the session construction, event
subscription, turn execution, shutdown — is unchanged.

## What this example demonstrates

- Importing ONLY the public SDK (`pkg/tau`); zero `internal/` imports.
- Constructing an `*AgentSession` via `tau.CreateAgentSession`.
- Subscribing to the full event stream via `sess.Subscribe()`.
- Running a single turn via `sess.Run(ctx, prompt)`.
- Clean shutdown via `sess.Shutdown(ctx)` (idempotent; closes the event bus).
- Using `tau.NewInMemoryManager` to avoid persisting state across runs.

## Embedding checklist

When embedding tau in your own program:

1. Construct an `LLMClient` (faux, Anthropic, OpenAI, or a custom
   provider registered via `tau.RegisterProvider`).
2. Call `tau.CreateAgentSession(ctx, tau.Options{...})` with `Cwd`,
   `Model`, `LLMClient`, `Tools`, `Settings`, and optionally
   `StateManager`.
3. Subscribe to events BEFORE calling `Run` so early events are not
   missed.
4. Drain the event channel in a goroutine; the channel is closed when
   the session shuts down.
5. Call `sess.Shutdown(ctx)` when done. It is idempotent.
