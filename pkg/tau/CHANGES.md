# pkg/tau CHANGES

This file tracks user-visible changes to the public SDK at
`pkg/tau/`. Every PR that touches `pkg/tau/` MUST update this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/);
versions track the Go module version in `go.mod`.

## Unreleased

### Added

- **Cross-session storage seam** — `Store` interface, `Entry` and
  `Query` types, and three typed sentinels for the cross-session
  context backend.
  - `Store` exposes `Put(ctx, Entry) error`, `Query(ctx, Query)
    ([]Entry, error)`, and `Close() error`. The runtime accepts any
    `Store` via `Options.Store`; nil disables storage features.
  - `Entry` carries `ID`, `Text`, `Tags`, `Embedding`, `Timestamp`,
    `Source`. ID is the primary key — two Puts with the same ID
    overwrite.
  - `Query` selects by `KeywordQuery` (case-insensitive substring),
    `EmbeddingQuery` (dense vector), `TagsQuery` (AND), `SinceQuery`
    (timestamp), `Limit`. Zero-value fields are ignored; the match is
    the AND of every non-zero field.
  - `NewFileStore(dir)` re-exports the reference file-backed backend.
    Each entry is one markdown file under `dir/<id>.md` with YAML-
    style frontmatter; writes are atomic (staged to `.tmp` then
    renamed under a per-store flock); file mode is `0600`, directory
    mode is `0700`. FileStore does NOT compute embeddings —
    `Query.EmbeddingQuery` returns `ErrUnsupportedQuery`.
  - `AgentSession.Store() Store` inspector returns the store supplied
    at construction, or nil when none was supplied. Mirrors the
    ownership contract of `Tools()` (fresh value, not a pointer into
    internal state).
  - Typed sentinels `ErrStoreClosed`, `ErrStoreReadOnly`,
    `ErrUnsupportedQuery`; all compatible with `errors.Is`.
- **Lifecycle contract — embedder owns the injected store.** The
  runtime NEVER calls `Close` on a store supplied via `Options.Store`.
  Unlike `StateManager` (which has a runtime-created default that the
  runtime closes on `Shutdown`), `Store` has no default — nil means
  "no store" — so there is nothing for the runtime to close. The
  embedder MUST `Close` the store when their process exits. Pinned by
  `TestContractStorageLifecycleNotClosedOnShutdown`.
- **Middleware seam** — three in-process extension hooks on the agent
  turn loop, registered via `Options.Middleware []any`:
  - `RequestMutator` mutates the outgoing `*Request` in place before
    the provider call. Gating: a non-nil error aborts the turn.
  - `ResponseObserver` inspects the completed `(*Request, *Response)`
    pair after the stream returns. Non-aborting: errors are logged via
    the standard `log` package and remaining observers still run.
  - `ToolInterceptor` wraps each tool execution. `BeforeToolCall` may
    short-circuit with a `*ToolResult` (skips `Execute`) or abort with
    an error. `AfterToolCall` observes the result non-aborting.
  - `Response` type alias for `llm.Message` — the accumulated assistant
    message returned by a single round-trip, used as the observer's
    response argument.
  - `ErrUnknownMiddlewareType` sentinel returned by `CreateAgentSession`
    when an `Options.Middleware` element satisfies none of the three
    interfaces; `errors.Is` identity is contract-tested.
- Middleware is in-process only; no gRPC adapter. The runtime takes a
  nil-slice fast path when no middleware is registered (zero interface
  dispatches on the turn loop). The runtime partitions the slice into
  three typed slices preserving registration order within each type;
  an element that satisfies multiple interfaces is invoked at every
  matching intercept point.
- Asymmetric error propagation is contract-tested: gating hooks
  (`RequestMutator`, `ToolInterceptor.BeforeToolCall`) abort the turn;
  observing hooks (`ResponseObserver`, `ToolInterceptor.AfterToolCall`)
  log via `log.Printf` and never abort.
- Contract coverage in `pkg/tau/contract_test.go` plus runtime coverage
  in `internal/agent/middleware_test.go`.

## v1.0.0

Initial public release of the tau Go SDK. The package was promoted
from an internal facade to a frozen v1.0 surface.

### Added

- **Provider registry** — `RegisterProvider`, `LookupProvider`,
  `MustRegisterProvider`, `Providers`, `ProviderFactory`,
  `ProviderOptions`. Built-in `anthropic` and `openai` factories
  self-register by default; disable both with the
  `-tags provider_builtins_off` build tag. Sentinel errors
  `ErrProviderNotFound` and `ErrProviderAlreadyRegistered`.
- **Typed client constructors** — `NewAnthropicClient(AnthropicOptions)`
  and `NewOpenAIClient(OpenAIOptions)` as thin wrappers around
  `LookupProvider`. `AnthropicOptions` carries `APIKey` / `BaseURL` /
  `Headers` / `Transport`.
- **Auth resolution** — `ResolveAuth(provider, explicit, authStore)`
  dispatches by provider name through the explicit → auth.json →
  environment → sigil chain.
- **Convenience constructors** — `DefaultSettings()`, `BuiltinTools()`
  (returns the seven built-in tools), `NewFauxProvider(responses...)`
  (deterministic offline LLMClient), `NewInMemoryManager(cwd)`.
- **State manager injection** — `Options.StateManager` field; when
  non-nil, the runtime uses it as-is and does NOT close it on
  Shutdown. When nil, the runtime creates and owns the default
  manager.
- **Event bus topic subscription** — `AgentSession.SubscribeTopics(
  topics ...Topic) <-chan SessionEvent` filters the event stream to
  the specified topics. Nine `TopicXxx` constants re-exported
  (`TopicSessionStart` ... `TopicSessionShutdown`).
- **Tool inventory inspector** — `AgentSession.Tools() []string`
  returns the registered tool names, sorted, as a fresh slice copy.
- **Slash command registry surface** — `Options.SlashCommands`
  injects a custom `*Registry`; `AgentSession.SlashCommands() []string`
  returns the registered names, sorted, no leading `/`. `NewRegistry`
  and `DefaultSlashRegistry` constructors re-exported.
- **Tool interface split** — `HeadlessTool` (functional subset:
  Name/Description/Parameters/Execute) separated from `Tool` (embeds
  `HeadlessTool`, adds `RenderCall`/`RenderResult`). `NoRender` mixin
  satisfies `Tool` with no-op render methods. `Options.Tools` and the
  registry widened to `[]HeadlessTool`. Fully source-compatible:
  every existing `Tool` still satisfies `HeadlessTool`.
- **Type aliases** — 34 alias declarations covering LLM types
  (`Message`, `ContentBlock`, `ToolCall`, `ToolResult`, `Request`,
  `Delta`, `TextDelta`, `ThinkingDelta`, `ToolCallDelta`, `UsageDelta`,
  `Final`, `StopReason` plus the five `StopReasonXxx` constants,
  `Usage`, `Cost`, `ToolUse`, `ToolResult`, `LLMClient`), tool types
  (`Tool`, `HeadlessTool`, `NoRender`, `Theme`, `ToolCall`,
  `ToolResult`, `Registry`), config types (`Settings`, `ModelAPI`,
  `AuthStore`), plugins (`Manager`), and 11 event types/topics
  (`SessionEvent`, `Topic`, the nine `Topic*` constants, the eight
  event structs, `MessageUpdateEvent` with its `Delta` payload).
- **Tool result constructors** — `NewTextResult(text)` and
  `NewErrorResult(text)` re-exported from `internal/tools` so
  embedders implementing custom `HeadlessTool`s do not have to
  assemble `[]ContentBlock{TextContent{...}}` by hand.
- **Sentinel errors** — `ErrRuntimeShutdown`, `ErrUnknownTool`,
  `ErrToolAlreadyRegistered`, `ErrUnknownCommand`, `ErrNotASlashCommand`,
  `ErrQuitRequested`, `ErrClearViewport`, `ErrShowTree`.
- **Contract test** — `pkg/tau/contract_test.go` is the copyable
  contract pattern (precedent: `hashicorp/go-plugin`). Embedders copy
  the file into their own package to pin the API surface they target;
  the file documents its own substitution points.
- **Examples** — `examples/sdk-embed/` is a minimal construct →
  subscribe → run → shutdown program. `examples/sdk-custom-provider/`
  registers a custom LLM provider via the registry and runs a single
  turn. Both import only `github.com/coevin/tau/pkg/tau`.
- **Documentation** — `pkg/tau/doc.go` rewritten as a complete
  package overview with sections for Lifecycle, Concurrency, Errors,
  Versioning. `docs/sdk/cookbook.md` covers six runnable patterns
  (custom tool, custom provider, headless batch mode, multi-session
  fan-out, in-memory state, custom slash commands).

### Changed

- `Options.Tools` widened from `[]Tool` to `[]HeadlessTool`.
  Source-compatible: every `Tool` satisfies `HeadlessTool`.
- `Options.SlashCommands` is a new `*Registry` field. Existing
  callers that omit it get the default built-in command set.

### Deviations from the original spec

- **`provider_builtins` build tag**: the spec prose called for
  `-tags provider_builtins=off` to disable built-in factories. Go's
  boolean build-tag semantics do not allow that pattern (positive
  tags are NOT default-on). Implemented as
  `-tags provider_builtins_off` instead — built-ins register by
  default; opt out with the negative tag.
