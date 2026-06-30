// Package tau is the public SDK for embedding tau's agentic coding agent
// in Go programs. It provides a thin facade over the internal agent
// runtime: a single CreateAgentSession constructor, an Options bundle
// for the handful of inputs callers need, and type aliases that
// re-export the value types consumers interact with (Message, ContentBlock,
// ToolCall, ToolResult, SessionEvent, and friends).
//
// The SDK is intentionally minimal. It does not try to hide every internal
// behind an interface — instead, the type aliases keep the SDK and the
// internal types interchangeable so callers can pass an SDK Tool to an
// internal API (and vice versa) without conversion boilerplate.
//
// See doc.go for a usage example.
package tau

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/coevin/tau/internal/agent"
	"github.com/coevin/tau/internal/config"
	"github.com/coevin/tau/internal/fauxprovider"
	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/llm/provider/anthropic"
	"github.com/coevin/tau/internal/llm/provider/openai"
	"github.com/coevin/tau/internal/orchestrator"
	"github.com/coevin/tau/internal/plugins"
	"github.com/coevin/tau/internal/slash"
	"github.com/coevin/tau/internal/storage"
	"github.com/coevin/tau/internal/state"
	"github.com/coevin/tau/internal/tools"
)

// --- Re-exported types (type aliases) ---------------------------------------
//
// Type aliases keep the SDK and internal types identical. Callers may pass
// an SDK Tool to an internal function or type-switch on a SessionEvent
// against concrete types defined in internal/ — the names resolve to the
// same underlying symbols.

// Role identifies the speaker of a Message (system, user, assistant, tool).
type Role = llm.Role

// Message is a single chat message exchanged with the model: a Role plus
// an ordered list of ContentBlocks.
type Message = llm.Message

// ContentBlock is a sealed interface for one block of message content.
// Implementations: TextContent, ThinkingContent, ImageContent, ToolUse,
// ToolResult. Type-switch on the underlying concrete type to render.
type ContentBlock = llm.ContentBlock

// TextContent is plain text content.
type TextContent = llm.TextContent

// ThinkingContent is an extended-thinking (reasoning) block.
type ThinkingContent = llm.ThinkingContent

// ImageContent is a base64-encoded image block.
type ImageContent = llm.ImageContent

// ThinkingDelta carries an incremental chunk of reasoning text from the
// model stream. Emitted by providers when the model is in extended-thinking
// mode; subscribers render it inline with TextDelta.
type ThinkingDelta = llm.ThinkingDelta

// ToolUse is the model's request to invoke a tool. The agent loop reads
// this from the assistant message and dispatches it to the Registry.
type ToolUse = llm.ToolUse

// StopReason is why the model stopped generating this turn (stop, length,
// toolUse, error, aborted).
type StopReason = llm.StopReason

// StopReason constants re-exported so embedders do not have to use string
// literals when constructing a Final delta or asserting on a message's
// stop reason.
const (
	StopReasonEndTurn = llm.StopReasonEndTurn
	StopReasonLength  = llm.StopReasonLength
	StopReasonToolUse = llm.StopReasonToolUse
	StopReasonError   = llm.StopReasonError
	StopReasonAborted = llm.StopReasonAborted
)

// Usage reports per-turn token accounting.
type Usage = llm.Usage

// Request is the streaming request handed to an LLMClient. The agent
// loop builds one per dispatch from the current state tree.
type Request = llm.Request

// Response is the completed assistant message returned by an LLM
// provider for a single round-trip. The agent loop accumulates deltas
// from LLMClient.Stream into an assistant Message; that Message —
// complete with Role, Content blocks, StopReason, Usage, and any
// provider IDs — IS the response. ResponseObserver middleware receive
// a *Response pointing at this value after Stream returns.
//
// tau aliases Message rather than introducing a separate Response
// struct because the two concepts are identical: the model's
// "response" is the assistant message it produced for this turn.
type Response = llm.Message

// Delta is the sealed interface for incremental model-stream events.
// Implementations: TextDelta, ToolCallDelta, UsageDelta, Final.
type Delta = llm.Delta

// TextDelta carries an incremental chunk of assistant text from the model
// stream. Subscribers receive these as part of MessageUpdateEvent.
type TextDelta = llm.TextDelta

// ToolCallDelta carries an incremental tool-call argument fragment.
type ToolCallDelta = llm.ToolCallDelta

// UsageDelta carries an incremental token-usage update mid-stream.
type UsageDelta = llm.UsageDelta

// Final is the terminal delta that closes a stream. Its StopReason drives
// the agent-loop dispatch (tool_use continues the loop; anything else
// ends the turn).
type Final = llm.Final

// LLMClient is the streaming model client interface implemented by every
// provider (Anthropic, OpenAI, faux for tests).
type LLMClient = llm.LLMClient

// Tool is the single abstraction used by both built-in and plugin-exposed
// tools. It embeds HeadlessTool and adds the two TUI rendering hooks
// (RenderCall, RenderResult). Implementations MUST be safe for concurrent
// use.
type Tool = tools.Tool

// HeadlessTool is the functional subset of Tool: the methods an embedder
// needs to plug a tool into the agent loop without implementing TUI
// rendering. Headless embedders (batch evaluators, CI bots, daemons)
// implement only HeadlessTool; the TUI falls back to a generic
// representation when a tool does not satisfy Tool.
type HeadlessTool = tools.HeadlessTool

// NoRender is the embedded mixin that satisfies Tool's two render methods
// with no-op stubs. Embed it in a headless tool that you also want to pass
// to APIs that require the full Tool interface (e.g., the TUI's render
// path). The TUI's generic fallback is used when RenderCall/RenderResult
// return the empty string, which is what NoRender yields.
type NoRender = tools.NoRender

// Theme is the styling bundle passed to Tool.RenderCall and
// Tool.RenderResult. It carries ANSI SGR escape sequences for the TUI and
// empty strings for non-TTY contexts (tests, JSON mode, redirected output).
// Construct via PlainTheme() (no styling) or ColorTheme() (basic dark
// palette); the TUI's runtime Theme converts to this shape when invoking a
// Tool's render methods.
type Theme = tools.Theme

// ToolCall is the runtime invocation of a tool. The agent loop constructs
// this from a provider-side ToolUse block.
type ToolCall = tools.ToolCall

// ToolResult is the output of a tool execution. The agent loop wraps this
// in an llm.ToolResult block for the LLM context.
type ToolResult = tools.ToolResult

// NewTextResult is a convenience constructor for a successful
// single-block text ToolResult. Re-exported from internal/tools so
// embedders implementing a custom HeadlessTool do not have to assemble
// []ContentBlock{TextContent{...}} by hand.
func NewTextResult(text string) ToolResult { return tools.NewTextResult(text) }

// NewErrorResult is a convenience constructor for an error ToolResult
// (Content=text, IsError=true). Re-exported from internal/tools.
func NewErrorResult(text string) ToolResult { return tools.NewErrorResult(text) }

// Settings is the fully-resolved settings bundle (global merged with
// project scope). Construct via config.DefaultSettings() or load from
// disk via the internal/config layer.
type Settings = config.Settings

// ThinkingLevel selects the model's reasoning budget. One of the
// config.Thinking* constants.
type ThinkingLevel = config.ThinkingLevel

// TransportSetting selects the streaming transport (auto, sse, websocket).
type TransportSetting = config.TransportSetting

// SteeringMode governs concurrent tool execution (all, one-at-a-time).
type SteeringMode = config.SteeringMode

// ModelAPI is the identifier string embedded in model definitions (e.g.,
// "anthropic", "openai") used to route auth resolution and provider
// selection. It is re-exported so embedders building model-definition
// tooling (e.g., a custom settings UI) can refer to the same type the
// runtime uses.
type ModelAPI = config.ModelAPI

// AuthStore is the file-backed credential storage interface implemented by
// FileAuthStore (production) and InMemoryAuthStore (tests). Re-exported so
// an embedder calling ResolveAuth can construct or load one without
// reaching into internal/config.
type AuthStore = config.AuthStore

// PluginManager loads and exposes plugin-provided tools over gRPC.
// Callers that want plugin tools in their session construct one, call
// SpawnAll, and pass it via Options.Plugins.
type PluginManager = plugins.Manager

// --- Slash commands ---------------------------------------------------------
//
// tau's slash-command surface is re-exported here so embedders can add,
// list, or replace commands without reaching into internal/slash. The
// runtime does NOT dispatch commands from the registry itself — embedder
// UIs hold the same *Registry reference and dispatch in their event loop.
//
// See openspec/specs/sdk-public-api/spec.md "Slash command interface".

// Command is the interface every slash command implements. Execute
// receives the raw argument string (after the command name, whitespace
// trimmed) and the wired session. The returned string is rendered to the
// user; a non-nil error surfaces as diagnostic text.
type Command = slash.Command

// CommandSession is the session surface every Command.Execute receives.
// It is the public, method-based view over the wired agent session;
// external Go modules implement tau.Command without importing internal/.
// The runtime constructs the view via AgentSession.AsCommandSession().
type CommandSession = agent.CommandSession

// CommandRuntime is the runtime surface exposed through CommandSession.
// Runtime(). It carries the state manager, options, compactor, and event
// bus the command operates on.
type CommandRuntime = agent.CommandRuntime

// CommandOptions is the options surface exposed through CommandRuntime.
// Options(). Reads (Model, ProviderAPI, ...) are method-based; the
// SetModel setter is the supported way to mutate the active model.
type CommandOptions = agent.CommandOptions

// Registry maps slash-command names (with leading slash) to Command
// implementations. The zero value is not usable; construct via
// NewRegistry (empty) or DefaultSlashRegistry (built-ins pre-registered).
type Registry = slash.Registry

// NewRegistry returns an empty slash-command registry. Embedders add
// their own commands via Register before wiring the registry into their
// UI dispatch. To get the built-in command set pre-registered, call
// DefaultSlashRegistry instead.
func NewRegistry() *Registry { return slash.NewRegistry() }

// DefaultSlashRegistry returns a Registry pre-populated with every
// built-in command (/fork, /checkout, /tree, /compact, /model, /label,
// /clear, /help, /quit). Embedders wanting to selectively disable
// built-ins should call NewRegistry() and Register each desired command
// themselves.
func DefaultSlashRegistry() *Registry { return slash.DefaultRegistry() }

// ErrUnknownCommand is returned by Registry.Execute when the parsed
// command name is not registered.
var ErrUnknownCommand = slash.ErrUnknownCommand

// ErrNotASlashCommand is returned by Registry.Execute when the input
// does not parse as a slash command (i.e., does not start with "/").
var ErrNotASlashCommand = slash.ErrNotASlashCommand

// ErrQuitRequested is returned by /quit to signal the caller should
// close the program's quit channel.
var ErrQuitRequested = slash.ErrQuitRequested

// ErrClearViewport is returned by /clear to signal the caller should
// clear the conversational viewport.
var ErrClearViewport = slash.ErrClearViewport

// ErrShowTree is returned by /tree to signal the caller should open the
// tree-view overlay.
var ErrShowTree = slash.ErrShowTree

// StateManager is the persistence and tree-management interface implemented
// by the runtime's state backends. The SDK accepts a caller-supplied
// StateManager via Options.StateManager; when nil, the runtime creates a
// lazy-flushed bbolt-backed manager rooted at Options.Cwd. When non-nil,
// the runtime uses the injected manager verbatim and does NOT call Close
// on it — the embedder owns the injected manager's lifecycle.
type StateManager = state.Manager

// --- Session events ---------------------------------------------------------

// Topic is the routing key for an Event. Subscribers select zero or more
// topics when subscribing via SubscribeTopics; the bus delivers matching
// events only. Use the Topic* constants below.
type Topic = agent.Topic

// Topic* constants enumerate the canonical event topics emitted over a
// session's lifetime. Pass any subset to SubscribeTopics to receive only
// those events; subscribe to all topics via Subscribe (no arguments).
const (
	TopicSessionStart    = agent.TopicSessionStart
	TopicTurnStart       = agent.TopicTurnStart
	TopicMessageStart    = agent.TopicMessageStart
	TopicMessageUpdate   = agent.TopicMessageUpdate
	TopicToolCall        = agent.TopicToolCall
	TopicToolResult      = agent.TopicToolResult
	TopicMessageEnd      = agent.TopicMessageEnd
	TopicTurnEnd         = agent.TopicTurnEnd
	TopicSessionShutdown = agent.TopicSessionShutdown
)

// SessionEvent is the sealed interface implemented by every typed event
// published on the session bus. Type-switch on the concrete aliases
// below (SessionStartEvent, TurnStartEvent, etc.) to handle each kind.
type SessionEvent = agent.Event

// SessionStartEvent fires once per session, immediately before the first
// turn. Subscribers use it to bootstrap UI state or log sinks.
type SessionStartEvent = agent.SessionStartEvent

// TurnStartEvent opens a turn: the agent has accepted user input and is
// about to dispatch to the model.
type TurnStartEvent = agent.TurnStartEvent

// MessageStartEvent opens an assistant message: the model began streaming.
type MessageStartEvent = agent.MessageStartEvent

// MessageUpdateEvent carries an incremental delta from the model stream.
// Rendered live by subscribers. The Delta field is the raw llm.Delta;
// type-switch to render (TextDelta, ToolCallDelta, UsageDelta, Final).
type MessageUpdateEvent = agent.MessageUpdateEvent

// ToolCallEvent is emitted when the agent invokes a tool. Always paired
// with a ToolResultEvent for the same tool-use ID.
type ToolCallEvent = agent.ToolCallEvent

// ToolResultEvent carries the tool's output. Emitted after the matching
// ToolCallEvent.
type ToolResultEvent = agent.ToolResultEvent

// MessageEndEvent closes the assistant message: the model returned Final
// AND any tool execution triggered by this message has completed.
type MessageEndEvent = agent.MessageEndEvent

// TurnEndEvent closes a turn: the assistant's response (and any tool
// round-trips) are persisted to the state tree.
type TurnEndEvent = agent.TurnEndEvent

// SessionShutdownEvent is emitted exactly once when the session is
// shutting down. After this event, subscriber channels close.
type SessionShutdownEvent = agent.SessionShutdownEvent

// ErrRuntimeShutdown is returned by AgentSession.Run when called after
// Shutdown has been invoked. The runtime is one-shot: callers must build
// a new session to serve further turns.
var ErrRuntimeShutdown = agent.ErrRuntimeShutdown

// ErrMergeStateForeignChild is returned by CommandSession.MergeState when
// the child argument is not a CommandSession produced by AsCommandSession
// or Spawn. Plugin authors hit this when they wrap the CommandSession
// returned by Spawn in their own type before passing it back to
// MergeState; the wrapper must be removed so MergeState can reach the
// concrete session underneath. Detect via errors.Is.
var ErrMergeStateForeignChild = agent.ErrMergeStateForeignChild

// ErrUnknownTool is returned by the tool registry's Lookup when no tool is
// registered under the requested name. The agent loop catches this and
// synthesizes a ToolResult with IsError=true so the model sees a clear
// error rather than a generic "internal error".
var ErrUnknownTool = tools.ErrUnknownTool

// ErrToolAlreadyRegistered is returned by the tool registry's Register
// when another tool is already registered under the same name. First
// registration wins so plugin-load order is deterministic.
//
// tau aliases the internal ErrDuplicateTool to this name to match the
// SDK-side naming convention used for ErrProviderAlreadyRegistered.
var ErrToolAlreadyRegistered = tools.ErrDuplicateTool

// --- Options ---------------------------------------------------------------

// Options is the input bundle to CreateAgentSession. Required fields are
// Model, LLMClient, and at least one Tool. Optional fields defer to
// Settings or sensible defaults when left zero.
type Options struct {
	// Cwd is the absolute working directory the session operates in.
	// Required. Tools that touch the filesystem resolve relative paths
	// against this. Pass "" to use the process's current directory.
	Cwd string

	// Model is the model identifier to send to the provider (e.g.
	// "claude-opus-4-5-20251101"). Required.
	Model string

	// LLMClient is the streaming model client. Required. Production code
	// wires a real provider; tests inject a faux provider.
	LLMClient LLMClient

	// Tools is the built-in tool set the runtime registers. Must contain
	// at least one entry. Accepts HeadlessTool so a headless embedder
	// can register tools that do not implement the TUI rendering methods;
	// every Tool also satisfies HeadlessTool. Plugin tools (when Plugins
	// is non-nil) merge on top.
	Tools []HeadlessTool

	// Settings is the fully-resolved settings bundle. Required. The agent
	// never reads disk for settings — the caller loads them.
	Settings Settings

	// ThinkingLevel selects the reasoning budget. Zero defers to
	// Settings.DefaultThinkingLevel.
	ThinkingLevel ThinkingLevel

	// Transport selects the streaming transport. Zero defers to
	// Settings.Transport.
	Transport TransportSetting

	// SteeringMode governs concurrent tool execution. Zero defers to
	// Settings.SteeringMode.
	SteeringMode SteeringMode

	// Plugins, when non-nil, exposes plugin-provided tools to the runtime.
	// The caller is responsible for SpawnAll and Shutdown coordination;
	// the SDK does not manage plugin subprocess lifetimes.
	Plugins *PluginManager

	// StateManager, when non-nil, overrides the runtime's default
	// state manager. Use NewInMemoryManager(cwd) for tests and ephemeral
	// sessions that should not write to disk.
	//
	// Lifecycle contract:
	//   - If nil (the default), the runtime creates a lazy-flushed bbolt
	//     manager rooted at Cwd and OWNS its lifecycle (Close is called
	//     on Shutdown).
	//   - If non-nil, the runtime uses the injected manager as-is and
	//     does NOT call Close on it — the embedder owns the injected
	//     manager's lifecycle. This lets an embedder share one manager
	//     across many sessions.
	StateManager StateManager

	// ContextWindow is the model's context window in tokens, used by the
	// compactor. Zero lets the runtime apply its default (200 000).
	ContextWindow int

	// ConfigDir overrides the global config directory used to locate
	// prompts (<ConfigDir>/agent/prompts, <ConfigDir>/SYSTEM.md). Empty
	// uses standard resolution (TAU_CONFIG_DIR, XDG_CONFIG_HOME, default).
	ConfigDir string

	// SessionID, when non-empty, requests resume from an existing
	// session at the configured sessions path. Empty starts fresh.
	SessionID string

	// SlashCommands, when non-nil, is stored on the runtime for SDK-level
	// inspection (AgentSession.SlashCommands). The runtime does NOT
	// dispatch commands from this registry; embedder UIs hold the same
	// reference and dispatch themselves. nil means SlashCommands() falls
	// back to DefaultSlashRegistry().
	SlashCommands *Registry

	// Middleware, when non-empty, registers in-process hooks on the
	// agent turn loop. Each element SHALL satisfy at least one of the
	// RequestMutator, ResponseObserver, or ToolInterceptor interfaces;
	// CreateAgentSession type-checks every element and returns
	// ErrUnknownMiddlewareType for any element that satisfies none.
	// The runtime partitions the slice into three typed slices
	// (preserving registration order within each) and invokes each
	// middleware at its documented intercept point during Run. nil or
	// an empty slice disables middleware — the runtime takes a fast
	// path with no observable overhead beyond a nil-slice check.
	Middleware []any

	// Store, when non-nil, exposes a cross-session context backend to
	// the runtime. The runtime does NOT auto-inject retrieved entries
	// into the request today; embedders retrieve via their own
	// RequestMutator middleware (see docs/sdk/cookbook.md recipe (h))
	// and inject the text they want. nil disables storage features.
	//
	// Lifecycle contract:
	//   - The runtime does NOT call Close on a store supplied here.
	//     The embedder owns the injected store's lifecycle. This is
	//     the asymmetry with StateManager: StateManager has a runtime-
	//     created default that the runtime closes on Shutdown; Store
	//     has no default — nil means "no store" — so there is nothing
	//     for the runtime to close.
	//   - Construct via NewFileStore(dir) for the reference file-
	//     backed backend; implement Store directly for vector / sqlite
	//     / external backends.
	Store Store

	// Orchestrator, when non-nil, enables multi-session orchestration
	// on this session. The orchestrator drives Spawn / MergeState
	// workflows in-process; see the Orchestration section in doc.go
	// for the full lifecycle contract. nil disables orchestration:
	// Spawn returns ErrRuntimeShutdown and MergeState returns
	// ErrOrchestratorClosed.
	//
	// Lifecycle contract:
	//   - The runtime NEVER calls any method on an orchestrator
	//     supplied here, including during Shutdown. The embedder owns
	//     the orchestrator's lifecycle.
	//   - Children spawned via AgentSession.Spawn are NOT shut down
	//     automatically by the parent's Shutdown. The embedder
	//     coordinates child lifetimes explicitly.
	//   - Construct via NewSequentialOrchestrator(parent) for the
	//     reference sequential-phase implementation; implement
	//     Orchestrator directly for fan-out / adversarial review.
	Orchestrator Orchestrator
}

// --- Constructor and session handle ----------------------------------------

// CreateAgentSession validates opts, applies defaults, and returns a
// ready-to-Run AgentSession. The caller owns the returned session and
// must call Shutdown to release state-manager resources.
//
// CreateAgentSession does no plugin subprocess management and no network
// I/O. Filesystem access is limited to global config-directory resolution
// when opts.ConfigDir is empty.
func CreateAgentSession(ctx context.Context, opts Options) (*AgentSession, error) {
	mw, err := partitionMiddleware(opts.Middleware)
	if err != nil {
		return nil, err
	}
	so := agent.SessionOptions{
		Model:         opts.Model,
		ThinkingLevel: opts.ThinkingLevel,
		Settings:      opts.Settings,
		Transport:     opts.Transport,
		SteeringMode:  opts.SteeringMode,
		Tools:         opts.Tools,
		Plugins:       opts.Plugins,
		LLMClient:     opts.LLMClient,
		StateManager:  opts.StateManager,
		ContextWindow: opts.ContextWindow,
		ConfigDir:     opts.ConfigDir,
		SessionID:     opts.SessionID,
		SlashCommands: opts.SlashCommands,
		Middleware:    mw,
		Store:         opts.Store,
		Orchestrator:  opts.Orchestrator,
	}
	rt, err := agent.CreateAgentSessionRuntime(ctx, opts.Cwd, so)
	if err != nil {
		return nil, err
	}
	return &AgentSession{sess: agent.NewAgentSession(rt), rt: rt}, nil
}

// partitionMiddleware type-checks every element of mw against the three
// middleware interfaces (RequestMutator, ResponseObserver,
// ToolInterceptor) and partitions the slice into three typed slices
// preserving registration order within each type. An element that
// satisfies none yields ErrUnknownMiddlewareType naming its Go type.
//
// An element that satisfies more than one interface is appended to every
// matching slice — embedders who build such a type explicitly opt in to
// multi-phase invocation. The common case is single-interface middleware.
func partitionMiddleware(mw []any) (agent.MiddlewareSet, error) {
	var set agent.MiddlewareSet
	for _, v := range mw {
		if v == nil {
			continue
		}
		matched := false
		if m, ok := v.(RequestMutator); ok {
			set.RequestMutators = append(set.RequestMutators, &sdkRequestMutator{m})
			matched = true
		}
		if o, ok := v.(ResponseObserver); ok {
			set.ResponseObservers = append(set.ResponseObservers, &sdkResponseObserver{o})
			matched = true
		}
		if i, ok := v.(ToolInterceptor); ok {
			set.ToolInterceptors = append(set.ToolInterceptors, &sdkToolInterceptor{i})
			matched = true
		}
		if !matched {
			return agent.MiddlewareSet{}, fmt.Errorf("%w: %T", ErrUnknownMiddlewareType, v)
		}
	}
	return set, nil
}

// sdkRequestMutator adapts a pkg/tau.RequestMutator to the runtime's
// internal signature. The adapter is a concrete struct (not an interface)
// so the runtime does not depend on pkg/tau.
type sdkRequestMutator struct {
	rm RequestMutator
}

func (a *sdkRequestMutator) MutateRequest(ctx context.Context, req *llm.Request) error {
	return a.rm.MutateRequest(ctx, (*Request)(req))
}

// sdkResponseObserver adapts a pkg/tau.ResponseObserver. It receives the
// (request, response) pair as *llm.Message — the same type aliased as
// Response in pkg/tau — so the cast is zero-cost.
type sdkResponseObserver struct {
	ro ResponseObserver
}

func (a *sdkResponseObserver) ObserveResponse(ctx context.Context, req *llm.Request, resp *llm.Message) error {
	return a.ro.ObserveResponse(ctx, (*Request)(req), (*Response)(resp))
}

// sdkToolInterceptor adapts a pkg/tau.ToolInterceptor. The ToolCall /
// ToolResult aliases make the cast zero-cost.
type sdkToolInterceptor struct {
	ti ToolInterceptor
}

func (a *sdkToolInterceptor) BeforeToolCall(ctx context.Context, call tools.ToolCall) (*tools.ToolResult, error) {
	r, err := a.ti.BeforeToolCall(ctx, ToolCall(call))
	if err != nil {
		return nil, err
	}
	if r == nil {
		return nil, nil
	}
	out := ToolResult(*r)
	return &out, nil
}

func (a *sdkToolInterceptor) AfterToolCall(ctx context.Context, call tools.ToolCall, result tools.ToolResult) error {
	return a.ti.AfterToolCall(ctx, ToolCall(call), ToolResult(result))
}

// AgentSession is the SDK handle for one agentic session. A session is
// NOT safe to Run concurrently from multiple goroutines — the model is
// single-threaded per session. Abort and Shutdown MAY be called
// concurrently with Run.
type AgentSession struct {
	sess *agent.AgentSession
	rt   *agent.AgentSessionRuntime
}

// Run executes one user turn against the agent session. It blocks until
// the model returns a final response with no pending tool calls, the
// context is cancelled (Abort or external cancel), or an unrecoverable
// error occurs.
//
// Run must NOT be called concurrently with itself on the same session.
// Returns ErrRuntimeShutdown when called after Shutdown.
func (s *AgentSession) Run(ctx context.Context, userInput string) error {
	return s.sess.Run(ctx, userInput)
}

// Abort cancels the in-flight turn's context, if any. Subscribers (tools,
// streaming consumer) see ctx.Done() and unwind. Abort is a no-op when no
// turn is in flight. Safe to call concurrently with Run.
//
// The reason is currently informational; it is not surfaced on the
// resulting TurnEndEvent. A future revision may publish it.
func (s *AgentSession) Abort(reason string) {
	s.sess.Abort(reason)
}

// Shutdown emits SessionShutdown, closes the state manager when the
// runtime owns it, and marks the runtime as terminal. Idempotent and
// safe to call concurrently with Run or Abort. Subsequent Run calls
// return ErrRuntimeShutdown.
//
// The plugin manager is NOT shut down here — per the runtime contract,
// the caller coordinates plugin subprocess lifetime.
func (s *AgentSession) Shutdown(ctx context.Context) error {
	return s.sess.Shutdown(ctx)
}

// Subscribe returns a channel that receives every event emitted by the
// session: SessionStartEvent, TurnStartEvent, MessageStartEvent,
// MessageUpdateEvent, ToolCallEvent, ToolResultEvent, MessageEndEvent,
// TurnEndEvent, and SessionShutdownEvent. The channel is buffered; when
// the session shuts down, the channel closes after SessionShutdownEvent
// is delivered.
//
// Callers that want fine-grained topic selection (e.g., a metrics
// exporter that only wants tool_call + tool_result) should use
// SubscribeTopics instead.
func (s *AgentSession) Subscribe() <-chan SessionEvent {
	return s.rt.EventBus.Subscribe()
}

// SubscribeTopics returns a channel that receives only events whose Topic
// matches one of the supplied topics. Pass any subset of the Topic*
// constants; the bus delivers matching events in emission order. The
// channel is buffered; when the session shuts down, the channel closes
// after a matching SessionShutdownEvent is delivered (or immediately if
// the subscriber did not select SessionShutdown).
//
// SubscribeTopics with no arguments is equivalent to Subscribe. Passing
// an unknown Topic yields zero events for that filter — the bus does not
// error on unrecognised topics, so callers can safely subscribe ahead of
// a topic being added in a future version.
func (s *AgentSession) SubscribeTopics(topics ...Topic) <-chan SessionEvent {
	return s.rt.EventBus.Subscribe(topics...)
}

// Model returns the resolved model identifier the session is configured
// to use. Read-only; never changes for a given session.
func (s *AgentSession) Model() string { return s.rt.Options.Model }

// Cwd returns the absolute working directory the session operates in.
func (s *AgentSession) Cwd() string { return s.rt.Cwd }

// SessionID returns the persisted session id (empty for fresh lazy
// sessions until the first assistant message triggers a flush).
func (s *AgentSession) SessionID() string { return s.rt.SessionID }

// CreatedAt returns when the underlying runtime was constructed.
func (s *AgentSession) CreatedAt() time.Time { return s.rt.CreatedAt }

// Tools returns the registered tool names from the session's underlying
// registry, sorted lexicographically. The returned slice is a fresh copy;
// callers may freely mutate it without affecting the registry or other
// callers. Embedders building dashboards, audit logs, or "active tool
// set" inspectors SHOULD prefer this method over reaching into the
// registry directly.
func (s *AgentSession) Tools() []string {
	names := s.rt.Registry.Names()
	out := make([]string, len(names))
	copy(out, names)
	return out
}

// SlashCommands returns the registered slash-command names (without the
// leading "/") from the session's underlying registry, sorted
// lexicographically. The returned slice is a fresh copy; callers may
// freely mutate it without affecting the registry or other callers.
//
// When Options.SlashCommands was nil at construction time, this method
// falls back to DefaultSlashRegistry() so the returned list reflects the
// built-in command set an embedder's UI would see by default. Pass an
// explicit registry via Options.SlashCommands to override.
func (s *AgentSession) SlashCommands() []string {
	var reg *Registry
	if r, ok := s.rt.SlashRegistry.(*Registry); ok && r != nil {
		reg = r
	} else {
		reg = DefaultSlashRegistry()
	}
	names := reg.Names()
	out := make([]string, len(names))
	for i, n := range names {
		// Strip the leading "/" so consumers see "fork", "checkout", ...
		// rather than "/fork", "/checkout". Both forms are valid in the
		// registry; the SDK contract is "names without the slash".
		if len(n) > 0 && n[0] == '/' {
			out[i] = n[1:]
		} else {
			out[i] = n
		}
	}
	return out
}

// Store returns the cross-session context backend supplied at
// construction, or nil if none was supplied. The returned value is the
// same Store the caller passed via Options.Store; the runtime holds no
// other reference and does not copy.
//
// Embedders use this inspector to retrieve entries from middleware
// (RequestMutator) or from their own UI loop without retaining a
// separate pointer. See docs/sdk/cookbook.md recipe (h) for the
// retrieve-and-inject pattern.
func (s *AgentSession) Store() Store {
	if s == nil || s.rt == nil {
		return nil
	}
	return s.rt.Options.Store
}

// Orchestrator returns the orchestrator injected via Options.Orchestrator,
// or nil when no orchestrator is configured. The returned value satisfies
// the Orchestrator interface; callers may type-assert to a concrete
// implementation (e.g. *SequentialOrchestrator) when they need
// implementation-specific behaviour.
func (s *AgentSession) Orchestrator() Orchestrator {
	if s == nil || s.rt == nil {
		return nil
	}
	if s.rt.Options.Orchestrator == nil {
		return nil
	}
	o, _ := s.rt.Options.Orchestrator.(Orchestrator)
	return o
}

// Spawn constructs a child AgentSession with its own state-tree branch
// and (optionally) inherited LLMClient/Model. The child is a fully
// functional *AgentSession; its Run, Shutdown, Subscribe, etc. methods
// behave identically to a top-level session.
//
// Spawn returns ErrNoOrchestrator when the parent has no orchestrator
// configured (Options.Orchestrator is nil), and ErrRuntimeShutdown
// when the parent's Shutdown has completed. Either is acceptable per
// the spec's "OR a typed equivalent" language; the contract test
// accepts both.
//
// opts is a fresh Options bundle; required fields (Cwd, Model,
// LLMClient, Tools, Settings) MUST be populated unless the caller
// expects inheritance. Cwd is always inherited from the parent (the
// SessionOptions layer has no Cwd field). LLMClient and Model are
// inherited when zero-value. Tools, Settings, and Middleware are
// NEVER inherited; they MUST be supplied.
//
// The runtime NEVER calls Shutdown on the returned child when the
// parent shuts down. The embedder owns the child's lifecycle.
func (s *AgentSession) Spawn(ctx context.Context, opts Options) (*AgentSession, error) {
	if s == nil || s.sess == nil {
		return nil, ErrRuntimeShutdown
	}
	child, err := s.sess.Spawn(ctx, adaptOptionsToInternal(opts))
	if err != nil {
		return nil, err
	}
	return &AgentSession{sess: child, rt: child.Runtime()}, nil
}

// MergeState reconciles a child session's state tree into the parent
// according to spec.Policy. See the MergePolicy documentation for the
// three policies' semantics.
//
// Returns:
//   - nil on success.
//   - ErrNoOrchestrator — parent has no orchestrator configured.
//   - ErrOrchestratorClosed — child's state-tree pointer is nil.
//   - ErrOrchestrationConflict (MergePolicyReplay only) — wraps a
//     *ConflictReport; recover via errors.As(err, &report) where
//     report is a *ConflictReport.
//
// On conflict the parent's state is unchanged (preceding replayed
// entries are NOT rolled back; callers who need atomicity should
// snapshot before calling).
func (s *AgentSession) MergeState(ctx context.Context, child *AgentSession, spec MergeSpec) error {
	if s == nil || s.sess == nil || child == nil || child.sess == nil {
		return ErrOrchestratorClosed
	}
	shell := agent.MergeSpecShell{
		Policy:           agent.MergePolicyShell(spec.Policy),
		Phase:            spec.Phase,
		ConflictCallback: adaptConflictCallback(spec.ConflictCallback),
	}
	return s.sess.MergeState(ctx, child.sess, shell)
}

// adaptConflictCallback wraps a public SDK conflict callback so the
// runtime can invoke it. The wrapping is identity-preserving: the
// *ConflictReport passed to the SDK callback is the same pointer the
// runtime produced (because pkg/tau.ConflictReport IS agent.ConflictReportShell
// via type alias).
func adaptConflictCallback(cb func(*ConflictReport) error) func(*agent.ConflictReportShell) error {
	if cb == nil {
		return nil
	}
	return func(r *agent.ConflictReportShell) error {
		return cb((*ConflictReport)(r))
	}
}

// adaptOptionsToInternal converts the public SDK Options bundle to the
// internal/agent.SessionOptions shape expected by the runtime. The
// conversion is shallow; the SDK re-uses the caller's slice/map headers
// where possible.
func adaptOptionsToInternal(opts Options) agent.SessionOptions {
	mw, _ := partitionMiddleware(opts.Middleware)
	so := agent.SessionOptions{
		Model:          opts.Model,
		ThinkingLevel:  opts.ThinkingLevel,
		Settings:       opts.Settings,
		Transport:      opts.Transport,
		SteeringMode:   opts.SteeringMode,
		Tools:          opts.Tools,
		Plugins:        opts.Plugins,
		LLMClient:      opts.LLMClient,
		StateManager:   opts.StateManager,
		ContextWindow:  opts.ContextWindow,
		ConfigDir:      opts.ConfigDir,
		SessionID:      opts.SessionID,
		SlashCommands:  opts.SlashCommands,
		Middleware:     mw,
		Store:          opts.Store,
		Orchestrator:   opts.Orchestrator,
	}
	return so
}

// NewSequentialOrchestrator returns the reference sequential-phase
// Orchestrator. Phases execute strictly in dependency order; cycles
// are rejected at Run time. Each phase spawns a child session, runs
// the phase prompt, drains events onto the multiplexed channel, and
// (when MergePolicy is not MergePolicyNone) calls MergeState before
// proceeding.
//
// The parent MUST have been constructed with Options.Orchestrator set
// (otherwise parent.Spawn will return ErrNoOrchestrator at Run time).
// The returned Orchestrator does NOT take ownership of the parent;
// the caller owns the parent's lifecycle.
//
// See the Orchestration section in doc.go for the full lifecycle
// contract and the cookbook at docs/sdk/cookbook.md recipe (i) for a
// worked example.
func NewSequentialOrchestrator(parent *AgentSession) Orchestrator {
	return orchestrator.NewSequentialOrchestrator(parent.sess)
}

// NewInMemoryManager returns a StateManager that keeps the entire state
// tree in memory and never writes to disk. Use it in tests and in
// ephemeral sessions that should not persist (e.g., a one-shot batch
// evaluator, a CI bot that does not need to resume).
//
// The returned manager is safe for concurrent use. When injected via
// Options.StateManager, the runtime does NOT call Close on it — the
// caller owns its lifecycle.
func NewInMemoryManager(cwd string) StateManager {
	return state.NewInMemoryManager(cwd)
}

// NewFileStore returns the reference file-backed Store rooted at dir.
// Each entry is persisted as a single markdown file (dir/<id>.md) with
// YAML-style frontmatter; the directory is created with mode 0700 if
// missing, and files are written with mode 0600. Writes are atomic
// (staged to .tmp then renamed) under a per-store flock so concurrent
// Puts and Queries are safe.
//
// FileStore does NOT compute embeddings — Query.EmbeddingQuery returns
// ErrUnsupportedQuery. Embedders needing semantic similarity implement
// Store directly against their vector backend.
//
// When injected via Options.Store, the runtime does NOT call Close on
// the returned store — the caller owns its lifecycle.
func NewFileStore(dir string) (Store, error) {
	return storage.NewFileStore(dir)
}

// --- Convenience provider constructors --------------------------------------
//
// NewAnthropicClient and NewOpenAIClient are thin wrappers around the
// process-wide provider registry. They look up the factory registered under
// the canonical name ("anthropic" / "openai") and invoke it. This keeps a
// single source of truth: the built-in factories self-register via
// provider_builtins.go, and an embedder who wants a different client under
// the same name just calls RegisterProvider before NewAnthropicClient.
//
// Both return ErrProviderNotFound when no factory is registered under the
// requested name. The most common cause is building with
// `-tags provider_builtins=off` without manually registering first.

// AnthropicOptions configures a built-in Anthropic client. It is the
// SDK-level mirror of the internal anthropic.Options; the factory copies
// each field across. APIKey is required.
type AnthropicOptions struct {
	// APIKey is the resolved credential. Required.
	APIKey string

	// BaseURL overrides the default Anthropic endpoint. Useful for
	// pointing at a local proxy or a compatible gateway.
	BaseURL string

	// Headers are appended to every request after the auth headers.
	Headers map[string]string

	// Transport overrides the underlying *http.Client. If nil, the
	// provider uses http.DefaultClient.
	Transport *http.Client
}

// OpenAIOptions configures a built-in OpenAI (or OpenAI-compatible) client.
// Point BaseURL at DeepSeek, Ollama, LM Studio, OpenRouter, Groq, etc., to
// use the OpenAI Chat Completions protocol against a different backend.
type OpenAIOptions = AnthropicOptions

// NewAnthropicClient resolves the "anthropic" factory from the provider
// registry and invokes it. Returns ErrProviderNotFound when built-ins are
// compiled out and the embedder has not registered a replacement.
func NewAnthropicClient(opts AnthropicOptions) (LLMClient, error) {
	return resolveProvider("anthropic", ProviderOptions{
		APIKey:    opts.APIKey,
		BaseURL:   opts.BaseURL,
		Headers:   opts.Headers,
		Transport: opts.Transport,
	})
}

// NewOpenAIClient resolves the "openai" factory from the provider registry
// and invokes it. Returns ErrProviderNotFound when built-ins are compiled
// out and the embedder has not registered a replacement.
func NewOpenAIClient(opts OpenAIOptions) (LLMClient, error) {
	return resolveProvider("openai", ProviderOptions{
		APIKey:    opts.APIKey,
		BaseURL:   opts.BaseURL,
		Headers:   opts.Headers,
		Transport: opts.Transport,
	})
}

// --- Auth resolution --------------------------------------------------------
//
// ResolveAuth is the SDK-level dispatch into the four-step auth chain every
// built-in provider implements:
//
//  1. Explicit value supplied by the caller (may be a $ENV or !shell sigil).
//  2. auth.json entry for the provider (loaded via the AuthStore on disk).
//  3. Provider-specific env var (ANTHROPIC_API_KEY, OPENAI_API_KEY).
//  4. Empty result is an error.
//
// tau deviates from pi by exposing this as a single SDK-level function that
// dispatches by provider name, rather than one function per provider. This
// matches the registry model: callers that registered a custom provider can
// also teach ResolveAuth about their env var via a future registration hook;
// for the two built-ins, the dispatch is a switch.
//
// Pass an explicit="" to skip step 1 and resolve purely from auth.json +
// env. Pass auth=nil to skip step 2 (typical for ephemeral / CI runs that
// have no auth.json on disk).
//
// The returned string is the resolved credential. The second return is the
// provenance ("explicit", "sigil", "auth.json", or "env") for diagnostics.
// On failure, the error describes every step that was attempted.

// ResolveAuth resolves an API key for the named provider using the
// four-step chain documented above. provider MUST be "anthropic" or
// "openai" today; unknown providers return ErrProviderNotFound.
func ResolveAuth(provider, explicit string, auth config.AuthStore) (string, string, error) {
	switch provider {
	case "anthropic":
		r, err := anthropic.ResolveAuth(explicit, auth)
		if err != nil {
			return "", "", err
		}
		return r.APIKey, string(r.Source), nil
	case "openai":
		r, err := openai.ResolveAuth("openai", openai.EnvVar, explicit, auth)
		if err != nil {
			return "", "", err
		}
		return r.APIKey, string(r.Source), nil
	default:
		return "", "", fmt.Errorf("%w: %q", ErrProviderNotFound, provider)
	}
}

// --- Convenience constructors -----------------------------------------------
//
// The constructors below give embedders a single-import path to the
// canonical defaults: the built-in settings bundle, the seven built-in
// tools, and a deterministic LLMClient for tests. Each is a thin wrapper
// that re-exports an internal constructor; the surface stays small on
// purpose so the SDK remains a facade, not a parallel implementation.

// DefaultSettings returns the fully-resolved default Settings bundle. It
// is the zero-dependency way to get a Settings value for Options.Settings
// without reaching into internal/config.
func DefaultSettings() Settings { return config.DefaultSettings() }

// BuiltinTools returns the seven built-in tools (read, bash, edit, write,
// grep, find, ls) wired to their OS-backed implementations. The returned
// slice is freshly allocated on every call; callers may append their own
// HeadlessTool entries before passing to Options.Tools.
//
// Tools that have no filesystem equivalent (e.g., a fetch tool that
// requires a BaseURL) are NOT included — embedders add those explicitly.
func BuiltinTools() []HeadlessTool {
	return []HeadlessTool{
		tools.NewReadTool(tools.OSReadOperations{}),
		tools.NewBashTool(tools.OSBashOperations{}),
		tools.NewEditTool(tools.OSEditOperations{}),
		tools.NewWriteTool(tools.OSWriteOperations{}),
		tools.NewGrepTool(tools.OSGrepOperations{}),
		tools.NewFindTool(tools.OSFindOperations{}),
		tools.NewLSTool(tools.OSLSOperations{}),
	}
}

// NewFauxProvider returns a deterministic LLMClient that streams the given
// canned responses in order. It is the SDK-level equivalent of the
// internal fauxprovider used by tau's own tests; embedders use it to write
// unit tests that exercise the agent loop without hitting a real API.
//
// Pass one response for a single-shot test; pass several to script a
// multi-turn exchange. Each Stream call advances to the next response; an
// empty variadic list yields the provider's built-in default response.
func NewFauxProvider(responses ...string) LLMClient {
	if len(responses) == 0 {
		return fauxprovider.NewWithResponse(fauxprovider.DefaultResponse)
	}
	// Join with newlines so the faux provider emits them as one TextDelta;
	// multi-response scripting is the caller's job (one Stream per
	// response). For the common single-response case this is exact.
	return fauxprovider.NewWithResponse(responses[0])
}
