// command_session.go — the public interfaces and adapter types that
// decouple slash commands (built-in and external plugin) from the
// concrete *AgentSession, *AgentSessionRuntime, and resolvedOptions
// types.
//
// The third parameter of slash.Command.Execute was previously
// *agent.AgentSession — an internal-package type that external Go
// modules cannot name. The widening here gives commands an interface
// (CommandSession) they can implement against; built-in commands
// receive the same interface via the same adapter.
//
// See openspec/specs/sdk-public-api/spec.md "Slash command interface"
// for the spec requirement this file implements. See
// openspec/changes/open-slash-command-surface/ for the change record.
package agent

import (
	"context"
	"errors"
	"time"

	"github.com/coevin/tau/internal/compaction"
	"github.com/coevin/tau/internal/config"
	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/state"
	"github.com/coevin/tau/internal/storage"
)

// CommandSession is the session-shaped surface a slash command (built-in
// or external plugin) receives via Command.Execute. It exposes the
// lifecycle, inspection, session-tree, subscription, and runtime
// accessor methods commands need, without leaking the concrete
// *AgentSession, *AgentSessionRuntime, or resolvedOptions types (all of
// which are internal-package types that external Go modules cannot
// name).
//
// The interface is satisfied by an unexported adapter type
// (commandSessionView) declared below. Internal callers obtain the
// adapter via (*AgentSession).AsCommandSession(); external code only
// sees the interface.
//
// The surface is sized to cover every method invoked by the built-in
// commands (/checkout, /compact, /fork, /label, /model, ...) plus the
// methods external plugin commands need (e.g. the SDD plugin's planned
// /apply, /archive). See design.md "Audit table" for the call-site
// enumeration.
type CommandSession interface {
	// Lifecycle.
	Run(ctx context.Context, userInput string) error
	Abort(reason string)
	Shutdown(ctx context.Context) error

	// Inspection.
	Model() string
	Cwd() string
	SessionID() string
	CreatedAt() time.Time
	Tools() []string
	SlashCommands() []string
	Store() storage.Store
	Orchestrator() any

	// Event subscription.
	Subscribe() <-chan Event
	SubscribeTopics(topics ...Topic) <-chan Event

	// Session tree.
	Spawn(ctx context.Context, opts SessionOptions) (CommandSession, error)
	MergeState(ctx context.Context, child CommandSession, spec MergeSpecShell) error

	// Runtime surface (power-user). Runtime() returns the wired
	// components commands reach today via session.Runtime() on the
	// concrete type.
	Runtime() CommandRuntime
}

// CommandRuntime exposes the runtime components commands reach today
// via session.Runtime() on the concrete *AgentSession. Each accessor
// corresponds to a public field on *AgentSessionRuntime (State, Options,
// Compactor, EventBus); the adapter turns the field read into a method
// call so the interface is implementable without leaking the runtime
// struct type.
type CommandRuntime interface {
	State() state.Manager
	Options() CommandOptions
	Compactor() *compaction.Compactor
	EventBus() *EventBus
}

// CommandOptions exposes the post-defaults options bundle commands reach
// today via rt.Options on the concrete runtime. Model() returns the
// active model id; SetModel(string) mutates it (replacing the pre-change
// direct field write `rt.Options.Model = args` used by /model). The
// remaining accessors are read-only; future setters can be added on
// demand without breaking consumers.
type CommandOptions interface {
	Model() string
	SetModel(string)
	LLMClient() llm.LLMClient
	ContextWindow() int
	KnownModels() []config.KnownModel
	ProviderAPI() config.ModelAPI
}

// ErrMergeStateForeignChild is returned by (*commandSessionView).MergeState
// when the child argument is not a commandSessionView produced by
// AsCommandSession or Spawn. Callers MAY detect this via errors.Is to
// distinguish "foreign child wrapper" from the underlying MergeState's
// own sentinels (ErrRuntimeShutdown, ErrNoOrchestrator, ErrOrchestratorClosed).
//
// External plugin authors encountering this sentinel have wrapped the
// CommandSession returned by Spawn in their own type before passing it
// back to MergeState; the wrapper must be removed so MergeState can
// reach the concrete *AgentSession underneath.
var ErrMergeStateForeignChild = errors.New("agent: MergeState child is not an agent session")

// commandSessionView adapts a *AgentSession to the CommandSession
// interface. It reads off the session's wired runtime (s.rt) for
// inspectors and forwards lifecycle calls to the underlying session.
// The adapter lives in package agent so it can read unexported fields
// (s.rt) without exposing them.
type commandSessionView struct {
	s *AgentSession
}

// Ensure commandSessionView satisfies CommandSession at compile time.
// If a future change adds a method to CommandSession without adding
// the corresponding method to commandSessionView, this assertion fails
// the build.
var _ CommandSession = commandSessionView{}

// AsCommandSession returns a CommandSession view over this session.
// Built-in slash commands and external plugins both receive this view
// via Registry.Execute; the view exposes the wired runtime without
// leaking the concrete *AgentSession type to external packages.
//
// Internal callers (TUI, RPC server, tests) pass the result to
// slash.Registry.Execute in place of the raw *AgentSession.
func (s *AgentSession) AsCommandSession() CommandSession {
	return commandSessionView{s: s}
}

// Run forwards to the underlying session's turn loop. See
// AgentSession.Run for the full contract.
func (v commandSessionView) Run(ctx context.Context, userInput string) error {
	return v.s.Run(ctx, userInput)
}

// Abort forwards to the underlying session's abort. See AgentSession.Abort.
func (v commandSessionView) Abort(reason string) {
	v.s.Abort(reason)
}

// Shutdown forwards to the underlying session's shutdown. See
// AgentSession.Shutdown.
func (v commandSessionView) Shutdown(ctx context.Context) error {
	return v.s.Shutdown(ctx)
}

// Model returns the resolved model identifier the session is configured
// to use. Read directly off the runtime's options bundle.
func (v commandSessionView) Model() string {
	if v.s == nil || v.s.rt == nil {
		return ""
	}
	return v.s.rt.Options.Model
}

// Cwd returns the absolute working directory the session operates in.
func (v commandSessionView) Cwd() string {
	if v.s == nil || v.s.rt == nil {
		return ""
	}
	return v.s.rt.Cwd
}

// SessionID returns the persisted session id (empty for fresh lazy
// sessions until the first assistant message triggers a flush).
func (v commandSessionView) SessionID() string {
	if v.s == nil || v.s.rt == nil {
		return ""
	}
	return v.s.rt.SessionID
}

// CreatedAt returns when the underlying runtime was constructed.
func (v commandSessionView) CreatedAt() time.Time {
	if v.s == nil || v.s.rt == nil {
		return time.Time{}
	}
	return v.s.rt.CreatedAt
}

// Tools returns the registered tool names from the session's underlying
// registry, sorted lexicographically. The returned slice is a fresh
// copy; callers may freely mutate it.
func (v commandSessionView) Tools() []string {
	if v.s == nil || v.s.rt == nil || v.s.rt.Registry == nil {
		return nil
	}
	return v.s.rt.Registry.Names()
}

// slashRegistryWithNames is an unnamed structural interface used to
// read the SlashRegistry field without importing internal/slash (which
// would create an import cycle). *slash.Registry satisfies this
// interface structurally.
type slashRegistryWithNames interface {
	Names() []string
}

// SlashCommands returns the registered slash-command names (without the
// leading "/") from the session's underlying slash registry. Returns
// nil when no registry is wired (the caller can fall back to its own
// default).
//
// The leading "/" is stripped to match the SDK facade's convention.
func (v commandSessionView) SlashCommands() []string {
	if v.s == nil || v.s.rt == nil || v.s.rt.SlashRegistry == nil {
		return nil
	}
	reg, ok := v.s.rt.SlashRegistry.(slashRegistryWithNames)
	if !ok || reg == nil {
		return nil
	}
	names := reg.Names()
	out := make([]string, len(names))
	for i, n := range names {
		if len(n) > 0 && n[0] == '/' {
			out[i] = n[1:]
		} else {
			out[i] = n
		}
	}
	return out
}

// Store returns the cross-session context backend supplied at
// construction, or nil if none was supplied.
func (v commandSessionView) Store() storage.Store {
	if v.s == nil || v.s.rt == nil {
		return nil
	}
	return v.s.rt.Options.Store
}

// Orchestrator returns the orchestrator injected via Options.Orchestrator,
// or nil when no orchestrator is configured. The returned value
// satisfies the SDK Orchestrator interface when the injected value
// does; callers type-assert as needed.
func (v commandSessionView) Orchestrator() any {
	if v.s == nil || v.s.rt == nil {
		return nil
	}
	return v.s.rt.Options.Orchestrator
}

// Subscribe returns a channel that receives every event emitted by the
// session. See EventBus.Subscribe for delivery semantics.
func (v commandSessionView) Subscribe() <-chan Event {
	if v.s == nil || v.s.rt == nil || v.s.rt.EventBus == nil {
		return nil
	}
	return v.s.rt.EventBus.Subscribe()
}

// SubscribeTopics returns a channel that receives only events whose
// Topic matches one of the supplied topics.
func (v commandSessionView) SubscribeTopics(topics ...Topic) <-chan Event {
	if v.s == nil || v.s.rt == nil || v.s.rt.EventBus == nil {
		return nil
	}
	return v.s.rt.EventBus.Subscribe(topics...)
}

// Spawn constructs a child session with its own state-tree branch.
// The returned CommandSession is backed by a commandSessionView
// wrapping the child *AgentSession; callers that need to pass the
// child back into MergeState should pass it through unchanged.
func (v commandSessionView) Spawn(ctx context.Context, opts SessionOptions) (CommandSession, error) {
	if v.s == nil {
		return nil, ErrRuntimeShutdown
	}
	child, err := v.s.Spawn(ctx, opts)
	if err != nil {
		return nil, err
	}
	return commandSessionView{s: child}, nil
}

// MergeState reconciles a child session's state tree into the parent's
// according to spec.Policy. The child MUST be a commandSessionView
// returned by Spawn (or by AsCommandSession on a *AgentSession); any
// other CommandSession implementation triggers
// ErrMergeStateForeignChild. A nil child triggers ErrOrchestratorClosed,
// matching the underlying MergeState's own nil-child behavior.
func (v commandSessionView) MergeState(ctx context.Context, child CommandSession, spec MergeSpecShell) error {
	if v.s == nil {
		return ErrRuntimeShutdown
	}
	if child == nil {
		return ErrOrchestratorClosed
	}
	cv, ok := child.(commandSessionView)
	if !ok {
		return ErrMergeStateForeignChild
	}
	return v.s.MergeState(ctx, cv.s, spec)
}

// Runtime returns the wired runtime as a CommandRuntime interface.
// Callers that only need State/Options/Compactor/EventBus should
// prefer the specific accessor over holding the interface.
func (v commandSessionView) Runtime() CommandRuntime {
	if v.s == nil || v.s.rt == nil {
		return nil
	}
	return commandRuntimeView{rt: v.s.rt}
}

// commandRuntimeView adapts a *AgentSessionRuntime to the
// CommandRuntime interface. Each method corresponds to a public field
// on the concrete runtime; the adapter turns the field read into a
// method call.
type commandRuntimeView struct {
	rt *AgentSessionRuntime
}

// Ensure commandRuntimeView satisfies CommandRuntime at compile time.
var _ CommandRuntime = commandRuntimeView{}

// State returns the state manager the runtime was constructed with.
func (v commandRuntimeView) State() state.Manager {
	if v.rt == nil {
		return nil
	}
	return v.rt.State
}

// Options returns the resolved options bundle as a CommandOptions
// interface. The returned adapter mutates the underlying options in
// place, so changes via SetModel are visible to subsequent turns.
func (v commandRuntimeView) Options() CommandOptions {
	if v.rt == nil {
		return nil
	}
	return commandOptionsView{opts: &v.rt.Options}
}

// Compactor returns the wired compactor (may be nil if compaction is
// disabled).
func (v commandRuntimeView) Compactor() *compaction.Compactor {
	if v.rt == nil {
		return nil
	}
	return v.rt.Compactor
}

// EventBus returns the session's event bus.
func (v commandRuntimeView) EventBus() *EventBus {
	if v.rt == nil {
		return nil
	}
	return v.rt.EventBus
}

// commandOptionsView adapts a *resolvedOptions to the CommandOptions
// interface. SetModel writes the underlying field directly so the
// mutation is visible to every subsequent read of rt.Options.Model
// in the same runtime.
type commandOptionsView struct {
	opts *resolvedOptions
}

// Ensure commandOptionsView satisfies CommandOptions at compile time.
var _ CommandOptions = commandOptionsView{}

// Model returns the active model identifier.
func (v commandOptionsView) Model() string {
	if v.opts == nil {
		return ""
	}
	return v.opts.Model
}

// SetModel mutates the active model identifier. The write goes to the
// same field the pre-change /model command wrote via rt.Options.Model = args;
// observable behavior is identical.
func (v commandOptionsView) SetModel(m string) {
	if v.opts == nil {
		return
	}
	v.opts.Model = m
}

// LLMClient returns the wired LLM client (may be nil if the runtime
// was constructed without one — unusual, but defensive).
func (v commandOptionsView) LLMClient() llm.LLMClient {
	if v.opts == nil {
		return nil
	}
	return v.opts.LLMClient
}

// ContextWindow returns the resolved context window in tokens.
func (v commandOptionsView) ContextWindow() int {
	if v.opts == nil {
		return 0
	}
	return v.opts.ContextWindow
}

// KnownModels returns the resolved known-models slice (may be empty).
// The returned slice aliases the underlying field; callers SHOULD NOT
// mutate it.
func (v commandOptionsView) KnownModels() []config.KnownModel {
	if v.opts == nil {
		return nil
	}
	return v.opts.KnownModels
}

// ProviderAPI returns the resolved provider API kind.
func (v commandOptionsView) ProviderAPI() config.ModelAPI {
	if v.opts == nil {
		return ""
	}
	return v.opts.ProviderAPI
}
