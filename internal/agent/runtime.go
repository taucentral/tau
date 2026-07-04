// runtime.go — wire every moving part into a single AgentSessionRuntime.
//
// CreateAgentSessionRuntime is the single entry point the CLI (Phase 9) and
// the public SDK (Phase 12) call to turn a SessionOptions bundle into a
// runnable agent. The factory:
//
//   1. Validates required fields (Model, LLMClient, ≥1 Tool).
//   2. Resolves defaults from Settings for any optional field the caller
//      left zero (ThinkingLevel, Transport, SteeringMode).
//   3. Resolves the global ConfigDir (caller-supplied or config.ConfigDir()).
//   4. Builds the EventBus (bounded, non-blocking).
//   5. Wires the StateManager (caller-supplied or a fresh lazy manager
//      rooted at cwd — no .bolt file is written until the first assistant
//      message).
//   6. Wires the TokenCounter (caller-supplied or the real BPE counter).
//   7. Constructs the Compactor with a Summarizer so above-threshold
//      sessions compact correctly; the Enabled flag is consulted by the
//      turn loop, not by the factory.
//   8. Builds the tool Registry: built-ins first, then plugin tools.
//      Plugin/built-in collisions are first-wins and reported via the
//      EventBus; registration never fails the factory for a duplicate.
//   9. Builds the prompts.Assembler and Loader from ConfigDir + cwd.
//  10. Builds a fresh FileMutationQueue (per-session, per-path lock).
//
// The factory does NOT call plugin.Manager.SpawnAll — that is the caller's
// job because spawn failures are observable at the application boundary
// (the CLI emits a diagnostic and continues with built-ins only). Likewise,
// plugin.Manager.Shutdown is the caller's job (the runtime does not own
// plugin subprocess lifetimes).

package agent

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/coevin/tau/internal/compaction"
	"github.com/coevin/tau/internal/config"
	"github.com/coevin/tau/internal/llm/tokencounter"
	"github.com/coevin/tau/internal/prompts"
	"github.com/coevin/tau/internal/state"
	"github.com/coevin/tau/internal/tools"
)

// AgentSessionRuntime is the wired bundle: every component the turn loop
// (session.go) needs to serve one user turn. The runtime is safe to read
// concurrently after CreateAgentSessionRuntime returns; mutable fields
// (shutdown coordination) are guarded by shutdownMu.
//
// The runtime is one-shot: Shutdown closes the state manager and marks
// the runtime as shut down. Subsequent Run calls return ErrRuntimeShutdown.
//
//nolint:revive // exported name is part of the public SDK API surface (pkg/tau).
type AgentSessionRuntime struct {
	// Cwd is the absolute working directory this session operates in.
	Cwd string
	// ConfigDir is the resolved global config directory.
	ConfigDir string
	// SessionID is the persisted session id (empty for fresh lazy
	// sessions until the first assistant message triggers a flush).
	SessionID string
	// CreatedAt is when the runtime was constructed. Emitted in the
	// SessionStartEvent payload.
	CreatedAt time.Time

	// Options is the resolved-options bundle (defaults applied).
	Options resolvedOptions

	// EventBus is the bounded pub/sub for agent events. Owned by the
	// runtime; Shutdown emits SessionShutdown and closes subscribers.
	EventBus *EventBus

	// State is the session's state-tree Manager. May be caller-owned
	// (opts.StateManager) or runtime-owned (factory-created).
	State state.Manager

	// Registry is the merged tool registry (built-ins + plugin tools).
	Registry *tools.Registry

	// SlashRegistry holds the slash-command registry, when supplied via
	// SDK Options.SlashCommands. nil means the embedder did not inject
	// one; SDK consumers can check via the *AgentSession.SlashCommands()
	// inspector, which falls back to the default built-in registry.
	//
	// Typed interface{} (not *slash.Registry) to avoid a circular
	// dependency between agent and slash. The SDK type-asserts when
	// reading.
	SlashRegistry interface{}

	// Compactor runs the compaction pipeline. Consulted by the turn
	// loop after every assistant message.
	Compactor *compaction.Compactor

	// Summarizer is the LLM-backed summarizer used by Compactor for
	// above-threshold sessions. nil when no LLMClient was supplied.
	Summarizer *compaction.Summarizer

	// Assembler composes the system prompt (built-in + global + project).
	Assembler *prompts.Assembler

	// TemplateLoader resolves named prompt templates from disk.
	TemplateLoader *prompts.Loader

	// MutationQueue serializes per-path file mutations across tool calls
	// in a single turn (concurrent tools writing the same file run in
	// arrival order, not racy).
	MutationQueue *tools.FileMutationQueue

	// ownsState is true when the factory created the State manager
	// (and thus Shutdown should Close it). False when the caller
	// injected a manager via opts.StateManager.
	ownsState bool

	// shutdownMu guards shutdownOnce, shutdownDone, and shutdownReason.
	shutdownMu sync.Mutex
	// shutdownOnce is the fence ensuring Shutdown runs its body exactly
	// once even under concurrent calls.
	shutdownDone   bool
	shutdownReason string
}

// ErrRuntimeShutdown is returned by Run when called after Shutdown has
// been invoked. The runtime is one-shot: callers must construct a new
// runtime to serve further turns.
var ErrRuntimeShutdown = errors.New("agent: runtime is shut down")

// CreateAgentSessionRuntime validates opts, applies defaults, and returns
// a wired runtime ready to serve turns via (*AgentSession).Run.
//
// The factory never reaches outside the values supplied in opts (plus cwd
// and the resolved ConfigDir). It performs no plugin subprocess management
// and no network I/O. Filesystem access is limited to config.ConfigDir()
// resolution when opts.ConfigDir is empty.
func CreateAgentSessionRuntime(ctx context.Context, cwd string, opts SessionOptions) (*AgentSessionRuntime, error) {
	// Honour ctx cancellation at every yield point in the factory.
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("agent: cancelled before runtime init: %w", err)
	}
	if err := opts.validate(); err != nil {
		return nil, err
	}
	if cwd == "" {
		return nil, errors.New("agent: cwd is required")
	}

	r := opts.resolve()

	// Resolve the global config directory. Caller-supplied wins; otherwise
	// fall back to the standard resolution (TAU_CONFIG_DIR → XDG → default).
	configDir := opts.ConfigDir
	if configDir == "" {
		cd, err := config.ConfigDir()
		if err != nil {
			return nil, fmt.Errorf("agent: resolve config dir: %w", err)
		}
		configDir = cd
	}

	// EventBus: one bounded bus per runtime.
	bus := NewEventBus(DefaultEventBuffer)

	// StateManager: caller-injected wins; otherwise a fresh lazy manager.
	var mgr state.Manager
	ownsState := false
	if r.StateManager != nil {
		mgr = r.StateManager
	} else {
		header := state.SessionHeaderPayload{
			Cwd:       cwd,
			Model:     r.Model,
			CreatedAt: time.Now().UTC(),
			Version:   state.CurrentSchemaVersion,
		}
		m, err := state.Create(cwd, header)
		if err != nil {
			return nil, fmt.Errorf("agent: create state manager: %w", err)
		}
		mgr = m
		ownsState = true
	}

	// TokenCounter: caller-injected wins; otherwise the real BPE counter.
	counter := r.TokenCounter
	if counter == nil {
		counter = tokencounter.New()
	}

	// Summarizer + Compactor. The summarizer needs the LLM client; if the
	// caller somehow wired a compactor without one, NewCompactor still
	// returns a usable value (above-threshold compaction will surface a
	// descriptive error rather than panic).
	summarizer := compaction.NewSummarizer(r.LLMClient, r.Model)
	reserve := 0
	if r.Settings.Compaction != nil && r.Settings.Compaction.ReserveTokens != nil {
		reserve = *r.Settings.Compaction.ReserveTokens
	}
	keepRecent := 0
	if r.Settings.Compaction != nil && r.Settings.Compaction.KeepRecentTokens != nil {
		keepRecent = *r.Settings.Compaction.KeepRecentTokens
	}
	// r.ContextWindow is the model's budget (resolvedOptions embeds the
	// caller's ContextWindow at options.go:192). Passing it through lets
	// NewCompactor clamp an oversized keepRecent to contextWindow/2.
	compactor := compaction.NewCompactor(counter, summarizer, reserve, keepRecent, r.ContextWindow)

	// Tool registry: built-ins first, plugin tools layered on top.
	// Plugin/built-in collisions are first-wins and reported via the bus
	// (the event is informational; registration does not fail).
	registry := tools.NewRegistry()
	for _, t := range r.BuiltinTools {
		if err := registry.Register(t); err != nil {
			return nil, fmt.Errorf("agent: register built-in tool %q: %w", t.Name(), err)
		}
	}
	if r.PluginManager != nil {
		for _, t := range r.PluginManager.Tools() {
			if err := registry.Register(t); err != nil {
				// First-wins: emit a single informational event and continue.
				bus.Publish(registrationCollisionEvent{
					When:   time.Now().UTC(),
					Name:   t.Name(),
					Reason: err.Error(),
				})
			}
		}
	}

	// Prompts assembler + template loader. Both are immutable after
	// construction; safe for concurrent use. The assembler adopts the
	// walk knobs from Settings.Prompts so users can configure the
	// ancestor-walk behavior via settings.json.
	var walkOpts prompts.WalkOpts
	if r.Settings.Prompts != nil {
		if r.Settings.Prompts.WalkToRoot != nil {
			walkOpts.WalkToRoot = *r.Settings.Prompts.WalkToRoot
		}
		if r.Settings.Prompts.MaxAncestorDepth != nil {
			walkOpts.MaxAncestorDepth = *r.Settings.Prompts.MaxAncestorDepth
		}
		if r.Settings.Prompts.StopDir != nil {
			walkOpts.StopDir = *r.Settings.Prompts.StopDir
		}
	}
	assembler := prompts.NewAssemblerWithWalk(configDir, cwd, walkOpts)
	templateLoader := prompts.NewLoader(configDir, cwd)

	// Per-session file mutation queue.
	queue := tools.NewFileMutationQueue()

	rt := &AgentSessionRuntime{
		Cwd:            cwd,
		ConfigDir:      configDir,
		SessionID:      opts.SessionID,
		CreatedAt:      time.Now().UTC(),
		Options:        r,
		EventBus:       bus,
		State:          mgr,
		Registry:       registry,
		SlashRegistry:  opts.SlashCommands,
		Compactor:      compactor,
		Summarizer:     summarizer,
		Assembler:      assembler,
		TemplateLoader: templateLoader,
		MutationQueue:  queue,
		ownsState:      ownsState,
	}
	return rt, nil
}

// registrationCollisionEvent is an internal event published when a plugin
// tool name collides with an already-registered tool. It is NOT part of
// the public Topic set; it is emitted on the bus so subscribers logging
// all events see the diagnostic. The topic is intentionally distinct from
// the agent-loop lifecycle topics so it does not confuse the TUI.
type registrationCollisionEvent struct {
	When   time.Time
	Name   string
	Reason string
}

// Topic implements Event. Returns a reserved diagnostic topic.
func (registrationCollisionEvent) Topic() Topic { return TopicDiagnostic }

// TopicDiagnostic is the reserved topic for non-lifecycle diagnostic
// events (e.g., tool registration collisions, plugin warnings). Not part
// of the canonical agent-loop event sequence.
const TopicDiagnostic Topic = "diagnostic"
