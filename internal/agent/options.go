// options.go — SessionOptions, the input to CreateAgentSessionRuntime.
//
// The options struct carries every decision the runtime factory cannot
// derive on its own: the user's model/transport/steering choices, the
// pre-loaded Settings, the plugin manager, the built-in tool set, and
// optional injection seams for tests (LLMClient, StateManager,
// TokenCounter).
//
// Design rule: the factory never reaches outside the values supplied in
// Options (+ the cwd parameter it takes). This keeps the agent layer
// independent of how Settings was loaded or how the plugin manager was
// constructed, and lets the test/e2e harness swap any single seam.

package agent

import (
	"github.com/coevin/tau/internal/config"
	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/llm/tokencounter"
	"github.com/coevin/tau/internal/plugins"
	"github.com/coevin/tau/internal/state"
	"github.com/coevin/tau/internal/tools"
)

// SessionOptions is the input bundle to CreateAgentSessionRuntime. The
// caller fills the required fields and any optional injection seams
// needed (LLMClient for tests, StateManager for in-memory state, etc.).
//
// Required fields:
//   - Model
//   - Settings (caller loads from disk via config layer)
//   - LLMClient (real provider in production; faux provider in tests)
//   - At least one entry in Tools (the built-in tool set)
//
// Optional fields:
//   - Plugins (nil disables plugin tool discovery)
//   - StateManager (nil lets the factory create one rooted at cwd)
//   - TokenCounter (nil lets the factory pick the BPE counter for Model)
//   - ThinkingLevel (zero value defers to Settings.DefaultThinkingLevel)
//   - Transport (zero value defers to Settings.Transport)
//   - SteeringMode (zero value defers to Settings.SteeringMode)
//   - SessionID (empty starts a new session; non-empty resumes)
//   - ContextWindow (0 lets the factory look it up from Settings/ModelsFile)
type SessionOptions struct {
	// Model is the model identifier to send to the LLM (e.g.
	// "claude-opus-4-5-20251101"). Required.
	Model string

	// ThinkingLevel selects the model's reasoning budget. One of
	// config.ThinkingOff / ThinkingMinimal / ThinkingLow / ThinkingMedium
	// / ThinkingHigh / ThinkingXHigh. Zero defers to
	// Settings.DefaultThinkingLevel.
	ThinkingLevel config.ThinkingLevel

	// Settings is the fully-resolved settings bundle (global merged
	// with project). The caller loads it; the agent never reads disk.
	Settings config.Settings

	// Transport selects the streaming transport. Zero defers to
	// Settings.Transport. The runtime converts to llm.Transport at the
	// request boundary.
	Transport config.TransportSetting

	// SteeringMode governs concurrent tool execution. One of
	// config.SteeringAll or config.SteeringOneAtATime. Zero defers to
	// Settings.SteeringMode.
	SteeringMode config.SteeringMode

	// Tools is the built-in tool set the runtime registers. Plugin
	// tools (if Plugins is non-nil) are merged on top. The slice
	// accepts HeadlessTool so a headless embedder can register tools
	// that do not implement the TUI rendering methods; the TUI falls
	// back to a generic representation for those tools.
	Tools []tools.HeadlessTool

	// Plugins, when non-nil, exposes plugin-provided tools to the
	// runtime. The runtime calls Tools() and Execute() on it; the
	// caller is responsible for SpawnAll and Shutdown coordination.
	Plugins *plugins.Manager

	// LLMClient is the streaming model client. Required. Tests inject
	// a faux provider; production wires the real provider for
	// Settings.Model.
	LLMClient llm.LLMClient

	// StateManager, when non-nil, overrides the factory's default
	// state manager. Used by tests to inject in-memory state.
	StateManager state.Manager

	// TokenCounter, when non-nil, drives the compactor. Defaults to
	// the real BPE counter for Model.
	TokenCounter tokencounter.TokenCounter

	// SessionID, when non-empty, requests resume from an existing
	// session at the configured sessions path. Empty starts fresh.
	SessionID string

	// ContextWindow is the model's context window in tokens, used by
	// the compactor. Zero lets the factory look it up from
	// Settings/ModelsFile (deferred until that integration exists).
	ContextWindow int

	// KnownModels is the flat list of model entries declared in the
	// user's models.json (top-level plus provider-attached). Slash
	// commands and TUI pickers consult this to validate user-supplied
	// model ids and to render listings. Empty when no models.json is
	// configured; callers SHOULD treat "empty" as "no validation
	// enforced, but warn the user the choice is not checked".
	KnownModels []config.KnownModel

	// ProviderAPI is the API family of the wired LLMClient ("anthropic",
	// "openai", ...). The /model slash command consults this to refuse
	// cross-API switches honestly (the runtime cannot rebuild the
	// LLMClient mid-session, so a switch that needs a different API
	// requires a restart). Empty for the faux provider and for tests
	// that don't care about model-switch semantics.
	ProviderAPI config.ModelAPI

	// ConfigDir overrides the global config directory used to locate
	// prompts (<ConfigDir>/agent/prompts, <ConfigDir>/SYSTEM.md). Empty
	// lets the factory call config.ConfigDir(). Tests inject a temp dir.
	ConfigDir string

	// SlashCommands, when non-nil, is the slash-command registry stored
	// on the runtime for SDK-level inspection. The runtime does NOT
	// dispatch commands from this registry; embedder UIs hold the same
	// reference and dispatch themselves. nil means "no registry
	// injected"; the SDK's AgentSession.SlashCommands() inspector falls
	// back to the default built-in registry in that case.
	//
	// The field is typed interface{} (not *slash.Registry) to avoid a
	// circular dependency between agent and slash. The SDK layer
	// type-asserts to *slash.Registry when reading.
	SlashCommands interface{}
}

// resolvedOptions is the post-defaults bundle the runtime actually
// consumes. Created by CreateAgentSessionRuntime from SessionOptions;
// not for public consumption.
type resolvedOptions struct {
	Model         string
	ThinkingLevel config.ThinkingLevel
	Settings      config.Settings
	Transport     config.TransportSetting
	SteeringMode  config.SteeringMode
	ContextWindow int
	KnownModels   []config.KnownModel
	ProviderAPI   config.ModelAPI
	LLMClient     llm.LLMClient
	StateManager  state.Manager
	TokenCounter  tokencounter.TokenCounter
	BuiltinTools  []tools.HeadlessTool
	PluginManager *plugins.Manager
}

// resolve applies defaults from Settings for any optional field that
// the caller left zero. Returns the resolved bundle. Caller must
// already have validated required fields.
func (o SessionOptions) resolve() resolvedOptions {
	r := resolvedOptions{
		Model:         o.Model,
		ThinkingLevel: o.ThinkingLevel,
		Settings:      o.Settings,
		Transport:     o.Transport,
		SteeringMode:  o.SteeringMode,
		ContextWindow: o.ContextWindow,
		KnownModels:   o.KnownModels,
		ProviderAPI:   o.ProviderAPI,
		LLMClient:     o.LLMClient,
		StateManager:  o.StateManager,
		TokenCounter:  o.TokenCounter,
		BuiltinTools:  o.Tools,
		PluginManager: o.Plugins,
	}
	if r.ThinkingLevel == "" && o.Settings.DefaultThinkingLevel != nil {
		r.ThinkingLevel = *o.Settings.DefaultThinkingLevel
	}
	if r.Transport == "" && o.Settings.Transport != nil {
		r.Transport = *o.Settings.Transport
	}
	if r.SteeringMode == "" && o.Settings.SteeringMode != nil {
		r.SteeringMode = *o.Settings.SteeringMode
	}
	return r
}

// validate returns an error if a required field is missing or a value
// is out of range. Called by CreateAgentSessionRuntime before wiring.
func (o SessionOptions) validate() error {
	if o.Model == "" {
		return errOptionsRequired{"Model"}
	}
	if o.LLMClient == nil {
		return errOptionsRequired{"LLMClient"}
	}
	if len(o.Tools) == 0 {
		return errOptionsRequired{"Tools"}
	}
	return nil
}

// errOptionsRequired is the sentinel for missing required SessionOptions
// fields. Fields are listed by name (not value) so the message is stable.
type errOptionsRequired struct{ Field string }

func (e errOptionsRequired) Error() string {
	return "agent: SessionOptions." + e.Field + " is required"
}
