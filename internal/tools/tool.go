// Package tools defines the Tool interface, Registry, and built-in tools.
//
// The Tool interface is the single abstraction used by both built-in tools
// and plugin-exposed tools. Every tool receives a typed ToolCall and returns
// a typed ToolResult; the agent loop is unaware of the tool's internal
// mechanics.
//
// Tools that touch the filesystem or spawn subprocesses expose an Operations
// interface (e.g., ReadOperations, BashOperations). The default
// implementation uses the real OS; tests inject fake implementations to
// avoid side effects.
package tools

import (
	"context"
	"encoding/json"

	"github.com/invopop/jsonschema"

	"github.com/taucentral/tau/internal/llm"
)

// HeadlessTool is the functional contract for a tool: the methods an
// embedder needs to implement in order to plug a tool into the agent loop.
// A headless embedder (batch evaluator, CI bot, daemon) implements only
// HeadlessTool and never has to think about TUI rendering.
//
// Implementations MUST be safe for concurrent use: the agent loop may
// invoke multiple Execute calls on the same HeadlessTool instance in
// parallel when Settings.SteeringMode is "all".
type HeadlessTool interface {
	// Name is the unique tool identifier. Lowercase ASCII, words
	// separated by dots (e.g., "read", "git.status"). The Registry
	// rejects duplicate names.
	Name() string

	// Description is a human- and model-readable summary of what the
	// tool does. Providers embed this verbatim in the tool definition.
	Description() string

	// Parameters returns the input JSON Schema. The schema MUST be
	// draft-2020-12 compliant so the provider can embed it directly.
	Parameters() jsonschema.Schema

	// Execute runs the tool. The context carries cancellation and
	// deadlines; tools MUST honor ctx.Done() promptly.
	//
	// call.Args is the raw JSON arguments payload from the model. Tools
	// validate it against their own Parameters() before running; a
	// validation failure produces a ToolResult with IsError=true and a
	// descriptive message rather than a returned error.
	//
	// A non-nil error is reserved for infrastructure failures (panic,
	// context cancellation, internal bug). Application-level failures
	// (file not found, command exit non-zero, validation error) are
	// returned as ToolResult.IsError=true.
	Execute(ctx context.Context, call ToolCall) (ToolResult, error)
}

// Tool is the contract the TUI consumes: every method on HeadlessTool plus
// the two rendering hooks the TUI needs to display a call and its result.
// Tools that ship as built-ins implement Tool directly so the TUI gets rich
// rendering; headless embedders implement HeadlessTool and the TUI falls
// back to a generic representation when a type assertion to Tool fails.
//
// Every existing Tool implementation satisfies HeadlessTool — the split is
// source-compatible. Callers that need a rendering-aware tool can also
// embed NoRender to satisfy the rendering methods with no-op fallbacks.
type Tool interface {
	HeadlessTool

	// RenderCall produces a human-readable representation of the tool
	// invocation for the TUI. The theme controls color/indentation; the
	// tools package uses ANSI codes directly so it has no TUI dependency.
	RenderCall(args json.RawMessage, theme *Theme) string

	// RenderResult produces a human-readable representation of the
	// tool's output. Used by the TUI when displaying completed tool
	// calls; never sent to the LLM.
	RenderResult(result ToolResult, theme *Theme) string
}

// NoRender is a mixin whose RenderCall and RenderResult return the empty
// string. Embed NoRender in a HeadlessTool to satisfy the Tool interface
// without writing real rendering code; the TUI's generic-render fallback
// kicks in for the empty strings.
//
// Example:
//
//	type myTool struct{ NoRender }
//	func (myTool) Name() string { return "my" }
//	// ... other HeadlessTool methods ...
//	// myTool now satisfies both HeadlessTool and Tool.
type NoRender struct{}

// RenderCall implements Tool by returning the empty string. The TUI
// substitutes its generic representation when the rendered string is empty.
func (NoRender) RenderCall(args json.RawMessage, theme *Theme) string { return "" }

// RenderResult implements Tool by returning the empty string. The TUI
// substitutes its generic representation when the rendered string is empty.
func (NoRender) RenderResult(result ToolResult, theme *Theme) string { return "" }

// Compile-time check: NoRender carries the render method set but is NOT
// itself a Tool (it has no Name/Description/Parameters/Execute). Embedding
// it into a HeadlessTool is what satisfies Tool.
var _ = (Tool)(nil)

// HydrationMode controls how the registry evaluates LazyHeadlessTool
// hydration triggers each turn. See design.md D1.2 for the rationale.
type HydrationMode string

const (
	// HydrationModeHeuristic is the default. The registry evaluates
	// triggers in order: AlwaysRender → Settings.Tools.AlwaysRender →
	// recent use within RecentUseWindow → Intent keyword match.
	HydrationModeHeuristic HydrationMode = "heuristic"

	// HydrationModeModelDeclared reserves a future two-round-trip
	// selection prompt. For v1, the registry treats it the same as
	// HydrationModeOff (everything renders) because the selection
	// round-trip is not yet implemented.
	HydrationModeModelDeclared HydrationMode = "model_declared"

	// HydrationModeOff bypasses the hydration heuristic entirely.
	// LazyHeadlessTool instances render eagerly as if they were
	// HeadlessTool instances.
	HydrationModeOff HydrationMode = "off"
)

// ToolTag carries lightweight metadata used by the hydration heuristic.
// Tools that want tag-based hydration return a ToolTag from Tag();
// everything else renders eagerly.
//
// See docs/sdk/cookbook.md for the recommended Intent vocabulary
// ("filesystem.read", "filesystem.write", "search", "memory.recall",
// "policy.enforce"). Intent is free-form; tau does not prescribe a
// structured vocabulary.
type ToolTag struct {
	// Intent is a free-form string describing when the tool is useful.
	// The registry substring-matches Intent (case-insensitive) against
	// the latest user message as one of the hydration triggers.
	Intent string

	// AlwaysRender, when true, forces the tool to render every turn
	// regardless of hydration mode or other triggers. Useful for tools
	// the model must always see.
	AlwaysRender bool

	// RecentUseWeight adjusts the recency heuristic. Reserved for
	// future use; zero means use Settings.Tools.RecentUseWindow.
	RecentUseWeight float64
}

// TurnSignals packages per-turn data the runtime derives from the state
// tree and settings, then passes to Registry.Schemas. The Registry
// consults these fields when evaluating hydration triggers for each
// LazyHeadlessTool.
//
// The runtime (internal/agent) constructs a TurnSignals value in
// buildRequest before calling Registry.Schemas; embedders never need
// to build one themselves.
type TurnSignals struct {
	// UserMessage is the latest user input. Intent matchers test
	// against this field (case-insensitive substring match).
	UserMessage string

	// RecentToolCalls lists tool names called within the last
	// RecentUseWindow turns. The runtime walks the state tree to
	// derive this list.
	RecentToolCalls []string

	// Mode is the active HydrationMode for this turn, copied from
	// Settings.Tools.HydrationMode.
	Mode HydrationMode

	// AlwaysRender is the per-deployment override list copied from
	// Settings.Tools.AlwaysRender. Tools named here render every turn
	// regardless of the heuristic.
	AlwaysRender []string

	// RecentUseWindow is the maximum turns-since-last-use a tool can
	// have before it falls out of RecentToolCalls. Copied from
	// Settings.Tools.RecentUseWindow.
	RecentUseWindow int
}

// LazyHeadlessTool is an opt-in interface that extends HeadlessTool with
// tag-based hydration. Tools that implement this interface are rendered
// conditionally per turn based on TurnSignals; tools that do not
// implement it render eagerly via the existing HeadlessTool.Parameters()
// path. The runtime type-asserts each registered tool against
// LazyHeadlessTool — no registration change is required.
//
// Hydrate returns the same jsonschema.Schema shape that Parameters()
// returns on eager tools. Eager and lazy tools share the same schema
// representation in Request.Tools. The runtime calls Hydrate when the
// tool matches a hydration trigger OR when a first-call miss occurs
// (the model called a hidden lazy tool).
type LazyHeadlessTool interface {
	HeadlessTool

	// Tag returns a lightweight descriptor used by the runtime to
	// decide whether to render the tool's full schema this turn.
	Tag() ToolTag

	// Hydrate returns the full JSON schema, called by the runtime
	// when the tool is selected for rendering (either by the
	// hydration heuristic or by a first-call miss). The schema MUST
	// match the shape returned by Parameters() on eager tools.
	Hydrate(ctx context.Context) (jsonschema.Schema, error)
}

// ToolCall is the runtime invocation of a tool. The agent loop constructs
// this from a provider-side ToolUse block.
type ToolCall struct {
	// ID is the provider-assigned tool-use id. Echoed in ToolResult so
	// the provider can correlate results with calls.
	ID string `json:"id"`
	// Name is the tool name (matches Tool.Name()).
	Name string `json:"name"`
	// Args is the raw JSON arguments payload from the model. Tools parse
	// this against their own schema.
	Args json.RawMessage `json:"args,omitempty"`
	// Cwd is the working directory the agent loop has chosen for the
	// session. Tools that operate on files resolve relative paths
	// against this; tools that spawn subprocesses chdir here.
	Cwd string `json:"cwd,omitempty"`
}

// ToolResult is the output of a tool execution. The agent loop wraps this
// in an llm.ToolResult block for the LLM context.
type ToolResult struct {
	// Content is the LLM-facing content. Typically a single TextContent;
	// image-returning tools (read on a PNG/JPG) use ImageContent.
	Content []llm.ContentBlock `json:"content"`
	// IsError indicates the tool failed. The provider renders this as a
	// tool-call failure to the model. Set when the tool detected an
	// application-level error (file missing, validation failed,
	// command exit non-zero).
	IsError bool `json:"isError"`
	// Metadata carries tool-specific data for UI rendering (exit codes,
	// line counts, paths). Not sent to the LLM. Keys are tool-defined;
	// see each tool's doc for the shape.
	Metadata map[string]any `json:"metadata,omitempty"`
}

// NewTextResult is a convenience constructor for a successful single-block
// text result. Most tools produce this shape.
func NewTextResult(text string) ToolResult {
	return ToolResult{
		Content: []llm.ContentBlock{llm.TextContent{Text: text}},
	}
}

// NewErrorResult is a convenience constructor for an error result. The text
// describes the failure for the model.
func NewErrorResult(text string) ToolResult {
	return ToolResult{
		Content: []llm.ContentBlock{llm.TextContent{Text: text}},
		IsError: true,
	}
}

// AsLLMResult converts the ToolResult to the llm-package ToolResult shape,
// correlating it with the originating tool-use id. The agent loop uses this
// when appending the result to the state tree.
func (r ToolResult) AsLLMResult(toolUseID string) llm.ToolResult {
	return llm.ToolResult{
		ToolUseID: toolUseID,
		Content:   r.Content,
		IsError:   r.IsError,
	}
}

// ParseArgs is a helper for tools to validate their arguments. It decodes
// raw into dst and returns a descriptive ToolResult on failure (rather than
// an error). The caller writes:
//
//	var args ReadArgs
//	if bad := ParseArgs(call.Args, &args, "read"); bad != nil {
//	    return *bad, nil
//	}
//
// dst MUST be a pointer.
func ParseArgs(raw json.RawMessage, dst any, toolName string) *ToolResult {
	if len(raw) == 0 {
		// Empty args is valid for tools that take no parameters; the
		// caller's struct will retain its zero value.
		return nil
	}
	if err := json.Unmarshal(raw, dst); err != nil {
		r := NewErrorResult(toolName + ": invalid arguments JSON: " + err.Error())
		return &r
	}
	return nil
}
