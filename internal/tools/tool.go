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

	"github.com/coevin/tau/internal/llm"
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
