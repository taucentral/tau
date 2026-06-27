// render.go — generic-render fallback for HeadlessTool.
//
// When the TUI needs to render a tool call or result but the registered
// tool does not satisfy Tool (i.e., it is a HeadlessTool-only
// implementation), it falls back to these helpers. The output is a short,
// always-non-empty summary derived from the tool's Name, Parameters, and
// the call's raw arguments or the result's content blocks. The TUI uses
// the same generic representation for built-in tools whose RenderCall or
// RenderResult returns the empty string (e.g., a tool that embeds NoRender).
//
// The helpers never return the empty string: callers can rely on the
// output being safe to display in a tool-bubble viewport.

package tools

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/coevin/tau/internal/llm"
)

// GenericRenderCall produces a non-empty, plain-text representation of a
// tool invocation when the registered tool does not supply its own
// RenderCall. The output is "tool <name>: <args>" where <args> is the raw
// JSON arguments payload verbatim (or "<no args>" when args is empty).
// Callers SHOULD pass the result through their theme wrapper; the helper
// itself emits no ANSI codes.
func GenericRenderCall(headless HeadlessTool, args json.RawMessage) string {
	if headless == nil {
		return "tool <nil>"
	}
	name := headless.Name()
	if name == "" {
		name = "<unnamed>"
	}
	if len(args) == 0 {
		return fmt.Sprintf("tool %s: <no args>", name)
	}
	compact := strings.TrimSpace(string(args))
	if compact == "" {
		compact = "<empty args>"
	}
	return fmt.Sprintf("tool %s: %s", name, compact)
}

// GenericRenderResult produces a non-empty, plain-text representation of a
// tool's output when the registered tool does not supply its own
// RenderResult. The output mirrors the success/error status, includes the
// tool's name when supplied, and joins every TextContent block with a
// separator. Non-text content blocks are summarised by their Go type name.
func GenericRenderResult(headless HeadlessTool, result ToolResult) string {
	name := ""
	if headless != nil {
		name = headless.Name()
	}
	prefix := "tool"
	if name != "" {
		prefix = "tool " + name
	}
	if result.IsError {
		prefix += " (error)"
	} else {
		prefix += " (ok)"
	}
	if len(result.Content) == 0 {
		return prefix + ": <no content>"
	}
	parts := make([]string, 0, len(result.Content))
	for _, block := range result.Content {
		switch v := block.(type) {
		case llm.TextContent:
			text := strings.TrimSpace(v.Text)
			if text == "" {
				text = "<empty text>"
			}
			parts = append(parts, text)
		default:
			parts = append(parts, fmt.Sprintf("<%T>", block))
		}
	}
	return prefix + ": " + strings.Join(parts, " | ")
}

// RenderCallOrFallback invokes t.RenderCall when t satisfies Tool and
// returns a non-empty string; otherwise it falls back to GenericRenderCall.
// TUI consumers SHOULD prefer this helper over a raw type assertion so
// they handle both the "not a Tool" and "is a Tool but returned empty"
// cases with one call.
func RenderCallOrFallback(headless HeadlessTool, args json.RawMessage, theme *Theme) string {
	if t, ok := headless.(Tool); ok {
		if s := t.RenderCall(args, theme); s != "" {
			return s
		}
	}
	return GenericRenderCall(headless, args)
}

// RenderResultOrFallback invokes t.RenderResult when t satisfies Tool and
// returns a non-empty string; otherwise it falls back to GenericRenderResult.
// TUI consumers SHOULD prefer this helper over a raw type assertion.
func RenderResultOrFallback(headless HeadlessTool, result ToolResult, theme *Theme) string {
	if t, ok := headless.(Tool); ok {
		if s := t.RenderResult(result, theme); s != "" {
			return s
		}
	}
	return GenericRenderResult(headless, result)
}
