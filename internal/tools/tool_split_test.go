// tool_split_test.go — verifies the Tool / HeadlessTool / NoRender split.
//
// The split is source-compatible (every existing Tool satisfies
// HeadlessTool). These tests exercise the three new paths:
//
//   (a) a struct implementing only HeadlessTool is registered, looked up,
//       and executed end-to-end via the same Registry the agent loop uses.
//   (b) a struct embedding NoRender satisfies Tool (compile-time) and
//       returns the empty string from both render methods at runtime.
//   (c) the generic-render fallback produces a non-empty string for a
//       HeadlessTool-only value and for an error and a success result.
package tools

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/invopop/jsonschema"

	"github.com/taucentral/tau/internal/llm"
)

// headlessOnly implements HeadlessTool and deliberately does NOT satisfy
// Tool (no RenderCall/RenderResult methods). Used to exercise (a) and the
// "type assertion to Tool fails" half of (c).
type headlessOnly struct {
	name string
}

func (h headlessOnly) Name() string                 { return h.name }
func (h headlessOnly) Description() string           { return "headless-only test tool" }
func (h headlessOnly) Parameters() jsonschema.Schema { return jsonschema.Schema{} }
func (h headlessOnly) Execute(ctx context.Context, call ToolCall) (ToolResult, error) {
	return NewTextResult("headless-only executed: " + string(call.Args)), nil
}

// Verify headlessOnly does NOT satisfy Tool at compile time. (This is a
// negative compile-time check — if headlessOnly ever accidentally picks up
// Render methods, this line would still compile and the test would need to
// switch to a different sentinel. The line below IS valid Go: an interface
// assertion against a type that lacks the methods just doesn't hold at
// runtime; the compile-time check happens in the test body via reflection.)
var _ HeadlessTool = headlessOnly{}

// noRenderTool embeds NoRender so it satisfies Tool without writing real
// render methods.
type noRenderTool struct {
	NoRender
	name string
}

func (t noRenderTool) Name() string                 { return t.name }
func (t noRenderTool) Description() string           { return "no-render test tool" }
func (t noRenderTool) Parameters() jsonschema.Schema { return jsonschema.Schema{} }
func (t noRenderTool) Execute(ctx context.Context, call ToolCall) (ToolResult, error) {
	return NewTextResult("no-render executed"), nil
}

// Verify noRenderTool satisfies BOTH interfaces at compile time.
var (
	_ HeadlessTool = noRenderTool{}
	_ Tool         = noRenderTool{}
)

func TestHeadlessToolRegisteredAndExecuted(t *testing.T) {
	// (a) a struct implementing only HeadlessTool is registered and
	// executed end-to-end via the same Registry the agent loop uses.
	r := NewRegistry()
	h := headlessOnly{name: "headless-only"}
	if err := r.Register(h); err != nil {
		t.Fatalf("Register(headlessOnly): %v", err)
	}

	got, err := r.Lookup("headless-only")
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	if _, ok := got.(Tool); ok {
		t.Fatalf("headlessOnly should NOT satisfy Tool; type assertion succeeded")
	}

	res, err := got.Execute(context.Background(), ToolCall{
		ID:   "1",
		Name: "headless-only",
		Args: json.RawMessage(`{"k":"v"}`),
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if len(res.Content) == 0 {
		t.Fatal("Execute returned empty content")
	}
	tc, ok := res.Content[0].(llm.TextContent)
	if !ok {
		t.Fatalf("Execute content[0] type = %T, want TextContent", res.Content[0])
	}
	if !strings.Contains(tc.Text, "headless-only executed") {
		t.Errorf("Execute output = %q, want substring 'headless-only executed'", tc.Text)
	}
}

func TestNoRenderSatisfiesTool(t *testing.T) {
	// (b) a struct embedding NoRender satisfies Tool and returns the
	// empty string from both render methods at runtime.
	tool := noRenderTool{name: "no-render"}

	// Compile-time check: tool satisfies Tool (the var _ above already
	// asserts this at package scope, but we exercise it again here for
	// the type assertion).
	var asTool Tool = tool
	_ = asTool

	args := json.RawMessage(`{"foo":"bar"}`)
	if got := tool.RenderCall(args, PlainTheme()); got != "" {
		t.Errorf("NoRender.RenderCall = %q, want empty", got)
	}
	res := NewTextResult("ok")
	if got := tool.RenderResult(res, PlainTheme()); got != "" {
		t.Errorf("NoRender.RenderResult = %q, want empty", got)
	}
}

func TestGenericRenderFallbackNonEmpty(t *testing.T) {
	// (c) the generic-render fallback produces a non-empty string.
	t.Run("headless-only call", func(t *testing.T) {
		h := headlessOnly{name: "headless-only"}
		got := RenderCallOrFallback(h, json.RawMessage(`{"x":1}`), PlainTheme())
		if got == "" {
			t.Fatal("RenderCallOrFallback returned empty for headless-only")
		}
		if !strings.Contains(got, "headless-only") {
			t.Errorf("output %q missing tool name", got)
		}
		if !strings.Contains(got, `"x":1`) {
			t.Errorf("output %q missing args", got)
		}
	})

	t.Run("headless-only result success", func(t *testing.T) {
		h := headlessOnly{name: "headless-only"}
		res := NewTextResult("done")
		got := RenderResultOrFallback(h, res, PlainTheme())
		if got == "" {
			t.Fatal("RenderResultOrFallback returned empty for success")
		}
		if !strings.Contains(got, "ok") {
			t.Errorf("output %q missing 'ok' marker", got)
		}
		if !strings.Contains(got, "done") {
			t.Errorf("output %q missing content", got)
		}
	})

	t.Run("headless-only result error", func(t *testing.T) {
		h := headlessOnly{name: "headless-only"}
		res := ToolResult{
			Content: []llm.ContentBlock{llm.TextContent{Text: "boom"}},
			IsError: true,
		}
		got := RenderResultOrFallback(h, res, PlainTheme())
		if got == "" {
			t.Fatal("RenderResultOrFallback returned empty for error")
		}
		if !strings.Contains(got, "error") {
			t.Errorf("output %q missing 'error' marker", got)
		}
	})

	t.Run("noRender falls back to generic", func(t *testing.T) {
		tool := noRenderTool{name: "no-render"}
		// RenderCall returns "" so the fallback must kick in.
		got := RenderCallOrFallback(tool, json.RawMessage(`{}`), PlainTheme())
		if got == "" {
			t.Fatal("RenderCallOrFallback returned empty for NoRender-embedded tool")
		}
		if !strings.Contains(got, "no-render") {
			t.Errorf("output %q missing tool name", got)
		}
	})

	t.Run("real Tool wins over fallback", func(t *testing.T) {
		// A real Tool (the built-in readTool) MUST return its own
		// rendering, not the generic fallback.
		tool := NewReadTool(OSReadOperations{})
		got := RenderCallOrFallback(tool, json.RawMessage(`{"path":"x.txt"}`), PlainTheme())
		if got == "" {
			t.Fatal("RenderCallOrFallback returned empty for readTool")
		}
		// The generic fallback prepends "tool "; the real renderer does
		// not. If the output starts with "tool ", the fallback fired
		// incorrectly.
		if strings.HasPrefix(got, "tool read: ") {
			t.Errorf("fallback fired for readTool; output = %q", got)
		}
	})
}

func TestHeadlessOnlyRejectsToolAssertion(t *testing.T) {
	// Sanity: a HeadlessTool-only type must NOT satisfy Tool at runtime.
	// This guards the agent-loop assumption: when the TUI type-asserts, it
	// must be able to fall through to the generic renderer.
	h := headlessOnly{name: "x"}
	if _, ok := interface{}(h).(Tool); ok {
		t.Fatal("headlessOnly unexpectedly satisfies Tool; the split is broken")
	}
	// Ensure the registry-stored value retains this property.
	r := NewRegistry()
	_ = r.Register(h)
	got, _ := r.Lookup("x")
	if _, ok := got.(Tool); ok {
		t.Fatal("registry-stored headlessOnly unexpectedly satisfies Tool")
	}
}

// errors.Is check that the registry still returns the typed sentinel for
// duplicates and unknown names after the HeadlessTool widening.
func TestRegistryHeadlessSentinels(t *testing.T) {
	r := NewRegistry()
	h := headlessOnly{name: "dup"}
	if err := r.Register(h); err != nil {
		t.Fatalf("first Register: %v", err)
	}
	err := r.Register(headlessOnly{name: "dup"})
	if !errors.Is(err, ErrDuplicateTool) {
		t.Errorf("second Register err = %v, want ErrDuplicateTool", err)
	}
	if _, err := r.Lookup("missing"); !errors.Is(err, ErrUnknownTool) {
		t.Errorf("Lookup(missing) err = %v, want ErrUnknownTool", err)
	}
}
