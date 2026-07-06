package openai

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/taucentral/tau/internal/llm"
)

// feed parses an SSE byte stream via the parser.
func feed(t *testing.T, rawSSE string) []llm.Delta {
	t.Helper()
	p := newStreamParser()
	var deltas []llm.Delta
	r := strings.NewReader(rawSSE)
	err := llm.ReadSSE(r, func(ev llm.SSEEvent) error {
		out, err := p.handle(ev)
		if err != nil {
			return err
		}
		deltas = append(deltas, out...)
		return nil
	})
	if err != nil {
		t.Fatalf("ReadSSE: %v", err)
	}
	return deltas
}

func TestStream_TextOnly(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		``,
		`data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	deltas := feed(t, sse)

	var texts []string
	var sawFinal, sawUsage bool
	for _, d := range deltas {
		switch v := d.(type) {
		case llm.TextDelta:
			texts = append(texts, v.Text)
		case llm.UsageDelta:
			sawUsage = true
			if v.InputTokens != 5 || v.OutputTokens != 2 {
				t.Errorf("Usage = %+v", v)
			}
		case llm.Final:
			sawFinal = true
			if v.StopReason != llm.StopReasonEndTurn {
				t.Errorf("StopReason = %q", v.StopReason)
			}
			if v.ResponseID != "chatcmpl-1" {
				t.Errorf("ResponseID = %q", v.ResponseID)
			}
		}
	}
	if len(texts) != 2 || texts[0] != "Hello" || texts[1] != " world" {
		t.Errorf("texts = %v", texts)
	}
	if !sawUsage || !sawFinal {
		t.Errorf("missing usage/final: usage=%v final=%v", sawUsage, sawFinal)
	}
}

func TestStream_ToolCallFragmented(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"id":"c","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read","arguments":""}}]},"finish_reason":null}]}`,
		``,
		`data: {"id":"c","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path"}}]},"finish_reason":null}]}`,
		``,
		`data: {"id":"c","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\":\"/x\"}"}}]},"finish_reason":null}]}`,
		``,
		`data: {"id":"c","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	deltas := feed(t, sse)

	var fragments []string
	var id, name string
	for _, d := range deltas {
		if td, ok := d.(llm.ToolCallDelta); ok {
			if td.ID != "" {
				id = td.ID
			}
			if td.Name != "" {
				name = td.Name
			}
			fragments = append(fragments, td.PartialInput)
		}
	}
	if id != "call_1" || name != "read" {
		t.Errorf("id=%q name=%q", id, name)
	}
	joined := strings.Join(fragments, "")
	if joined != `{"path":"/x"}` {
		t.Errorf("joined = %q", joined)
	}
	// Validate JSON.
	var parsed map[string]any
	if err := json.Unmarshal([]byte(joined), &parsed); err != nil {
		t.Errorf("invalid JSON: %v", err)
	}
}

func TestStream_MultipleToolCalls(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"id":"c","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"noop","arguments":"{}"}}]},"finish_reason":null}]}`,
		``,
		`data: {"id":"c","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"b","type":"function","function":{"name":"noop","arguments":"{}"}}]},"finish_reason":null}]}`,
		``,
		`data: {"id":"c","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	deltas := feed(t, sse)

	// Should have two distinct tool calls (by id).
	seen := make(map[string]int)
	for _, d := range deltas {
		if td, ok := d.(llm.ToolCallDelta); ok {
			seen[td.ID]++
		}
	}
	if len(seen) != 2 {
		t.Errorf("got %d distinct tool call ids, want 2", len(seen))
	}
}

func TestStream_ToolCallAfterText(t *testing.T) {
	// When text precedes a tool call, tau_index for the tool call must be
	// 1 (text is at 0).
	sse := strings.Join([]string{
		`data: {"id":"c","model":"m","choices":[{"index":0,"delta":{"content":"thinking..."},"finish_reason":null}]}`,
		``,
		`data: {"id":"c","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"x","arguments":""}}]},"finish_reason":null}]}`,
		``,
		`data: {"id":"c","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	deltas := feed(t, sse)

	var textIdx, toolIdx = -1, -1
	for _, d := range deltas {
		switch v := d.(type) {
		case llm.TextDelta:
			textIdx = v.ContentIndex
		case llm.ToolCallDelta:
			toolIdx = v.ContentIndex
		}
	}
	if textIdx != 0 {
		t.Errorf("text ContentIndex = %d, want 0", textIdx)
	}
	if toolIdx != 1 {
		t.Errorf("tool ContentIndex = %d, want 1", toolIdx)
	}
}

func TestStream_ReasoningContent(t *testing.T) {
	// Some compat providers stream reasoning_content (e.g., DeepSeek).
	sse := strings.Join([]string{
		`data: {"id":"c","model":"deepseek","choices":[{"index":0,"delta":{"reasoning_content":"thinking..."},"finish_reason":null}]}`,
		``,
		`data: {"id":"c","model":"deepseek","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":null}]}`,
		``,
		`data: {"id":"c","model":"deepseek","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	deltas := feed(t, sse)

	var sawThinking, sawText bool
	for _, d := range deltas {
		switch v := d.(type) {
		case llm.ThinkingDelta:
			sawThinking = true
			if v.Text != "thinking..." {
				t.Errorf("thinking = %q", v.Text)
			}
		case llm.TextDelta:
			sawText = true
			if v.Text != "answer" {
				t.Errorf("text = %q", v.Text)
			}
		}
	}
	if !sawThinking || !sawText {
		t.Errorf("missing thinking=%v text=%v", sawThinking, sawText)
	}
}

func TestStream_StopReasonMapping(t *testing.T) {
	cases := map[string]llm.StopReason{
		"stop":           llm.StopReasonEndTurn,
		"end":            llm.StopReasonEndTurn,
		"length":         llm.StopReasonLength,
		"function_call":  llm.StopReasonToolUse,
		"tool_calls":     llm.StopReasonToolUse,
		"content_filter": llm.StopReasonError,
		"network_error":  llm.StopReasonError,
	}
	for input, want := range cases {
		if got := mapStopReason(input); got != want {
			t.Errorf("mapStopReason(%q) = %v, want %v", input, got, want)
		}
	}
}

func TestStream_DoneWithoutFinishReason(t *testing.T) {
	// Some compat providers send [DONE] without finish_reason.
	sse := strings.Join([]string{
		`data: {"id":"c","model":"m","choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	deltas := feed(t, sse)
	// Should still see a Final and a (zero) UsageDelta.
	var sawFinal, sawUsage bool
	for _, d := range deltas {
		switch d.(type) {
		case llm.Final:
			sawFinal = true
		case llm.UsageDelta:
			sawUsage = true
		}
	}
	if !sawFinal || !sawUsage {
		t.Errorf("missing final=%v usage=%v", sawFinal, sawUsage)
	}
}

func TestStream_ChunkErrorEmitsFinal(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"error":{"message":"rate limited","type":"rate_limit"}}`,
		``,
	}, "\n")
	deltas := feed(t, sse)
	if len(deltas) != 1 {
		t.Fatalf("deltas len = %d", len(deltas))
	}
	f, ok := deltas[0].(llm.Final)
	if !ok {
		t.Fatalf("type = %T", deltas[0])
	}
	if f.StopReason != llm.StopReasonError || f.Err == nil {
		t.Errorf("Final = %+v", f)
	}
}

func TestStream_AfterDoneSilentlySkips(t *testing.T) {
	// After [DONE] the parser should silently ignore further events.
	p := newStreamParser()
	_, _ = p.handle(llm.SSEEvent{Type: "message", Data: "[DONE]"})
	out, err := p.handle(llm.SSEEvent{Type: "message", Data: `{"choices":[]}`})
	if err != nil {
		t.Errorf("err = %v", err)
	}
	if len(out) != 0 {
		t.Errorf("out = %v, want no deltas", out)
	}
}

func TestStream_FinishReasonBundledWithUsageEmitsFinalImmediately(t *testing.T) {
	// Some OpenAI-compatible providers (and OpenAI without include_usage)
	// emit finish_reason and usage in the same chunk. The parser must emit
	// the Final immediately in that case rather than deferring.
	sse := strings.Join([]string{
		`data: {"id":"c","model":"m","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`,
		``,
		`data: {"id":"c","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	deltas := feed(t, sse)

	var sawUsage, sawFinal bool
	for _, d := range deltas {
		switch v := d.(type) {
		case llm.UsageDelta:
			sawUsage = true
			if v.InputTokens != 2 || v.OutputTokens != 1 {
				t.Errorf("Usage = %+v", v)
			}
		case llm.Final:
			sawFinal = true
			if v.StopReason != llm.StopReasonEndTurn {
				t.Errorf("StopReason = %q", v.StopReason)
			}
		}
	}
	if !sawUsage || !sawFinal {
		t.Errorf("missing usage=%v final=%v (bundled path)", sawUsage, sawFinal)
	}
}
