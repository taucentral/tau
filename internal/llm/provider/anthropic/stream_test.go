package anthropic

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/taucentral/tau/internal/llm"
)

// feedEvents parses a sequence of SSE event blocks (as they'd appear on the
// wire, minus the HTTP framing) and feeds them to a fresh parser. Returns
// all emitted deltas.
func feedEvents(t *testing.T, rawSSE string) []llm.Delta {
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
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"claude-opus-4-5","usage":{"input_tokens":10,"output_tokens":0}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	deltas := feedEvents(t, sse)

	// Filter by type — we want: TextDelta x2, UsageDelta, Final
	var texts []string
	var sawUsage, sawFinal bool
	for _, d := range deltas {
		switch v := d.(type) {
		case llm.TextDelta:
			texts = append(texts, v.Text)
		case llm.UsageDelta:
			sawUsage = true
			if v.InputTokens != 10 || v.OutputTokens != 2 {
				t.Errorf("Usage = %+v", v)
			}
		case llm.Final:
			sawFinal = true
			if v.StopReason != llm.StopReasonEndTurn {
				t.Errorf("StopReason = %q", v.StopReason)
			}
			if v.ResponseID != "msg_1" {
				t.Errorf("ResponseID = %q", v.ResponseID)
			}
			if v.ResponseModel != "claude-opus-4-5" {
				t.Errorf("ResponseModel = %q", v.ResponseModel)
			}
		}
	}
	if len(texts) != 2 || texts[0] != "Hello" || texts[1] != " world" {
		t.Errorf("texts = %v", texts)
	}
	if !sawUsage {
		t.Errorf("UsageDelta missing")
	}
	if !sawFinal {
		t.Errorf("Final missing")
	}
}

func TestStream_ToolUseAssembled(t *testing.T) {
	sse := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":5,"output_tokens":0}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call_1","name":"read","input":{}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"/tmp/x\"}"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":10}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	deltas := feedEvents(t, sse)

	var fragments []string
	var id, name string
	var sawFinal bool
	for _, d := range deltas {
		switch v := d.(type) {
		case llm.ToolCallDelta:
			if v.ID != "" {
				id = v.ID
			}
			if v.Name != "" {
				name = v.Name
			}
			fragments = append(fragments, v.PartialInput)
		case llm.Final:
			sawFinal = true
			if v.StopReason != llm.StopReasonToolUse {
				t.Errorf("StopReason = %q", v.StopReason)
			}
		}
	}
	if id != "call_1" || name != "read" {
		t.Errorf("id=%q name=%q", id, name)
	}
	joined := strings.Join(fragments, "")
	if joined != `{"path":"/tmp/x"}` {
		t.Errorf("joined = %q", joined)
	}
	if !sawFinal {
		t.Errorf("Final missing")
	}

	// Verify the assembled JSON is valid.
	var parsed map[string]any
	if err := json.Unmarshal([]byte(joined), &parsed); err != nil {
		t.Errorf("invalid JSON: %v", err)
	}
}

func TestStream_ThinkingBlocks(t *testing.T) {
	sse := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":1,"output_tokens":0}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"reasoning..."}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_123"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	deltas := feedEvents(t, sse)

	var thinkingText, signature string
	for _, d := range deltas {
		switch v := d.(type) {
		case llm.ThinkingDelta:
			if v.Text != "" {
				thinkingText += v.Text
			}
			if v.Signature != "" {
				signature = v.Signature
			}
		}
	}
	if thinkingText != "reasoning..." {
		t.Errorf("thinkingText = %q", thinkingText)
	}
	if signature != "sig_123" {
		t.Errorf("signature = %q", signature)
	}
}

func TestStream_StopReasonMapping(t *testing.T) {
	cases := map[string]llm.StopReason{
		"end_turn":      llm.StopReasonEndTurn,
		"stop_sequence": llm.StopReasonEndTurn,
		"pause_turn":    llm.StopReasonEndTurn,
		"max_tokens":    llm.StopReasonLength,
		"tool_use":      llm.StopReasonToolUse,
		"refusal":       llm.StopReasonError,
		"sensitive":     llm.StopReasonError,
	}
	for input, want := range cases {
		if got := mapStopReason(input); got != want {
			t.Errorf("mapStopReason(%q) = %v, want %v", input, got, want)
		}
	}
}

func TestStream_RedactedThinking(t *testing.T) {
	sse := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"m","model":"m","usage":{"input_tokens":1,"output_tokens":0}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"opaque"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	deltas := feedEvents(t, sse)
	var sawRedacted bool
	for _, d := range deltas {
		if td, ok := d.(llm.ThinkingDelta); ok && td.Text == "[redacted]" {
			sawRedacted = true
		}
	}
	if !sawRedacted {
		t.Errorf("redacted thinking stand-in missing")
	}
}

func TestStream_StreamError(t *testing.T) {
	sse := strings.Join([]string{
		`event: error`,
		`data: {"error":{"type":"overloaded_error","message":"Overloaded"}}`,
		``,
	}, "\n")
	deltas := feedEvents(t, sse)
	if len(deltas) != 1 {
		t.Fatalf("deltas len = %d, want 1", len(deltas))
	}
	f, ok := deltas[0].(llm.Final)
	if !ok {
		t.Fatalf("delta type = %T", deltas[0])
	}
	if f.StopReason != llm.StopReasonError {
		t.Errorf("StopReason = %q", f.StopReason)
	}
	if f.Err == nil {
		t.Errorf("Err is nil")
	}
}

func TestStream_PingIgnored(t *testing.T) {
	sse := strings.Join([]string{
		`event: ping`,
		`data: {}`,
		``,
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"m","model":"m","usage":{"input_tokens":1,"output_tokens":0}}}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	deltas := feedEvents(t, sse)
	// Should only see UsageDelta + Final (no text deltas from ping).
	for _, d := range deltas {
		if td, ok := d.(llm.TextDelta); ok && td.Text != "" {
			t.Errorf("ping produced text delta: %+v", td)
		}
	}
}

func TestStream_AfterMessageStopRejectsFurtherEvents(t *testing.T) {
	p := newStreamParser()
	_, _ = p.handle(llm.SSEEvent{Type: "message_stop", Data: `{"type":"message_stop"}`})
	_, err := p.handle(llm.SSEEvent{Type: "message_stop", Data: `{"type":"message_stop"}`})
	if !errors.Is(err, ErrStreamClosed) {
		t.Errorf("err = %v, want ErrStreamClosed", err)
	}
}
