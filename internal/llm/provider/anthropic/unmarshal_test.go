package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/coevin/tau/internal/llm"
)

func TestUnmarshalResponse_TextOnly(t *testing.T) {
	resp := wireResponse{
		ID:         "msg_abc",
		Model:      "claude-opus-4-5",
		StopReason: "end_turn",
		Role:       "assistant",
		Content: []wireBlock{
			{Type: "text", Text: "Hello"},
		},
		Usage: wireResponseUsage{
			InputTokens:  10,
			OutputTokens: 5,
		},
	}
	msg, err := unmarshalResponse(resp)
	if err != nil {
		t.Fatalf("unmarshalResponse: %v", err)
	}
	if msg.Role != llm.RoleAssistant {
		t.Errorf("Role = %q", msg.Role)
	}
	if msg.Model != "claude-opus-4-5" || msg.ResponseID != "msg_abc" {
		t.Errorf("Model/ResponseID = %q/%q", msg.Model, msg.ResponseID)
	}
	if msg.StopReason != llm.StopReasonEndTurn {
		t.Errorf("StopReason = %q", msg.StopReason)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("Content len = %d", len(msg.Content))
	}
	tc, ok := msg.Content[0].(llm.TextContent)
	if !ok || tc.Text != "Hello" {
		t.Errorf("Content[0] = %+v", msg.Content[0])
	}
	if msg.Usage == nil || msg.Usage.Input != 10 || msg.Usage.Output != 5 {
		t.Errorf("Usage = %+v", msg.Usage)
	}
}

func TestUnmarshalResponse_ToolUse(t *testing.T) {
	resp := wireResponse{
		ID:         "msg_1",
		Model:      "claude-opus-4-5",
		StopReason: "tool_use",
		Content: []wireBlock{
			{Type: "tool_use", ID: "call_1", Name: "read", Input: json.RawMessage(`{"path":"/x"}`)},
		},
	}
	msg, err := unmarshalResponse(resp)
	if err != nil {
		t.Fatalf("unmarshalResponse: %v", err)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("Content len = %d", len(msg.Content))
	}
	tu, ok := msg.Content[0].(llm.ToolUse)
	if !ok {
		t.Fatalf("Content[0] type = %T", msg.Content[0])
	}
	if tu.ID != "call_1" || tu.Name != "read" {
		t.Errorf("ToolUse = %+v", tu)
	}
	if string(tu.Input) != `{"path":"/x"}` {
		t.Errorf("Input = %s", tu.Input)
	}
}

func TestUnmarshalResponse_ToolUseMissingInputDefaultsToEmptyJSON(t *testing.T) {
	resp := wireResponse{
		Content: []wireBlock{{Type: "tool_use", ID: "c", Name: "noop"}},
	}
	msg, _ := unmarshalResponse(resp)
	tu, _ := msg.Content[0].(llm.ToolUse)
	if string(tu.Input) != "{}" {
		t.Errorf("Input = %s, want {}", tu.Input)
	}
}

func TestUnmarshalResponse_ThinkingWithSignature(t *testing.T) {
	resp := wireResponse{
		Content: []wireBlock{
			{Type: "thinking", Thinking: "reasoning", Signature: "sig_123"},
		},
	}
	msg, _ := unmarshalResponse(resp)
	tc, ok := msg.Content[0].(llm.ThinkingContent)
	if !ok {
		t.Fatalf("Content[0] type = %T", msg.Content[0])
	}
	if tc.Thinking != "reasoning" || tc.Signature != "sig_123" {
		t.Errorf("ThinkingContent = %+v", tc)
	}
}

func TestUnmarshalResponse_RedactedThinking(t *testing.T) {
	resp := wireResponse{
		Content: []wireBlock{
			{Type: "redacted_thinking", Signature: "opaque"},
		},
	}
	msg, _ := unmarshalResponse(resp)
	tc, ok := msg.Content[0].(llm.ThinkingContent)
	if !ok {
		t.Fatalf("Content[0] type = %T", msg.Content[0])
	}
	if !tc.Redacted {
		t.Errorf("Redacted should be true")
	}
	if tc.Signature != "opaque" {
		t.Errorf("Signature = %q", tc.Signature)
	}
}

func TestUnmarshalResponse_Image(t *testing.T) {
	resp := wireResponse{
		Content: []wireBlock{
			{Type: "image", Source: &wireImageSource{Type: "base64", MediaType: "image/png", Data: "abc"}},
		},
	}
	msg, _ := unmarshalResponse(resp)
	ic, ok := msg.Content[0].(llm.ImageContent)
	if !ok {
		t.Fatalf("Content[0] type = %T", msg.Content[0])
	}
	if ic.Data != "abc" || ic.MimeType != "image/png" {
		t.Errorf("ImageContent = %+v", ic)
	}
}

func TestUnmarshalResponse_ImageMissingSource(t *testing.T) {
	resp := wireResponse{
		Content: []wireBlock{{Type: "image"}},
	}
	_, err := unmarshalResponse(resp)
	if err == nil {
		t.Errorf("expected error for image without source")
	}
}

func TestUnmarshalResponse_ToolResult(t *testing.T) {
	resp := wireResponse{
		Content: []wireBlock{
			{
				Type:      "tool_result",
				ToolUseID: "call_1",
				Content:   []wireBlock{{Type: "text", Text: "file"}},
			},
		},
	}
	msg, _ := unmarshalResponse(resp)
	tr, ok := msg.Content[0].(llm.ToolResult)
	if !ok {
		t.Fatalf("Content[0] type = %T", msg.Content[0])
	}
	if tr.ToolUseID != "call_1" {
		t.Errorf("ToolUseID = %q", tr.ToolUseID)
	}
	if len(tr.Content) != 1 {
		t.Fatalf("nested Content len = %d", len(tr.Content))
	}
}

func TestUnmarshalResponse_UnknownBlockType(t *testing.T) {
	resp := wireResponse{
		Content: []wireBlock{{Type: "bogus"}},
	}
	_, err := unmarshalResponse(resp)
	if err == nil {
		t.Errorf("expected error for unknown block type")
	}
}

// Marshal-then-unmarshal round-trip: verify the wire shapes produced by
// marshal.go parse cleanly via unmarshal.go.
func TestMarshalUnmarshal_RoundTrip_AssistantMessage(t *testing.T) {
	req := llm.Request{
		Model: "m",
		Messages: []llm.Message{
			{
				Role: llm.RoleAssistant,
				Content: []llm.ContentBlock{
					llm.TextContent{Text: "thinking about it"},
					llm.ToolUse{ID: "c1", Name: "read", Input: json.RawMessage(`{"path":"/x"}`)},
				},
			},
		},
	}
	out, _ := marshalRequest(req)
	// Take the marshaled assistant message and convert to a wireResponse
	// shape that unmarshalResponse can consume.
	if len(out.Messages) != 1 {
		t.Fatalf("Messages len = %d", len(out.Messages))
	}
	resp := wireResponse{
		ID:         "msg_x",
		Model:      "m",
		StopReason: "tool_use",
		Content:    out.Messages[0].Content,
	}
	msg, err := unmarshalResponse(resp)
	if err != nil {
		t.Fatalf("unmarshalResponse: %v", err)
	}
	if len(msg.Content) != 2 {
		t.Fatalf("Content len = %d", len(msg.Content))
	}
	if _, ok := msg.Content[0].(llm.TextContent); !ok {
		t.Errorf("Content[0] = %T", msg.Content[0])
	}
	tu, ok := msg.Content[1].(llm.ToolUse)
	if !ok {
		t.Fatalf("Content[1] = %T", msg.Content[1])
	}
	if tu.ID != "c1" || tu.Name != "read" || string(tu.Input) != `{"path":"/x"}` {
		t.Errorf("ToolUse = %+v", tu)
	}
}
