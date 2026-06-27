package llm

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestTextContent_RoundTrip(t *testing.T) {
	tc := TextContent{Text: "hello world"}
	data, err := json.Marshal(tc)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var got TextContent
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if got.Text != tc.Text {
		t.Errorf("Text = %q, want %q", got.Text, tc.Text)
	}
}

func TestTextContent_TypeTag(t *testing.T) {
	tc := TextContent{Text: "x"}
	data, _ := json.Marshal(tc)
	if !strings.Contains(string(data), `"type":"text"`) {
		t.Errorf("missing type tag in %s", data)
	}
}

func TestTextContent_WrongTypeTagRejected(t *testing.T) {
	payload := []byte(`{"type":"image","text":"x"}`)
	var tc TextContent
	if err := json.Unmarshal(payload, &tc); err == nil {
		t.Errorf("expected error for wrong type tag")
	}
}

func TestImageContent_RoundTrip(t *testing.T) {
	ic := ImageContent{Data: "base64data==", MimeType: "image/png"}
	data, err := json.Marshal(ic)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var got ImageContent
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if got.Data != ic.Data || got.MimeType != ic.MimeType {
		t.Errorf("round-trip mismatch: %+v vs %+v", got, ic)
	}
}

func TestThinkingContent_SignatureOmittedWhenEmpty(t *testing.T) {
	tc := ThinkingContent{Thinking: "thought"}
	data, _ := json.Marshal(tc)
	if strings.Contains(string(data), "signature") {
		t.Errorf("empty signature should be omitted: %s", data)
	}
}

func TestThinkingContent_WithSignature(t *testing.T) {
	tc := ThinkingContent{Thinking: "thought", Signature: "sig123", Redacted: true}
	data, _ := json.Marshal(tc)
	var got ThinkingContent
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if got.Signature != tc.Signature || !got.Redacted {
		t.Errorf("round-trip lost fields: %+v", got)
	}
}

func TestToolUse_RoundTrip(t *testing.T) {
	tu := ToolUse{
		ID:    "call_1",
		Name:  "read",
		Input: json.RawMessage(`{"path":"/tmp/x"}`),
	}
	data, err := json.Marshal(tu)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var got ToolUse
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if got.ID != tu.ID || got.Name != tu.Name {
		t.Errorf("round-trip mismatch: %+v", got)
	}
	// Input is raw JSON — compare after re-marshalling both sides.
	gotInput, _ := json.Marshal(got.Input)
	wantInput, _ := json.Marshal(tu.Input)
	if string(gotInput) != string(wantInput) {
		t.Errorf("Input mismatch: %s vs %s", gotInput, wantInput)
	}
}

func TestToolResult_RoundTrip(t *testing.T) {
	tr := ToolResult{
		ToolUseID: "call_1",
		Content: []ContentBlock{
			TextContent{Text: "file contents"},
		},
		IsError: false,
	}
	data, err := json.Marshal(tr)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var got ToolResult
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if got.ToolUseID != tr.ToolUseID {
		t.Errorf("ToolUseID = %q", got.ToolUseID)
	}
	if len(got.Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(got.Content))
	}
	tc, ok := got.Content[0].(TextContent)
	if !ok {
		t.Fatalf("Content[0] type = %T, want TextContent", got.Content[0])
	}
	if tc.Text != "file contents" {
		t.Errorf("Text = %q", tc.Text)
	}
}

func TestToolResult_NestedImageContent(t *testing.T) {
	tr := ToolResult{
		ToolUseID: "call_1",
		Content: []ContentBlock{
			ImageContent{Data: "abc", MimeType: "image/jpeg"},
		},
	}
	data, err := json.Marshal(tr)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var got ToolResult
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	ic, ok := got.Content[0].(ImageContent)
	if !ok {
		t.Fatalf("Content[0] type = %T, want ImageContent", got.Content[0])
	}
	if ic.Data != "abc" || ic.MimeType != "image/jpeg" {
		t.Errorf("round-trip mismatch: %+v", ic)
	}
}

func TestUnmarshalContentBlock_Text(t *testing.T) {
	cb, err := UnmarshalContentBlock([]byte(`{"type":"text","text":"hi"}`))
	if err != nil {
		t.Fatalf("UnmarshalContentBlock: %v", err)
	}
	tc, ok := cb.(TextContent)
	if !ok {
		t.Fatalf("type = %T, want TextContent", cb)
	}
	if tc.Text != "hi" {
		t.Errorf("Text = %q", tc.Text)
	}
}

func TestUnmarshalContentBlock_Image(t *testing.T) {
	cb, err := UnmarshalContentBlock([]byte(`{"type":"image","data":"x","mimeType":"image/png"}`))
	if err != nil {
		t.Fatalf("UnmarshalContentBlock: %v", err)
	}
	if _, ok := cb.(ImageContent); !ok {
		t.Fatalf("type = %T, want ImageContent", cb)
	}
}

func TestUnmarshalContentBlock_Thinking(t *testing.T) {
	cb, err := UnmarshalContentBlock([]byte(`{"type":"thinking","thinking":"..."}`))
	if err != nil {
		t.Fatalf("UnmarshalContentBlock: %v", err)
	}
	if _, ok := cb.(ThinkingContent); !ok {
		t.Fatalf("type = %T, want ThinkingContent", cb)
	}
}

func TestUnmarshalContentBlock_ToolUse(t *testing.T) {
	cb, err := UnmarshalContentBlock([]byte(`{"type":"toolUse","id":"c","name":"n","input":{}}`))
	if err != nil {
		t.Fatalf("UnmarshalContentBlock: %v", err)
	}
	if _, ok := cb.(ToolUse); !ok {
		t.Fatalf("type = %T, want ToolUse", cb)
	}
}

func TestUnmarshalContentBlock_ToolResult(t *testing.T) {
	cb, err := UnmarshalContentBlock([]byte(`{"type":"toolResult","toolUseId":"c","content":[{"type":"text","text":"ok"}]}`))
	if err != nil {
		t.Fatalf("UnmarshalContentBlock: %v", err)
	}
	tr, ok := cb.(ToolResult)
	if !ok {
		t.Fatalf("type = %T, want ToolResult", cb)
	}
	if len(tr.Content) != 1 {
		t.Errorf("Content len = %d, want 1", len(tr.Content))
	}
}

func TestUnmarshalContentBlock_UnknownType(t *testing.T) {
	_, err := UnmarshalContentBlock([]byte(`{"type":"bogus"}`))
	if err == nil {
		t.Fatal("expected error for unknown type")
	}
	if !strings.Contains(err.Error(), "unknown content block type") {
		t.Errorf("err = %v", err)
	}
}

func TestUnmarshalContentBlock_Empty(t *testing.T) {
	_, err := UnmarshalContentBlock([]byte(``))
	if err == nil {
		t.Fatal("expected error for empty input")
	}
}

func TestMessage_RoundTrip(t *testing.T) {
	now := time.Date(2025, 6, 15, 12, 30, 0, 0, time.UTC)
	msg := Message{
		Role: RoleAssistant,
		Content: []ContentBlock{
			TextContent{Text: "I'll read the file"},
			ThinkingContent{Thinking: "User wants to read /tmp/x", Signature: "sig"},
			ToolUse{ID: "call_1", Name: "read", Input: json.RawMessage(`{"path":"/tmp/x"}`)},
		},
		Model:      "claude-opus-4-5",
		ProviderID: "anthropic",
		ResponseID: "msg_abc",
		Usage: &Usage{
			Input:       100,
			Output:      50,
			CacheRead:   0,
			CacheWrite:  100,
			TotalTokens: 250,
			Cost:        Cost{Input: 0.001, Output: 0.002, Total: 0.003},
		},
		StopReason: StopReasonToolUse,
		Timestamp:  now,
	}
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	var got Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if got.Role != RoleAssistant {
		t.Errorf("Role = %q", got.Role)
	}
	if len(got.Content) != 3 {
		t.Fatalf("Content len = %d, want 3", len(got.Content))
	}
	if _, ok := got.Content[0].(TextContent); !ok {
		t.Errorf("Content[0] type = %T", got.Content[0])
	}
	if _, ok := got.Content[1].(ThinkingContent); !ok {
		t.Errorf("Content[1] type = %T", got.Content[1])
	}
	tu, ok := got.Content[2].(ToolUse)
	if !ok {
		t.Fatalf("Content[2] type = %T", got.Content[2])
	}
	if tu.ID != "call_1" || tu.Name != "read" {
		t.Errorf("ToolUse = %+v", tu)
	}
	if got.Model != "claude-opus-4-5" {
		t.Errorf("Model = %q", got.Model)
	}
	if got.Usage == nil || got.Usage.Input != 100 || got.Usage.TotalTokens != 250 {
		t.Errorf("Usage mismatch: %+v", got.Usage)
	}
	if got.StopReason != StopReasonToolUse {
		t.Errorf("StopReason = %q", got.StopReason)
	}
	if !got.Timestamp.Equal(now) {
		t.Errorf("Timestamp = %v, want %v", got.Timestamp, now)
	}
}

func TestMessage_ToolRoleRoundTrip(t *testing.T) {
	msg := Message{
		Role:       RoleTool,
		ToolCallID: "call_1",
		ToolName:   "read",
		Content: []ContentBlock{
			TextContent{Text: "file contents here"},
		},
		Timestamp: time.Now(),
	}
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var got Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if got.Role != RoleTool {
		t.Errorf("Role = %q", got.Role)
	}
	if got.ToolCallID != "call_1" || got.ToolName != "read" {
		t.Errorf("tool call id/name lost: %+v", got)
	}
}

func TestMessage_SystemRoleRoundTrip(t *testing.T) {
	msg := Message{
		Role: RoleSystem,
		Content: []ContentBlock{
			TextContent{Text: "You are a coding agent."},
		},
		Timestamp: time.Now(),
	}
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var got Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if got.Role != RoleSystem {
		t.Errorf("Role = %q", got.Role)
	}
	if len(got.Content) != 1 {
		t.Fatalf("Content len = %d", len(got.Content))
	}
}

func TestMessage_EmptyContent(t *testing.T) {
	msg := Message{
		Role:      RoleUser,
		Content:   nil,
		Timestamp: time.Now(),
	}
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var got Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if len(got.Content) != 0 {
		t.Errorf("Content len = %d, want 0", len(got.Content))
	}
}

func TestContentBlock_SealedInterface(t *testing.T) {
	// Compile-time check: only types in this package implement isContentBlock().
	var _ ContentBlock = TextContent{}
	var _ ContentBlock = ImageContent{}
	var _ ContentBlock = ThinkingContent{}
	var _ ContentBlock = ToolUse{}
	var _ ContentBlock = ToolResult{}
}

func TestStopReason_Constants(t *testing.T) {
	cases := []struct {
		got, want StopReason
	}{
		{StopReasonEndTurn, "stop"},
		{StopReasonLength, "length"},
		{StopReasonToolUse, "toolUse"},
		{StopReasonError, "error"},
		{StopReasonAborted, "aborted"},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("constant = %q, want %q", c.got, c.want)
		}
	}
}

func TestRole_Constants(t *testing.T) {
	cases := []struct {
		got, want Role
	}{
		{RoleSystem, "system"},
		{RoleUser, "user"},
		{RoleAssistant, "assistant"},
		{RoleTool, "tool"},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("constant = %q, want %q", c.got, c.want)
		}
	}
}
