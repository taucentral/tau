package openai

import (
	"encoding/json"
	"testing"

	"github.com/taucentral/tau/internal/llm"
)

func TestUnmarshalResponse_TextOnly(t *testing.T) {
	content := "Hello"
	resp := chatResponse{
		ID:    "chatcmpl-1",
		Model: "gpt-4o",
		Choices: []responseChoice{
			{
				Index:        0,
				FinishReason: "stop",
				Message: responseMessage{
					Role:    "assistant",
					Content: &content,
				},
			},
		},
		Usage: &chunkUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
	}
	msg, err := unmarshalResponse(resp)
	if err != nil {
		t.Fatalf("unmarshalResponse: %v", err)
	}
	if msg.Role != llm.RoleAssistant {
		t.Errorf("Role = %q", msg.Role)
	}
	if msg.Model != "gpt-4o" || msg.ResponseID != "chatcmpl-1" {
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

func TestUnmarshalResponse_ToolCalls(t *testing.T) {
	content := "calling"
	resp := chatResponse{
		ID:    "c",
		Model: "gpt-4o",
		Choices: []responseChoice{
			{
				FinishReason: "tool_calls",
				Message: responseMessage{
					Role:    "assistant",
					Content: &content,
					ToolCalls: []wireToolCall{
						{
							ID:   "call_1",
							Type: "function",
							Function: wireToolFunction{
								Name:      "read",
								Arguments: `{"path":"/x"}`,
							},
						},
					},
				},
			},
		},
	}
	msg, err := unmarshalResponse(resp)
	if err != nil {
		t.Fatalf("unmarshalResponse: %v", err)
	}
	if msg.StopReason != llm.StopReasonToolUse {
		t.Errorf("StopReason = %q", msg.StopReason)
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
	if tu.ID != "call_1" || tu.Name != "read" || string(tu.Input) != `{"path":"/x"}` {
		t.Errorf("ToolUse = %+v", tu)
	}
}

func TestUnmarshalResponse_MissingIDRejected(t *testing.T) {
	content := "x"
	resp := chatResponse{
		Choices: []responseChoice{{
			Message: responseMessage{
				Role:    "assistant",
				Content: &content,
				ToolCalls: []wireToolCall{{
					ID:       "", // missing
					Type:     "function",
					Function: wireToolFunction{Name: "x"},
				}},
			},
		}},
	}
	_, err := unmarshalResponse(resp)
	if err == nil {
		t.Errorf("expected error for missing tool call id")
	}
}

func TestUnmarshalResponse_NoChoicesRejected(t *testing.T) {
	resp := chatResponse{}
	_, err := unmarshalResponse(resp)
	if err == nil {
		t.Errorf("expected error for empty choices")
	}
}

func TestUnmarshalResponse_ReasoningContent(t *testing.T) {
	content := "answer"
	resp := chatResponse{
		ID:    "c",
		Model: "deepseek",
		Choices: []responseChoice{{
			FinishReason: "stop",
			Message: responseMessage{
				Role:             "assistant",
				Content:          &content,
				ReasoningContent: "thinking...",
			},
		}},
	}
	msg, _ := unmarshalResponse(resp)
	if len(msg.Content) != 2 {
		t.Fatalf("Content len = %d", len(msg.Content))
	}
	// Thinking should come before text.
	if _, ok := msg.Content[0].(llm.ThinkingContent); !ok {
		t.Errorf("Content[0] = %T, want ThinkingContent", msg.Content[0])
	}
	if _, ok := msg.Content[1].(llm.TextContent); !ok {
		t.Errorf("Content[1] = %T, want TextContent", msg.Content[1])
	}
}

func TestMarshalUnmarshal_RoundTrip_AssistantWithToolCall(t *testing.T) {
	req := llm.Request{
		Model: "gpt-4o",
		Messages: []llm.Message{
			{
				Role: llm.RoleAssistant,
				Content: []llm.ContentBlock{
					llm.TextContent{Text: "calling"},
					llm.ToolUse{ID: "c1", Name: "read", Input: json.RawMessage(`{"path":"/x"}`)},
				},
			},
		},
	}
	out, _ := marshalRequest(req)
	content := string(out.Messages[0].Content)
	// Trim the JSON quotes.
	content = content[1 : len(content)-1]
	resp := chatResponse{
		ID:    "msg_x",
		Model: "gpt-4o",
		Choices: []responseChoice{{
			FinishReason: "tool_calls",
			Message: responseMessage{
				Role:      "assistant",
				Content:   &content,
				ToolCalls: out.Messages[0].ToolCalls,
			},
		}},
	}
	msg, err := unmarshalResponse(resp)
	if err != nil {
		t.Fatalf("unmarshalResponse: %v", err)
	}
	if len(msg.Content) != 2 {
		t.Fatalf("Content len = %d", len(msg.Content))
	}
	tu, ok := msg.Content[1].(llm.ToolUse)
	if !ok {
		t.Fatalf("Content[1] = %T", msg.Content[1])
	}
	if tu.ID != "c1" || tu.Name != "read" || string(tu.Input) != `{"path":"/x"}` {
		t.Errorf("ToolUse = %+v", tu)
	}
}
