package anthropic

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/coevin/tau/internal/llm"
)

func TestMarshalRequest_TextOnly(t *testing.T) {
	maxTok := 1024
	temp := 0.7
	req := llm.Request{
		Model:       "claude-opus-4-5",
		MaxTokens:   &maxTok,
		Temperature: &temp,
		System: []llm.ContentBlock{
			llm.TextContent{Text: "You are helpful."},
		},
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.TextContent{Text: "hi"}}},
		},
	}
	out, err := marshalRequest(req)
	if err != nil {
		t.Fatalf("marshalRequest: %v", err)
	}
	if !out.Stream {
		t.Errorf("Stream should be true")
	}
	if out.Model != "claude-opus-4-5" {
		t.Errorf("Model = %q", out.Model)
	}
	if out.MaxTokens != 1024 {
		t.Errorf("MaxTokens = %d", out.MaxTokens)
	}
	if len(out.System) != 1 || out.System[0].Text != "You are helpful." {
		t.Errorf("System = %+v", out.System)
	}
	if len(out.Messages) != 1 {
		t.Fatalf("Messages len = %d", len(out.Messages))
	}
	if out.Messages[0].Role != "user" {
		t.Errorf("Messages[0].Role = %q", out.Messages[0].Role)
	}
	if len(out.Messages[0].Content) != 1 {
		t.Fatalf("Messages[0].Content len = %d", len(out.Messages[0].Content))
	}
	if out.Messages[0].Content[0].Type != "text" || out.Messages[0].Content[0].Text != "hi" {
		t.Errorf("Messages[0].Content[0] = %+v", out.Messages[0].Content[0])
	}
}

func TestMarshalRequest_DefaultMaxTokens(t *testing.T) {
	req := llm.Request{Model: "m"}
	out, err := marshalRequest(req)
	if err != nil {
		t.Fatalf("marshalRequest: %v", err)
	}
	if out.MaxTokens != defaultMaxTokens {
		t.Errorf("MaxTokens = %d, want %d", out.MaxTokens, defaultMaxTokens)
	}
}

func TestMarshalRequest_AssistantMessageWithToolUse(t *testing.T) {
	req := llm.Request{
		Model: "m",
		Messages: []llm.Message{
			{
				Role: llm.RoleAssistant,
				Content: []llm.ContentBlock{
					llm.TextContent{Text: "calling tool"},
					llm.ToolUse{ID: "call_1", Name: "read", Input: json.RawMessage(`{"path":"/x"}`)},
				},
			},
		},
	}
	out, err := marshalRequest(req)
	if err != nil {
		t.Fatalf("marshalRequest: %v", err)
	}
	if len(out.Messages[0].Content) != 2 {
		t.Fatalf("Content len = %d", len(out.Messages[0].Content))
	}
	if out.Messages[0].Content[0].Type != "text" {
		t.Errorf("Content[0].Type = %q", out.Messages[0].Content[0].Type)
	}
	tu := out.Messages[0].Content[1]
	if tu.Type != "tool_use" || tu.ID != "call_1" || tu.Name != "read" {
		t.Errorf("tool_use block = %+v", tu)
	}
	if string(tu.Input) != `{"path":"/x"}` {
		t.Errorf("Input = %s", tu.Input)
	}
}

func TestMarshalRequest_ToolRoleBecomesUserToolResult(t *testing.T) {
	req := llm.Request{
		Model: "m",
		Messages: []llm.Message{
			{
				Role:       llm.RoleTool,
				ToolCallID: "call_1",
				Content: []llm.ContentBlock{
					llm.ToolResult{
						ToolUseID: "call_1",
						Content:   []llm.ContentBlock{llm.TextContent{Text: "file contents"}},
					},
				},
			},
		},
	}
	out, err := marshalRequest(req)
	if err != nil {
		t.Fatalf("marshalRequest: %v", err)
	}
	if len(out.Messages) != 1 {
		t.Fatalf("Messages len = %d", len(out.Messages))
	}
	if out.Messages[0].Role != "user" {
		t.Errorf("Role = %q, want user (tool results as user)", out.Messages[0].Role)
	}
	tr := out.Messages[0].Content[0]
	if tr.Type != "tool_result" || tr.ToolUseID != "call_1" {
		t.Errorf("tool_result block = %+v", tr)
	}
	if len(tr.Content) != 1 || tr.Content[0].Text != "file contents" {
		t.Errorf("tool_result content = %+v", tr.Content)
	}
}

func TestMarshalRequest_ToolResultIsErrorPropagated(t *testing.T) {
	req := llm.Request{
		Model: "m",
		Messages: []llm.Message{
			{
				Role: llm.RoleTool,
				Content: []llm.ContentBlock{
					llm.ToolResult{
						ToolUseID: "c1",
						Content:   []llm.ContentBlock{llm.TextContent{Text: "permission denied"}},
						IsError:   true,
					},
				},
			},
		},
	}
	out, _ := marshalRequest(req)
	if !out.Messages[0].Content[0].IsError {
		t.Errorf("IsError not propagated")
	}
}

func TestMarshalRequest_ImageContent(t *testing.T) {
	req := llm.Request{
		Model: "m",
		Messages: []llm.Message{
			{
				Role: llm.RoleUser,
				Content: []llm.ContentBlock{
					llm.ImageContent{Data: "base64abc", MimeType: "image/png"},
				},
			},
		},
	}
	out, _ := marshalRequest(req)
	b := out.Messages[0].Content[0]
	if b.Type != "image" || b.Source == nil {
		t.Fatalf("image block wrong: %+v", b)
	}
	if b.Source.Type != "base64" || b.Source.MediaType != "image/png" || b.Source.Data != "base64abc" {
		t.Errorf("source = %+v", b.Source)
	}
}

func TestMarshalRequest_ThinkingConfig(t *testing.T) {
	budget := 4096
	req := llm.Request{
		Model:          "m",
		MaxTokens:      &budget,
		ThinkingBudget: &budget,
	}
	out, err := marshalRequest(req)
	if err != nil {
		t.Fatalf("marshalRequest: %v", err)
	}
	if out.Thinking == nil || out.Thinking.Type != "enabled" || out.Thinking.BudgetTokens != 4096 {
		t.Errorf("Thinking = %+v", out.Thinking)
	}
}

func TestMarshalRequest_ThinkingConfigDisabledWhenZero(t *testing.T) {
	zero := 0
	req := llm.Request{Model: "m", ThinkingBudget: &zero}
	out, _ := marshalRequest(req)
	if out.Thinking != nil {
		t.Errorf("Thinking should be nil for budget=0")
	}
}

func TestMarshalRequest_Tools(t *testing.T) {
	req := llm.Request{
		Model: "m",
		Tools: []llm.ToolSchema{
			{
				Name:        "read",
				Description: "Read a file",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`),
			},
		},
	}
	out, _ := marshalRequest(req)
	if len(out.Tools) != 1 {
		t.Fatalf("Tools len = %d", len(out.Tools))
	}
	if out.Tools[0].Name != "read" || out.Tools[0].Description != "Read a file" {
		t.Errorf("Tools[0] = %+v", out.Tools[0])
	}
	if string(out.Tools[0].InputSchema) == "" {
		t.Errorf("InputSchema empty")
	}
}

func TestMarshalRequest_SystemRoleInMessagesRejected(t *testing.T) {
	req := llm.Request{
		Model: "m",
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: []llm.ContentBlock{llm.TextContent{Text: "system"}}},
		},
	}
	_, err := marshalRequest(req)
	if err == nil {
		t.Errorf("expected error for system-role message in messages")
	}
}

func TestMarshalRequest_JSONRoundTrip(t *testing.T) {
	// Verify the marshaled request is valid JSON and produces expected keys.
	maxTok := 1024
	req := llm.Request{
		Model:     "claude-opus-4-5",
		MaxTokens: &maxTok,
		System:    []llm.ContentBlock{llm.TextContent{Text: "system"}},
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.TextContent{Text: "hi"}}}},
	}
	out, _ := marshalRequest(req)
	data, err := json.Marshal(out)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	s := string(data)
	for _, want := range []string{
		`"model":"claude-opus-4-5"`,
		`"stream":true`,
		`"max_tokens":1024`,
		`"system":[{"type":"text","text":"system"}]`,
	} {
		if !strings.Contains(s, want) {
			t.Errorf("missing %q in %s", want, s)
		}
	}
}
