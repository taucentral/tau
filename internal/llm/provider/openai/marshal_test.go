package openai

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/taucentral/tau/internal/llm"
)

func TestMarshalRequest_TextOnly(t *testing.T) {
	maxTok := 1024
	temp := 0.7
	req := llm.Request{
		Model:       "gpt-4o",
		MaxTokens:   &maxTok,
		Temperature: &temp,
		System:      []llm.ContentBlock{llm.TextContent{Text: "Be nice."}},
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
	if out.Model != "gpt-4o" {
		t.Errorf("Model = %q", out.Model)
	}
	if out.StreamOptions == nil || !out.StreamOptions.IncludeUsage {
		t.Errorf("StreamOptions.IncludeUsage should be true")
	}
	// First message should be the system prompt.
	if len(out.Messages) != 2 {
		t.Fatalf("Messages len = %d, want 2", len(out.Messages))
	}
	if out.Messages[0].Role != "system" {
		t.Errorf("Messages[0].Role = %q", out.Messages[0].Role)
	}
	if string(out.Messages[0].Content) != `"Be nice."` {
		t.Errorf("Messages[0].Content = %s", out.Messages[0].Content)
	}
	if out.Messages[1].Role != "user" {
		t.Errorf("Messages[1].Role = %q", out.Messages[1].Role)
	}
	if string(out.Messages[1].Content) != `"hi"` {
		t.Errorf("Messages[1].Content = %s", out.Messages[1].Content)
	}
}

func TestMarshalRequest_AssistantWithToolUse(t *testing.T) {
	req := llm.Request{
		Model: "gpt-4o",
		Messages: []llm.Message{
			{
				Role: llm.RoleAssistant,
				Content: []llm.ContentBlock{
					llm.TextContent{Text: "calling"},
					llm.ToolUse{ID: "call_1", Name: "read", Input: json.RawMessage(`{"path":"/x"}`)},
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
	wm := out.Messages[0]
	if wm.Role != "assistant" {
		t.Errorf("Role = %q", wm.Role)
	}
	if string(wm.Content) != `"calling"` {
		t.Errorf("Content = %s", wm.Content)
	}
	if len(wm.ToolCalls) != 1 {
		t.Fatalf("ToolCalls len = %d", len(wm.ToolCalls))
	}
	tc := wm.ToolCalls[0]
	if tc.ID != "call_1" || tc.Function.Name != "read" || tc.Function.Arguments != `{"path":"/x"}` {
		t.Errorf("ToolCall = %+v", tc)
	}
}

func TestMarshalRequest_AssistantWithOnlyToolUseHasNullContent(t *testing.T) {
	req := llm.Request{
		Model: "gpt-4o",
		Messages: []llm.Message{
			{
				Role: llm.RoleAssistant,
				Content: []llm.ContentBlock{
					llm.ToolUse{ID: "c1", Name: "noop", Input: json.RawMessage(`{}`)},
				},
			},
		},
	}
	out, _ := marshalRequest(req)
	if string(out.Messages[0].Content) != "null" {
		t.Errorf("Content = %s, want null", out.Messages[0].Content)
	}
}

func TestMarshalRequest_ToolResultMessage(t *testing.T) {
	req := llm.Request{
		Model: "gpt-4o",
		Messages: []llm.Message{
			{
				Role:       llm.RoleTool,
				ToolCallID: "call_1",
				ToolName:   "read",
				Content:    []llm.ContentBlock{llm.TextContent{Text: "file contents"}},
			},
		},
	}
	out, _ := marshalRequest(req)
	if len(out.Messages) != 1 {
		t.Fatalf("Messages len = %d", len(out.Messages))
	}
	wm := out.Messages[0]
	if wm.Role != "tool" {
		t.Errorf("Role = %q", wm.Role)
	}
	if wm.ToolCallID != "call_1" {
		t.Errorf("ToolCallID = %q", wm.ToolCallID)
	}
	if wm.Name != "read" {
		t.Errorf("Name = %q", wm.Name)
	}
	if string(wm.Content) != `"file contents"` {
		t.Errorf("Content = %s", wm.Content)
	}
}

func TestMarshalRequest_UserWithImage(t *testing.T) {
	req := llm.Request{
		Model: "gpt-4o",
		Messages: []llm.Message{
			{
				Role: llm.RoleUser,
				Content: []llm.ContentBlock{
					llm.TextContent{Text: "what's this?"},
					llm.ImageContent{Data: "abc", MimeType: "image/png"},
				},
			},
		},
	}
	out, _ := marshalRequest(req)
	wm := out.Messages[0]
	// Content should be a JSON array, not a string.
	if !strings.HasPrefix(string(wm.Content), "[") {
		t.Errorf("Content should be array for image; got %s", wm.Content)
	}
	var parts []wireContentPart
	if err := json.Unmarshal(wm.Content, &parts); err != nil {
		t.Fatalf("unmarshal content: %v", err)
	}
	if len(parts) != 2 {
		t.Fatalf("parts len = %d", len(parts))
	}
	if parts[0].Type != "text" || parts[0].Text != "what's this?" {
		t.Errorf("parts[0] = %+v", parts[0])
	}
	if parts[1].Type != "image_url" || parts[1].ImageURL == nil {
		t.Errorf("parts[1] = %+v", parts[1])
	}
	if !strings.HasPrefix(parts[1].ImageURL.URL, "data:image/png;base64,abc") {
		t.Errorf("image URL = %q", parts[1].ImageURL.URL)
	}
}

func TestMarshalRequest_Tools(t *testing.T) {
	req := llm.Request{
		Model: "gpt-4o",
		Tools: []llm.ToolSchema{
			{
				Name:        "bash",
				Description: "Run a command",
				Parameters:  json.RawMessage(`{"type":"object"}`),
			},
		},
	}
	out, _ := marshalRequest(req)
	if len(out.Tools) != 1 {
		t.Fatalf("Tools len = %d", len(out.Tools))
	}
	if out.Tools[0].Type != "function" {
		t.Errorf("Type = %q", out.Tools[0].Type)
	}
	if out.Tools[0].Function.Name != "bash" {
		t.Errorf("Name = %q", out.Tools[0].Function.Name)
	}
}

func TestMarshalRequest_JSONShape(t *testing.T) {
	maxTok := 4096
	req := llm.Request{
		Model:     "gpt-4o",
		MaxTokens: &maxTok,
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.TextContent{Text: "hi"}}}},
	}
	out, _ := marshalRequest(req)
	data, _ := json.Marshal(out)
	s := string(data)
	for _, want := range []string{
		`"model":"gpt-4o"`,
		`"stream":true`,
		`"max_tokens":4096`,
		`"stream_options":{"include_usage":true}`,
	} {
		if !strings.Contains(s, want) {
			t.Errorf("missing %q in %s", want, s)
		}
	}
}

func TestMarshalRequest_SystemMessageInMessagesTolerated(t *testing.T) {
	// A system-role message that ended up in Messages (not System) should
	// be marshaled as role=system rather than rejected.
	req := llm.Request{
		Model: "gpt-4o",
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: []llm.ContentBlock{llm.TextContent{Text: "system here"}}},
		},
	}
	out, err := marshalRequest(req)
	if err != nil {
		t.Fatalf("marshalRequest: %v", err)
	}
	if out.Messages[0].Role != "system" {
		t.Errorf("Role = %q", out.Messages[0].Role)
	}
}
