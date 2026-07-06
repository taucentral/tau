// marshal.go — convert llm.Request to OpenAI /v1/chat/completions wire payload.
//
// Wire format highlights:
//
//   - System prompt is the first message with role="system" (or role="developer"
//     for reasoning models — tau uses "system" which all current OpenAI
//     models accept).
//   - Assistant text and tool calls live in separate fields: content (string)
//     and tool_calls (array). tau concatenates all text blocks into content.
//   - Tool results are role="tool" messages with tool_call_id.
//   - Image inputs use the multi-content-part form (an array of typed parts).
//
// This file is the inverse of unmarshal.go; round-trip fidelity is enforced
// by tests.

package openai

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/taucentral/tau/internal/llm"
)

// DefaultBaseURL is the production OpenAI endpoint.
const DefaultBaseURL = "https://api.openai.com/v1"

// ProviderID is the canonical name for this provider.
const ProviderID = "openai"

// chatRequest is the JSON body sent to /v1/chat/completions.
type chatRequest struct {
	Model         string        `json:"model"`
	Messages      []wireMessage `json:"messages"`
	Tools         []wireTool    `json:"tools,omitempty"`
	MaxTokens     *int          `json:"max_tokens,omitempty"`
	Temperature   *float64      `json:"temperature,omitempty"`
	Stream        bool          `json:"stream"`
	StreamOptions *streamOpts   `json:"stream_options,omitempty"`
	Stop          []string      `json:"stop,omitempty"`
}

// streamOpts asks OpenAI to include a final usage chunk.
type streamOpts struct {
	IncludeUsage bool `json:"include_usage"`
}

// wireMessage is one entry in the messages array.
type wireMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content,omitempty"` // string OR multi-part array
	ToolCalls  []wireToolCall  `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	Name       string          `json:"name,omitempty"` // optional for tool messages
}

// wireToolCall is one entry in assistant tool_calls.
type wireToolCall struct {
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type"` // "function"
	Function wireToolFunction `json:"function"`
}

// wireToolFunction is the inner function shape.
type wireToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON string (NOT an object)
}

// wireTool is one entry in the tools array.
type wireTool struct {
	Type     string      `json:"type"` // "function"
	Function wireToolDef `json:"function"`
}

// wireToolDef is the inner function definition.
type wireToolDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// wireContentPart is one entry in a multi-part user message (text or image).
type wireContentPart struct {
	Type     string        `json:"type"`
	Text     string        `json:"text,omitempty"`
	ImageURL *wireImageURL `json:"image_url,omitempty"`
}

// wireImageURL is the OpenAI image shape.
type wireImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// marshalRequest converts a tau Request to the OpenAI wire shape.
func marshalRequest(req llm.Request) (chatRequest, error) {
	out := chatRequest{
		Model:         req.Model,
		Stream:        true,
		StreamOptions: &streamOpts{IncludeUsage: true},
		MaxTokens:     req.MaxTokens,
		Temperature:   req.Temperature,
		Stop:          req.Stop,
	}

	// System prompt as a leading system message.
	if len(req.System) > 0 {
		systemText, err := systemToText(req.System)
		if err != nil {
			return chatRequest{}, fmt.Errorf("openai: system: %w", err)
		}
		if systemText != "" {
			out.Messages = append(out.Messages, wireMessage{
				Role:    "system",
				Content: json.RawMessage(`"` + jsonEscapeString(systemText) + `"`),
			})
		}
	}

	// Conversation messages.
	for i, m := range req.Messages {
		wm, err := marshalMessage(m)
		if err != nil {
			return chatRequest{}, fmt.Errorf("openai: messages[%d]: %w", i, err)
		}
		out.Messages = append(out.Messages, wm)
	}

	// Tools.
	for i, ts := range req.Tools {
		if err := ts.Validate(); err != nil {
			return chatRequest{}, fmt.Errorf("openai: tools[%d]: %w", i, err)
		}
		out.Tools = append(out.Tools, wireTool{
			Type: "function",
			Function: wireToolDef{
				Name:        ts.Name,
				Description: ts.Description,
				Parameters:  ts.Parameters,
			},
		})
	}

	return out, nil
}

// marshalMessage converts one tau Message to one OpenAI wire message.
func marshalMessage(m llm.Message) (wireMessage, error) {
	switch m.Role {
	case llm.RoleSystem:
		// Tolerate system messages in the messages array (callers SHOULD put
		// them in Request.System, but it's not always possible with a
		// forked conversation).
		return wireMessage{
			Role:    "system",
			Content: json.RawMessage(`"` + jsonEscapeString(blocksToText(m.Content)) + `"`),
		}, nil
	case llm.RoleUser:
		content, err := blocksToContent(m.Content)
		if err != nil {
			return wireMessage{}, err
		}
		return wireMessage{Role: "user", Content: content}, nil
	case llm.RoleAssistant:
		wm := wireMessage{Role: "assistant"}
		// Concatenate all text blocks into content; gather tool_use blocks.
		var textParts []string
		for _, b := range m.Content {
			switch v := b.(type) {
			case llm.TextContent:
				textParts = append(textParts, v.Text)
			case llm.ToolUse:
				input := string(v.Input)
				if input == "" {
					input = "{}"
				}
				wm.ToolCalls = append(wm.ToolCalls, wireToolCall{
					ID:   v.ID,
					Type: "function",
					Function: wireToolFunction{
						Name:      v.Name,
						Arguments: input,
					},
				})
			case llm.ThinkingContent:
				// OpenAI Chat Completions has no native thinking channel.
				// Drop silently — callers that need thinking should use the
				// Anthropic provider or the OpenAI Responses API (future).
			default:
				return wireMessage{}, fmt.Errorf("openai: assistant block type %T not supported", b)
			}
		}
		if len(textParts) > 0 {
			joined := strings.Join(textParts, "")
			wm.Content = json.RawMessage(`"` + jsonEscapeString(joined) + `"`)
		} else {
			// OpenAI requires content to be null when only tool_calls are present.
			wm.Content = json.RawMessage(`null`)
		}
		return wm, nil
	case llm.RoleTool:
		// Tool result message — extract text content and set tool_call_id.
		text := blocksToText(m.Content)
		return wireMessage{
			Role:       "tool",
			Content:    json.RawMessage(`"` + jsonEscapeString(text) + `"`),
			ToolCallID: m.ToolCallID,
			Name:       m.ToolName,
		}, nil
	default:
		return wireMessage{}, fmt.Errorf("openai: unknown role %q", m.Role)
	}
}

// systemToText concatenates system ContentBlocks (must all be Text).
func systemToText(blocks []llm.ContentBlock) (string, error) {
	var b strings.Builder
	for _, blk := range blocks {
		tc, ok := blk.(llm.TextContent)
		if !ok {
			return "", fmt.Errorf("system content block of type %T not supported", blk)
		}
		b.WriteString(tc.Text)
	}
	return b.String(), nil
}

// blocksToText concatenates text from a slice of mixed blocks (used for
// tool-result messages where OpenAI only takes a string content).
func blocksToText(blocks []llm.ContentBlock) string {
	var b strings.Builder
	for _, blk := range blocks {
		switch v := blk.(type) {
		case llm.TextContent:
			b.WriteString(v.Text)
		case llm.ToolResult:
			// Inline tool_result's nested text.
			b.WriteString(blocksToText(v.Content))
		}
	}
	return b.String()
}

// blocksToContent decides whether to emit a string content or a multi-part
// array based on whether any image blocks are present.
func blocksToContent(blocks []llm.ContentBlock) (json.RawMessage, error) {
	hasImage := false
	for _, b := range blocks {
		if _, ok := b.(llm.ImageContent); ok {
			hasImage = true
			break
		}
	}
	if !hasImage {
		// String content path.
		return json.RawMessage(`"` + jsonEscapeString(blocksToText(blocks)) + `"`), nil
	}
	// Multi-part array path.
	parts := make([]wireContentPart, 0, len(blocks))
	for _, b := range blocks {
		switch v := b.(type) {
		case llm.TextContent:
			parts = append(parts, wireContentPart{Type: "text", Text: v.Text})
		case llm.ImageContent:
			parts = append(parts, wireContentPart{
				Type:     "image_url",
				ImageURL: &wireImageURL{URL: "data:" + v.MimeType + ";base64," + v.Data},
			})
		case llm.ToolResult:
			// Inline as text parts.
			for _, sub := range v.Content {
				if tc, ok := sub.(llm.TextContent); ok {
					parts = append(parts, wireContentPart{Type: "text", Text: tc.Text})
				}
			}
		}
	}
	data, err := json.Marshal(parts)
	if err != nil {
		return nil, err
	}
	return json.RawMessage(data), nil
}

// jsonEscapeString returns the inside of a JSON string literal (without the
// surrounding quotes) for s. Used to inline string values into RawMessage.
func jsonEscapeString(s string) string {
	b, err := json.Marshal(s)
	if err != nil {
		// Should never happen for a string input.
		return ""
	}
	// Trim the surrounding quotes that json.Marshal adds.
	return string(b[1 : len(b)-1])
}
