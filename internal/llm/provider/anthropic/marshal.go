// marshal.go — convert llm.Request to Anthropic /v1/messages wire payload.
//
// Wire format highlights:
//
//   - system is a top-level field (not in messages), accepting a string or
//     an array of {type:"text",text:"..."} blocks. tau always uses the
//     array form for cache_control fidelity.
//   - messages carry role "user" or "assistant"; tool results are encoded
//     as user-role messages with tool_result content blocks.
//   - tools is an array of {name, description, input_schema}.
//
// Reversibility: the unmarshal.go file converts the same wire payload back
// into a tau Message; round-trip fidelity is enforced by tests.

package anthropic

import (
	"encoding/json"
	"fmt"

	"github.com/coevin/tau/internal/llm"
)

// Anthropic API version this provider targets.
const APIVersion = "2023-06-01"

// DefaultBaseURL is the production Anthropic API endpoint.
const DefaultBaseURL = "https://api.anthropic.com"

// messagesRequest is the JSON body sent to /v1/messages.
type messagesRequest struct {
	Model         string           `json:"model"`
	Messages      []wireMessage    `json:"messages"`
	System        []wireTextBlock  `json:"system,omitempty"`
	Tools         []wireTool       `json:"tools,omitempty"`
	MaxTokens     int              `json:"max_tokens"`
	Temperature   *float64         `json:"temperature,omitempty"`
	StopSequences []string         `json:"stop_sequences,omitempty"`
	Stream        bool             `json:"stream"`
	Thinking      *wireThinkingCfg `json:"thinking,omitempty"`
}

// wireMessage is one entry in the messages array.
type wireMessage struct {
	Role    string      `json:"role"`
	Content []wireBlock `json:"content"`
}

// wireBlock is a tagged union — the Type field selects the variant.
type wireBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	Thinking  string          `json:"thinking,omitempty"`
	Signature string          `json:"signature,omitempty"`
	// tool_result fields:
	ToolUseID string      `json:"tool_use_id,omitempty"`
	Content   []wireBlock `json:"content,omitempty"`
	IsError   bool        `json:"is_error,omitempty"`
	// image fields:
	Source *wireImageSource `json:"source,omitempty"`
}

// wireTextBlock is one block in the top-level system field.
type wireTextBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// wireTool is one entry in the tools array.
type wireTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

// wireImageSource is the Anthropic image source shape.
type wireImageSource struct {
	Type      string `json:"type"` // always "base64"
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

// wireThinkingCfg enables extended thinking.
type wireThinkingCfg struct {
	Type         string `json:"type"` // "enabled"
	BudgetTokens int    `json:"budget_tokens"`
}

// defaultMaxTokens is used when req.MaxTokens is nil. Anthropic requires a
// max_tokens field; we default to 4096 (the typical chat default).
const defaultMaxTokens = 4096

// marshalRequest converts a tau Request to the Anthropic wire shape.
func marshalRequest(req llm.Request) (messagesRequest, error) {
	out := messagesRequest{
		Model:         req.Model,
		Stream:        true,
		StopSequences: req.Stop,
		Temperature:   req.Temperature,
	}
	if req.MaxTokens != nil {
		out.MaxTokens = *req.MaxTokens
	} else {
		out.MaxTokens = defaultMaxTokens
	}
	if req.ThinkingBudget != nil && *req.ThinkingBudget > 0 {
		out.Thinking = &wireThinkingCfg{
			Type:         "enabled",
			BudgetTokens: *req.ThinkingBudget,
		}
	}

	// System blocks.
	for _, b := range req.System {
		switch v := b.(type) {
		case llm.TextContent:
			out.System = append(out.System, wireTextBlock{Type: "text", Text: v.Text})
		default:
			return messagesRequest{}, fmt.Errorf("anthropic: system content block of type %T not supported", b)
		}
	}

	// Messages.
	for i, m := range req.Messages {
		wm, err := marshalMessage(m)
		if err != nil {
			return messagesRequest{}, fmt.Errorf("anthropic: messages[%d]: %w", i, err)
		}
		out.Messages = append(out.Messages, wm...)
	}

	// Tools.
	for i, ts := range req.Tools {
		wt, err := marshalTool(ts)
		if err != nil {
			return messagesRequest{}, fmt.Errorf("anthropic: tools[%d]: %w", i, err)
		}
		out.Tools = append(out.Tools, wt)
	}

	return out, nil
}

// marshalMessage converts one tau Message to one or more Anthropic messages.
// A tau RoleTool Message produces a user-role message with a tool_result
// block; everything else maps 1:1.
func marshalMessage(m llm.Message) ([]wireMessage, error) {
	switch m.Role {
	case llm.RoleSystem:
		// System should be in req.System, not req.Messages. Tolerate by
		// converting to a system-role... actually Anthropic doesn't accept
		// system-role messages. Promote to user role.
		return nil, fmt.Errorf("anthropic: system-role messages must be in Request.System, not Request.Messages")
	case llm.RoleUser:
		blocks, err := marshalContentBlocks(m.Content)
		if err != nil {
			return nil, err
		}
		return []wireMessage{{Role: "user", Content: blocks}}, nil
	case llm.RoleAssistant:
		blocks, err := marshalContentBlocks(m.Content)
		if err != nil {
			return nil, err
		}
		return []wireMessage{{Role: "assistant", Content: blocks}}, nil
	case llm.RoleTool:
		// Tool result → user-role message with tool_result block(s).
		var blocks []wireBlock
		for _, b := range m.Content {
			tr, ok := b.(llm.ToolResult)
			if !ok {
				return nil, fmt.Errorf("anthropic: RoleTool message has non-ToolResult content %T", b)
			}
			tb, err := marshalToolResult(tr)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, tb)
		}
		return []wireMessage{{Role: "user", Content: blocks}}, nil
	default:
		return nil, fmt.Errorf("anthropic: unknown role %q", m.Role)
	}
}

// marshalContentBlocks converts tau ContentBlocks to Anthropic wire blocks.
func marshalContentBlocks(blocks []llm.ContentBlock) ([]wireBlock, error) {
	out := make([]wireBlock, 0, len(blocks))
	for _, b := range blocks {
		wb, err := marshalContentBlock(b)
		if err != nil {
			return nil, err
		}
		out = append(out, wb)
	}
	return out, nil
}

// marshalContentBlock converts one tau ContentBlock to one Anthropic wire block.
func marshalContentBlock(b llm.ContentBlock) (wireBlock, error) {
	switch v := b.(type) {
	case llm.TextContent:
		return wireBlock{Type: "text", Text: v.Text}, nil
	case llm.ThinkingContent:
		wb := wireBlock{Type: "thinking", Thinking: v.Thinking}
		if v.Signature != "" {
			wb.Signature = v.Signature
		}
		return wb, nil
	case llm.ImageContent:
		return wireBlock{
			Type: "image",
			Source: &wireImageSource{
				Type:      "base64",
				MediaType: v.MimeType,
				Data:      v.Data,
			},
		}, nil
	case llm.ToolUse:
		input := v.Input
		if len(input) == 0 {
			input = json.RawMessage("{}")
		}
		return wireBlock{Type: "tool_use", ID: v.ID, Name: v.Name, Input: input}, nil
	case llm.ToolResult:
		return marshalToolResult(v)
	default:
		return wireBlock{}, fmt.Errorf("anthropic: unknown content block type %T", b)
	}
}

// marshalToolResult converts a tau ToolResult to an Anthropic tool_result block.
func marshalToolResult(r llm.ToolResult) (wireBlock, error) {
	content, err := marshalContentBlocks(r.Content)
	if err != nil {
		return wireBlock{}, err
	}
	return wireBlock{
		Type:      "tool_result",
		ToolUseID: r.ToolUseID,
		Content:   content,
		IsError:   r.IsError,
	}, nil
}

// marshalTool converts a tau ToolSchema to an Anthropic tool definition.
func marshalTool(ts llm.ToolSchema) (wireTool, error) {
	if err := ts.Validate(); err != nil {
		return wireTool{}, err
	}
	return wireTool{
		Name:        ts.Name,
		Description: ts.Description,
		InputSchema: ts.Parameters,
	}, nil
}
