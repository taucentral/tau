// unmarshal.go — convert Anthropic /v1/messages JSON to tau Message.
//
// Used by:
//
//   - Fixture-based tests that load an Anthropic response payload and check
//     it round-trips back to the original Request shape.
//   - Future direct-mode (non-streaming) responses, when added.
//
// The reverse mapping must be lossless for every shape marshal.go produces;
// round-trip fidelity is enforced by tests in unmarshal_test.go.

package anthropic

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/coevin/tau/internal/llm"
)

// wireResponse is the top-level shape of an Anthropic /v1/messages response
// (both streaming message_start envelope and non-streaming response).
type wireResponse struct {
	ID         string            `json:"id"`
	Model      string            `json:"model"`
	StopReason string            `json:"stop_reason"`
	Role       string            `json:"role"` // always "assistant"
	Content    []wireBlock       `json:"content"`
	Usage      wireResponseUsage `json:"usage"`
}

// wireResponseUsage is the Anthropic usage object in a response.
type wireResponseUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
}

// unmarshalResponse converts a wireResponse to a tau assistant Message.
func unmarshalResponse(resp wireResponse) (llm.Message, error) {
	msg := llm.Message{
		Role:       llm.RoleAssistant,
		Model:      resp.Model,
		ProviderID: "anthropic",
		ResponseID: resp.ID,
		StopReason: mapStopReason(resp.StopReason),
		Timestamp:  time.Now(),
		// Real LLM response; the runtime stamps short-circuited
		// (cached) messages with Source="cache".
		Source: "llm",
	}
	if resp.Usage.InputTokens > 0 || resp.Usage.OutputTokens > 0 {
		msg.Usage = &llm.Usage{
			Input:       resp.Usage.InputTokens,
			Output:      resp.Usage.OutputTokens,
			CacheRead:   resp.Usage.CacheReadInputTokens,
			CacheWrite:  resp.Usage.CacheCreationInputTokens,
			TotalTokens: resp.Usage.InputTokens + resp.Usage.OutputTokens,
		}
	}
	for i, b := range resp.Content {
		block, err := unmarshalBlock(b)
		if err != nil {
			return llm.Message{}, fmt.Errorf("anthropic: content[%d]: %w", i, err)
		}
		msg.Content = append(msg.Content, block)
	}
	return msg, nil
}

// unmarshalBlock converts a wireBlock to a tau ContentBlock.
func unmarshalBlock(b wireBlock) (llm.ContentBlock, error) {
	switch b.Type {
	case "text":
		return llm.TextContent{Text: b.Text}, nil
	case "thinking":
		return llm.ThinkingContent{Thinking: b.Thinking, Signature: b.Signature}, nil
	case "redacted_thinking":
		return llm.ThinkingContent{Thinking: "", Signature: b.Signature, Redacted: true}, nil
	case "image":
		if b.Source == nil {
			return nil, fmt.Errorf("anthropic: image block missing source")
		}
		return llm.ImageContent{Data: b.Source.Data, MimeType: b.Source.MediaType}, nil
	case "tool_use":
		input := b.Input
		if len(input) == 0 {
			input = json.RawMessage("{}")
		}
		return llm.ToolUse{ID: b.ID, Name: b.Name, Input: input}, nil
	case "tool_result":
		var content []llm.ContentBlock
		for _, cb := range b.Content {
			block, err := unmarshalBlock(cb)
			if err != nil {
				return nil, err
			}
			content = append(content, block)
		}
		return llm.ToolResult{ToolUseID: b.ToolUseID, Content: content, IsError: b.IsError}, nil
	default:
		return nil, fmt.Errorf("anthropic: unknown wire block type %q", b.Type)
	}
}
