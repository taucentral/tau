// unmarshal.go — convert OpenAI /v1/chat/completions JSON to tau Message.
//
// Used by:
//
//   - Fixture-based tests that load an OpenAI (non-streaming) response and
//     check it round-trips through marshal.go.
//   - Future non-streaming helpers.
//
// Reversibility: marshal.go produces shapes that unmarshal.go can parse back,
// for the assistant-message case. (Tool results only flow one way.)

package openai

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/coevin/tau/internal/llm"
)

// chatResponse is the top-level shape of a non-streaming OpenAI response.
type chatResponse struct {
	ID      string           `json:"id"`
	Model   string           `json:"model"`
	Choices []responseChoice `json:"choices"`
	Usage   *chunkUsage      `json:"usage"`
}

// responseChoice is one entry in chatResponse.choices.
type responseChoice struct {
	Index        int             `json:"index"`
	Message      responseMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

// responseMessage is the assistant message in a non-streaming response.
type responseMessage struct {
	Role             string         `json:"role"`
	Content          *string        `json:"content"`
	ReasoningContent string         `json:"reasoning_content,omitempty"`
	ToolCalls        []wireToolCall `json:"tool_calls,omitempty"`
}

// unmarshalResponse converts a chatResponse to a tau assistant Message.
func unmarshalResponse(resp chatResponse) (llm.Message, error) {
	if len(resp.Choices) == 0 {
		return llm.Message{}, fmt.Errorf("openai: response has no choices")
	}
	choice := resp.Choices[0]
	msg := llm.Message{
		Role:       llm.RoleAssistant,
		Model:      resp.Model,
		ProviderID: ProviderID,
		ResponseID: resp.ID,
		StopReason: mapStopReason(choice.FinishReason),
		Timestamp:  time.Now(),
		// Real LLM response; the runtime stamps short-circuited
		// (cached) messages with Source="cache".
		Source: "llm",
	}
	if choice.Message.Content != nil && *choice.Message.Content != "" {
		msg.Content = append(msg.Content, llm.TextContent{Text: *choice.Message.Content})
	}
	if choice.Message.ReasoningContent != "" {
		// Place thinking BEFORE the text (matches streaming order).
		msg.Content = append([]llm.ContentBlock{llm.ThinkingContent{Thinking: choice.Message.ReasoningContent}}, msg.Content...)
	}
	for i, tc := range choice.Message.ToolCalls {
		input := json.RawMessage(tc.Function.Arguments)
		if len(input) == 0 {
			input = json.RawMessage("{}")
		}
		if tc.ID == "" {
			return llm.Message{}, fmt.Errorf("openai: tool_calls[%d] missing id", i)
		}
		msg.Content = append(msg.Content, llm.ToolUse{
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: input,
		})
	}
	if resp.Usage != nil {
		msg.Usage = &llm.Usage{
			Input:       resp.Usage.PromptTokens,
			Output:      resp.Usage.CompletionTokens,
			TotalTokens: resp.Usage.TotalTokens,
		}
	}
	return msg, nil
}
