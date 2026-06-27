// stream.go — decode OpenAI Chat Completions SSE events into llm.Delta values.
//
// OpenAI streams one JSON chunk per line; each chunk has the shape:
//
//	{
//	  "id": "chatcmpl-...",
//	  "model": "gpt-4o",
//	  "choices": [{"index": 0, "delta": {...}, "finish_reason": null}],
//	  "usage": {...}   // only on the final chunk when include_usage=true
//	}
//
// And the stream terminates with the literal line `data: [DONE]`.
//
// Each delta has at most one of:
//
//   - delta.content       — incremental assistant text
//   - delta.role          — first chunk, sets role (no-op for us)
//   - delta.tool_calls[]  — incremental tool-call fragments (keyed by index)
//   - delta.reasoning_content — reasoning output (some compat providers)
//
// Tool call assembly: each tool_call has an `index` field (0, 1, ...) which
// identifies which tool call within the message. The first chunk for a given
// index carries id and name; subsequent chunks for that index carry only
// argument fragments. We map OpenAI's tool_call.index to a tau ContentIndex
// using the rule: tau_index = (1 if any text seen else 0) + openai_index.

package openai

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/coevin/tau/internal/llm"
)

// streamParser translates OpenAI SSE events into llm.Delta values.
type streamParser struct {
	// responseID is captured from the first chunk's "id" field.
	responseID string
	// modelUsed is captured from the first chunk's "model" field.
	modelUsed string
	// textStarted tracks whether we've emitted any text delta (and thus
	// consumed tau_index 0).
	textStarted bool
	// toolCallIDs tracks openai_index → id (populated on first delta for an index).
	toolCallIDs map[int]string
	// toolCallNames tracks openai_index → name (populated on first delta).
	toolCallNames map[int]string
	// usageSeen indicates we've already emitted a UsageDelta.
	usageSeen bool
	// pendingReason is the finish_reason seen on a content chunk, waiting
	// for either a usage chunk or [DONE] to be turned into a Final.
	pendingReason string
	// finalEmitted indicates we've already emitted a Final. Subsequent
	// chunks (usage-only, [DONE]) are skipped silently.
	finalEmitted bool
}

func newStreamParser() *streamParser {
	return &streamParser{
		toolCallIDs:   make(map[int]string),
		toolCallNames: make(map[int]string),
	}
}

// ErrStreamClosed is returned after [DONE] is processed.
var ErrStreamClosed = errors.New("openai: stream already closed by [DONE]")

// handle consumes one SSE event from the OpenAI stream.
func (p *streamParser) handle(ev llm.SSEEvent) ([]llm.Delta, error) {
	if ev.Type != "message" {
		// OpenAI doesn't use named events; everything is "message".
		return nil, nil
	}
	data := strings.TrimSpace(ev.Data)
	if data == "[DONE]" {
		return p.handleDone(), nil
	}
	// After a Final has been emitted, silently skip subsequent chunks
	// (usage-only chunks, terminal events).
	if p.finalEmitted {
		return nil, nil
	}
	// Try to decode as a chunk. Some OpenAI-compatible providers emit error
	// objects instead of chunks; detect and surface.
	var chunk chatChunk
	if err := json.Unmarshal([]byte(data), &chunk); err != nil {
		return nil, fmt.Errorf("openai: decode chunk: %w (data=%s)", err, data)
	}
	if chunk.Error != nil {
		p.finalEmitted = true
		return []llm.Delta{llm.Final{
			StopReason: llm.StopReasonError,
			Err:        fmt.Errorf("openai: %s", chunk.Error.Message),
		}}, nil
	}
	return p.handleChunk(chunk)
}

// chatChunk is the JSON shape of one streamed chunk.
type chatChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Model   string        `json:"model"`
	Choices []chunkChoice `json:"choices"`
	Usage   *chunkUsage   `json:"usage"`
	// Error is non-standard but several OpenAI-compatible providers (Ollama,
	// LM Studio) emit it instead of HTTP status codes.
	Error *chunkError `json:"error,omitempty"`
}

// chunkChoice is one entry in chunk.choices.
type chunkChoice struct {
	Index        int        `json:"index"`
	Delta        chunkDelta `json:"delta"`
	FinishReason *string    `json:"finish_reason"`
}

// chunkDelta is the incremental update.
type chunkDelta struct {
	Role             string          `json:"role,omitempty"`
	Content          string          `json:"content,omitempty"`
	ReasoningContent string          `json:"reasoning_content,omitempty"`
	ToolCalls        []chunkToolCall `json:"tool_calls,omitempty"`
}

// chunkToolCall is one entry in delta.tool_calls.
type chunkToolCall struct {
	Index    int               `json:"index"`
	ID       string            `json:"id,omitempty"`
	Type     string            `json:"type,omitempty"`
	Function chunkToolFunction `json:"function"`
}

// chunkToolFunction is the inner function shape.
type chunkToolFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// chunkUsage is the per-stream token accounting (final chunk only).
type chunkUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// chunkError is the OpenAI-compatible error shape.
type chunkError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

// handleChunk processes one parsed chunk.
func (p *streamParser) handleChunk(c chatChunk) ([]llm.Delta, error) {
	if p.responseID == "" && c.ID != "" {
		p.responseID = c.ID
	}
	if p.modelUsed == "" && c.Model != "" {
		p.modelUsed = c.Model
	}
	var out []llm.Delta
	for _, choice := range c.Choices {
		// Text delta.
		if choice.Delta.Content != "" {
			if !p.textStarted {
				p.textStarted = true
			}
			out = append(out, llm.TextDelta{ContentIndex: 0, Text: choice.Delta.Content})
		}
		// Reasoning delta — treat as ThinkingDelta (provider-specific, but tau
		// exposes a unified thinking channel).
		if choice.Delta.ReasoningContent != "" {
			// Reasoning comes before content in practice; tau_index 0 is the
			// thinking slot. If text has also started, this is an unusual
			// ordering; we still emit at ContentIndex 0 — the accumulator
			// will type-collide-reject, but no provider actually does this.
			out = append(out, llm.ThinkingDelta{ContentIndex: 0, Text: choice.Delta.ReasoningContent})
		}
		// Tool call deltas.
		for _, tc := range choice.Delta.ToolCalls {
			id := tc.ID
			name := tc.Function.Name
			if id != "" {
				p.toolCallIDs[tc.Index] = id
			} else {
				id = p.toolCallIDs[tc.Index]
			}
			if name != "" {
				p.toolCallNames[tc.Index] = name
			} else {
				name = p.toolCallNames[tc.Index]
			}
			tauIdx := p.tauIndexForToolCall(tc.Index)
			out = append(out, llm.ToolCallDelta{
				ContentIndex: tauIdx,
				ID:           id,
				Name:         name,
				PartialInput: tc.Function.Arguments,
			})
		}
		// Finish reason.
		if choice.FinishReason != nil {
			out = append(out, p.handleFinish(*choice.FinishReason, c.Usage)...)
			return out, nil
		}
	}
	// Usage-only chunk (final chunk when include_usage=true). This is where
	// the deferred Final from a prior finish_reason chunk is emitted.
	if c.Usage != nil && len(c.Choices) == 0 && !p.usageSeen {
		p.usageSeen = true
		out = append(out, llm.UsageDelta{
			InputTokens:  c.Usage.PromptTokens,
			OutputTokens: c.Usage.CompletionTokens,
		})
		if p.pendingReason != "" && !p.finalEmitted {
			p.finalEmitted = true
			reason := p.pendingReason
			p.pendingReason = ""
			out = append(out, llm.Final{
				StopReason:    mapStopReason(reason),
				ResponseID:    p.responseID,
				ResponseModel: p.modelUsed,
			})
		}
	}
	return out, nil
}

// handleFinish records the finish_reason. The Final is emitted eagerly only
// when usage is bundled in the same chunk; otherwise the reason is stashed in
// pendingReason and the Final is deferred until the subsequent usage-only
// chunk (or [DONE]) arrives. OpenAI's include_usage=true stream sends the
// finish_reason in one chunk and the usage in the NEXT chunk — emitting the
// Final immediately would cause the usage chunk to be silently skipped.
func (p *streamParser) handleFinish(reason string, usage *chunkUsage) []llm.Delta {
	var out []llm.Delta
	if usage != nil && !p.usageSeen {
		p.usageSeen = true
		out = append(out, llm.UsageDelta{
			InputTokens:  usage.PromptTokens,
			OutputTokens: usage.CompletionTokens,
		})
	}
	if usage != nil {
		// Usage bundled in the finish chunk: no subsequent usage chunk will
		// arrive, so emit Final now.
		p.finalEmitted = true
		out = append(out, llm.Final{
			StopReason:    mapStopReason(reason),
			ResponseID:    p.responseID,
			ResponseModel: p.modelUsed,
		})
		return out
	}
	// Stash the reason; the Final will be emitted when the usage chunk (or
	// [DONE]) arrives.
	p.pendingReason = reason
	return out
}

// handleDone is invoked when the parser sees `data: [DONE]`. If a Final was
// already emitted, returns nil. Otherwise we synthesize one, using any
// pending finish_reason captured from a prior chunk. If no reason was seen
// at all, default to EndTurn (some compat providers send [DONE] with no
// finish_reason).
func (p *streamParser) handleDone() []llm.Delta {
	if p.finalEmitted {
		return nil
	}
	p.finalEmitted = true
	reason := p.pendingReason
	if reason == "" {
		reason = "stop"
	}
	p.pendingReason = ""
	stopReason := mapStopReason(reason)
	if !p.usageSeen {
		return []llm.Delta{
			llm.UsageDelta{},
			llm.Final{StopReason: stopReason, ResponseID: p.responseID, ResponseModel: p.modelUsed},
		}
	}
	return []llm.Delta{
		llm.Final{StopReason: stopReason, ResponseID: p.responseID, ResponseModel: p.modelUsed},
	}
}

// tauIndexForToolCall maps an OpenAI tool_call index to a tau ContentIndex.
// tau_index = (1 if any text seen else 0) + openai_index.
func (p *streamParser) tauIndexForToolCall(openaiIdx int) int {
	offset := 0
	if p.textStarted {
		offset = 1
	}
	return offset + openaiIdx
}

// mapStopReason translates OpenAI finish_reason to tau StopReason.
func mapStopReason(reason string) llm.StopReason {
	switch reason {
	case "stop", "end":
		return llm.StopReasonEndTurn
	case "length":
		return llm.StopReasonLength
	case "function_call", "tool_calls":
		return llm.StopReasonToolUse
	case "content_filter", "network_error":
		return llm.StopReasonError
	default:
		return llm.StopReasonEndTurn
	}
}
