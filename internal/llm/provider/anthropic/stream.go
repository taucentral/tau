// stream.go — decode Anthropic SSE events into llm.Delta values.
//
// Anthropic emits these event types over the SSE stream:
//
//   - message_start       — top-level message envelope with input usage
//   - content_block_start — opens a new block at a given content index
//   - content_block_delta — incremental update to an open block
//   - content_block_stop  — closes a block
//   - message_delta       — stop_reason / final usage update
//   - message_stop        — terminal marker
//   - ping                — heartbeat
//   - error               — mid-stream error
//
// This file does NOT do the HTTP request — that's in client.go. It exposes
// a parser that consumes decoded SSE events and emits llm.Delta values,
// which the agent loop consumes. The split keeps the parser trivially
// testable: feed it an []byte, get back an []llm.Delta.

package anthropic

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/taucentral/tau/internal/llm"
)

// streamParser translates Anthropic SSE events into llm.Delta values.
// It tracks open content blocks by index so it can route deltas to the
// right block (text/thinking/tool_use) and assemble tool-use IDs.
type streamParser struct {
	// blockType[index] = "text" | "thinking" | "tool_use" | "redacted_thinking"
	blockType map[int]string
	// blockID[index] = tool_use id (only for tool_use blocks)
	blockID map[int]string
	// blockName[index] = tool_use name (only for tool_use blocks)
	blockName map[int]string

	usage           *llm.UsageDelta
	stopReason      llm.StopReason
	responseID      string
	modelUsed       string
	finalErr        error
	seenMessageStop bool
}

func newStreamParser() *streamParser {
	return &streamParser{
		blockType: make(map[int]string),
		blockID:   make(map[int]string),
		blockName: make(map[int]string),
	}
}

// ErrStreamClosed is returned by parser.Next after message_stop.
var ErrStreamClosed = errors.New("anthropic: stream already closed by message_stop")

// handle consumes one Anthropic SSE event. Returns the deltas to emit, or
// an error if the event is malformed. After message_stop the parser
// refuses further events.
func (p *streamParser) handle(ev llm.SSEEvent) ([]llm.Delta, error) {
	if p.seenMessageStop {
		return nil, ErrStreamClosed
	}
	if ev.Type == "ping" {
		return nil, nil
	}
	if ev.Type == "error" {
		var eErr struct {
			Error struct {
				Type    string `json:"type"`
				Message string `json:"message"`
			} `json:"error"`
		}
		_ = json.Unmarshal([]byte(ev.Data), &eErr)
		p.finalErr = fmt.Errorf("anthropic: stream error: %s: %s", eErr.Error.Type, eErr.Error.Message)
		p.seenMessageStop = true
		return []llm.Delta{llm.Final{StopReason: llm.StopReasonError, Err: p.finalErr}}, nil
	}
	// All other events carry a "type" field inside the JSON payload.
	var probe struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal([]byte(ev.Data), &probe); err != nil {
		return nil, fmt.Errorf("anthropic: decode event type: %w (data=%s)", err, ev.Data)
	}
	switch probe.Type {
	case "message_start":
		return p.handleMessageStart([]byte(ev.Data))
	case "content_block_start":
		return p.handleContentBlockStart([]byte(ev.Data))
	case "content_block_delta":
		return p.handleContentBlockDelta([]byte(ev.Data))
	case "content_block_stop":
		return p.handleContentBlockStop([]byte(ev.Data))
	case "message_delta":
		return p.handleMessageDelta([]byte(ev.Data))
	case "message_stop":
		return p.handleMessageStop()
	default:
		// Unknown event types are ignored (forward-compat).
		return nil, nil
	}
}

func (p *streamParser) handleMessageStart(data []byte) ([]llm.Delta, error) {
	var ev struct {
		Message struct {
			ID    string `json:"id"`
			Model string `json:"model"`
			Usage struct {
				InputTokens              int `json:"input_tokens"`
				OutputTokens             int `json:"output_tokens"`
				CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
				CacheReadInputTokens     int `json:"cache_read_input_tokens"`
			} `json:"usage"`
		} `json:"message"`
	}
	if err := json.Unmarshal(data, &ev); err != nil {
		return nil, fmt.Errorf("anthropic: message_start: %w", err)
	}
	p.responseID = ev.Message.ID
	p.modelUsed = ev.Message.Model
	p.usage = &llm.UsageDelta{
		InputTokens:      ev.Message.Usage.InputTokens,
		OutputTokens:     ev.Message.Usage.OutputTokens,
		CacheReadTokens:  ev.Message.Usage.CacheReadInputTokens,
		CacheWriteTokens: ev.Message.Usage.CacheCreationInputTokens,
	}
	return nil, nil
}

func (p *streamParser) handleContentBlockStart(data []byte) ([]llm.Delta, error) {
	var ev struct {
		Index        int             `json:"index"`
		ContentBlock json.RawMessage `json:"content_block"`
	}
	if err := json.Unmarshal(data, &ev); err != nil {
		return nil, fmt.Errorf("anthropic: content_block_start: %w", err)
	}
	var probe struct {
		Type string `json:"type"`
		ID   string `json:"id"`
		Name string `json:"name"`
	}
	if err := json.Unmarshal(ev.ContentBlock, &probe); err != nil {
		return nil, fmt.Errorf("anthropic: content_block_start.probe: %w", err)
	}
	p.blockType[ev.Index] = probe.Type
	switch probe.Type {
	case "text", "thinking":
		// No delta to emit; the start carries an empty text.
		return nil, nil
	case "tool_use":
		p.blockID[ev.Index] = probe.ID
		p.blockName[ev.Index] = probe.Name
		// Emit a zero-input tool-call delta to establish the block.
		return []llm.Delta{llm.ToolCallDelta{
			ContentIndex: ev.Index,
			ID:           probe.ID,
			Name:         probe.Name,
			PartialInput: "",
		}}, nil
	case "redacted_thinking":
		// Redacted thinking block — emit a thinking delta carrying a
		// stand-in marker; the upstream redacted content is opaque to us.
		p.blockType[ev.Index] = "thinking"
		return []llm.Delta{llm.ThinkingDelta{ContentIndex: ev.Index, Text: "[redacted]"}}, nil
	default:
		return nil, fmt.Errorf("anthropic: content_block_start: unknown block type %q", probe.Type)
	}
}

func (p *streamParser) handleContentBlockDelta(data []byte) ([]llm.Delta, error) {
	var ev struct {
		Index int             `json:"index"`
		Delta json.RawMessage `json:"delta"`
	}
	if err := json.Unmarshal(data, &ev); err != nil {
		return nil, fmt.Errorf("anthropic: content_block_delta: %w", err)
	}
	var probe struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(ev.Delta, &probe); err != nil {
		return nil, fmt.Errorf("anthropic: content_block_delta.probe: %w", err)
	}
	bt, ok := p.blockType[ev.Index]
	if !ok {
		return nil, fmt.Errorf("anthropic: content_block_delta for unknown index %d", ev.Index)
	}
	switch probe.Type {
	case "text_delta":
		var d struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(ev.Delta, &d); err != nil {
			return nil, err
		}
		return []llm.Delta{llm.TextDelta{ContentIndex: ev.Index, Text: d.Text}}, nil
	case "thinking_delta":
		var d struct {
			Thinking string `json:"thinking"`
		}
		if err := json.Unmarshal(ev.Delta, &d); err != nil {
			return nil, err
		}
		return []llm.Delta{llm.ThinkingDelta{ContentIndex: ev.Index, Text: d.Thinking}}, nil
	case "signature_delta":
		var d struct {
			Signature string `json:"signature"`
		}
		if err := json.Unmarshal(ev.Delta, &d); err != nil {
			return nil, err
		}
		// Attach signature to the most recent thinking block at this index.
		// Since signatures follow their thinking block, we emit them as a
		// signature-bearing ThinkingDelta.
		if bt != "thinking" {
			return nil, fmt.Errorf("anthropic: signature_delta on non-thinking block at index %d", ev.Index)
		}
		return []llm.Delta{llm.ThinkingDelta{ContentIndex: ev.Index, Signature: d.Signature}}, nil
	case "input_json_delta":
		var d struct {
			PartialJSON string `json:"partial_json"`
		}
		if err := json.Unmarshal(ev.Delta, &d); err != nil {
			return nil, err
		}
		return []llm.Delta{llm.ToolCallDelta{
			ContentIndex: ev.Index,
			ID:           p.blockID[ev.Index],
			PartialInput: d.PartialJSON,
		}}, nil
	default:
		return nil, fmt.Errorf("anthropic: content_block_delta: unknown delta type %q", probe.Type)
	}
}

func (p *streamParser) handleContentBlockStop(data []byte) ([]llm.Delta, error) {
	var ev struct {
		Index int `json:"index"`
	}
	if err := json.Unmarshal(data, &ev); err != nil {
		return nil, fmt.Errorf("anthropic: content_block_stop: %w", err)
	}
	// No delta — the accumulator has already assembled everything from the
	// prior content_block_delta events. Free the per-index tracking.
	delete(p.blockType, ev.Index)
	delete(p.blockID, ev.Index)
	delete(p.blockName, ev.Index)
	return nil, nil
}

func (p *streamParser) handleMessageDelta(data []byte) ([]llm.Delta, error) {
	var ev struct {
		Delta struct {
			StopReason   string  `json:"stop_reason"`
			StopSequence *string `json:"stop_sequence"`
		} `json:"delta"`
		Usage struct {
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(data, &ev); err != nil {
		return nil, fmt.Errorf("anthropic: message_delta: %w", err)
	}
	if ev.Delta.StopReason != "" {
		p.stopReason = mapStopReason(ev.Delta.StopReason)
	}
	if p.usage != nil {
		p.usage.OutputTokens = ev.Usage.OutputTokens
	} else {
		p.usage = &llm.UsageDelta{OutputTokens: ev.Usage.OutputTokens}
	}
	return nil, nil
}

func (p *streamParser) handleMessageStop() ([]llm.Delta, error) {
	p.seenMessageStop = true
	var deltas []llm.Delta
	if p.usage != nil {
		deltas = append(deltas, *p.usage)
	}
	final := llm.Final{
		StopReason:    p.stopReason,
		ResponseID:    p.responseID,
		ResponseModel: p.modelUsed,
	}
	deltas = append(deltas, final)
	return deltas, nil
}

// mapStopReason translates Anthropic stop_reason to tau StopReason.
func mapStopReason(reason string) llm.StopReason {
	switch reason {
	case "end_turn", "stop_sequence", "pause_turn":
		return llm.StopReasonEndTurn
	case "max_tokens":
		return llm.StopReasonLength
	case "tool_use":
		return llm.StopReasonToolUse
	case "refusal", "sensitive":
		return llm.StopReasonError
	default:
		return llm.StopReasonEndTurn
	}
}
