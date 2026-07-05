// stream.go — merge a Delta stream into an assistant Message.
//
// The agent loop consumes a provider's <-chan Delta incrementally so the TUI
// can render text as it arrives. For state persistence (and for the simpler
// non-streaming tests) we also need to collapse the whole stream into a
// single assistant Message. That collapse is what this file provides.
//
// Accumulator semantics:
//
//   - TextDelta is appended to the text block at ContentIndex. If the block
//     doesn't exist yet, one is created.
//   - ThinkingDelta is appended to the thinking block at ContentIndex, with
//     Signature captured on first non-empty sight.
//   - ToolCallDelta fragments with the same ContentIndex are concatenated
//     into a single ToolUse block at ContentIndex. The first delta seen for
//     a content index latches ID and Name.
//   - UsageDelta populates Message.Usage.
//   - Final sets StopReason / ResponseID / ResponseModel / Err. Final MUST
//     be the last delta before channel close.
//
// The accumulator tracks content blocks by their ContentIndex slot, not by
// tool-use id, because a single assistant message can interleave text,
// thinking, and multiple tool calls in any order.

package llm

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
)

// StreamAccumulator consumes a sequence of Delta values and produces an
// assistant Message. The zero value is NOT ready to use — call NewAccumulator.
//
// Accumulator is NOT safe for concurrent use. The agent loop owns a single
// goroutine that reads from the delta channel and calls Accumulate.
type StreamAccumulator struct {
	model      string
	providerID string

	// blocks indexed by ContentIndex. Holes are filled with nil until a
	// delta arrives for that index.
	blocks []ContentBlock

	// For ToolUse blocks we track the running JSON fragment so subsequent
	// deltas append to it. Indexed by ContentIndex.
	toolInputFragments map[int]*strings.Builder

	// ThinkingBlock signatures captured so far — used to handle signature
	// deltas that arrive after the thinking content.
	thinkingSigs map[int]string

	usage      *Usage
	stopReason StopReason
	responseID string
	modelUsed  string
	finalErr   error
	seenFinal  bool
}

// NewAccumulator returns a ready-to-use StreamAccumulator. model and
// providerID are stamped on the resulting Message (use the model id the
// request was issued with; ResponseModel from Final overrides this if set).
func NewAccumulator(model, providerID string) *StreamAccumulator {
	return &StreamAccumulator{
		model:              model,
		providerID:         providerID,
		toolInputFragments: make(map[int]*strings.Builder),
		thinkingSigs:       make(map[int]string),
	}
}

// Accumulate processes one delta. Returns an error if the delta violates the
// protocol (e.g., a delta arriving after Final, or a malformed tool fragment).
// Once Final is seen, subsequent deltas are rejected.
func (a *StreamAccumulator) Accumulate(d Delta) error {
	if a.seenFinal {
		return fmt.Errorf("llm: delta arrived after Final: %T", d)
	}
	switch v := d.(type) {
	case TextDelta:
		return a.accumulateText(v)
	case ThinkingDelta:
		return a.accumulateThinking(v)
	case ToolCallDelta:
		return a.accumulateToolCall(v)
	case UsageDelta:
		return a.accumulateUsage(v)
	case Final:
		return a.accumulateFinal(v)
	default:
		return fmt.Errorf("llm: unknown delta type %T", d)
	}
}

// Message returns the assistant Message assembled so far. May be called at
// any point; the returned Message is a snapshot and safe to retain.
//
// Calling Message before Final is seen produces a partial message — useful
// for live TUI updates. After Final it produces the persisted shape.
func (a *StreamAccumulator) Message() Message {
	msg := Message{
		Role:       RoleAssistant,
		Content:    a.snapshotBlocks(),
		Model:      a.modelUsed,
		ProviderID: a.providerID,
		ResponseID: a.responseID,
		StopReason: a.stopReason,
		Timestamp:  time.Now(),
		// Identifies the message as a real LLM response. The runtime
		// stamps short-circuited (cached) messages with Source="cache"
		// before they reach ResponseObserver. Empty-string defaults to
		// "llm" semantics for backward compatibility.
		Source: "llm",
	}
	if a.usage != nil {
		u := *a.usage
		msg.Usage = &u
	}
	if a.modelUsed == "" {
		msg.Model = a.model
	}
	return msg
}

// Err returns the error from Final.Err, or nil. Convenience for callers that
// only care whether the stream ended cleanly.
func (a *StreamAccumulator) Err() error {
	return a.finalErr
}

// snapshotBlocks returns the current content-block slice. Tool-use blocks
// get their Input field populated from the accumulated fragment.
func (a *StreamAccumulator) snapshotBlocks() []ContentBlock {
	out := make([]ContentBlock, len(a.blocks))
	for i, b := range a.blocks {
		if b == nil {
			out[i] = nil
			continue
		}
		switch v := b.(type) {
		case ToolUse:
			// Rebuild with accumulated fragment.
			frag, ok := a.toolInputFragments[i]
			if ok && frag != nil {
				raw := frag.String()
				if raw == "" {
					raw = "{}"
				}
				v.Input = []byte(raw)
			} else if len(v.Input) == 0 {
				v.Input = []byte("{}")
			}
			out[i] = v
		case ThinkingContent:
			// Apply captured signature if any.
			if sig, ok := a.thinkingSigs[i]; ok && v.Signature == "" {
				v.Signature = sig
			}
			out[i] = v
		default:
			out[i] = b
		}
	}
	// Trim trailing nils.
	for len(out) > 0 && out[len(out)-1] == nil {
		out = out[:len(out)-1]
	}
	return out
}

func (a *StreamAccumulator) accumulateText(d TextDelta) error {
	a.ensureSlot(d.ContentIndex)
	existing, ok := a.blocks[d.ContentIndex].(TextContent)
	if ok {
		existing.Text += d.Text
		a.blocks[d.ContentIndex] = existing
		return nil
	}
	// If the slot was empty or held a non-text block, create a new one.
	// (A non-text collision indicates a provider protocol bug; we keep the
	// existing block and discard the text delta, but log via error.)
	if a.blocks[d.ContentIndex] != nil {
		return fmt.Errorf("llm: TextDelta at index %d collides with %T", d.ContentIndex, a.blocks[d.ContentIndex])
	}
	a.blocks[d.ContentIndex] = TextContent{Text: d.Text}
	return nil
}

func (a *StreamAccumulator) accumulateThinking(d ThinkingDelta) error {
	a.ensureSlot(d.ContentIndex)
	existing, ok := a.blocks[d.ContentIndex].(ThinkingContent)
	if ok {
		existing.Thinking += d.Text
		if d.Signature != "" {
			existing.Signature = d.Signature
		}
		a.blocks[d.ContentIndex] = existing
		return nil
	}
	if a.blocks[d.ContentIndex] != nil {
		return fmt.Errorf("llm: ThinkingDelta at index %d collides with %T", d.ContentIndex, a.blocks[d.ContentIndex])
	}
	a.blocks[d.ContentIndex] = ThinkingContent{
		Thinking:  d.Text,
		Signature: d.Signature,
	}
	return nil
}

func (a *StreamAccumulator) accumulateToolCall(d ToolCallDelta) error {
	a.ensureSlot(d.ContentIndex)
	existing, ok := a.blocks[d.ContentIndex].(ToolUse)
	if ok {
		// Subsequent fragment for the same tool-use: append input.
		if d.ID != "" && existing.ID == "" {
			existing.ID = d.ID
		}
		if d.Name != "" && existing.Name == "" {
			existing.Name = d.Name
		}
		a.blocks[d.ContentIndex] = existing
	} else if a.blocks[d.ContentIndex] != nil {
		return fmt.Errorf("llm: ToolCallDelta at index %d collides with %T", d.ContentIndex, a.blocks[d.ContentIndex])
	} else {
		// First fragment: create the block.
		a.blocks[d.ContentIndex] = ToolUse{ID: d.ID, Name: d.Name}
	}
	if a.toolInputFragments[d.ContentIndex] == nil {
		a.toolInputFragments[d.ContentIndex] = &strings.Builder{}
	}
	a.toolInputFragments[d.ContentIndex].WriteString(d.PartialInput)
	return nil
}

func (a *StreamAccumulator) accumulateUsage(d UsageDelta) error {
	if a.usage != nil {
		// Some providers update usage incrementally; merge rather than overwrite.
		a.usage.Input = maxInt(d.InputTokens, a.usage.Input)
		a.usage.Output = maxInt(d.OutputTokens, a.usage.Output)
		a.usage.CacheRead = maxInt(d.CacheReadTokens, a.usage.CacheRead)
		a.usage.CacheWrite = maxInt(d.CacheWriteTokens, a.usage.CacheWrite)
		a.usage.TotalTokens = a.usage.Input + a.usage.Output
		return nil
	}
	a.usage = &Usage{
		Input:       d.InputTokens,
		Output:      d.OutputTokens,
		CacheRead:   d.CacheReadTokens,
		CacheWrite:  d.CacheWriteTokens,
		TotalTokens: d.InputTokens + d.OutputTokens,
	}
	return nil
}

func (a *StreamAccumulator) accumulateFinal(f Final) error {
	a.seenFinal = true
	a.stopReason = f.StopReason
	a.responseID = f.ResponseID
	if f.ResponseModel != "" {
		a.modelUsed = f.ResponseModel
	}
	a.finalErr = f.Err
	if f.Err != nil && a.stopReason == "" {
		a.stopReason = StopReasonError
	}
	return nil
}

func (a *StreamAccumulator) ensureSlot(index int) {
	for len(a.blocks) <= index {
		a.blocks = append(a.blocks, nil)
	}
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// CollectStream drains a Delta channel and returns the assembled Message.
// Returns an error if:
//   - the channel closed without a Final (ErrFinalMissing),
//   - Accumulate returned an error for any delta,
//   - Final.Err was set.
//
// The caller's ctx bounds the wait. If ctx is cancelled before the channel
// closes, the partial Message assembled so far is discarded and ctx.Err() is
// returned.
func CollectStream(ctx context.Context, ch <-chan Delta, model, providerID string) (Message, error) {
	acc := NewAccumulator(model, providerID)
	for {
		select {
		case <-ctx.Done():
			return Message{}, ctx.Err()
		case d, ok := <-ch:
			if !ok {
				if !acc.seenFinal {
					return Message{}, ErrFinalMissing
				}
				if err := acc.Err(); err != nil {
					return acc.Message(), err
				}
				return acc.Message(), nil
			}
			if err := acc.Accumulate(d); err != nil {
				return acc.Message(), err
			}
		}
	}
}

// ErrChannelClosed is returned when an internal helper sees a channel close
// unexpectedly. Not used by CollectStream directly (which returns
// ErrFinalMissing), but exposed for custom consumers.
var ErrChannelClosed = errors.New("delta channel closed unexpectedly")
