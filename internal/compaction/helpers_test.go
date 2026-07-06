package compaction

import (
	"context"
	"encoding/json"
	"time"

	"github.com/taucentral/tau/internal/llm"
	"github.com/taucentral/tau/internal/llm/tokencounter"
	"github.com/taucentral/tau/internal/state"
)

// nowAt returns a fixed base time advanced by the given offset; useful for
// producing deterministic timestamps in tests where recency ordering matters.
func nowAt(offset time.Duration) time.Time {
	base := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	return base.Add(offset)
}

// mkMessage builds a Message entry with the given role and a single text block.
// ts may be zero for "do not care" — recency tests pass explicit timestamps.
func mkMessage(id, parentID string, role llm.Role, text string, ts time.Time) state.Entry {
	return state.Entry{
		ID:        id,
		ParentID:  parentID,
		Kind:      state.KindMessage,
		Timestamp: ts,
		Payload:   state.MessagePayload{Role: role, Content: []llm.ContentBlock{llm.TextContent{Text: text}}},
	}
}

// mkToolUseMessage builds an assistant Message whose sole content block is a
// ToolUse with the given name and JSON input.
func mkToolUseMessage(id, parentID, toolName, toolUseID string, input map[string]any, ts time.Time) state.Entry {
	raw, _ := json.Marshal(input)
	return state.Entry{
		ID:       id,
		ParentID: parentID,
		Kind:     state.KindMessage,
		Timestamp: func() time.Time {
			if ts.IsZero() {
				return nowAt(0)
			}
			return ts
		}(),
		Payload: state.MessagePayload{
			Role: llm.RoleAssistant,
			Content: []llm.ContentBlock{
				llm.ToolUse{ID: toolUseID, Name: toolName, Input: raw},
			},
		},
	}
}

// mkToolResultMessage builds a user Message whose sole content block is a
// ToolResult responding to the given toolUseID.
func mkToolResultMessage(id, parentID, toolUseID, resultText string, ts time.Time) state.Entry {
	return state.Entry{
		ID:        id,
		ParentID:  parentID,
		Kind:      state.KindMessage,
		Timestamp: ts,
		Payload: state.MessagePayload{
			Role: llm.RoleUser,
			Content: []llm.ContentBlock{
				llm.ToolResult{
					ToolUseID: toolUseID,
					Content:   []llm.ContentBlock{llm.TextContent{Text: resultText}},
				},
			},
		},
	}
}

// mkLabel builds a Label entry.
func mkLabel(id, parentID, label string, ts time.Time) state.Entry {
	return state.Entry{
		ID:        id,
		ParentID:  parentID,
		Kind:      state.KindLabel,
		Timestamp: ts,
		Payload:   state.LabelPayload{Label: label},
	}
}

// mkSessionInfo builds a SessionInfo entry.
func mkSessionInfo(id, parentID, key, value string, ts time.Time) state.Entry {
	return state.Entry{
		ID:        id,
		ParentID:  parentID,
		Kind:      state.KindSessionInfo,
		Timestamp: ts,
		Payload:   state.SessionInfoPayload{Key: key, Value: value},
	}
}

// mkCompaction builds a Compaction entry.
func mkCompaction(id, parentID, summary, firstKept string, ts time.Time) state.Entry {
	return state.Entry{
		ID:        id,
		ParentID:  parentID,
		Kind:      state.KindCompaction,
		Timestamp: ts,
		Payload:   state.CompactionPayload{Summary: summary, FirstKeptEntryID: firstKept},
	}
}

// mkBranchSummary builds a BranchSummary entry.
func mkBranchSummary(id, parentID, summary string, ts time.Time) state.Entry {
	return state.Entry{
		ID:        id,
		ParentID:  parentID,
		Kind:      state.KindBranchSummary,
		Timestamp: ts,
		Payload:   state.BranchSummaryPayload{Summary: summary},
	}
}

// mkSessionHeader builds a SessionHeader root entry. Used as the root of every
// test tree so state.NewTree accepts it.
func mkSessionHeader(id string) state.Entry {
	return state.Entry{
		ID:        id,
		Kind:      state.KindSessionHeader,
		Timestamp: nowAt(0),
		Payload:   state.SessionHeaderPayload{SessionID: id, Version: state.CurrentSchemaVersion},
	}
}

// detCounter is a deterministic TokenCounter that scores text as len(text)/charsPerToken
// (rounded up). It lets cut-point tests reason about budgets exactly without
// depending on a BPE encoding.
type detCounter struct {
	charsPerToken int
}

func (d detCounter) Count(model, text string) int {
	if d.charsPerToken <= 0 {
		return len(text)
	}
	return (len(text) + d.charsPerToken - 1) / d.charsPerToken
}

func (d detCounter) CountMessages(model string, msgs []llm.Message) int {
	total := 0
	for _, m := range msgs {
		for _, b := range m.Content {
			if tc, ok := b.(llm.TextContent); ok {
				total += d.Count(model, tc.Text)
				continue
			}
			total += 5
		}
		total += 4 // per-message framing
	}
	return total
}

// Compile-time check.
var _ tokencounter.TokenCounter = detCounter{}

// fakeLLM is an LLMClient that emits a scripted sequence of deltas. It records
// the last Request it saw so tests can assert on the prompt.
type fakeLLM struct {
	deltas []llm.Delta
	err    error
	last   llm.Request
	gotReq bool
}

func (f *fakeLLM) Stream(ctx context.Context, req llm.Request) (<-chan llm.Delta, error) {
	f.last = req
	f.gotReq = true
	if f.err != nil {
		return nil, f.err
	}
	ch := make(chan llm.Delta, len(f.deltas))
	go func() {
		defer close(ch)
		for _, d := range f.deltas {
			select {
			case <-ctx.Done():
				return
			case ch <- d:
			}
		}
	}()
	return ch, nil
}

// lastPromptText returns the concatenated text of the last request's user
// message. Tests use it to assert which prompt template ran.
func (f *fakeLLM) lastPromptText() string {
	if len(f.last.Messages) == 0 {
		return ""
	}
	var out string
	for _, b := range f.last.Messages[0].Content {
		if tc, ok := b.(llm.TextContent); ok {
			out += tc.Text
		}
	}
	return out
}

// Compile-time check.
var _ llm.LLMClient = (*fakeLLM)(nil)
