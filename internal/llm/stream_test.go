package llm

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestAccumulator_TextOnly(t *testing.T) {
	acc := NewAccumulator("m", "p")
	deltas := []Delta{
		TextDelta{ContentIndex: 0, Text: "Hello "},
		TextDelta{ContentIndex: 0, Text: "world"},
		UsageDelta{InputTokens: 10, OutputTokens: 5},
		Final{StopReason: StopReasonEndTurn},
	}
	for _, d := range deltas {
		if err := acc.Accumulate(d); err != nil {
			t.Fatalf("Accumulate: %v", err)
		}
	}
	msg := acc.Message()
	if len(msg.Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(msg.Content))
	}
	tc, ok := msg.Content[0].(TextContent)
	if !ok {
		t.Fatalf("Content[0] = %T, want TextContent", msg.Content[0])
	}
	if tc.Text != "Hello world" {
		t.Errorf("Text = %q", tc.Text)
	}
	if msg.Usage == nil || msg.Usage.Output != 5 {
		t.Errorf("Usage lost: %+v", msg.Usage)
	}
	if msg.StopReason != StopReasonEndTurn {
		t.Errorf("StopReason = %q", msg.StopReason)
	}
}

func TestAccumulator_ToolUseAssembledFromFragments(t *testing.T) {
	acc := NewAccumulator("m", "p")
	deltas := []Delta{
		ToolCallDelta{ContentIndex: 0, ID: "call_1", Name: "read", PartialInput: `{"path":"`},
		ToolCallDelta{ContentIndex: 0, PartialInput: `/tmp/x"}`},
		Final{StopReason: StopReasonToolUse},
	}
	for _, d := range deltas {
		if err := acc.Accumulate(d); err != nil {
			t.Fatalf("Accumulate: %v", err)
		}
	}
	msg := acc.Message()
	if len(msg.Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(msg.Content))
	}
	tu, ok := msg.Content[0].(ToolUse)
	if !ok {
		t.Fatalf("Content[0] = %T", msg.Content[0])
	}
	if tu.ID != "call_1" || tu.Name != "read" {
		t.Errorf("ID/Name = %q/%q", tu.ID, tu.Name)
	}
	// Validate the concatenated input is valid JSON.
	var got map[string]any
	if err := json.Unmarshal(tu.Input, &got); err != nil {
		t.Fatalf("input JSON invalid: %v; raw=%s", err, tu.Input)
	}
	if got["path"] != "/tmp/x" {
		t.Errorf("path = %v", got["path"])
	}
}

func TestAccumulator_ThinkingAndTextInterleaved(t *testing.T) {
	acc := NewAccumulator("m", "p")
	deltas := []Delta{
		ThinkingDelta{ContentIndex: 0, Text: "Thinking..."},
		TextDelta{ContentIndex: 1, Text: "Answer"},
		Final{StopReason: StopReasonEndTurn},
	}
	for _, d := range deltas {
		if err := acc.Accumulate(d); err != nil {
			t.Fatalf("Accumulate: %v", err)
		}
	}
	msg := acc.Message()
	if len(msg.Content) != 2 {
		t.Fatalf("Content len = %d, want 2", len(msg.Content))
	}
	if _, ok := msg.Content[0].(ThinkingContent); !ok {
		t.Errorf("Content[0] = %T", msg.Content[0])
	}
	if _, ok := msg.Content[1].(TextContent); !ok {
		t.Errorf("Content[1] = %T", msg.Content[1])
	}
}

func TestAccumulator_ThinkingSignatureCaptured(t *testing.T) {
	acc := NewAccumulator("m", "p")
	deltas := []Delta{
		ThinkingDelta{ContentIndex: 0, Text: "thought"},
		ThinkingDelta{ContentIndex: 0, Signature: "sig123"},
		Final{StopReason: StopReasonEndTurn},
	}
	for _, d := range deltas {
		_ = acc.Accumulate(d)
	}
	msg := acc.Message()
	tc, ok := msg.Content[0].(ThinkingContent)
	if !ok {
		t.Fatalf("Content[0] = %T", msg.Content[0])
	}
	if tc.Signature != "sig123" {
		t.Errorf("Signature = %q", tc.Signature)
	}
}

func TestAccumulator_MultipleToolCallsByIndex(t *testing.T) {
	acc := NewAccumulator("m", "p")
	deltas := []Delta{
		ToolCallDelta{ContentIndex: 0, ID: "a", Name: "read", PartialInput: `{"path":"a"`},
		ToolCallDelta{ContentIndex: 0, PartialInput: `}`},
		ToolCallDelta{ContentIndex: 1, ID: "b", Name: "read", PartialInput: `{"path":"b"}`},
		Final{StopReason: StopReasonToolUse},
	}
	for _, d := range deltas {
		if err := acc.Accumulate(d); err != nil {
			t.Fatalf("Accumulate: %v", err)
		}
	}
	msg := acc.Message()
	if len(msg.Content) != 2 {
		t.Fatalf("Content len = %d, want 2", len(msg.Content))
	}
}

func TestAccumulator_FinalErrorPropagates(t *testing.T) {
	acc := NewAccumulator("m", "p")
	wantErr := errors.New("provider exploded")
	_ = acc.Accumulate(TextDelta{ContentIndex: 0, Text: "partial"})
	_ = acc.Accumulate(Final{Err: wantErr})
	if !errors.Is(acc.Err(), wantErr) {
		t.Errorf("Err = %v, want %v", acc.Err(), wantErr)
	}
	msg := acc.Message()
	if msg.StopReason != StopReasonError {
		t.Errorf("StopReason = %q, want %q", msg.StopReason, StopReasonError)
	}
}

func TestAccumulator_DeltaAfterFinalRejected(t *testing.T) {
	acc := NewAccumulator("m", "p")
	_ = acc.Accumulate(Final{StopReason: StopReasonEndTurn})
	err := acc.Accumulate(TextDelta{ContentIndex: 0, Text: "late"})
	if err == nil {
		t.Errorf("expected error for delta after Final")
	}
}

func TestAccumulator_TypeCollisionRejected(t *testing.T) {
	acc := NewAccumulator("m", "p")
	_ = acc.Accumulate(TextDelta{ContentIndex: 0, Text: "x"})
	err := acc.Accumulate(ThinkingDelta{ContentIndex: 0, Text: "y"})
	if err == nil {
		t.Errorf("expected error for type collision")
	}
	if !strings.Contains(err.Error(), "collides") {
		t.Errorf("err = %v", err)
	}
}

func TestAccumulator_UsageDeltaMerged(t *testing.T) {
	acc := NewAccumulator("m", "p")
	_ = acc.Accumulate(UsageDelta{InputTokens: 10, OutputTokens: 5})
	_ = acc.Accumulate(UsageDelta{InputTokens: 12, OutputTokens: 8})
	_ = acc.Accumulate(Final{StopReason: StopReasonEndTurn})
	msg := acc.Message()
	if msg.Usage == nil {
		t.Fatal("Usage is nil")
	}
	if msg.Usage.Input != 12 || msg.Usage.Output != 8 {
		t.Errorf("Usage = %+v", msg.Usage)
	}
	if msg.Usage.TotalTokens != 20 {
		t.Errorf("TotalTokens = %d, want 20", msg.Usage.TotalTokens)
	}
}

func TestAccumulator_ResponseModelOverridesRequestModel(t *testing.T) {
	acc := NewAccumulator("requested-model", "p")
	_ = acc.Accumulate(Final{StopReason: StopReasonEndTurn, ResponseModel: "actual-model"})
	msg := acc.Message()
	if msg.Model != "actual-model" {
		t.Errorf("Model = %q, want actual-model", msg.Model)
	}
}

func TestAccumulator_ResponseIDCaptured(t *testing.T) {
	acc := NewAccumulator("m", "p")
	_ = acc.Accumulate(Final{StopReason: StopReasonEndTurn, ResponseID: "msg_abc123"})
	msg := acc.Message()
	if msg.ResponseID != "msg_abc123" {
		t.Errorf("ResponseID = %q", msg.ResponseID)
	}
}

func TestCollectStream_HappyPath(t *testing.T) {
	ch := make(chan Delta, 4)
	ch <- TextDelta{ContentIndex: 0, Text: "hi"}
	ch <- UsageDelta{InputTokens: 1, OutputTokens: 1}
	ch <- Final{StopReason: StopReasonEndTurn}
	close(ch)
	msg, err := CollectStream(context.Background(), ch, "m", "p")
	if err != nil {
		t.Fatalf("CollectStream: %v", err)
	}
	if len(msg.Content) != 1 {
		t.Errorf("Content len = %d", len(msg.Content))
	}
}

func TestCollectStream_MissingFinal(t *testing.T) {
	ch := make(chan Delta, 1)
	ch <- TextDelta{ContentIndex: 0, Text: "x"}
	close(ch)
	_, err := CollectStream(context.Background(), ch, "m", "p")
	if !errors.Is(err, ErrFinalMissing) {
		t.Errorf("err = %v, want ErrFinalMissing", err)
	}
}

func TestCollectStream_PropagatesFinalErr(t *testing.T) {
	wantErr := errors.New("rate limited")
	ch := make(chan Delta, 2)
	ch <- Final{Err: wantErr}
	close(ch)
	_, err := CollectStream(context.Background(), ch, "m", "p")
	if !errors.Is(err, wantErr) {
		t.Errorf("err = %v, want %v", err, wantErr)
	}
}

func TestCollectStream_RespectsCtxCancel(t *testing.T) {
	ch := make(chan Delta) // blocks forever, no close
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		cancel()
	}()
	_, err := CollectStream(ctx, ch, "m", "p")
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

func TestAccumulator_PartialSnapshot(t *testing.T) {
	// Calling Message() before Final should return a usable snapshot.
	acc := NewAccumulator("m", "p")
	_ = acc.Accumulate(TextDelta{ContentIndex: 0, Text: "part"})
	msg := acc.Message()
	if len(msg.Content) != 1 {
		t.Fatalf("snapshot Content len = %d", len(msg.Content))
	}
	// Continue accumulating.
	_ = acc.Accumulate(TextDelta{ContentIndex: 0, Text: " 2"})
	_ = acc.Accumulate(Final{StopReason: StopReasonEndTurn})
	msg = acc.Message()
	tc, _ := msg.Content[0].(TextContent)
	if tc.Text != "part 2" {
		t.Errorf("Text = %q", tc.Text)
	}
}

func TestAccumulator_EmptyInputDefaultsToEmptyJSON(t *testing.T) {
	// A tool_use with no input fragments should default to "{}" not "".
	acc := NewAccumulator("m", "p")
	_ = acc.Accumulate(ToolCallDelta{ContentIndex: 0, ID: "x", Name: "noop"})
	_ = acc.Accumulate(Final{StopReason: StopReasonToolUse})
	msg := acc.Message()
	tu, _ := msg.Content[0].(ToolUse)
	if string(tu.Input) != "{}" {
		t.Errorf("Input = %q, want {}", tu.Input)
	}
}
