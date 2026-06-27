package compaction

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/state"
)

func TestSummarizer_NilClientReturnsError(t *testing.T) {
	s := &Summarizer{Client: nil, Model: "x"}
	_, err := s.Summarize(context.Background(), "", nil, FileTracking{})
	if err == nil {
		t.Errorf("expected error on nil client")
	}
}

func TestSummarizer_FirstTimeUsesInitialPrompt(t *testing.T) {
	fc := &fakeLLM{
		deltas: []llm.Delta{
			llm.TextDelta{Text: "## Goal\nDo the thing"},
			llm.Final{},
		},
	}
	s := NewSummarizer(fc, "test-model")
	out, err := s.Summarize(context.Background(), "", []state.Entry{
		mkMessage("u1", "h", "user", "hello", nowAt(0)),
	}, FileTracking{})
	if err != nil {
		t.Fatalf("Summarize: %v", err)
	}
	if !strings.Contains(out, "## Goal") {
		t.Errorf("summary missing '## Goal': %q", out)
	}
	prompt := fc.lastPromptText()
	if strings.Contains(prompt, "Previous summary:") {
		t.Errorf("first-time prompt should not contain 'Previous summary:', got: %q", prompt)
	}
	if !strings.Contains(prompt, "Conversation to summarize") {
		t.Errorf("first-time prompt missing 'Conversation to summarize': %q", prompt)
	}
}

func TestSummarizer_UpdateUsesUpdatePrompt(t *testing.T) {
	fc := &fakeLLM{
		deltas: []llm.Delta{
			llm.TextDelta{Text: "## Goal\nUpdated"},
			llm.Final{},
		},
	}
	s := NewSummarizer(fc, "test-model")
	prev := "## Goal\nOld goal"
	out, err := s.Summarize(context.Background(), prev, []state.Entry{
		mkMessage("u1", "h", "user", "more", nowAt(0)),
	}, FileTracking{})
	if err != nil {
		t.Fatalf("Summarize: %v", err)
	}
	if !strings.Contains(out, "Updated") {
		t.Errorf("expected updated summary, got %q", out)
	}
	prompt := fc.lastPromptText()
	if !strings.Contains(prompt, "Previous summary:") {
		t.Errorf("update prompt should contain 'Previous summary:', got: %q", prompt)
	}
	if !strings.Contains(prompt, prev) {
		t.Errorf("update prompt should embed previous summary, got: %q", prompt)
	}
}

func TestSummarizer_FileTrackingIncludedInPrompt(t *testing.T) {
	fc := &fakeLLM{
		deltas: []llm.Delta{
			llm.TextDelta{Text: "## Goal\nx"},
			llm.Final{},
		},
	}
	s := NewSummarizer(fc, "test-model")
	ft := FileTracking{
		Reads: []FileOperation{{Path: "/a", Operation: "read"}},
		Modifications: []FileOperation{
			{Path: "/b", Operation: "modify"},
		},
	}
	_, err := s.Summarize(context.Background(), "", nil, ft)
	if err != nil {
		t.Fatalf("Summarize: %v", err)
	}
	prompt := fc.lastPromptText()
	if !strings.Contains(prompt, "/a") || !strings.Contains(prompt, "/b") {
		t.Errorf("prompt missing file paths: %s", prompt)
	}
	if !strings.Contains(prompt, "Files read:") {
		t.Errorf("prompt missing 'Files read:' header")
	}
}

func TestSummarizer_StreamStartError(t *testing.T) {
	fc := &fakeLLM{err: errors.New("boom")}
	s := NewSummarizer(fc, "test-model")
	_, err := s.Summarize(context.Background(), "", nil, FileTracking{})
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Errorf("expected wrapped 'boom' error, got %v", err)
	}
}

func TestSummarizer_FinalErrorReturned(t *testing.T) {
	fc := &fakeLLM{
		deltas: []llm.Delta{
			llm.Final{Err: errors.New("provider exploded")},
		},
	}
	s := NewSummarizer(fc, "test-model")
	_, err := s.Summarize(context.Background(), "", nil, FileTracking{})
	if err == nil || !strings.Contains(err.Error(), "provider exploded") {
		t.Errorf("expected wrapped 'provider exploded', got %v", err)
	}
}

func TestSummarizer_NoFinalReturnsErrFinalMissing(t *testing.T) {
	// Channel closes without emitting Final.
	fc := &fakeLLM{
		deltas: []llm.Delta{
			llm.TextDelta{Text: "no final"},
		},
	}
	s := NewSummarizer(fc, "test-model")
	_, err := s.Summarize(context.Background(), "", nil, FileTracking{})
	if !errors.Is(err, llm.ErrFinalMissing) {
		t.Errorf("expected ErrFinalMissing, got %v", err)
	}
}

func TestSummarizer_EmptyOutputReturnsError(t *testing.T) {
	// Output is just whitespace → trimmed to empty → must error.
	fc := &fakeLLM{
		deltas: []llm.Delta{
			llm.TextDelta{Text: "   \n\t  "},
			llm.Final{},
		},
	}
	s := NewSummarizer(fc, "test-model")
	_, err := s.Summarize(context.Background(), "", nil, FileTracking{})
	if err == nil || !strings.Contains(err.Error(), "empty summary") {
		t.Errorf("expected 'empty summary' error, got %v", err)
	}
}

func TestSummarizer_StreamRequestShape(t *testing.T) {
	fc := &fakeLLM{
		deltas: []llm.Delta{
			llm.TextDelta{Text: "ok"},
			llm.Final{},
		},
	}
	s := NewSummarizer(fc, "test-model-123")
	_, err := s.Summarize(context.Background(), "", []state.Entry{
		mkMessage("u1", "h", "user", "hi", nowAt(0)),
	}, FileTracking{})
	if err != nil {
		t.Fatalf("Summarize: %v", err)
	}
	if fc.last.Model != "test-model-123" {
		t.Errorf("request Model = %q, want test-model-123", fc.last.Model)
	}
	if fc.last.MaxTokens == nil || *fc.last.MaxTokens != summarizationMaxTokens {
		if fc.last.MaxTokens != nil {
			t.Errorf("MaxTokens = %d, want %d", *fc.last.MaxTokens, summarizationMaxTokens)
		} else {
			t.Errorf("MaxTokens is nil, want %d", summarizationMaxTokens)
		}
	}
	if len(fc.last.Messages) != 1 {
		t.Fatalf("Messages len = %d, want 1", len(fc.last.Messages))
	}
	if fc.last.Messages[0].Role != llm.RoleUser {
		t.Errorf("Messages[0].Role = %q, want user", fc.last.Messages[0].Role)
	}
}

func TestSummarizer_ContextCancellation_StartError(t *testing.T) {
	// Cancel context before Stream is invoked; fakeLLM's goroutine honors
	// ctx via the Stream-start error path (cancel before goroutine launch
	// surfaces as the configured error).
	fc := &fakeLLM{err: context.Canceled}
	s := NewSummarizer(fc, "test-model")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := s.Summarize(ctx, "", nil, FileTracking{})
	if err == nil {
		t.Errorf("expected error on canceled ctx, got nil")
	}
}

func TestRenderEntriesForPrompt_Kinds(t *testing.T) {
	entries := []state.Entry{
		mkMessage("u1", "h", "user", "hello", nowAt(0)),
		mkMessage("a1", "u1", "assistant", "world", nowAt(0)),
		mkBranchSummary("b1", "a1", "forked", nowAt(0)),
		mkCompaction("c1", "b1", "earlier summary", "x", nowAt(0)),
		mkLabel("l1", "c1", "tag", nowAt(0)),
	}
	got := renderEntriesForPrompt(entries)
	if !strings.Contains(got, "[user]") {
		t.Errorf("missing [user]: %s", got)
	}
	if !strings.Contains(got, "[assistant]") {
		t.Errorf("missing [assistant]: %s", got)
	}
	if !strings.Contains(got, "[branch-summary]") {
		t.Errorf("missing [branch-summary]: %s", got)
	}
	if !strings.Contains(got, "[prior-compaction]") {
		t.Errorf("missing [prior-compaction]: %s", got)
	}
	if !strings.Contains(got, "earlier summary") {
		t.Errorf("missing prior summary text: %s", got)
	}
	// Labels should NOT be rendered.
	if strings.Contains(got, "[label]") {
		t.Errorf("labels should not appear: %s", got)
	}
}

func TestRenderEntriesForPrompt_Empty(t *testing.T) {
	got := renderEntriesForPrompt(nil)
	if got != "(no conversational entries)" {
		t.Errorf("empty render = %q, want placeholder", got)
	}
}

func TestRenderEntriesForPrompt_ToolUseToolResult(t *testing.T) {
	entries := []state.Entry{
		mkToolUseMessage("a1", "h", "read", "tu1", map[string]any{"path": "/x"}, nowAt(0)),
		mkToolResultMessage("u1", "a1", "tu1", "content-of-x", nowAt(0)),
	}
	got := renderEntriesForPrompt(entries)
	if !strings.Contains(got, "(tool read") {
		t.Errorf("missing tool rendering: %s", got)
	}
	if !strings.Contains(got, "(tool-result") {
		t.Errorf("missing tool-result rendering: %s", got)
	}
	if !strings.Contains(got, "content-of-x") {
		t.Errorf("missing tool-result content: %s", got)
	}
}

func TestRenderEntriesForPrompt_SkipsThinking(t *testing.T) {
	// ThinkingContent is omitted from the rendered prompt.
	entries := []state.Entry{
		{
			Kind:      state.KindMessage,
			Timestamp: nowAt(0),
			Payload: state.MessagePayload{
				Role: llm.RoleAssistant,
				Content: []llm.ContentBlock{
					llm.ThinkingContent{Thinking: "internal-reasoning"},
					llm.TextContent{Text: "answer"},
				},
			},
		},
	}
	got := renderEntriesForPrompt(entries)
	if strings.Contains(got, "internal-reasoning") {
		t.Errorf("thinking content should be elided: %s", got)
	}
	if !strings.Contains(got, "answer") {
		t.Errorf("text content missing: %s", got)
	}
}

func TestSummarizationPrompts_NonEmpty(t *testing.T) {
	// Smoke: the prompt templates exist and are non-trivially long.
	if len(SummarizationPrompt) < 100 {
		t.Errorf("SummarizationPrompt is suspiciously short: %d", len(SummarizationPrompt))
	}
	if len(UpdateSummarizationPrompt) < 100 {
		t.Errorf("UpdateSummarizationPrompt is suspiciously short: %d", len(UpdateSummarizationPrompt))
	}
}

func TestSummarizationMaxTokens_Reasonable(t *testing.T) {
	// The cap should be high enough to fit all six sections without
	// truncation, but not so high that the model rambles.
	if summarizationMaxTokens < 256 || summarizationMaxTokens > 8192 {
		t.Errorf("summarizationMaxTokens = %d, out of [256, 8192]", summarizationMaxTokens)
	}
}
