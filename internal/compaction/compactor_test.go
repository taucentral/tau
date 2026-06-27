package compaction

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/state"
)

func TestMaybeCompact_NilCounter(t *testing.T) {
	c := &Compactor{Counter: nil}
	_, err := c.MaybeCompact(context.Background(), state.NewInMemoryManager("/x"), "m", 100)
	if err == nil {
		t.Errorf("expected error on nil Counter")
	}
}

func TestMaybeCompact_NilManager(t *testing.T) {
	c := NewCompactor(detCounter{charsPerToken: 1}, nil, 0)
	_, err := c.MaybeCompact(context.Background(), nil, "m", 100)
	if err == nil {
		t.Errorf("expected error on nil manager")
	}
}

func TestMaybeCompact_EmptySession(t *testing.T) {
	mgr := state.NewInMemoryManager("/x")
	c := NewCompactor(detCounter{charsPerToken: 1}, nil, 0)
	res, err := c.MaybeCompact(context.Background(), mgr, "m", 100)
	if err != nil {
		t.Fatalf("MaybeCompact: %v", err)
	}
	if res.Compacted {
		t.Errorf("Compacted = true on empty session")
	}
	if res.Reason == "" {
		t.Errorf("Reason should explain skip")
	}
}

func TestMaybeCompact_BelowThreshold(t *testing.T) {
	mgr := state.NewInMemoryManager("/x")
	// Append a small conversation.
	_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
		Role:    llm.RoleUser,
		Content: []llm.ContentBlock{llm.TextContent{Text: "hi"}},
	}})
	leaf, _ := mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
		Role:    llm.RoleAssistant,
		Content: []llm.ContentBlock{llm.TextContent{Text: "hello"}},
	}})

	c := NewCompactor(detCounter{charsPerToken: 1}, nil, 0)
	// Huge context window → no compaction needed.
	res, err := c.MaybeCompact(context.Background(), mgr, "test-model", 10_000)
	if err != nil {
		t.Fatalf("MaybeCompact: %v", err)
	}
	if res.Compacted {
		t.Errorf("Compacted = true; should be below threshold")
	}
	if res.PreCompactionTokens == 0 {
		t.Errorf("PreCompactionTokens should be set even on skip")
	}
	if res.Reason != "below threshold" {
		t.Errorf("Reason = %q, want 'below threshold'", res.Reason)
	}
	// Leaf unchanged (no Compaction entry appended).
	if mgr.LeafID() != leaf {
		t.Errorf("LeafID changed despite skip")
	}
}

func TestMaybeCompact_AboveThresholdCompacts(t *testing.T) {
	mgr := state.NewInMemoryManager("/x")
	// Long user message → exceeds tiny budget.
	_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
		Role:    llm.RoleUser,
		Content: []llm.ContentBlock{llm.TextContent{Text: strings.Repeat("a", 200)}},
	}})
	leaf, _ := mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
		Role:    llm.RoleAssistant,
		Content: []llm.ContentBlock{llm.TextContent{Text: strings.Repeat("b", 200)}},
	}})

	fc := &fakeLLM{
		deltas: []llm.Delta{
			llm.TextDelta{Text: "## Goal\nTest goal"},
			llm.Final{},
		},
	}
	summarizer := NewSummarizer(fc, "test-model")
	c := NewCompactor(detCounter{charsPerToken: 1}, summarizer, 0)
	// contextWindow = 100, ReserveTokens = DefaultReserveTokens(8192).
	// Threshold = 100 - 8192 = -8092. Everything triggers.
	res, err := c.MaybeCompact(context.Background(), mgr, "test-model", 100)
	if err != nil {
		t.Fatalf("MaybeCompact: %v", err)
	}
	if !res.Compacted {
		t.Fatalf("expected Compacted=true; got %+v", res)
	}
	if res.ArchivedCount == 0 {
		t.Errorf("ArchivedCount = 0, want > 0")
	}
	if res.CompactionEntryID == "" {
		t.Errorf("CompactionEntryID should be set")
	}
	if res.FirstKeptEntryID == "" {
		t.Errorf("FirstKeptEntryID should be set")
	}
	if !strings.Contains(res.Summary, "Test goal") {
		t.Errorf("Summary = %q, want 'Test goal'", res.Summary)
	}
	// Leaf should now be the new Compaction entry.
	if mgr.LeafID() != res.CompactionEntryID {
		t.Errorf("LeafID = %q, want %q", mgr.LeafID(), res.CompactionEntryID)
	}
	// Previous leaf should be different.
	if mgr.LeafID() == leaf {
		t.Errorf("leaf should have advanced past old leaf %q", leaf)
	}
}

func TestMaybeCompact_AboveThresholdNoSummarizerErrors(t *testing.T) {
	mgr := state.NewInMemoryManager("/x")
	_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
		Role:    llm.RoleUser,
		Content: []llm.ContentBlock{llm.TextContent{Text: strings.Repeat("a", 200)}},
	}})

	c := NewCompactor(detCounter{charsPerToken: 1}, nil, 0)
	_, err := c.MaybeCompact(context.Background(), mgr, "test-model", 10)
	if err == nil {
		t.Errorf("expected error when above-threshold and no summarizer")
	}
}

func TestMaybeCompact_PostCompactionTokensLower(t *testing.T) {
	mgr := state.NewInMemoryManager("/x")
	// 4 long messages → big pre-compaction count.
	for i := 0; i < 4; i++ {
		_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
			Role:    llm.RoleUser,
			Content: []llm.ContentBlock{llm.TextContent{Text: strings.Repeat("x", 100)}},
		}})
	}

	fc := &fakeLLM{
		deltas: []llm.Delta{
			llm.TextDelta{Text: "## Goal\nx"},
			llm.Final{},
		},
	}
	c := NewCompactor(detCounter{charsPerToken: 1}, NewSummarizer(fc, "test-model"), 0)
	res, err := c.MaybeCompact(context.Background(), mgr, "test-model", 100)
	if err != nil {
		t.Fatalf("MaybeCompact: %v", err)
	}
	if !res.Compacted {
		t.Fatalf("expected compaction; got %+v", res)
	}
	if res.PostCompactionTokens >= res.PreCompactionTokens {
		t.Errorf("post (%d) >= pre (%d); compaction should reduce tokens",
			res.PostCompactionTokens, res.PreCompactionTokens)
	}
}

func TestMaybeCompact_IterativeUpdateWithPreviousSummary(t *testing.T) {
	// First run: produce a Compaction entry. Second run: the compactor
	// must locate that prior summary and pass it to the summarizer (so
	// the UPDATE prompt runs, not the initial prompt).
	mgr := state.NewInMemoryManager("/x")
	for i := 0; i < 2; i++ {
		_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
			Role:    llm.RoleUser,
			Content: []llm.ContentBlock{llm.TextContent{Text: strings.Repeat("u", 100)}},
		}})
	}

	fc := &fakeLLM{
		deltas: []llm.Delta{
			llm.TextDelta{Text: "## Goal\nFirst"},
			llm.Final{},
		},
	}
	c := NewCompactor(detCounter{charsPerToken: 1}, NewSummarizer(fc, "test-model"), 0)
	if _, err := c.MaybeCompact(context.Background(), mgr, "test-model", 50); err != nil {
		t.Fatalf("first MaybeCompact: %v", err)
	}
	firstPrompt := fc.lastPromptText()
	if strings.Contains(firstPrompt, "Previous summary:") {
		t.Errorf("first compaction should use initial prompt: %s", firstPrompt)
	}

	// Add more entries to push above threshold again.
	for i := 0; i < 3; i++ {
		_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
			Role:    llm.RoleUser,
			Content: []llm.ContentBlock{llm.TextContent{Text: strings.Repeat("v", 100)}},
		}})
	}

	fc.deltas = []llm.Delta{
		llm.TextDelta{Text: "## Goal\nUpdated"},
		llm.Final{},
	}
	if _, err := c.MaybeCompact(context.Background(), mgr, "test-model", 50); err != nil {
		t.Fatalf("second MaybeCompact: %v", err)
	}
	secondPrompt := fc.lastPromptText()
	if !strings.Contains(secondPrompt, "Previous summary:") {
		t.Errorf("second compaction should use UPDATE prompt: %s", secondPrompt)
	}
	if !strings.Contains(secondPrompt, "## Goal\nFirst") {
		t.Errorf("UPDATE prompt should embed previous summary: %s", secondPrompt)
	}
}

func TestMaybeCompact_RespectsCustomReserveTokens(t *testing.T) {
	// reserveTokens > contextWindow → threshold negative → always triggers.
	// Use a small conversation and verify compaction runs.
	mgr := state.NewInMemoryManager("/x")
	_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
		Role:    llm.RoleUser,
		Content: []llm.ContentBlock{llm.TextContent{Text: "hi"}},
	}})

	fc := &fakeLLM{deltas: []llm.Delta{
		llm.TextDelta{Text: "## Goal\nx"},
		llm.Final{},
	}}
	c := NewCompactor(detCounter{charsPerToken: 1}, NewSummarizer(fc, "test-model"), 10_000)
	res, err := c.MaybeCompact(context.Background(), mgr, "test-model", 100)
	if err != nil {
		t.Fatalf("MaybeCompact: %v", err)
	}
	if !res.Compacted {
		t.Errorf("expected Compacted=true with reserve > window")
	}
}

func TestNewCompactor_DefaultsReserveTokens(t *testing.T) {
	c := NewCompactor(detCounter{charsPerToken: 1}, nil, 0)
	if c.ReserveTokens != DefaultReserveTokens {
		t.Errorf("ReserveTokens = %d, want default %d", c.ReserveTokens, DefaultReserveTokens)
	}
}

func TestNewCompactor_NegativeReserveBecomesDefault(t *testing.T) {
	c := NewCompactor(detCounter{charsPerToken: 1}, nil, -1)
	if c.ReserveTokens != DefaultReserveTokens {
		t.Errorf("ReserveTokens = %d, want default %d", c.ReserveTokens, DefaultReserveTokens)
	}
}

func TestMaybeCompact_ProtectionConfigPropagates(t *testing.T) {
	// A SessionInfo entry in the would-be-archived region must force the
	// cut deeper.
	mgr := state.NewInMemoryManager("/x")
	_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
		Role:    llm.RoleUser,
		Content: []llm.ContentBlock{llm.TextContent{Text: "old"}},
	}})
	// SessionInfo entry — protected.
	infoID, _ := mgr.Append(state.Entry{Kind: state.KindSessionInfo, Payload: state.SessionInfoPayload{Key: "k", Value: "v"}})
	_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
		Role:    llm.RoleUser,
		Content: []llm.ContentBlock{llm.TextContent{Text: strings.Repeat("x", 100)}},
	}})

	fc := &fakeLLM{deltas: []llm.Delta{
		llm.TextDelta{Text: "## Goal\nx"},
		llm.Final{},
	}}
	c := NewCompactor(detCounter{charsPerToken: 1}, NewSummarizer(fc, "test-model"), 0)
	res, err := c.MaybeCompact(context.Background(), mgr, "test-model", 50)
	if err != nil {
		t.Fatalf("MaybeCompact: %v", err)
	}
	if !res.Compacted {
		t.Fatalf("expected compaction")
	}
	// The FirstKeptEntryID should be at-or-older than the SessionInfo
	// entry (the cut was extended backward to include it).
	tr, _ := mgr.Tree()
	keptPath, err := tr.Path(res.FirstKeptEntryID)
	if err != nil {
		t.Fatalf("Path(%q): %v", res.FirstKeptEntryID, err)
	}
	found := false
	for _, e := range keptPath {
		if e.ID == infoID {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("protected SessionInfo %q not in kept path from leaf to %q", infoID, res.FirstKeptEntryID)
	}
}

func TestMaybeCompact_StreamErrorBubblesUp(t *testing.T) {
	// The fakeLLM returns an error from Stream start; the compactor
	// should wrap it and return an error.
	mgr := state.NewInMemoryManager("/x")
	_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
		Role:    llm.RoleUser,
		Content: []llm.ContentBlock{llm.TextContent{Text: strings.Repeat("x", 100)}},
	}})
	fc := &fakeLLM{err: errors.New("stream-start-failure")}
	c := NewCompactor(detCounter{charsPerToken: 1}, NewSummarizer(fc, "test-model"), 0)
	_, err := c.MaybeCompact(context.Background(), mgr, "test-model", 50)
	if err == nil || !strings.Contains(err.Error(), "stream-start-failure") {
		t.Errorf("expected wrapped 'stream-start-failure', got %v", err)
	}
}

func TestMaybeCompact_CompactionEntryPreservesSummary(t *testing.T) {
	mgr := state.NewInMemoryManager("/x")
	_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
		Role:    llm.RoleUser,
		Content: []llm.ContentBlock{llm.TextContent{Text: strings.Repeat("x", 100)}},
	}})
	fc := &fakeLLM{deltas: []llm.Delta{
		llm.TextDelta{Text: "## Goal\nTest\n## Constraints & Preferences\nnone"},
		llm.Final{},
	}}
	c := NewCompactor(detCounter{charsPerToken: 1}, NewSummarizer(fc, "test-model"), 0)
	res, err := c.MaybeCompact(context.Background(), mgr, "test-model", 50)
	if err != nil {
		t.Fatalf("MaybeCompact: %v", err)
	}
	// Inspect the new Compaction entry's payload.
	tr, _ := mgr.Tree()
	cp, ok := tr.Get(res.CompactionEntryID)
	if !ok {
		t.Fatalf("Compaction entry not in tree")
	}
	cpp, ok := cp.Payload.(state.CompactionPayload)
	if !ok {
		t.Fatalf("payload = %T, want CompactionPayload", cp.Payload)
	}
	if cpp.Summary != res.Summary {
		t.Errorf("payload Summary %q != result Summary %q", cpp.Summary, res.Summary)
	}
	if cpp.FirstKeptEntryID != res.FirstKeptEntryID {
		t.Errorf("payload FirstKeptEntryID %q != result %q", cpp.FirstKeptEntryID, res.FirstKeptEntryID)
	}
}

func TestMaybeCompact_SummarizerErrorBubblesUp(t *testing.T) {
	mgr := state.NewInMemoryManager("/x")
	_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
		Role:    llm.RoleUser,
		Content: []llm.ContentBlock{llm.TextContent{Text: strings.Repeat("x", 100)}},
	}})
	fc := &fakeLLM{err: errors.New("summarizer exploded")}
	c := NewCompactor(detCounter{charsPerToken: 1}, NewSummarizer(fc, "test-model"), 0)
	_, err := c.MaybeCompact(context.Background(), mgr, "test-model", 50)
	if err == nil || !strings.Contains(err.Error(), "summarizer exploded") {
		t.Errorf("expected wrapped 'summarizer exploded', got %v", err)
	}
}

func TestFindPreviousSummary_FindsMostRecent(t *testing.T) {
	walk := []state.Entry{
		mkMessage("u2", "c1", "user", "x", nowAt(0)),
		mkCompaction("c1", "c0", "OLD", "fke0", nowAt(0)),
		mkCompaction("c0", "h", "OLDER", "fkeH", nowAt(0)),
		mkSessionHeader("h"),
	}
	// walk[0] is leaf, walk[1] is most-recent Compaction.
	got := findPreviousSummary(walk, 1)
	if got != "OLD" {
		t.Errorf("findPreviousSummary = %q, want 'OLD'", got)
	}
}

func TestFindPreviousSummary_SearchesEntireWalk(t *testing.T) {
	// Per spec scenario "Iterative summary update", the previous summary
	// is passed to the UPDATE prompt regardless of whether its Compaction
	// entry sits in the kept or archived region. So findPreviousSummary
	// searches the whole walk.
	walk := []state.Entry{
		mkMessage("u2", "c1", "user", "x", nowAt(0)),
		mkCompaction("c1", "c0", "OLD", "fke0", nowAt(0)),
		mkSessionHeader("h"),
	}
	// cutIdx=0 → kept region is walk[0]; Compaction is at walk[1] (archived).
	// The function MUST still find it.
	got := findPreviousSummary(walk, 0)
	if got != "OLD" {
		t.Errorf("findPreviousSummary = %q, want 'OLD' (search whole walk)", got)
	}
}

func TestFindPreviousSummary_NoneAtAll(t *testing.T) {
	walk := []state.Entry{
		mkMessage("u1", "h", "user", "x", nowAt(0)),
		mkSessionHeader("h"),
	}
	got := findPreviousSummary(walk, 0)
	if got != "" {
		t.Errorf("findPreviousSummary = %q, want empty (no Compaction)", got)
	}
}
