package compaction

import (
	"context"
	"errors"
	"log"
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
	c := NewCompactor(detCounter{charsPerToken: 1}, nil, 0, 0, 0)
	_, err := c.MaybeCompact(context.Background(), nil, "m", 100)
	if err == nil {
		t.Errorf("expected error on nil manager")
	}
}

func TestMaybeCompact_EmptySession(t *testing.T) {
	mgr := state.NewInMemoryManager("/x")
	c := NewCompactor(detCounter{charsPerToken: 1}, nil, 0, 0, 0)
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

	c := NewCompactor(detCounter{charsPerToken: 1}, nil, 0, 0, 0)
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
	c := NewCompactor(detCounter{charsPerToken: 1}, summarizer, 0, 0, 0)
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

	c := NewCompactor(detCounter{charsPerToken: 1}, nil, 0, 0, 0)
	_, err := c.MaybeCompact(context.Background(), mgr, "test-model", 10)
	if err == nil {
		t.Errorf("expected error when above-threshold and no summarizer")
	}
}

func TestMaybeCompact_PostCompactionTokensLower(t *testing.T) {
	// Walk (leaf→root): 4 user messages × 104 tokens = 416 tokens, plus header(0).
	// Floor = 50 (small). cutPoints = [0,1,2,3]. Default cut = 3.
	// Loop:
	//   i=0: running=104. >=50! Largest cutPoint<=0 is 0. cutIndex=0. break.
	// Cut = 0 (leaf only kept). Archived = walk[1..4] = u3, u2, u1, header.
	// Pre = 416. Post = walk[0] (104) + Compaction entry (~9) = 113. Post < pre.
	mgr := state.NewInMemoryManager("/x")
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
	c := NewCompactor(detCounter{charsPerToken: 1}, NewSummarizer(fc, "test-model"), 0, 50, 0)
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
	c := NewCompactor(detCounter{charsPerToken: 1}, NewSummarizer(fc, "test-model"), 0, 0, 0)
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
	c := NewCompactor(detCounter{charsPerToken: 1}, NewSummarizer(fc, "test-model"), 10_000, 0, 0)
	res, err := c.MaybeCompact(context.Background(), mgr, "test-model", 100)
	if err != nil {
		t.Fatalf("MaybeCompact: %v", err)
	}
	if !res.Compacted {
		t.Errorf("expected Compacted=true with reserve > window")
	}
}

func TestNewCompactor_DefaultsReserveTokens(t *testing.T) {
	c := NewCompactor(detCounter{charsPerToken: 1}, nil, 0, 0, 0)
	if c.ReserveTokens != DefaultReserveTokens {
		t.Errorf("ReserveTokens = %d, want default %d", c.ReserveTokens, DefaultReserveTokens)
	}
}

func TestNewCompactor_NegativeReserveBecomesDefault(t *testing.T) {
	c := NewCompactor(detCounter{charsPerToken: 1}, nil, -1, 0, 0)
	if c.ReserveTokens != DefaultReserveTokens {
		t.Errorf("ReserveTokens = %d, want default %d", c.ReserveTokens, DefaultReserveTokens)
	}
}

func TestMaybeCompact_ProtectionConfigPropagates(t *testing.T) {
	// Protected SessionInfo entries that fall in the would-be-archived
	// region must be pulled into the kept region (the cut is extended
	// backward to include them). This test sets up a walk where the
	// floor forces a cut that would archive the SessionInfo, and verifies
	// the cut is extended.
	//
	// Walk (leaf→root): user2(0, big), info(1, SessionInfo), user1(2, small), header(3, 0).
	// With a small floor, the cut would normally be near the leaf (user2),
	// archiving info, user1, header. The protected info forces the cut to
	// extend backward (toward root) past it.
	mgr := state.NewInMemoryManager("/x")
	_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
		Role:    llm.RoleUser,
		Content: []llm.ContentBlock{llm.TextContent{Text: strings.Repeat("x", 100)}},
	}})
	// SessionInfo entry — protected.
	infoID, _ := mgr.Append(state.Entry{Kind: state.KindSessionInfo, Payload: state.SessionInfoPayload{Key: "k", Value: "v"}})
	_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
		Role:    llm.RoleUser,
		Content: []llm.ContentBlock{llm.TextContent{Text: "old"}},
	}})

	fc := &fakeLLM{deltas: []llm.Delta{
		llm.TextDelta{Text: "## Goal\nx"},
		llm.Final{},
	}}
	// Floor = 5. With walk tokens [user2=104, info=0, user1=7, header=0]:
	//   cutPoints = [0, 2] (user2, user1; info not eligible).
	//   Default cutIndex = 2. Loop:
	//     i=0: running=104. >=5! Largest cutPoint<=0 is 0. cutIndex=0. break.
	//   Initial cutIndex = 0 (user2). Protected scan:
	//     j=1 (info): protected → cut=1.
	//     j=2 (user1): not protected.
	//     j=3 (header): not protected.
	//   Final cut = 1 (info is the oldest kept). user1 and header archived.
	// FirstKeptEntryID = walk[1].ID = infoID. So the cut was extended
	// backward from user2 (0) to info (1) to protect info.
	c := NewCompactor(detCounter{charsPerToken: 1}, NewSummarizer(fc, "test-model"), 0, 5, 0)
	res, err := c.MaybeCompact(context.Background(), mgr, "test-model", 50)
	if err != nil {
		t.Fatalf("MaybeCompact: %v", err)
	}
	if !res.Compacted {
		t.Fatalf("expected compaction")
	}
	if res.FirstKeptEntryID != infoID {
		t.Errorf("FirstKeptEntryID = %q, want %q (protected info, cut extended backward)",
			res.FirstKeptEntryID, infoID)
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
	c := NewCompactor(detCounter{charsPerToken: 1}, NewSummarizer(fc, "test-model"), 0, 0, 0)
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
	c := NewCompactor(detCounter{charsPerToken: 1}, NewSummarizer(fc, "test-model"), 0, 0, 0)
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
	c := NewCompactor(detCounter{charsPerToken: 1}, NewSummarizer(fc, "test-model"), 0, 0, 0)
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

// Test 5.1: integration test for the floor semantic end-to-end through
// MaybeCompact. Constructs a walk with ~50000 tokens, runs MaybeCompact with
// keepRecentTokens=20000, and asserts the kept region (PostCompactionTokens,
// excluding the Compaction entry's contribution) stays at or below 20000.
//
// Per the floor semantic (design.md D1.1, D1.2): FindCutPoint walks backward
// accumulating tokens, cuts at the closest eligible cut at-or-newer than the
// floor crossing. The kept region is therefore approximately == keepRecentTokens
// (modulo the granularity of message boundaries).
func TestMaybeCompact_FloorKeepsAtMostKeepRecentTokens(t *testing.T) {
	mgr := state.NewInMemoryManager("/x")
	// 500 user messages × 100 chars = 500 × 104 = 52000 tokens. Each msg
	// = 100 chars + 4 framing = 104 tokens (detCounter{charsPerToken:1}).
	for i := 0; i < 500; i++ {
		_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
			Role:    llm.RoleUser,
			Content: []llm.ContentBlock{llm.TextContent{Text: strings.Repeat("x", 100)}},
		}})
	}

	fc := &fakeLLM{deltas: []llm.Delta{
		llm.TextDelta{Text: "## Goal\nx"},
		llm.Final{},
	}}
	// keepRecentTokens = 20000; contextWindow = 60000 (>20000 so clamp
	// doesn't fire; clamp limit is 30000).
	c := NewCompactor(detCounter{charsPerToken: 1}, NewSummarizer(fc, "test-model"), 0, 20000, 60000)
	// contextWindow = 50000, ReserveTokens = default 8192. Threshold = 41808.
	// Total = ~52000 > 41808 → triggers compaction.
	res, err := c.MaybeCompact(context.Background(), mgr, "test-model", 50000)
	if err != nil {
		t.Fatalf("MaybeCompact: %v", err)
	}
	if !res.Compacted {
		t.Fatalf("expected compaction; got %+v", res)
	}
	// The kept region's tokens (pre-Compaction-entry) = walk[0..cutIdx].
	// PostCompactionTokens includes the Compaction entry's summary (~9
	// tokens). The kept region alone is PostCompactionTokens minus the
	// summary's contribution. We want the kept region ≤ 20000 + one message's
	// tolerance (the floor crossing happens at message granularity).
	//
	// Summary text "## Goal\nx" = 9 chars = 9 tokens + 4 framing = 13.
	// kept region ≈ PostCompactionTokens - 13.
	keptRegionTokens := res.PostCompactionTokens - 13
	// Allow up to 104 tokens of overage (one message) because the floor
	// is crossed at message granularity.
	if keptRegionTokens > 20000+104 {
		t.Errorf("kept region = %d tokens, want ≤ %d (floor 20000 + one message tolerance)",
			keptRegionTokens, 20000+104)
	}
	// Kept region must also be at least 20000 (the floor minus one message
	// of underflow — we stop AT the crossing, so we may stop just below).
	if keptRegionTokens < 20000-104 {
		t.Errorf("kept region = %d tokens, want ≥ %d (floor 20000 − one message tolerance)",
			keptRegionTokens, 20000-104)
	}
}

// Test 5.2: clamp integration test. Constructs a Compactor with an oversized
// keepRecentTokens (500000) and a finite contextWindow (200000), asserts the
// stored c.keepRecent is clamped to 100000 (contextWindow/2), and the warning
// log fires with the expected message.
func TestNewCompactor_ClampsKeepRecentToContextWindowHalf(t *testing.T) {
	// Capture log output. log.Printf writes to the standard logger; swap
	// its output for a buffer we can inspect, and restore it on exit.
	var buf strings.Builder
	origOut := log.Writer()
	origFlags := log.Flags()
	log.SetOutput(&buf)
	log.SetFlags(0) // drop timestamps for a stable substring match
	t.Cleanup(func() {
		log.SetOutput(origOut)
		log.SetFlags(origFlags)
	})

	c := NewCompactor(detCounter{charsPerToken: 1}, nil, 0, 500000, 200000)
	if c.keepRecent != 100000 {
		t.Errorf("keepRecent = %d, want 100000 (clamped to contextWindow/2 = 200000/2)",
			c.keepRecent)
	}
	logMsg := buf.String()
	if !strings.Contains(logMsg, "clamped keepRecentTokens from 500000 to 100000") {
		t.Errorf("log output = %q, want substring 'clamped keepRecentTokens from 500000 to 100000'", logMsg)
	}
	if !strings.Contains(logMsg, "contextWindow/2") {
		t.Errorf("log output = %q, want substring 'contextWindow/2'", logMsg)
	}
}

// Test 5.2b: when keepRecentTokens ≤ contextWindow/2, no clamping fires and
// no warning is logged. Guards against false-positive warnings.
func TestNewCompactor_NoClampWhenKeepRecentUnderHalf(t *testing.T) {
	var buf strings.Builder
	origOut := log.Writer()
	origFlags := log.Flags()
	log.SetOutput(&buf)
	log.SetFlags(0)
	t.Cleanup(func() {
		log.SetOutput(origOut)
		log.SetFlags(origFlags)
	})

	c := NewCompactor(detCounter{charsPerToken: 1}, nil, 0, 20000, 200000)
	if c.keepRecent != 20000 {
		t.Errorf("keepRecent = %d, want 20000 (under clamp, no change)", c.keepRecent)
	}
	if buf.String() != "" {
		t.Errorf("log output = %q, want empty (no clamp expected)", buf.String())
	}
}
