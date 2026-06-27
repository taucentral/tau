package compaction

import (
	"context"
	"strings"
	"testing"

	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/state"
)

// TestIntegration_ArchivedEntriesStillVisibleInTree covers compaction spec
// scenario "Archived entries still visible in tree": after the compactor
// writes a Compaction entry, every archived entry remains reachable via
// Tree() and Path() — only BuildContext filters them out.
func TestIntegration_ArchivedEntriesStillVisibleInTree(t *testing.T) {
	mgr := state.NewInMemoryManager("/x")
	// 4 long user messages → force a cut.
	msgIDs := make([]string, 4)
	for i := 0; i < 4; i++ {
		id, err := mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
			Role:    llm.RoleUser,
			Content: []llm.ContentBlock{llm.TextContent{Text: strings.Repeat("x", 100)}},
		}})
		if err != nil {
			t.Fatalf("Append %d: %v", i, err)
		}
		msgIDs[i] = id
	}
	preTree, err := mgr.Tree()
	if err != nil {
		t.Fatalf("pre-Tree: %v", err)
	}
	preLen := preTree.Len()

	fc := &fakeLLM{deltas: []llm.Delta{
		llm.TextDelta{Text: "## Goal\nx"},
		llm.Final{},
	}}
	c := NewCompactor(detCounter{charsPerToken: 1}, NewSummarizer(fc, "test-model"), 0)
	if _, err := c.MaybeCompact(context.Background(), mgr, "test-model", 50); err != nil {
		t.Fatalf("MaybeCompact: %v", err)
	}

	postTree, err := mgr.Tree()
	if err != nil {
		t.Fatalf("post-Tree: %v", err)
	}
	// Post-compaction: every prior entry present PLUS the new Compaction entry.
	if postTree.Len() != preLen+1 {
		t.Errorf("post Tree Len = %d, want %d (pre + 1 compaction entry)", postTree.Len(), preLen+1)
	}
	// Every archived message ID is still reachable.
	for _, id := range msgIDs {
		if _, ok := postTree.Get(id); !ok {
			t.Errorf("archived message %q missing from Tree()", id)
		}
	}
}

// TestIntegration_ReBranchPastCompaction covers compaction spec scenario
// "Re-branch past a compaction": after compaction archives entries E10..E100,
// /checkout E50 moves the leaf to E50 and subsequent BuildContext calls
// include the pre-compaction state in context.
func TestIntegration_ReBranchPastCompaction(t *testing.T) {
	mgr := state.NewInMemoryManager("/x")
	// Build chain: u1 ("PRE-COMPACTION-MARKER"), u2..u4 (long, force cut).
	markerID, err := mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
		Role:    llm.RoleUser,
		Content: []llm.ContentBlock{llm.TextContent{Text: "PRE-COMPACTION-MARKER"}},
	}})
	if err != nil {
		t.Fatalf("Append marker: %v", err)
	}
	for i := 0; i < 3; i++ {
		_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
			Role:    llm.RoleUser,
			Content: []llm.ContentBlock{llm.TextContent{Text: strings.Repeat("y", 100)}},
		}})
	}

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

	// Sanity: post-compaction context should NOT contain the marker (it
	// was archived).
	ctx, err := mgr.BuildContext(context.Background())
	if err != nil {
		t.Fatalf("post-compact BuildContext: %v", err)
	}
	for _, m := range ctx.Messages {
		for _, b := range m.Content {
			if tc, ok := b.(llm.TextContent); ok {
				if strings.Contains(tc.Text, "PRE-COMPACTION-MARKER") {
					t.Errorf("PRE-COMPACTION-MARKER leaked into post-compact context")
				}
			}
		}
	}

	// Now re-branch back to the marker. The leaf moves; subsequent
	// BuildContext includes the pre-compaction state in context (the
	// marker is reachable root→leaf without hitting a Compaction entry
	// that would archive it).
	if err := mgr.SetLeaf(markerID); err != nil {
		t.Fatalf("SetLeaf(marker): %v", err)
	}
	if mgr.LeafID() != markerID {
		t.Errorf("LeafID = %q, want %q", mgr.LeafID(), markerID)
	}
	ctx2, err := mgr.BuildContext(context.Background())
	if err != nil {
		t.Fatalf("post-checkout BuildContext: %v", err)
	}
	// The marker MUST be in context now.
	found := false
	for _, m := range ctx2.Messages {
		for _, b := range m.Content {
			if tc, ok := b.(llm.TextContent); ok {
				if strings.Contains(tc.Text, "PRE-COMPACTION-MARKER") {
					found = true
				}
			}
		}
	}
	if !found {
		t.Errorf("after re-branch to %q, PRE-COMPACTION-MARKER missing from context", markerID)
	}
}

// TestIntegration_PostCompactionContextUsesMostRecentCompaction covers the
// state-tree spec scenario "Post-compaction context" end-to-end: when
// multiple Compaction entries exist, BuildContext uses the MOST RECENT one
// (the one closest to the leaf) and drops everything older than its
// FirstKeptEntryID.
func TestIntegration_PostCompactionContextUsesMostRecentCompaction(t *testing.T) {
	mgr := state.NewInMemoryManager("/x")
	_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
		Role:    llm.RoleUser,
		Content: []llm.ContentBlock{llm.TextContent{Text: "ancient"}},
	}})
	firstNewerID, _ := mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
		Role:    llm.RoleUser,
		Content: []llm.ContentBlock{llm.TextContent{Text: "after-first-compaction"}},
	}})
	// Old compaction: FirstKeptEntryID = firstNewerID (so "ancient" is archived).
	_, _ = mgr.Append(state.Entry{Kind: state.KindCompaction, Payload: state.CompactionPayload{
		Summary:          "FIRST SUMMARY",
		FirstKeptEntryID: firstNewerID,
	}})
	secondNewerID, _ := mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
		Role:    llm.RoleUser,
		Content: []llm.ContentBlock{llm.TextContent{Text: "after-second-compaction"}},
	}})
	// New compaction: FirstKeptEntryID = secondNewerID.
	_, _ = mgr.Append(state.Entry{Kind: state.KindCompaction, Payload: state.CompactionPayload{
		Summary:          "SECOND SUMMARY",
		FirstKeptEntryID: secondNewerID,
	}})

	ctx, err := mgr.BuildContext(context.Background())
	if err != nil {
		t.Fatalf("BuildContext: %v", err)
	}
	// Expected: [SECOND SUMMARY synthetic, "after-second-compaction"].
	// "FIRST SUMMARY" and "ancient" and "after-first-compaction" must NOT appear.
	if len(ctx.Messages) != 2 {
		t.Fatalf("Messages len = %d, want 2 (most-recent summary + 1 kept message): %+v", len(ctx.Messages), ctx.Messages)
	}
	for _, m := range ctx.Messages {
		for _, b := range m.Content {
			if tc, ok := b.(llm.TextContent); ok {
				for _, forbidden := range []string{"FIRST SUMMARY", "ancient", "after-first-compaction"} {
					if strings.Contains(tc.Text, forbidden) {
						t.Errorf("forbidden token %q in context: %q", forbidden, tc.Text)
					}
				}
			}
		}
	}
	// First message must be the SECOND summary synthetic.
	tc0, _ := ctx.Messages[0].Content[0].(llm.TextContent)
	if !strings.Contains(tc0.Text, "SECOND SUMMARY") {
		t.Errorf("Messages[0] = %q, want SECOND SUMMARY", tc0.Text)
	}
}

// TestIntegration_FileReadsSurviveCompaction covers compaction spec scenario
// "File reads survive compaction": when compaction archives entries with
// tool calls, the most-recent read of each distinct file (up to the cap)
// remains in context.
func TestIntegration_FileReadsSurviveCompaction(t *testing.T) {
	mgr := state.NewInMemoryManager("/x")
	// Build a chain where read tool calls land in the archived region,
	// then a long message forces a cut.
	paths := []string{"/file-a", "/file-b", "/file-c"}
	parent := ""
	for _, p := range paths {
		readMsg := state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
			Role: llm.RoleAssistant,
			Content: []llm.ContentBlock{
				llm.ToolUse{ID: "tu-" + p, Name: "read", Input: []byte(`{"path":"` + p + `"}`)},
			},
		}}
		id, err := mgr.Append(readMsg)
		if err != nil {
			t.Fatalf("Append read %s: %v", p, err)
		}
		_ = parent
		parent = id
	}
	// Long message → triggers compaction; cut goes past the read calls.
	_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
		Role:    llm.RoleUser,
		Content: []llm.ContentBlock{llm.TextContent{Text: strings.Repeat("z", 200)}},
	}})

	fc := &fakeLLM{deltas: []llm.Delta{
		llm.TextDelta{Text: "## Goal\nread files"},
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

	// The most-recent read of each distinct path must remain in context.
	// (Per spec: "the most recent read of C, B, and A (up to the cap)
	// remains in context".)
	ctx, err := mgr.BuildContext(context.Background())
	if err != nil {
		t.Fatalf("BuildContext: %v", err)
	}
	// Collect all ToolUse block paths still in context.
	gotPaths := map[string]bool{}
	for _, m := range ctx.Messages {
		for _, b := range m.Content {
			if tu, ok := b.(llm.ToolUse); ok {
				if tu.Name == "read" {
					// Path is in input JSON; extract.
					path := extractPath(tu.Input)
					if path != "" {
						gotPaths[path] = true
					}
				}
			}
		}
	}
	for _, want := range paths {
		if !gotPaths[want] {
			t.Errorf("expected read of %s to survive compaction; got paths = %v", want, gotPaths)
		}
	}
}

// TestIntegration_CompactionThenSummarizeChain verifies that running the
// compactor twice produces an UPDATE summarizer call (carrying the prior
// summary) the second time.
func TestIntegration_CompactionThenSummarizeChain(t *testing.T) {
	mgr := state.NewInMemoryManager("/x")
	// First batch of long messages.
	for i := 0; i < 3; i++ {
		_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
			Role:    llm.RoleUser,
			Content: []llm.ContentBlock{llm.TextContent{Text: strings.Repeat("a", 100)}},
		}})
	}

	fc := &fakeLLM{deltas: []llm.Delta{
		llm.TextDelta{Text: "## Goal\nfirst"},
		llm.Final{},
	}}
	c := NewCompactor(detCounter{charsPerToken: 1}, NewSummarizer(fc, "test-model"), 0)

	if _, err := c.MaybeCompact(context.Background(), mgr, "test-model", 50); err != nil {
		t.Fatalf("first MaybeCompact: %v", err)
	}
	if strings.Contains(fc.lastPromptText(), "Previous summary:") {
		t.Errorf("first compaction should use initial prompt, not UPDATE")
	}

	// Append more messages → triggers second compaction.
	for i := 0; i < 3; i++ {
		_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
			Role:    llm.RoleUser,
			Content: []llm.ContentBlock{llm.TextContent{Text: strings.Repeat("b", 100)}},
		}})
	}
	fc.deltas = []llm.Delta{
		llm.TextDelta{Text: "## Goal\nsecond"},
		llm.Final{},
	}
	if _, err := c.MaybeCompact(context.Background(), mgr, "test-model", 50); err != nil {
		t.Fatalf("second MaybeCompact: %v", err)
	}
	if !strings.Contains(fc.lastPromptText(), "Previous summary:") {
		t.Errorf("second compaction should use UPDATE prompt")
	}
	if !strings.Contains(fc.lastPromptText(), "## Goal\nfirst") {
		t.Errorf("UPDATE prompt missing previous summary text")
	}
}

// TestIntegration_CompactionPreservesToolPairIntegrity covers compaction
// spec scenario "Never split a tool pair": a cut must not orphan a ToolUse
// by archiving its matching ToolResult (or vice versa).
func TestIntegration_CompactionPreservesToolPairIntegrity(t *testing.T) {
	mgr := state.NewInMemoryManager("/x")
	// ToolUse (assistant) → ToolResult (user).
	_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
		Role: llm.RoleAssistant,
		Content: []llm.ContentBlock{
			llm.ToolUse{ID: "tu1", Name: "read", Input: []byte(`{"path":"/x"}`)},
		},
	}})
	_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
		Role: llm.RoleUser,
		Content: []llm.ContentBlock{
			llm.ToolResult{ToolUseID: "tu1", Content: []llm.ContentBlock{llm.TextContent{Text: "result-of-x"}}},
		},
	}})
	// Long user message → triggers compaction with a tight budget.
	_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{
		Role:    llm.RoleUser,
		Content: []llm.ContentBlock{llm.TextContent{Text: strings.Repeat("q", 100)}},
	}})

	fc := &fakeLLM{deltas: []llm.Delta{
		llm.TextDelta{Text: "## Goal\nx"},
		llm.Final{},
	}}
	c := NewCompactor(detCounter{charsPerToken: 1}, NewSummarizer(fc, "test-model"), 0)
	if _, err := c.MaybeCompact(context.Background(), mgr, "test-model", 50); err != nil {
		t.Fatalf("MaybeCompact: %v", err)
	}

	ctx, err := mgr.BuildContext(context.Background())
	if err != nil {
		t.Fatalf("BuildContext: %v", err)
	}
	// If either half of the tool pair survived, BOTH halves must be
	// present (no orphaned ToolUse or ToolResult).
	hasUse, hasResult := false, false
	for _, m := range ctx.Messages {
		for _, b := range m.Content {
			if tu, ok := b.(llm.ToolUse); ok && tu.ID == "tu1" {
				hasUse = true
			}
			if tr, ok := b.(llm.ToolResult); ok && tr.ToolUseID == "tu1" {
				hasResult = true
			}
		}
	}
	if hasUse != hasResult {
		t.Errorf("tool pair split: hasUse=%v hasResult=%v (must match)", hasUse, hasResult)
	}
}
