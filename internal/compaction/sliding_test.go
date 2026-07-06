package compaction

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/taucentral/tau/internal/state"
)

func TestArchive_WritesCompactionEntry(t *testing.T) {
	mgr := state.NewInMemoryManager("/cwd")
	// Build a tree: header + u1 + a1 + u2 (leaf).
	u1, _ := mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{Role: "user"}})
	_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{Role: "assistant"}})
	u2, _ := mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{Role: "user"}})

	tree, err := mgr.Tree()
	if err != nil {
		t.Fatalf("Tree: %v", err)
	}
	walk, err := tree.WalkFromLeaf(u2)
	if err != nil {
		t.Fatalf("WalkFromLeaf: %v", err)
	}
	// walk[0]=u2, walk[1]=a1, walk[2]=u1, walk[3]=header (leaf→root).
	// Cut at index 2 → keep u2, a1; archive u1.
	// Note: header is at index 3, but we are not archiving it.
	summary := "## Goal\nTest summary\n"
	res, err := Archive(mgr, walk, 2, summary)
	if err != nil {
		t.Fatalf("Archive: %v", err)
	}
	if res.ArchivedCount != 1 {
		t.Errorf("ArchivedCount = %d, want 1", res.ArchivedCount)
	}
	// FirstKeptEntryID should be walk[2].ID = u1 (the oldest KEPT entry).
	if res.FirstKeptEntryID != u1 {
		t.Errorf("FirstKeptEntryID = %q, want %q", res.FirstKeptEntryID, u1)
	}
	// A Compaction entry should have been appended.
	if mgr.LeafID() != res.CompactionEntryID {
		t.Errorf("LeafID = %q, want CompactionEntryID %q", mgr.LeafID(), res.CompactionEntryID)
	}
	tree2, err := mgr.Tree()
	if err != nil {
		t.Fatalf("Tree after Archive: %v", err)
	}
	cp, ok := tree2.Get(res.CompactionEntryID)
	if !ok {
		t.Fatalf("Compaction entry not in tree")
	}
	cpp, _ := cp.Payload.(state.CompactionPayload)
	if cpp.Summary != summary {
		t.Errorf("stored summary = %q, want %q", cpp.Summary, summary)
	}
	if cpp.FirstKeptEntryID != u1 {
		t.Errorf("stored FirstKeptEntryID = %q, want %q", cpp.FirstKeptEntryID, u1)
	}
}

func TestArchive_InvalidCutIndex(t *testing.T) {
	mgr := state.NewInMemoryManager("/cwd")
	id, _ := mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{Role: "user"}})
	tree, _ := mgr.Tree()
	walk, _ := tree.WalkFromLeaf(id)

	cases := []int{-1, -2, len(walk), len(walk) + 5}
	for _, cut := range cases {
		if _, err := Archive(mgr, walk, cut, "x"); err == nil {
			t.Errorf("cut=%d: expected error, got nil", cut)
		}
	}
}

func TestArchive_CutAtLeafArchivesRest(t *testing.T) {
	// Cut at index 0 (the leaf) keeps walk[0] and archives walk[1:].
	// ArchivedCount = len(walk) - 0 - 1 = 1 (the header in this small tree).
	mgr := state.NewInMemoryManager("/cwd")
	id, _ := mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{Role: "user"}})
	tree, _ := mgr.Tree()
	walk, _ := tree.WalkFromLeaf(id)
	res, err := Archive(mgr, walk, 0, "summary")
	if err != nil {
		t.Fatalf("Archive: %v", err)
	}
	if res.ArchivedCount != len(walk)-1 {
		t.Errorf("ArchivedCount = %d, want %d (everything older than leaf)", res.ArchivedCount, len(walk)-1)
	}
	if res.FirstKeptEntryID != id {
		t.Errorf("FirstKeptEntryID = %q, want %q", res.FirstKeptEntryID, id)
	}
}

func TestArchive_CutAtRootArchivesNothing(t *testing.T) {
	// Cut at the last walk index (the root) means everything is kept;
	// nothing is older than the root, so ArchivedCount = 0. This mirrors
	// the compactor's `cutIdx >= len(walk)-1` skip branch.
	mgr := state.NewInMemoryManager("/cwd")
	_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{Role: "user"}})
	u2, _ := mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{Role: "user"}})
	tree, _ := mgr.Tree()
	walk, _ := tree.WalkFromLeaf(u2)
	rootIdx := len(walk) - 1
	res, err := Archive(mgr, walk, rootIdx, "summary")
	if err != nil {
		t.Fatalf("Archive: %v", err)
	}
	if res.ArchivedCount != 0 {
		t.Errorf("ArchivedCount = %d, want 0 (cut at root)", res.ArchivedCount)
	}
}

func TestArchive_DoesNotDeleteArchivedEntries(t *testing.T) {
	// Per spec scenario "Archived entries still visible in tree", Archive
	// does NOT remove archived entries — they stay in the tree and remain
	// visible via Tree()/Path(). Only BuildContext filters them out (via
	// FirstKeptEntryID on the Compaction entry).
	mgr := state.NewInMemoryManager("/cwd")
	_, _ = mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{Role: "user"}})
	u2, _ := mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{Role: "user"}})
	preTreeLen := func() int {
		t.Helper()
		tr, err := mgr.Tree()
		if err != nil {
			t.Fatal(err)
		}
		return tr.Len()
	}()
	tree, _ := mgr.Tree()
	walk, _ := tree.WalkFromLeaf(u2)
	// Cut at 1 (keep u2; archive u1).
	if _, err := Archive(mgr, walk, 1, "summary"); err != nil {
		t.Fatalf("Archive: %v", err)
	}
	// Post-archive: tree grew by 1 (the new Compaction entry), no deletions.
	post := func() int {
		tr, err := mgr.Tree()
		if err != nil {
			t.Fatal(err)
		}
		return tr.Len()
	}()
	if post != preTreeLen+1 {
		t.Errorf("post-archive tree Len = %d, want %d (entries preserved + 1 compaction entry)", post, preTreeLen+1)
	}
}

func TestArchive_SummaryWithSpecialChars(t *testing.T) {
	// The summary field should round-trip arbitrary content (newlines,
	// quotes, backslashes) without truncation.
	mgr := state.NewInMemoryManager("/cwd")
	id, _ := mgr.Append(state.Entry{Kind: state.KindMessage, Payload: state.MessagePayload{Role: "user"}})
	tree, _ := mgr.Tree()
	walk, _ := tree.WalkFromLeaf(id)

	summary := "## Goal\n\"multi-line\"\nsummary with \\ backslash and \t tab\n"
	res, err := Archive(mgr, walk, 0, summary)
	if err != nil {
		t.Fatalf("Archive: %v", err)
	}
	tr, _ := mgr.Tree()
	cp, _ := tr.Get(res.CompactionEntryID)
	cpp, _ := cp.Payload.(state.CompactionPayload)
	if cpp.Summary != summary {
		t.Errorf("summary round-trip mismatch:\nwant=%q\ngot=%q", summary, cpp.Summary)
	}
}

func TestArchive_SummaryRoundTripThroughJSON(t *testing.T) {
	// In-memory manager doesn't exercise JSON; verify via re-marshal.
	summary := strings.Repeat("a", 1000) // long summary
	entry := state.Entry{
		ID:        "x",
		Kind:      state.KindCompaction,
		Timestamp: nowAt(0),
		Payload:   state.CompactionPayload{Summary: summary, FirstKeptEntryID: "abc"},
	}
	data, err := json.Marshal(entry)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got state.Entry
	if err := got.UnmarshalJSON(data); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	cpp, ok := got.Payload.(state.CompactionPayload)
	if !ok {
		t.Fatalf("payload = %T, want CompactionPayload", got.Payload)
	}
	if cpp.Summary != summary {
		t.Errorf("summary len = %d, want %d", len(cpp.Summary), len(summary))
	}
}

func TestSlidingResult_Fields(t *testing.T) {
	// Smoke: SlidingResult fields are visible and as expected.
	r := SlidingResult{CompactionEntryID: "c1", FirstKeptEntryID: "f1", ArchivedCount: 7}
	if r.CompactionEntryID != "c1" || r.FirstKeptEntryID != "f1" || r.ArchivedCount != 7 {
		t.Errorf("SlidingResult field mismatch: %+v", r)
	}
}
