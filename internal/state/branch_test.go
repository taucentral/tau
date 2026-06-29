package state

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/coevin/tau/internal/llm"
)

// newBranchOnBolt sets up a parent boltManager (with a couple of messages)
// and returns it plus a branchManager created from its current leaf. Used
// by the shadow-semantics tests below.
func newBranchOnBolt(t *testing.T) (parent Manager, branch Manager) {
	t.Helper()
	parent, _ = newBoltManager(t)
	appendMessage(t, parent, llm.RoleUser, "p1")
	appendMessage(t, parent, llm.RoleAssistant, "p2") // triggers flush
	branch = NewBranchManager(parent, parent.(*boltManager).cwd)
	return parent, branch
}

// TestBranch_AppendGoesToShadow_ParentUntouched verifies the core
// shadow-write invariant: Append on the branch advances the branch leaf
// but does NOT advance the parent's leaf, and the parent's Tree() does
// not see the shadow entries.
func TestBranch_AppendGoesToShadow_ParentUntouched(t *testing.T) {
	parent, branch := newBranchOnBolt(t)

	parentLeafBefore := parent.LeafID()
	parentTreeBefore, err := parent.Tree()
	if err != nil {
		t.Fatalf("parent.Tree before: %v", err)
	}
	parentLenBefore := parentTreeBefore.Len()

	// Append to the branch.
	id1 := appendMessage(t, branch, llm.RoleUser, "branch-u1")

	// Branch leaf advances to the new entry.
	if got := branch.LeafID(); got != id1 {
		t.Errorf("branch.LeafID = %q, want %q", got, id1)
	}

	// Parent leaf is unchanged.
	if got := parent.LeafID(); got != parentLeafBefore {
		t.Errorf("parent.LeafID changed: %q -> %q (shadow writes must not mutate parent)",
			parentLeafBefore, got)
	}

	// Parent's Tree does NOT see the new entry.
	parentTreeAfter, err := parent.Tree()
	if err != nil {
		t.Fatalf("parent.Tree after: %v", err)
	}
	if parentTreeAfter.Len() != parentLenBefore {
		t.Errorf("parent tree Len = %d, want %d (shadow leaked into parent)",
			parentTreeAfter.Len(), parentLenBefore)
	}
	if _, ok := parentTreeAfter.Get(id1); ok {
		t.Error("shadow entry id1 visible in parent's Tree; expected isolation")
	}
}

// TestBranch_TreeReflectsParentPlusShadow verifies that the branch's own
// Tree() is the union of the parent's entries and the shadow, forming a
// valid single-root tree.
func TestBranch_TreeReflectsParentPlusShadow(t *testing.T) {
	parent, branch := newBranchOnBolt(t)

	parentTree, err := parent.Tree()
	if err != nil {
		t.Fatalf("parent.Tree: %v", err)
	}
	parentLen := parentTree.Len()

	id1 := appendMessage(t, branch, llm.RoleUser, "branch-u1")
	id2 := appendMessage(t, branch, llm.RoleAssistant, "branch-a2")

	branchTree, err := branch.Tree()
	if err != nil {
		t.Fatalf("branch.Tree: %v", err)
	}

	// Branch tree = parent + 2 shadow entries.
	if want := parentLen + 2; branchTree.Len() != want {
		t.Errorf("branch tree Len = %d, want %d", branchTree.Len(), want)
	}
	if _, ok := branchTree.Get(id1); !ok {
		t.Error("id1 missing from branch tree")
	}
	if _, ok := branchTree.Get(id2); !ok {
		t.Error("id2 missing from branch tree")
	}

	// id2's parent must be id1 (chained in shadow).
	e2, ok := branchTree.Get(id2)
	if !ok {
		t.Fatalf("missing id2 in branch tree")
	}
	if e2.ParentID != id1 {
		t.Errorf("id2.ParentID = %q, want %q", e2.ParentID, id1)
	}

	// The branch's root must match the parent's root (shared store).
	if branchTree.RootID() != parentTree.RootID() {
		t.Errorf("branch root = %q, parent root = %q; want equal",
			branchTree.RootID(), parentTree.RootID())
	}
}

// TestBranch_BuildContextWalksShadow verifies that BuildContext on the
// branch surfaces messages from both the parent path and the shadow.
func TestBranch_BuildContextWalksShadow(t *testing.T) {
	parent, branch := newBranchOnBolt(t)
	_ = parent

	appendMessage(t, branch, llm.RoleUser, "branch-u")
	appendMessage(t, branch, llm.RoleAssistant, "branch-a")

	ctx, err := branch.BuildContext(context.Background())
	if err != nil {
		t.Fatalf("BuildContext: %v", err)
	}

	// Parent contributed 2 messages (u + a); shadow added 2 more.
	if want := 4; len(ctx.Messages) != want {
		t.Fatalf("ctx.Messages = %d, want %d", len(ctx.Messages), want)
	}

	// Order: parent-user, parent-assistant, branch-user, branch-assistant.
	wantRoles := []llm.Role{
		llm.RoleUser, llm.RoleAssistant,
		llm.RoleUser, llm.RoleAssistant,
	}
	for i, want := range wantRoles {
		if ctx.Messages[i].Role != want {
			t.Errorf("Messages[%d].Role = %q, want %q", i, ctx.Messages[i].Role, want)
		}
	}
}

// TestBranch_BranchShadowHelperReturnsCopy verifies that the exported
// BranchShadow helper returns the shadow entries (defensively copied) for a
// branch manager, and (nil, false) for non-branch managers.
func TestBranch_BranchShadowHelperReturnsCopy(t *testing.T) {
	_, branch := newBranchOnBolt(t)
	appendMessage(t, branch, llm.RoleUser, "shadow1")
	appendMessage(t, branch, llm.RoleUser, "shadow2")

	got, ok := BranchShadow(branch)
	if !ok {
		t.Fatal("BranchShadow(branch) returned ok=false; want true")
	}
	if len(got) != 2 {
		t.Fatalf("BranchShadow len = %d, want 2", len(got))
	}

	// Mutating the returned slice must not corrupt the branch's internal
	// state (defensive copy contract).
	got[0] = Entry{ID: "MUTATED"}
	gotAgain, _ := BranchShadow(branch)
	if gotAgain[0].ID == "MUTATED" {
		t.Error("BranchShadow did not return a defensive copy")
	}

	// Non-branch managers return (nil, false).
	parent, _ := newBoltManager(t)
	if _, ok := BranchShadow(parent); ok {
		t.Error("BranchShadow on boltManager returned ok=true; want false")
	}
}

// TestBranch_AppendAtDoesNotAdvanceLeaf verifies that AppendAt on a branch
// writes to the shadow but leaves the leaf pointer alone. This is the
// leaf-preserving primitive MergeState relies on for integration.
func TestBranch_AppendAtDoesNotAdvanceLeaf(t *testing.T) {
	_, branch := newBranchOnBolt(t)
	leafBefore := branch.LeafID()

	// AppendAt off the current leaf.
	id, err := branch.AppendAt(Entry{
		Kind: KindMessage,
		Payload: MessagePayload{
			Role:    llm.RoleAssistant,
			Content: []llm.ContentBlock{llm.TextContent{Text: "attached"}},
		},
	}, leafBefore)
	if err != nil {
		t.Fatalf("AppendAt: %v", err)
	}

	// Leaf unchanged.
	if got := branch.LeafID(); got != leafBefore {
		t.Errorf("leaf = %q, want %q (AppendAt must not advance leaf)", got, leafBefore)
	}

	// New entry is visible in the branch's tree.
	tree, err := branch.Tree()
	if err != nil {
		t.Fatalf("Tree: %v", err)
	}
	if _, ok := tree.Get(id); !ok {
		t.Error("AppendAt entry missing from branch tree")
	}
}

// TestBranch_SetLeafRejectsUnknownID verifies that SetLeaf on a branch
// rejects IDs not in the parent cache or the shadow.
func TestBranch_SetLeafRejectsUnknownID(t *testing.T) {
	_, branch := newBranchOnBolt(t)
	err := branch.SetLeaf("nonexistent-id")
	if !errors.Is(err, ErrInvalidBranch) {
		t.Errorf("err = %v, want ErrInvalidBranch", err)
	}
}

// TestBranch_SetLeafAcceptsShadowID verifies that branching to an earlier
// shadow entry works (the branch can rewind within its own writes).
func TestBranch_SetLeafAcceptsShadowID(t *testing.T) {
	_, branch := newBranchOnBolt(t)
	id1 := appendMessage(t, branch, llm.RoleUser, "first")
	appendMessage(t, branch, llm.RoleUser, "second")

	// Rewind to id1.
	if err := branch.SetLeaf(id1); err != nil {
		t.Fatalf("SetLeaf(id1): %v", err)
	}
	if got := branch.LeafID(); got != id1 {
		t.Errorf("leaf = %q, want %q", got, id1)
	}
}

// TestBranch_BranchWithSummaryUnsupported verifies that the branch
// manager itself never summarizes — the agent layer composes Branch +
// BranchSummary entries.
func TestBranch_BranchWithSummaryUnsupported(t *testing.T) {
	_, branch := newBranchOnBolt(t)
	_, err := branch.BranchWithSummary(context.Background(), "x", nil)
	if !errors.Is(err, ErrBranchWithSummaryUnsupported) {
		t.Errorf("err = %v, want ErrBranchWithSummaryUnsupported", err)
	}
}

// TestBranch_CloseDiscardsShadow_NoOrphans verifies the "no orphans on
// MergePolicyNone" requirement: after Close, the branch's shadow is
// discarded and the parent's tree contains zero entries from the child's
// shadow. Reopening or re-reading the parent shows nothing from the branch.
func TestBranch_CloseDiscardsShadow_NoOrphans(t *testing.T) {
	parent, branch := newBranchOnBolt(t)

	parentTreeBefore, err := parent.Tree()
	if err != nil {
		t.Fatalf("parent.Tree before: %v", err)
	}
	parentLenBefore := parentTreeBefore.Len()

	appendMessage(t, branch, llm.RoleUser, "shadow1")
	appendMessage(t, branch, llm.RoleUser, "shadow2")

	// Confirm shadow has entries.
	shadow, ok := BranchShadow(branch)
	if !ok || len(shadow) != 2 {
		t.Fatalf("pre-close BranchShadow len = %d, want 2", len(shadow))
	}
	shadowIDs := map[string]struct{}{}
	for _, e := range shadow {
		shadowIDs[e.ID] = struct{}{}
	}

	// Close the branch (simulates MergePolicyNone: the agent discards the
	// shadow without integrating it into the parent).
	if err := branch.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Parent's tree must be unchanged.
	parentTreeAfter, err := parent.Tree()
	if err != nil {
		t.Fatalf("parent.Tree after: %v", err)
	}
	if parentTreeAfter.Len() != parentLenBefore {
		t.Errorf("parent tree Len = %d, want %d (Close leaked shadow entries into parent)",
			parentTreeAfter.Len(), parentLenBefore)
	}
	// None of the shadow IDs appear in the parent's tree.
	for id := range shadowIDs {
		if _, ok := parentTreeAfter.Get(id); ok {
			t.Errorf("shadow id %q appeared in parent tree after Close; expected no orphans", id)
		}
	}

	// Post-close branch operations return ErrManagerClosed.
	if _, err := branch.Append(Entry{Kind: KindMessage, Payload: MessagePayload{}}); !errors.Is(err, ErrManagerClosed) {
		t.Errorf("post-close Append err = %v, want ErrManagerClosed", err)
	}

	// Shadow is gone.
	gotShadow, _ := BranchShadow(branch)
	if len(gotShadow) != 0 {
		t.Errorf("post-close BranchShadow len = %d, want 0", len(gotShadow))
	}
}

// TestBranch_IsBranchAndIsClosedHelpers verifies the exported type-check
// helpers used by agent.MergeState to pick the integration path.
func TestBranch_IsBranchAndIsClosedHelpers(t *testing.T) {
	parent, branch := newBranchOnBolt(t)

	if !IsBranch(branch) {
		t.Error("IsBranch(branch) = false; want true")
	}
	if IsBranch(parent) {
		t.Error("IsBranch(parent bolt) = true; want false")
	}

	if IsClosed(branch) {
		t.Error("IsClosed(branch) = true before Close; want false")
	}
	if err := branch.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if !IsClosed(branch) {
		t.Error("IsClosed(branch) = false after Close; want true")
	}
}

// TestBranch_IntegrationViaAppendAt verifies the MergePolicyAppend
// integration contract: the parent uses AppendAt to attach each shadow
// entry in order, advancing its leaf at the end. No orphan entries are
// created; the parent's tree grows by exactly len(shadow).
func TestBranch_IntegrationViaAppendAt(t *testing.T) {
	parent, branch := newBranchOnBolt(t)

	parentTreeBefore, err := parent.Tree()
	if err != nil {
		t.Fatalf("parent.Tree before: %v", err)
	}
	parentLenBefore := parentTreeBefore.Len()

	// Branch writes.
	uID := appendMessage(t, branch, llm.RoleUser, "branch-user")
	aID := appendMessage(t, branch, llm.RoleAssistant, "branch-assistant")

	shadow, ok := BranchShadow(branch)
	if !ok {
		t.Fatal("BranchShadow returned ok=false on a branch")
	}
	if len(shadow) != 2 {
		t.Fatalf("shadow len = %d, want 2", len(shadow))
	}

	// Integration: parent.AppendAt each shadow entry in order.
	parentLeaf := parent.LeafID()
	for _, e := range shadow {
		newID, err := parent.AppendAt(e, parentLeaf)
		if err != nil {
			t.Fatalf("AppendAt: %v", err)
		}
		parentLeaf = newID
	}
	if err := parent.SetLeaf(parentLeaf); err != nil {
		t.Fatalf("SetLeaf: %v", err)
	}

	// The parent's tree now includes both shadow entries (by payload), and
	// the new leaf matches the final AppendAt result.
	parentTreeAfter, err := parent.Tree()
	if err != nil {
		t.Fatalf("parent.Tree after: %v", err)
	}
	if want := parentLenBefore + 2; parentTreeAfter.Len() != want {
		t.Errorf("parent tree Len = %d, want %d", parentTreeAfter.Len(), want)
	}
	if got := parent.LeafID(); got != parentLeaf {
		t.Errorf("parent leaf = %q, want %q", got, parentLeaf)
	}

	// The branch's shadow IDs (uID, aID) are NOT in the parent — AppendAt
	// allocates fresh IDs. This is by design: integration copies payloads,
	// not IDs. The plan's acceptance criterion is "parent tree contains
	// zero entries from the child's shadow" when discarding (None policy);
	// for Append policy, the payloads are integrated but the IDs differ.
	if _, ok := parentTreeAfter.Get(uID); ok {
		t.Error("branch uID appeared verbatim in parent tree; AppendAt should allocate fresh IDs")
	}
	if _, ok := parentTreeAfter.Get(aID); ok {
		t.Error("branch aID appeared verbatim in parent tree; AppendAt should allocate fresh IDs")
	}
}

// TestBranch_ParentConcurrencySafeUnderConcurrentAppends verifies that
// the branch's shadow writes and concurrent reads of the parent's Tree
// do not race. The parent Manager is documented as concurrency-safe;
// this test pins that property under the race detector.
func TestBranch_ParentConcurrencySafeUnderConcurrentAppends(t *testing.T) {
	parent, branch := newBranchOnBolt(t)

	const writers = 4
	const writesEach = 25
	var wg sync.WaitGroup
	for i := 0; i < writers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < writesEach; j++ {
				appendMessage(t, branch, llm.RoleUser, "concurrent")
			}
		}()
	}
	// Concurrent reader on the parent.
	readerDone := make(chan struct{})
	go func() {
		defer close(readerDone)
		for i := 0; i < 50; i++ {
			if _, err := parent.Tree(); err != nil {
				t.Errorf("parent.Tree: %v", err)
				return
			}
		}
	}()

	wg.Wait()
	<-readerDone

	// All writes landed in the shadow.
	shadow, _ := BranchShadow(branch)
	if want := writers * writesEach; len(shadow) != want {
		t.Errorf("shadow len = %d, want %d", len(shadow), want)
	}
}

// TestBranch_OnLazyParent verifies that a branchManager can sit on top of
// a lazy parent that has NOT yet flushed to disk. The parent's
// synthetic-root materialization makes Tree()/LeafID() work pre-flush, so
// the branch captures the synthetic root ID as branchAt and shadows
// subsequent writes correctly.
func TestBranch_OnLazyParent(t *testing.T) {
	withConfigDir(t)
	cwd := t.TempDir()
	parent, err := Create(cwd, SessionHeaderPayload{Cwd: cwd})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	defer parent.Close()

	// Append a user message on the parent (no flush yet).
	appendMessage(t, parent, llm.RoleUser, "pre-flush-parent")

	branch := NewBranchManager(parent, cwd)
	branchAt := parent.LeafID()
	if got := branch.LeafID(); got != branchAt {
		t.Errorf("branch.LeafID = %q, want parent leaf %q", got, branchAt)
	}

	// Append on the branch must not advance the parent's leaf.
	appendMessage(t, branch, llm.RoleUser, "shadow-u")
	if got := parent.LeafID(); got != branchAt {
		t.Errorf("parent leaf changed under branch: %q (want %q)", got, branchAt)
	}

	// Branch tree includes both parent and shadow entries.
	tree, err := branch.Tree()
	if err != nil {
		t.Fatalf("branch.Tree: %v", err)
	}
	// lazy parent materializes a synthetic root at Create, so the parent
	// has root + 1 user entry = 2 entries; the branch adds 1 shadow entry.
	if tree.Len() != 3 {
		t.Errorf("branch tree Len = %d, want 3", tree.Len())
	}
}
