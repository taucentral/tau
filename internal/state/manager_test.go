package state

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/coevin/tau/internal/config"
	"github.com/coevin/tau/internal/llm"
)

// withConfigDir points TAU_CONFIG_DIR at a temp dir for the duration of
// the test so CreateManager / List / etc. do not touch the real home.
func withConfigDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("TAU_CONFIG_DIR", dir)
	return dir
}

// newBoltManager creates a fresh boltManager under a temp config dir.
// Returns the manager and the cwd it was created with.
func newBoltManager(t *testing.T) (Manager, string) {
	t.Helper()
	withConfigDir(t)
	cwd := t.TempDir()
	header := SessionHeaderPayload{
		Cwd:      cwd,
		Model:    "test-model",
		Provider: "test-provider",
	}
	mgr, err := CreateManager(cwd, header)
	if err != nil {
		t.Fatalf("CreateManager: %v", err)
	}
	t.Cleanup(func() { _ = mgr.Close() })
	return mgr, cwd
}

// appendMessage is a test helper that appends a Message entry with a
// single TextContent block of the given text. Fatals on error, returns
// the new entry ID.
func appendMessage(t *testing.T, mgr Manager, role llm.Role, text string) string {
	t.Helper()
	return appendMessageBlocks(t, mgr, role, llm.TextContent{Text: text})
}

// appendMessageBlocks appends a Message entry with arbitrary content
// blocks. Used for tool-use / tool-result tests that need non-text blocks.
func appendMessageBlocks(t *testing.T, mgr Manager, role llm.Role, blocks ...llm.ContentBlock) string {
	t.Helper()
	id, err := mgr.Append(Entry{
		Kind: KindMessage,
		Payload: MessagePayload{
			Role:    role,
			Content: blocks,
		},
	})
	if err != nil {
		t.Fatalf("Append: %v", err)
	}
	return id
}

func TestCreateManager_PersistsFileAndAdvancesLeaf(t *testing.T) {
	mgr, cwd := newBoltManager(t)
	// Append a message and verify LeafID advances.
	id := appendMessage(t, mgr, llm.RoleUser, "hello")
	if mgr.LeafID() != id {
		t.Errorf("LeafID after Append = %q, want %q", mgr.LeafID(), id)
	}
	// File exists on disk.
	sessions, err := mgr.List(cwd)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(sessions) != 1 {
		t.Fatalf("len(sessions) = %d, want 1", len(sessions))
	}
	if sessions[0].Path == "" {
		t.Error("SessionInfo.Path is empty")
	}
}

func TestCreateManager_HonorsConfigDirEnv(t *testing.T) {
	cfgDir := withConfigDir(t)
	cwd := t.TempDir()
	mgr, err := CreateManager(cwd, SessionHeaderPayload{Cwd: cwd})
	if err != nil {
		t.Fatalf("CreateManager: %v", err)
	}
	defer mgr.Close()
	// The session file must live under cfgDir.
	abs, err := filepath.Abs(cfgDir)
	if err != nil {
		t.Fatal(err)
	}
	var found []string
	_ = filepath.WalkDir(abs, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil //nolint:nilerr // skip unreadable entries per filepath.WalkDir convention
		}
		if strings.HasSuffix(path, ".bolt") {
			found = append(found, path)
		}
		return nil
	})
	// CreateManager does not flush until Append; the file may not yet
	// exist. Append one entry, then verify.
	if _, err := mgr.Append(Entry{Kind: KindMessage, Payload: MessagePayload{Role: llm.RoleUser}}); err != nil {
		t.Fatalf("Append: %v", err)
	}
	sessions, err := mgr.List(cwd)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(sessions) != 1 {
		t.Fatalf("expected 1 session under cfg dir, got %d", len(sessions))
	}
	if !strings.HasPrefix(sessions[0].Path, abs) {
		t.Errorf("session path %q not under cfg dir %q", sessions[0].Path, abs)
	}
}

func TestBoltManager_Append_AdvancesLeafChain(t *testing.T) {
	mgr, _ := newBoltManager(t)
	a := appendMessage(t, mgr, llm.RoleUser, "a")
	b := appendMessage(t, mgr, llm.RoleAssistant, "b")
	c := appendMessage(t, mgr, llm.RoleUser, "c")
	if mgr.LeafID() != c {
		t.Errorf("LeafID = %q, want %q", mgr.LeafID(), c)
	}
	tree, err := mgr.Tree()
	if err != nil {
		t.Fatalf("Tree: %v", err)
	}
	path, err := tree.Path(c)
	if err != nil {
		t.Fatalf("Path: %v", err)
	}
	// Path includes the SessionHeader as root plus a, b, c.
	if len(path) != 4 {
		t.Errorf("Path len = %d, want 4 (header + a + b + c)", len(path))
	}
	// Verify ParentID chain.
	if path[1].ID != a || path[2].ID != b || path[3].ID != c {
		t.Errorf("Path IDs = %s, want %s,%s,%s after root",
			[]string{path[1].ID, path[2].ID, path[3].ID}, a, b, c)
	}
}

func TestBoltManager_Append_DerivesKindFromPayload(t *testing.T) {
	mgr, _ := newBoltManager(t)
	// Caller supplies a MISMATCHED Kind; manager should override with the
	// payload-derived kind.
	id, err := mgr.Append(Entry{
		Kind:    KindLabel, // intentionally wrong
		Payload: MessagePayload{Role: llm.RoleUser},
	})
	if err != nil {
		t.Fatalf("Append: %v", err)
	}
	tree, err := mgr.Tree()
	if err != nil {
		t.Fatalf("Tree: %v", err)
	}
	e, _ := tree.Get(id)
	if e.Kind != KindMessage {
		t.Errorf("Kind = %q, want %q (derived from payload)", e.Kind, KindMessage)
	}
}

func TestBoltManager_Branch_MovesLeaf(t *testing.T) {
	mgr, _ := newBoltManager(t)
	a := appendMessage(t, mgr, llm.RoleUser, "a")
	b := appendMessage(t, mgr, llm.RoleAssistant, "b")
	c := appendMessage(t, mgr, llm.RoleUser, "c")
	// Branch back to a (abandoning b and c).
	if err := mgr.Branch(a); err != nil {
		t.Fatalf("Branch: %v", err)
	}
	if mgr.LeafID() != a {
		t.Errorf("LeafID after Branch = %q, want %q", mgr.LeafID(), a)
	}
	// New append descends from a, not c.
	d := appendMessage(t, mgr, llm.RoleAssistant, "d")
	tree, err := mgr.Tree()
	if err != nil {
		t.Fatalf("Tree: %v", err)
	}
	e, _ := tree.Get(d)
	if e.ParentID != a {
		t.Errorf("new entry ParentID = %q, want %q", e.ParentID, a)
	}
	// b and c are still in the tree (path-abandonment is non-destructive).
	if _, ok := tree.Get(b); !ok {
		t.Error("b missing after Branch")
	}
	if _, ok := tree.Get(c); !ok {
		t.Error("c missing after Branch")
	}
}

func TestBoltManager_Branch_UnknownIDReturnsErrInvalidBranch(t *testing.T) {
	mgr, _ := newBoltManager(t)
	err := mgr.Branch("nonexistent")
	if !errors.Is(err, ErrInvalidBranch) {
		t.Errorf("err = %v, want ErrInvalidBranch", err)
	}
}

func TestBoltManager_SetLeaf_UnknownIDReturnsErrInvalidBranch(t *testing.T) {
	mgr, _ := newBoltManager(t)
	err := mgr.SetLeaf("nonexistent")
	if !errors.Is(err, ErrInvalidBranch) {
		t.Errorf("err = %v, want ErrInvalidBranch", err)
	}
}

func TestBoltManager_BranchWithSummary_UnsupportedSentinel(t *testing.T) {
	mgr, _ := newBoltManager(t)
	_, err := mgr.BranchWithSummary(context.Background(), "x", nil)
	if !errors.Is(err, ErrBranchWithSummaryUnsupported) {
		t.Errorf("err = %v, want ErrBranchWithSummaryUnsupported", err)
	}
}

func TestBoltManager_Tree_RejectsEmpty(t *testing.T) {
	// A freshly-created boltManager has just the SessionHeader, so Tree
	// should succeed with one entry. Skip "empty" since CreateManager
	// always writes the header.
	mgr, _ := newBoltManager(t)
	tree, err := mgr.Tree()
	if err != nil {
		t.Fatalf("Tree: %v", err)
	}
	if tree.Len() != 1 {
		t.Errorf("Len = %d, want 1 (session header)", tree.Len())
	}
	if tree.Root().Kind != KindSessionHeader {
		t.Errorf("Root Kind = %q, want %q", tree.Root().Kind, KindSessionHeader)
	}
}

func TestBoltManager_Close_IdempotentAndBlocksSubsequentOps(t *testing.T) {
	mgr, _ := newBoltManager(t)
	if err := mgr.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := mgr.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
	_, err := mgr.Append(Entry{Kind: KindMessage, Payload: MessagePayload{}})
	if !errors.Is(err, ErrManagerClosed) {
		t.Errorf("Append after Close = %v, want ErrManagerClosed", err)
	}
}

func TestBoltManager_List_ReturnsNewestFirst(t *testing.T) {
	withConfigDir(t)
	cwd := t.TempDir()
	// Create three sessions with different mtimes.
	m1, err := CreateManager(cwd, SessionHeaderPayload{Cwd: cwd})
	if err != nil {
		t.Fatal(err)
	}
	m1.Close()
	// Bump mtime by touching the file.
	touchSessionMtime(t, cwd, 1*time.Hour)
	m2, _ := CreateManager(cwd, SessionHeaderPayload{Cwd: cwd})
	m2.Close()
	touchSessionMtime(t, cwd, 2*time.Hour)
	m3, _ := CreateManager(cwd, SessionHeaderPayload{Cwd: cwd})
	m3.Close()
	touchSessionMtime(t, cwd, 3*time.Hour)

	// Use any boltManager's List — they share the same filesystem view.
	m, _ := CreateManager(cwd, SessionHeaderPayload{Cwd: cwd})
	defer m.Close()
	sessions, err := m.List(cwd)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(sessions) != 4 {
		t.Fatalf("len = %d, want 4", len(sessions))
	}
	// Newest first: the third-bumped session must come first.
	for i := 0; i < len(sessions)-1; i++ {
		if sessions[i].LastActive.Before(sessions[i+1].LastActive) {
			t.Errorf("List not newest-first at %d: %v before %v", i, sessions[i].LastActive, sessions[i+1].LastActive)
		}
	}
}

// touchSessionMtime reaches into the sessions dir and bumps the most
// recently-created .bolt file's mtime by delta. This simulates sessions
// of varying ages for List ordering tests without sleeping.
func touchSessionMtime(t *testing.T, cwd string, delta time.Duration) {
	t.Helper()
	dir, err := config.SessionsDir(cwd)
	if err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var newest os.DirEntry
	var newestMtime time.Time
	for _, ent := range entries {
		if !strings.HasSuffix(ent.Name(), ".bolt") {
			continue
		}
		info, err := ent.Info()
		if err != nil {
			continue
		}
		if newest == nil || info.ModTime().After(newestMtime) {
			newest = ent
			newestMtime = info.ModTime()
		}
	}
	if newest == nil {
		t.Fatal("no .bolt file to touch")
	}
	path := filepath.Join(dir, newest.Name())
	now := time.Now().Add(delta)
	if err := os.Chtimes(path, now, now); err != nil {
		t.Fatalf("Chtimes: %v", err)
	}
}

func TestBoltManager_ContinueRecent_NoSessionReturnsErrNoSession(t *testing.T) {
	withConfigDir(t)
	cwd := t.TempDir()
	mgr, err := CreateManager(cwd, SessionHeaderPayload{Cwd: cwd})
	if err != nil {
		t.Fatal(err)
	}
	defer mgr.Close()
	_, err = mgr.ContinueRecent(t.TempDir()) // different cwd
	if !errors.Is(err, ErrNoSession) {
		t.Errorf("err = %v, want ErrNoSession", err)
	}
}

func TestBoltManager_ContinueRecent_OpensNewest(t *testing.T) {
	withConfigDir(t)
	cwd := t.TempDir()
	// Create two sessions; the second one is newer.
	m1, _ := CreateManager(cwd, SessionHeaderPayload{Cwd: cwd})
	m1.Close()
	m2, _ := CreateManager(cwd, SessionHeaderPayload{Cwd: cwd})
	appendMessage(t, m2, llm.RoleUser, "in second session")
	m2.Close()
	touchSessionMtime(t, cwd, 1*time.Hour) // make m2 newest

	probe, _ := CreateManager(cwd, SessionHeaderPayload{Cwd: cwd})
	defer probe.Close()
	recent, err := probe.ContinueRecent(cwd)
	if err != nil {
		t.Fatalf("ContinueRecent: %v", err)
	}
	defer recent.Close()
	tree, err := recent.Tree()
	if err != nil {
		t.Fatalf("Tree: %v", err)
	}
	// The newest session (m2) had one message appended; the older one
	// had zero. We can tell them apart by entry count.
	if tree.Len() < 2 {
		t.Errorf("Len = %d, want >= 2 (header + at least one entry from m2)", tree.Len())
	}
}

func TestBoltManager_CreateBranchedSession_CopiesPath(t *testing.T) {
	mgr, cwd := newBoltManager(t)
	appendMessage(t, mgr, llm.RoleUser, "u1")
	branchedAt := appendMessage(t, mgr, llm.RoleAssistant, "a1")
	appendMessage(t, mgr, llm.RoleUser, "u2")
	appendMessage(t, mgr, llm.RoleAssistant, "a2")

	// Branch the session at branchedAt into a new session.
	child, err := mgr.CreateBranchedSession(branchedAt)
	if err != nil {
		t.Fatalf("CreateBranchedSession: %v", err)
	}
	defer child.Close()

	tree, err := child.Tree()
	if err != nil {
		t.Fatalf("Tree: %v", err)
	}
	// Child has: header + u1 + a1 = 3 entries (u2/a2 excluded).
	if tree.Len() != 3 {
		t.Errorf("child Len = %d, want 3", tree.Len())
	}
	// Child's leaf equals the last appended entry (a1 copy).
	childLeaf := child.LeafID()
	gotLeaf, _ := tree.Get(childLeaf)
	if gotLeaf.Kind != KindMessage {
		t.Errorf("child leaf Kind = %q, want Message", gotLeaf.Kind)
	}
	mp, _ := gotLeaf.Payload.(MessagePayload)
	if len(mp.Content) != 1 {
		t.Fatalf("child leaf Content len = %d", len(mp.Content))
	}
	tc, _ := mp.Content[0].(llm.TextContent)
	if tc.Text != "a1" {
		t.Errorf("child leaf text = %q, want a1", tc.Text)
	}
	// Child session is discoverable via List.
	sessions, err := mgr.List(cwd)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(sessions) != 2 {
		t.Errorf("sessions = %d, want 2 (parent + child)", len(sessions))
	}
}

func TestBoltManager_InMemory_ReturnsInMemoryManager(t *testing.T) {
	mgr, _ := newBoltManager(t)
	inMem := mgr.InMemory(t.TempDir())
	defer inMem.Close()
	id := appendMessage(t, inMem, llm.RoleUser, "ephemeral")
	if inMem.LeafID() != id {
		t.Errorf("in-memory LeafID = %q, want %q", inMem.LeafID(), id)
	}
	// Verify it's not persisted: List on its cwd should be empty (since
	// we never created a backing file).
	// The InMemory manager doesn't have a way to query "did I persist?";
	// we instead check that opening a fresh boltManager on the same cwd
	// shows no sessions.
	sessions, err := mgr.List(inMem.(*inMemoryManager).cwd)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	for _, s := range sessions {
		if strings.Contains(s.Path, id) {
			t.Errorf("in-memory entry leaked to disk: %s", s.Path)
		}
	}
}

func TestBoltManager_ForkFrom_CopiesFromSource(t *testing.T) {
	src, cwd := newBoltManager(t)
	appendMessage(t, src, llm.RoleUser, "u")
	appendMessage(t, src, llm.RoleAssistant, "a")
	// Re-open the source path so we can pass it to ForkFrom.
	src.Close()
	sessions, err := listSessionsInDir(mustSessionsDir(t, cwd), cwd)
	if err != nil {
		t.Fatal(err)
	}
	if len(sessions) != 1 {
		t.Fatalf("sessions = %d, want 1", len(sessions))
	}

	dst, err := CreateManager(cwd, SessionHeaderPayload{Cwd: cwd})
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Close()
	forked, err := dst.ForkFrom(sessions[0].Path)
	if err != nil {
		t.Fatalf("ForkFrom: %v", err)
	}
	defer forked.Close()
	tree, err := forked.Tree()
	if err != nil {
		t.Fatalf("Tree: %v", err)
	}
	if tree.Len() != 3 { // header + u + a
		t.Errorf("forked Len = %d, want 3", tree.Len())
	}
}

func mustSessionsDir(t *testing.T, cwd string) string {
	t.Helper()
	dir, err := config.SessionsDir(cwd)
	if err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestBoltManager_ListAll_ScansEveryCwd(t *testing.T) {
	withConfigDir(t)
	cwd1 := t.TempDir()
	cwd2 := t.TempDir()
	m1, _ := CreateManager(cwd1, SessionHeaderPayload{Cwd: cwd1})
	appendMessage(t, m1, llm.RoleUser, "x")
	m1.Close()
	m2, _ := CreateManager(cwd2, SessionHeaderPayload{Cwd: cwd2})
	appendMessage(t, m2, llm.RoleUser, "y")
	m2.Close()
	// Use an in-memory manager to call ListAll so we don't add an extra
	// backing file by creating yet another session.
	all, err := NewInMemoryManager(cwd1).ListAll()
	if err != nil {
		t.Fatalf("ListAll: %v", err)
	}
	if len(all) != 2 {
		t.Errorf("ListAll = %d sessions, want 2: %+v", len(all), all)
	}
}

func TestBoltManager_BuildContext_BasicOrdering(t *testing.T) {
	mgr, _ := newBoltManager(t)
	appendMessage(t, mgr, llm.RoleUser, "first user")
	appendMessage(t, mgr, llm.RoleAssistant, "first assistant")
	appendMessage(t, mgr, llm.RoleUser, "second user")

	ctx, err := mgr.BuildContext(context.Background())
	if err != nil {
		t.Fatalf("BuildContext: %v", err)
	}
	if len(ctx.Messages) != 3 {
		t.Fatalf("Messages len = %d, want 3", len(ctx.Messages))
	}
	if ctx.Messages[0].Role != llm.RoleUser {
		t.Errorf("Messages[0].Role = %q, want user", ctx.Messages[0].Role)
	}
	if ctx.Messages[1].Role != llm.RoleAssistant {
		t.Errorf("Messages[1].Role = %q, want assistant", ctx.Messages[1].Role)
	}
	tc0, _ := ctx.Messages[0].Content[0].(llm.TextContent)
	if tc0.Text != "first user" {
		t.Errorf("Messages[0] text = %q", tc0.Text)
	}
}

func TestBoltManager_BuildContext_CompactionSummaryReplacesOlder(t *testing.T) {
	mgr, _ := newBoltManager(t)
	oldID := appendMessage(t, mgr, llm.RoleUser, "old user")
	appendMessage(t, mgr, llm.RoleAssistant, "old assistant")
	// Append a Compaction entry pointing at oldID as FirstKeptEntryID.
	_, err := mgr.Append(Entry{
		Kind: KindCompaction,
		Payload: CompactionPayload{
			Summary:          "earlier conversation was about X",
			FirstKeptEntryID: oldID,
		},
	})
	if err != nil {
		t.Fatalf("Append Compaction: %v", err)
	}
	appendMessage(t, mgr, llm.RoleUser, "new user")
	appendMessage(t, mgr, llm.RoleAssistant, "new assistant")

	ctx, err := mgr.BuildContext(context.Background())
	if err != nil {
		t.Fatalf("BuildContext: %v", err)
	}
	if len(ctx.Messages) != 5 {
		t.Fatalf("Messages len = %d, want 5 (summary + old user + old assistant + new user + new assistant)", len(ctx.Messages))
	}
	// First message is the compaction summary.
	tc0, _ := ctx.Messages[0].Content[0].(llm.TextContent)
	if !strings.Contains(tc0.Text, "earlier conversation was about X") {
		t.Errorf("Messages[0] = %q, want compaction summary", tc0.Text)
	}
	if !strings.HasPrefix(tc0.Text, "[Compacted earlier conversation]") {
		t.Errorf("Messages[0] missing synthetic prefix: %q", tc0.Text)
	}
}

func TestBoltManager_BuildContext_CompactionDropsOlderThanFirstKept(t *testing.T) {
	mgr, _ := newBoltManager(t)
	// Build a chain where the compaction's FirstKeptEntryID is in the
	// middle, so older entries should be dropped.
	drop1 := appendMessage(t, mgr, llm.RoleUser, "drop1")
	appendMessage(t, mgr, llm.RoleAssistant, "drop2")
	keep1 := appendMessage(t, mgr, llm.RoleUser, "keep1")
	// Compact: FirstKeptEntryID = keep1. drop1 and drop2 must vanish.
	_, err := mgr.Append(Entry{
		Kind: KindCompaction,
		Payload: CompactionPayload{
			Summary:          "summary of earlier material",
			FirstKeptEntryID: keep1,
		},
	})
	if err != nil {
		t.Fatalf("Append Compaction: %v", err)
	}
	appendMessage(t, mgr, llm.RoleAssistant, "keep2")

	ctx, err := mgr.BuildContext(context.Background())
	if err != nil {
		t.Fatalf("BuildContext: %v", err)
	}
	for _, m := range ctx.Messages {
		for _, b := range m.Content {
			if tc, ok := b.(llm.TextContent); ok {
				// "drop1"/"drop2" are the user-facing texts of the
				// dropped entries; they must NOT appear. "drop" as a
				// substring in test-code only.
				if strings.Contains(tc.Text, "drop1") || strings.Contains(tc.Text, "drop2") {
					t.Errorf("dropped text leaked into context: %q", tc.Text)
				}
			}
		}
	}
	// Summary + keep1 + keep2 = 3 messages.
	if len(ctx.Messages) != 3 {
		t.Errorf("Messages len = %d, want 3 (summary + keep1 + keep2)", len(ctx.Messages))
	}
	_ = drop1 // used for narrative clarity
}

func TestBoltManager_BuildContext_BranchSummaryInjected(t *testing.T) {
	mgr, _ := newBoltManager(t)
	_, err := mgr.Append(Entry{
		Kind:    KindBranchSummary,
		Payload: BranchSummaryPayload{Summary: "parent branch did X"},
	})
	if err != nil {
		t.Fatalf("Append BranchSummary: %v", err)
	}
	appendMessage(t, mgr, llm.RoleUser, "u")
	appendMessage(t, mgr, llm.RoleAssistant, "a")

	ctx, err := mgr.BuildContext(context.Background())
	if err != nil {
		t.Fatalf("BuildContext: %v", err)
	}
	// First message is the parent-branch-summary synthetic.
	if len(ctx.Messages) < 3 {
		t.Fatalf("Messages len = %d, want >= 3", len(ctx.Messages))
	}
	tc0, _ := ctx.Messages[0].Content[0].(llm.TextContent)
	if !strings.Contains(tc0.Text, "parent branch did X") {
		t.Errorf("Messages[0] = %q, want parent-branch summary", tc0.Text)
	}
	if !strings.HasPrefix(tc0.Text, "[Parent branch summary]") {
		t.Errorf("Messages[0] missing synthetic prefix: %q", tc0.Text)
	}
}

func TestBoltManager_BuildContext_ToolPairIntegrity(t *testing.T) {
	mgr, _ := newBoltManager(t)
	// Build a chain with a ToolUse in the assistant turn and the
	// matching ToolResult in the next user turn. Then insert a Compaction
	// whose FirstKeptEntryID is the user/tool-result entry — meaning the
	// ToolUse would be in the "dropped" region. The pair-integrity rule
	// must pull the ToolUse back in.
	toolUseMsg := appendMessageBlocks(t, mgr, llm.RoleAssistant, withToolUse("tu_1", "bash")...)
	toolResultMsg := appendMessageBlocks(t, mgr, llm.RoleUser, withToolResult("tu_1")...)
	// Compact pointing at the toolResult entry as the cutoff.
	_, err := mgr.Append(Entry{
		Kind: KindCompaction,
		Payload: CompactionPayload{
			Summary:          "summary placeholder",
			FirstKeptEntryID: toolResultMsg,
		},
	})
	if err != nil {
		t.Fatalf("Append Compaction: %v", err)
	}
	appendMessage(t, mgr, llm.RoleAssistant, "ok")

	ctx, err := mgr.BuildContext(context.Background())
	if err != nil {
		t.Fatalf("BuildContext: %v", err)
	}

	// The tool_use entry should have been pulled back into context
	// despite being older than FirstKeptEntryID. We don't assert the
	// exact message count (which depends on synthetic placement) —
	// instead we scan content for the tool_use block.
	var foundToolUse, foundToolResult bool
	for _, m := range ctx.Messages {
		for _, b := range m.Content {
			if tu, ok := b.(llm.ToolUse); ok && tu.ID == "tu_1" {
				foundToolUse = true
			}
			if tr, ok := b.(llm.ToolResult); ok && tr.ToolUseID == "tu_1" {
				foundToolResult = true
			}
		}
	}
	if !foundToolUse {
		t.Error("ToolUse tu_1 missing — pair integrity rule failed")
	}
	if !foundToolResult {
		t.Error("ToolResult tu_1 missing")
	}
	_ = toolUseMsg
}

// withToolUse returns a content slice containing a ToolUse block with the
// given id and name, suitable for MessagePayload.Content.
func withToolUse(id, name string) []llm.ContentBlock {
	return []llm.ContentBlock{llm.ToolUse{
		ID:    id,
		Name:  name,
		Input: json.RawMessage(`{}`),
	}}
}

// withToolResult returns a content slice containing a ToolResult block
// referencing the given tool-use id.
func withToolResult(toolUseID string) []llm.ContentBlock {
	return []llm.ContentBlock{llm.ToolResult{
		ToolUseID: toolUseID,
		Content:   []llm.ContentBlock{llm.TextContent{Text: "ok"}},
	}}
}

func TestInMemoryManager_AppendAndBranch(t *testing.T) {
	m := NewInMemoryManager(t.TempDir())
	defer m.Close()
	// NewInMemoryManager seeds a SessionHeader root automatically.
	rootID := m.LeafID()
	if rootID == "" {
		t.Fatal("LeafID empty after construction; expected seeded root")
	}
	a := appendMessage(t, m, llm.RoleUser, "a")
	b := appendMessage(t, m, llm.RoleAssistant, "b")
	if err := m.Branch(a); err != nil {
		t.Fatalf("Branch: %v", err)
	}
	c := appendMessage(t, m, llm.RoleAssistant, "c")
	tree, err := m.Tree()
	if err != nil {
		t.Fatalf("Tree: %v", err)
	}
	parent, _ := tree.Get(c)
	if parent.ParentID != a {
		t.Errorf("after Branch, c.ParentID = %q, want %q", parent.ParentID, a)
	}
	// b is still in the tree.
	if _, ok := tree.Get(b); !ok {
		t.Error("b missing after Branch")
	}
}

func TestInMemoryManager_BuildContext(t *testing.T) {
	m := NewInMemoryManager(t.TempDir())
	defer m.Close()
	appendMessage(t, m, llm.RoleUser, "u")
	appendMessage(t, m, llm.RoleAssistant, "a")
	ctx, err := m.BuildContext(context.Background())
	if err != nil {
		t.Fatalf("BuildContext: %v", err)
	}
	if len(ctx.Messages) != 2 {
		t.Errorf("Messages len = %d, want 2", len(ctx.Messages))
	}
}

func TestInMemoryManager_CloseClearsState(t *testing.T) {
	m := NewInMemoryManager(t.TempDir()).(*inMemoryManager)
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if m.leafID != "" {
		t.Errorf("leafID after Close = %q, want empty", m.leafID)
	}
	if len(m.entries) != 0 {
		t.Errorf("entries after Close = %d, want 0", len(m.entries))
	}
}
