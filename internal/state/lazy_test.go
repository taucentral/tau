package state

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/taucentral/tau/internal/config"
	"github.com/taucentral/tau/internal/llm"
)

// newLazyManager creates a lazyManager under an isolated config dir and
// registered cleanup.
func newLazyManager(t *testing.T) (Manager, string) {
	t.Helper()
	withConfigDir(t)
	cwd := t.TempDir()
	m, err := Create(cwd, SessionHeaderPayload{Cwd: cwd, Model: "test-model"})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	return m, cwd
}

// countBoltFiles counts *.bolt files under cfgRoot/sessions/*/.
func countBoltFiles(t *testing.T, cfgRoot string) int {
	t.Helper()
	abs, err := filepath.Abs(cfgRoot)
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	_ = filepath.WalkDir(abs, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil //nolint:nilerr // skip unreadable entries per filepath.WalkDir convention
		}
		if strings.HasSuffix(path, ".bolt") {
			count++
		}
		return nil
	})
	return count
}

func TestLazyManager_NoFileBeforeAssistantAppend(t *testing.T) {
	cfgDir := withConfigDir(t)
	cwd := t.TempDir()
	m, err := Create(cwd, SessionHeaderPayload{Cwd: cwd})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	defer m.Close()

	// Append non-assistant messages.
	appendMessage(t, m, llm.RoleSystem, "system")
	appendMessage(t, m, llm.RoleUser, "user1")
	appendMessage(t, m, llm.RoleUser, "user2")

	if got := countBoltFiles(t, cfgDir); got != 0 {
		t.Errorf("after non-assistant appends, bolt files = %d, want 0", got)
	}
}

func TestLazyManager_AssistantAppendTriggersFlush(t *testing.T) {
	cfgDir := withConfigDir(t)
	cwd := t.TempDir()
	m, err := Create(cwd, SessionHeaderPayload{Cwd: cwd})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	defer m.Close()

	appendMessage(t, m, llm.RoleUser, "user1")
	appendMessage(t, m, llm.RoleUser, "user2")
	if got := countBoltFiles(t, cfgDir); got != 0 {
		t.Fatalf("premature flush: bolt files = %d, want 0", got)
	}

	// First assistant message triggers flush.
	appendMessage(t, m, llm.RoleAssistant, "assistant1")
	if got := countBoltFiles(t, cfgDir); got != 1 {
		t.Errorf("after assistant Append, bolt files = %d, want 1", got)
	}
}

func TestLazyManager_CloseBeforeFlush_NoFileWritten(t *testing.T) {
	cfgDir := withConfigDir(t)
	cwd := t.TempDir()
	m, err := Create(cwd, SessionHeaderPayload{Cwd: cwd})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	appendMessage(t, m, llm.RoleUser, "abandoned")
	if err := m.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if got := countBoltFiles(t, cfgDir); got != 0 {
		t.Errorf("after close-before-flush, bolt files = %d, want 0", got)
	}
}

func TestLazyManager_PostFlushDelegates(t *testing.T) {
	m, cwd := newLazyManager(t)
	appendMessage(t, m, llm.RoleUser, "u1")
	appendMessage(t, m, llm.RoleAssistant, "a1") // triggers flush

	// Post-flush appends should be persisted (visible via List).
	postID := appendMessage(t, m, llm.RoleUser, "u2")
	if m.LeafID() != postID {
		t.Errorf("post-flush LeafID = %q, want %q", m.LeafID(), postID)
	}
	sessions, err := m.List(cwd)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(sessions) != 1 {
		t.Fatalf("sessions = %d, want 1 (flushed file)", len(sessions))
	}
	// bbolt holds an flock on the file: close m before re-opening.
	path := sessions[0].Path
	if err := m.Close(); err != nil {
		t.Fatalf("Close before reopen: %v", err)
	}
	reopened, err := OpenManager(path, cwd)
	if err != nil {
		t.Fatalf("OpenManager: %v", err)
	}
	defer reopened.Close()
	tree, err := reopened.Tree()
	if err != nil {
		t.Fatalf("Tree: %v", err)
	}
	// header + u1 + a1 + u2 = 4 entries.
	if tree.Len() != 4 {
		t.Errorf("reopened tree Len = %d, want 4", tree.Len())
	}
}

func TestLazyManager_BufferedBranchWorks(t *testing.T) {
	cfgDir := withConfigDir(t)
	cwd := t.TempDir()
	m, err := Create(cwd, SessionHeaderPayload{Cwd: cwd})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	defer m.Close()
	a := appendMessage(t, m, llm.RoleUser, "a")
	b := appendMessage(t, m, llm.RoleUser, "b")
	c := appendMessage(t, m, llm.RoleUser, "c")

	// Branch back to a in buffer mode.
	if err := m.Branch(a); err != nil {
		t.Fatalf("Branch: %v", err)
	}
	if m.LeafID() != a {
		t.Errorf("LeafID = %q, want %q", m.LeafID(), a)
	}
	d := appendMessage(t, m, llm.RoleAssistant, "d") // triggers flush
	_ = d

	// Verify post-flush tree: header + a + b + c + d.
	tree, err := m.Tree()
	if err != nil {
		t.Fatalf("Tree: %v", err)
	}
	if tree.Len() != 5 {
		t.Errorf("tree Len = %d, want 5 (header + a + b + c + d)", tree.Len())
	}
	// d's parent must be a (we branched to a before appending d).
	de, _ := tree.Get(d)
	if de.ParentID != a {
		t.Errorf("d.ParentID = %q, want %q", de.ParentID, a)
	}
	// File was created.
	if got := countBoltFiles(t, cfgDir); got != 1 {
		t.Errorf("bolt files = %d, want 1", got)
	}
	_ = b
	_ = c
}

func TestLazyManager_BufferedBuildContextWorks(t *testing.T) {
	m, _ := newLazyManager(t)
	appendMessage(t, m, llm.RoleUser, "u1")
	appendMessage(t, m, llm.RoleUser, "u2")

	// BuildContext should work in buffer mode (no flush yet).
	ctx, err := m.BuildContext(context.Background())
	if err != nil {
		t.Fatalf("BuildContext: %v", err)
	}
	if len(ctx.Messages) != 2 {
		t.Errorf("Messages len = %d, want 2", len(ctx.Messages))
	}
}

func TestLazyManager_BufferedTreeWorks(t *testing.T) {
	m, _ := newLazyManager(t)
	a := appendMessage(t, m, llm.RoleUser, "a")
	b := appendMessage(t, m, llm.RoleUser, "b")

	tree, err := m.Tree()
	if err != nil {
		t.Fatalf("Tree: %v", err)
	}
	// Both entries are present (plus their implicit chain via ParentID).
	if _, ok := tree.Get(a); !ok {
		t.Error("a not in tree")
	}
	if _, ok := tree.Get(b); !ok {
		t.Error("b not in tree")
	}
	path, err := tree.Path(b)
	if err != nil {
		t.Fatalf("Path: %v", err)
	}
	// Path is [synthetic-root, a, b] — the materialized SessionHeader
	// root plus the two appended entries.
	if len(path) != 3 {
		t.Errorf("path len = %d, want 3 (root, a, b)", len(path))
	}
}

func TestLazyManager_SetLeaf_UnknownIDReturnsErrInvalidBranch(t *testing.T) {
	m, _ := newLazyManager(t)
	err := m.SetLeaf("nonexistent")
	if !errors.Is(err, ErrInvalidBranch) {
		t.Errorf("err = %v, want ErrInvalidBranch", err)
	}
}

func TestLazyManager_Tree_HasSyntheticRootBeforeFlush(t *testing.T) {
	withConfigDir(t)
	cwd := t.TempDir()
	m, err := Create(cwd, SessionHeaderPayload{Cwd: cwd})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	defer m.Close()
	// Create materializes a synthetic SessionHeader root in the buffer
	// so Tree(), LeafID(), and AppendAt() work before flush. The tree
	// should contain exactly one entry (the root).
	tree, err := m.Tree()
	if err != nil {
		t.Fatalf("Tree: %v", err)
	}
	leaf := m.LeafID()
	if leaf == "" {
		t.Fatal("LeafID is empty; expected the synthetic root ID")
	}
	path, err := tree.Path(leaf)
	if err != nil {
		t.Fatalf("Path: %v", err)
	}
	if len(path) != 1 {
		t.Errorf("path len = %d, want 1 (synthetic root only)", len(path))
	}
	if path[0].Kind != KindSessionHeader {
		t.Errorf("root kind = %v, want KindSessionHeader", path[0].Kind)
	}
}

func TestLazyManager_Close_Idempotent(t *testing.T) {
	m, _ := newLazyManager(t)
	if err := m.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := m.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
}

func TestLazyManager_Close_PostFlush_DelegatesToBolt(t *testing.T) {
	m, _ := newLazyManager(t)
	appendMessage(t, m, llm.RoleAssistant, "trigger flush")
	if err := m.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := m.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
}

func TestLazyManager_BranchWithSummary_PreFlush_ReturnsUnsupported(t *testing.T) {
	m, _ := newLazyManager(t)
	_, err := m.BranchWithSummary(context.Background(), "x", nil)
	if !errors.Is(err, ErrBranchWithSummaryUnsupported) {
		t.Errorf("err = %v, want ErrBranchWithSummaryUnsupported", err)
	}
}

func TestLazyManager_RequiresNonEmptyCwd(t *testing.T) {
	_, err := Create("", SessionHeaderPayload{})
	if err == nil {
		t.Fatal("expected error on empty cwd, got nil")
	}
}

func TestLazyManager_FlushPersistsAllBufferedEntries(t *testing.T) {
	withConfigDir(t)
	cwd := t.TempDir()
	m, err := Create(cwd, SessionHeaderPayload{Cwd: cwd})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	// Buffer a long chain before flushing.
	appendMessage(t, m, llm.RoleUser, "u1")
	appendMessage(t, m, llm.RoleAssistant, "a1")
	appendMessage(t, m, llm.RoleUser, "u2")
	appendMessage(t, m, llm.RoleAssistant, "a2")
	appendMessage(t, m, llm.RoleUser, "u3")

	// Verify file exists with all entries.
	sessions, err := m.List(cwd)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(sessions) != 1 {
		t.Fatalf("sessions = %d, want 1", len(sessions))
	}
	path := sessions[0].Path
	// bbolt holds an flock on the file: close m before re-opening.
	if err := m.Close(); err != nil {
		t.Fatalf("Close before reopen: %v", err)
	}
	reopened, err := OpenManager(path, cwd)
	if err != nil {
		t.Fatalf("OpenManager: %v", err)
	}
	defer reopened.Close()
	tree, err := reopened.Tree()
	if err != nil {
		t.Fatalf("Tree: %v", err)
	}
	if tree.Len() != 6 { // header + 5 messages
		t.Errorf("reopened Len = %d, want 6", tree.Len())
	}
	// Sanity: re-opened BuildContext has 5 messages in order.
	ctx, err := reopened.BuildContext(context.Background())
	if err != nil {
		t.Fatalf("BuildContext: %v", err)
	}
	if len(ctx.Messages) != 5 {
		t.Errorf("reopened ctx messages = %d, want 5", len(ctx.Messages))
	}
	// Order check: u1, a1, u2, a2, u3
	wantRoles := []llm.Role{llm.RoleUser, llm.RoleAssistant, llm.RoleUser, llm.RoleAssistant, llm.RoleUser}
	for i, want := range wantRoles {
		if ctx.Messages[i].Role != want {
			t.Errorf("Messages[%d].Role = %q, want %q", i, ctx.Messages[i].Role, want)
		}
	}
}

// Verify lazyManager plays well with the global listSessionsInDir helper
// by exercising List before flush.
func TestLazyManager_List_PreFlush_ReturnsEmpty(t *testing.T) {
	m, cwd := newLazyManager(t)
	appendMessage(t, m, llm.RoleUser, "buffered")
	sessions, err := m.List(cwd)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(sessions) != 0 {
		t.Errorf("pre-flush List = %d, want 0", len(sessions))
	}
}

// Sanity: lazy sessions are discoverable by ListAll once flushed.
func TestLazyManager_ListAll_IncludesFlushedSession(t *testing.T) {
	withConfigDir(t)
	cwd := t.TempDir()
	m, err := Create(cwd, SessionHeaderPayload{Cwd: cwd})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	defer m.Close()
	appendMessage(t, m, llm.RoleUser, "u")
	appendMessage(t, m, llm.RoleAssistant, "a") // flush

	all, err := NewInMemoryManager(cwd).ListAll()
	if err != nil {
		t.Fatalf("ListAll: %v", err)
	}
	if len(all) != 1 {
		t.Errorf("ListAll = %d, want 1", len(all))
	}
	// sanity: the SessionInfo.Path resolves to a real file
	for _, si := range all {
		if _, err := os.Stat(si.Path); err != nil {
			t.Errorf("SessionInfo.Path %q: %v", si.Path, err)
		}
	}
}

// Verify lazyManager + config.SessionsDir integration: the flushed file
// lands in the encoded-cwd directory.
func TestLazyManager_FlushedFileLandsInEncodedCwdDir(t *testing.T) {
	withConfigDir(t)
	cwd := t.TempDir()
	m, err := Create(cwd, SessionHeaderPayload{Cwd: cwd})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	defer m.Close()
	appendMessage(t, m, llm.RoleAssistant, "trigger flush")

	expectedDir, err := config.SessionsDir(cwd)
	if err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(expectedDir)
	if err != nil {
		t.Fatalf("ReadDir %s: %v", expectedDir, err)
	}
	if len(entries) != 1 {
		t.Errorf("entries in encoded dir = %d, want 1", len(entries))
	}
}
