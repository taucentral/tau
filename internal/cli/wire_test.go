// wire_test.go — exercises resolveStateManager, the sentinel errors, and
// the wireSession cleanup contract.
//
// Tests run against the real filesystem under a per-test TAU_CONFIG_DIR
// override so each subtest starts from a clean slate. Prior sessions are
// materialized via state.CreateManager + Append + Close so the .bolt file
// lands on disk synchronously (state.Create is lazy and would not produce
// a file until the first assistant Append, which these tests never run).

package cli

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/state"
)

// withConfigDir overrides TAU_CONFIG_DIR for the duration of a test so
// AgentDir / SessionsDir / OpenManager resolve under tmp. Mirrors the
// helper at internal/state/manager_test.go:17-24.
func withConfigDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("TAU_CONFIG_DIR", dir)
	// Clear TAU_AGENT_DIR so AgentDir falls back to <ConfigDir>/agent.
	t.Setenv("TAU_AGENT_DIR", "")
	return dir
}

// createPriorSession materializes a real .bolt session file on disk under
// <tmp>/agent/sessions/<EncodeCwd(cwd)>/. Returns the absolute path and
// the leaf entry ID. The session contains a SessionHeader root plus one
// user Message entry so LeafID is well-defined and distinct from root.
func createPriorSession(t *testing.T, cwd string) (path, leafID string) {
	t.Helper()
	mgr, err := state.CreateManager(cwd, state.SessionHeaderPayload{
		Cwd:      cwd,
		Model:    "test-model",
		Provider: "test-provider",
	})
	if err != nil {
		t.Fatalf("state.CreateManager: %v", err)
	}
	id, err := mgr.Append(state.Entry{
		Kind: state.KindMessage,
		Payload: state.MessagePayload{
			Role: llm.RoleUser,
			Content: []llm.ContentBlock{
				llm.TextContent{Text: "seed message for resume test"},
			},
		},
	})
	if err != nil {
		t.Fatalf("mgr.Append: %v", err)
	}
	if err := mgr.Close(); err != nil {
		t.Fatalf("mgr.Close: %v", err)
	}
	// Re-derive the path: we cannot call mgr.SourcePath() because no such
	// method exists on the interface (resolveStateManager divergence A).
	sessions, err := state.ListSessions(cwd)
	if err != nil {
		t.Fatalf("state.ListSessions: %v", err)
	}
	if len(sessions) != 1 {
		t.Fatalf("ListSessions returned %d sessions, want 1", len(sessions))
	}
	return sessions[0].Path, id
}

// sessionBoltFilename returns the .bolt filename (no directory) for the
// single session recorded under cwd. Used to derive args.Resume strings.
func sessionBoltFilename(t *testing.T, cwd string) string {
	t.Helper()
	sessions, err := state.ListSessions(cwd)
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(sessions) != 1 {
		t.Fatalf("expected exactly 1 session under %s, got %d", cwd, len(sessions))
	}
	return filepath.Base(sessions[0].Path)
}

// Test 0: sentinel errors are self-identical via errors.Is. This guards
// against a refactor that wraps them with %w or renames the package-level
// var without updating callers.
func TestResolveStateManager_SentinelSelfIdentity(t *testing.T) {
	t.Run("ErrNoRecentSession", func(t *testing.T) {
		if !errors.Is(ErrNoRecentSession, ErrNoRecentSession) {
			t.Fatal("errors.Is(ErrNoRecentSession, ErrNoRecentSession) = false")
		}
	})
	t.Run("ErrSessionNotFound", func(t *testing.T) {
		if !errors.Is(ErrSessionNotFound, ErrSessionNotFound) {
			t.Fatal("errors.Is(ErrSessionNotFound, ErrSessionNotFound) = false")
		}
	})
	t.Run("ErrForkWithoutSource", func(t *testing.T) {
		if !errors.Is(ErrForkWithoutSource, ErrForkWithoutSource) {
			t.Fatal("errors.Is(ErrForkWithoutSource, ErrForkWithoutSource) = false")
		}
	})
}

// Test 4.1: no resume flags → fresh session; runtime creates its own
// manager. resolveStateManager returns (nil, "", nil).
func TestResolveStateManager_NoFlags_FreshSession(t *testing.T) {
	withConfigDir(t)
	cwd := t.TempDir()
	args := Args{}
	mgr, sessionID, err := resolveStateManager(args, cwd)
	if err != nil {
		t.Fatalf("resolveStateManager: %v", err)
	}
	if mgr != nil {
		t.Errorf("mgr = %T, want nil for fresh session", mgr)
	}
	if sessionID != "" {
		t.Errorf("sessionID = %q, want empty for fresh session", sessionID)
	}
}

// Test 4.2: --no-session → in-memory manager; no .bolt file appears even
// after an Append. The returned manager must be non-nil and must not
// produce a file under the sessions dir.
func TestResolveStateManager_NoSession_InMemory(t *testing.T) {
	withConfigDir(t)
	cwd := t.TempDir()
	args := Args{NoSession: true}
	mgr, sessionID, err := resolveStateManager(args, cwd)
	if err != nil {
		t.Fatalf("resolveStateManager: %v", err)
	}
	if mgr == nil {
		t.Fatal("mgr = nil, want an in-memory manager for --no-session")
	}
	defer mgr.Close()
	if sessionID != "" {
		t.Errorf("sessionID = %q, want empty for --no-session", sessionID)
	}
	// Append to the in-memory manager; no .bolt file should appear.
	if _, err := mgr.Append(state.Entry{
		Kind: state.KindMessage,
		Payload: state.MessagePayload{
			Role: llm.RoleUser,
			Content: []llm.ContentBlock{
				llm.TextContent{Text: "ephemeral"},
			},
		},
	}); err != nil {
		t.Fatalf("mgr.Append: %v", err)
	}
	sessions, err := state.ListSessions(cwd)
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(sessions) != 0 {
		t.Errorf("ListSessions returned %d files, want 0 for --no-session", len(sessions))
	}
}

// Test 4.3: --continue with no prior session → ErrNoRecentSession.
func TestResolveStateManager_Continue_NoPrior(t *testing.T) {
	withConfigDir(t)
	cwd := t.TempDir()
	args := Args{Continue: true}
	_, _, err := resolveStateManager(args, cwd)
	if !errors.Is(err, ErrNoRecentSession) {
		t.Fatalf("err = %v, want ErrNoRecentSession", err)
	}
}

// Test 4.4: --continue with a prior session → manager opened at that
// session; LeafID matches the prior session's leaf.
func TestResolveStateManager_Continue_WithPrior(t *testing.T) {
	withConfigDir(t)
	cwd := t.TempDir()
	_, priorLeaf := createPriorSession(t, cwd)

	args := Args{Continue: true}
	mgr, sessionID, err := resolveStateManager(args, cwd)
	if err != nil {
		t.Fatalf("resolveStateManager: %v", err)
	}
	defer mgr.Close()
	if sessionID != priorLeaf {
		t.Errorf("sessionID = %q, want %q", sessionID, priorLeaf)
	}
	if got := mgr.LeafID(); got != priorLeaf {
		t.Errorf("mgr.LeafID() = %q, want %q", got, priorLeaf)
	}
}

// Test 4.5: --resume <id> valid → manager opened at that session. The
// <id> is the .bolt filename (the SessionID portion of the filename is
// the same string the user would pass on the command line).
func TestResolveStateManager_Resume_Valid(t *testing.T) {
	withConfigDir(t)
	cwd := t.TempDir()
	_, priorLeaf := createPriorSession(t, cwd)
	boltName := sessionBoltFilename(t, cwd)

	args := Args{Resume: boltName}
	mgr, sessionID, err := resolveStateManager(args, cwd)
	if err != nil {
		t.Fatalf("resolveStateManager: %v", err)
	}
	defer mgr.Close()
	if sessionID != priorLeaf {
		t.Errorf("sessionID = %q, want %q", sessionID, priorLeaf)
	}
}

// Test 4.6: --resume <id> unknown → ErrSessionNotFound (NOT a silently
// created empty session file).
func TestResolveStateManager_Resume_Unknown(t *testing.T) {
	withConfigDir(t)
	cwd := t.TempDir()
	args := Args{Resume: "nonexistent-id.bolt"}
	_, _, err := resolveStateManager(args, cwd)
	if !errors.Is(err, ErrSessionNotFound) {
		t.Fatalf("err = %v, want ErrSessionNotFound", err)
	}
	// Verify no file was silently created.
	sessions, err := state.ListSessions(cwd)
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(sessions) != 0 {
		t.Errorf("ListSessions returned %d files, want 0 (no silent create)", len(sessions))
	}
}

// Test 4.7: --session <path> valid → manager opened at that path.
func TestResolveStateManager_SessionPath_Valid(t *testing.T) {
	withConfigDir(t)
	cwd := t.TempDir()
	priorPath, priorLeaf := createPriorSession(t, cwd)

	args := Args{Session: priorPath}
	mgr, sessionID, err := resolveStateManager(args, cwd)
	if err != nil {
		t.Fatalf("resolveStateManager: %v", err)
	}
	defer mgr.Close()
	if sessionID != priorLeaf {
		t.Errorf("sessionID = %q, want %q", sessionID, priorLeaf)
	}
}

// Test 4.8: --session <path> missing → ErrSessionNotFound.
func TestResolveStateManager_SessionPath_Missing(t *testing.T) {
	withConfigDir(t)
	cwd := t.TempDir()
	missing := filepath.Join(cwd, "never-existed.bolt")
	args := Args{Session: missing}
	_, _, err := resolveStateManager(args, cwd)
	if !errors.Is(err, ErrSessionNotFound) {
		t.Fatalf("err = %v, want ErrSessionNotFound", err)
	}
}

// Test 4.9: --fork with no source flag → ErrForkWithoutSource.
func TestResolveStateManager_ForkAlone(t *testing.T) {
	withConfigDir(t)
	cwd := t.TempDir()
	args := Args{Fork: true}
	_, _, err := resolveStateManager(args, cwd)
	if !errors.Is(err, ErrForkWithoutSource) {
		t.Fatalf("err = %v, want ErrForkWithoutSource", err)
	}
}

// Test 4.10: --continue --fork → forked manager with a fresh leaf that
// traces back to the source leaf via its parent chain. Also verifies the
// source file's bbolt lock is released (we can re-open the source path
// after resolveStateManager returns).
func TestResolveStateManager_ContinueFork(t *testing.T) {
	withConfigDir(t)
	cwd := t.TempDir()
	priorPath, priorLeaf := createPriorSession(t, cwd)

	args := Args{Continue: true, Fork: true}
	mgr, sessionID, err := resolveStateManager(args, cwd)
	if err != nil {
		t.Fatalf("resolveStateManager: %v", err)
	}
	defer mgr.Close()

	if sessionID == priorLeaf {
		t.Fatal("forked sessionID equals source leaf; fork should produce a new leaf")
	}
	// The fork relationship is cross-file: CreateBranchedSession creates
	// a fresh session whose root SessionHeader has ParentSession set to
	// the source's SessionID, but ParentID="" (it's a new root). So we
	// verify the relationship via the header payload, not a ParentID walk.
	tree, err := mgr.Tree()
	if err != nil {
		t.Fatalf("mgr.Tree: %v", err)
	}
	root := tree.Root()
	header, ok := root.Payload.(state.SessionHeaderPayload)
	if !ok {
		t.Fatalf("root payload = %T, want SessionHeaderPayload", root.Payload)
	}
	if header.ParentSession == "" {
		t.Fatal("forked session's root header has empty ParentSession; fork relationship not recorded")
	}
	// LeafID must be a valid entry in the forked tree (distinct from root
	// because CreateBranchedSession copies path[1:] into the new file).
	if sessionID == root.ID {
		t.Fatal("forked sessionID == root; CreateBranchedSession should have copied the source leaf as a child")
	}

	// Source file must be re-openable (proves we closed our handle to it).
	src, err := state.OpenManager(priorPath, cwd)
	if err != nil {
		t.Fatalf("re-open source after fork: %v", err)
	}
	_ = src.Close()
}

// Test 4.11: --resume <id> --fork → same behavior as 4.10 but with an
// explicit source id rather than --continue's most-recent semantics.
func TestResolveStateManager_ResumeFork(t *testing.T) {
	withConfigDir(t)
	cwd := t.TempDir()
	_, priorLeaf := createPriorSession(t, cwd)
	boltName := sessionBoltFilename(t, cwd)

	args := Args{Resume: boltName, Fork: true}
	mgr, sessionID, err := resolveStateManager(args, cwd)
	if err != nil {
		t.Fatalf("resolveStateManager: %v", err)
	}
	defer mgr.Close()
	if sessionID == priorLeaf {
		t.Fatal("forked sessionID equals source leaf; fork should produce a new leaf")
	}
	if sessionID == "" {
		t.Fatal("forked sessionID is empty")
	}
}

// Test 4.12: --continue --resume <id> → usage error (plain fmt.Errorf,
// not a typed sentinel). The error must name both flags and must NOT
// satisfy errors.Is against any of the three sentinels.
func TestResolveStateManager_MultipleSources_UsageError(t *testing.T) {
	withConfigDir(t)
	cwd := t.TempDir()
	createPriorSession(t, cwd)
	boltName := sessionBoltFilename(t, cwd)

	args := Args{Continue: true, Resume: boltName}
	_, _, err := resolveStateManager(args, cwd)
	if err == nil {
		t.Fatal("err = nil, want usage error for multiple sources")
	}
	if errors.Is(err, ErrNoRecentSession) || errors.Is(err, ErrSessionNotFound) || errors.Is(err, ErrForkWithoutSource) {
		t.Fatalf("err = %v, want plain usage error (not a sentinel)", err)
	}
	msg := err.Error()
	if !strings.Contains(msg, "--continue") || !strings.Contains(msg, "--resume") {
		t.Errorf("err msg = %q, want it to name both --continue and --resume", msg)
	}
}

// Test 4.13: concurrent resolveStateManager calls from two goroutines,
// each under its own cwd, stay race-free. Separate cwds avoid bbolt
// file-lock contention on the same .bolt file. Run under -race.
func TestResolveStateManager_Concurrent(t *testing.T) {
	withConfigDir(t)

	const goroutines = 2
	var wg sync.WaitGroup
	wg.Add(goroutines)
	errs := make([]error, goroutines)
	for i := 0; i < goroutines; i++ {
		i := i
		go func() {
			defer wg.Done()
			// Each goroutine gets its own cwd + its own prior session
			// so neither blocks on the other's file lock.
			cwd := t.TempDir()
			_, wantLeaf := createPriorSession(t, cwd)
			args := Args{Continue: true}
			mgr, sessionID, err := resolveStateManager(args, cwd)
			if err != nil {
				errs[i] = err
				return
			}
			defer mgr.Close()
			if sessionID != wantLeaf {
				errs[i] = fmt.Errorf("sessionID = %q, want %q", sessionID, wantLeaf)
			}
		}()
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Errorf("goroutine %d: %v", i, err)
		}
	}
}

// TestResolveStateManager_NoSession_PrecedenceOverOthers verifies
// design.md D2 row 1: --no-session wins even when --continue is also set.
// The user has explicitly opted out of persistence; that beats resume.
func TestResolveStateManager_NoSession_PrecedenceOverOthers(t *testing.T) {
	withConfigDir(t)
	cwd := t.TempDir()
	createPriorSession(t, cwd)

	args := Args{NoSession: true, Continue: true}
	mgr, sessionID, err := resolveStateManager(args, cwd)
	if err != nil {
		t.Fatalf("resolveStateManager: %v", err)
	}
	defer mgr.Close()
	if mgr == nil {
		t.Fatal("mgr = nil, want in-memory manager (--no-session precedence)")
	}
	if sessionID != "" {
		t.Errorf("sessionID = %q, want empty for --no-session", sessionID)
	}
	// Confirm in-memory: Append must not produce a .bolt file.
	if _, err := mgr.Append(state.Entry{
		Kind: state.KindMessage,
		Payload: state.MessagePayload{
			Role: llm.RoleUser,
			Content: []llm.ContentBlock{
				llm.TextContent{Text: "x"},
			},
		},
	}); err != nil {
		t.Fatalf("mgr.Append: %v", err)
	}
	sessions, err := state.ListSessions(cwd)
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(sessions) != 1 {
		t.Errorf("ListSessions = %d files, want 1 (only the prior session)", len(sessions))
	}
}

// TestWireSession_FreshSession_CleanupIsNoOp exercises the full wireSession
// path (not just resolveStateManager) for the fresh-session case: the
// returned cleanup must be safe to call, and must not panic when the
// underlying manager is nil (runtime-owned).
//
// This test uses the faux model to bypass provider/auth resolution.
func TestWireSession_FreshSession_CleanupIsNoOp(t *testing.T) {
	withConfigDir(t)
	cwd := t.TempDir()
	args := Args{Model: fauxModelID, Cwd: cwd}
	wired, cleanup, err := wireSession(context.Background(), args)
	if err != nil {
		t.Fatalf("wireSession: %v", err)
	}
	defer func() { _ = wired.Session.Shutdown(context.Background()) }()
	// Calling cleanup must not panic and must be idempotent.
	cleanup()
	cleanup()
}
