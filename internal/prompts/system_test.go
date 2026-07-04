package prompts

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/coevin/tau/internal/llm"
)

// writeFile is a small helper for tests: writes content to path,
// creating parent dirs. Fails the test on error.
func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

// mkdirGit is a helper for tests: creates a `.git` directory marker
// in dir so resolveStopDir treats it as a repo root and the walk
// stops there. Keeps tests isolated from any real `.git` ancestors.
func mkdirGit(t *testing.T, dir string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Join(dir, ".git"), 0o755); err != nil {
		t.Fatalf("mkdir .git in %s: %v", dir, err)
	}
}

// blockText extracts the text payload of a ContentBlock, failing the
// test if the block isn't llm.TextContent.
func blockText(t *testing.T, b llm.ContentBlock) string {
	t.Helper()
	tc, ok := b.(llm.TextContent)
	if !ok {
		t.Fatalf("block = %T, want llm.TextContent", b)
	}
	return tc.Text
}

// TestAssembler_BuiltInOnly verifies that an Assembler with no
// ConfigDir or ProjectDir produces just the built-in baseline block.
func TestAssembler_BuiltInOnly(t *testing.T) {
	a := &Assembler{BuiltIn: "IDENTITY"}
	blocks, err := a.Assemble(context.Background())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if len(blocks) != 1 {
		t.Fatalf("expected 1 block, got %d", len(blocks))
	}
	txt, ok := blocks[0].(llm.TextContent)
	if !ok {
		t.Fatalf("block = %T, want llm.TextContent", blocks[0])
	}
	if !strings.Contains(txt.Text, "IDENTITY") {
		t.Errorf("block missing built-in content: %q", txt.Text)
	}
}

// TestAssembler_NilSafe verifies that a nil *Assembler returns no
// blocks without panicking.
func TestAssembler_NilSafe(t *testing.T) {
	var nilA *Assembler
	got, err := nilA.Assemble(context.Background())
	if err != nil {
		t.Fatalf("nil Assembler Assemble: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("nil Assembler returned %d blocks", len(got))
	}
}

// TestAssembler_ProjectFilesAppended verifies that AGENTS.md and
// CLAUDE.md in the project dir produce blocks after the built-in.
//
// A `.git` marker in proj isolates the walk from any real ancestors
// so the test is deterministic across machines.
func TestAssembler_ProjectFilesAppended(t *testing.T) {
	proj := t.TempDir()
	mkdirGit(t, proj)
	writeFile(t, filepath.Join(proj, "AGENTS.md"), "agents body")
	writeFile(t, filepath.Join(proj, "CLAUDE.md"), "claude body")
	a := &Assembler{ProjectDir: proj, BuiltIn: "IDENTITY"}
	blocks, err := a.Assemble(context.Background())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if len(blocks) != 3 {
		t.Fatalf("expected 3 blocks (built-in + agents + claude), got %d", len(blocks))
	}
	for i, label := range []string{"IDENTITY", "agents body", "claude body"} {
		txt := blocks[i].(llm.TextContent).Text
		if !strings.Contains(txt, label) {
			t.Errorf("block %d (%q) missing %q", i, txt, label)
		}
	}
}

// TestAssembler_GlobalSystem verifies that <ConfigDir>/SYSTEM.md
// appears between the built-in and project files.
func TestAssembler_GlobalSystem(t *testing.T) {
	conf := t.TempDir()
	proj := t.TempDir()
	mkdirGit(t, proj)
	writeFile(t, filepath.Join(conf, "SYSTEM.md"), "global rules")
	writeFile(t, filepath.Join(proj, "AGENTS.md"), "project rules")
	a := &Assembler{ConfigDir: conf, ProjectDir: proj, BuiltIn: "IDENTITY"}
	blocks, err := a.Assemble(context.Background())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if len(blocks) != 3 {
		t.Fatalf("expected 3 blocks, got %d", len(blocks))
	}
	if !strings.Contains(blocks[0].(llm.TextContent).Text, "IDENTITY") {
		t.Errorf("block 0 should be identity")
	}
	if !strings.Contains(blocks[1].(llm.TextContent).Text, "global rules") {
		t.Errorf("block 1 should be global rules")
	}
	if !strings.Contains(blocks[2].(llm.TextContent).Text, "project rules") {
		t.Errorf("block 2 should be project rules")
	}
}

// TestAssembler_EmptyFilesSkipped verifies that empty files do not
// produce empty blocks (silent skip, not failure).
func TestAssembler_EmptyFilesSkipped(t *testing.T) {
	proj := t.TempDir()
	mkdirGit(t, proj)
	writeFile(t, filepath.Join(proj, "AGENTS.md"), "")
	writeFile(t, filepath.Join(proj, "CLAUDE.md"), "   \n  \t  ")
	a := &Assembler{ProjectDir: proj, BuiltIn: "IDENTITY"}
	blocks, err := a.Assemble(context.Background())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if len(blocks) != 1 {
		t.Fatalf("expected 1 block (built-in only), got %d", len(blocks))
	}
}

// TestAssembler_MissingFilesOK verifies that missing files are not
// errors; only genuine read failures (permissions, etc.) propagate.
func TestAssembler_MissingFilesOK(t *testing.T) {
	tmp := t.TempDir()
	a := &Assembler{
		ConfigDir:  filepath.Join(tmp, "missing-config"),
		ProjectDir: filepath.Join(tmp, "missing-project"),
		BuiltIn:    "IDENTITY",
	}
	blocks, err := a.Assemble(context.Background())
	if err != nil {
		t.Fatalf("Assemble with missing dirs: %v", err)
	}
	if len(blocks) != 1 {
		t.Errorf("expected 1 block, got %d", len(blocks))
	}
}

// TestWalk_SingleDirNoWalk covers task 4.1: when ProjectDir contains
// a `.git` marker, the walk resolves stopDir = ProjectDir and reads
// only ProjectDir/CLAUDE.md. Output matches today's behavior modulo
// the new "(cwd)" label.
func TestWalk_SingleDirNoWalk(t *testing.T) {
	proj := t.TempDir()
	mkdirGit(t, proj)
	writeFile(t, filepath.Join(proj, "CLAUDE.md"), "only-cwd-body")
	a := &Assembler{ProjectDir: proj, BuiltIn: "ID"}
	blocks, err := a.Assemble(context.Background())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	// Expect built-in + one CLAUDE.md block.
	if len(blocks) != 2 {
		t.Fatalf("expected 2 blocks, got %d: %+v", len(blocks), blocks)
	}
	txt := blockText(t, blocks[1])
	if !strings.Contains(txt, "only-cwd-body") {
		t.Errorf("block 1 missing body: %q", txt)
	}
	if !strings.Contains(txt, "project claude (cwd)") {
		t.Errorf("block 1 missing (cwd) label: %q", txt)
	}
	if strings.Contains(txt, "repo-root") || strings.Contains(txt, " up)") {
		t.Errorf("single-file project should not have repo-root/N-up labels: %q", txt)
	}
}

// TestWalk_TwoLevelBothPresent covers task 4.2: parent + child both
// present; parent appears first (outer-to-inner).
func TestWalk_TwoLevelBothPresent(t *testing.T) {
	root := t.TempDir()
	mkdirGit(t, root)
	child := filepath.Join(root, "child")
	writeFile(t, filepath.Join(child, "CLAUDE.md"), "child-body")
	writeFile(t, filepath.Join(root, "CLAUDE.md"), "root-body")
	a := &Assembler{ProjectDir: child, BuiltIn: "ID"}
	blocks, err := a.Assemble(context.Background())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	// Built-in + root + child.
	if len(blocks) != 3 {
		t.Fatalf("expected 3 blocks, got %d", len(blocks))
	}
	rootTxt := blockText(t, blocks[1])
	childTxt := blockText(t, blocks[2])
	if !strings.Contains(rootTxt, "root-body") {
		t.Errorf("block 1 should be root: %q", rootTxt)
	}
	if !strings.Contains(rootTxt, "(repo-root)") {
		t.Errorf("root block missing (repo-root) label: %q", rootTxt)
	}
	if !strings.Contains(childTxt, "child-body") {
		t.Errorf("block 2 should be child: %q", childTxt)
	}
	if !strings.Contains(childTxt, "(cwd)") {
		t.Errorf("child block missing (cwd) label: %q", childTxt)
	}
}

// TestWalk_TwoLevelOnlyChild covers task 4.3: parent missing, child
// present. Only the child block appears.
func TestWalk_TwoLevelOnlyChild(t *testing.T) {
	root := t.TempDir()
	mkdirGit(t, root)
	child := filepath.Join(root, "child")
	writeFile(t, filepath.Join(child, "CLAUDE.md"), "child-only")
	a := &Assembler{ProjectDir: child, BuiltIn: "ID"}
	blocks, err := a.Assemble(context.Background())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if len(blocks) != 2 {
		t.Fatalf("expected 2 blocks (built-in + child), got %d", len(blocks))
	}
	txt := blockText(t, blocks[1])
	if !strings.Contains(txt, "child-only") {
		t.Errorf("block 1 missing child body: %q", txt)
	}
	if !strings.Contains(txt, "(cwd)") {
		t.Errorf("child block missing (cwd) label: %q", txt)
	}
}

// TestWalk_TwoLevelOnlyParent covers task 4.4: parent present, child
// missing. Only the parent block appears.
func TestWalk_TwoLevelOnlyParent(t *testing.T) {
	root := t.TempDir()
	mkdirGit(t, root)
	child := filepath.Join(root, "child")
	writeFile(t, filepath.Join(root, "CLAUDE.md"), "root-only")
	a := &Assembler{ProjectDir: child, BuiltIn: "ID"}
	blocks, err := a.Assemble(context.Background())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if len(blocks) != 2 {
		t.Fatalf("expected 2 blocks (built-in + root), got %d", len(blocks))
	}
	txt := blockText(t, blocks[1])
	if !strings.Contains(txt, "root-only") {
		t.Errorf("block 1 missing root body: %q", txt)
	}
	if !strings.Contains(txt, "(repo-root)") {
		t.Errorf("root block missing (repo-root) label: %q", txt)
	}
}

// TestWalk_StopsAtGitMarker covers task 4.5: three-level deep with
// `.git` at the middle level. Walk reads from `.git` down; does not
// read above the marker.
func TestWalk_StopsAtGitMarker(t *testing.T) {
	tmp := t.TempDir()
	root := filepath.Join(tmp, "a")
	mid := filepath.Join(root, "b")
	leaf := filepath.Join(mid, "c")
	mkdirGit(t, root)
	// Plant files at three levels PLUS one above the marker.
	writeFile(t, filepath.Join(root, "CLAUDE.md"), "root-body")
	writeFile(t, filepath.Join(mid, "CLAUDE.md"), "mid-body")
	writeFile(t, filepath.Join(leaf, "CLAUDE.md"), "leaf-body")
	writeFile(t, filepath.Join(tmp, "CLAUDE.md"), "ABOVE-MARKER")
	a := &Assembler{ProjectDir: leaf, BuiltIn: "ID"}
	blocks, err := a.Assemble(context.Background())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	// Built-in + root + mid + leaf (no above-marker).
	if len(blocks) != 4 {
		t.Fatalf("expected 4 blocks, got %d", len(blocks))
	}
	for i, want := range []string{"root-body", "mid-body", "leaf-body"} {
		txt := blockText(t, blocks[i+1])
		if !strings.Contains(txt, want) {
			t.Errorf("block %d = %q, want body %q", i+1, txt, want)
		}
	}
	// Verify no block contains the above-marker content.
	for _, b := range blocks {
		if strings.Contains(blockText(t, b), "ABOVE-MARKER") {
			t.Errorf("walk read above the .git marker")
		}
	}
}

// TestWalk_WalkToRootIgnoresGit covers task 4.6: WalkToRoot=true
// continues past the `.git` marker toward the filesystem root.
func TestWalk_WalkToRootIgnoresGit(t *testing.T) {
	tmp := t.TempDir()
	root := filepath.Join(tmp, "a")
	mid := filepath.Join(root, "b")
	leaf := filepath.Join(mid, "c")
	mkdirGit(t, root)
	writeFile(t, filepath.Join(tmp, "CLAUDE.md"), "above-marker-body")
	writeFile(t, filepath.Join(leaf, "CLAUDE.md"), "leaf-body")
	a := &Assembler{ProjectDir: leaf, WalkToRoot: true, BuiltIn: "ID"}
	blocks, err := a.Assemble(context.Background())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	// Find whether any block contains the above-marker body.
	found := false
	for _, b := range blocks {
		if strings.Contains(blockText(t, b), "above-marker-body") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("WalkToRoot=true did not read above the .git marker; blocks: %+v", blocks)
	}
}

// TestWalk_MaxAncestorDepthCap covers task 4.7: MaxAncestorDepth=1
// reads only cwd and cwd/.. (one level up).
func TestWalk_MaxAncestorDepthCap(t *testing.T) {
	tmp := t.TempDir()
	lvl0 := filepath.Join(tmp, "a") // 3 up
	lvl1 := filepath.Join(lvl0, "b") // 2 up
	lvl2 := filepath.Join(lvl1, "c") // 1 up
	lvl3 := filepath.Join(lvl2, "d") // cwd
	mkdirGit(t, lvl0) // marker far away; MaxAncestorDepth should bind first
	writeFile(t, filepath.Join(lvl0, "CLAUDE.md"), "three-up-body")
	writeFile(t, filepath.Join(lvl1, "CLAUDE.md"), "two-up-body")
	writeFile(t, filepath.Join(lvl2, "CLAUDE.md"), "one-up-body")
	writeFile(t, filepath.Join(lvl3, "CLAUDE.md"), "cwd-body")
	a := &Assembler{ProjectDir: lvl3, MaxAncestorDepth: 1, BuiltIn: "ID"}
	blocks, err := a.Assemble(context.Background())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	// Built-in + one-up + cwd (maxDepth=1 = cwd and one parent only).
	if len(blocks) != 3 {
		t.Fatalf("expected 3 blocks, got %d", len(blocks))
	}
	for _, b := range blocks {
		txt := blockText(t, b)
		if strings.Contains(txt, "two-up-body") || strings.Contains(txt, "three-up-body") {
			t.Errorf("MaxAncestorDepth=1 walked past one parent: %q", txt)
		}
	}
	if !strings.Contains(blockText(t, blocks[1]), "one-up-body") {
		t.Errorf("block 1 should be one-up: %q", blockText(t, blocks[1]))
	}
	if !strings.Contains(blockText(t, blocks[2]), "cwd-body") {
		t.Errorf("block 2 should be cwd: %q", blockText(t, blocks[2]))
	}
}

// TestWalk_FifthGlobalLookup covers task 4.8: <ConfigDir>/SYSTEM.md
// and <ConfigDir>/CLAUDE.md both appear, in that order, before the
// ancestor walks.
func TestWalk_FifthGlobalLookup(t *testing.T) {
	conf := t.TempDir()
	proj := t.TempDir()
	mkdirGit(t, proj)
	writeFile(t, filepath.Join(conf, "SYSTEM.md"), "global-system")
	writeFile(t, filepath.Join(conf, "CLAUDE.md"), "user-claude")
	writeFile(t, filepath.Join(proj, "CLAUDE.md"), "proj-claude")
	a := &Assembler{ConfigDir: conf, ProjectDir: proj, BuiltIn: "ID"}
	blocks, err := a.Assemble(context.Background())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	// ID + global-system + user-claude + proj-claude.
	if len(blocks) != 4 {
		t.Fatalf("expected 4 blocks, got %d", len(blocks))
	}
	want := []string{"ID", "global-system", "user-claude", "proj-claude"}
	for i, w := range want {
		txt := blockText(t, blocks[i])
		if !strings.Contains(txt, w) {
			t.Errorf("block %d = %q, want %q", i, txt, w)
		}
	}
	// Verify the user claude block label.
	if !strings.Contains(blockText(t, blocks[2]), "[source: user claude]") {
		t.Errorf("block 2 missing 'user claude' label: %q", blockText(t, blocks[2]))
	}
}

// TestWalk_EmptyConfigDir covers task 4.9: ConfigDir = "" produces
// no global blocks but the ancestor walk still works.
func TestWalk_EmptyConfigDir(t *testing.T) {
	proj := t.TempDir()
	mkdirGit(t, proj)
	writeFile(t, filepath.Join(proj, "CLAUDE.md"), "proj-body")
	a := &Assembler{ConfigDir: "", ProjectDir: proj, BuiltIn: "ID"}
	blocks, err := a.Assemble(context.Background())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	// Built-in + proj CLAUDE.md only (no global blocks).
	if len(blocks) != 2 {
		t.Fatalf("expected 2 blocks, got %d", len(blocks))
	}
	for _, b := range blocks {
		txt := blockText(t, b)
		if strings.Contains(txt, "[source: global system]") || strings.Contains(txt, "[source: user claude]") {
			t.Errorf("empty ConfigDir produced a global block: %q", txt)
		}
	}
}

// TestWalk_SymlinkFollow covers task 4.10: ProjectDir/CLAUDE.md is a
// symlink to a file outside the project. os.ReadFile follows the
// symlink and reads the target content.
func TestWalk_SymlinkFollow(t *testing.T) {
	proj := t.TempDir()
	mkdirGit(t, proj)
	// Place the target outside proj so we can detect leakage.
	target := filepath.Join(t.TempDir(), "secret.md")
	writeFile(t, target, "symlink-target-body")
	if err := os.Symlink(target, filepath.Join(proj, "CLAUDE.md")); err != nil {
		t.Fatalf("symlink: %v", err)
	}
	a := &Assembler{ProjectDir: proj, BuiltIn: "ID"}
	blocks, err := a.Assemble(context.Background())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if len(blocks) != 2 {
		t.Fatalf("expected 2 blocks, got %d", len(blocks))
	}
	txt := blockText(t, blocks[1])
	if !strings.Contains(txt, "symlink-target-body") {
		t.Errorf("symlink target content not read: %q", txt)
	}
}

// TestWalk_PerTurnReread covers task 4.11: Assemble re-reads files
// every call. Modifying a file between calls is visible on the next
// Assemble.
func TestWalk_PerTurnReread(t *testing.T) {
	proj := t.TempDir()
	mkdirGit(t, proj)
	claudePath := filepath.Join(proj, "CLAUDE.md")
	writeFile(t, claudePath, "v1-body")
	a := &Assembler{ProjectDir: proj, BuiltIn: "ID"}
	b1, err := a.Assemble(context.Background())
	if err != nil {
		t.Fatalf("Assemble (1): %v", err)
	}
	if !strings.Contains(blockText(t, b1[1]), "v1-body") {
		t.Errorf("first call missing v1: %q", blockText(t, b1[1]))
	}
	// Edit the file in place.
	writeFile(t, claudePath, "v2-body")
	b2, err := a.Assemble(context.Background())
	if err != nil {
		t.Fatalf("Assemble (2): %v", err)
	}
	if !strings.Contains(blockText(t, b2[1]), "v2-body") {
		t.Errorf("second call missing v2 (no re-read): %q", blockText(t, b2[1]))
	}
	if strings.Contains(blockText(t, b2[1]), "v1-body") {
		t.Errorf("second call returned stale content: %q", blockText(t, b2[1]))
	}
}

// TestWalk_ConcurrentCalls covers task 4.12: two goroutines call
// Assemble simultaneously against the same Assembler. Run with
// -race; the Assembler is immutable so no race should fire.
func TestWalk_ConcurrentCalls(t *testing.T) {
	proj := t.TempDir()
	mkdirGit(t, proj)
	writeFile(t, filepath.Join(proj, "CLAUDE.md"), "concurrent-body")
	a := &Assembler{ProjectDir: proj, BuiltIn: "ID"}
	const N = 8
	var wg sync.WaitGroup
	wg.Add(N)
	errs := make(chan error, N)
	for i := 0; i < N; i++ {
		go func() {
			defer wg.Done()
			if _, err := a.Assemble(context.Background()); err != nil {
				errs <- err
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("concurrent Assemble failed: %v", err)
		}
	}
}

// TestNewAssembler_DefaultsPreserveBehavior verifies that the
// zero-value WalkOpts from NewAssembler preserves today's behavior:
// for a project with a .git marker at ProjectDir, only one CLAUDE.md
// block appears (no ancestor walk past ProjectDir).
func TestNewAssembler_DefaultsPreserveBehavior(t *testing.T) {
	proj := t.TempDir()
	mkdirGit(t, proj)
	writeFile(t, filepath.Join(proj, "CLAUDE.md"), "body")
	a := NewAssembler("", proj)
	if a.StopDir != "" || a.WalkToRoot || a.MaxAncestorDepth != 0 {
		t.Errorf("NewAssembler did not return zero-value walk opts: %+v", a)
	}
	blocks, err := a.Assemble(context.Background())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	// NewAssembler sets BuiltIn=BuiltInSystem, so expect built-in + CLAUDE.md.
	if len(blocks) != 2 {
		t.Errorf("expected 2 blocks (built-in + CLAUDE.md), got %d", len(blocks))
	}
	txt := blockText(t, blocks[1])
	if !strings.Contains(txt, "(cwd)") {
		t.Errorf("default walk for repo-root project should label block (cwd): %q", txt)
	}
}
