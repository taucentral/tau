package prompts

import (
	"context"
	"os"
	"path/filepath"
	"strings"
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
func TestAssembler_ProjectFilesAppended(t *testing.T) {
	proj := t.TempDir()
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
	a := &Assembler{
		ConfigDir:  filepath.Join(t.TempDir(), "missing-config"),
		ProjectDir: filepath.Join(t.TempDir(), "missing-project"),
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
