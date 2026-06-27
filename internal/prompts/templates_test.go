package prompts

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestLoader_ProjectShadowsGlobal verifies that the project copy of a
// template wins when both directories have the same name.
func TestLoader_ProjectShadowsGlobal(t *testing.T) {
	conf := t.TempDir()
	proj := t.TempDir()
	writeFile(t, filepath.Join(conf, "agent", "prompts", "commit.md"), "global template")
	writeFile(t, filepath.Join(proj, ".tau", "prompts", "commit.md"), "project template")

	l := NewLoader(conf, proj)
	got, err := l.Load("commit")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !strings.Contains(got, "project template") {
		t.Errorf("expected project template to win; got %q", got)
	}
}

// TestLoader_GlobalFallback verifies that the global copy is returned
// when no project copy exists.
func TestLoader_GlobalFallback(t *testing.T) {
	conf := t.TempDir()
	proj := t.TempDir()
	writeFile(t, filepath.Join(conf, "agent", "prompts", "commit.md"), "global template")
	l := NewLoader(conf, proj)
	got, err := l.Load("commit")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !strings.Contains(got, "global template") {
		t.Errorf("expected global template; got %q", got)
	}
}

// TestLoader_NotFound verifies that a missing template returns
// ErrTemplateNotFound.
func TestLoader_NotFound(t *testing.T) {
	l := NewLoader(t.TempDir(), t.TempDir())
	_, err := l.Load("missing")
	if !errors.Is(err, ErrTemplateNotFound) {
		t.Errorf("expected ErrTemplateNotFound, got %v", err)
	}
}

// TestLoader_InvalidNames verifies that invalid template names are
// rejected before any filesystem access.
func TestLoader_InvalidNames(t *testing.T) {
	cases := []string{
		"",          // empty
		".hidden",   // leading dot
		"UPPER",     // uppercase
		"has space", // space
		"has/slash", // path separator
		"with:colon",
		strings.Repeat("a", 65), // too long
	}
	l := NewLoader(t.TempDir(), t.TempDir())
	for _, name := range cases {
		if _, err := l.Load(name); !errors.Is(err, ErrInvalidTemplateName) {
			t.Errorf("Load(%q) err = %v, want ErrInvalidTemplateName", name, err)
		}
	}
}

// TestLoader_ValidNames verifies that names with dots, dashes,
// underscores, and digits are accepted.
func TestLoader_ValidNames(t *testing.T) {
	conf := t.TempDir()
	writeFile(t, filepath.Join(conf, "agent", "prompts", "v1.2.md"), "body")
	writeFile(t, filepath.Join(conf, "agent", "prompts", "commit_message.md"), "body")
	writeFile(t, filepath.Join(conf, "agent", "prompts", "review-pr.md"), "body")
	l := NewLoader(conf, "")
	for _, name := range []string{"v1.2", "commit_message", "review-pr"} {
		if _, err := l.Load(name); err != nil {
			t.Errorf("Load(%q): %v", name, err)
		}
	}
}

// TestLoader_Available enumerates both dirs, dedupes, and sorts.
func TestLoader_Available(t *testing.T) {
	conf := t.TempDir()
	proj := t.TempDir()
	writeFile(t, filepath.Join(conf, "agent", "prompts", "alpha.md"), "")
	writeFile(t, filepath.Join(conf, "agent", "prompts", "beta.md"), "")
	writeFile(t, filepath.Join(proj, ".tau", "prompts", "gamma.md"), "")
	writeFile(t, filepath.Join(proj, ".tau", "prompts", "alpha.md"), "") // shadow
	writeFile(t, filepath.Join(proj, ".tau", "prompts", "ignored.txt"), "")

	l := NewLoader(conf, proj)
	got, err := l.Available()
	if err != nil {
		t.Fatalf("Available: %v", err)
	}
	want := []string{"alpha", "beta", "gamma"}
	if len(got) != len(want) {
		t.Fatalf("Available = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("Available[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

// TestLoader_AvailableNeitherDirExists verifies Available tolerates
// missing dirs.
func TestLoader_AvailableNeitherDirExists(t *testing.T) {
	l := NewLoader(
		filepath.Join(t.TempDir(), "nope-conf"),
		filepath.Join(t.TempDir(), "nope-proj"),
	)
	got, err := l.Available()
	if err != nil {
		t.Fatalf("Available: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("Available should be empty; got %v", got)
	}
}

// TestLoader_NilSafe verifies that a Loader with empty dirs returns
// ErrTemplateNotFound for any name (no panic).
func TestLoader_NilSafe(t *testing.T) {
	l := &Loader{}
	_, err := l.Load("anything")
	if !errors.Is(err, ErrTemplateNotFound) {
		t.Errorf("expected ErrTemplateNotFound, got %v", err)
	}
	got, err := l.Available()
	if err != nil {
		t.Fatalf("Available: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("Available should be empty; got %v", got)
	}
}

// TestLoader_ReadFailurePropagates verifies that genuine read failures
// (e.g., a directory in place of the template file) are returned
// rather than swallowed as "not found".
func TestLoader_ReadFailurePropagates(t *testing.T) {
	conf := t.TempDir()
	// Place a directory at commit.md so ReadFile fails with EISDIR.
	dirPath := filepath.Join(conf, "agent", "prompts", "commit.md")
	if err := os.MkdirAll(dirPath, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	l := NewLoader(conf, "")
	_, err := l.Load("commit")
	if err == nil {
		t.Fatalf("expected read error, got nil")
	}
	if errors.Is(err, ErrTemplateNotFound) {
		t.Errorf("error should not be ErrTemplateNotFound for read failure; got %v", err)
	}
}
