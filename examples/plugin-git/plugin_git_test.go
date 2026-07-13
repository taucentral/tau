// plugin_git_test.go — automated bit-rot check for the reference git plugin.
//
// This test builds the plugin, installs it, sets up a fixture git repo,
// and calls git.status / git.diff through the Manager to verify the
// plugin still works. It skips when `git` is not on PATH or in -short mode.
package main

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/taucentral/tau/internal/llm"
	"github.com/taucentral/tau/internal/plugins"
	"github.com/taucentral/tau/internal/tools"
)

// TestPluginGit_StatusAndDiff builds the plugin, installs it into a temp
// plugins dir, creates a git fixture with one committed file and one
// uncommitted modification, then verifies git.status reflects the modification
// and git.diff shows the delta.
func TestPluginGit_StatusAndDiff(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping plugin-git test in -short mode")
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH; skipping plugin-git test")
	}

	// Build the plugin binary.
	binDir := t.TempDir()
	binPath := filepath.Join(binDir, "tau-plugin-git")
	if runtime.GOOS == "windows" {
		binPath += ".exe"
	}
	buildCmd := exec.Command("go", "build", "-o="+binPath, ".")
	var buildErr strings.Builder
	buildCmd.Stderr = &buildErr
	if err := buildCmd.Run(); err != nil {
		t.Fatalf("go build plugin-git: %v\n%s", err, buildErr.String())
	}

	// Install into a plugins dir.
	pluginsDir := filepath.Join(t.TempDir(), "plugins")
	if err := os.MkdirAll(pluginsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(pluginsDir, "tau-plugin-git")
	if runtime.GOOS == "windows" {
		dst += ".exe"
	}
	if err := os.Rename(binPath, dst); err != nil {
		t.Fatalf("rename: %v", err)
	}

	// Create a Manager and spawn.
	hostSrv := plugins.NoopHostServer()
	mgr, err := plugins.NewManager(pluginsDir, "", hostSrv)
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	if _, err := mgr.SpawnAll(context.Background()); err != nil {
		t.Fatalf("SpawnAll: %v", err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = mgr.Shutdown(ctx)
	})

	// Create a git fixture repo.
	repoDir := t.TempDir()
	if err := initGitRepo(repoDir); err != nil {
		t.Fatalf("initGitRepo: %v", err)
	}

	// Write and commit README.md.
	readme := filepath.Join(repoDir, "README.md")
	if err := os.WriteFile(readme, []byte("# initial\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := runGitCmd(repoDir, "add", "README.md"); err != nil {
		t.Fatalf("git add: %v", err)
	}
	if err := runGitCmd(repoDir, "commit", "-m", "initial"); err != nil {
		t.Fatalf("git commit: %v", err)
	}

	// Modify README.md without committing.
	if err := os.WriteFile(readme, []byte("# modified\nsecond line\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Call git.status through the plugin.
	statusResult, err := mgr.Execute(context.Background(), tools.ToolCall{
		ID:   "tu_status",
		Name: "git.status",
		Args: json.RawMessage(`{}`),
		Cwd:  repoDir,
	})
	if err != nil {
		t.Fatalf("git.status execute: %v", err)
	}
	if statusResult.IsError {
		t.Fatalf("git.status returned error: %+v", statusResult.Content)
	}
	// The status JSON is embedded somewhere in the content blocks. Serialize
	// everything and search for README.md.
	rawStatus, _ := json.Marshal(statusResult.Content)
	if !strings.Contains(string(rawStatus), "README.md") {
		t.Errorf("git.status output missing README.md; result: %s", string(rawStatus))
	}

	// Call git.diff through the plugin.
	diffResult, err := mgr.Execute(context.Background(), tools.ToolCall{
		ID:   "tu_diff",
		Name: "git.diff",
		Args: json.RawMessage(`{}`),
		Cwd:  repoDir,
	})
	if err != nil {
		t.Fatalf("git.diff execute: %v", err)
	}
	if diffResult.IsError {
		t.Fatalf("git.diff returned error: %+v", diffResult.Content)
	}
	// The diff output should mention README.md and show the modification.
	rawDiff, _ := json.Marshal(diffResult.Content)
	diffStr := string(rawDiff)
	if !strings.Contains(diffStr, "README.md") {
		t.Errorf("git.diff output missing README.md; result: %s", diffStr)
	}
	if !strings.Contains(diffStr, "modified") && !strings.Contains(diffStr, "second line") {
		t.Errorf("git.diff output missing modification; result: %s", diffStr)
	}

	// Silence unused import guard.
	_ = llm.TextContent{}
}

// initGitRepo initializes a git repo in dir and configures a test user so
// commits don't fail on missing identity.
func initGitRepo(dir string) error {
	if err := runGitCmd(dir, "init"); err != nil {
		return err
	}
	if err := runGitCmd(dir, "config", "user.email", "test@example.com"); err != nil {
		return err
	}
	if err := runGitCmd(dir, "config", "user.name", "Test User"); err != nil {
		return err
	}
	if err := runGitCmd(dir, "config", "commit.gpgsign", "false"); err != nil {
		return err
	}
	return nil
}

// runGitCmd runs git with the given args in dir. Distinct from main.go's
// runGit which returns ([]byte, error).
func runGitCmd(dir string, args ...string) error {
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		return &gitCmdError{err: err, output: string(out)}
	}
	return nil
}

type gitCmdError struct {
	err    error
	output string
}

func (e *gitCmdError) Error() string {
	return e.err.Error() + ": " + e.output
}
