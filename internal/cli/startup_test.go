package cli

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/taucentral/tau/internal/config"
)

func TestMaybeSetup_NoSetupFlag_Skips(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("TAU_CONFIG_DIR", dir)
	if err := maybeSetup(context.Background(), Args{NoSetup: true}); err != nil {
		t.Fatalf("maybeSetup: %v", err)
	}
	// Verify nothing was written.
	if _, err := os.Stat(filepath.Join(dir, "agent", "settings.json")); !os.IsNotExist(err) {
		t.Errorf("expected no settings.json after --no-setup, got err=%v", err)
	}
}

func TestMaybeSetup_ExistingSettings_Skips(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("TAU_CONFIG_DIR", dir)
	agentDir := filepath.Join(dir, "agent")
	if err := os.MkdirAll(agentDir, 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	existing := []byte(`{"theme":"dark"}`)
	if err := os.WriteFile(filepath.Join(agentDir, "settings.json"), existing, 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := maybeSetup(context.Background(), Args{}); err != nil {
		t.Fatalf("maybeSetup: %v", err)
	}
	got, err := os.ReadFile(filepath.Join(agentDir, "settings.json"))
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if !bytes.Equal(got, existing) {
		t.Errorf("maybeSetup overwrote pre-existing settings.json:\n got: %s\nwant: %s", got, existing)
	}
}

func TestMaybeSetup_NonTTY_WritesDefaultConfig(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("TAU_CONFIG_DIR", dir)
	// In `go test`, stdin/stdout are not TTYs.
	if err := maybeSetup(context.Background(), Args{}); err != nil {
		t.Fatalf("maybeSetup: %v", err)
	}
	settingsPath := filepath.Join(dir, "agent", "settings.json")
	data, err := os.ReadFile(settingsPath)
	if err != nil {
		t.Fatalf("read settings.json: %v", err)
	}
	if len(data) == 0 {
		t.Fatalf("settings.json is empty")
	}
	// Verify the file parses as Settings.
	storage, err := config.NewFileSettingsStorage(filepath.Join(dir, "agent"), filepath.Join(dir, "agent"), true)
	if err != nil {
		t.Fatalf("storage: %v", err)
	}
	defer storage.Close()
	s, err := storage.Load(context.Background())
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if s.DefaultThinkingLevel == nil || *s.DefaultThinkingLevel != config.ThinkingOff {
		t.Errorf("expected default thinking level off, got %+v", s.DefaultThinkingLevel)
	}
}

func TestPromptSetup_ParsesAnswers(t *testing.T) {
	in := strings.NewReader("light\nclaude-opus-4-5\ny\n")
	var out bytes.Buffer
	answers, err := promptSetup(in, &out)
	if err != nil {
		t.Fatalf("promptSetup: %v", err)
	}
	if answers.Theme != "light" {
		t.Errorf("Theme = %q, want light", answers.Theme)
	}
	if answers.DefaultModel != "claude-opus-4-5" {
		t.Errorf("DefaultModel = %q, want claude-opus-4-5", answers.DefaultModel)
	}
	if !answers.EnableAnalytics {
		t.Errorf("EnableAnalytics = false, want true")
	}
}

func TestPromptSetup_EmptyDefaults(t *testing.T) {
	in := strings.NewReader("\n\n\n")
	var out bytes.Buffer
	answers, err := promptSetup(in, &out)
	if err != nil {
		t.Fatalf("promptSetup: %v", err)
	}
	if answers.Theme != "dark" {
		t.Errorf("default Theme = %q, want dark", answers.Theme)
	}
	if answers.DefaultModel != "" {
		t.Errorf("default DefaultModel = %q, want empty", answers.DefaultModel)
	}
	if answers.EnableAnalytics {
		t.Errorf("default EnableAnalytics = true, want false")
	}
}

func TestPromptSetup_BadThemeFallsBackToDefault(t *testing.T) {
	in := strings.NewReader("garbage\n\nn\n")
	var out bytes.Buffer
	answers, err := promptSetup(in, &out)
	if err != nil {
		t.Fatalf("promptSetup: %v", err)
	}
	if answers.Theme != "dark" {
		t.Errorf("garbage Theme = %q, want dark default", answers.Theme)
	}
}

func TestNormalizeYesNo(t *testing.T) {
	cases := []struct {
		in   string
		def  bool
		want bool
	}{
		{"", false, false},
		{"", true, true},
		{"y", false, true},
		{"Y", false, true},
		{"yes", false, true},
		{"n", true, false},
		{"no", true, false},
		{"garbage", false, false},
	}
	for _, c := range cases {
		if got := normalizeYesNo(c.in, c.def); got != c.want {
			t.Errorf("normalizeYesNo(%q, %v) = %v, want %v", c.in, c.def, got, c.want)
		}
	}
}

func TestWriteDefaultConfig_CreatesSettingsJSON(t *testing.T) {
	dir := t.TempDir()
	agentDir := filepath.Join(dir, "agent")
	if err := writeDefaultConfig(agentDir); err != nil {
		t.Fatalf("writeDefaultConfig: %v", err)
	}
	info, err := os.Stat(filepath.Join(agentDir, "settings.json"))
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Errorf("settings.json mode = %v, want 0600", info.Mode().Perm())
	}
}

func TestWriteSetupAnswers_PersistsAnswers(t *testing.T) {
	dir := t.TempDir()
	agentDir := filepath.Join(dir, "agent")
	answers := setupAnswers{
		Theme:           "light",
		DefaultModel:    "claude-opus-4-5",
		EnableAnalytics: true,
	}
	if err := writeSetupAnswers(agentDir, answers); err != nil {
		t.Fatalf("writeSetupAnswers: %v", err)
	}
	storage, err := config.NewFileSettingsStorage(agentDir, agentDir, true)
	if err != nil {
		t.Fatalf("storage: %v", err)
	}
	defer storage.Close()
	s, err := storage.Load(context.Background())
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if s.Theme == nil || *s.Theme != "light" {
		t.Errorf("Theme not persisted: %+v", s.Theme)
	}
	if s.DefaultModel == nil || *s.DefaultModel != "claude-opus-4-5" {
		t.Errorf("DefaultModel not persisted: %+v", s.DefaultModel)
	}
	if s.EnableAnalytics == nil || !*s.EnableAnalytics {
		t.Errorf("EnableAnalytics not persisted: %+v", s.EnableAnalytics)
	}
}

// TestMaybeSetup_PipedInputAutoSkips ensures that when stdin is piped
// (not a TTY), the wizard is auto-skipped even without --no-setup.
// In `go test`, stdin is not a TTY, so this matches the CI scenario.
func TestMaybeSetup_PipedInputAutoSkips(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("TAU_CONFIG_DIR", dir)

	// Replace stdin with a pipe (non-TTY) to make the test explicit.
	orig := os.Stdin
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	os.Stdin = r
	defer func() {
		os.Stdin = orig
		_ = r.Close()
		_ = w.Close()
	}()

	// Write some data so the pipe isn't EOF immediately; we just need
	// stdin to be non-TTY.
	_, _ = io.WriteString(w, "this should be ignored\n")

	if err := maybeSetup(context.Background(), Args{}); err != nil {
		t.Fatalf("maybeSetup: %v", err)
	}

	// The non-TTY path writes defaults; verify Theme is unset (default).
	storage, _ := config.NewFileSettingsStorage(filepath.Join(dir, "agent"), filepath.Join(dir, "agent"), true)
	defer storage.Close()
	s, err := storage.Load(context.Background())
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	// The non-TTY path doesn't prompt, so Theme must be nil.
	if s.Theme != nil {
		t.Errorf("non-TTY path wrote Theme=%v, expected nil", *s.Theme)
	}
}
