package tui

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/charmbracelet/lipgloss"
)

func TestDefaultTheme(t *testing.T) {
	tests := []struct {
		name string
		want string
	}{
		{"dark", "dark"},
		{"light", "light"},
		{"", "dark"},      // empty falls back to dark
		{"bogus", "dark"}, // unknown falls back to dark
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DefaultTheme(tt.name)
			if got.Name != tt.want {
				t.Errorf("DefaultTheme(%q).Name = %q, want %q", tt.name, got.Name, tt.want)
			}
		})
	}
}

func TestLoadTheme_BuiltIn(t *testing.T) {
	got, err := LoadTheme("/tmp", "dark")
	if err != nil {
		t.Fatalf("LoadTheme dark: %v", err)
	}
	if got.Name != "dark" {
		t.Errorf("Name = %q, want 'dark'", got.Name)
	}

	got, err = LoadTheme("/tmp", "light")
	if err != nil {
		t.Fatalf("LoadTheme light: %v", err)
	}
	if got.Name != "light" {
		t.Errorf("Name = %q, want 'light'", got.Name)
	}
}

func TestLoadTheme_MissingCustomFallsBack(t *testing.T) {
	dir := t.TempDir()
	got, err := LoadTheme(dir, "nonexistent")
	if err != nil {
		t.Fatalf("LoadTheme: err = %v, want nil", err)
	}
	if got.Name != "dark" {
		t.Errorf("Name = %q, want 'dark' fallback", got.Name)
	}
}

func TestLoadTheme_EmptyName(t *testing.T) {
	dir := t.TempDir()
	got, err := LoadTheme(dir, "")
	if err != nil {
		t.Fatalf("LoadTheme(''): err = %v", err)
	}
	if got.Name != "dark" {
		t.Errorf("Name = %q, want 'dark'", got.Name)
	}
}

func TestLoadTheme_CustomJSON(t *testing.T) {
	dir := t.TempDir()
	themesDir := filepath.Join(dir, "themes")
	if err := os.MkdirAll(themesDir, 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	data := `{
		"name": "ocean",
		"primary": "#7aa2f7",
		"secondary": "#bb9af7",
		"accent": "#e0af68",
		"muted": "#565f89",
		"userMsg": "#7dcfff",
		"assistantMsg": "#c0caf5",
		"toolName": "#73daca",
		"error": "#f7768e",
		"success": "#9ece6a",
		"warning": "#e0af68",
		"indentation": 4
	}`
	path := filepath.Join(themesDir, "ocean.json")
	if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	got, err := LoadTheme(dir, "ocean")
	if err != nil {
		t.Fatalf("LoadTheme ocean: %v", err)
	}
	if got.Name != "ocean" {
		t.Errorf("Name = %q, want 'ocean'", got.Name)
	}
	if got.Indentation != 4 {
		t.Errorf("Indentation = %d, want 4", got.Indentation)
	}
}

func TestLoadTheme_MalformedJSON(t *testing.T) {
	dir := t.TempDir()
	themesDir := filepath.Join(dir, "themes")
	if err := os.MkdirAll(themesDir, 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	path := filepath.Join(themesDir, "broken.json")
	if err := os.WriteFile(path, []byte(`{not-json`), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	_, err := LoadTheme(dir, "broken")
	if err == nil {
		t.Errorf("LoadTheme broken: err = nil, want parse error")
	}
}

func TestLoadTheme_CustomFillsDefaults(t *testing.T) {
	dir := t.TempDir()
	themesDir := filepath.Join(dir, "themes")
	if err := os.MkdirAll(themesDir, 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	// Minimal JSON — missing name and indentation.
	data := `{"primary":"#ff0000"}`
	path := filepath.Join(themesDir, "minimal.json")
	if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	got, err := LoadTheme(dir, "minimal")
	if err != nil {
		t.Fatalf("LoadTheme: %v", err)
	}
	if got.Name != "custom" {
		t.Errorf("Name = %q, want 'custom' default", got.Name)
	}
	if got.Indentation != 2 {
		t.Errorf("Indentation = %d, want default 2", got.Indentation)
	}
}

func TestThemeStyles(t *testing.T) {
	th := DarkTheme
	// Verify each style carries the expected foreground colour.
	// lipgloss strips colours in non-TTY environments, so we compare
	// the style property directly rather than the rendered output.
	tests := []struct {
		name  string
		style lipgloss.Style
		color lipgloss.Color
	}{
		{"User", th.UserStyle(), th.UserMsg},
		{"Assistant", th.AssistantStyle(), th.AssistantMsg},
		{"Tool", th.ToolStyle(), th.ToolName},
		{"Error", th.ErrorStyle(), th.ErrorColor},
		{"Muted", th.MutedStyle(), th.Muted},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.style.GetForeground()
			// lipgloss.Color is a string-based type; compare values.
			if got != tt.color {
				t.Errorf("%s foreground = %v, want %v", tt.name, got, tt.color)
			}
		})
	}
}

func TestThemeIndent(t *testing.T) {
	th := DarkTheme
	indent := th.Indent()
	if len(indent) != th.Indentation {
		t.Errorf("Indent() len = %d, want %d", len(indent), th.Indentation)
	}

	th2 := Theme{Indentation: 0}
	if th2.Indent() != "" {
		t.Errorf("Indent() with 0 = %q, want empty", th2.Indent())
	}
}

func TestDarkAndLightThemesDiffer(t *testing.T) {
	if DarkTheme.Name == LightTheme.Name {
		t.Error("dark and light themes have the same name")
	}
	if DarkTheme.Primary == LightTheme.Primary {
		t.Error("dark and light themes share the same primary colour")
	}
}
