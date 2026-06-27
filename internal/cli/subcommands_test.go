package cli

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coevin/tau/internal/config"
)

func TestRunConfigSubcommand_Path(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("TAU_CONFIG_DIR", dir)
	var out bytes.Buffer
	orig := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w
	defer func() {
		os.Stdout = orig
		_ = w.Close()
		_, _ = io.Copy(&out, r)
	}()
	err := runConfigSubcommand(context.Background(), Args{
		Subcommand:     "config",
		SubcommandArgs: []string{"--path"},
	})
	if err != nil {
		t.Fatalf("runConfigSubcommand: %v", err)
	}
	_ = w.Close()
	_, _ = io.Copy(&out, r)
	got := strings.TrimSpace(out.String())
	want := filepath.Join(dir, "agent")
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestRunConfigSubcommand_NoFlagsPrintsBoth(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("TAU_CONFIG_DIR", dir)

	// Capture stdout via a temp file to avoid pipe races.
	tmp := filepath.Join(t.TempDir(), "out.txt")
	f, err := os.Create(tmp)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	orig := os.Stdout
	os.Stdout = f
	err = runConfigSubcommand(context.Background(), Args{Subcommand: "config"})
	os.Stdout = orig
	_ = f.Close()
	if err != nil {
		t.Fatalf("runConfigSubcommand: %v", err)
	}
	data, err := os.ReadFile(tmp)
	if err != nil {
		t.Fatalf("read tmp: %v", err)
	}
	s := string(data)
	if !strings.Contains(s, "Agent directory:") {
		t.Errorf("missing 'Agent directory:' in output:\n%s", s)
	}
	if !strings.Contains(s, "Settings file:") {
		t.Errorf("missing 'Settings file:' in output:\n%s", s)
	}
	if !strings.Contains(s, filepath.Join(dir, "agent")) {
		t.Errorf("missing agent dir path in output:\n%s", s)
	}
}

func TestRunUpdateSubcommand_StubsToStderr(t *testing.T) {
	// Capture stderr.
	orig := os.Stderr
	r, w, _ := os.Pipe()
	os.Stderr = w
	err := runUpdateSubcommand(context.Background(), Args{Subcommand: "update"})
	_ = w.Close()
	os.Stderr = orig
	if err != nil {
		t.Fatalf("runUpdateSubcommand returned error: %v", err)
	}
	var buf bytes.Buffer
	_, _ = io.Copy(&buf, r)
	got := buf.String()
	if !strings.Contains(got, "self-update") || !strings.Contains(got, "package manager") {
		t.Errorf("expected message about self-update + package manager, got: %q", got)
	}
}

func TestListModels_EmptyFile(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("TAU_CONFIG_DIR", dir)

	// Capture stdout to a temp file (avoid pipe races with the tabwriter).
	tmp := filepath.Join(t.TempDir(), "out.txt")
	f, err := os.Create(tmp)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	orig := os.Stdout
	os.Stdout = f
	err = listModels(context.Background(), Args{})
	os.Stdout = orig
	_ = f.Close()
	if err != nil {
		t.Fatalf("listModels: %v", err)
	}
	data, err := os.ReadFile(tmp)
	if err != nil {
		t.Fatalf("read tmp: %v", err)
	}
	s := string(data)
	if !strings.Contains(s, "PROVIDER") || !strings.Contains(s, "MODEL ID") {
		t.Errorf("missing header row:\n%s", s)
	}
	// Built-in models must be present.
	for _, m := range builtinModels {
		if !strings.Contains(s, m.ID) {
			t.Errorf("missing built-in model %q in output:\n%s", m.ID, s)
		}
	}
}

func TestListModels_WithModelsFile(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("TAU_CONFIG_DIR", dir)
	agentDir := filepath.Join(dir, "agent")
	if err := os.MkdirAll(agentDir, 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	modelsJSON := `{
		"providers": {
			"ollama": {
				"baseUrl": "http://localhost:11434/v1",
				"api": "openai",
				"models": [{"id": "llama3", "contextWindow": 8000}]
			}
		},
		"models": [{"id": "custom-model", "api": "anthropic", "contextWindow": 200000}]
	}`
	if err := os.WriteFile(filepath.Join(agentDir, "models.json"), []byte(modelsJSON), 0o600); err != nil {
		t.Fatalf("write models.json: %v", err)
	}

	tmp := filepath.Join(t.TempDir(), "out.txt")
	f, err := os.Create(tmp)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	orig := os.Stdout
	os.Stdout = f
	err = listModels(context.Background(), Args{})
	os.Stdout = orig
	_ = f.Close()
	if err != nil {
		t.Fatalf("listModels: %v", err)
	}
	data, _ := os.ReadFile(tmp)
	s := string(data)
	if !strings.Contains(s, "llama3") {
		t.Errorf("missing provider-attached model llama3:\n%s", s)
	}
	if !strings.Contains(s, "custom-model") {
		t.Errorf("missing top-level custom-model:\n%s", s)
	}
	if !strings.Contains(s, "claude-opus-4-5") {
		t.Errorf("built-ins should still appear alongside file entries:\n%s", s)
	}
}

func TestListModels_BadModelsFileReturnsError(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("TAU_CONFIG_DIR", dir)
	agentDir := filepath.Join(dir, "agent")
	if err := os.MkdirAll(agentDir, 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "models.json"), []byte("{not json"), 0o600); err != nil {
		t.Fatalf("write models.json: %v", err)
	}
	err := listModels(context.Background(), Args{})
	if err == nil {
		t.Fatalf("expected error for malformed models.json")
	}
}

func TestMergeModelLists_BuiltinsPlusFile(t *testing.T) {
	mf := &config.ModelsFile{
		Models: []config.ModelDefinition{
			{ID: "custom", API: config.APIAnthropic, ContextWindow: 1000},
		},
		Providers: map[string]config.ProviderDefinition{
			"ollama": {
				API: config.APIOpenAI,
				Models: []config.ModelDefinition{
					{ID: "llama3", ContextWindow: 8000},
				},
			},
		},
	}
	got := mergeModelLists(builtinModels, mf)
	ids := make(map[string]bool, len(got))
	for _, m := range got {
		ids[m.ID] = true
	}
	if !ids["custom"] || !ids["llama3"] || !ids["claude-opus-4-5"] {
		t.Errorf("missing expected IDs in merged list: %v", ids)
	}
}

func TestMergeModelLists_ShadowsBuiltinByID(t *testing.T) {
	mf := &config.ModelsFile{
		Models: []config.ModelDefinition{
			{ID: "claude-opus-4-5", API: config.APIAnthropic, ContextWindow: 999},
		},
	}
	got := mergeModelLists(builtinModels, mf)
	var found config.ModelDefinition
	for _, m := range got {
		if m.ID == "claude-opus-4-5" {
			found = m
		}
	}
	if found.ContextWindow != 999 {
		t.Errorf("expected file entry to shadow builtin (ctx=999), got %d", found.ContextWindow)
	}
}

func TestFormatContextWindow(t *testing.T) {
	cases := []struct {
		in   int
		want string
	}{
		{0, "-"},
		{-1, "-"},
		{500, "500"},
		{1000, "1k"},
		{128000, "128k"},
		{200000, "200k"},
	}
	for _, c := range cases {
		if got := formatContextWindow(c.in); got != c.want {
			t.Errorf("formatContextWindow(%d) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestFormatCost(t *testing.T) {
	cases := []struct {
		name string
		cost *config.ModelCost
		want string
	}{
		{"nil", nil, "-"},
		{"input-only", &config.ModelCost{Input: 3}, "3/0"},
		{"full", &config.ModelCost{Input: 15, Output: 75}, "15/75"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := formatCost(c.cost); got != c.want {
				t.Errorf("got %q, want %q", got, c.want)
			}
		})
	}
}

func TestHasSubcommandArg(t *testing.T) {
	args := Args{SubcommandArgs: []string{"--path", "extra"}}
	if !hasSubcommandArg(args, "--path") {
		t.Errorf("expected --path to be found")
	}
	if hasSubcommandArg(args, "--missing") {
		t.Errorf("--missing should not be found")
	}
}
