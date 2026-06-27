package config

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadModelsFile_MissingFile_ReturnsEmpty(t *testing.T) {
	mf, err := LoadModelsFile(filepath.Join(t.TempDir(), "does-not-exist.json"))
	if err != nil {
		t.Fatalf("missing file should not error: %v", err)
	}
	if mf == nil {
		t.Fatal("expected non-nil ModelsFile")
	}
	if len(mf.Providers) != 0 || len(mf.Models) != 0 {
		t.Errorf("expected empty ModelsFile, got %+v", mf)
	}
}

func TestLoadModelsFile_ValidOllamaProvider(t *testing.T) {
	const data = `{
	  "providers": {
	    "ollama": {
	      "baseUrl": "http://localhost:11434/v1",
	      "api":     "openai",
	      "models":  [{"id": "llama3", "contextWindow": 8192}]
	    }
	  },
	  "models": [{"id": "gpt-4o", "api": "openai", "contextWindow": 128000}]
	}`
	path := writeTemp(t, data)
	mf, err := LoadModelsFile(path)
	if err != nil {
		t.Fatalf("LoadModelsFile: %v", err)
	}
	if _, ok := mf.Providers["ollama"]; !ok {
		t.Errorf("ollama provider missing")
	}
	if got := mf.Providers["ollama"].API; got != APIOpenAI {
		t.Errorf("ollama.api = %q, want openai", got)
	}
	if len(mf.Models) != 1 {
		t.Errorf("len(Models) = %d, want 1", len(mf.Models))
	}
}

func TestLoadModelsFile_UnknownFieldRejected(t *testing.T) {
	const data = `{"providers":{"x":{"completelyUnknown":"typo"}}}`
	path := writeTemp(t, data)
	_, err := LoadModelsFile(path)
	if err == nil {
		t.Fatalf("expected schema violation for unknown field")
	}
	if !errors.Is(err, ErrSchemaViolation) {
		t.Errorf("err = %v, want ErrSchemaViolation", err)
	}
}

func TestLoadModelsFile_InvalidAPI_Rejected(t *testing.T) {
	const data = `{"providers":{"x":{"api":"udp"}}}`
	path := writeTemp(t, data)
	_, err := LoadModelsFile(path)
	if err == nil {
		t.Fatalf("expected schema violation for invalid api")
	}
	if !errors.Is(err, ErrSchemaViolation) {
		t.Errorf("err = %v, want ErrSchemaViolation", err)
	}
}

func TestLoadModelsFile_MissingID_Rejected(t *testing.T) {
	const data = `{"models":[{"api":"openai"}]}`
	path := writeTemp(t, data)
	_, err := LoadModelsFile(path)
	if err == nil {
		t.Fatalf("expected schema violation for missing id")
	}
}

func TestModelsFile_JSONRoundTrip(t *testing.T) {
	in := &ModelsFile{
		Providers: map[string]ProviderDefinition{
			"openai": {
				API:     APIOpenAI,
				BaseURL: "https://api.openai.com/v1",
				ModelOverrides: map[string]ModelDefinition{
					"gpt-4o": {ContextWindow: 192000},
				},
			},
		},
		Models: []ModelDefinition{{ID: "claude-opus-4-5", API: APIAnthropic}},
	}
	data, err := json.MarshalIndent(in, "", "  ")
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	out := &ModelsFile{}
	if err := strictJSONDecode(data, out); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if out.Providers["openai"].API != APIOpenAI {
		t.Errorf("API = %q", out.Providers["openai"].API)
	}
	if out.Providers["openai"].ModelOverrides["gpt-4o"].ContextWindow != 192000 {
		t.Errorf("override contextWindow lost")
	}
}

func TestResolveModel_ProviderOverrideWins(t *testing.T) {
	providers := map[string]ProviderDefinition{
		"openai": {
			API:     APIOpenAI,
			BaseURL: "default-base",
			Models: []ModelDefinition{
				{ID: "gpt-4o", ContextWindow: 128000},
			},
			ModelOverrides: map[string]ModelDefinition{
				"gpt-4o": {ContextWindow: 192000},
			},
		},
	}
	got := ResolveModel(providers, nil, "openai", "gpt-4o")
	if got == nil {
		t.Fatal("expected model, got nil")
	}
	if got.ContextWindow != 192000 {
		t.Errorf("ContextWindow = %d, want 192000 (override wins)", got.ContextWindow)
	}
	if got.API != APIOpenAI {
		t.Errorf("API = %q, want openai (inherited from provider)", got.API)
	}
	if got.BaseURL != "default-base" {
		t.Errorf("BaseURL = %q, want \"default-base\" (inherited)", got.BaseURL)
	}
}

func TestResolveModel_TopLevelFallback(t *testing.T) {
	got := ResolveModel(nil, []ModelDefinition{{ID: "claude", API: APIAnthropic}}, "any", "claude")
	if got == nil {
		t.Fatal("expected top-level model, got nil")
	}
	if got.API != APIAnthropic {
		t.Errorf("API = %q", got.API)
	}
}

func TestResolveModel_NotFound_ReturnsNil(t *testing.T) {
	if got := ResolveModel(nil, nil, "x", "nope"); got != nil {
		t.Errorf("expected nil, got %+v", got)
	}
}

func TestResolveModel_InheritsProviderHeaders(t *testing.T) {
	providers := map[string]ProviderDefinition{
		"gw": {
			API:     APIOpenAI,
			BaseURL: "https://gw.example.com",
			Headers: map[string]string{
				"X-Gateway": "true",
				"X-Tenant":  "acme",
			},
			Models: []ModelDefinition{
				{
					ID:      "gpt-4o",
					Headers: map[string]string{"X-Tenant": "override"}, // overrides one
				},
			},
		},
	}
	got := ResolveModel(providers, nil, "gw", "gpt-4o")
	if got == nil {
		t.Fatal("nil")
	}
	if got.Headers["X-Gateway"] != "true" {
		t.Errorf("inherited header missing")
	}
	if got.Headers["X-Tenant"] != "override" {
		t.Errorf("model override should win on conflict")
	}
}

func TestAllKnownModels_NilFile_ReturnsNil(t *testing.T) {
	var mf *ModelsFile
	if got := mf.AllKnownModels(); got != nil {
		t.Errorf("expected nil, got %v", got)
	}
}

func TestAllKnownModels_EmptyFile_ReturnsEmpty(t *testing.T) {
	mf := &ModelsFile{}
	got := mf.AllKnownModels()
	if len(got) != 0 {
		t.Errorf("expected empty, got %v", got)
	}
}

func TestAllKnownModels_CombinesTopLevelAndProviderAttached(t *testing.T) {
	mf := &ModelsFile{
		Providers: map[string]ProviderDefinition{
			"ollama": {
				API: APIOpenAI,
				Models: []ModelDefinition{
					{ID: "llama3"},
					{ID: "qwen2"},
				},
				ModelOverrides: map[string]ModelDefinition{
					"llama3": {ContextWindow: 16384},
				},
			},
		},
		Models: []ModelDefinition{
			{ID: "claude-opus-4-5", API: APIAnthropic},
		},
	}
	got := mf.AllKnownModels()

	// Expect 1 top-level + 2 provider Models + 1 provider ModelOverride = 4.
	if len(got) != 4 {
		t.Fatalf("len = %d, want 4 (%+v)", len(got), got)
	}

	// First entry is the top-level model (Provider="").
	if got[0].Provider != "" || got[0].Model.ID != "claude-opus-4-5" {
		t.Errorf("got[0] = %+v, want top-level claude-opus-4-5", got[0])
	}

	// Provider-attached entries inherit API from the provider.
	var llamaEntries int
	var sawQwen bool
	for _, km := range got {
		if km.Provider == "ollama" && km.Model.ID == "llama3" {
			llamaEntries++
			if km.Model.API != APIOpenAI {
				t.Errorf("llama3 API = %q, want openai (inherited)", km.Model.API)
			}
		}
		if km.Provider == "ollama" && km.Model.ID == "qwen2" {
			sawQwen = true
		}
	}
	if !sawQwen {
		t.Errorf("qwen2 missing from %v", got)
	}
	// llama3 appears twice: once from Models, once from ModelOverrides.
	if llamaEntries != 2 {
		t.Errorf("llama3 entries = %d, want 2 (Models + ModelOverrides)", llamaEntries)
	}
}

func writeTemp(t *testing.T, contents string) string {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "models-*.json")
	if err != nil {
		t.Fatalf("CreateTemp: %v", err)
	}
	defer f.Close()
	if _, err := f.WriteString(contents); err != nil {
		t.Fatalf("WriteString: %v", err)
	}
	return f.Name()
}
