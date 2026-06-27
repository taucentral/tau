package config

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// TestPhase2IntegrationSmokeTest is the end-to-end Phase 2 verification:
// TAU_CONFIG_DIR → ConfigDir → FileSettingsStorage (global + project) →
// FileAuthStore → FileTrustStore → ResolveValue (env + literal).
//
// All assertions are inside this single test; no separate binary.
func TestPhase2IntegrationSmokeTest(t *testing.T) {
	tmp := t.TempDir()
	// Override TAU_CONFIG_DIR so ConfigDir() resolves under tmp.
	t.Setenv("TAU_CONFIG_DIR", tmp)
	t.Setenv("TAU_AGENT_DIR", "")
	// And set an env var ResolveValue can resolve.
	t.Setenv("TAU_INTEGRATION_KEY", "sk-integration-12345")

	// 1. ConfigDir resolves to the override.
	gotConfigDir, err := ConfigDir()
	if err != nil {
		t.Fatalf("ConfigDir: %v", err)
	}
	if gotConfigDir != tmp {
		t.Errorf("ConfigDir = %q, want %q", gotConfigDir, tmp)
	}

	// 2. AgentDir under tmp/agent.
	agentDir, err := AgentDir()
	if err != nil {
		t.Fatalf("AgentDir: %v", err)
	}
	if want := filepath.Join(tmp, "agent"); agentDir != want {
		t.Errorf("AgentDir = %q, want %q", agentDir, want)
	}
	if err := MkdirAll(agentDir); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}

	// 3. FileSettingsStorage global write + reload.
	cwd := filepath.Join(tmp, "project")
	if err := MkdirAll(cwd); err != nil {
		t.Fatal(err)
	}
	st, err := NewFileSettingsStorage(agentDir, cwd, true)
	if err != nil {
		t.Fatalf("NewFileSettingsStorage: %v", err)
	}
	model := "test-model"
	if err := st.Save(context.Background(), ScopeGlobal, func(c Settings) Settings {
		c.DefaultModel = &model
		return c
	}); err != nil {
		t.Fatalf("Save global: %v", err)
	}

	// 4. Project overrides ReserveTokens only; global DefaultModel must survive.
	reserve := 4096
	if err := st.Save(context.Background(), ScopeProject, func(c Settings) Settings {
		c.Compaction = &CompactionSettings{ReserveTokens: &reserve}
		return c
	}); err != nil {
		t.Fatalf("Save project: %v", err)
	}

	// Re-open the storage to verify the merge is computed from disk state.
	st2, err := NewFileSettingsStorage(agentDir, cwd, true)
	if err != nil {
		t.Fatalf("Re-open storage: %v", err)
	}
	got, err := st2.Load(context.Background())
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got.DefaultModel == nil || *got.DefaultModel != model {
		t.Errorf("DefaultModel = %v, want %q", got.DefaultModel, model)
	}
	if got.Compaction == nil || got.Compaction.ReserveTokens == nil || *got.Compaction.ReserveTokens != reserve {
		t.Errorf("Compaction.ReserveTokens lost across reload")
	}

	// 5. Auth store write + reload + 0600 mode.
	authPath := filepath.Join(agentDir, "auth.json")
	auth := NewFileAuthStore(authPath, nil)
	if err := auth.Set("anthropic", "sk-secret"); err != nil {
		t.Fatalf("Auth Set: %v", err)
	}
	info, err := os.Stat(authPath)
	if err != nil {
		t.Fatalf("stat auth: %v", err)
	}
	if info.Mode().Perm() != FileMode {
		t.Errorf("auth file mode = %o, want %o", info.Mode().Perm(), FileMode)
	}
	auth2 := NewFileAuthStore(authPath, nil)
	if v, ok := auth2.Get("anthropic"); !ok || v != "sk-secret" {
		t.Errorf("Auth reload: Get = %q,%t", v, ok)
	}

	// 6. Trust store round trip.
	trustPath := filepath.Join(agentDir, "trust.json")
	trust := NewFileTrustStore(trustPath)
	if err := trust.Load(); err != nil {
		t.Fatalf("Trust Load: %v", err)
	}
	if err := trust.SetTrust("/tmp/example", TrustTrusted); err != nil {
		t.Fatalf("SetTrust: %v", err)
	}
	if err := trust.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}
	trust2 := NewFileTrustStore(trustPath)
	if err := trust2.Load(); err != nil {
		t.Fatalf("Reload trust: %v", err)
	}
	if ok, _ := trust2.IsTrusted("/tmp/example"); !ok {
		t.Errorf("IsTrusted after reload = false, want true")
	}

	// 7. ResolveValue: env and literal paths.
	gotKey, err := ResolveValue("$TAU_INTEGRATION_KEY")
	if err != nil {
		t.Fatalf("ResolveValue env: %v", err)
	}
	if gotKey != "sk-integration-12345" {
		t.Errorf("ResolveValue env = %q", gotKey)
	}
	gotLit, err := ResolveValue("just-a-literal")
	if err != nil {
		t.Fatalf("ResolveValue literal: %v", err)
	}
	if gotLit != "just-a-literal" {
		t.Errorf("ResolveValue literal = %q", gotLit)
	}
}
