package config

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func TestMergeSettings_ProjectWins(t *testing.T) {
	gModel := "gpt-4o"
	pModel := "claude-opus-4-5"
	g := Settings{DefaultModel: &gModel}
	p := Settings{DefaultModel: &pModel}
	eff := mergeSettings(g, p)
	if eff.DefaultModel == nil || *eff.DefaultModel != pModel {
		t.Errorf("DefaultModel = %v, want %q", eff.DefaultModel, pModel)
	}
}

func TestMergeSettings_DeepNestedStruct(t *testing.T) {
	// Global has ReserveTokens=8192; project only sets Enabled=false.
	// After merge: ReserveTokens should survive from global.
	reserve := 8192
	enabled := true
	g := Settings{Compaction: &CompactionSettings{ReserveTokens: &reserve, Enabled: &enabled}}
	disabled := false
	p := Settings{Compaction: &CompactionSettings{Enabled: &disabled}}
	eff := mergeSettings(g, p)
	if eff.Compaction == nil {
		t.Fatal("Compaction nil after merge")
	}
	if eff.Compaction.Enabled == nil || *eff.Compaction.Enabled {
		t.Errorf("Enabled = %v, want false", eff.Compaction.Enabled)
	}
	if eff.Compaction.ReserveTokens == nil || *eff.Compaction.ReserveTokens != reserve {
		t.Errorf("ReserveTokens = %v, want %d (should survive from global)", eff.Compaction.ReserveTokens, reserve)
	}
}

func TestMergeSettings_SliceReplaceNotAppend(t *testing.T) {
	g := Settings{NpmCommand: []string{"npm"}}
	p := Settings{NpmCommand: []string{"mise", "exec", "node@20", "--", "npm"}}
	eff := mergeSettings(g, p)
	if got, want := len(eff.NpmCommand), 5; got != want {
		t.Errorf("len(NpmCommand) = %d, want %d (replace, not append)", got, want)
	}
	if eff.NpmCommand[0] != "mise" {
		t.Errorf("NpmCommand[0] = %q, want mise", eff.NpmCommand[0])
	}
}

func TestMergeSettings_NilOverridePreservesBase(t *testing.T) {
	model := "gpt-4o"
	g := Settings{DefaultModel: &model}
	p := Settings{} // nil DefaultModel
	eff := mergeSettings(g, p)
	if eff.DefaultModel == nil || *eff.DefaultModel != model {
		t.Errorf("DefaultModel = %v, want %q (nil override should preserve base)", eff.DefaultModel, model)
	}
}

func TestMergeSettings_NilBaseTakesOverride(t *testing.T) {
	g := Settings{}
	model := "claude"
	p := Settings{DefaultModel: &model}
	eff := mergeSettings(g, p)
	if eff.DefaultModel == nil || *eff.DefaultModel != model {
		t.Errorf("DefaultModel = %v, want %q", eff.DefaultModel, model)
	}
}

func TestMergeSettings_DeepCopy_NoAlias(t *testing.T) {
	reserve := 8192
	g := Settings{Compaction: &CompactionSettings{ReserveTokens: &reserve}}
	p := Settings{}
	eff := mergeSettings(g, p)
	if eff.Compaction == nil || eff.Compaction.ReserveTokens == nil {
		t.Fatal("missing field after merge")
	}
	*eff.Compaction.ReserveTokens = 1
	if reserve != 8192 {
		t.Errorf("merge aliased base: base.ReserveTokens mutated to %d", reserve)
	}
}

func TestInMemory_LoadMissing_ReturnsDefaults(t *testing.T) {
	s := NewInMemorySettingsStorage(true)
	got, err := s.Load(context.Background())
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got.Compaction == nil || got.Compaction.Enabled == nil || !*got.Compaction.Enabled {
		t.Errorf("Load on empty store should return DefaultSettings, got %+v", got)
	}
}

func TestInMemory_SaveGlobal_LoadReflects(t *testing.T) {
	s := NewInMemorySettingsStorage(true)
	model := "claude"
	err := s.Save(context.Background(), ScopeGlobal, func(c Settings) Settings {
		c.DefaultModel = &model
		return c
	})
	if err != nil {
		t.Fatalf("Save: %v", err)
	}
	got, err := s.Load(context.Background())
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got.DefaultModel == nil || *got.DefaultModel != model {
		t.Errorf("DefaultModel = %v, want %q", got.DefaultModel, model)
	}
}

func TestInMemory_SaveProject_OnUntrusted_Errors(t *testing.T) {
	s := NewInMemorySettingsStorage(false)
	err := s.Save(context.Background(), ScopeProject, func(c Settings) Settings {
		return c
	})
	if !errors.Is(err, ErrUntrustedProject) {
		t.Errorf("err = %v, want ErrUntrustedProject", err)
	}
}

func TestInMemory_Load_OnUntrusted_SkipsProject(t *testing.T) {
	s := NewInMemorySettingsStorage(false)
	model := "claude"
	// Even if project somehow has data, untrusted load must not see it.
	s.project.DefaultModel = &model
	got, err := s.Load(context.Background())
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got.DefaultModel != nil && *got.DefaultModel == model {
		t.Errorf("untrusted Load leaked project DefaultModel = %q", model)
	}
}

func TestInMemory_ConcurrentSave_Serializes(t *testing.T) {
	s := NewInMemorySettingsStorage(true)
	const n = 50
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			_ = s.Save(context.Background(), ScopeGlobal, func(c Settings) Settings {
				if c.NpmCommand == nil {
					c.NpmCommand = []string{}
				}
				c.NpmCommand = append(c.NpmCommand, "x")
				return c
			})
		}()
	}
	wg.Wait()
	got, _ := s.Load(context.Background())
	if len(got.NpmCommand) != n {
		t.Errorf("len(NpmCommand) = %d, want %d (serialization lost writes)", len(got.NpmCommand), n)
	}
}

func TestFileSettingsStorage_RoundTrip(t *testing.T) {
	tmp := t.TempDir()
	agentDir := filepath.Join(tmp, "agent")
	cwd := filepath.Join(tmp, "cwd")
	if err := MkdirAll(agentDir); err != nil {
		t.Fatal(err)
	}
	if err := MkdirAll(cwd); err != nil {
		t.Fatal(err)
	}
	st, err := NewFileSettingsStorage(agentDir, cwd, true)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	model := "claude-opus-4-5"
	if err := st.Save(context.Background(), ScopeGlobal, func(c Settings) Settings {
		c.DefaultModel = &model
		return c
	}); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Re-open to verify persistence.
	st2, err := NewFileSettingsStorage(agentDir, cwd, true)
	if err != nil {
		t.Fatalf("Re-New: %v", err)
	}
	got, err := st2.Load(context.Background())
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got.DefaultModel == nil || *got.DefaultModel != model {
		t.Errorf("DefaultModel = %v, want %q", got.DefaultModel, model)
	}
}

func TestFileSettingsStorage_ProjectMergeKeepsGlobal(t *testing.T) {
	tmp := t.TempDir()
	agentDir := filepath.Join(tmp, "agent")
	cwd := filepath.Join(tmp, "cwd")
	MkdirAll(agentDir)
	MkdirAll(cwd)
	st, _ := NewFileSettingsStorage(agentDir, cwd, true)

	gModel := "gpt-4o"
	if err := st.Save(context.Background(), ScopeGlobal, func(c Settings) Settings {
		c.DefaultModel = &gModel
		return c
	}); err != nil {
		t.Fatal(err)
	}
	reserve := 4096
	if err := st.Save(context.Background(), ScopeProject, func(c Settings) Settings {
		c.Compaction = &CompactionSettings{ReserveTokens: &reserve}
		return c
	}); err != nil {
		t.Fatal(err)
	}

	got, _ := st.Load(context.Background())
	if got.DefaultModel == nil || *got.DefaultModel != gModel {
		t.Errorf("DefaultModel = %v, want %q (global must survive project merge)", got.DefaultModel, gModel)
	}
	if got.Compaction == nil || got.Compaction.ReserveTokens == nil || *got.Compaction.ReserveTokens != reserve {
		t.Errorf("Compaction.ReserveTokens = %v, want %d", got.Compaction, reserve)
	}
}

func TestFileSettingsStorage_UntrustedSave_Rejected(t *testing.T) {
	tmp := t.TempDir()
	st, _ := NewFileSettingsStorage(filepath.Join(tmp, "agent"), filepath.Join(tmp, "cwd"), false)
	err := st.Save(context.Background(), ScopeProject, func(c Settings) Settings { return c })
	if !errors.Is(err, ErrUntrustedProject) {
		t.Errorf("err = %v, want ErrUntrustedProject", err)
	}
	// And no file should have been created.
	if _, err := os.Stat(filepath.Join(tmp, "cwd", ".tau", "settings.json")); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("project file should not exist after rejected save")
	}
}

func TestFileSettingsStorage_AtomicWrite_TempFileCleanedUp(t *testing.T) {
	tmp := t.TempDir()
	agentDir := filepath.Join(tmp, "agent")
	cwd := filepath.Join(tmp, "cwd")
	MkdirAll(agentDir)
	MkdirAll(cwd)
	st, _ := NewFileSettingsStorage(agentDir, cwd, true)
	model := "x"
	if err := st.Save(context.Background(), ScopeGlobal, func(c Settings) Settings {
		c.DefaultModel = &model
		return c
	}); err != nil {
		t.Fatal(err)
	}
	// The temp file should not exist after a successful save.
	if _, err := os.Stat(filepath.Join(agentDir, "settings.json.tmp")); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("temp file should not exist after successful save")
	}
}

func TestFileSettingsStorage_RejectsMalformedJSON(t *testing.T) {
	tmp := t.TempDir()
	agentDir := filepath.Join(tmp, "agent")
	cwd := filepath.Join(tmp, "cwd")
	MkdirAll(agentDir)
	MkdirAll(cwd)
	// Write garbage directly to the global file.
	if err := os.WriteFile(filepath.Join(agentDir, "settings.json"), []byte("{not valid"), FileMode); err != nil {
		t.Fatal(err)
	}
	st, _ := NewFileSettingsStorage(agentDir, cwd, true)
	if _, err := st.Load(context.Background()); err == nil {
		t.Errorf("Load should reject malformed JSON")
	}
}

func TestFileSettingsStorage_Close_IsNoOpAndAllowsReuse(t *testing.T) {
	tmp := t.TempDir()
	agentDir := filepath.Join(tmp, "agent")
	cwd := filepath.Join(tmp, "cwd")
	MkdirAll(agentDir)
	MkdirAll(cwd)
	st, _ := NewFileSettingsStorage(agentDir, cwd, true)
	// Close must not error and must not prevent subsequent operations
	// (file locks are per-operation, not held across calls).
	if err := st.Close(); err != nil {
		t.Errorf("Close returned error: %v", err)
	}
	m := "post-close-model"
	if err := st.Save(context.Background(), ScopeGlobal, func(c Settings) Settings {
		c.DefaultModel = &m
		return c
	}); err != nil {
		t.Fatalf("Save after Close: %v", err)
	}
	got, err := st.Load(context.Background())
	if err != nil {
		t.Fatalf("Load after Close: %v", err)
	}
	if got.DefaultModel == nil || *got.DefaultModel != m {
		t.Errorf("DefaultModel = %v, want %q", got.DefaultModel, m)
	}
}

func TestInMemorySettingsStorage_Close_IsNoOp(t *testing.T) {
	st := NewInMemorySettingsStorage(true)
	if err := st.Close(); err != nil {
		t.Errorf("Close returned error: %v", err)
	}
	// The in-memory store does not hold resources either; Close is a no-op
	// and does not invalidate the store.
	if _, err := st.Load(context.Background()); err != nil {
		t.Errorf("Load after Close: %v", err)
	}
}
