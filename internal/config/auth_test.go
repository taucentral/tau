package config

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func TestInMemoryAuthStore_SetGetDelete(t *testing.T) {
	s := NewInMemoryAuthStore()
	if _, ok := s.Get("anthropic"); ok {
		t.Errorf("Get on empty store returned ok=true")
	}
	if err := s.Set("anthropic", "sk-abc"); err != nil {
		t.Fatalf("Set: %v", err)
	}
	if v, ok := s.Get("anthropic"); !ok || v != "sk-abc" {
		t.Errorf("Get = %q,%t; want \"sk-abc\",true", v, ok)
	}
	if err := s.Set("openai", "sk-xyz"); err != nil {
		t.Fatalf("Set: %v", err)
	}
	entries, err := s.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(entries) != 2 {
		t.Errorf("len(entries) = %d, want 2", len(entries))
	}
	if err := s.Delete("anthropic"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if _, ok := s.Get("anthropic"); ok {
		t.Errorf("Get after Delete returned ok=true")
	}
	if v, ok := s.Get("openai"); !ok || v != "sk-xyz" {
		t.Errorf("Get(\"openai\") after Delete = %q,%t; want \"sk-xyz\",true", v, ok)
	}
}

func TestInMemoryAuthStore_EmptyAPIKeyClears(t *testing.T) {
	s := NewInMemoryAuthStore()
	_ = s.Set("anthropic", "sk-abc")
	if err := s.Set("anthropic", ""); err != nil {
		t.Fatalf("Set(\"\", \"\"): %v", err)
	}
	if _, ok := s.Get("anthropic"); ok {
		t.Errorf("empty Set should clear the entry")
	}
}

func TestInMemoryAuthStore_Save_ReplacesAll(t *testing.T) {
	s := NewInMemoryAuthStore()
	_ = s.Set("anthropic", "sk-old1")
	_ = s.Set("openai", "sk-old2")
	if err := s.Save(map[string]ProviderAuth{
		"google":   {APIKey: "sk-google"},
		"deepseek": {APIKey: "sk-ds"},
	}); err != nil {
		t.Fatalf("Save: %v", err)
	}
	// Old entries are gone.
	if _, ok := s.Get("anthropic"); ok {
		t.Errorf("anthropic should have been replaced")
	}
	if _, ok := s.Get("openai"); ok {
		t.Errorf("openai should have been replaced")
	}
	// New entries are present.
	if v, ok := s.Get("google"); !ok || v != "sk-google" {
		t.Errorf("google = %q,%t; want sk-google,true", v, ok)
	}
	if v, ok := s.Get("deepseek"); !ok || v != "sk-ds" {
		t.Errorf("deepseek = %q,%t; want sk-ds,true", v, ok)
	}
	// Save with empty map clears everything.
	if err := s.Save(map[string]ProviderAuth{}); err != nil {
		t.Fatalf("Save empty: %v", err)
	}
	entries, err := s.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(entries) != 0 {
		t.Errorf("after empty Save, entries = %d, want 0", len(entries))
	}
}

func TestFileAuthStore_FreshWrite_Creates0600(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "agent", "auth.json")
	s := NewFileAuthStore(path, nil)
	if err := s.Set("anthropic", "sk-abc"); err != nil {
		t.Fatalf("Set: %v", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if info.Mode().Perm() != FileMode {
		t.Errorf("file mode = %o, want %o", info.Mode().Perm(), FileMode)
	}
}

func TestFileAuthStore_LooseModeTightenedAndDiagnosticEmitted(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "auth.json")
	if err := os.WriteFile(path, []byte(`{"openai":{"apiKey":"sk-old"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	count := 0
	s := NewFileAuthStore(path, func(msg string) {
		count++
	})
	// Trigger a load via Get.
	if _, ok := s.Get("openai"); !ok {
		t.Fatalf("Get should see the existing key")
	}
	if count != 1 {
		t.Errorf("diag count = %d, want 1", count)
	}
	// Verify mode tightened.
	info, _ := os.Stat(path)
	if info.Mode().Perm() != FileMode {
		t.Errorf("file mode = %o, want %o", info.Mode().Perm(), FileMode)
	}
	// Second load should not re-emit.
	_, _ = s.Get("openai")
	if count != 1 {
		t.Errorf("diag count = %d after second load, want 1", count)
	}
}

func TestFileAuthStore_PersistedAcrossInstances(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "agent", "auth.json")
	s1 := NewFileAuthStore(path, nil)
	if err := s1.Set("anthropic", "sk-abc"); err != nil {
		t.Fatalf("Set: %v", err)
	}
	s2 := NewFileAuthStore(path, nil)
	if v, ok := s2.Get("anthropic"); !ok || v != "sk-abc" {
		t.Errorf("Get after re-open = %q,%t; want \"sk-abc\",true", v, ok)
	}
}

func TestFileAuthStore_GetOnMissing_ReturnsFalse(t *testing.T) {
	tmp := t.TempDir()
	s := NewFileAuthStore(filepath.Join(tmp, "auth.json"), nil)
	if _, ok := s.Get("nobody"); ok {
		t.Errorf("Get on missing provider returned true")
	}
}

func TestFileAuthStore_DeleteRemovesOnlyNamed(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "auth.json")
	s := NewFileAuthStore(path, nil)
	_ = s.Set("anthropic", "sk-a")
	_ = s.Set("openai", "sk-b")
	if err := s.Delete("anthropic"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if _, ok := s.Get("anthropic"); ok {
		t.Errorf("anthropic should be gone")
	}
	if v, ok := s.Get("openai"); !ok || v != "sk-b" {
		t.Errorf("openai should still be present: %q,%t", v, ok)
	}
}

func TestFileAuthStore_ConcurrentSave_Serializes(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "agent", "auth.json")
	s := NewFileAuthStore(path, nil)
	const n = 25
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(j int) {
			defer wg.Done()
			_ = s.Set(providerName(j), "sk")
		}(i)
	}
	wg.Wait()
	got, _ := s.Load()
	if len(got) != n {
		t.Errorf("len = %d, want %d", len(got), n)
	}
}

func TestFileAuthStore_RejectsUnknownFields(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "auth.json")
	if err := os.WriteFile(path, []byte(`{"anthropic":{"apiKeey":"sk"}}`), FileMode); err != nil {
		t.Fatal(err)
	}
	s := NewFileAuthStore(path, nil)
	_, err := s.Load()
	if err == nil {
		t.Errorf("Load should reject unknown fields")
	}
}

func TestFileAuthStore_SaveReplacesAll(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "auth.json")
	s := NewFileAuthStore(path, nil)
	_ = s.Set("anthropic", "sk-a")
	_ = s.Set("openai", "sk-b")
	if err := s.Save(map[string]ProviderAuth{"google": {APIKey: "sk-c"}}); err != nil {
		t.Fatalf("Save: %v", err)
	}
	got, _ := s.Load()
	if len(got) != 1 {
		t.Errorf("len = %d, want 1", len(got))
	}
	if _, ok := got["google"]; !ok {
		t.Errorf("google should be present")
	}
	if _, ok := got["anthropic"]; ok {
		t.Errorf("anthropic should be gone after Save replaced all")
	}
}

func providerName(i int) string {
	const alphabet = "abcdefghijklmnopqrstuvwxyz"
	if i < len(alphabet) {
		return "p" + string(alphabet[i])
	}
	return "p" + string(rune('A'+i-len(alphabet)))
}
