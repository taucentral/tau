package config

import (
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func TestFileTrustStore_EmptyReturnsFalse(t *testing.T) {
	s := NewFileTrustStore(filepath.Join(t.TempDir(), "trust.json"))
	if err := s.Load(); err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := s.IsTrusted("/any/cwd")
	if err != nil {
		t.Fatalf("IsTrusted: %v", err)
	}
	if got {
		t.Errorf("IsTrusted on empty store = true, want false")
	}
}

func TestFileTrustStore_EmptyCwd_ReturnsFalse(t *testing.T) {
	s := NewFileTrustStore(filepath.Join(t.TempDir(), "trust.json"))
	_ = s.Load()
	got, _ := s.IsTrusted("")
	if got {
		t.Errorf("IsTrusted(\"\") = true, want false")
	}
}

func TestFileTrustStore_SetThenIsTrusted(t *testing.T) {
	s := NewFileTrustStore(filepath.Join(t.TempDir(), "trust.json"))
	_ = s.Load()
	if err := s.SetTrust("/home/x/proj", TrustTrusted); err != nil {
		t.Fatalf("SetTrust: %v", err)
	}
	got, _ := s.IsTrusted("/home/x/proj")
	if !got {
		t.Errorf("IsTrusted after SetTrust(trusted) = false, want true")
	}
}

func TestFileTrustStore_SetUntrusted_ReturnsFalse(t *testing.T) {
	s := NewFileTrustStore(filepath.Join(t.TempDir(), "trust.json"))
	_ = s.Load()
	_ = s.SetTrust("/foo", TrustUntrusted)
	got, _ := s.IsTrusted("/foo")
	if got {
		t.Errorf("IsTrusted after SetTrust(untrusted) = true, want false")
	}
}

func TestFileTrustStore_PersistRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "subdir", "trust.json")
	s1 := NewFileTrustStore(path)
	if err := s1.Load(); err != nil {
		t.Fatalf("Load: %v", err)
	}
	if err := s1.SetTrust("/tmp/proj", TrustTrusted); err != nil {
		t.Fatalf("SetTrust: %v", err)
	}
	if err := s1.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Re-open from disk.
	s2 := NewFileTrustStore(path)
	if err := s2.Load(); err != nil {
		t.Fatalf("Re-load: %v", err)
	}
	got, _ := s2.IsTrusted("/tmp/proj")
	if !got {
		t.Errorf("IsTrusted after reload = false, want true")
	}

	// File mode should be 0600.
	info, _ := os.Stat(path)
	if info.Mode().Perm() != FileMode {
		t.Errorf("file mode = %o, want %o", info.Mode().Perm(), FileMode)
	}
}

func TestFileTrustStore_OverwriteExistingEntry(t *testing.T) {
	s := NewFileTrustStore(filepath.Join(t.TempDir(), "trust.json"))
	_ = s.Load()
	_ = s.SetTrust("/foo", TrustTrusted)
	_ = s.SetTrust("/foo", TrustUntrusted)
	got, _ := s.IsTrusted("/foo")
	if got {
		t.Errorf("IsTrusted after overwrite = true, want false")
	}
}

func TestFileTrustStore_InvalidStatus_Rejected(t *testing.T) {
	s := NewFileTrustStore(filepath.Join(t.TempDir(), "trust.json"))
	_ = s.Load()
	if err := s.SetTrust("/foo", TrustStatus("bogus")); !errors.Is(err, ErrSchemaViolation) {
		t.Errorf("err = %v, want ErrSchemaViolation", err)
	}
}

func TestFileTrustStore_MalformedFile_Rejected(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trust.json")
	if err := os.WriteFile(path, []byte("{not json"), FileMode); err != nil {
		t.Fatal(err)
	}
	s := NewFileTrustStore(path)
	if err := s.Load(); err == nil {
		t.Errorf("Load should reject malformed file")
	}
}

func TestFileTrustStore_LooseModeTightenedOnLoad(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trust.json")
	if err := os.WriteFile(path, []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
	s := NewFileTrustStore(path)
	if err := s.Load(); err != nil {
		t.Fatalf("Load: %v", err)
	}
	info, _ := os.Stat(path)
	if info.Mode().Perm() != FileMode {
		t.Errorf("mode = %o, want %o (Load should tighten)", info.Mode().Perm(), FileMode)
	}
}

func TestFileTrustStore_InvalidStatusValueInFile_Rejected(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trust.json")
	if err := os.WriteFile(path, []byte(`{"/foo":"bogus"}`), FileMode); err != nil {
		t.Fatal(err)
	}
	s := NewFileTrustStore(path)
	if err := s.Load(); !errors.Is(err, ErrSchemaViolation) {
		t.Errorf("err = %v, want ErrSchemaViolation", err)
	}
}

func TestFileTrustStore_NoSave_NoDirtyWrite(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trust.json")
	s := NewFileTrustStore(path)
	_ = s.Load()
	_ = s.SetTrust("/foo", TrustTrusted)
	// Don't call Save — file should not exist.
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("file should not exist before Save")
	}
}

func TestFileTrustStore_ConcurrentSave_Serializes(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trust.json")
	s := NewFileTrustStore(path)
	_ = s.Load()
	const n = 25
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(j int) {
			defer wg.Done()
			_ = s.SetTrust(filepath.Join("/x", string(rune('a'+j%26))), TrustTrusted)
			_ = s.Save()
		}(i)
	}
	wg.Wait()
	// All 25 entries should be present (or however many distinct paths).
	// The 26-rune cycling produces 26 distinct names; we used 25 of them.
	if got := len(s.m); got != 25 {
		t.Errorf("len(s.m) = %d, want 25", got)
	}
}
