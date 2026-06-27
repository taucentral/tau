package state

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// newTestStore opens a store backed by a fresh temp file. The cleanup is
// registered with t.Cleanup so callers don't have to remember to close.
func newTestStore(t *testing.T) *Store {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "test.bolt")
	s, err := OpenStore(path)
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func TestOpenStore_CreatesFileWithBuckets(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.bolt")
	s, err := OpenStore(path)
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	defer s.Close()
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("bolt file not created: %v", err)
	}
	// Buckets exist: writing to entries should not error.
	if err := s.Append(Entry{ID: "abc12345", Kind: KindLabel, Payload: LabelPayload{Label: "x"}}); err != nil {
		t.Errorf("Append after OpenStore: %v", err)
	}
}

func TestOpenStore_FileModeIs0600(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.bolt")
	s, err := OpenStore(path)
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	defer s.Close()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	// Mask off type bits.
	mode := info.Mode().Perm()
	if mode != 0600 {
		t.Errorf("file mode = %o, want 0600", mode)
	}
}

func TestStore_AppendAndGet(t *testing.T) {
	s := newTestStore(t)
	original := Entry{
		ID:       "abc12345",
		ParentID: "parent00",
		Kind:     KindLabel,
		Payload:  LabelPayload{Label: "v1.0"},
	}
	if err := s.Append(original); err != nil {
		t.Fatalf("Append: %v", err)
	}
	got, err := s.Get("abc12345")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got.ID != original.ID {
		t.Errorf("ID = %q, want %q", got.ID, original.ID)
	}
	if got.Kind != KindLabel {
		t.Errorf("Kind = %q, want %q", got.Kind, KindLabel)
	}
	lp, ok := got.Payload.(LabelPayload)
	if !ok {
		t.Fatalf("Payload = %T, want LabelPayload", got.Payload)
	}
	if lp.Label != "v1.0" {
		t.Errorf("Label = %q, want %q", lp.Label, "v1.0")
	}
}

func TestStore_Append_RejectsEmptyID(t *testing.T) {
	s := newTestStore(t)
	err := s.Append(Entry{Kind: KindLabel, Payload: LabelPayload{}})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "empty ID") {
		t.Errorf("err = %v, want 'empty ID'", err)
	}
}

func TestStore_Append_RejectsDuplicate(t *testing.T) {
	s := newTestStore(t)
	e := Entry{ID: "dup00001", Kind: KindLabel, Payload: LabelPayload{}}
	if err := s.Append(e); err != nil {
		t.Fatalf("first Append: %v", err)
	}
	err := s.Append(e)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "already exists") {
		t.Errorf("err = %v, want 'already exists'", err)
	}
}

func TestStore_AppendBatch_Atomic(t *testing.T) {
	s := newTestStore(t)
	entries := []Entry{
		{ID: "b1", ParentID: "b0", Kind: KindLabel, Payload: LabelPayload{}},
		{ID: "b0", ParentID: "", Kind: KindSessionHeader, Payload: SessionHeaderPayload{SessionID: "s"}},
		{ID: "b2", ParentID: "b1", Kind: KindLabel, Payload: LabelPayload{}},
	}
	if err := s.AppendBatch(entries); err != nil {
		t.Fatalf("AppendBatch: %v", err)
	}
	for _, id := range []string{"b0", "b1", "b2"} {
		if _, err := s.Get(id); err != nil {
			t.Errorf("Get(%q): %v", id, err)
		}
	}
}

func TestStore_AppendBatch_RollsBackOnError(t *testing.T) {
	s := newTestStore(t)
	// Pre-add the duplicate so the second entry in the batch fails.
	if err := s.Append(Entry{ID: "dup", Kind: KindLabel, Payload: LabelPayload{}}); err != nil {
		t.Fatalf("pre-Append: %v", err)
	}
	// Batch: a new entry (would succeed alone) followed by the duplicate.
	// The whole batch must roll back, leaving the new entry NOT in the store.
	batch := []Entry{
		{ID: "new", Kind: KindLabel, Payload: LabelPayload{}},
		{ID: "dup", Kind: KindLabel, Payload: LabelPayload{}},
	}
	err := s.AppendBatch(batch)
	if err == nil {
		t.Fatal("expected batch to fail, got nil")
	}
	// "new" must NOT be present — atomicity guarantee.
	if _, err := s.Get("new"); !errors.Is(err, ErrEntryNotFound) {
		t.Errorf("err = %v, want ErrEntryNotFound (atomic rollback failed)", err)
	}
}

func TestStore_Get_MissingReturnsErrEntryNotFound(t *testing.T) {
	s := newTestStore(t)
	_, err := s.Get("nope")
	if !errors.Is(err, ErrEntryNotFound) {
		t.Errorf("err = %v, want ErrEntryNotFound", err)
	}
}

func TestStore_All_ReturnsEveryEntry(t *testing.T) {
	s := newTestStore(t)
	for _, id := range []string{"a", "b", "c"} {
		if err := s.Append(Entry{ID: id, Kind: KindLabel, Payload: LabelPayload{}}); err != nil {
			t.Fatalf("Append(%q): %v", id, err)
		}
	}
	got, err := s.All()
	if err != nil {
		t.Fatalf("All: %v", err)
	}
	if len(got) != 3 {
		t.Errorf("len = %d, want 3", len(got))
	}
	seen := map[string]bool{}
	for _, e := range got {
		seen[e.ID] = true
	}
	for _, want := range []string{"a", "b", "c"} {
		if !seen[want] {
			t.Errorf("missing %q in All() result: %v", want, seen)
		}
	}
}

func TestStore_Exists(t *testing.T) {
	s := newTestStore(t)
	if err := s.Append(Entry{ID: "x", Kind: KindLabel, Payload: LabelPayload{}}); err != nil {
		t.Fatalf("Append: %v", err)
	}
	if yes, err := s.Exists("x"); err != nil || !yes {
		t.Errorf("Exists(x) = %v, %v, want true nil", yes, err)
	}
	if yes, err := s.Exists("missing"); err != nil || yes {
		t.Errorf("Exists(missing) = %v, %v, want false nil", yes, err)
	}
}

func TestStore_LeafRoundTrip(t *testing.T) {
	s := newTestStore(t)
	// Empty before set.
	got, err := s.LeafID()
	if err != nil {
		t.Fatalf("LeafID before set: %v", err)
	}
	if got != "" {
		t.Errorf("LeafID = %q, want empty", got)
	}
	if err := s.SetLeaf("abc12345"); err != nil {
		t.Fatalf("SetLeaf: %v", err)
	}
	got, err = s.LeafID()
	if err != nil {
		t.Fatalf("LeafID: %v", err)
	}
	if got != "abc12345" {
		t.Errorf("LeafID = %q, want abc12345", got)
	}
	// Overwrite works.
	if err := s.SetLeaf("def67890"); err != nil {
		t.Fatalf("SetLeaf overwrite: %v", err)
	}
	got, _ = s.LeafID()
	if got != "def67890" {
		t.Errorf("LeafID = %q, want def67890", got)
	}
}

func TestStore_InfoRoundTrip(t *testing.T) {
	s := newTestStore(t)
	info, err := s.Info()
	if err != nil {
		t.Fatalf("Info before set: %v", err)
	}
	if info != nil {
		t.Errorf("Info = %v, want nil", info)
	}
	want := map[string]string{"exit": "done", "turns": "42"}
	if err := s.SetInfo(want); err != nil {
		t.Fatalf("SetInfo: %v", err)
	}
	got, err := s.Info()
	if err != nil {
		t.Fatalf("Info: %v", err)
	}
	if got["exit"] != "done" || got["turns"] != "42" || len(got) != 2 {
		t.Errorf("Info = %v, want %v", got, want)
	}
}

func TestStore_LabelsRoundTrip(t *testing.T) {
	s := newTestStore(t)
	if err := s.SetLabels([]string{"after-merge", "verified"}); err != nil {
		t.Fatalf("SetLabels: %v", err)
	}
	got, err := s.Labels()
	if err != nil {
		t.Fatalf("Labels: %v", err)
	}
	if len(got) != 2 || got[0] != "after-merge" || got[1] != "verified" {
		t.Errorf("Labels = %v, want [after-merge verified]", got)
	}
}

func TestStore_Path(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "p.bolt")
	s, err := OpenStore(path)
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	defer s.Close()
	if got := s.Path(); got != path {
		t.Errorf("Path = %q, want %q", got, path)
	}
}

func TestStore_Close_IdempotentAndEnforcesClosed(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "c.bolt")
	s, err := OpenStore(path)
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
	if err := s.Append(Entry{ID: "x", Kind: KindLabel, Payload: LabelPayload{}}); !errors.Is(err, ErrStoreClosed) {
		t.Errorf("Append after Close: err = %v, want ErrStoreClosed", err)
	}
	if _, err := s.Get("x"); !errors.Is(err, ErrStoreClosed) {
		t.Errorf("Get after Close: err = %v, want ErrStoreClosed", err)
	}
	if _, err := s.All(); !errors.Is(err, ErrStoreClosed) {
		t.Errorf("All after Close: err = %v, want ErrStoreClosed", err)
	}
	if _, err := s.LeafID(); !errors.Is(err, ErrStoreClosed) {
		t.Errorf("LeafID after Close: err = %v, want ErrStoreClosed", err)
	}
}

func TestStore_ReopenPreservesEntries(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "r.bolt")
	s, err := OpenStore(path)
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	if err := s.Append(Entry{ID: "persist", Kind: KindLabel, Payload: LabelPayload{Label: "v1"}}); err != nil {
		t.Fatalf("Append: %v", err)
	}
	if err := s.SetLeaf("persist"); err != nil {
		t.Fatalf("SetLeaf: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	// Reopen and verify.
	s2, err := OpenStore(path)
	if err != nil {
		t.Fatalf("reOpen: %v", err)
	}
	defer s2.Close()
	got, err := s2.Get("persist")
	if err != nil {
		t.Fatalf("Get after reopen: %v", err)
	}
	lp, _ := got.Payload.(LabelPayload)
	if lp.Label != "v1" {
		t.Errorf("Label after reopen = %q, want v1", lp.Label)
	}
	leaf, _ := s2.LeafID()
	if leaf != "persist" {
		t.Errorf("LeafID after reopen = %q, want persist", leaf)
	}
}
