// filestore_test.go — internal tests for the FileStore reference backend.
//
// Coverage table (each test maps to a scenario in
// openspec/changes/add-storage-seam/specs/cross-session-storage/spec.md):
//
//   - TestFileStorePutQueryRoundTrip            : (a) round-trip a single entry
//   - TestFileStoreKeywordQueryCaseInsensitive  : (b) keyword query matches
//   - TestFileStoreTagsQueryAND                 : (c) tags query (all queried)
//   - TestFileStoreSinceQueryFilters            : (d) since query by timestamp
//   - TestFileStoreLimitCapsResults             : (e) limit caps results
//   - TestFileStoreEmbeddingQueryUnsupported    : (f) embedding -> ErrUnsupportedQuery
//   - TestFileStoreMissingDirCreated0700        : (g) directory created 0700
//   - TestFileStoreFileMode0600                 : (h) file mode 0600
//   - TestFileStoreConcurrentPutNoRace          : (i) two goroutines Put without race
//   - TestFileStoreRejectsEmptyID               : typed error on empty ID
//   - TestFileStoreQueryAfterClose              : ErrStoreClosed after Close
//   - TestFileStoreOverwriteByID                : same-ID Put overwrites

package storage

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func newTestStore(t *testing.T) *FileStore {
	t.Helper()
	dir := t.TempDir()
	s, err := NewFileStore(dir)
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func TestFileStorePutQueryRoundTrip(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()
	when := time.Date(2026, 6, 28, 14, 32, 1, 0, time.UTC)
	in := Entry{
		ID:        "decisions/auth",
		Text:      "Use OAuth2 with PKCE for all CLI auth flows.",
		Tags:      []string{"decision", "auth"},
		Timestamp: when,
		Source:    "session-abc",
	}
	if err := s.Put(ctx, in); err != nil {
		t.Fatalf("Put: %v", err)
	}
	out, err := s.Query(ctx, Query{Limit: 10})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("Query returned %d entries, want 1", len(out))
	}
	got := out[0]
	if got.ID != in.ID {
		t.Errorf("ID = %q, want %q", got.ID, in.ID)
	}
	if got.Text != in.Text {
		t.Errorf("Text = %q, want %q", got.Text, in.Text)
	}
	if got.Source != in.Source {
		t.Errorf("Source = %q, want %q", got.Source, in.Source)
	}
	if !got.Timestamp.Equal(in.Timestamp) {
		t.Errorf("Timestamp = %v, want %v", got.Timestamp, in.Timestamp)
	}
	if len(got.Tags) != 2 || got.Tags[0] != "decision" || got.Tags[1] != "auth" {
		t.Errorf("Tags = %v, want [decision auth]", got.Tags)
	}
}

func TestFileStoreKeywordQueryCaseInsensitive(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()
	_ = s.Put(ctx, Entry{ID: "a", Text: "Use OAuth2 with PKCE", Timestamp: time.Now()})
	_ = s.Put(ctx, Entry{ID: "b", Text: "Backend uses postgres", Timestamp: time.Now()})

	out, err := s.Query(ctx, Query{KeywordQuery: "OAUTH"})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("case-insensitive substring: got %d, want 1", len(out))
	}
	if out[0].ID != "a" {
		t.Errorf("matched ID = %q, want %q", out[0].ID, "a")
	}
}

func TestFileStoreTagsQueryAND(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()
	_ = s.Put(ctx, Entry{ID: "a", Text: "a", Tags: []string{"decision", "auth"}, Timestamp: time.Now()})
	_ = s.Put(ctx, Entry{ID: "b", Text: "b", Tags: []string{"decision"}, Timestamp: time.Now()})
	_ = s.Put(ctx, Entry{ID: "c", Text: "c", Tags: []string{"auth"}, Timestamp: time.Now()})

	out, err := s.Query(ctx, Query{TagsQuery: []string{"decision", "auth"}})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("AND tags: got %d, want 1 (only entry with both tags)", len(out))
	}
	if out[0].ID != "a" {
		t.Errorf("matched ID = %q, want %q", out[0].ID, "a")
	}
}

func TestFileStoreSinceQueryFilters(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()
	t0 := time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC)
	t1 := time.Date(2026, 6, 28, 0, 0, 0, 0, time.UTC)
	_ = s.Put(ctx, Entry{ID: "old", Text: "old", Timestamp: t0})
	_ = s.Put(ctx, Entry{ID: "new", Text: "new", Timestamp: t1})

	out, err := s.Query(ctx, Query{SinceQuery: t1})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("SinceQuery: got %d, want 1", len(out))
	}
	if out[0].ID != "new" {
		t.Errorf("SinceQuery matched ID = %q, want %q", out[0].ID, "new")
	}
}

func TestFileStoreLimitCapsResults(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()
	for i := 0; i < 5; i++ {
		_ = s.Put(ctx, Entry{
			ID:        "e" + string(rune('a'+i)),
			Text:      "entry",
			Timestamp: time.Now(),
		})
		// Stagger mtimes so the mtime-descending order is stable.
		time.Sleep(2 * time.Millisecond)
	}
	out, err := s.Query(ctx, Query{Limit: 2})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(out) != 2 {
		t.Errorf("Limit: got %d, want 2", len(out))
	}
}

func TestFileStoreEmbeddingQueryUnsupported(t *testing.T) {
	s := newTestStore(t)
	_, err := s.Query(context.Background(), Query{EmbeddingQuery: []float32{0.1, 0.2}})
	if !errors.Is(err, ErrUnsupportedQuery) {
		t.Errorf("EmbeddingQuery err = %v, want errors.Is(ErrUnsupportedQuery)", err)
	}
}

func TestFileStoreMissingDirCreated0700(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "nested", "store")
	s, err := NewFileStore(dir)
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	info, err := os.Stat(dir)
	if err != nil {
		t.Fatalf("stat store dir: %v", err)
	}
	// Directory mode bits must be exactly 0700 (the umask-adjusted
	// equivalent; on most Linux systems MkdirAll yields 0700 verbatim
	// when umask allows it). We check the perm bits only.
	if got := info.Mode().Perm(); got != DirectoryMode {
		t.Errorf("store dir mode = %v, want %v", got, DirectoryMode)
	}
}

func TestFileStoreFileMode0600(t *testing.T) {
	dir := t.TempDir()
	s, err := NewFileStore(dir)
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	if err := s.Put(context.Background(), Entry{
		ID:        "x",
		Text:      "secret value",
		Timestamp: time.Now(),
	}); err != nil {
		t.Fatalf("Put: %v", err)
	}
	info, err := os.Stat(filepath.Join(dir, "x.md"))
	if err != nil {
		t.Fatalf("stat entry file: %v", err)
	}
	if got := info.Mode().Perm(); got != FileMode {
		t.Errorf("entry file mode = %v, want %v", got, FileMode)
	}
}

func TestFileStoreConcurrentPutNoRace(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()
	const N = 20
	var wg sync.WaitGroup
	wg.Add(N)
	for i := 0; i < N; i++ {
		go func(i int) {
			defer wg.Done()
			err := s.Put(ctx, Entry{
				ID:        "concurrent-" + string(rune('a'+i%26)) + string(rune('a'+(i/26)%26)),
				Text:      "concurrent",
				Timestamp: time.Now(),
			})
			if err != nil {
				t.Errorf("goroutine %d Put: %v", i, err)
			}
		}(i)
	}
	wg.Wait()
	// The race detector is the primary assertion here: if two Put
	// paths touched shared state unsafely, -race would report it. The
	// count assertion is a sanity check.
	out, err := s.Query(ctx, Query{Limit: N + 5})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(out) != N {
		t.Errorf("after %d concurrent Puts, Query returned %d, want %d", N, len(out), N)
	}
}

func TestFileStoreRejectsEmptyID(t *testing.T) {
	s := newTestStore(t)
	err := s.Put(context.Background(), Entry{ID: "", Text: "no id"})
	if err == nil {
		t.Fatal("Put with empty ID: got nil error, want one")
	}
	if !strings.Contains(err.Error(), "ID") {
		t.Errorf("Put err = %v, want message containing 'ID'", err)
	}
}

func TestFileStoreQueryAfterClose(t *testing.T) {
	dir := t.TempDir()
	s, err := NewFileStore(dir)
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	_ = s.Put(context.Background(), Entry{ID: "a", Text: "before close", Timestamp: time.Now()})
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	_, err = s.Query(context.Background(), Query{})
	if !errors.Is(err, ErrStoreClosed) {
		t.Errorf("Query after Close: err = %v, want errors.Is(ErrStoreClosed)", err)
	}
	err = s.Put(context.Background(), Entry{ID: "b", Text: "after close"})
	if !errors.Is(err, ErrStoreClosed) {
		t.Errorf("Put after Close: err = %v, want errors.Is(ErrStoreClosed)", err)
	}
}

func TestFileStoreOverwriteByID(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()
	_ = s.Put(ctx, Entry{ID: "k", Text: "v1", Timestamp: time.Now()})
	time.Sleep(2 * time.Millisecond) // distinct mtime
	_ = s.Put(ctx, Entry{ID: "k", Text: "v2", Timestamp: time.Now()})
	out, err := s.Query(ctx, Query{Limit: 10})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("overwrite: got %d entries, want 1 (same ID overwrites)", len(out))
	}
	if out[0].Text != "v2" {
		t.Errorf("overwrite text = %q, want %q", out[0].Text, "v2")
	}
}

// TestFileStorePathSeparatorInIDIsSanitised covers a defence-in-depth
// invariant: an embedder-supplied ID containing "/" or ".." cannot
// escape the store directory or create subdirectories. The sanitiser
// rewrites those characters to "_".
func TestFileStorePathSeparatorInIDIsSanitised(t *testing.T) {
	dir := t.TempDir()
	s, err := NewFileStore(dir)
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })

	for _, id := range []string{"a/b", "..escape", "with:colon"} {
		if err := s.Put(context.Background(), Entry{ID: id, Text: "x", Timestamp: time.Now()}); err != nil {
			t.Errorf("Put(%q): %v", id, err)
		}
	}
	// Ensure no file landed outside the store dir.
	_ = filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		// Every regular file must be directly under dir (one level).
		if d.IsDir() {
			return nil
		}
		rel, _ := filepath.Rel(dir, path)
		if strings.Contains(rel, string(os.PathSeparator)) && rel != filepath.Join(".store.lock") {
			// .store.lock is the flock file at the dir root; that's
			// allowed. Any other subpath means sanitisation failed.
			t.Errorf("entry file escaped store dir: %q", rel)
		}
		return nil
	})
}

