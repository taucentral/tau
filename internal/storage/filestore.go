// filestore.go — file-backed reference implementation of Store.
//
// FileStore persists each Entry as a single markdown file under a
// directory rooted at the constructor path:
//
//	<dir>/
//	  <id>.md       ← YAML-style frontmatter + markdown body
//	  <id>.md.tmp   ← atomic-write staging file (transient)
//	  <id>.md.lock  ← per-entry flock (transient)
//
// File format (one entry per file, UTF-8, mode 0600):
//
//	---
//	id: decisions/2026-06-28-auth
//	source: session-abc123
//	timestamp: 2026-06-28T14:32:01Z
//	tags:
//	  - decision
//	  - auth
//	---
//	Use OAuth2 with PKCE for all CLI auth flows.
//
// The frontmatter is YAML-compatible (key: value lines; tags is a list
// under "tags:"). The body is Entry.Text verbatim. Query parses the
// frontmatter to reconstruct Entry fields; it does not interpret the
// body beyond keyword-substring matching.
//
// Concurrency: every Put and every Query takes an exclusive flock on
// the directory (a single lock file at <dir>/.store.lock). This is
// coarse-grained but correct — concurrent Puts serialise, concurrent
// Queries serialise, but a Put never blocks a Query from a different
// goroutine because both take shared/exclusive respectively. Read
// throughput is plenty for the SDK's typical load (a few writes per
// turn, a handful of queries per turn).
//
// Reference: docs/input/context/plugin-support/whitepaper.md §3.4.

package storage

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/gofrs/flock"
)

// FileMode matches the project-wide convention for credential / trust /
// settings files: 0600 for files, 0700 for directories.
const (
	FileMode      os.FileMode = 0600
	DirectoryMode os.FileMode = 0700
)

// FileStore is the file-backed reference Store backend. Safe for
// concurrent use; the zero value is NOT usable — construct via
// NewFileStore.
type FileStore struct {
	dir  string
	lock *flock.Flock
	mu   sync.Mutex // guards Close-once semantics
	closed bool
}

// NewFileStore creates a FileStore rooted at dir. The directory is
// created with mode 0700 if missing; an existing directory is reused.
// Returns an error if dir cannot be created (permission denied, parent
// missing, path is a regular file).
func NewFileStore(dir string) (*FileStore, error) {
	if dir == "" {
		return nil, errors.New("storage: NewFileStore requires a non-empty dir")
	}
	// MkdirAll handles the "already exists" case correctly. The mode
	// applies only to newly-created directories; existing dirs keep
	// their current mode.
	if err := os.MkdirAll(dir, DirectoryMode); err != nil {
		return nil, fmt.Errorf("storage: create store dir %q: %w", dir, err)
	}
	return &FileStore{
		dir:  dir,
		lock: flock.New(filepath.Join(dir, ".store.lock")),
	}, nil
}

// Put writes (or overwrites) entry e as dir/<id>.md. The write is
// atomic: the body is staged to a .tmp file and renamed under an
// exclusive flock. Returns a typed error for the documented failure
// modes; the underlying os.Rename error surfaces verbatim otherwise.
func (s *FileStore) Put(ctx context.Context, e Entry) error {
	if s == nil {
		return errors.New("storage: Put on nil FileStore")
	}
	if e.ID == "" {
		return errors.New("storage: Put requires Entry.ID")
	}
	// Honour ctx cancellation before the (potentially blocking) lock
	// acquisition. The flock library does not take a context.
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return ErrStoreClosed
	}
	s.mu.Unlock()

	if err := s.lock.Lock(); err != nil {
		return fmt.Errorf("storage: acquire write lock: %w", err)
	}
	defer func() { _ = s.lock.Unlock() }()

	path := s.entryPath(e.ID)
	tmp := path + ".tmp"

	var buf bytes.Buffer
	writeFrontmatter(&buf, e)
	buf.WriteString(e.Text)

	if err := os.WriteFile(tmp, buf.Bytes(), FileMode); err != nil {
		return fmt.Errorf("storage: stage entry %q: %w", e.ID, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		// Best-effort cleanup of the staging file so the directory
		// does not accumulate .tmp litter on rename failure.
		_ = os.Remove(tmp)
		return fmt.Errorf("storage: commit entry %q: %w", e.ID, err)
	}
	return nil
}

// Query returns entries matching q, sorted by file mtime descending.
// Zero-value Query fields are ignored; the match is the AND of every
// non-zero field. EmbeddingQuery returns ErrUnsupportedQuery — FileStore
// does not compute embeddings.
func (s *FileStore) Query(ctx context.Context, q Query) ([]Entry, error) {
	if s == nil {
		return nil, errors.New("storage: Query on nil FileStore")
	}
	if len(q.EmbeddingQuery) > 0 {
		return nil, ErrUnsupportedQuery
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil, ErrStoreClosed
	}
	s.mu.Unlock()

	// Shared lock so concurrent Queries don't block each other; a Put
	// takes an exclusive lock and serialises against running Queries.
	if err := s.lock.RLock(); err != nil {
		return nil, fmt.Errorf("storage: acquire read lock: %w", err)
	}
	defer func() { _ = s.lock.Unlock() }()

	matches, err := os.ReadDir(s.dir)
	if err != nil {
		return nil, fmt.Errorf("storage: read store dir: %w", err)
	}

	// Build the candidate list first: name, mtime, error. Sort by mtime
	// descending so a Limit applied after filtering returns the most
	// recent matches.
	type candidate struct {
		path  string
		mtime time.Time
	}
	var cands []candidate
	for _, m := range matches {
		if m.IsDir() {
			continue
		}
		name := m.Name()
		if !strings.HasSuffix(name, ".md") {
			continue
		}
		// Skip lock files and staging files explicitly.
		if strings.HasSuffix(name, ".lock") || strings.HasSuffix(name, ".tmp") {
			continue
		}
		info, err := m.Info()
		if err != nil {
			continue
		}
		cands = append(cands, candidate{path: filepath.Join(s.dir, name), mtime: info.ModTime()})
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].mtime.After(cands[j].mtime) })

	limit := q.Limit
	if limit <= 0 {
		limit = DefaultLimit
	}

	out := make([]Entry, 0, limit)
	for _, c := range cands {
		if len(out) >= limit {
			break
		}
		data, err := os.ReadFile(c.path)
		if err != nil {
			continue // best-effort: skip unreadable entries
		}
		entry, ok := parseEntry(data)
		if !ok {
			continue
		}
		if !matchQuery(entry, q) {
			continue
		}
		out = append(out, entry)
	}
	return out, nil
}

// Close releases any held resources. The flock is per-call (taken and
// released inside Put and Query), so Close is effectively a no-op
// beyond marking the store closed. Subsequent Put / Query return
// ErrStoreClosed. Safe to call multiple times.
func (s *FileStore) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	return nil
}

// entryPath returns the absolute path of the file backing id. The id
// is sanitised so path separators cannot escape the store directory.
func (s *FileStore) entryPath(id string) string {
	safe := sanitiseID(id)
	return filepath.Join(s.dir, safe+".md")
}

// sanitiseID rewrites path separators and other filesystem-unsafe
// characters so an embedder-supplied ID cannot escape the store
// directory or create subdirectories. "/" becomes "_" (matching the
// decisions/<date> convention used by the cookbook recipe).
func sanitiseID(id string) string {
	if id == "" {
		return ""
	}
	r := strings.NewReplacer(
		string(os.PathSeparator), "_",
		"/", "_",
		"\\", "_",
		":", "_",
		"..", "_",
	)
	return r.Replace(id)
}

// matchQuery reports whether entry satisfies every non-zero field of q.
// The match is the AND of every populated field.
func matchQuery(e Entry, q Query) bool {
	if q.KeywordQuery != "" {
		// Case-insensitive substring on Text.
		if !strings.Contains(strings.ToLower(e.Text), strings.ToLower(q.KeywordQuery)) {
			return false
		}
	}
	if len(q.TagsQuery) > 0 {
		// Every queried tag must be present (AND).
		have := make(map[string]bool, len(e.Tags))
		for _, t := range e.Tags {
			have[t] = true
		}
		for _, want := range q.TagsQuery {
			if !have[want] {
				return false
			}
		}
	}
	if !q.SinceQuery.IsZero() {
		if e.Timestamp.Before(q.SinceQuery) {
			return false
		}
	}
	return true
}

// writeFrontmatter emits a YAML-style frontmatter block to w. The
// block is delimited by "---" lines; fields appear in a stable order
// (id, source, timestamp, tags) so diffs are minimal across writes.
//
// Empty optional fields are omitted rather than rendered as zero
// values. The round-trip is preserved because parseEntry treats a
// missing key as the field's zero value (empty Source, zero Timestamp,
// nil Tags). ID is always written because every Put requires a
// non-empty ID.
func writeFrontmatter(w *bytes.Buffer, e Entry) {
	w.WriteString("---\n")
	fmt.Fprintf(w, "id: %s\n", yamlEscape(e.ID))
	if e.Source != "" {
		fmt.Fprintf(w, "source: %s\n", yamlEscape(e.Source))
	}
	if !e.Timestamp.IsZero() {
		fmt.Fprintf(w, "timestamp: %s\n", e.Timestamp.UTC().Format(time.RFC3339))
	}
	if len(e.Tags) > 0 {
		w.WriteString("tags:\n")
		for _, t := range e.Tags {
			fmt.Fprintf(w, "  - %s\n", yamlEscape(t))
		}
	}
	w.WriteString("---\n")
}

// yamlEscape returns s with characters that would break single-line
// YAML scalar parsing wrapped in double quotes. The implementation is
// conservative: any value containing ":", "#", leading "-", leading
// " ", or embedded newline is quoted. This covers every field that
// FileStore writes (ids, sources, tags); arbitrary user text is in
// the body, not in frontmatter.
func yamlEscape(s string) string {
	if s == "" {
		return `""`
	}
	needsQuote := false
	if strings.ContainsAny(s, ":#\n\"") {
		needsQuote = true
	}
	if strings.HasPrefix(s, "-") || strings.HasPrefix(s, " ") || strings.HasSuffix(s, " ") {
		needsQuote = true
	}
	if !needsQuote {
		return s
	}
	// Escape backslashes and double quotes, then wrap in quotes.
	r := strings.NewReplacer(`\`, `\\`, `"`, `\"`)
	return `"` + r.Replace(s) + `"`
}

// parseEntry parses the file format produced by writeFrontmatter back
// into an Entry. Returns (Entry, false) if the file does not contain a
// parseable frontmatter block. The body (everything after the closing
// "---") is the Entry.Text.
func parseEntry(data []byte) (Entry, bool) {
	var e Entry
	lines := bytes.Split(data, []byte("\n"))
	if len(lines) < 2 {
		return e, false
	}
	if strings.TrimSpace(string(lines[0])) != "---" {
		return e, false
	}
	// Find the closing "---".
	closeIdx := -1
	for i := 1; i < len(lines); i++ {
		if strings.TrimSpace(string(lines[i])) == "---" {
			closeIdx = i
			break
		}
	}
	if closeIdx == -1 {
		return e, false
	}
	// Parse the frontmatter lines (1..closeIdx-1).
	i := 1
	for i < closeIdx {
		line := strings.TrimSpace(string(lines[i]))
		i++
		if line == "" {
			continue
		}
		key, val, ok := splitKV(line)
		if !ok {
			continue
		}
		switch key {
		case "id":
			e.ID = yamlUnquote(val)
		case "source":
			e.Source = yamlUnquote(val)
		case "timestamp":
			t, err := time.Parse(time.RFC3339, yamlUnquote(val))
			if err == nil {
				e.Timestamp = t
			}
		case "tags":
			// Inline empty list: "tags: []".
			rest := strings.TrimSpace(strings.TrimPrefix(line, "tags:"))
			if rest == "[]" || rest == "" {
				// List form: collect subsequent "  - <tag>" lines.
				for i < closeIdx {
					tagLine := strings.TrimSpace(string(lines[i]))
					if !strings.HasPrefix(tagLine, "- ") {
						break
					}
					e.Tags = append(e.Tags, yamlUnquote(strings.TrimSpace(strings.TrimPrefix(tagLine, "- "))))
					i++
				}
			} else {
				// Single-tag shorthand: "tags: foo".
				e.Tags = []string{yamlUnquote(rest)}
			}
		}
	}
	// Body: everything after closeIdx. Skip the closing "---" line
	// itself; the leading newline after the body is preserved.
	if closeIdx+1 < len(lines) {
		bodyLines := lines[closeIdx+1:]
		body := strings.Join(sliceOfByteSlices(bodyLines), "\n")
		e.Text = body
	}
	return e, true
}

// sliceOfByteSlices converts [][]byte to []string via string conversion
// without the trailing-empty-line artefact that bytes.Join would leave
// when the original file ended with a newline.
func sliceOfByteSlices(bs [][]byte) []string {
	out := make([]string, len(bs))
	for i, b := range bs {
		out[i] = string(b)
	}
	return out
}

// splitKV splits a "key: value" frontmatter line. Returns ok=false if
// the line has no colon.
func splitKV(line string) (string, string, bool) {
	idx := strings.Index(line, ":")
	if idx == -1 {
		return "", "", false
	}
	return strings.TrimSpace(line[:idx]), strings.TrimSpace(line[idx+1:]), true
}

// yamlUnquote reverses yamlEscape's double-quoting for simple values.
// Returns the input unchanged when it is not a double-quoted string.
func yamlUnquote(s string) string {
	s = strings.TrimSpace(s)
	if len(s) >= 2 && s[0] == '"' && s[len(s)-1] == '"' {
		inner := s[1 : len(s)-1]
		r := strings.NewReplacer(`\"`, `"`, `\\`, `\`)
		return r.Replace(inner)
	}
	return s
}
