package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fakeLSOps is an in-memory LSOperations for unit tests.
type fakeLSOps struct {
	dirs    map[string]bool           // absolute path -> is directory
	entries map[string][]fakeDirEntry // directory abs path -> children
	statErr error                     // if set, returned from IsDirectory
	readErr error                     // if set, returned from ReadDir
}

func newFakeLSOps() *fakeLSOps {
	return &fakeLSOps{
		dirs:    map[string]bool{},
		entries: map[string][]fakeDirEntry{},
	}
}

// addFile registers a regular file at absolutePath with size and mtime.
// Parent directories are marked as directories automatically and receive
// the entry in their children list.
func (f *fakeLSOps) addFile(absolutePath string, size int64, mtime time.Time) {
	dir := filepath.Dir(absolutePath)
	f.ensureDir(dir)
	name := filepath.Base(absolutePath)
	f.entries[dir] = append(f.entries[dir], fakeDirEntry{
		name:  name,
		isDir: false,
		mode:  0644,
		size:  size,
		mtime: mtime,
	})
}

// addDir registers an empty directory at absolutePath (and ancestors).
func (f *fakeLSOps) addDir(absolutePath string) {
	f.ensureDir(absolutePath)
}

func (f *fakeLSOps) ensureDir(absolutePath string) {
	if absolutePath == "" || absolutePath == "." || absolutePath == "/" {
		return
	}
	f.dirs[absolutePath] = true
	if _, ok := f.entries[absolutePath]; !ok {
		f.entries[absolutePath] = nil
	}
	// Ensure ancestor exists and has this dir as a child entry.
	parent := filepath.Dir(absolutePath)
	if parent != absolutePath {
		f.ensureDir(parent)
		name := filepath.Base(absolutePath)
		// Avoid duplicate entries.
		already := false
		for _, e := range f.entries[parent] {
			if e.name == name {
				already = true
				break
			}
		}
		if !already {
			f.entries[parent] = append(f.entries[parent], fakeDirEntry{
				name:  name,
				isDir: true,
				mode:  0755 | fs.ModeDir,
			})
		}
	}
}

func (f *fakeLSOps) IsDirectory(p string) (bool, error) {
	if f.statErr != nil {
		return false, f.statErr
	}
	if f.dirs[p] {
		return true, nil
	}
	// Not a directory; check if it exists as a file via parent lookup.
	parent := filepath.Dir(p)
	name := filepath.Base(p)
	for _, e := range f.entries[parent] {
		if e.name == name && !e.isDir {
			return false, nil
		}
	}
	return false, &fs.PathError{Op: "stat", Path: p, Err: fs.ErrNotExist}
}

func (f *fakeLSOps) ReadDir(p string) ([]fs.DirEntry, error) {
	if f.readErr != nil {
		return nil, f.readErr
	}
	entries, ok := f.entries[p]
	if !ok {
		return nil, &fs.PathError{Op: "readdir", Path: p, Err: fs.ErrNotExist}
	}
	out := make([]fs.DirEntry, 0, len(entries))
	for _, e := range entries {
		out = append(out, e)
	}
	return out, nil
}

// Verify fakeLSOps satisfies LSOperations at compile time.
var _ LSOperations = (*fakeLSOps)(nil)

func makeLSCall(args any) ToolCall {
	raw, _ := json.Marshal(args)
	return ToolCall{
		ID:   "ls-call",
		Name: "ls",
		Args: raw,
		Cwd:  "/test/cwd",
	}
}

// ---- basics ----

func TestLSTool_NameDescriptionParameters(t *testing.T) {
	tool := NewLSTool(nil)
	if tool.Name() != "ls" {
		t.Errorf("Name = %q, want ls", tool.Name())
	}
	if tool.Description() == "" {
		t.Errorf("Description should not be empty")
	}
	s := tool.Parameters()
	if s.Type != "object" {
		t.Errorf("Type = %q, want object", s.Type)
	}
	raw, _ := s.MarshalJSON()
	if !strings.Contains(string(raw), `"path"`) {
		t.Errorf("schema should mention path: %s", string(raw))
	}
}

func TestLSTool_NilOpsDefaultsToOS(t *testing.T) {
	tool := NewLSTool(nil)
	if _, ok := tool.(*lsTool).ops.(OSLSOperations); !ok {
		t.Errorf("nil ops should default to OSLSOperations, got %T", tool.(*lsTool).ops)
	}
}

// ---- happy path ----

func TestLSTool_ListsDirAndFiles(t *testing.T) {
	ops := newFakeLSOps()
	mtime := time.Date(2024, 1, 15, 10, 30, 0, 0, time.UTC)
	ops.addDir("/test/cwd/subdir")
	ops.addFile("/test/cwd/file.go", 1234, mtime)
	ops.addFile("/test/cwd/another.txt", 567, mtime)
	tool := NewLSTool(ops)

	res, err := tool.Execute(context.Background(), makeLSCall(lsArgs{}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	// Directories first, then files alphabetically. Expected order:
	// subdir/, another.txt, file.go.
	lines := strings.Split(got, "\n")
	if len(lines) < 3 {
		t.Fatalf("expected >= 3 lines, got %d: %s", len(lines), got)
	}
	// Strip any trailing notice block before asserting order.
	var entries []string
	for _, l := range lines {
		if l == "" || strings.HasPrefix(l, "[") {
			continue
		}
		entries = append(entries, l)
	}
	if len(entries) != 3 {
		t.Fatalf("expected 3 entries, got %d: %v", len(entries), entries)
	}
	if !strings.Contains(entries[0], "subdir/") {
		t.Errorf("dir should come first: %v", entries)
	}
	if !strings.Contains(entries[1], "another.txt") {
		t.Errorf("file 'another.txt' should come second: %v", entries)
	}
	if !strings.Contains(entries[2], "file.go") {
		t.Errorf("file 'file.go' should come third: %v", entries)
	}
	// Type indicators present.
	if !strings.HasPrefix(strings.TrimSpace(entries[0]), "d") {
		t.Errorf("dir row should start with 'd': %q", entries[0])
	}
	if !strings.HasPrefix(strings.TrimSpace(entries[2]), "-") {
		t.Errorf("file row should start with '-': %q", entries[2])
	}
	// Size and mtime are present.
	if !strings.Contains(entries[2], "1234") {
		t.Errorf("file size missing from row: %q", entries[2])
	}
	if !strings.Contains(entries[2], "2024-01-15 10:30") {
		t.Errorf("mtime missing or wrong format: %q", entries[2])
	}
}

func TestLSTool_EmptyDirectory(t *testing.T) {
	ops := newFakeLSOps()
	ops.addDir("/test/cwd/empty")
	tool := NewLSTool(ops)
	res, err := tool.Execute(context.Background(), makeLSCall(lsArgs{Path: "empty"}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	if !strings.Contains(got, "empty") {
		t.Errorf("expected empty marker, got: %s", got)
	}
}

func TestLSTool_CustomPath(t *testing.T) {
	ops := newFakeLSOps()
	mtime := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	ops.addDir("/test/cwd/src")
	ops.addFile("/test/cwd/src/a.go", 10, mtime)
	ops.addFile("/test/cwd/outside.go", 100, mtime)
	tool := NewLSTool(ops)

	res, err := tool.Execute(context.Background(), makeLSCall(lsArgs{Path: "src"}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	got := ResultText(res)
	if !strings.Contains(got, "a.go") {
		t.Errorf("expected a.go under src: %s", got)
	}
	if strings.Contains(got, "outside.go") {
		t.Errorf("outside.go should not be in src listing: %s", got)
	}
}

func TestLSTool_DefaultPathIsCwd(t *testing.T) {
	ops := newFakeLSOps()
	mtime := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	ops.addFile("/test/cwd/root.go", 10, mtime)
	tool := NewLSTool(ops)

	// Empty path → defaults to "." → resolves to call.Cwd.
	res, err := tool.Execute(context.Background(), makeLSCall(lsArgs{}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(ResultText(res), "root.go") {
		t.Errorf("default path should list cwd: %s", ResultText(res))
	}
}

// ---- error paths ----

func TestLSTool_MissingCwd(t *testing.T) {
	tool := NewLSTool(newFakeLSOps())
	raw, _ := json.Marshal(lsArgs{})
	call := ToolCall{ID: "ls", Name: "ls", Args: raw, Cwd: ""}
	res, err := tool.Execute(context.Background(), call)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !res.IsError || !strings.Contains(ResultText(res), "cwd") {
		t.Errorf("expected missing-cwd error, got: %+v", res)
	}
}

func TestLSTool_InvalidJSON(t *testing.T) {
	tool := NewLSTool(newFakeLSOps())
	call := ToolCall{
		ID:   "ls",
		Name: "ls",
		Args: json.RawMessage(`{"path": broken`),
		Cwd:  "/test/cwd",
	}
	res, err := tool.Execute(context.Background(), call)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !res.IsError {
		t.Errorf("expected error result for invalid JSON, got: %+v", res)
	}
}

func TestLSTool_PathNotFound(t *testing.T) {
	tool := NewLSTool(newFakeLSOps())
	res, err := tool.Execute(context.Background(), makeLSCall(lsArgs{Path: "missing"}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !res.IsError || !strings.Contains(ResultText(res), "not found") {
		t.Errorf("expected not-found error, got: %+v", res)
	}
}

func TestLSTool_NotADirectory(t *testing.T) {
	ops := newFakeLSOps()
	mtime := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	ops.addFile("/test/cwd/file.go", 10, mtime)
	tool := NewLSTool(ops)
	res, err := tool.Execute(context.Background(), makeLSCall(lsArgs{Path: "file.go"}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !res.IsError || !strings.Contains(ResultText(res), "not a directory") {
		t.Errorf("expected not-a-directory error, got: %+v", res)
	}
}

func TestLSTool_StatError(t *testing.T) {
	ops := newFakeLSOps()
	ops.statErr = errors.New("simulated permission denied")
	tool := NewLSTool(ops)
	res, err := tool.Execute(context.Background(), makeLSCall(lsArgs{}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !res.IsError {
		t.Errorf("expected error result for stat failure, got: %+v", res)
	}
}

func TestLSTool_ReadDirError(t *testing.T) {
	ops := newFakeLSOps()
	ops.addDir("/test/cwd")
	ops.readErr = errors.New("simulated readdir failure")
	tool := NewLSTool(ops)
	res, err := tool.Execute(context.Background(), makeLSCall(lsArgs{}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !res.IsError || !strings.Contains(ResultText(res), "readdir") {
		t.Errorf("expected readdir error, got: %+v", res)
	}
}

func TestLSTool_CtxCanceled(t *testing.T) {
	tool := NewLSTool(newFakeLSOps())
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := tool.Execute(ctx, makeLSCall(lsArgs{}))
	if err == nil {
		t.Errorf("expected ctx.Err() to surface, got nil")
	}
}

// ---- format helpers ----

func TestFormatLSSize(t *testing.T) {
	if got := formatLSSize(0); got != "0" {
		t.Errorf("formatLSSize(0) = %q", got)
	}
	if got := formatLSSize(1234); got != "1234" {
		t.Errorf("formatLSSize(1234) = %q", got)
	}
	if got := formatLSSize(-1); got != "-" {
		t.Errorf("formatLSSize(-1) = %q, want -", got)
	}
}

func TestFormatLSPaddedSize(t *testing.T) {
	if got := formatLSPaddedSize(5, 4); got != "   5" {
		t.Errorf("got %q", got)
	}
	if got := formatLSPaddedSize(12345, 3); got != "12345" {
		t.Errorf("got %q", got)
	}
}

func TestTypeIndicatorFor(t *testing.T) {
	if got := typeIndicatorFor(true, fs.ModeDir); got != "d" {
		t.Errorf("dir indicator = %q, want d", got)
	}
	if got := typeIndicatorFor(false, 0); got != "-" {
		t.Errorf("file indicator = %q, want -", got)
	}
	if got := typeIndicatorFor(false, fs.ModeSymlink); got != "l" {
		t.Errorf("symlink indicator = %q, want l", got)
	}
}

func TestLSDisplayName(t *testing.T) {
	if got := lsDisplayName("foo", true); got != "foo/" {
		t.Errorf("got %q", got)
	}
	if got := lsDisplayName("foo", false); got != "foo" {
		t.Errorf("got %q", got)
	}
}

func TestBuildLSNotices_NoTruncation(t *testing.T) {
	notices := buildLSNotices(5, "abc", "abc")
	if len(notices) != 0 {
		t.Errorf("expected no notices, got %v", notices)
	}
}

func TestBuildLSNotices_Truncated(t *testing.T) {
	raw := strings.Repeat("a", 1024)
	capped := strings.Repeat("a", 512)
	notices := buildLSNotices(100, raw, capped)
	if len(notices) != 1 {
		t.Fatalf("expected 1 notice, got %v", notices)
	}
	if !strings.Contains(notices[0], "truncated") {
		t.Errorf("notice should mention truncation: %v", notices)
	}
}

// TestLSTool_ExecuteTruncatesManyEntries drives Execute with more entries
// than DefaultMaxLines and verifies the truncation notice fires.
func TestLSTool_ExecuteTruncatesManyEntries(t *testing.T) {
	ops := newFakeLSOps()
	mtime := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	ops.addDir("/test/cwd")
	for i := 0; i < DefaultMaxLines+50; i++ {
		ops.addFile(fmt.Sprintf("/test/cwd/file-%04d.txt", i), 10, mtime)
	}
	tool := NewLSTool(ops)

	res, err := tool.Execute(context.Background(), makeLSCall(lsArgs{}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	if !strings.Contains(got, "truncated") {
		t.Errorf("expected truncation notice in output: %q", got)
	}
}

// ---- RenderCall / RenderResult ----

func TestLSTool_RenderCall_Default(t *testing.T) {
	tool := NewLSTool(nil)
	got := tool.RenderCall(json.RawMessage(`{}`), PlainTheme())
	if !strings.Contains(got, "ls") {
		t.Errorf("missing tool name: %q", got)
	}
	if !strings.Contains(got, ".") {
		t.Errorf("default path should show '.': %q", got)
	}
}

func TestLSTool_RenderCall_CustomPath(t *testing.T) {
	tool := NewLSTool(nil)
	got := tool.RenderCall(json.RawMessage(`{"path":"src"}`), PlainTheme())
	if !strings.Contains(got, "src") {
		t.Errorf("missing custom path: %q", got)
	}
}

func TestLSTool_RenderResult_Text(t *testing.T) {
	tool := NewLSTool(nil)
	res := NewTextResult("d  src/")
	got := tool.RenderResult(res, ColorTheme())
	if !strings.Contains(got, "src") {
		t.Errorf("missing content: %q", got)
	}
}

func TestLSTool_RenderResult_Error(t *testing.T) {
	tool := NewLSTool(nil)
	res := NewErrorResult("not a directory")
	got := tool.RenderResult(res, ColorTheme())
	if !strings.Contains(got, "error") {
		t.Errorf("missing error marker: %q", got)
	}
}

// ---- real-FS smoke test ----

func TestLSTool_RealFS_ListsTestDir(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping real-FS test in -short mode")
	}
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	tool := NewLSTool(OSLSOperations{})
	raw, _ := json.Marshal(lsArgs{})
	res, err := tool.Execute(context.Background(), ToolCall{
		ID:   "ls-real",
		Name: "ls",
		Args: raw,
		Cwd:  wd,
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	// The tools test directory contains ls.go, ls_test.go, find.go, etc.
	// ls.go itself must appear in the listing.
	if !strings.Contains(got, "ls.go") {
		t.Errorf("expected ls.go in real-FS listing, got: %s", got)
	}
}
