package tools

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeWriteOps is an in-memory WriteOperations for tests.
type fakeWriteOps struct {
	mu          sync.Mutex
	files       map[string][]byte
	writeErr    error
	writeCount  int
	lastWritten []byte
	statErr     error           // if set, returned for Stat on any path
	dirPaths    map[string]bool // paths that should report IsDir=true
}

func newFakeWriteOps() *fakeWriteOps {
	return &fakeWriteOps{
		files:    map[string][]byte{},
		dirPaths: map[string]bool{},
	}
}

func (f *fakeWriteOps) WriteFile(p string, content []byte) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.writeErr != nil {
		return f.writeErr
	}
	stored := make([]byte, len(content))
	copy(stored, content)
	f.files[p] = stored
	f.writeCount++
	f.lastWritten = stored
	return nil
}

func (f *fakeWriteOps) Stat(p string) (fs.FileInfo, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.statErr != nil {
		return nil, f.statErr
	}
	if data, ok := f.files[p]; ok {
		return fakeFileInfo{name: filepath.Base(p), size: int64(len(data))}, nil
	}
	if f.dirPaths[p] {
		return fakeFileInfo{name: filepath.Base(p), isDir: true}, nil
	}
	return nil, &fs.PathError{Op: "stat", Path: p, Err: fs.ErrNotExist}
}

// fakeFileInfo is a minimal fs.FileInfo for tests.
type fakeFileInfo struct {
	name  string
	size  int64
	isDir bool
	mtime time.Time
}

func (f fakeFileInfo) Name() string       { return f.name }
func (f fakeFileInfo) Size() int64        { return f.size }
func (f fakeFileInfo) Mode() fs.FileMode  { return 0644 }
func (f fakeFileInfo) ModTime() time.Time { return f.mtime }
func (f fakeFileInfo) IsDir() bool        { return f.isDir }
func (f fakeFileInfo) Sys() any           { return nil }

func makeWriteCall(args any) ToolCall {
	raw, _ := json.Marshal(args)
	return ToolCall{
		ID:   "write-call",
		Name: "write",
		Args: raw,
		Cwd:  "/test/cwd",
	}
}

// ---- OSWriteOperations ----

func TestOSWriteOperations_WriteFileAtomic(t *testing.T) {
	ops := OSWriteOperations{}
	dir := t.TempDir()
	p := filepath.Join(dir, "x.txt")
	if err := ops.WriteFile(p, []byte("hello")); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	data, _ := os.ReadFile(p)
	if string(data) != "hello" {
		t.Errorf("data = %q, want hello", string(data))
	}
	// No leftover temp files.
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".x.txt.") {
			t.Errorf("leftover temp: %s", e.Name())
		}
	}
}

func TestOSWriteOperations_OverwritePreservesMode(t *testing.T) {
	ops := OSWriteOperations{}
	dir := t.TempDir()
	p := filepath.Join(dir, "x.txt")
	_ = os.WriteFile(p, []byte("orig"), 0600)
	_ = ops.WriteFile(p, []byte("new"))
	fi, _ := os.Stat(p)
	if fi.Mode().Perm() != 0600 {
		t.Errorf("mode = %o, want 0600", fi.Mode().Perm())
	}
}

func TestOSWriteOperations_NewFileDefaultMode(t *testing.T) {
	ops := OSWriteOperations{}
	dir := t.TempDir()
	p := filepath.Join(dir, "new.txt")
	_ = ops.WriteFile(p, []byte("data"))
	fi, _ := os.Stat(p)
	if fi.Mode().Perm() != 0644 {
		t.Errorf("mode = %o, want 0644", fi.Mode().Perm())
	}
}

func TestOSWriteOperations_Stat(t *testing.T) {
	ops := OSWriteOperations{}
	dir := t.TempDir()
	p := filepath.Join(dir, "x.txt")
	_ = os.WriteFile(p, []byte("hi"), 0644)
	fi, err := ops.Stat(p)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	if fi.Size() != 2 {
		t.Errorf("Size = %d, want 2", fi.Size())
	}
	if _, err := ops.Stat(filepath.Join(dir, "missing.txt")); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("err = %v, want ErrNotExist", err)
	}
}

// ---- isWithinCwd ----

func TestIsWithinCwd(t *testing.T) {
	cwd := "/test/cwd"
	cases := []struct {
		name   string
		target string
		want   bool
	}{
		{"exact cwd", cwd, true},
		{"descendant", "/test/cwd/sub/file.txt", true},
		{"direct child", "/test/cwd/file.txt", true},
		{"sibling", "/test/other/file.txt", false},
		{"parent", "/test", false},
		{"unrelated abs", "/other/path", false},
		{"relative target", "rel.txt", false},
		{"relative cwd", "rel", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := isWithinCwd(tc.target, cwd); got != tc.want {
				t.Errorf("isWithinCwd(%q, %q) = %v, want %v", tc.target, cwd, got, tc.want)
			}
		})
	}
}

// ---- writeTool.Execute — happy path ----

func TestWriteTool_NewFile(t *testing.T) {
	ops := newFakeWriteOps()
	tool := NewWriteTool(ops)
	res, err := tool.Execute(context.Background(), makeWriteCall(writeArgs{
		Path:    "new.txt",
		Content: "fresh content",
	}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	if string(ops.files["/test/cwd/new.txt"]) != "fresh content" {
		t.Errorf("file = %q, want fresh content", string(ops.files["/test/cwd/new.txt"]))
	}
	if !strings.Contains(ResultText(res), "Wrote") {
		t.Errorf("result should say Wrote: %q", ResultText(res))
	}
}

func TestWriteTool_Overwrite(t *testing.T) {
	ops := newFakeWriteOps()
	ops.files["/test/cwd/x.txt"] = []byte("old")
	tool := NewWriteTool(ops)
	res, _ := tool.Execute(context.Background(), makeWriteCall(writeArgs{
		Path:    "x.txt",
		Content: "new",
	}))
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	if !strings.Contains(ResultText(res), "Overwrote") {
		t.Errorf("result should say Overwrote: %q", ResultText(res))
	}
	if string(ops.files["/test/cwd/x.txt"]) != "new" {
		t.Errorf("file = %q, want new", string(ops.files["/test/cwd/x.txt"]))
	}
}

func TestWriteTool_OutOfCwdRefused(t *testing.T) {
	ops := newFakeWriteOps()
	tool := NewWriteTool(ops)
	res, _ := tool.Execute(context.Background(), makeWriteCall(writeArgs{
		Path:    "/elsewhere/x.txt",
		Content: "data",
	}))
	if !res.IsError {
		t.Errorf("expected IsError on out-of-cwd write")
	}
	if !strings.Contains(ResultText(res), "outside cwd") {
		t.Errorf("error should mention outside cwd: %s", ResultText(res))
	}
}

func TestWriteTool_OutOfCwdAllowed(t *testing.T) {
	ops := newFakeWriteOps()
	tool := NewWriteTool(ops)
	res, _ := tool.Execute(context.Background(), makeWriteCall(writeArgs{
		Path:          "/elsewhere/x.txt",
		Content:       "data",
		AllowOutOfCwd: true,
	}))
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	if string(ops.files["/elsewhere/x.txt"]) != "data" {
		t.Errorf("file = %q, want data", string(ops.files["/elsewhere/x.txt"]))
	}
}

func TestWriteTool_DirectoryRefused(t *testing.T) {
	ops := newFakeWriteOps()
	ops.dirPaths["/test/cwd/subdir"] = true
	tool := NewWriteTool(ops)
	res, _ := tool.Execute(context.Background(), makeWriteCall(writeArgs{
		Path:    "subdir",
		Content: "data",
	}))
	if !res.IsError {
		t.Errorf("expected IsError on directory target")
	}
}

func TestWriteTool_MissingPath(t *testing.T) {
	tool := NewWriteTool(newFakeWriteOps())
	res, _ := tool.Execute(context.Background(), makeWriteCall(writeArgs{Content: "x"}))
	if !res.IsError {
		t.Errorf("expected IsError on missing path")
	}
}

func TestWriteTool_MissingCwd(t *testing.T) {
	tool := NewWriteTool(newFakeWriteOps())
	raw, _ := json.Marshal(writeArgs{Path: "x.txt", Content: "x"})
	res, _ := tool.Execute(context.Background(), ToolCall{
		ID:   "x",
		Name: "write",
		Args: raw,
		Cwd:  "",
	})
	if !res.IsError {
		t.Errorf("expected IsError on missing cwd")
	}
}

func TestWriteTool_InvalidJSON(t *testing.T) {
	tool := NewWriteTool(newFakeWriteOps())
	res, err := tool.Execute(context.Background(), ToolCall{
		ID:   "x",
		Name: "write",
		Args: json.RawMessage(`{"path": invalid`),
		Cwd:  "/cwd",
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !res.IsError {
		t.Errorf("invalid JSON should yield IsError")
	}
}

func TestWriteTool_NilOpsDefaultsToOS(t *testing.T) {
	tool := NewWriteTool(nil)
	if wt, ok := tool.(*writeTool); !ok || wt.ops == nil {
		t.Errorf("nil ops should default to OSWriteOperations")
	}
}

// ---- writeTool.Name / Description / Parameters ----

func TestWriteTool_NameDescriptionParameters(t *testing.T) {
	tool := NewWriteTool(newFakeWriteOps())
	if tool.Name() != "write" {
		t.Errorf("Name = %q, want write", tool.Name())
	}
	if tool.Description() == "" {
		t.Errorf("Description should not be empty")
	}
	s := tool.Parameters()
	if s.Type != "object" {
		t.Errorf("Type = %q, want object", s.Type)
	}
	raw, _ := s.MarshalJSON()
	if !strings.Contains(string(raw), `"content"`) {
		t.Errorf("schema should mention content: %s", string(raw))
	}
	if !strings.Contains(string(raw), `"allowOutOfCwd"`) {
		t.Errorf("schema should mention allowOutOfCwd: %s", string(raw))
	}
}

// ---- writeTool.RenderCall / RenderResult ----

func TestWriteTool_RenderCall_Plain(t *testing.T) {
	tool := NewWriteTool(newFakeWriteOps())
	got := tool.RenderCall(json.RawMessage(`{"path":"x.txt","content":"hello"}`), PlainTheme())
	if !strings.Contains(got, "write") || !strings.Contains(got, "x.txt") {
		t.Errorf("RenderCall missing name/path: %q", got)
	}
	if !strings.Contains(got, "5 bytes") {
		t.Errorf("RenderCall should show content size: %q", got)
	}
}

func TestWriteTool_RenderCall_AllowOutOfCwd(t *testing.T) {
	tool := NewWriteTool(newFakeWriteOps())
	got := tool.RenderCall(json.RawMessage(`{"path":"x","content":"y","allowOutOfCwd":true}`), ColorTheme())
	if !strings.Contains(got, "allow-out-of-cwd") {
		t.Errorf("RenderCall should show allow-out-of-cwd marker: %q", got)
	}
}

func TestWriteTool_RenderResult_Text(t *testing.T) {
	tool := NewWriteTool(newFakeWriteOps())
	res := NewTextResult("Wrote x.txt (5 bytes).")
	got := tool.RenderResult(res, PlainTheme())
	if !strings.Contains(got, "Wrote") {
		t.Errorf("RenderResult missing header: %q", got)
	}
}

func TestWriteTool_RenderResult_Error(t *testing.T) {
	tool := NewWriteTool(newFakeWriteOps())
	res := NewErrorResult("permission denied")
	got := tool.RenderResult(res, ColorTheme())
	if !strings.Contains(got, "permission denied") {
		t.Errorf("RenderResult missing error: %q", got)
	}
}

// ---- end-to-end real FS ----

func TestWriteTool_RealFSCreateOverwrite(t *testing.T) {
	dir := t.TempDir()
	tool := NewWriteTool(OSWriteOperations{})
	absFile := filepath.Join(dir, "real.txt")

	// Create.
	res, err := tool.Execute(context.Background(), ToolCall{
		ID:   "x",
		Name: "write",
		Args: jsonRawMustMarshal(writeArgs{Path: absFile, Content: "first"}),
		Cwd:  dir,
	})
	if err != nil {
		t.Fatalf("Execute (create): %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error (create): %s", ResultText(res))
	}
	data, _ := os.ReadFile(absFile)
	if string(data) != "first" {
		t.Errorf("first write: data = %q", string(data))
	}

	// Overwrite.
	res2, _ := tool.Execute(context.Background(), ToolCall{
		ID:   "x",
		Name: "write",
		Args: jsonRawMustMarshal(writeArgs{Path: absFile, Content: "second"}),
		Cwd:  dir,
	})
	if res2.IsError {
		t.Fatalf("unexpected error (overwrite): %s", ResultText(res2))
	}
	if !strings.Contains(ResultText(res2), "Overwrote") {
		t.Errorf("overwrite should say Overwrote: %q", ResultText(res2))
	}
	data2, _ := os.ReadFile(absFile)
	if string(data2) != "second" {
		t.Errorf("overwrite: data = %q, want second", string(data2))
	}
}

func TestWriteTool_RealFS_OutOfCwdRefused(t *testing.T) {
	dir := t.TempDir()
	tool := NewWriteTool(OSWriteOperations{})
	// Use a path outside dir.
	outside := filepath.Join(filepath.Dir(dir), "outside-"+filepath.Base(dir)+".txt")
	res, _ := tool.Execute(context.Background(), ToolCall{
		ID:   "x",
		Name: "write",
		Args: jsonRawMustMarshal(writeArgs{Path: outside, Content: "x"}),
		Cwd:  dir,
	})
	if !res.IsError {
		t.Errorf("expected IsError on out-of-cwd write")
	}
}

// ---- concurrency serialization ----

func TestWriteTool_ConcurrentSamePathSerializes(t *testing.T) {
	// Two writes to the same file: the final content should be one of
	// the two, with no partial-write corruption.
	ops := newFakeWriteOps()
	tool := NewWriteTool(ops)
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		_, _ = tool.Execute(context.Background(), makeWriteCall(writeArgs{
			Path: "f.txt", Content: strings.Repeat("A", 1000),
		}))
	}()
	go func() {
		defer wg.Done()
		_, _ = tool.Execute(context.Background(), makeWriteCall(writeArgs{
			Path: "f.txt", Content: strings.Repeat("B", 1000),
		}))
	}()
	wg.Wait()
	data := string(ops.files["/test/cwd/f.txt"])
	if len(data) != 1000 {
		t.Errorf("final length = %d, want 1000 (race?)", len(data))
	}
	if data != strings.Repeat("A", 1000) && data != strings.Repeat("B", 1000) {
		t.Errorf("final content mixed, not all-A or all-B")
	}
}
