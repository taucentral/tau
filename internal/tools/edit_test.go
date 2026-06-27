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
)

// fakeEditOps is an in-memory EditOperations for unit tests.
type fakeEditOps struct {
	mu          sync.Mutex
	files       map[string][]byte
	accessErr   error
	readErr     error
	writeErr    error
	writeCount  int
	lastWritten []byte
}

func newFakeEditOps() *fakeEditOps {
	return &fakeEditOps{files: map[string][]byte{}}
}

func (f *fakeEditOps) ReadFile(p string) ([]byte, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.readErr != nil {
		return nil, f.readErr
	}
	if data, ok := f.files[p]; ok {
		// Return a copy so callers can mutate without surprising us.
		out := make([]byte, len(data))
		copy(out, data)
		return out, nil
	}
	return nil, &fs.PathError{Op: "open", Path: p, Err: fs.ErrNotExist}
}

func (f *fakeEditOps) WriteFile(p string, content []byte) error {
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

func (f *fakeEditOps) Access(p string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.accessErr != nil {
		return f.accessErr
	}
	data, ok := f.files[p]
	if !ok {
		return &fs.PathError{Op: "open", Path: p, Err: fs.ErrNotExist}
	}
	_ = data
	return nil
}

func makeEditCall(args any) ToolCall {
	raw, _ := json.Marshal(args)
	return ToolCall{
		ID:   "edit-call",
		Name: "edit",
		Args: raw,
		Cwd:  "/test/cwd",
	}
}

// ---- OSEditOperations ----

func TestOSEditOperations_ReadWriteAccess(t *testing.T) {
	ops := OSEditOperations{}
	dir := t.TempDir()
	p := filepath.Join(dir, "x.txt")
	if err := os.WriteFile(p, []byte("hello"), 0600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	data, err := ops.ReadFile(p)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if string(data) != "hello" {
		t.Errorf("data = %q, want hello", string(data))
	}
	if err := ops.Access(p); err != nil {
		t.Errorf("Access: %v", err)
	}
	if err := ops.WriteFile(p, []byte("world")); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	data2, _ := ops.ReadFile(p)
	if string(data2) != "world" {
		t.Errorf("after write: data = %q, want world", string(data2))
	}
}

func TestOSEditOperations_AccessMissing(t *testing.T) {
	ops := OSEditOperations{}
	dir := t.TempDir()
	if err := ops.Access(filepath.Join(dir, "nope.txt")); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("err = %v, want ErrNotExist", err)
	}
}

func TestOSEditOperations_AccessDir(t *testing.T) {
	ops := OSEditOperations{}
	dir := t.TempDir()
	if err := ops.Access(dir); err == nil {
		t.Errorf("expected error on directory access")
	}
}

func TestOSEditOperations_WriteFileAtomic(t *testing.T) {
	ops := OSEditOperations{}
	dir := t.TempDir()
	p := filepath.Join(dir, "atomic.txt")
	if err := os.WriteFile(p, []byte("orig"), 0600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if err := ops.WriteFile(p, []byte("replaced")); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	// No leftover temp files.
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".atomic.txt.") {
			t.Errorf("leftover temp file: %s", e.Name())
		}
	}
	// Content matches.
	data, _ := os.ReadFile(p)
	if string(data) != "replaced" {
		t.Errorf("data = %q, want replaced", string(data))
	}
}

func TestOSEditOperations_WriteFilePreservesMode(t *testing.T) {
	ops := OSEditOperations{}
	dir := t.TempDir()
	p := filepath.Join(dir, "mode.txt")
	_ = os.WriteFile(p, []byte("orig"), 0600)
	_ = ops.WriteFile(p, []byte("new"))
	fi, err := os.Stat(p)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	if fi.Mode().Perm() != 0600 {
		t.Errorf("mode = %o, want 0600", fi.Mode().Perm())
	}
}

// ---- splitLines ----

func TestSplitLines(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  []string
	}{
		{"empty", "", nil},
		{"no newline", "abc", []string{"abc"}},
		{"trailing newline", "a\nb\n", []string{"a\n", "b\n"}},
		{"no trailing newline multiline", "a\nb", []string{"a\n", "b"}},
		{"only newlines", "\n\n", []string{"\n", "\n"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := splitLines([]byte(tc.input))
			if len(got) != len(tc.want) {
				t.Errorf("len = %d, want %d (%q vs %q)", len(got), len(tc.want), got, tc.want)
				return
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Errorf("[%d] = %q, want %q", i, got[i], tc.want[i])
				}
			}
		})
	}
}

// ---- editTool.Execute — happy path ----

func TestEditTool_Replace(t *testing.T) {
	ops := newFakeEditOps()
	ops.files["/test/cwd/x.txt"] = []byte("hello world\nfoo bar\n")
	tool := NewEditTool(ops)
	res, err := tool.Execute(context.Background(), makeEditCall(editArgs{
		Path:      "x.txt",
		OldString: "foo bar",
		NewString: "baz qux",
	}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	if !strings.Contains(got, "Edited") {
		t.Errorf("missing Edited header: %q", got)
	}
	if !strings.Contains(got, "-foo bar") {
		t.Errorf("diff missing -foo bar: %q", got)
	}
	if !strings.Contains(got, "+baz qux") {
		t.Errorf("diff missing +baz qux: %q", got)
	}
	// File was updated.
	if string(ops.files["/test/cwd/x.txt"]) != "hello world\nbaz qux\n" {
		t.Errorf("file content = %q, want hello+baz", string(ops.files["/test/cwd/x.txt"]))
	}
}

func TestEditTool_NewFileRefused(t *testing.T) {
	tool := NewEditTool(newFakeEditOps())
	res, _ := tool.Execute(context.Background(), makeEditCall(editArgs{
		Path:      "missing.txt",
		OldString: "x",
		NewString: "y",
	}))
	if !res.IsError {
		t.Errorf("expected IsError on missing file")
	}
	if !strings.Contains(ResultText(res), "does not exist") {
		t.Errorf("error should mention 'does not exist': %s", ResultText(res))
	}
}

func TestEditTool_NotFound(t *testing.T) {
	ops := newFakeEditOps()
	ops.files["/test/cwd/x.txt"] = []byte("hello world\n")
	tool := NewEditTool(ops)
	res, _ := tool.Execute(context.Background(), makeEditCall(editArgs{
		Path:      "x.txt",
		OldString: "missing text",
		NewString: "anything",
	}))
	if !res.IsError {
		t.Errorf("expected IsError on missing oldString")
	}
	if !strings.Contains(ResultText(res), "not found") {
		t.Errorf("error should mention 'not found': %s", ResultText(res))
	}
}

func TestEditTool_NotUnique(t *testing.T) {
	ops := newFakeEditOps()
	ops.files["/test/cwd/x.txt"] = []byte("dup\ndup\ndup\n")
	tool := NewEditTool(ops)
	res, _ := tool.Execute(context.Background(), makeEditCall(editArgs{
		Path:      "x.txt",
		OldString: "dup",
		NewString: "unique",
	}))
	if !res.IsError {
		t.Errorf("expected IsError on duplicate oldString")
	}
	if !strings.Contains(ResultText(res), "3 times") {
		t.Errorf("error should mention occurrence count: %s", ResultText(res))
	}
}

func TestEditTool_EmptyOldString(t *testing.T) {
	tool := NewEditTool(newFakeEditOps())
	res, _ := tool.Execute(context.Background(), makeEditCall(editArgs{
		Path:      "x.txt",
		OldString: "",
		NewString: "y",
	}))
	if !res.IsError {
		t.Errorf("empty oldString should be rejected")
	}
}

func TestEditTool_IdenticalStrings(t *testing.T) {
	tool := NewEditTool(newFakeEditOps())
	res, _ := tool.Execute(context.Background(), makeEditCall(editArgs{
		Path:      "x.txt",
		OldString: "same",
		NewString: "same",
	}))
	if !res.IsError {
		t.Errorf("identical old/new should be rejected")
	}
}

func TestEditTool_MissingPath(t *testing.T) {
	tool := NewEditTool(newFakeEditOps())
	res, _ := tool.Execute(context.Background(), makeEditCall(editArgs{
		OldString: "x",
		NewString: "y",
	}))
	if !res.IsError {
		t.Errorf("missing path should be rejected")
	}
}

func TestEditTool_DeleteBlock(t *testing.T) {
	ops := newFakeEditOps()
	ops.files["/test/cwd/x.txt"] = []byte("a\nDELETE ME\nb\n")
	tool := NewEditTool(ops)
	res, _ := tool.Execute(context.Background(), makeEditCall(editArgs{
		Path:      "x.txt",
		OldString: "DELETE ME\n",
		NewString: "",
	}))
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	if string(ops.files["/test/cwd/x.txt"]) != "a\nb\n" {
		t.Errorf("file after delete = %q, want a\\nb\\n", string(ops.files["/test/cwd/x.txt"]))
	}
}

func TestEditTool_InvalidJSON(t *testing.T) {
	tool := NewEditTool(newFakeEditOps())
	res, err := tool.Execute(context.Background(), ToolCall{
		ID:   "x",
		Name: "edit",
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

func TestEditTool_NilOpsDefaultsToOS(t *testing.T) {
	tool := NewEditTool(nil)
	if et, ok := tool.(*editTool); !ok || et.ops == nil {
		t.Errorf("nil ops should default to OSEditOperations")
	}
}

// ---- editTool.Name / Description / Parameters ----

func TestEditTool_NameDescriptionParameters(t *testing.T) {
	tool := NewEditTool(newFakeEditOps())
	if tool.Name() != "edit" {
		t.Errorf("Name = %q, want edit", tool.Name())
	}
	if tool.Description() == "" {
		t.Errorf("Description should not be empty")
	}
	s := tool.Parameters()
	if s.Type != "object" {
		t.Errorf("Type = %q, want object", s.Type)
	}
	raw, _ := s.MarshalJSON()
	if !strings.Contains(string(raw), `"oldString"`) {
		t.Errorf("schema should mention oldString: %s", string(raw))
	}
	if !strings.Contains(string(raw), `"newString"`) {
		t.Errorf("schema should mention newString: %s", string(raw))
	}
}

// ---- editTool — concurrency serialization ----

func TestEditTool_ConcurrentSamePathSerializes(t *testing.T) {
	// Two edits to the same file targeting independent lines. Both must
	// be reflected in the final content regardless of arrival order; if
	// the FileMutationQueue failed to serialize, the second write would
	// clobber the first and one edit would be lost.
	//
	// Order-dependent assertions (e.g. B's oldString being the result of
	// A) are deliberately avoided here because Go's goroutine scheduler
	// does not guarantee arrival order. The "second sees first's result"
	// property is covered separately by TestEditTool_SequentialSecondSeesFirst.
	ops := newFakeEditOps()
	ops.files["/test/cwd/x.txt"] = []byte("alpha\nbeta\n")
	tool := NewEditTool(ops)

	var wg sync.WaitGroup
	errs := make([]error, 2)
	wg.Add(2)
	go func() {
		defer wg.Done()
		_, errs[0] = tool.Execute(context.Background(), makeEditCall(editArgs{
			Path:      "x.txt",
			OldString: "alpha",
			NewString: "alpha-x",
		}))
	}()
	go func() {
		defer wg.Done()
		_, errs[1] = tool.Execute(context.Background(), makeEditCall(editArgs{
			Path:      "x.txt",
			OldString: "beta",
			NewString: "beta-y",
		}))
	}()
	wg.Wait()
	if errs[0] != nil {
		t.Errorf("first edit errored: %v", errs[0])
	}
	if errs[1] != nil {
		t.Errorf("second edit errored: %v", errs[1])
	}
	final := string(ops.files["/test/cwd/x.txt"])
	if !strings.Contains(final, "alpha-x") {
		t.Errorf("first edit's change lost (queue did not serialize): %q", final)
	}
	if !strings.Contains(final, "beta-y") {
		t.Errorf("second edit's change lost (queue did not serialize): %q", final)
	}
}

func TestEditTool_SequentialSecondSeesFirst(t *testing.T) {
	// Spec: "Concurrent edits to the same file... the second edit sees
	// the first edit's result." Verified here with explicit sequencing
	// (no arrival-order race) so the assertion is deterministic.
	ops := newFakeEditOps()
	ops.files["/test/cwd/x.txt"] = []byte("alpha\nbeta\n")
	tool := NewEditTool(ops)

	if _, err := tool.Execute(context.Background(), makeEditCall(editArgs{
		Path:      "x.txt",
		OldString: "alpha",
		NewString: "alpha-edited",
	})); err != nil {
		t.Fatalf("first edit failed: %v", err)
	}
	if _, err := tool.Execute(context.Background(), makeEditCall(editArgs{
		Path:      "x.txt",
		OldString: "alpha-edited",
		NewString: "alpha-double",
	})); err != nil {
		t.Fatalf("second edit failed: %v", err)
	}
	final := string(ops.files["/test/cwd/x.txt"])
	if !strings.Contains(final, "alpha-double") {
		t.Errorf("second edit did not see first edit's result: %q", final)
	}
}

// ---- editTool.RenderCall / RenderResult ----

func TestEditTool_RenderCall_Plain(t *testing.T) {
	tool := NewEditTool(newFakeEditOps())
	got := tool.RenderCall(json.RawMessage(`{"path":"x.txt","oldString":"foo","newString":"bar"}`), PlainTheme())
	if !strings.Contains(got, "edit") || !strings.Contains(got, "x.txt") {
		t.Errorf("RenderCall missing tool name / path: %q", got)
	}
	if !strings.Contains(got, "foo") {
		t.Errorf("RenderCall should show oldString summary: %q", got)
	}
}

func TestEditTool_RenderResult_Text(t *testing.T) {
	tool := NewEditTool(newFakeEditOps())
	res := NewTextResult("Edited x.txt\n-old\n+new")
	got := tool.RenderResult(res, ColorTheme())
	if !strings.Contains(got, "Edited") {
		t.Errorf("RenderResult missing header: %q", got)
	}
}

func TestEditTool_RenderResult_Error(t *testing.T) {
	tool := NewEditTool(newFakeEditOps())
	res := NewErrorResult("file not unique")
	got := tool.RenderResult(res, ColorTheme())
	if !strings.Contains(got, "file not unique") {
		t.Errorf("RenderResult missing error text: %q", got)
	}
}

// ---- editTool — diff computation ----

func TestComputeUnifiedDiff_Replace(t *testing.T) {
	original := []byte("alpha\nbeta\n")
	updated := []byte("alpha\nBETA\n")
	got, err := computeUnifiedDiff("file.txt", original, updated)
	if err != nil {
		t.Fatalf("computeUnifiedDiff: %v", err)
	}
	if !strings.Contains(got, "-beta") {
		t.Errorf("diff missing -beta: %s", got)
	}
	if !strings.Contains(got, "+BETA") {
		t.Errorf("diff missing +BETA: %s", got)
	}
	if !strings.HasPrefix(strings.TrimSpace(got), "---") {
		t.Errorf("diff should start with ---: %s", got)
	}
}

func TestComputeUnifiedDiff_NoChange(t *testing.T) {
	original := []byte("same\n")
	got, err := computeUnifiedDiff("file.txt", original, original)
	if err != nil {
		t.Fatalf("computeUnifiedDiff: %v", err)
	}
	if got != "" {
		t.Errorf("expected empty diff for identical content: %q", got)
	}
}

func TestComputeUnifiedDiff_Insertion(t *testing.T) {
	original := []byte("a\nb\n")
	updated := []byte("a\nX\nb\n")
	got, _ := computeUnifiedDiff("f.txt", original, updated)
	if !strings.Contains(got, "+X") {
		t.Errorf("diff missing +X: %s", got)
	}
}

func TestComputeUnifiedDiff_Deletion(t *testing.T) {
	original := []byte("a\nX\nb\n")
	updated := []byte("a\nb\n")
	got, _ := computeUnifiedDiff("f.txt", original, updated)
	if !strings.Contains(got, "-X") {
		t.Errorf("diff missing -X: %s", got)
	}
}

func TestComputeUnifiedDiff_NoTrailingNewline(t *testing.T) {
	original := []byte("a\nb")
	updated := []byte("a\nB")
	got, _ := computeUnifiedDiff("f.txt", original, updated)
	if !strings.Contains(got, "No newline at end of file") {
		t.Errorf("diff should note missing newline: %s", got)
	}
}

// ---- end-to-end through real OS ----

func TestEditTool_RealFSReplace(t *testing.T) {
	dir := t.TempDir()
	tool := NewEditTool(OSEditOperations{})
	file := filepath.Join(dir, "x.txt")
	_ = os.WriteFile(file, []byte("hello\nworld\n"), 0644)
	res, err := tool.Execute(context.Background(), ToolCall{
		ID:   "real",
		Name: "edit",
		Args: jsonRawMustMarshal(editArgs{Path: file, OldString: "world", NewString: "WORLD"}),
		Cwd:  dir,
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	data, _ := os.ReadFile(file)
	if string(data) != "hello\nWORLD\n" {
		t.Errorf("file content = %q, want hello+WORLD", string(data))
	}
}

// jsonRawMustMarshal is a tiny helper for tests.
func jsonRawMustMarshal(v any) json.RawMessage {
	raw, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return raw
}
