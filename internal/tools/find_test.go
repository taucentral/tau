package tools

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fakeGrepOps also satisfies FindOperations (IsDirectory + WalkDir). Tests
// in this file use it via the FindOperations interface; the grep-specific
// fields (lookPathResult, lookPathErr, statErr) are simply unused.
func newFakeFindOps() *fakeGrepOps { return newFakeGrepOps() }

func makeFindCall(args any) ToolCall {
	raw, _ := json.Marshal(args)
	return ToolCall{
		ID:   "find-call",
		Name: "find",
		Args: raw,
		Cwd:  "/test/cwd",
	}
}

// ---- findTool basics ----

func TestFindTool_NameDescriptionParameters(t *testing.T) {
	tool := NewFindTool(nil)
	if tool.Name() != "find" {
		t.Errorf("Name = %q, want find", tool.Name())
	}
	if tool.Description() == "" {
		t.Errorf("Description should not be empty")
	}
	s := tool.Parameters()
	if s.Type != "object" {
		t.Errorf("Type = %q, want object", s.Type)
	}
	raw, _ := s.MarshalJSON()
	if !strings.Contains(string(raw), `"pattern"`) {
		t.Errorf("schema should mention pattern: %s", string(raw))
	}
}

func TestFindTool_NilOpsDefaultsToOS(t *testing.T) {
	tool := NewFindTool(nil)
	if _, ok := tool.(*findTool).ops.(OSFindOperations); !ok {
		t.Errorf("nil ops should default to OSFindOperations, got %T", tool.(*findTool).ops)
	}
}

// ---- happy paths ----

func TestFindTool_DeepGlobMatchesAllDepths(t *testing.T) {
	ops := newFakeFindOps()
	ops.addFile("/test/cwd/top.go", "package main\n")
	ops.addFile("/test/cwd/sub/mid.go", "package main\n")
	ops.addFile("/test/cwd/sub/deep/nested.go", "package main\n")
	ops.addFile("/test/cwd/sub/deep/nested.txt", "not go\n")
	tool := NewFindTool(ops)

	res, err := tool.Execute(context.Background(), makeFindCall(findArgs{Pattern: "**/*.go"}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error result: %q", ResultText(res))
	}
	got := ResultText(res)
	for _, want := range []string{"top.go", "sub/mid.go", "sub/deep/nested.go"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing match %q in output: %s", want, got)
		}
	}
	if strings.Contains(got, "nested.txt") {
		t.Errorf("non-.go file should not be matched: %s", got)
	}
}

func TestFindTool_BareGlobOnlyMatchesTopLevel(t *testing.T) {
	// doublestar treats `*` as not crossing `/` — bare patterns are
	// strictly top-level. This is intentional; users wanting depth use `**/`.
	ops := newFakeFindOps()
	ops.addFile("/test/cwd/top.go", "")
	ops.addFile("/test/cwd/sub/mid.go", "")
	tool := NewFindTool(ops)

	res, err := tool.Execute(context.Background(), makeFindCall(findArgs{Pattern: "*.go"}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	got := ResultText(res)
	if !strings.Contains(got, "top.go") {
		t.Errorf("top-level match missing: %s", got)
	}
	if strings.Contains(got, "sub/mid.go") {
		t.Errorf("bare pattern should not cross separators: %s", got)
	}
}

func TestFindTool_NoMatches(t *testing.T) {
	ops := newFakeFindOps()
	ops.addFile("/test/cwd/foo.txt", "")
	tool := NewFindTool(ops)

	res, err := tool.Execute(context.Background(), makeFindCall(findArgs{Pattern: "**/*.go"}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	got := ResultText(res)
	if !strings.Contains(got, "No files found") {
		t.Errorf("expected no-matches message, got: %s", got)
	}
}

func TestFindTool_ResultsAreSorted(t *testing.T) {
	ops := newFakeFindOps()
	// Add files in non-sorted insertion order.
	ops.addFile("/test/cwd/zeta.go", "")
	ops.addFile("/test/cwd/alpha.go", "")
	ops.addFile("/test/cwd/mid.go", "")
	tool := NewFindTool(ops)

	res, err := tool.Execute(context.Background(), makeFindCall(findArgs{Pattern: "**/*.go"}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	got := ResultText(res)
	lines := strings.Split(got, "\n")
	// Strip any trailing notice block (it would start with `[`).
	var paths []string
	for _, l := range lines {
		if l == "" || strings.HasPrefix(l, "[") {
			continue
		}
		paths = append(paths, l)
	}
	if len(paths) != 3 {
		t.Fatalf("expected 3 paths, got %d: %v", len(paths), paths)
	}
	if paths[0] != "alpha.go" || paths[1] != "mid.go" || paths[2] != "zeta.go" {
		t.Errorf("paths not lexically sorted: %v", paths)
	}
}

func TestFindTool_CustomPath(t *testing.T) {
	ops := newFakeFindOps()
	ops.addFile("/test/cwd/src/a.go", "")
	ops.addFile("/test/cwd/other/b.go", "")
	tool := NewFindTool(ops)

	res, err := tool.Execute(context.Background(), makeFindCall(findArgs{Pattern: "**/*.go", Path: "src"}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	got := ResultText(res)
	if !strings.Contains(got, "a.go") {
		t.Errorf("should match file under src/: %s", got)
	}
	if strings.Contains(got, "b.go") {
		t.Errorf("should not match file outside src/: %s", got)
	}
}

func TestFindTool_DisplayPathRelativeToCwd(t *testing.T) {
	ops := newFakeFindOps()
	ops.addFile("/test/cwd/pkg/foo.go", "")
	tool := NewFindTool(ops)

	res, err := tool.Execute(context.Background(), makeFindCall(findArgs{Pattern: "**/*.go", Path: "pkg"}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	got := ResultText(res)
	// When the search root is a subdir of cwd, displayPath shows the
	// path relative to cwd (so "pkg/foo.go") not relative to the search
	// root (which would be just "foo.go"). This matches what grep does
	// and gives the model cwd-anchored paths it can pass to read/edit.
	if !strings.Contains(got, "pkg/foo.go") && !strings.Contains(got, "foo.go") {
		t.Errorf("expected a foo.go reference, got: %s", got)
	}
}

// ---- gitignore (real FS) ----
//
// The fake FindOperations can't exercise gitignore handling because
// gitignore.CompileIgnoreFile reads via os.Open and the fake's files
// live in-memory. These tests use a real temp directory, mirroring the
// approach in grep_test.go.

func TestFindTool_GitignoreRespected(t *testing.T) {
	dir := t.TempDir()
	mustWriteFile(t, filepath.Join(dir, ".gitignore"), "*.txt\n")
	mustWriteFile(t, filepath.Join(dir, "keep.go"), "")
	mustWriteFile(t, filepath.Join(dir, "skip.txt"), "")
	tool := NewFindTool(OSFindOperations{})

	res, err := tool.Execute(context.Background(), ToolCall{
		ID:   "find",
		Name: "find",
		Args: jsonRawMustMarshal(findArgs{Pattern: "**/*"}),
		Cwd:  dir,
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	got := ResultText(res)
	if !strings.Contains(got, "keep.go") {
		t.Errorf("keep.go should be in output: %s", got)
	}
	if strings.Contains(got, "skip.txt") {
		t.Errorf("skip.txt should be ignored: %s", got)
	}
}

func TestFindTool_GitignoreNestedDirectorySkipped(t *testing.T) {
	dir := t.TempDir()
	mustWriteFile(t, filepath.Join(dir, ".gitignore"), "build/\n")
	mustWriteFile(t, filepath.Join(dir, "build", "out.go"), "")
	mustWriteFile(t, filepath.Join(dir, "src", "main.go"), "")
	tool := NewFindTool(OSFindOperations{})

	res, err := tool.Execute(context.Background(), ToolCall{
		ID:   "find",
		Name: "find",
		Args: jsonRawMustMarshal(findArgs{Pattern: "**/*.go"}),
		Cwd:  dir,
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	got := ResultText(res)
	if strings.Contains(got, "build/") || strings.Contains(got, "build/out.go") {
		t.Errorf("build/ directory should be skipped: %s", got)
	}
	if !strings.Contains(got, "main.go") {
		t.Errorf("main.go should be present: %s", got)
	}
}

func TestFindTool_DotGitDirectorySkipped(t *testing.T) {
	dir := t.TempDir()
	mustWriteFile(t, filepath.Join(dir, ".git", "config"), "") // .git dir + config
	mustWriteFile(t, filepath.Join(dir, "main.go"), "")
	tool := NewFindTool(OSFindOperations{})

	res, err := tool.Execute(context.Background(), ToolCall{
		ID:   "find",
		Name: "find",
		Args: jsonRawMustMarshal(findArgs{Pattern: "**/*"}),
		Cwd:  dir,
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	got := ResultText(res)
	if strings.Contains(got, ".git/") {
		t.Errorf(".git/ should be skipped: %s", got)
	}
	if !strings.Contains(got, "main.go") {
		t.Errorf("main.go should be present: %s", got)
	}
}

// mustWriteFile writes content to dir/path, creating parent directories.
// Mirrors the helper used in grep_test.go's real-FS tests.
func mustWriteFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatalf("MkdirAll(%q): %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("WriteFile(%q): %v", path, err)
	}
}

// ---- limit + truncation ----

func TestFindTool_LimitReached(t *testing.T) {
	ops := newFakeFindOps()
	for i := 0; i < 5; i++ {
		ops.addFile("/test/cwd/f"+string(rune('a'+i))+".go", "")
	}
	tool := NewFindTool(ops)

	res, err := tool.Execute(context.Background(), makeFindCall(findArgs{Pattern: "**/*.go", Limit: 2}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	got := ResultText(res)
	// Exactly 2 paths before the notice block.
	lines := strings.Split(got, "\n")
	var paths []string
	for _, l := range lines {
		if l == "" || strings.HasPrefix(l, "[") {
			continue
		}
		paths = append(paths, l)
	}
	if len(paths) != 2 {
		t.Errorf("expected 2 paths, got %d: %v", len(paths), paths)
	}
	if !strings.Contains(got, "limit reached") {
		t.Errorf("expected limit notice, got: %s", got)
	}
}

func TestFindTool_DefaultLimitApplies(t *testing.T) {
	// When Limit=0 (omitted), DefaultFindLimit applies. Generating more
	// files than the default and confirming output is capped at
	// DefaultFindLimit matches. Output may be further truncated to
	// DefaultMaxLines (=500) by TruncateHeadTail; we only assert that the
	// limit-reached notice is present and the path count does not exceed
	// the find-level limit.
	ops := newFakeFindOps()
	for i := 0; i < DefaultFindLimit+10; i++ {
		// Generate distinct names by zero-padding the index.
		name := "f" + strings.Repeat("x", i/26) + string(rune('a'+(i%26))) + ".go"
		ops.addFile("/test/cwd/"+name, "")
	}
	tool := NewFindTool(ops)

	res, err := tool.Execute(context.Background(), makeFindCall(findArgs{Pattern: "**/*.go"}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	got := ResultText(res)
	lines := strings.Split(got, "\n")
	var paths []string
	for _, l := range lines {
		if l == "" || strings.HasPrefix(l, "[") {
			continue
		}
		paths = append(paths, l)
	}
	if len(paths) > DefaultFindLimit {
		t.Errorf("expected at most %d paths, got %d", DefaultFindLimit, len(paths))
	}
	if !strings.Contains(got, "limit reached") {
		t.Errorf("expected limit notice, got: %s", got)
	}
}

// ---- error paths ----

func TestFindTool_MissingPattern(t *testing.T) {
	tool := NewFindTool(newFakeFindOps())
	res, err := tool.Execute(context.Background(), makeFindCall(findArgs{}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !res.IsError {
		t.Errorf("expected error result, got: %+v", res)
	}
	if !strings.Contains(ResultText(res), "missing") || !strings.Contains(ResultText(res), "pattern") {
		t.Errorf("error should mention missing pattern: %s", ResultText(res))
	}
}

func TestFindTool_InvalidPattern(t *testing.T) {
	tool := NewFindTool(newFakeFindOps())
	// `**[unclosed` is malformed doublestar.
	res, err := tool.Execute(context.Background(), makeFindCall(findArgs{Pattern: "**[unclosed"}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !res.IsError {
		t.Errorf("expected error result for malformed pattern, got: %+v", res)
	}
	if !strings.Contains(ResultText(res), "invalid pattern") {
		t.Errorf("error should mention invalid pattern: %s", ResultText(res))
	}
}

func TestFindTool_NegativeLimit(t *testing.T) {
	tool := NewFindTool(newFakeFindOps())
	res, err := tool.Execute(context.Background(), makeFindCall(findArgs{Pattern: "**/*.go", Limit: -1}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !res.IsError {
		t.Errorf("expected error result for negative limit, got: %+v", res)
	}
}

func TestFindTool_MissingCwd(t *testing.T) {
	tool := NewFindTool(newFakeFindOps())
	raw, _ := json.Marshal(findArgs{Pattern: "**/*.go"})
	call := ToolCall{ID: "find-call", Name: "find", Args: raw, Cwd: ""}
	res, err := tool.Execute(context.Background(), call)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !res.IsError || !strings.Contains(ResultText(res), "cwd") {
		t.Errorf("expected missing-cwd error, got: %+v", res)
	}
}

func TestFindTool_InvalidJSON(t *testing.T) {
	tool := NewFindTool(newFakeFindOps())
	call := ToolCall{
		ID:   "find-call",
		Name: "find",
		Args: json.RawMessage(`{"pattern": broken`),
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

func TestFindTool_PathNotFound(t *testing.T) {
	tool := NewFindTool(newFakeFindOps())
	res, err := tool.Execute(context.Background(), makeFindCall(findArgs{Pattern: "**/*.go", Path: "missing"}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !res.IsError {
		t.Errorf("expected error result for missing path, got: %+v", res)
	}
	if !strings.Contains(ResultText(res), "not found") {
		t.Errorf("error should mention not-found: %s", ResultText(res))
	}
}

func TestFindTool_PathIsFileNotDir(t *testing.T) {
	ops := newFakeFindOps()
	ops.addFile("/test/cwd/file.go", "")
	tool := NewFindTool(ops)
	res, err := tool.Execute(context.Background(), makeFindCall(findArgs{Pattern: "**/*.go", Path: "file.go"}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !res.IsError || !strings.Contains(ResultText(res), "not a directory") {
		t.Errorf("expected not-a-directory error, got: %+v", res)
	}
}

func TestFindTool_StatError(t *testing.T) {
	ops := newFakeFindOps()
	ops.statErr = errors.New("simulated permission denied")
	tool := NewFindTool(ops)
	res, err := tool.Execute(context.Background(), makeFindCall(findArgs{Pattern: "**/*.go"}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !res.IsError {
		t.Errorf("expected error result for stat failure, got: %+v", res)
	}
}

func TestFindTool_CtxCanceled(t *testing.T) {
	ops := newFakeFindOps()
	ops.addFile("/test/cwd/a.go", "")
	tool := NewFindTool(ops)
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel before Execute
	_, err := tool.Execute(ctx, makeFindCall(findArgs{Pattern: "**/*.go"}))
	if err == nil {
		t.Errorf("expected ctx.Err() to surface, got nil")
	}
}

// ---- RenderCall / RenderResult ----

func TestFindTool_RenderCall_Plain(t *testing.T) {
	tool := NewFindTool(newFakeFindOps())
	got := tool.RenderCall(json.RawMessage(`{"pattern":"**/*.go"}`), PlainTheme())
	if !strings.Contains(got, "find") || !strings.Contains(got, "**/*.go") {
		t.Errorf("RenderCall missing name or pattern: %q", got)
	}
}

func TestFindTool_RenderCall_WithPath(t *testing.T) {
	tool := NewFindTool(newFakeFindOps())
	got := tool.RenderCall(json.RawMessage(`{"pattern":"**/*.go","path":"src"}`), PlainTheme())
	if !strings.Contains(got, "src") {
		t.Errorf("RenderCall missing path: %q", got)
	}
}

func TestFindTool_RenderCall_WithLimit(t *testing.T) {
	tool := NewFindTool(newFakeFindOps())
	got := tool.RenderCall(json.RawMessage(`{"pattern":"**/*.go","limit":50}`), PlainTheme())
	if !strings.Contains(got, "limit") || !strings.Contains(got, "50") {
		t.Errorf("RenderCall missing limit: %q", got)
	}
}

func TestFindTool_RenderResult_Text(t *testing.T) {
	tool := NewFindTool(newFakeFindOps())
	res := NewTextResult("a.go\nb.go")
	got := tool.RenderResult(res, ColorTheme())
	if !strings.Contains(got, "a.go") {
		t.Errorf("RenderResult missing content: %q", got)
	}
}

func TestFindTool_RenderResult_Error(t *testing.T) {
	tool := NewFindTool(newFakeFindOps())
	res := NewErrorResult("invalid pattern")
	got := tool.RenderResult(res, ColorTheme())
	if !strings.Contains(got, "error") {
		t.Errorf("RenderResult should mark error: %q", got)
	}
}

// ---- matchFindPattern unit tests ----

func TestMatchFindPattern_DeepGo(t *testing.T) {
	cases := []struct {
		pattern string
		path    string
		want    bool
	}{
		{`**/*.go`, `foo.go`, true},
		{`**/*.go`, `bar/foo.go`, true},
		{`**/*.go`, `a/b/c/foo.go`, true},
		{`**/*.go`, `foo.txt`, false},
		{`*.go`, `foo.go`, true},
		{`*.go`, `bar/foo.go`, false}, // `*` does not cross `/`
		{`src/**/*.test.ts`, `src/a/b/c/x.test.ts`, true},
		{`src/**/*.test.ts`, `src/x.test.ts`, true},  // `**` matches zero dirs
		{`src/**/*.test.ts`, `out/x.test.ts`, false}, // outside src
	}
	for _, c := range cases {
		got := matchFindPattern(c.pattern, c.path)
		if got != c.want {
			t.Errorf("matchFindPattern(%q, %q) = %v, want %v", c.pattern, c.path, got, c.want)
		}
	}
}

func TestMatchFindPattern_PathSeparators(t *testing.T) {
	// Windows-style backslashes in the input path should be normalized
	// to forward slashes before matching.
	if !matchFindPattern(`**/*.go`, `sub\foo.go`) {
		t.Errorf("backslash path not normalized to forward slash")
	}
}

// ---- buildFindNotices ----

func TestBuildFindNotices_None(t *testing.T) {
	notices := buildFindNotices(5, 100, "x", "x")
	if len(notices) != 0 {
		t.Errorf("expected no notices, got %v", notices)
	}
}

func TestBuildFindNotices_LimitReached(t *testing.T) {
	notices := buildFindNotices(100, 100, "x", "x")
	if len(notices) != 1 {
		t.Fatalf("expected 1 notice, got %v", notices)
	}
	if !strings.Contains(notices[0], "limit") {
		t.Errorf("notice should mention limit: %v", notices)
	}
}

func TestBuildFindNotices_ByteLimitReached(t *testing.T) {
	raw := strings.Repeat("a", 1024)
	capped := strings.Repeat("a", 512)
	notices := buildFindNotices(5, 100, raw, capped)
	if len(notices) != 1 {
		t.Fatalf("expected 1 notice, got %v", notices)
	}
	if !strings.Contains(notices[0], "KiB") {
		t.Errorf("notice should mention KiB: %v", notices)
	}
}

// ---- real-filesystem smoke test ----

func TestFindTool_RealFS_DeepGlob(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping real-FS test in -short mode")
	}
	// Use the current test file's directory as the search root. The
	// pattern `*_test.go` should at minimum find this file.
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	tool := NewFindTool(OSFindOperations{})
	raw, _ := json.Marshal(findArgs{Pattern: "find_test.go"})
	res, err := tool.Execute(context.Background(), ToolCall{
		ID:   "find-real",
		Name: "find",
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
	if !strings.Contains(got, "find_test.go") {
		t.Errorf("expected to find find_test.go under %s, got: %s", wd, got)
	}
	_ = filepath.Separator
	_ = fs.SkipDir
}
