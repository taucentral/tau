package tools

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeGrepOps is an in-memory GrepOperations for unit tests. The fake
// forces the pure-Go matcher by default (LookPath returns ErrNotFound);
// tests can flip lookPathResult to simulate rg being installed.
type fakeGrepOps struct {
	mu             sync.Mutex
	dirs           map[string]bool   // absolute path -> is directory
	files          map[string][]byte // absolute path -> content
	lookPathResult string            // if non-empty, LookPath returns this
	lookPathErr    error             // if set and result empty, LookPath returns this
	statErr        error             // if set, returned from IsDirectory
}

func newFakeGrepOps() *fakeGrepOps {
	return &fakeGrepOps{
		dirs:  map[string]bool{},
		files: map[string][]byte{},
	}
}

// addFile registers a regular file at absolutePath with the given content.
// Parent directories are marked as directories automatically.
func (f *fakeGrepOps) addFile(absolutePath, content string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.files[absolutePath] = []byte(content)
	// Mark parent directories.
	dir := filepath.Dir(absolutePath)
	for dir != "/" && dir != "." && dir != "" {
		f.dirs[dir] = true
		dir = filepath.Dir(dir)
	}
}

// addDir registers a directory (and ancestors) at absolutePath.
func (f *fakeGrepOps) addDir(absolutePath string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.dirs[absolutePath] = true
	dir := filepath.Dir(absolutePath)
	for dir != "/" && dir != "." && dir != "" {
		f.dirs[dir] = true
		dir = filepath.Dir(dir)
	}
}

func (f *fakeGrepOps) IsDirectory(p string) (bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.statErr != nil {
		return false, f.statErr
	}
	if f.dirs[p] {
		return true, nil
	}
	if _, ok := f.files[p]; ok {
		return false, nil
	}
	return false, &fs.PathError{Op: "stat", Path: p, Err: fs.ErrNotExist}
}

func (f *fakeGrepOps) ReadFile(p string) ([]byte, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if data, ok := f.files[p]; ok {
		out := make([]byte, len(data))
		copy(out, data)
		return out, nil
	}
	return nil, &fs.PathError{Op: "open", Path: p, Err: fs.ErrNotExist}
}

// WalkDir walks the fake tree by flattening files and dirs into a sorted
// list. Non-directory ancestors of files are synthesized as directory
// entries; this matches the contract that a real WalkDir would honor.
func (f *fakeGrepOps) WalkDir(root string, fn fs.WalkDirFunc) error {
	f.mu.Lock()
	knownDirs := map[string]bool{}
	for d := range f.dirs {
		knownDirs[d] = true
	}
	knownFiles := map[string][]byte{}
	for p, data := range f.files {
		knownFiles[p] = data
	}
	f.mu.Unlock()

	// Collect the set of paths to visit: the root, all directories, all
	// files, but only those equal to or under root.
	var allPaths []string
	allPaths = append(allPaths, root)
	for d := range knownDirs {
		if d == root || strings.HasPrefix(d, root+string(filepath.Separator)) {
			allPaths = append(allPaths, d)
		}
	}
	for p := range knownFiles {
		if p == root || strings.HasPrefix(p, root+string(filepath.Separator)) {
			allPaths = append(allPaths, p)
		}
	}
	sort.Strings(allPaths)

	for _, p := range allPaths {
		isDir := knownDirs[p]
		if !isDir {
			if _, ok := knownFiles[p]; !ok && p == root {
				isDir = true
			}
		}
		mode := fs.ModeDir.Perm()
		if !isDir {
			mode = 0644
		}
		d := fakeDirEntry{name: filepath.Base(p), isDir: isDir, mode: mode}
		if err := fn(p, d, nil); err != nil {
			if errors.Is(err, fs.SkipDir) {
				continue
			}
			return err
		}
	}
	return nil
}

func (f *fakeGrepOps) LookPath(name string) (string, error) {
	if f.lookPathResult != "" {
		return f.lookPathResult, nil
	}
	if f.lookPathErr != nil {
		return "", f.lookPathErr
	}
	return "", exec.ErrNotFound
}

// fakeDirEntry is a minimal fs.DirEntry for tests.
type fakeDirEntry struct {
	name  string
	isDir bool
	mode  fs.FileMode
	size  int64
	mtime time.Time
}

func (d fakeDirEntry) Name() string      { return d.name }
func (d fakeDirEntry) IsDir() bool       { return d.isDir }
func (d fakeDirEntry) Type() fs.FileMode { return d.mode }
func (d fakeDirEntry) Info() (fs.FileInfo, error) {
	return fakeFileInfo{name: d.name, isDir: d.isDir, size: d.size, mtime: d.mtime}, nil
}

func makeGrepCall(args any) ToolCall {
	raw, _ := json.Marshal(args)
	return ToolCall{
		ID:   "grep-call",
		Name: "grep",
		Args: raw,
		Cwd:  "/test/cwd",
	}
}

// ---- OSGrepOperations end-to-end ----

func TestOSGrepOperations_IsDirectory(t *testing.T) {
	ops := OSGrepOperations{}
	dir := t.TempDir()
	file := filepath.Join(dir, "x.txt")
	_ = os.WriteFile(file, []byte("hi"), 0644)

	got, err := ops.IsDirectory(dir)
	if err != nil {
		t.Fatalf("IsDirectory(%q): %v", dir, err)
	}
	if !got {
		t.Errorf("dir should be a directory")
	}
	got, err = ops.IsDirectory(file)
	if err != nil {
		t.Fatalf("IsDirectory(%q): %v", file, err)
	}
	if got {
		t.Errorf("file should not be a directory")
	}
	_, err = ops.IsDirectory(filepath.Join(dir, "missing"))
	if !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("missing path should return ErrNotExist, got %v", err)
	}
}

func TestOSGrepOperations_ReadFile(t *testing.T) {
	ops := OSGrepOperations{}
	dir := t.TempDir()
	p := filepath.Join(dir, "x.txt")
	_ = os.WriteFile(p, []byte("hello"), 0644)
	data, err := ops.ReadFile(p)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if string(data) != "hello" {
		t.Errorf("data = %q, want hello", string(data))
	}
}

func TestOSGrepOperations_WalkDir(t *testing.T) {
	ops := OSGrepOperations{}
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "a.txt"), []byte("a"), 0644)
	_ = os.Mkdir(filepath.Join(dir, "sub"), 0755)
	_ = os.WriteFile(filepath.Join(dir, "sub", "b.txt"), []byte("b"), 0644)

	var visited []string
	err := ops.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		visited = append(visited, filepath.ToSlash(strings.TrimPrefix(p, dir+string(filepath.Separator))))
		return nil
	})
	if err != nil {
		t.Fatalf("WalkDir: %v", err)
	}
	joined := strings.Join(visited, ",")
	for _, want := range []string{".", "a.txt", "sub", "sub/b.txt"} {
		if !strings.Contains(joined, want) {
			t.Errorf("walk should visit %q: %s", want, joined)
		}
	}
}

func TestOSGrepOperations_LookPath(t *testing.T) {
	ops := OSGrepOperations{}
	// "sh" exists on POSIX; "cmd" or similar on Windows. Just test that
	// the call doesn't panic and returns a sensible result for a name
	// that definitely does not exist.
	_, err := ops.LookPath("definitely-not-a-real-binary-xyz-12345")
	if err == nil {
		t.Errorf("expected error for fake binary name")
	}
}

// ---- parseRgJSON ----

func TestParseRgJSON_Empty(t *testing.T) {
	matches := parseRgJSON(nil)
	if len(matches) != 0 {
		t.Errorf("empty input = %d matches, want 0", len(matches))
	}
}

func TestParseRgJSON_SingleMatch(t *testing.T) {
	input := []byte(`{"type":"match","data":{"path":{"text":"foo.txt"},"line_number":3,"lines":{"text":"hello\n"}}}` + "\n")
	matches := parseRgJSON(input)
	if len(matches) != 1 {
		t.Fatalf("got %d matches, want 1", len(matches))
	}
	m := matches[0]
	if m.filePath != "foo.txt" {
		t.Errorf("filePath = %q, want foo.txt", m.filePath)
	}
	if m.lineNumber != 3 {
		t.Errorf("lineNumber = %d, want 3", m.lineNumber)
	}
	if !strings.Contains(m.lineText, "hello") {
		t.Errorf("lineText = %q, want contains hello", m.lineText)
	}
}

func TestParseRgJSON_MultipleMatches(t *testing.T) {
	var b strings.Builder
	for i := 1; i <= 3; i++ {
		b.WriteString(`{"type":"match","data":{"path":{"text":"f.txt"},"line_number":` + itoa(i) + `,"lines":{"text":"x"}}}` + "\n")
	}
	matches := parseRgJSON([]byte(b.String()))
	if len(matches) != 3 {
		t.Fatalf("got %d matches, want 3", len(matches))
	}
	if matches[2].lineNumber != 3 {
		t.Errorf("last match line = %d, want 3", matches[2].lineNumber)
	}
}

func TestParseRgJSON_NonMatchEventsSkipped(t *testing.T) {
	input := []byte(`{"type":"begin","data":{"path":{"text":"foo.txt"}}}` + "\n" +
		`{"type":"summary","data":{"elapsed_total":{"secs":0,"nanos":1000},"stats":{"matches":1}}}` + "\n" +
		`{"type":"match","data":{"path":{"text":"foo.txt"},"line_number":1,"lines":{"text":"hi"}}}` + "\n" +
		`{"type":"end","data":{"path":{"text":"foo.txt"}}}` + "\n")
	matches := parseRgJSON(input)
	if len(matches) != 1 {
		t.Errorf("got %d matches, want 1 (only type=match counts)", len(matches))
	}
}

func TestParseRgJSON_MalformedLinesSkipped(t *testing.T) {
	input := []byte("not json at all\n" +
		`{"type":"match","data":{"path":{"text":"f.txt"},"line_number":1,"lines":{"text":"x"}}}` + "\n" +
		"") // trailing blank
	matches := parseRgJSON(input)
	if len(matches) != 1 {
		t.Errorf("got %d matches, want 1 (bad lines skipped)", len(matches))
	}
}

func TestParseRgJSON_MatchMissingPathDropped(t *testing.T) {
	input := []byte(`{"type":"match","data":{"path":{"text":""},"line_number":1,"lines":{"text":"x"}}}` + "\n")
	matches := parseRgJSON(input)
	if len(matches) != 0 {
		t.Errorf("match with empty path should be dropped, got %d", len(matches))
	}
}

// ---- displayPath ----

func TestDisplayPath_DirectoryRoot(t *testing.T) {
	got := displayPath("/root/sub/file.txt", "/root/sub", true, "/root")
	if got != "file.txt" {
		t.Errorf("displayPath = %q, want file.txt", got)
	}
}

func TestDisplayPath_NestedUnderDir(t *testing.T) {
	got := displayPath("/root/sub/a/b.txt", "/root/sub", true, "/root")
	if got != "a/b.txt" {
		t.Errorf("displayPath = %q, want a/b.txt", got)
	}
}

func TestDisplayPath_FileRoot(t *testing.T) {
	got := displayPath("/root/x.txt", "/root/x.txt", false, "/root")
	if got != "x.txt" {
		t.Errorf("displayPath = %q, want x.txt", got)
	}
}

// ---- truncateGrepLine ----

func TestTruncateGrepLine_Short(t *testing.T) {
	flag := &linesTruncatedFlag{}
	got := truncateGrepLine("hello", flag)
	if got != "hello" {
		t.Errorf("got %q, want hello", got)
	}
	if flag.v {
		t.Errorf("flag should not be set for short line")
	}
}

func TestTruncateGrepLine_Long(t *testing.T) {
	flag := &linesTruncatedFlag{}
	long := strings.Repeat("x", GrepMaxLineLength+10)
	got := truncateGrepLine(long, flag)
	if !flag.v {
		t.Errorf("flag should be set for long line")
	}
	if len(got) > GrepMaxLineLength+10 {
		t.Errorf("got length %d, should be capped near %d", len(got), GrepMaxLineLength)
	}
	if !strings.HasSuffix(got, "…") {
		t.Errorf("long line should end with ellipsis: ...%q", got[len(got)-8:])
	}
}

// ---- globMatch ----

func TestGlobMatch_Simple(t *testing.T) {
	if !globMatch("*.go", "main.go") {
		t.Errorf("*.go should match main.go")
	}
	if globMatch("*.go", "main.js") {
		t.Errorf("*.go should not match main.js")
	}
}

func TestGlobMatch_WithSeparator(t *testing.T) {
	// Glob with a path separator matches against the whole path.
	if !globMatch("src/*.go", "src/main.go") {
		t.Errorf("src/*.go should match src/main.go")
	}
	if globMatch("src/*.go", "lib/main.go") {
		t.Errorf("src/*.go should not match lib/main.go")
	}
}

// ---- buildGrepNotices / appendNotices ----

func TestBuildGrepNotices_None(t *testing.T) {
	notices := buildGrepNotices(5, 100, "abc", "abc", false)
	if len(notices) != 0 {
		t.Errorf("no notices expected, got %v", notices)
	}
}

func TestBuildGrepNotices_LimitReached(t *testing.T) {
	notices := buildGrepNotices(100, 100, "abc", "abc", false)
	if len(notices) != 1 {
		t.Fatalf("got %d notices, want 1", len(notices))
	}
	if !strings.Contains(notices[0], "limit") {
		t.Errorf("expected limit notice: %q", notices[0])
	}
}

func TestBuildGrepNotices_ByteLimitReached(t *testing.T) {
	raw := strings.Repeat("a", 1000)
	capped := strings.Repeat("a", 500)
	notices := buildGrepNotices(5, 100, raw, capped, false)
	found := false
	for _, n := range notices {
		if strings.Contains(n, "KiB") {
			found = true
		}
	}
	if !found {
		t.Errorf("expected byte-limit notice: %v", notices)
	}
}

func TestBuildGrepNotices_LineTruncated(t *testing.T) {
	notices := buildGrepNotices(1, 100, "abc", "abc", true)
	found := false
	for _, n := range notices {
		if strings.Contains(n, "lines truncated") {
			found = true
		}
	}
	if !found {
		t.Errorf("expected line-truncated notice: %v", notices)
	}
}

func TestAppendNotices_Empty(t *testing.T) {
	got := appendNotices("output", nil)
	if got != "output" {
		t.Errorf("got %q, want output", got)
	}
}

func TestAppendNotices_Joined(t *testing.T) {
	got := appendNotices("output", []string{"one", "two"})
	if !strings.Contains(got, "[one. two]") {
		t.Errorf("expected joined notices: %q", got)
	}
}

// ---- splitLinesString ----

func TestSplitLinesString(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  []string
	}{
		{"empty", "", nil},
		{"plain", "a\nb\n", []string{"a", "b", ""}},
		{"no trailing newline", "a\nb", []string{"a", "b"}},
		{"crlf", "a\r\nb\r\n", []string{"a", "b", ""}},
		{"cr only", "a\rb\r", []string{"a", "b", ""}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := splitLinesString(tc.input)
			if len(got) != len(tc.want) {
				t.Errorf("len = %d, want %d (%v vs %v)", len(got), len(tc.want), got, tc.want)
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

// ---- grepTool.Execute — happy path via pure-Go fallback ----

func TestGrepTool_SingleFile(t *testing.T) {
	ops := newFakeGrepOps()
	ops.addFile("/test/cwd/x.txt", "alpha\nTODO: fix\nbeta\n")
	tool := NewGrepTool(ops)
	res, err := tool.Execute(context.Background(), makeGrepCall(grepArgs{
		Pattern: "TODO",
		Path:    "x.txt",
	}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	if !strings.Contains(got, "x.txt:2: TODO: fix") {
		t.Errorf("missing match annotation: %q", got)
	}
}

func TestGrepTool_Directory(t *testing.T) {
	ops := newFakeGrepOps()
	ops.addFile("/test/cwd/a.txt", "TODO one\n")
	ops.addFile("/test/cwd/b.txt", "nothing here\nTODO two\n")
	ops.addFile("/test/cwd/sub/c.txt", "TODO three\n")
	ops.addDir("/test/cwd/sub")
	tool := NewGrepTool(ops)
	res, _ := tool.Execute(context.Background(), makeGrepCall(grepArgs{
		Pattern: "TODO",
	}))
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	for _, want := range []string{"a.txt:1: TODO one", "b.txt:2: TODO two", "sub/c.txt:1: TODO three"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing match %q in output: %s", want, got)
		}
	}
}

func TestGrepTool_IgnoreCase(t *testing.T) {
	ops := newFakeGrepOps()
	ops.addFile("/test/cwd/x.txt", "Hello\nHELLO\nhello\n")
	tool := NewGrepTool(ops)
	res, _ := tool.Execute(context.Background(), makeGrepCall(grepArgs{
		Pattern:    "hello",
		IgnoreCase: true,
	}))
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	if strings.Count(got, "x.txt:") != 3 {
		t.Errorf("ignore-case should match all three, got: %s", got)
	}
}

func TestGrepTool_Literal(t *testing.T) {
	ops := newFakeGrepOps()
	ops.addFile("/test/cwd/x.txt", "a.b\na.c\n")
	tool := NewGrepTool(ops)
	res, _ := tool.Execute(context.Background(), makeGrepCall(grepArgs{
		Pattern: "a.b",
		Literal: true,
	}))
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	if !strings.Contains(got, "a.b") {
		t.Errorf("literal should match a.b: %s", got)
	}
	if strings.Contains(got, "a.c") {
		t.Errorf("literal should not match a.c (regex would): %s", got)
	}
}

func TestGrepTool_RegexMeta_LiteralFalse(t *testing.T) {
	ops := newFakeGrepOps()
	ops.addFile("/test/cwd/x.txt", "foo\nbar\nfooo\n")
	tool := NewGrepTool(ops)
	res, _ := tool.Execute(context.Background(), makeGrepCall(grepArgs{
		Pattern: "fo+",
	}))
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	// Should match foo and fooo but not bar.
	if strings.Count(got, "x.txt:") != 2 {
		t.Errorf("regex should match 2 lines, got: %s", got)
	}
}

func TestGrepTool_GlobFilter(t *testing.T) {
	ops := newFakeGrepOps()
	ops.addFile("/test/cwd/a.go", "TODO\n")
	ops.addFile("/test/cwd/b.txt", "TODO\n")
	tool := NewGrepTool(ops)
	res, _ := tool.Execute(context.Background(), makeGrepCall(grepArgs{
		Pattern: "TODO",
		Glob:    "*.go",
	}))
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	if !strings.Contains(got, "a.go:1: TODO") {
		t.Errorf("glob should include a.go: %s", got)
	}
	if strings.Contains(got, "b.txt") {
		t.Errorf("glob should exclude b.txt: %s", got)
	}
}

func TestGrepTool_Context(t *testing.T) {
	ops := newFakeGrepOps()
	ops.addFile("/test/cwd/x.txt", "l1\nl2\nMATCH\nl4\nl5\n")
	tool := NewGrepTool(ops)
	res, _ := tool.Execute(context.Background(), makeGrepCall(grepArgs{
		Pattern: "MATCH",
		Context: 1,
	}))
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	// Match line plus one context line on each side.
	if !strings.Contains(got, "x.txt:3: MATCH") {
		t.Errorf("missing match line: %s", got)
	}
	if !strings.Contains(got, "x.txt-2- l2") {
		t.Errorf("missing pre-context: %s", got)
	}
	if !strings.Contains(got, "x.txt-4- l4") {
		t.Errorf("missing post-context: %s", got)
	}
}

func TestGrepTool_LimitReached(t *testing.T) {
	ops := newFakeGrepOps()
	var b strings.Builder
	for i := 1; i <= 50; i++ {
		b.WriteString("TODO\n")
	}
	ops.addFile("/test/cwd/x.txt", b.String())
	tool := NewGrepTool(ops)
	res, _ := tool.Execute(context.Background(), makeGrepCall(grepArgs{
		Pattern: "TODO",
		Limit:   5,
	}))
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	if strings.Count(got, "x.txt:") != 5 {
		t.Errorf("limit=5 should produce 5 matches, got: %s", got)
	}
	if !strings.Contains(got, "limit reached") {
		t.Errorf("limit notice missing: %s", got)
	}
}

func TestGrepTool_NoMatches(t *testing.T) {
	ops := newFakeGrepOps()
	ops.addFile("/test/cwd/x.txt", "alpha\nbeta\n")
	tool := NewGrepTool(ops)
	res, _ := tool.Execute(context.Background(), makeGrepCall(grepArgs{
		Pattern: "gamma",
	}))
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	if got != "No matches found" {
		t.Errorf("got %q, want 'No matches found'", got)
	}
}

func TestGrepTool_LongLineTruncated(t *testing.T) {
	ops := newFakeGrepOps()
	long := strings.Repeat("x", GrepMaxLineLength+50)
	ops.addFile("/test/cwd/x.txt", long+"\n")
	tool := NewGrepTool(ops)
	res, _ := tool.Execute(context.Background(), makeGrepCall(grepArgs{
		Pattern: "x",
		Limit:   1,
	}))
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	if !strings.Contains(got, "lines truncated") {
		t.Errorf("expected line-truncated notice: %s", got)
	}
}

func TestGrepTool_GitignoreRespected(t *testing.T) {
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "keep.txt"), []byte("TODO keep\n"), 0644)
	_ = os.WriteFile(filepath.Join(dir, "ignore.txt"), []byte("TODO ignore\n"), 0644)
	_ = os.WriteFile(filepath.Join(dir, ".gitignore"), []byte("ignore.txt\n"), 0644)
	tool := NewGrepTool(OSGrepOperations{})
	res, _ := tool.Execute(context.Background(), ToolCall{
		ID:   "g",
		Name: "grep",
		Args: jsonRawMustMarshal(grepArgs{Pattern: "TODO"}),
		Cwd:  dir,
	})
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	if !strings.Contains(got, "keep.txt") {
		t.Errorf("keep.txt should be searched: %s", got)
	}
	if strings.Contains(got, "ignore.txt") {
		t.Errorf("ignore.txt should be skipped: %s", got)
	}
}

func TestGrepTool_DotGitSkipped(t *testing.T) {
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "keep.txt"), []byte("TODO keep\n"), 0644)
	_ = os.Mkdir(filepath.Join(dir, ".git"), 0755)
	_ = os.WriteFile(filepath.Join(dir, ".git", "config"), []byte("TODO secret\n"), 0644)
	tool := NewGrepTool(OSGrepOperations{})
	res, _ := tool.Execute(context.Background(), ToolCall{
		ID:   "g",
		Name: "grep",
		Args: jsonRawMustMarshal(grepArgs{Pattern: "TODO"}),
		Cwd:  dir,
	})
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	if strings.Contains(got, ".git") {
		t.Errorf(".git should be skipped: %s", got)
	}
}

// ---- grepTool.Execute — error paths ----

func TestGrepTool_MissingPattern(t *testing.T) {
	tool := NewGrepTool(newFakeGrepOps())
	res, _ := tool.Execute(context.Background(), makeGrepCall(grepArgs{Path: "."}))
	if !res.IsError {
		t.Errorf("expected IsError on missing pattern")
	}
}

func TestGrepTool_InvalidRegex(t *testing.T) {
	tool := NewGrepTool(newFakeGrepOps())
	res, _ := tool.Execute(context.Background(), makeGrepCall(grepArgs{Pattern: "[unclosed"}))
	if !res.IsError {
		t.Errorf("expected IsError on invalid regex")
	}
	if !strings.Contains(ResultText(res), "invalid pattern") {
		t.Errorf("error should mention invalid pattern: %s", ResultText(res))
	}
}

func TestGrepTool_PathNotFound(t *testing.T) {
	tool := NewGrepTool(newFakeGrepOps())
	res, _ := tool.Execute(context.Background(), makeGrepCall(grepArgs{
		Pattern: "x",
		Path:    "missing-dir",
	}))
	if !res.IsError {
		t.Errorf("expected IsError on missing path")
	}
	if !strings.Contains(ResultText(res), "not found") {
		t.Errorf("error should mention not found: %s", ResultText(res))
	}
}

func TestGrepTool_MissingCwd(t *testing.T) {
	tool := NewGrepTool(newFakeGrepOps())
	raw, _ := json.Marshal(grepArgs{Pattern: "x"})
	res, _ := tool.Execute(context.Background(), ToolCall{
		ID:   "g",
		Name: "grep",
		Args: raw,
		Cwd:  "",
	})
	if !res.IsError {
		t.Errorf("expected IsError on missing cwd")
	}
}

func TestGrepTool_InvalidJSON(t *testing.T) {
	tool := NewGrepTool(newFakeGrepOps())
	res, err := tool.Execute(context.Background(), ToolCall{
		ID:   "g",
		Name: "grep",
		Args: json.RawMessage(`{"pattern": invalid`),
		Cwd:  "/cwd",
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !res.IsError {
		t.Errorf("invalid JSON should yield IsError")
	}
}

func TestGrepTool_NegativeContext(t *testing.T) {
	tool := NewGrepTool(newFakeGrepOps())
	res, _ := tool.Execute(context.Background(), makeGrepCall(grepArgs{
		Pattern: "x",
		Context: -1,
	}))
	if !res.IsError {
		t.Errorf("negative context should yield IsError")
	}
}

func TestGrepTool_NegativeLimit(t *testing.T) {
	tool := NewGrepTool(newFakeGrepOps())
	res, _ := tool.Execute(context.Background(), makeGrepCall(grepArgs{
		Pattern: "x",
		Limit:   -1,
	}))
	if !res.IsError {
		t.Errorf("negative limit should yield IsError")
	}
}

func TestGrepTool_NilOpsDefaultsToOS(t *testing.T) {
	tool := NewGrepTool(nil)
	if gt, ok := tool.(*grepTool); !ok || gt.ops == nil {
		t.Errorf("nil ops should default to OSGrepOperations")
	}
}

func TestGrepTool_CtxCanceled(t *testing.T) {
	ops := newFakeGrepOps()
	ops.addFile("/test/cwd/x.txt", "TODO\n")
	tool := NewGrepTool(ops)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := tool.Execute(ctx, makeGrepCall(grepArgs{Pattern: "TODO"}))
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

// ---- grepTool.Name / Description / Parameters ----

func TestGrepTool_NameDescriptionParameters(t *testing.T) {
	tool := NewGrepTool(newFakeGrepOps())
	if tool.Name() != "grep" {
		t.Errorf("Name = %q, want grep", tool.Name())
	}
	if tool.Description() == "" {
		t.Errorf("Description should not be empty")
	}
	s := tool.Parameters()
	if s.Type != "object" {
		t.Errorf("Type = %q, want object", s.Type)
	}
	raw, _ := s.MarshalJSON()
	for _, field := range []string{`"pattern"`, `"path"`, `"glob"`, `"ignoreCase"`, `"literal"`, `"context"`, `"limit"`} {
		if !strings.Contains(string(raw), field) {
			t.Errorf("schema missing %s: %s", field, string(raw))
		}
	}
}

// ---- grepTool.RenderCall / RenderResult ----

func TestGrepTool_RenderCall_Plain(t *testing.T) {
	tool := NewGrepTool(newFakeGrepOps())
	got := tool.RenderCall(json.RawMessage(`{"pattern":"TODO","path":"src"}`), PlainTheme())
	if !strings.Contains(got, "grep") {
		t.Errorf("missing tool name: %q", got)
	}
	if !strings.Contains(got, "/TODO/") {
		t.Errorf("missing pattern: %q", got)
	}
	if !strings.Contains(got, "src") {
		t.Errorf("missing path: %q", got)
	}
}

func TestGrepTool_RenderCall_Full(t *testing.T) {
	tool := NewGrepTool(newFakeGrepOps())
	got := tool.RenderCall(json.RawMessage(`{"pattern":"x","glob":"*.go","limit":50,"ignoreCase":true,"literal":true}`), ColorTheme())
	for _, marker := range []string{"*.go", "limit", "ignore-case", "literal"} {
		if !strings.Contains(got, marker) {
			t.Errorf("missing %q in: %q", marker, got)
		}
	}
}

func TestGrepTool_RenderResult_Text(t *testing.T) {
	tool := NewGrepTool(newFakeGrepOps())
	res := NewTextResult("x.txt:1: hello")
	got := tool.RenderResult(res, PlainTheme())
	if !strings.Contains(got, "hello") {
		t.Errorf("missing text: %q", got)
	}
}

func TestGrepTool_RenderResult_Error(t *testing.T) {
	tool := NewGrepTool(newFakeGrepOps())
	res := NewErrorResult("invalid pattern")
	got := tool.RenderResult(res, ColorTheme())
	if !strings.Contains(got, "invalid pattern") {
		t.Errorf("missing error text: %q", got)
	}
}

// ---- ignoreStack ----

func TestIgnoreStack_NoMatches(t *testing.T) {
	s := newIgnoreStack()
	if s.matchesFile("anything") {
		t.Errorf("empty stack should not match")
	}
}

func TestIgnoreStack_AddAndMatch(t *testing.T) {
	dir := t.TempDir()
	ignorePath := filepath.Join(dir, ".gitignore")
	_ = os.WriteFile(ignorePath, []byte("*.tmp\n"), 0644)
	ig, ok := tryReadGitignore(ignorePath)
	if !ok {
		t.Fatalf("tryReadGitignore returned !ok")
	}
	s := newIgnoreStack()
	s.push(dir)
	s.add(ig, dir)
	if !s.matchesFile("foo.tmp") {
		t.Errorf("foo.tmp should be ignored")
	}
	if s.matchesFile("foo.txt") {
		t.Errorf("foo.txt should not be ignored")
	}
}

func TestTryReadGitignore_MissingFile(t *testing.T) {
	ig, ok := tryReadGitignore("/definitely-not-a-path-xyz/.gitignore")
	if ok || ig != nil {
		t.Errorf("missing .gitignore should return (nil, false)")
	}
}

// ---- end-to-end through real OS (pure-Go path forced) ----

func TestGrepTool_RealFS_PureGoFallback(t *testing.T) {
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "a.txt"), []byte("alpha TODO\nbeta\n"), 0644)
	_ = os.WriteFile(filepath.Join(dir, "b.txt"), []byte("gamma TODO\n"), 0644)
	// Force pure-Go path by wrapping OS ops with a LookPath that fails.
	ops := &rgAbsentOps{inner: OSGrepOperations{}}
	tool := NewGrepTool(ops)
	res, err := tool.Execute(context.Background(), ToolCall{
		ID:   "g",
		Name: "grep",
		Args: jsonRawMustMarshal(grepArgs{Pattern: "TODO"}),
		Cwd:  dir,
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	if strings.Count(got, "TODO") != 2 {
		t.Errorf("expected 2 TODO matches, got: %s", got)
	}
}

// rgAbsentOps wraps OSGrepOperations but claims rg is never installed.
type rgAbsentOps struct {
	inner OSGrepOperations
}

func (o *rgAbsentOps) IsDirectory(p string) (bool, error) { return o.inner.IsDirectory(p) }
func (o *rgAbsentOps) ReadFile(p string) ([]byte, error)  { return o.inner.ReadFile(p) }
func (o *rgAbsentOps) WalkDir(root string, fn fs.WalkDirFunc) error {
	return o.inner.WalkDir(root, fn)
}
func (o *rgAbsentOps) LookPath(name string) (string, error) { return "", exec.ErrNotFound }

// ---- end-to-end through real OS (ripgrep path, when installed) ----

func TestGrepTool_RealFS_Ripgrep(t *testing.T) {
	if _, err := exec.LookPath("rg"); err != nil {
		t.Skip("rg not on PATH; skipping ripgrep integration test")
	}
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "a.txt"), []byte("alpha TODO\nbeta\n"), 0644)
	_ = os.WriteFile(filepath.Join(dir, "b.txt"), []byte("gamma TODO\n"), 0644)
	tool := NewGrepTool(OSGrepOperations{})
	res, err := tool.Execute(context.Background(), ToolCall{
		ID:   "g",
		Name: "grep",
		Args: jsonRawMustMarshal(grepArgs{Pattern: "TODO"}),
		Cwd:  dir,
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	if strings.Count(got, "TODO") != 2 {
		t.Errorf("expected 2 TODO matches via ripgrep, got: %s", got)
	}
}

func TestGrepTool_RealFS_Ripgrep_NoMatches(t *testing.T) {
	if _, err := exec.LookPath("rg"); err != nil {
		t.Skip("rg not on PATH; skipping ripgrep integration test")
	}
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "a.txt"), []byte("alpha\n"), 0644)
	tool := NewGrepTool(OSGrepOperations{})
	res, _ := tool.Execute(context.Background(), ToolCall{
		ID:   "g",
		Name: "grep",
		Args: jsonRawMustMarshal(grepArgs{Pattern: "ZZZ"}),
		Cwd:  dir,
	})
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	if ResultText(res) != "No matches found" {
		t.Errorf("got %q, want 'No matches found'", ResultText(res))
	}
}

// itoa is a tiny helper to avoid strconv in test data assembly.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
