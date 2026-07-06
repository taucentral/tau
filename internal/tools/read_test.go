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

	"github.com/taucentral/tau/internal/llm"
)

// fakeReadOps is an in-memory ReadOperations for unit tests.
type fakeReadOps struct {
	files     map[string][]byte // absolute path → content
	accessErr error             // if set, Access always returns this
	readErr   error             // if set, ReadFile always returns this
	mimeErr   error             // if set, DetectImageMimeType always returns this
}

func newFakeReadOps() *fakeReadOps {
	return &fakeReadOps{files: map[string][]byte{}}
}

func (f *fakeReadOps) ReadFile(p string) ([]byte, error) {
	if f.readErr != nil {
		return nil, f.readErr
	}
	if data, ok := f.files[p]; ok {
		return data, nil
	}
	return nil, &fs.PathError{Op: "open", Path: p, Err: fs.ErrNotExist}
}

func (f *fakeReadOps) Access(p string) error {
	if f.accessErr != nil {
		return f.accessErr
	}
	if _, ok := f.files[p]; !ok {
		return &fs.PathError{Op: "open", Path: p, Err: fs.ErrNotExist}
	}
	return nil
}

func (f *fakeReadOps) DetectImageMimeType(p string) (string, error) {
	if f.mimeErr != nil {
		return "", f.mimeErr
	}
	data, ok := f.files[p]
	if !ok {
		return "", nil
	}
	return DetectImageMimeTypeFromBytes(data), nil
}

// makeCall builds a ToolCall with a fixed cwd.
func makeCall(args any) ToolCall {
	raw, _ := json.Marshal(args)
	return ToolCall{
		ID:   "test-call",
		Name: "read",
		Args: raw,
		Cwd:  "/test/cwd",
	}
}

// ResultText concatenates all TextContent blocks in res. Image blocks and
// other types are skipped. Used by tests that want to assert on the text
// payload of a single-block TextContent result.
func ResultText(res ToolResult) string {
	var b strings.Builder
	for _, blk := range res.Content {
		if tc, ok := blk.(llm.TextContent); ok {
			b.WriteString(tc.Text)
		}
	}
	return b.String()
}

// ---- OSReadOperations ----

func TestOSReadOperations_ReadFile(t *testing.T) {
	ops := OSReadOperations{}
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
}

func TestOSReadOperations_Access(t *testing.T) {
	ops := OSReadOperations{}
	dir := t.TempDir()
	p := filepath.Join(dir, "exists.txt")
	_ = os.WriteFile(p, []byte("x"), 0600)
	if err := ops.Access(p); err != nil {
		t.Errorf("Access on existing file: %v", err)
	}
	if err := ops.Access(filepath.Join(dir, "nope.txt")); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("Access on missing file: err=%v, want ErrNotExist", err)
	}
}

func TestOSReadOperations_DetectImageMimeType(t *testing.T) {
	ops := OSReadOperations{}
	dir := t.TempDir()
	png := filepath.Join(dir, "x.png")
	_ = os.WriteFile(png, []byte{
		0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
		0x00, 0x00, 0x00, 0x0d, 'I', 'H', 'D', 'R',
	}, 0600)
	mt, err := ops.DetectImageMimeType(png)
	if err != nil {
		t.Fatalf("DetectImageMimeType: %v", err)
	}
	if mt != "image/png" {
		t.Errorf("mt = %q, want image/png", mt)
	}
	txt := filepath.Join(dir, "x.txt")
	_ = os.WriteFile(txt, []byte("plain text"), 0600)
	mt2, _ := ops.DetectImageMimeType(txt)
	if mt2 != "" {
		t.Errorf("mt = %q, want empty", mt2)
	}
}

// ---- DetectImageMimeTypeFromBytes ----

func TestDetectImageMimeTypeFromBytes(t *testing.T) {
	cases := []struct {
		name string
		buf  []byte
		want string
	}{
		{"jpeg", []byte{0xff, 0xd8, 0xff, 0xe0}, "image/jpeg"},
		{"jpeg2000 excluded", []byte{0xff, 0xd8, 0xff, 0xf7}, ""},
		{"png valid", []byte{
			0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
			0x00, 0x00, 0x00, 0x0d, 'I', 'H', 'D', 'R',
		}, "image/png"},
		{"png corrupted (no IHDR)", []byte{
			0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
			0x00, 0x00, 0x00, 0x0d, 'X', 'X', 'X', 'X',
		}, ""},
		{"gif", []byte{'G', 'I', 'F', '8', '7', 'a'}, "image/gif"},
		{"webp", []byte{
			'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'E', 'B', 'P',
		}, "image/webp"},
		{"text", []byte("hello world"), ""},
		{"empty", nil, ""},
		{"short jpeg header", []byte{0xff, 0xd8}, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := DetectImageMimeTypeFromBytes(tc.buf)
			if got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

// ---- IsBinary ----

func TestIsBinary(t *testing.T) {
	if !IsBinary([]byte("text\x00with nul")) {
		t.Errorf("text with NUL should be binary")
	}
	if IsBinary([]byte("plain text no nul bytes here")) {
		t.Errorf("text without NUL should not be binary")
	}
	if IsBinary(nil) {
		t.Errorf("nil should not be binary")
	}
	long := strings.Repeat("x", 10_000)
	if IsBinary([]byte(long)) {
		t.Errorf("long text should not be binary")
	}
	// NUL past the 8 KiB window is not detected.
	large := []byte(strings.Repeat("x", 9000))
	large[8500] = 0
	if IsBinary(large) {
		t.Errorf("NUL past 8KiB window should not be detected")
	}
	// NUL within the window is detected.
	large2 := []byte(strings.Repeat("x", 9000))
	large2[100] = 0
	if !IsBinary(large2) {
		t.Errorf("NUL within 8KiB window should be detected")
	}
}

// ---- ResolveWithinCwd ----

func TestResolveWithinCwd(t *testing.T) {
	abs, err := ResolveWithinCwd("/abs/path.txt", "/cwd")
	if err != nil || abs != "/abs/path.txt" {
		t.Errorf("absolute path: got %q, %v", abs, err)
	}
	rel, err := ResolveWithinCwd("rel.txt", "/cwd")
	if err != nil {
		t.Fatalf("relative path: %v", err)
	}
	want := filepath.Join("/cwd", "rel.txt")
	if rel != want {
		t.Errorf("relative path: got %q, want %q", rel, want)
	}
	if _, err := ResolveWithinCwd("rel.txt", ""); err == nil {
		t.Errorf("empty cwd should error")
	}
	if _, err := ResolveWithinCwd("", "/cwd"); err == nil {
		t.Errorf("empty path should error")
	}
}

// ---- readTool.Execute — text path ----

func TestReadTool_SimpleText(t *testing.T) {
	ops := newFakeReadOps()
	ops.files["/test/cwd/hello.txt"] = []byte("hello world")
	tool := NewReadTool(ops)
	res, err := tool.Execute(context.Background(), makeCall(readArgs{Path: "hello.txt"}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if res.IsError {
		t.Errorf("unexpected error result: %s", ResultText(res))
	}
	if got := ResultText(res); got != "hello world" {
		t.Errorf("text = %q, want \"hello world\"", got)
	}
}

func TestReadTool_MissingPath(t *testing.T) {
	tool := NewReadTool(newFakeReadOps())
	res, err := tool.Execute(context.Background(), makeCall(readArgs{}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !res.IsError {
		t.Errorf("expected IsError on missing path")
	}
}

func TestReadTool_FileNotFound(t *testing.T) {
	tool := NewReadTool(newFakeReadOps())
	res, err := tool.Execute(context.Background(), makeCall(readArgs{Path: "missing.txt"}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !res.IsError {
		t.Errorf("expected IsError on missing file")
	}
	if !strings.Contains(ResultText(res), "missing.txt") {
		t.Errorf("error should mention the path: %s", ResultText(res))
	}
}

func TestReadTool_AccessDenied(t *testing.T) {
	ops := newFakeReadOps()
	ops.accessErr = fs.ErrPermission
	tool := NewReadTool(ops)
	res, err := tool.Execute(context.Background(), makeCall(readArgs{Path: "x.txt"}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !res.IsError {
		t.Errorf("expected IsError on access denied")
	}
}

func TestReadTool_BinaryRefused(t *testing.T) {
	ops := newFakeReadOps()
	ops.files["/test/cwd/bin.dat"] = []byte{0x00, 0x01, 0x02, 0x00, 0x03}
	tool := NewReadTool(ops)
	res, err := tool.Execute(context.Background(), makeCall(readArgs{Path: "bin.dat"}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !res.IsError {
		t.Errorf("expected IsError on binary file")
	}
	if !strings.Contains(strings.ToLower(ResultText(res)), "binary") {
		t.Errorf("error should mention binary: %s", ResultText(res))
	}
}

// ---- readTool.Execute — offset/limit ----

func TestReadTool_Offset(t *testing.T) {
	ops := newFakeReadOps()
	ops.files["/test/cwd/multi.txt"] = []byte("line1\nline2\nline3\nline4\nline5")
	tool := NewReadTool(ops)
	offset := 3
	res, _ := tool.Execute(context.Background(), makeCall(readArgs{Path: "multi.txt", Offset: &offset}))
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	if !strings.HasPrefix(got, "line3") {
		t.Errorf("offset=3 should start with line3: %q", got)
	}
}

func TestReadTool_Limit(t *testing.T) {
	ops := newFakeReadOps()
	ops.files["/test/cwd/multi.txt"] = []byte("line1\nline2\nline3\nline4\nline5")
	tool := NewReadTool(ops)
	limit := 2
	res, _ := tool.Execute(context.Background(), makeCall(readArgs{Path: "multi.txt", Limit: &limit}))
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	if !strings.HasPrefix(got, "line1\nline2") {
		t.Errorf("limit=2 should show line1,line2: %q", got)
	}
	if !strings.Contains(got, "offset=3") {
		t.Errorf("continuation hint should point at offset=3: %q", got)
	}
}

func TestReadTool_OffsetAndLimit(t *testing.T) {
	ops := newFakeReadOps()
	ops.files["/test/cwd/multi.txt"] = []byte("line1\nline2\nline3\nline4\nline5")
	tool := NewReadTool(ops)
	offset := 2
	limit := 2
	res, _ := tool.Execute(context.Background(), makeCall(readArgs{Path: "multi.txt", Offset: &offset, Limit: &limit}))
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	if !strings.HasPrefix(got, "line2\nline3") {
		t.Errorf("offset=2,limit=2 should show line2,line3: %q", got)
	}
}

func TestReadTool_OffsetOutOfBounds(t *testing.T) {
	ops := newFakeReadOps()
	ops.files["/test/cwd/multi.txt"] = []byte("one\ntwo")
	tool := NewReadTool(ops)
	offset := 99
	res, _ := tool.Execute(context.Background(), makeCall(readArgs{Path: "multi.txt", Offset: &offset}))
	if !res.IsError {
		t.Errorf("expected IsError on offset out of bounds")
	}
	if !strings.Contains(ResultText(res), "offset") {
		t.Errorf("error should mention offset: %s", ResultText(res))
	}
}

func TestReadTool_OffsetZero(t *testing.T) {
	ops := newFakeReadOps()
	ops.files["/test/cwd/x.txt"] = []byte("content")
	tool := NewReadTool(ops)
	offset := 0
	res, _ := tool.Execute(context.Background(), makeCall(readArgs{Path: "x.txt", Offset: &offset}))
	if !res.IsError {
		t.Errorf("offset=0 must be rejected (1-indexed)")
	}
}

func TestReadTool_LimitNegative(t *testing.T) {
	ops := newFakeReadOps()
	ops.files["/test/cwd/x.txt"] = []byte("content")
	tool := NewReadTool(ops)
	limit := -1
	res, _ := tool.Execute(context.Background(), makeCall(readArgs{Path: "x.txt", Limit: &limit}))
	if !res.IsError {
		t.Errorf("limit=-1 must be rejected")
	}
}

// ---- readTool.Execute — truncation ----

func TestReadTool_Truncation(t *testing.T) {
	ops := newFakeReadOps()
	// Many lines → triggers DefaultMaxLines (500). Each line ~30 chars so
	// we stay under DefaultMaxBytes.
	var lines []string
	for i := 0; i < 10_000; i++ {
		lines = append(lines, "line-"+numberStr(i))
	}
	ops.files["/test/cwd/big.txt"] = []byte(strings.Join(lines, "\n"))
	tool := NewReadTool(ops)
	res, _ := tool.Execute(context.Background(), makeCall(readArgs{Path: "big.txt"}))
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	if !strings.Contains(got, "Use offset=") {
		t.Errorf("truncated output should include offset hint: ...%s", tailStr(got, 200))
	}
}

func TestReadTool_FirstLineExceedsByteLimit(t *testing.T) {
	ops := newFakeReadOps()
	huge := strings.Repeat("x", DefaultMaxBytes+1)
	ops.files["/test/cwd/huge.txt"] = []byte(huge)
	tool := NewReadTool(ops)
	res, _ := tool.Execute(context.Background(), makeCall(readArgs{Path: "huge.txt"}))
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	if !strings.Contains(got, "sed -n") {
		t.Errorf("first-line-oversize should suggest bash sed fallback: %s...", headStr(got, 200))
	}
}

// ---- readTool.Execute — image path ----

func TestReadTool_PNG(t *testing.T) {
	ops := newFakeReadOps()
	pngHeader := []byte{
		0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
		0x00, 0x00, 0x00, 0x0d, 'I', 'H', 'D', 'R',
		0, 0, 0, 1,
	}
	ops.files["/test/cwd/img.png"] = pngHeader
	tool := NewReadTool(ops)
	res, err := tool.Execute(context.Background(), makeCall(readArgs{Path: "img.png"}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	if len(res.Content) != 2 {
		t.Fatalf("expected 2 content blocks (note + image), got %d", len(res.Content))
	}
	if _, ok := res.Content[1].(llm.ImageContent); !ok {
		t.Errorf("second block should be ImageContent, got %T", res.Content[1])
	}
}

func TestReadTool_JPEG(t *testing.T) {
	ops := newFakeReadOps()
	jpg := append([]byte{0xff, 0xd8, 0xff, 0xe0}, []byte("rest of file")...)
	ops.files["/test/cwd/x.jpg"] = jpg
	tool := NewReadTool(ops)
	res, _ := tool.Execute(context.Background(), makeCall(readArgs{Path: "x.jpg"}))
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	if len(res.Content) != 2 {
		t.Fatalf("expected 2 content blocks, got %d", len(res.Content))
	}
}

// ---- readTool.Execute — ctx cancellation ----

func TestReadTool_CtxCanceled(t *testing.T) {
	ops := newFakeReadOps()
	ops.files["/test/cwd/x.txt"] = []byte("content")
	tool := NewReadTool(ops)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := tool.Execute(ctx, makeCall(readArgs{Path: "x.txt"}))
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

// ---- readTool.Execute — absolute path ----

func TestReadTool_AbsolutePath(t *testing.T) {
	ops := newFakeReadOps()
	dir := t.TempDir()
	absFile := filepath.Join(dir, "abs.txt")
	ops.files[absFile] = []byte("abs content")
	tool := NewReadTool(ops)
	res, _ := tool.Execute(context.Background(), makeCall(readArgs{Path: absFile}))
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	if got := ResultText(res); got != "abs content" {
		t.Errorf("text = %q, want \"abs content\"", got)
	}
}

// ---- readTool.Execute — invalid args ----

func TestReadTool_InvalidJSON(t *testing.T) {
	tool := NewReadTool(newFakeReadOps())
	res, err := tool.Execute(context.Background(), ToolCall{
		ID:   "x",
		Name: "read",
		Args: json.RawMessage(`{"path": invalid`),
		Cwd:  "/cwd",
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !res.IsError {
		t.Errorf("invalid JSON should yield IsError result")
	}
}

func TestReadTool_NilOpsDefaultsToOS(t *testing.T) {
	tool := NewReadTool(nil)
	if rt, ok := tool.(*readTool); !ok || rt.ops == nil {
		t.Errorf("nil ops should default to OSReadOperations")
	}
}

// ---- readTool.Name / Description / Parameters ----

func TestReadTool_NameDescriptionParameters(t *testing.T) {
	tool := NewReadTool(newFakeReadOps())
	if tool.Name() != "read" {
		t.Errorf("Name = %q, want read", tool.Name())
	}
	if tool.Description() == "" {
		t.Errorf("Description should not be empty")
	}
	s := tool.Parameters()
	if s.Type != "object" {
		t.Errorf("Parameters.Type = %q, want object", s.Type)
	}
	if s.Properties == nil {
		t.Errorf("Properties should not be nil")
	}
}

func TestReadTool_ParametersRoundTripJSON(t *testing.T) {
	tool := NewReadTool(newFakeReadOps())
	s := tool.Parameters()
	raw, err := s.MarshalJSON()
	if err != nil {
		t.Fatalf("MarshalJSON: %v", err)
	}
	str := string(raw)
	if !strings.Contains(str, `"path"`) {
		t.Errorf("schema missing \"path\" property: %s", str)
	}
	if strings.Contains(str, "$schema") {
		t.Errorf("schema should strip $schema: %s", str)
	}
	if strings.Contains(str, "$id") {
		t.Errorf("schema should strip $id: %s", str)
	}
}

// ---- readTool.RenderCall / RenderResult ----

func TestReadTool_RenderCall_Plain(t *testing.T) {
	tool := NewReadTool(newFakeReadOps())
	got := tool.RenderCall(json.RawMessage(`{"path":"x.txt"}`), PlainTheme())
	if !strings.Contains(got, "read") || !strings.Contains(got, "x.txt") {
		t.Errorf("RenderCall missing name/path: %q", got)
	}
}

func TestReadTool_RenderCall_WithOffsetLimit(t *testing.T) {
	tool := NewReadTool(newFakeReadOps())
	got := tool.RenderCall(json.RawMessage(`{"path":"x.txt","offset":10,"limit":5}`), PlainTheme())
	if !strings.Contains(got, "10-14") {
		t.Errorf("RenderCall should show line range 10-14: %q", got)
	}
}

func TestReadTool_RenderResult_Text(t *testing.T) {
	tool := NewReadTool(newFakeReadOps())
	res := NewTextResult("the text output")
	got := tool.RenderResult(res, PlainTheme())
	if !strings.Contains(got, "the text output") {
		t.Errorf("RenderResult missing text: %q", got)
	}
}

func TestReadTool_RenderResult_Error(t *testing.T) {
	tool := NewReadTool(newFakeReadOps())
	res := NewErrorResult("file missing")
	got := tool.RenderResult(res, ColorTheme())
	if !strings.Contains(got, "file missing") {
		t.Errorf("RenderResult missing error text: %q", got)
	}
}

// ---- helpers ----

func numberStr(i int) string {
	if i == 0 {
		return "0"
	}
	digits := []byte{}
	for i > 0 {
		digits = append([]byte{byte('0' + i%10)}, digits...)
		i /= 10
	}
	return string(digits)
}

func headStr(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func tailStr(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return "..." + s[len(s)-n:]
}
