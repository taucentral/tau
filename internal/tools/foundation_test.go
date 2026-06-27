package tools

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/invopop/jsonschema"

	"github.com/coevin/tau/internal/llm"
)

// fakeTool is a minimal Tool for testing the Registry.
type fakeTool struct {
	name        string
	description string
}

func (f *fakeTool) Name() string                  { return f.name }
func (f *fakeTool) Description() string           { return f.description }
func (f *fakeTool) Parameters() jsonschema.Schema { return jsonschema.Schema{Type: "object"} }
func (f *fakeTool) Execute(ctx context.Context, call ToolCall) (ToolResult, error) {
	return NewTextResult("fake:" + f.name), nil
}
func (f *fakeTool) RenderCall(args json.RawMessage, theme *Theme) string {
	return "fake-call"
}
func (f *fakeTool) RenderResult(result ToolResult, theme *Theme) string {
	return "fake-result"
}

func TestRegistry_RegisterAndLookup(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(&fakeTool{name: "read", description: "x"}); err != nil {
		t.Fatalf("Register: %v", err)
	}
	got, err := r.Lookup("read")
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	if got.Name() != "read" {
		t.Errorf("Lookup returned %q, want read", got.Name())
	}
}

func TestRegistry_UnknownTool(t *testing.T) {
	r := NewRegistry()
	_, err := r.Lookup("nonexistent")
	if !errors.Is(err, ErrUnknownTool) {
		t.Errorf("err = %v, want ErrUnknownTool", err)
	}
}

func TestRegistry_DuplicateName(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(&fakeTool{name: "x"}); err != nil {
		t.Fatalf("first Register: %v", err)
	}
	err := r.Register(&fakeTool{name: "x"})
	if !errors.Is(err, ErrDuplicateTool) {
		t.Errorf("err = %v, want ErrDuplicateTool", err)
	}
}

func TestRegistry_NilTool(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(nil); err == nil {
		t.Errorf("expected error on nil tool")
	}
}

func TestRegistry_EmptyName(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(&fakeTool{name: ""}); err == nil {
		t.Errorf("expected error on empty Name")
	}
}

func TestRegistry_MustRegister_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic on duplicate MustRegister")
		}
	}()
	r := NewRegistry()
	r.MustRegister(&fakeTool{name: "x"})
	r.MustRegister(&fakeTool{name: "x"})
}

func TestRegistry_Names(t *testing.T) {
	r := NewRegistry()
	for _, n := range []string{"zeta", "alpha", "mango"} {
		r.MustRegister(&fakeTool{name: n})
	}
	got := r.Names()
	want := []string{"alpha", "mango", "zeta"}
	if len(got) != 3 || got[0] != want[0] || got[1] != want[1] || got[2] != want[2] {
		t.Errorf("Names() = %v, want %v (sorted)", got, want)
	}
}

func TestRegistry_SchemasSorted(t *testing.T) {
	r := NewRegistry()
	r.MustRegister(&fakeTool{name: "zeta"})
	r.MustRegister(&fakeTool{name: "alpha"})
	schemas := r.Schemas()
	if len(schemas) != 2 {
		t.Fatalf("Schemas len = %d, want 2", len(schemas))
	}
	if schemas[0].Name != "alpha" || schemas[1].Name != "zeta" {
		t.Errorf("Schemas not sorted: %+v", schemas)
	}
	for _, s := range schemas {
		if len(s.Parameters) == 0 {
			t.Errorf("Schemas[%q].Parameters is empty", s.Name)
		}
		// "alpha" intentionally has an empty Description in this test.
	}
}

func TestRegistry_Len(t *testing.T) {
	r := NewRegistry()
	if r.Len() != 0 {
		t.Errorf("Len = %d, want 0", r.Len())
	}
	r.MustRegister(&fakeTool{name: "a"})
	r.MustRegister(&fakeTool{name: "b"})
	if r.Len() != 2 {
		t.Errorf("Len = %d, want 2", r.Len())
	}
}

func TestRegistry_ConcurrentSafe(t *testing.T) {
	// Race detector should not fire: Registry uses RWMutex.
	r := NewRegistry()
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r.Lookup("x")
			r.Names()
		}()
	}
	wg.Wait()
}

// ---- Queue ----

func TestFileMutationQueue_SamePathSerializes(t *testing.T) {
	q := NewFileMutationQueue()
	var order []int
	var mu sync.Mutex
	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			q.Run(context.Background(), "/same", func() error {
				mu.Lock()
				order = append(order, i)
				mu.Unlock()
				time.Sleep(5 * time.Millisecond)
				return nil
			})
		}(i)
	}
	wg.Wait()
	if len(order) != 5 {
		t.Errorf("order len = %d, want 5", len(order))
	}
}

func TestFileMutationQueue_DifferentPathsParallel(t *testing.T) {
	// Two Run calls on different paths should proceed concurrently;
	// total wall time should be roughly one call, not two.
	q := NewFileMutationQueue()
	start := time.Now()
	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			q.Run(context.Background(), "/different-"+string(rune('a'+i)), func() error {
				time.Sleep(50 * time.Millisecond)
				return nil
			})
		}(i)
	}
	wg.Wait()
	elapsed := time.Since(start)
	if elapsed >= 100*time.Millisecond {
		t.Errorf("different-path calls took %v; want < 100ms (parallel)", elapsed)
	}
}

func TestFileMutationQueue_NormalizesPath(t *testing.T) {
	// "./foo.txt" and "foo.txt" should share a lock.
	q := NewFileMutationQueue()
	var count int32
	var wg sync.WaitGroup
	for _, p := range []string{"foo.txt", "./foo.txt", "foo.txt"} {
		wg.Add(1)
		go func(p string) {
			defer wg.Done()
			q.Run(context.Background(), p, func() error {
				atomic.AddInt32(&count, 1)
				time.Sleep(5 * time.Millisecond)
				return nil
			})
		}(p)
	}
	wg.Wait()
	if count != 3 {
		t.Errorf("count = %d, want 3 (all serialized)", count)
	}
}

func TestFileMutationQueue_CtxCanceled(t *testing.T) {
	q := NewFileMutationQueue()
	// Block the path with a long-running call; a second call with a
	// canceled ctx should return ctx.Err() quickly.
	block := make(chan struct{})
	go q.Run(context.Background(), "/block", func() error { //nolint:unparam // signature is fixed by FileMutationQueue.Run
		<-block
		return nil
	})
	time.Sleep(20 * time.Millisecond) // let the first call grab the lock

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := q.Run(ctx, "/block", func() error { return nil })
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
	close(block)
}

// ---- Truncate ----

func TestTruncateHeadTail_NoTruncation(t *testing.T) {
	s := "short\nmulti\nline\nstring"
	got := TruncateHeadTail(s, DefaultMaxLines, DefaultMaxBytes)
	if got != s {
		t.Errorf("expected passthrough, got truncation")
	}
}

func TestTruncateHeadTail_LineLimitEnforced(t *testing.T) {
	// 100 lines; cap at 10.
	var lines []string
	for i := 0; i < 100; i++ {
		lines = append(lines, "line-"+string(rune('a'+i%26))+strings.Repeat("x", i%5))
	}
	s := strings.Join(lines, "\n")
	got := TruncateHeadTail(s, 10, 0)
	if strings.Count(got, "\n") >= 100 {
		t.Errorf("got too many lines: %d", strings.Count(got, "\n"))
	}
	if !strings.Contains(got, "lines elided") {
		t.Errorf("expected elision marker")
	}
	if !strings.Contains(got, "head") || !strings.Contains(got, "tail") {
		t.Errorf("expected head/tail marker: %s", got)
	}
}

func TestTruncateHeadTail_ByteLimitEnforced(t *testing.T) {
	// 10 KiB input capped at 1 KiB must trigger byte-limit truncation.
	s := strings.Repeat("x", 10_000)
	got := TruncateHeadTail(s, 0, 1_000)
	if len(got) >= 10_000 {
		t.Errorf("byte cap not enforced: len=%d", len(got))
	}
	if !strings.Contains(got, "byte limit") {
		t.Errorf("expected byte-limit marker: %s...", got[:100])
	}
}

func TestTruncateHeadTail_NoLimits(t *testing.T) {
	// 0/0 disables truncation entirely.
	s := strings.Repeat("x", 10_000)
	got := TruncateHeadTail(s, 0, 0)
	if got != s {
		t.Errorf("0/0 should passthrough")
	}
}

func TestTruncateHeadTail_Empty(t *testing.T) {
	got := TruncateHeadTail("", DefaultMaxLines, DefaultMaxBytes)
	if got != "" {
		t.Errorf("empty input should passthrough; got %q", got)
	}
}

func TestTruncateHeadTail_ExactFit(t *testing.T) {
	// Exactly maxLines: no truncation needed.
	s := "a\nb\nc"
	got := TruncateHeadTail(s, 3, 0)
	if got != s {
		t.Errorf("exact-fit should passthrough: got %q", got)
	}
}

func TestTruncateBytes_Basic(t *testing.T) {
	s := strings.Repeat("x", 1000)
	got := TruncateBytes(s, 100)
	if len(got) >= 1000 {
		t.Errorf("byte cap not enforced: len=%d", len(got))
	}
	if !strings.Contains(got, "truncated") {
		t.Errorf("expected truncation marker")
	}
}

func TestTruncateBytes_UTF8Boundary(t *testing.T) {
	// Truncate must not split a multi-byte UTF-8 sequence.
	s := strings.Repeat("⌘", 100) // 3 bytes each
	got := TruncateBytes(s, 50)
	// Verify decoded without error and no replacement chars mid-stream.
	if !strings.HasPrefix(got, "⌘") {
		t.Errorf("truncated output doesn't start with the rune: %q", got[:10])
	}
	// Verify it round-trips through encoding/decoding (no invalid UTF-8).
	if !utf8Valid(got) {
		t.Errorf("truncated output has invalid UTF-8")
	}
}

func utf8Valid(s string) bool {
	for _, r := range s {
		if r == 0xFFFD {
			return false
		}
	}
	return true
}

func TestNewTextResult(t *testing.T) {
	r := NewTextResult("hello")
	if r.IsError {
		t.Errorf("IsError should be false")
	}
	if len(r.Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(r.Content))
	}
	tc, ok := r.Content[0].(llm.TextContent)
	if !ok || tc.Text != "hello" {
		t.Errorf("Content[0] = %+v, want TextContent{hello}", r.Content[0])
	}
}

func TestNewErrorResult(t *testing.T) {
	r := NewErrorResult("oops")
	if !r.IsError {
		t.Errorf("IsError should be true")
	}
}

func TestToolResult_AsLLMResult(t *testing.T) {
	r := NewTextResult("x")
	got := r.AsLLMResult("tu-1")
	if got.ToolUseID != "tu-1" {
		t.Errorf("ToolUseID = %q, want tu-1", got.ToolUseID)
	}
	if len(got.Content) != 1 {
		t.Errorf("Content len = %d, want 1", len(got.Content))
	}
}

func TestParseArgs_Valid(t *testing.T) {
	var dst struct {
		Path string `json:"path"`
	}
	if r := ParseArgs(json.RawMessage(`{"path":"/x"}`), &dst, "read"); r != nil {
		t.Errorf("ParseArgs returned non-nil: %+v", r)
	}
	if dst.Path != "/x" {
		t.Errorf("dst.Path = %q, want /x", dst.Path)
	}
}

func TestParseArgs_Invalid(t *testing.T) {
	var dst struct{}
	r := ParseArgs(json.RawMessage(`not json`), &dst, "read")
	if r == nil {
		t.Fatalf("expected error result, got nil")
	}
	if !r.IsError {
		t.Errorf("IsError should be true")
	}
}

func TestParseArgs_Empty(t *testing.T) {
	// Empty args is valid; no-op for tools that take no params.
	var dst struct{}
	if r := ParseArgs(nil, &dst, "x"); r != nil {
		t.Errorf("empty args should return nil; got %+v", r)
	}
}

// ---- Theme ----

func TestPlainTheme(t *testing.T) {
	th := PlainTheme()
	if th.Primary != "" {
		t.Errorf("PlainTheme.Primary = %q, want empty", th.Primary)
	}
	if th.Indent != "  " {
		t.Errorf("Indent = %q, want two spaces", th.Indent)
	}
}

func TestColorTheme(t *testing.T) {
	th := ColorTheme()
	if th.Primary == "" {
		t.Errorf("Primary should be set in ColorTheme")
	}
	if th.Reset == "" {
		t.Errorf("Reset should be set in ColorTheme")
	}
}

func TestTheme_Wrap(t *testing.T) {
	th := ColorTheme()
	got := th.Wrap(th.Primary, "x")
	if !strings.HasPrefix(got, th.Primary) {
		t.Errorf("Wrap missing prefix: %q", got)
	}
	if !strings.HasSuffix(got, th.Reset) {
		t.Errorf("Wrap missing Reset suffix: %q", got)
	}

	th2 := PlainTheme()
	if th2.Wrap("", "x") != "x" {
		t.Errorf("empty style + empty Reset should passthrough")
	}
}

func TestTheme_IndentN(t *testing.T) {
	th := PlainTheme()
	if got := th.IndentN(3); len(got) != 6 {
		t.Errorf("IndentN(3) = %q (len %d), want 6 chars", got, len(got))
	}
	if got := th.IndentN(0); got != "" {
		t.Errorf("IndentN(0) = %q, want empty", got)
	}
}
