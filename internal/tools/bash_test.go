package tools

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeBashOps is an in-memory BashOperations for unit tests. The first
// matching handler (added later overrides earlier) is invoked.
type fakeBashOps struct {
	mu       sync.Mutex
	handlers []fakeBashHandler
	// fallback is invoked when no handler matches the command.
	fallback func(ctx context.Context, command, cwd string, opts BashExecOptions) (BashExecResult, error)
}

// fakeBashHandler is a single canned response keyed by command prefix.
type fakeBashHandler struct {
	match   string // command must start with this
	run     func(command string) ([]byte, []byte, int)
	exitErr error // if non-nil, returned from Exec
}

func newFakeBashOps() *fakeBashOps {
	return &fakeBashOps{}
}

// add installs a handler. Handlers are checked in install order; the
// first whose match is a prefix of command runs.
func (f *fakeBashOps) add(match string, run func(command string) ([]byte, []byte, int)) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.handlers = append(f.handlers, fakeBashHandler{match: match, run: run})
}

func (f *fakeBashOps) Exec(ctx context.Context, command, cwd string, opts BashExecOptions) (BashExecResult, error) {
	if err := ctx.Err(); err != nil {
		return BashExecResult{ExitCode: -1}, err
	}
	f.mu.Lock()
	handlers := append([]fakeBashHandler(nil), f.handlers...)
	fallback := f.fallback
	f.mu.Unlock()

	var chunks []byte
	for _, h := range handlers {
		if strings.HasPrefix(command, h.match) {
			stdout, stderr, code := h.run(command)
			if opts.OnData != nil {
				if len(stdout) > 0 {
					opts.OnData(stdout)
					chunks = append(chunks, stdout...)
				}
				if len(stderr) > 0 {
					opts.OnData(stderr)
					chunks = append(chunks, stderr...)
				}
			}
			_ = chunks
			return BashExecResult{ExitCode: code, Stdout: stdout, Stderr: stderr}, h.exitErr
		}
	}
	if fallback != nil {
		return fallback(ctx, command, cwd, opts)
	}
	// Default: behave like /bin/true (exit 0, no output).
	return BashExecResult{ExitCode: 0}, nil
}

func makeBashCall(args any) ToolCall {
	raw, _ := json.Marshal(args)
	return ToolCall{
		ID:   "bash-call",
		Name: "bash",
		Args: raw,
		Cwd:  "/test/cwd",
	}
}

// ---- OSBashOperations end-to-end ----

func TestOSBashOperations_EchoSuccess(t *testing.T) {
	ops := OSBashOperations{}
	res, err := ops.Exec(context.Background(), "echo hello", t.TempDir(), BashExecOptions{})
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	if res.ExitCode != 0 {
		t.Errorf("ExitCode = %d, want 0", res.ExitCode)
	}
	if !strings.Contains(string(res.Stdout), "hello") {
		t.Errorf("stdout = %q, want contains hello", string(res.Stdout))
	}
}

func TestOSBashOperations_ExitCodeNonZero(t *testing.T) {
	ops := OSBashOperations{}
	res, err := ops.Exec(context.Background(), "exit 7", t.TempDir(), BashExecOptions{})
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	if res.ExitCode != 7 {
		t.Errorf("ExitCode = %d, want 7", res.ExitCode)
	}
}

func TestOSBashOperations_MissingCwd(t *testing.T) {
	ops := OSBashOperations{}
	_, err := ops.Exec(context.Background(), "echo hi", "/nonexistent-dir-12345", BashExecOptions{})
	if err == nil {
		t.Errorf("expected error on missing cwd")
	}
}

func TestOSBashOperations_EmptyCommand(t *testing.T) {
	ops := OSBashOperations{}
	_, err := ops.Exec(context.Background(), "", "/tmp", BashExecOptions{})
	if err == nil {
		t.Errorf("expected error on empty command")
	}
}

func TestOSBashOperations_Streaming(t *testing.T) {
	ops := OSBashOperations{}
	var chunks []byte
	var mu sync.Mutex
	opts := BashExecOptions{
		OnData: func(b []byte) {
			mu.Lock()
			chunks = append(chunks, b...)
			mu.Unlock()
		},
	}
	res, err := ops.Exec(context.Background(), "printf 'line1\\nline2\\nline3'", t.TempDir(), opts)
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if !strings.Contains(string(chunks), "line1") {
		t.Errorf("OnData should have seen line1: %q", string(chunks))
	}
	if !strings.Contains(string(res.Stdout), "line3") {
		t.Errorf("Stdout should have line3: %q", string(res.Stdout))
	}
}

func TestOSBashOperations_CtxCanceled(t *testing.T) {
	ops := OSBashOperations{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := ops.Exec(ctx, "echo hello", t.TempDir(), BashExecOptions{})
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

func TestOSBashOperations_Timeout(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping timeout test in short mode")
	}
	ops := OSBashOperations{}
	opts := BashExecOptions{Timeout: 100 * time.Millisecond}
	_, err := ops.Exec(context.Background(), "sleep 5", t.TempDir(), opts)
	if err == nil {
		t.Errorf("expected timeout error")
		return
	}
	if !strings.Contains(err.Error(), "timeout") {
		t.Errorf("err should mention timeout: %v", err)
	}
}

// ---- bashTool.Execute ----

func TestBashTool_SimpleSuccess(t *testing.T) {
	ops := newFakeBashOps()
	ops.add("echo", func(c string) ([]byte, []byte, int) {
		return []byte("hello world\n"), nil, 0
	})
	tool := NewBashTool(ops)
	res, err := tool.Execute(context.Background(), makeBashCall(bashArgs{Command: "echo hello"}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if res.IsError {
		t.Errorf("unexpected error result: %s", ResultText(res))
	}
	if !strings.Contains(ResultText(res), "hello world") {
		t.Errorf("output missing content: %q", ResultText(res))
	}
}

func TestBashTool_ExitNonZero(t *testing.T) {
	ops := newFakeBashOps()
	ops.add("false", func(c string) ([]byte, []byte, int) {
		return nil, []byte("error: nope"), 1
	})
	tool := NewBashTool(ops)
	res, err := tool.Execute(context.Background(), makeBashCall(bashArgs{Command: "false"}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !res.IsError {
		t.Errorf("expected IsError on non-zero exit")
	}
	if !strings.Contains(ResultText(res), "exited with code 1") {
		t.Errorf("error should mention exit code: %q", ResultText(res))
	}
	if !strings.Contains(ResultText(res), "error: nope") {
		t.Errorf("error should include stderr: %q", ResultText(res))
	}
	if res.Metadata["exitCode"].(int) != 1 {
		t.Errorf("metadata exitCode = %v, want 1", res.Metadata["exitCode"])
	}
}

func TestBashTool_NoOutput(t *testing.T) {
	ops := newFakeBashOps()
	ops.add("true", func(c string) ([]byte, []byte, int) {
		return nil, nil, 0
	})
	tool := NewBashTool(ops)
	res, _ := tool.Execute(context.Background(), makeBashCall(bashArgs{Command: "true"}))
	if res.IsError {
		t.Errorf("unexpected error: %s", ResultText(res))
	}
	if ResultText(res) != "(no output)" {
		t.Errorf("empty output should yield placeholder: %q", ResultText(res))
	}
}

func TestBashTool_MissingCommand(t *testing.T) {
	tool := NewBashTool(newFakeBashOps())
	res, _ := tool.Execute(context.Background(), makeBashCall(bashArgs{}))
	if !res.IsError {
		t.Errorf("expected IsError on missing command")
	}
}

func TestBashTool_MissingCwd(t *testing.T) {
	tool := NewBashTool(newFakeBashOps())
	raw, _ := json.Marshal(bashArgs{Command: "echo hi"})
	res, _ := tool.Execute(context.Background(), ToolCall{
		ID:   "x",
		Name: "bash",
		Args: raw,
		Cwd:  "",
	})
	if !res.IsError {
		t.Errorf("expected IsError on missing cwd")
	}
}

func TestBashTool_StderrCaptured(t *testing.T) {
	ops := newFakeBashOps()
	ops.add("warn", func(c string) ([]byte, []byte, int) {
		return []byte("stdout line\n"), []byte("stderr warning\n"), 0
	})
	tool := NewBashTool(ops)
	res, _ := tool.Execute(context.Background(), makeBashCall(bashArgs{Command: "warn"}))
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	if !strings.Contains(got, "stdout line") {
		t.Errorf("missing stdout: %q", got)
	}
	if !strings.Contains(got, "stderr warning") {
		t.Errorf("missing stderr: %q", got)
	}
	if !strings.Contains(got, "[stderr]") {
		t.Errorf("missing stderr separator: %q", got)
	}
}

func TestBashTool_Truncation(t *testing.T) {
	ops := newFakeBashOps()
	// 5_000 lines of output — should trigger DefaultMaxLines=500.
	var lines []string
	for i := 0; i < 5_000; i++ {
		lines = append(lines, "line-"+numberStr(i))
	}
	ops.add("seq", func(c string) ([]byte, []byte, int) {
		return []byte(strings.Join(lines, "\n")), nil, 0
	})
	tool := NewBashTool(ops)
	res, _ := tool.Execute(context.Background(), makeBashCall(bashArgs{Command: "seq"}))
	if res.IsError {
		t.Fatalf("unexpected error: %s", ResultText(res))
	}
	got := ResultText(res)
	// Per spec: first 250 lines + ellipsis + last 250 lines.
	if !strings.Contains(got, "lines elided") && !strings.Contains(got, "byte limit") {
		t.Errorf("expected truncation marker: %s...", headStr(got, 100))
	}
}

func TestBashTool_StreamingViaCallback(t *testing.T) {
	ops := newFakeBashOps()
	ops.add("stream", func(c string) ([]byte, []byte, int) {
		// Caller-supplied OnData is invoked by fakeBashOps.Exec.
		return []byte("hello-from-stream\n"), nil, 0
	})
	// Wrap to capture OnData invocations.
	called := make(chan []byte, 4)
	wrapped := &bashOpsWithHook{
		inner: ops,
		hook:  func(b []byte) { called <- b },
	}
	tool := NewBashTool(wrapped)
	_, err := tool.Execute(context.Background(), makeBashCall(bashArgs{Command: "stream"}))
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	// The fake invokes OnData once per non-empty stream.
	select {
	case <-called:
		// good
	case <-time.After(100 * time.Millisecond):
		t.Errorf("OnData callback was not invoked")
	}
}

// bashOpsWithHook wraps a BashOperations to capture OnData chunks for tests.
type bashOpsWithHook struct {
	inner BashOperations
	hook  func([]byte)
}

func (w *bashOpsWithHook) Exec(ctx context.Context, command, cwd string, opts BashExecOptions) (BashExecResult, error) {
	userOnData := opts.OnData
	opts.OnData = func(b []byte) {
		w.hook(b)
		if userOnData != nil {
			userOnData(b)
		}
	}
	return w.inner.Exec(ctx, command, cwd, opts)
}

// ---- bashTool.Execute — args validation ----

func TestBashTool_InvalidJSON(t *testing.T) {
	tool := NewBashTool(newFakeBashOps())
	res, err := tool.Execute(context.Background(), ToolCall{
		ID:   "x",
		Name: "bash",
		Args: json.RawMessage(`{"command": invalid`),
		Cwd:  "/cwd",
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !res.IsError {
		t.Errorf("invalid JSON should yield IsError result")
	}
}

func TestBashTool_NegativeTimeout(t *testing.T) {
	tool := NewBashTool(newFakeBashOps())
	to := -1
	res, _ := tool.Execute(context.Background(), makeBashCall(bashArgs{Command: "echo hi", Timeout: &to}))
	if !res.IsError {
		t.Errorf("negative timeout should yield IsError")
	}
}

func TestBashTool_NilOpsDefaultsToOS(t *testing.T) {
	tool := NewBashTool(nil)
	if bt, ok := tool.(*bashTool); !ok || bt.ops == nil {
		t.Errorf("nil ops should default to OSBashOperations")
	}
}

// ---- bashTool.Execute — ctx cancellation propagates ----

func TestBashTool_CtxCanceled(t *testing.T) {
	ops := newFakeBashOps()
	// A handler that simulates a long-running command via fallback.
	ops.fallback = func(ctx context.Context, command, cwd string, opts BashExecOptions) (BashExecResult, error) {
		<-ctx.Done()
		return BashExecResult{ExitCode: -1}, ctx.Err()
	}
	tool := NewBashTool(ops)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := tool.Execute(ctx, makeBashCall(bashArgs{Command: "long-running"}))
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

// ---- bashTool.Name / Description / Parameters ----

func TestBashTool_NameDescriptionParameters(t *testing.T) {
	tool := NewBashTool(newFakeBashOps())
	if tool.Name() != "bash" {
		t.Errorf("Name = %q, want bash", tool.Name())
	}
	if tool.Description() == "" {
		t.Errorf("Description should not be empty")
	}
	s := tool.Parameters()
	if s.Type != "object" {
		t.Errorf("Parameters.Type = %q, want object", s.Type)
	}
	raw, _ := s.MarshalJSON()
	if !strings.Contains(string(raw), `"command"`) {
		t.Errorf("schema should mention command: %s", string(raw))
	}
}

// ---- bashTool.RenderCall / RenderResult ----

func TestBashTool_RenderCall_Plain(t *testing.T) {
	tool := NewBashTool(newFakeBashOps())
	got := tool.RenderCall(json.RawMessage(`{"command":"ls -la"}`), PlainTheme())
	if !strings.Contains(got, "ls -la") {
		t.Errorf("RenderCall missing command: %q", got)
	}
}

func TestBashTool_RenderCall_WithTimeout(t *testing.T) {
	tool := NewBashTool(newFakeBashOps())
	got := tool.RenderCall(json.RawMessage(`{"command":"sleep 60","timeout":30}`), PlainTheme())
	if !strings.Contains(got, "timeout") {
		t.Errorf("RenderCall should show timeout: %q", got)
	}
}

func TestBashTool_RenderResult_Text(t *testing.T) {
	tool := NewBashTool(newFakeBashOps())
	res := NewTextResult("hello")
	got := tool.RenderResult(res, PlainTheme())
	if !strings.Contains(got, "hello") {
		t.Errorf("RenderResult missing text: %q", got)
	}
}

func TestBashTool_RenderResult_Error(t *testing.T) {
	tool := NewBashTool(newFakeBashOps())
	res := NewErrorResult("command failed")
	got := tool.RenderResult(res, ColorTheme())
	if !strings.Contains(got, "command failed") {
		t.Errorf("RenderResult missing error: %q", got)
	}
}

// ---- helpers shared with other test files ----
