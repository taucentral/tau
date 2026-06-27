package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/invopop/jsonschema"

	"github.com/coevin/tau/internal/llm"
)

// BashExecOptions controls the runtime behavior of a BashOperations.Exec
// call. The callbacks (if non-nil) receive streaming output as it is
// produced; the final accumulated bytes are always returned in the result.
type BashExecOptions struct {
	// OnData is invoked with each chunk of stdout/stderr output as it
	// arrives. Chunks are NOT line-buffered; they reflect the underlying
	// pipe buffer boundaries. May be nil.
	OnData func(chunk []byte)
	// Timeout, if greater than zero, kills the process after the
	// duration. A zero value means no timeout.
	Timeout time.Duration
	// Env overrides the process environment. If nil, the current
	// process environment is inherited.
	Env []string
}

// BashExecResult is the terminal state of an Exec call.
type BashExecResult struct {
	// ExitCode is the process exit code. -1 indicates the process was
	// killed (by signal, timeout, or context cancellation).
	ExitCode int
	// Stdout is the accumulated stdout bytes.
	Stdout []byte
	// Stderr is the accumulated stderr bytes.
	Stderr []byte
	// Signal is the signal name that killed the process, if any (e.g.
	// "KILL", "TERM"). Empty for normal exits.
	Signal string
}

// BashOperations abstracts the I/O surface of the bash tool. The default
// implementation uses the real OS; tests inject fake implementations.
type BashOperations interface {
	// Exec runs command in cwd and returns the accumulated output and
	// exit code. Streaming output (if opts.OnData is set) is delivered
	// as it arrives. ctx cancellation kills the process tree.
	Exec(ctx context.Context, command, cwd string, opts BashExecOptions) (BashExecResult, error)
}

// OSBashOperations is the default BashOperations backed by the user's
// shell. It uses /bin/sh on POSIX and cmd.exe on Windows; detached
// descendants are killed on ctx cancellation via process-group signaling.
type OSBashOperations struct{}

// Exec runs command via the platform shell and streams output.
func (OSBashOperations) Exec(ctx context.Context, command, cwd string, opts BashExecOptions) (BashExecResult, error) {
	if command == "" {
		return BashExecResult{ExitCode: -1}, errors.New("bash: empty command")
	}
	if cwd == "" {
		return BashExecResult{ExitCode: -1}, errors.New("bash: empty cwd")
	}
	if _, err := os.Stat(cwd); err != nil {
		return BashExecResult{ExitCode: -1}, fmt.Errorf("bash: cwd %q: %v", cwd, err)
	}

	cmd := buildShellCommand(ctx, command, cwd, opts.Env)
	var stdoutBuf, stderrBuf bytes.Buffer
	var mu sync.Mutex // guards stdoutBuf/stderrBuf writes from concurrent OnData
	stdoutWriter := newTeeWriter(&stdoutBuf, &mu, opts.OnData)
	stderrWriter := newTeeWriter(&stderrBuf, &mu, opts.OnData)
	cmd.Stdout = stdoutWriter
	cmd.Stderr = stderrWriter

	// exec.CommandContext surfaces ctx cancellation through Start's error
	// path; check ctx.Err() first so an already-cancelled context returns
	// the bare context.Canceled sentinel the caller expects (errors.Is).
	if err := ctx.Err(); err != nil {
		return BashExecResult{ExitCode: -1}, err
	}
	if err := cmd.Start(); err != nil {
		return BashExecResult{ExitCode: -1}, fmt.Errorf("bash: start: %w", err)
	}

	// Wait goroutine; closes when the process exits.
	waitCh := make(chan error, 1)
	go func() { waitCh <- cmd.Wait() }()

	var timeoutCh <-chan time.Time
	if opts.Timeout > 0 {
		timer := time.NewTimer(opts.Timeout)
		defer timer.Stop()
		timeoutCh = timer.C
	}

	result := BashExecResult{}
	timedOut := false
	canceled := false

	select {
	case err := <-waitCh:
		// Process exited (or failed to). Capture exit code.
		setExitState(&result, cmd, err)
	case <-timeoutCh:
		timedOut = true
		killProcess(cmd)
		<-waitCh
		setExitState(&result, cmd, nil)
	case <-ctx.Done():
		canceled = true
		killProcess(cmd)
		<-waitCh
		setExitState(&result, cmd, nil)
	}

	result.Stdout = stdoutBuf.Bytes()
	result.Stderr = stderrBuf.Bytes()

	switch {
	case canceled:
		return result, ctx.Err()
	case timedOut:
		return result, fmt.Errorf("bash: timeout after %s", opts.Timeout)
	}
	return result, nil
}

// buildShellCommand constructs the *exec.Cmd for command using the platform
// shell. On POSIX, /bin/sh -c is used. On Windows, cmd /c is used. ctx is
// plumbed via exec.CommandContext so the runtime schedules a SIGKILL on
// cancellation; Exec additionally watches ctx.Done itself to drain the
// child's exit state before returning.
func buildShellCommand(ctx context.Context, command, cwd string, env []string) *exec.Cmd {
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.CommandContext(ctx, "cmd.exe", "/c", command)
	} else {
		cmd = exec.CommandContext(ctx, "/bin/sh", "-c", command)
	}
	cmd.Dir = cwd
	if env != nil {
		cmd.Env = env
	}
	// Put the child in its own process group so we can kill the tree
	// on timeout/cancellation. On Windows this is a no-op (Setpgid is
	// ignored); Windows uses CREATE_NEW_PROCESS_GROUP via syscall but
	// that requires syscall.SysProcAttr — out of scope for v1.
	cmd.SysProcAttr = sysProcAttrForKill()
	return cmd
}

// killProcess sends the process a KILL signal. Idempotent.
func killProcess(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	// Best-effort process-group kill; falls back to single-process kill.
	_ = killProcessGroup(cmd)
}

// setExitState populates result's ExitCode/Signal from cmd.Wait() output.
func setExitState(result *BashExecResult, cmd *exec.Cmd, waitErr error) {
	if cmd.ProcessState == nil {
		result.ExitCode = -1
		return
	}
	result.ExitCode = cmd.ProcessState.ExitCode()
	if waitErr != nil {
		var ee *exec.ExitError
		if errors.As(waitErr, &ee) && ee.ProcessState != nil {
			result.ExitCode = ee.ExitCode()
			result.Signal = signalNameFromState(ee.ProcessState)
		}
	}
}

// teeWriter is an io.Writer that writes to a destination buffer (under a
// shared mutex) and also invokes a callback for each chunk.
type teeWriter struct {
	dst  io.Writer
	mu   *sync.Mutex
	hook func([]byte)
}

func newTeeWriter(dst io.Writer, mu *sync.Mutex, hook func([]byte)) *teeWriter {
	if mu == nil {
		mu = &sync.Mutex{}
	}
	return &teeWriter{dst: dst, mu: mu, hook: hook}
}

func (t *teeWriter) Write(p []byte) (int, error) {
	t.mu.Lock()
	n, err := t.dst.Write(p)
	t.mu.Unlock()
	if t.hook != nil && n > 0 {
		// Copy p so the caller can reuse the buffer.
		chunk := make([]byte, n)
		copy(chunk, p[:n])
		t.hook(chunk)
	}
	return n, err
}

// bashArgs is the input schema for the bash tool.
type bashArgs struct {
	Command string `json:"command" jsonschema:"description=Bash command to execute."`
	// Timeout is the maximum execution time in seconds. Zero or omitted
	// means no timeout.
	Timeout *int `json:"timeout,omitempty" jsonschema:"description=Timeout in seconds. Zero or omitted means no timeout.,minimum=0"`
}

// bashTool implements the "bash" built-in tool.
type bashTool struct {
	ops BashOperations
}

// NewBashTool returns a bash Tool backed by ops. A nil ops defaults to
// OSBashOperations.
func NewBashTool(ops BashOperations) Tool {
	if ops == nil {
		ops = OSBashOperations{}
	}
	return &bashTool{ops: ops}
}

// Name returns the tool's unique identifier.
func (t *bashTool) Name() string { return "bash" }

// Description returns the model-facing description of the tool's behavior.
func (t *bashTool) Description() string {
	return fmt.Sprintf(
		"Execute a bash command in the agent's working directory. stdout and stderr "+
			"are captured and returned. Output is truncated to %d lines and %d KiB "+
			"using head-and-tail truncation (preserving the first and last portions). "+
			"A non-zero exit code produces an IsError result with the exit code in the "+
			"message. Pass timeout (seconds) to bound long-running commands.",
		DefaultMaxLines, DefaultMaxBytes/1024,
	)
}

// Parameters returns the input JSON Schema (draft 2020-12) for the bash tool.
func (t *bashTool) Parameters() jsonschema.Schema {
	return ReflectSchema(&bashArgs{})
}

// Execute runs the bash command and returns the accumulated output. Non-zero
// exit codes produce an IsError result.
func (t *bashTool) Execute(ctx context.Context, call ToolCall) (ToolResult, error) {
	if err := ctx.Err(); err != nil {
		return ToolResult{}, err
	}
	var args bashArgs
	if bad := ParseArgs(call.Args, &args, "bash"); bad != nil {
		return *bad, nil
	}
	if args.Command == "" {
		return NewErrorResult("bash: missing required parameter \"command\""), nil
	}
	if args.Timeout != nil && *args.Timeout < 0 {
		return NewErrorResult(fmt.Sprintf("bash: timeout must be >= 0, got %d", *args.Timeout)), nil
	}
	if call.Cwd == "" {
		return NewErrorResult("bash: missing cwd"), nil
	}

	opts := BashExecOptions{}
	if args.Timeout != nil && *args.Timeout > 0 {
		opts.Timeout = time.Duration(*args.Timeout) * time.Second
	}

	result, err := t.ops.Exec(ctx, args.Command, call.Cwd, opts)
	if err != nil {
		// Context cancellation should propagate as an error, not as an
		// IsError ToolResult — the agent loop distinguishes the two.
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return ToolResult{}, err
		}
		// Timeout, spawn failure, etc.
		text := formatBashFailure(result, err)
		return ToolResult{
			Content: []llm.ContentBlock{llm.TextContent{Text: text}},
			IsError: true,
			Metadata: map[string]any{
				"exitCode": result.ExitCode,
				"signal":   result.Signal,
				"stderr":   string(result.Stderr),
			},
		}, nil
	}

	combined := combineOutput(result.Stdout, result.Stderr)
	truncated := TruncateHeadTail(combined, DefaultMaxLines, DefaultMaxBytes)

	if result.ExitCode != 0 {
		text := truncated
		if text == "" {
			text = "(no output)"
		}
		text = text + "\n\n[Command exited with code " + strconv.Itoa(result.ExitCode) + "]"
		return ToolResult{
			Content: []llm.ContentBlock{llm.TextContent{Text: text}},
			IsError: true,
			Metadata: map[string]any{
				"exitCode": result.ExitCode,
				"signal":   result.Signal,
			},
		}, nil
	}

	if truncated == "" {
		truncated = "(no output)"
	}
	return NewTextResult(truncated), nil
}

// formatBashFailure produces an IsError-friendly summary that includes
// stderr, the cause, and any partial stdout.
func formatBashFailure(result BashExecResult, cause error) string {
	var b strings.Builder
	if len(result.Stdout) > 0 {
		b.WriteString(strings.TrimRight(string(result.Stdout), "\n"))
		b.WriteString("\n")
	}
	if len(result.Stderr) > 0 {
		b.WriteString(strings.TrimRight(string(result.Stderr), "\n"))
		b.WriteString("\n")
	}
	b.WriteString("[")
	b.WriteString(cause.Error())
	b.WriteString("]")
	return b.String()
}

// combineOutput merges stdout and stderr into a single string with a
// separator if both are present.
func combineOutput(stdout, stderr []byte) string {
	var b strings.Builder
	if len(stdout) > 0 {
		b.Write(stdout)
	}
	if len(stderr) > 0 {
		if b.Len() > 0 && !strings.HasSuffix(b.String(), "\n") {
			b.WriteString("\n")
		}
		b.WriteString("[stderr]\n")
		b.Write(stderr)
	}
	return b.String()
}

// RenderCall produces a TUI-friendly representation of the invocation.
// Format: `$ <command> (timeout Ns)` when a timeout is set.
func (t *bashTool) RenderCall(args json.RawMessage, theme *Theme) string {
	var a bashArgs
	_ = json.Unmarshal(args, &a)
	cmd := a.Command
	if cmd == "" {
		cmd = "..."
	}
	out := theme.Wrap(theme.Primary, "$") + " " + theme.Wrap(theme.Accent, cmd)
	if a.Timeout != nil && *a.Timeout > 0 {
		out += " " + theme.Wrap(theme.Muted, fmt.Sprintf("(timeout %ds)", *a.Timeout))
	}
	return out
}

// RenderResult produces a TUI-friendly representation of the result. Errors
// are prefixed in the theme's Error color.
func (t *bashTool) RenderResult(result ToolResult, theme *Theme) string {
	prefix := ""
	if result.IsError {
		prefix = theme.Wrap(theme.Error, "error: ")
	}
	return prefix + renderContentBlocks(result.Content, theme)
}
