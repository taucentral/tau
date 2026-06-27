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

	"github.com/invopop/jsonschema"
)

// ErrOutOfCwd is returned by Write when the target path is outside the
// agent's working directory and the caller did not opt in via
// allowOutOfCwd.
var ErrOutOfCwd = errors.New("tools/write: path is outside cwd (set allowOutOfCwd to override)")

// WriteOperations abstracts the I/O surface of the write tool. The
// default implementation uses the real OS; tests inject fake ones.
type WriteOperations interface {
	// WriteFile writes content atomically (temp + rename). The mode of
	// the existing file (if any) is preserved; new files use mode 0644.
	WriteFile(absolutePath string, content []byte) error
	// Stat returns the os.FileInfo for the path, or os.ErrNotExist if
	// the path does not exist. Used to detect overwrites and to refuse
	// to write to directories.
	Stat(absolutePath string) (fs.FileInfo, error)
}

// OSWriteOperations is the default WriteOperations backed by the real
// filesystem.
type OSWriteOperations struct{}

// WriteFile writes content to a sibling temp file and renames it over
// the target. The temp file lives in the same directory so the rename
// is atomic on POSIX.
func (OSWriteOperations) WriteFile(p string, content []byte) error {
	dir := filepath.Dir(p)
	base := filepath.Base(p)
	tmp, err := os.CreateTemp(dir, "."+base+".*.tmp")
	if err != nil {
		return fmt.Errorf("write: create temp: %v", err)
	}
	tmpName := tmp.Name()
	defer func() { _ = os.Remove(tmpName) }()
	if _, err := tmp.Write(content); err != nil {
		tmp.Close()
		return fmt.Errorf("write: write temp: %v", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("write: close temp: %v", err)
	}
	if fi, err := os.Stat(p); err == nil {
		_ = os.Chmod(tmpName, fi.Mode())
	} else {
		// New file: use a sane default.
		_ = os.Chmod(tmpName, 0644)
	}
	if err := os.Rename(tmpName, p); err != nil {
		return fmt.Errorf("write: rename temp: %v", err)
	}
	return nil
}

// Stat delegates to os.Stat.
func (OSWriteOperations) Stat(p string) (fs.FileInfo, error) {
	return os.Stat(p)
}

// writeArgs is the input schema for the write tool.
type writeArgs struct {
	Path string `json:"path" jsonschema:"description=Path to the file to write. May be relative to the agent cwd or absolute."`
	// Content is the new file contents in full. The previous contents
	// are overwritten.
	Content string `json:"content" jsonschema:"description=The new file contents in full. Replaces any existing content."`
	// AllowOutOfCwd, when true, permits writing to paths outside the
	// agent's cwd. Default false. Set this only when the caller has
	// verified the destination is safe.
	AllowOutOfCwd bool `json:"allowOutOfCwd,omitempty" jsonschema:"description=Allow writes outside the agent cwd. Default false."`
}

// writeTool implements the "write" built-in tool.
type writeTool struct {
	ops   WriteOperations
	queue *FileMutationQueue
}

// NewWriteTool returns a write Tool backed by ops. Calls are serialized
// per-file through a FileMutationQueue. A nil ops defaults to
// OSWriteOperations.
func NewWriteTool(ops WriteOperations) Tool {
	if ops == nil {
		ops = OSWriteOperations{}
	}
	return &writeTool{ops: ops, queue: NewFileMutationQueue()}
}

// Name returns the tool's unique identifier.
func (t *writeTool) Name() string { return "write" }

// Description returns the model-facing description of the tool's behavior.
func (t *writeTool) Description() string {
	return "Write content to a file, overwriting any existing content. Writes are " +
		"atomic (temp + rename) so a crash mid-write never leaves a partial file. " +
		"Paths outside the agent's cwd are refused unless allowOutOfCwd=true is set. " +
		"For in-place edits to existing files, prefer the edit tool."
}

// Parameters returns the input JSON Schema (draft 2020-12) for the write tool.
func (t *writeTool) Parameters() jsonschema.Schema {
	return ReflectSchema(&writeArgs{})
}

// Execute validates args, enforces the cwd guard, and writes content
// atomically.
func (t *writeTool) Execute(ctx context.Context, call ToolCall) (ToolResult, error) {
	if err := ctx.Err(); err != nil {
		return ToolResult{}, err
	}
	var args writeArgs
	if bad := ParseArgs(call.Args, &args, "write"); bad != nil {
		return *bad, nil
	}
	if args.Path == "" {
		return NewErrorResult("write: missing required parameter \"path\""), nil
	}
	if call.Cwd == "" {
		return NewErrorResult("write: missing cwd"), nil
	}

	absolutePath, err := ResolveWithinCwd(args.Path, call.Cwd)
	if err != nil {
		//nolint:nilerr // application error → ToolResult per tool.go contract
		return NewErrorResult("write: " + err.Error()), nil
	}
	if !args.AllowOutOfCwd && !isWithinCwd(absolutePath, call.Cwd) {
		return NewErrorResult(fmt.Errorf(
			"write: %w (%q resolves outside cwd %q)",
			ErrOutOfCwd, args.Path, call.Cwd,
		).Error()), nil
	}

	var result ToolResult
	var execErr error
	queueErr := t.queue.Run(ctx, absolutePath, func() error {
		result, execErr = t.executeLocked(ctx, absolutePath, args)
		return nil
	})
	if queueErr != nil {
		return ToolResult{}, queueErr
	}
	if execErr != nil {
		if errors.Is(execErr, context.Canceled) || errors.Is(execErr, context.DeadlineExceeded) {
			return ToolResult{}, execErr
		}
		return ToolResult{}, execErr
	}
	return result, nil
}

// executeLocked does the actual write while the per-file lock is held.
func (t *writeTool) executeLocked(ctx context.Context, absolutePath string, args writeArgs) (ToolResult, error) {
	if err := ctx.Err(); err != nil {
		return ToolResult{}, err
	}
	// Refuse to write over a directory.
	if fi, err := t.ops.Stat(absolutePath); err == nil && fi.IsDir() {
		return NewErrorResult(fmt.Sprintf("write: %q is a directory", args.Path)), nil
	}

	// Determine the action description for the success message.
	wasOverwrite := false
	if _, err := t.ops.Stat(absolutePath); err == nil {
		wasOverwrite = true
	}

	if err := t.ops.WriteFile(absolutePath, []byte(args.Content)); err != nil {
		return NewErrorResult(fmt.Sprintf("write: %v", err)), nil
	}

	verb := "Wrote"
	if wasOverwrite {
		verb = "Overwrote"
	}
	out := fmt.Sprintf("%s %q (%d bytes).\n", verb, args.Path, len(args.Content))
	return NewTextResult(out), nil
}

// isWithinCwd returns true if target is the same as cwd or a descendant
// of cwd. Both arguments MUST be absolute; relative inputs return false.
func isWithinCwd(target, cwd string) bool {
	if !filepath.IsAbs(target) || !filepath.IsAbs(cwd) {
		return false
	}
	rel, err := filepath.Rel(cwd, target)
	if err != nil {
		return false
	}
	if rel == "." {
		return true
	}
	// filepath.Rel returns ".."-prefixed paths when target is outside cwd.
	if strings.HasPrefix(rel, "..") {
		return false
	}
	// On Windows, paths on different drives produce an error from Rel;
	// already handled above.
	return true
}

// RenderCall produces a TUI-friendly representation of the invocation.
// Format: `write <path> (N bytes)`.
func (t *writeTool) RenderCall(args json.RawMessage, theme *Theme) string {
	var a writeArgs
	_ = json.Unmarshal(args, &a)
	path := a.Path
	if path == "" {
		path = "?"
	}
	out := theme.Wrap(theme.Primary, "write") + " " + theme.Wrap(theme.Accent, path)
	if a.Content != "" {
		out += " " + theme.Wrap(theme.Muted, fmt.Sprintf("(%d bytes)", len(a.Content)))
	}
	if a.AllowOutOfCwd {
		out += " " + theme.Wrap(theme.Warning, "[allow-out-of-cwd]")
	}
	return out
}

// RenderResult produces a TUI-friendly representation of the result.
func (t *writeTool) RenderResult(result ToolResult, theme *Theme) string {
	prefix := ""
	if result.IsError {
		prefix = theme.Wrap(theme.Error, "error: ")
	}
	return prefix + renderContentBlocks(result.Content, theme)
}
