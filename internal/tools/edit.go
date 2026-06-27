package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/invopop/jsonschema"
	diffpkg "github.com/sourcegraph/go-diff/diff"
)

// ErrNotUnique is returned by Edit when oldString appears zero or multiple
// times in the file. The agent loop surfaces this as an IsError result.
var ErrNotUnique = errors.New("tools/edit: oldString is not unique in file")

// ErrNotFound is returned by Edit when oldString does not appear in the file.
var ErrNotFound = errors.New("tools/edit: oldString not found in file")

// EditOperations abstracts the I/O surface of the edit tool. The default
// implementation uses the real OS; tests inject fake implementations.
type EditOperations interface {
	// ReadFile returns the file's current content.
	ReadFile(absolutePath string) ([]byte, error)
	// WriteFile replaces the file's content atomically (temp + rename).
	// The mode of the existing file (if any) is preserved; new files use
	// mode 0644.
	WriteFile(absolutePath string, content []byte) error
	// Access verifies the file is readable AND writable.
	Access(absolutePath string) error
}

// OSEditOperations is the default EditOperations backed by the real
// filesystem.
type OSEditOperations struct{}

// ReadFile delegates to os.ReadFile.
func (OSEditOperations) ReadFile(p string) ([]byte, error) { return os.ReadFile(p) }

// WriteFile writes content to a sibling temp file and renames it over
// the target. The temp file lives in the same directory so the rename
// is atomic on POSIX (same filesystem).
func (OSEditOperations) WriteFile(p string, content []byte) error {
	dir := filepath.Dir(p)
	base := filepath.Base(p)
	tmp, err := os.CreateTemp(dir, "."+base+".*.tmp")
	if err != nil {
		return fmt.Errorf("edit: create temp: %v", err)
	}
	tmpName := tmp.Name()
	// Best-effort cleanup if anything below fails.
	defer func() { _ = os.Remove(tmpName) }()
	if _, err := tmp.Write(content); err != nil {
		tmp.Close()
		return fmt.Errorf("edit: write temp: %v", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("edit: close temp: %v", err)
	}
	// Inherit mode from the existing file if possible.
	if fi, err := os.Stat(p); err == nil {
		_ = os.Chmod(tmpName, fi.Mode())
	}
	if err := os.Rename(tmpName, p); err != nil {
		return fmt.Errorf("edit: rename temp: %v", err)
	}
	return nil
}

// Access verifies the file is readable and writable. For a missing file,
// it returns os.ErrNotExist (callers must distinguish "file doesn't exist
// yet, refuse to create" from "file exists but is read-only").
func (OSEditOperations) Access(p string) error {
	fi, err := os.Stat(p)
	if err != nil {
		return err
	}
	if fi.IsDir() {
		return fmt.Errorf("edit: %q is a directory", p)
	}
	// Try opening for read/write to flush out permission issues.
	f, err := os.OpenFile(p, os.O_RDWR, 0)
	if err != nil {
		return err
	}
	return f.Close()
}

// editArgs is the input schema for the edit tool.
type editArgs struct {
	Path      string `json:"path" jsonschema:"description=Path to the file to edit. May be relative to the agent cwd or absolute."`
	OldString string `json:"oldString" jsonschema:"description=Exact text to find in the file. Must appear exactly once."`
	NewString string `json:"newString" jsonschema:"description=Replacement text. May be empty to delete oldString."`
}

// editTool implements the "edit" built-in tool.
type editTool struct {
	ops   EditOperations
	queue *FileMutationQueue
}

// NewEditTool returns an edit Tool backed by ops. Calls are serialized
// per-file through a FileMutationQueue so concurrent edits to the same
// file don't race. A nil ops defaults to OSEditOperations.
func NewEditTool(ops EditOperations) Tool {
	if ops == nil {
		ops = OSEditOperations{}
	}
	return &editTool{ops: ops, queue: NewFileMutationQueue()}
}

// Name returns the tool's unique identifier.
func (t *editTool) Name() string { return "edit" }

// Description returns the model-facing description of the tool's behavior.
func (t *editTool) Description() string {
	return "Replace a unique block of text in a file. The oldString MUST appear " +
		"exactly once; if it appears zero or multiple times, the edit is refused " +
		"with an IsError result. The file is written atomically (temp + rename) " +
		"and the result includes a unified diff of the change."
}

// Parameters returns the input JSON Schema (draft 2020-12) for the edit tool.
func (t *editTool) Parameters() jsonschema.Schema {
	return ReflectSchema(&editArgs{})
}

// Execute validates arguments, acquires the per-file mutation lock, finds
// oldString in the file, replaces it with newString, writes the result
// atomically, and returns a unified diff.
func (t *editTool) Execute(ctx context.Context, call ToolCall) (ToolResult, error) {
	if err := ctx.Err(); err != nil {
		return ToolResult{}, err
	}
	var args editArgs
	if bad := ParseArgs(call.Args, &args, "edit"); bad != nil {
		return *bad, nil
	}
	if args.Path == "" {
		return NewErrorResult("edit: missing required parameter \"path\""), nil
	}
	if args.OldString == "" {
		return NewErrorResult("edit: missing required parameter \"oldString\" (empty oldString is not allowed)"), nil
	}
	if args.OldString == args.NewString {
		return NewErrorResult("edit: oldString and newString are identical; nothing to do"), nil
	}
	if call.Cwd == "" {
		return NewErrorResult("edit: missing cwd"), nil
	}

	absolutePath, err := ResolveWithinCwd(args.Path, call.Cwd)
	if err != nil {
		//nolint:nilerr // application error → ToolResult per tool.go contract
		return NewErrorResult("edit: " + err.Error()), nil
	}

	// Serialize per-file.
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
		// Shouldn't happen — executeLocked converts errors to ToolResult.
		return ToolResult{}, execErr
	}
	return result, nil
}

// executeLocked does the read-modify-write while the per-file lock is held.
// All application-level errors are returned as IsError ToolResult, not as
// Go errors; only ctx cancellation bubbles up.
func (t *editTool) executeLocked(ctx context.Context, absolutePath string, args editArgs) (ToolResult, error) {
	if err := ctx.Err(); err != nil {
		return ToolResult{}, err
	}
	if err := t.ops.Access(absolutePath); err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return NewErrorResult(fmt.Sprintf(
				"edit: %q does not exist. Use the write tool to create new files.",
				args.Path,
			)), nil
		}
		return NewErrorResult(fmt.Sprintf("edit: cannot access %q: %v", args.Path, err)), nil
	}

	original, err := t.ops.ReadFile(absolutePath)
	if err != nil {
		return NewErrorResult(fmt.Sprintf("edit: read %q: %v", args.Path, err)), nil
	}

	count := bytes.Count(original, []byte(args.OldString))
	if count == 0 {
		return NewErrorResult(fmt.Errorf(
			"edit: %w (oldString was not found in %q)",
			ErrNotFound, args.Path,
		).Error()), nil
	}
	if count > 1 {
		return NewErrorResult(fmt.Errorf(
			"edit: %w (oldString appeared %d times in %q)",
			ErrNotUnique, count, args.Path,
		).Error()), nil
	}

	updated := bytes.Replace(original, []byte(args.OldString), []byte(args.NewString), 1)

	if err := t.ops.WriteFile(absolutePath, updated); err != nil {
		return NewErrorResult(fmt.Sprintf("edit: write %q: %v", args.Path, err)), nil
	}

	diffText, diffErr := computeUnifiedDiff(args.Path, original, updated)
	if diffErr != nil {
		// Don't fail the edit if diff computation breaks; just omit the diff.
		diffText = ""
	}

	header := fmt.Sprintf("Edited %q (%d → %d bytes).\n", args.Path, len(original), len(updated))
	out := header + diffText
	return NewTextResult(out), nil
}

// computeUnifiedDiff returns a unified diff between original and updated.
// Both are byte slices; the diff is computed line-wise. The path is used
// as the file name in the diff header.
func computeUnifiedDiff(displayPath string, original, updated []byte) (string, error) {
	hunk := buildDiffHunk(original, updated)
	if hunk == nil {
		return "", nil
	}
	fd := &diffpkg.FileDiff{
		OrigName: displayPath,
		NewName:  displayPath,
		Hunks:    []*diffpkg.Hunk{hunk},
	}
	out, err := diffpkg.PrintFileDiff(fd)
	if err != nil {
		return "", err
	}
	return string(out), nil
}

// buildDiffHunk constructs a single go-diff Hunk covering the change from
// original to updated. It uses a simple whole-file diff (no Myers). The
// hunk's body lines are prefixed with '-' for removed and '+' for added.
func buildDiffHunk(original, updated []byte) *diffpkg.Hunk {
	origLines := splitLines(original)
	newLines := splitLines(updated)

	// Find common prefix.
	commonPrefix := 0
	maxPrefix := len(origLines)
	if len(newLines) < maxPrefix {
		maxPrefix = len(newLines)
	}
	for commonPrefix < maxPrefix && origLines[commonPrefix] == newLines[commonPrefix] {
		commonPrefix++
	}
	// Find common suffix.
	commonSuffix := 0
	maxSuffix := len(origLines) - commonPrefix
	if len(newLines)-commonPrefix < maxSuffix {
		maxSuffix = len(newLines) - commonPrefix
	}
	for commonSuffix < maxSuffix &&
		origLines[len(origLines)-1-commonSuffix] == newLines[len(newLines)-1-commonSuffix] {
		commonSuffix++
	}

	removed := origLines[commonPrefix : len(origLines)-commonSuffix]
	added := newLines[commonPrefix : len(newLines)-commonSuffix]
	if len(removed) == 0 && len(added) == 0 {
		return nil
	}

	var body bytes.Buffer
	for _, l := range removed {
		body.WriteString("-")
		body.WriteString(l)
		if !strings.HasSuffix(l, "\n") {
			body.WriteString("\n")
			body.WriteString("\\ No newline at end of file\n")
		}
	}
	for _, l := range added {
		body.WriteString("+")
		body.WriteString(l)
		if !strings.HasSuffix(l, "\n") {
			body.WriteString("\n")
			body.WriteString("\\ No newline at end of file\n")
		}
	}

	// Hunk header is 1-indexed for both start lines. When the change is
	// an insertion (no removed lines), the start line is the line number
	// AFTER which the insertion happens — which is commonPrefix+1 in
	// 1-indexed terms. Standard diff convention: if removed count is 0,
	// origStartLine is commonPrefix (0-indexed) → +1 makes it the line
	// before the insertion; some tools use commonPrefix+1 instead. We
	// follow GNU diff: count 0 means the start line is commonPrefix.
	origStart := int32(commonPrefix)
	newStart := int32(commonPrefix)
	if len(removed) > 0 {
		origStart++ // 1-indexed
	}
	if len(added) > 0 {
		newStart++ // 1-indexed
	}
	// GNU diff convention: when there are zero lines on a side, the
	// start line number is the line BEFORE the change (one less than
	// the would-be 1-indexed line). Achieve that by NOT incrementing
	// when count is 0.
	if len(removed) == 0 {
		origStart = int32(commonPrefix)
	}
	if len(added) == 0 {
		newStart = int32(commonPrefix)
	}

	return &diffpkg.Hunk{
		OrigStartLine: origStart,
		OrigLines:     int32(len(removed)),
		NewStartLine:  newStart,
		NewLines:      int32(len(added)),
		Body:          body.Bytes(),
		// Section name omitted; not needed for tool-output display.
	}
}

// splitLines splits b into lines, preserving the trailing newline on each
// line. The last element may lack a newline if b doesn't end with one.
// Returns nil for empty input.
func splitLines(b []byte) []string {
	if len(b) == 0 {
		return nil
	}
	lines := []string{}
	start := 0
	for i := 0; i < len(b); i++ {
		if b[i] == '\n' {
			lines = append(lines, string(b[start:i+1]))
			start = i + 1
		}
	}
	if start < len(b) {
		lines = append(lines, string(b[start:]))
	}
	return lines
}

// RenderCall produces a TUI-friendly representation of the invocation.
// Format: `edit <path>` on a single line; full edit body shown by TUI
// expansion downstream.
func (t *editTool) RenderCall(args json.RawMessage, theme *Theme) string {
	var a editArgs
	_ = json.Unmarshal(args, &a)
	path := a.Path
	if path == "" {
		path = "?"
	}
	out := theme.Wrap(theme.Primary, "edit") + " " + theme.Wrap(theme.Accent, path)
	if a.OldString != "" {
		// Summarize the oldString: first non-empty line, truncated.
		first := a.OldString
		if i := strings.Index(first, "\n"); i >= 0 {
			first = first[:i]
		}
		if len(first) > 60 {
			first = first[:60] + "..."
		}
		out += " " + theme.Wrap(theme.Muted, strconv.Quote(first))
	}
	return out
}

// RenderResult produces a TUI-friendly representation of the result.
// Errors are prefixed in the theme's Error color; success shows the diff
// with +/- lines tinted.
func (t *editTool) RenderResult(result ToolResult, theme *Theme) string {
	prefix := ""
	if result.IsError {
		prefix = theme.Wrap(theme.Error, "error: ")
	}
	body := renderContentBlocks(result.Content, theme)
	if !result.IsError {
		// Tint diff lines.
		var b strings.Builder
		for _, line := range strings.Split(body, "\n") {
			switch {
			case strings.HasPrefix(line, "-") && !strings.HasPrefix(line, "---"):
				b.WriteString(theme.Wrap(theme.Error, line))
			case strings.HasPrefix(line, "+") && !strings.HasPrefix(line, "+++"):
				b.WriteString(theme.Wrap(theme.Success, line))
			default:
				b.WriteString(line)
			}
			b.WriteString("\n")
		}
		body = strings.TrimRight(b.String(), "\n")
	}
	return prefix + body
}
