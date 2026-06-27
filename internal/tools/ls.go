package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
)

// LSTimeLayout is the timestamp format used in ls output. Picked to be
// compact and unambiguous across locales; the model can parse it directly.
const LSTimeLayout = "2006-01-02 15:04"

// ErrLSPathNotFound is returned by ls when the target path does not exist.
var ErrLSPathNotFound = errors.New("tools/ls: path not found")

// ErrLSNotADirectory is returned by ls when the target path is not a
// directory. ls lists the contents of a single directory; use read for
// individual files.
var ErrLSNotADirectory = errors.New("tools/ls: not a directory")

// LSOperations abstracts the I/O surface of the ls tool. The default
// implementation uses the real OS; tests inject fake implementations to
// avoid filesystem side effects.
type LSOperations interface {
	// IsDirectory returns true if absolutePath exists and is a directory.
	IsDirectory(absolutePath string) (bool, error)
	// ReadDir returns the entries within absolutePath, sorted by name.
	// The returned entries expose Name, IsDir, and Info via fs.DirEntry.
	ReadDir(absolutePath string) ([]fs.DirEntry, error)
}

// OSLSOperations is the default LSOperations backed by the real OS.
type OSLSOperations struct{}

// IsDirectory delegates to os.Stat and reports IsDir.
func (OSLSOperations) IsDirectory(p string) (bool, error) {
	fi, err := os.Stat(p)
	if err != nil {
		return false, err
	}
	return fi.IsDir(), nil
}

// ReadDir delegates to os.ReadDir.
func (OSLSOperations) ReadDir(p string) ([]fs.DirEntry, error) {
	return os.ReadDir(p)
}

// lsArgs is the input schema for the ls tool.
type lsArgs struct {
	// Path is the directory to list. Defaults to the agent cwd.
	Path string `json:"path,omitempty" jsonschema:"description=Directory to list (default: current directory)."`
}

// lsTool implements the "ls" built-in tool.
type lsTool struct {
	ops LSOperations
}

// NewLSTool returns an ls Tool backed by ops. A nil ops defaults to
// OSLSOperations.
func NewLSTool(ops LSOperations) Tool {
	if ops == nil {
		ops = OSLSOperations{}
	}
	return &lsTool{ops: ops}
}

// Name returns the tool's unique identifier.
func (t *lsTool) Name() string { return "ls" }

// Description returns the model-facing description of the tool's behavior.
func (t *lsTool) Description() string {
	return fmt.Sprintf(
		"List the contents of a single directory. Returns one entry per line with "+
			"a type indicator (`d` for directory, `-` for file), name (directories "+
			"carry a trailing slash), size in bytes, and modification time formatted "+
			"as %q. Entries are sorted with directories first, then files, "+
			"alphabetically within each group. Output is capped at %d lines / %d KiB. "+
			"For recursive listings use the find tool.",
		LSTimeLayout, DefaultMaxLines, DefaultMaxBytes/1024,
	)
}

// Parameters returns the input JSON Schema (draft 2020-12) for the ls tool.
func (t *lsTool) Parameters() jsonschema.Schema {
	return ReflectSchema(&lsArgs{})
}

// Execute validates args, resolves the target directory, reads it, and
// returns a formatted listing.
func (t *lsTool) Execute(ctx context.Context, call ToolCall) (ToolResult, error) {
	if err := ctx.Err(); err != nil {
		return ToolResult{}, err
	}
	var args lsArgs
	if bad := ParseArgs(call.Args, &args, "ls"); bad != nil {
		return *bad, nil
	}
	if call.Cwd == "" {
		return NewErrorResult("ls: missing cwd"), nil
	}

	searchPath := args.Path
	if searchPath == "" {
		searchPath = "."
	}
	absolutePath, err := ResolveWithinCwd(searchPath, call.Cwd)
	if err != nil {
		//nolint:nilerr // application error → ToolResult per tool.go contract
		return NewErrorResult("ls: " + err.Error()), nil
	}

	isDir, err := t.ops.IsDirectory(absolutePath)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return NewErrorResult(fmt.Errorf("ls: %w (%q)", ErrLSPathNotFound, args.Path).Error()), nil
		}
		return NewErrorResult(fmt.Sprintf("ls: stat %q: %v", args.Path, err)), nil
	}
	if !isDir {
		return NewErrorResult(fmt.Errorf("ls: %w (%q)", ErrLSNotADirectory, args.Path).Error()), nil
	}

	entries, err := t.ops.ReadDir(absolutePath)
	if err != nil {
		return NewErrorResult(fmt.Sprintf("ls: read %q: %v", args.Path, err)), nil
	}

	rows := t.buildRows(entries)
	if len(rows) == 0 {
		return NewTextResult("(empty directory)"), nil
	}
	out := formatLSRows(rows)
	capped := TruncateHeadTail(out, DefaultMaxLines, DefaultMaxBytes)
	notices := buildLSNotices(len(rows), out, capped)
	return NewTextResult(appendNotices(capped, notices)), nil
}

// lsRow is the per-entry record used to format ls output.
type lsRow struct {
	typeIndicator string // "d" or "-"
	name          string // entry name; directories carry a trailing slash
	size          int64
	mtime         time.Time
}

// buildRows converts directory entries to lsRows, sorted directories first
// then files, alphabetically within each group. Hidden entries (leading
// dot) are included; the spec describes ls as a plain listing tool, and
// hiding dotfiles would surprise the model. Callers wanting gitignore
// semantics should use find.
func (t *lsTool) buildRows(entries []fs.DirEntry) []lsRow {
	var dirs, files []lsRow
	for _, e := range entries {
		info, err := e.Info()
		// If Info() fails (e.g., dangling symlink), fall back to a
		// zero-value FileInfo so we still list the entry by name.
		if err != nil {
			row := lsRow{typeIndicator: typeIndicatorFor(e.IsDir(), e.Type()), name: lsDisplayName(e.Name(), e.IsDir())}
			if e.IsDir() {
				dirs = append(dirs, row)
			} else {
				files = append(files, row)
			}
			continue
		}
		row := lsRow{
			typeIndicator: typeIndicatorFor(info.IsDir(), info.Mode()),
			name:          lsDisplayName(info.Name(), info.IsDir()),
			size:          info.Size(),
			mtime:         info.ModTime(),
		}
		if info.IsDir() {
			dirs = append(dirs, row)
		} else {
			files = append(files, row)
		}
	}
	sort.Slice(dirs, func(i, j int) bool { return dirs[i].name < dirs[j].name })
	sort.Slice(files, func(i, j int) bool { return files[i].name < files[j].name })
	return append(dirs, files...)
}

// typeIndicatorFor returns "d" for directories and "-" for anything else.
// Special file types (symlinks, devices, sockets) all collapse to "-" so
// the column stays binary; mode permission bits are intentionally not
// surfaced (the model can use bash for `ls -l` if it needs them).
func typeIndicatorFor(isDir bool, mode os.FileMode) string {
	if isDir {
		return "d"
	}
	if mode&os.ModeSymlink != 0 {
		return "l"
	}
	return "-"
}

// lsDisplayName appends a trailing slash to directory names so the type
// is unambiguous even when the type column is hidden (e.g., when the
// model quotes the name back to read/edit).
func lsDisplayName(name string, isDir bool) string {
	if isDir {
		return name + "/"
	}
	return name
}

// formatLSRows renders rows in fixed-width columns. Widths are derived
// from the actual data so every column lines up regardless of name length.
// Output is the type indicator, name (padded), size (right-aligned), and
// mtime.
func formatLSRows(rows []lsRow) string {
	maxName := len("name")
	maxSize := len("size")
	maxType := len("t")
	for _, r := range rows {
		if len(r.typeIndicator) > maxType {
			maxType = len(r.typeIndicator)
		}
		if len(r.name) > maxName {
			maxName = len(r.name)
		}
		if w := len(formatLSSize(r.size)); w > maxSize {
			maxSize = w
		}
	}
	// Cap name column to keep output readable when one entry has an
	// abnormally long name (e.g., a minified bundle). 60 chars matches
	// common terminal width headroom; longer names still display, just
	// not padded to their full length.
	if maxName > 60 {
		maxName = 60
	}
	var b strings.Builder
	for i, r := range rows {
		if i > 0 {
			b.WriteString("\n")
		}
		name := r.name
		if len(name) > maxName {
			name = name[:maxName]
		}
		fmt.Fprintf(&b, "%-*s  %-*s  %s  %s",
			maxType, r.typeIndicator,
			maxName, name,
			formatLSPaddedSize(r.size, maxSize),
			r.mtime.Format(LSTimeLayout),
		)
	}
	return b.String()
}

// formatLSSize renders a byte count for the ls column. Directories (size<0)
// render as "-" since directory sizes are filesystem-dependent and not
// meaningful for the model.
func formatLSSize(size int64) string {
	if size < 0 {
		return "-"
	}
	return fmt.Sprintf("%d", size)
}

// formatLSPaddedSize is formatLSSize with right-padding to width.
func formatLSPaddedSize(size int64, width int) string {
	s := formatLSSize(size)
	if len(s) >= width {
		return s
	}
	return strings.Repeat(" ", width-len(s)) + s
}

// buildLSNotices assembles the `[...]` suffix appended to truncated ls
// output. rowCount is the number of entries that produced the output;
// raw is the pre-truncation output; capped is the post-truncation output.
func buildLSNotices(rowCount int, raw, capped string) []string {
	var notices []string
	if len(capped) < len(raw) {
		notices = append(notices, fmt.Sprintf(
			"output truncated: directory has %d entries, %d KiB limit reached. Use a more specific path",
			rowCount, DefaultMaxBytes/1024,
		))
	}
	return notices
}

// RenderCall produces a TUI-friendly representation of the invocation.
func (t *lsTool) RenderCall(args json.RawMessage, theme *Theme) string {
	var a lsArgs
	_ = json.Unmarshal(args, &a)
	out := theme.Wrap(theme.Primary, "ls")
	target := a.Path
	if target == "" {
		target = "."
	}
	return out + " " + theme.Wrap(theme.Muted, target)
}

// RenderResult produces a TUI-friendly representation of the result.
func (t *lsTool) RenderResult(result ToolResult, theme *Theme) string {
	prefix := ""
	if result.IsError {
		prefix = theme.Wrap(theme.Error, "error: ")
	}
	return prefix + renderContentBlocks(result.Content, theme)
}
