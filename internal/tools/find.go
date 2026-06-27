package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/bmatcuk/doublestar/v4"
	"github.com/invopop/jsonschema"
)

// DefaultFindLimit caps the number of paths returned by a single find
// invocation. Find output is one path per line; thousands of matches would
// crowd the LLM context. The user can override via the limit parameter.
const DefaultFindLimit = 1000

// ErrFindPathNotFound is returned by find when the target path does not
// exist.
var ErrFindPathNotFound = errors.New("tools/find: path not found")

// ErrFindInvalidPattern is returned by find when the pattern fails to
// compile as a doublestar glob.
var ErrFindInvalidPattern = errors.New("tools/find: invalid pattern")

// FindOperations abstracts the I/O surface of the find tool. The default
// implementation uses the real OS; tests inject fake implementations to
// avoid filesystem side effects.
type FindOperations interface {
	// IsDirectory returns true if absolutePath exists and is a directory.
	// Returns os.ErrNotExist (wrapped) if the path is missing.
	IsDirectory(absolutePath string) (bool, error)
	// WalkDir walks the file tree rooted at root, invoking fn for each
	// file and directory. Implementations MUST honor gitignore-style
	// external filtering (the caller applies gitignore internally); this
	// is just a directory traversal primitive.
	WalkDir(root string, fn fs.WalkDirFunc) error
}

// OSFindOperations is the default FindOperations backed by the real OS.
type OSFindOperations struct{}

// IsDirectory delegates to os.Stat and reports IsDir.
func (OSFindOperations) IsDirectory(p string) (bool, error) {
	fi, err := os.Stat(p)
	if err != nil {
		return false, err
	}
	return fi.IsDir(), nil
}

// WalkDir delegates to filepath.WalkDir.
func (OSFindOperations) WalkDir(root string, fn fs.WalkDirFunc) error {
	return filepath.WalkDir(root, fn)
}

// findArgs is the input schema for the find tool.
type findArgs struct {
	// Pattern is a glob interpreted with bmatcuk/doublestar semantics
	// (supports `**` for "any number of path segments"). It is matched
	// against the path relative to the search root, with forward slashes
	// on all platforms. Examples: "**/*.go" matches any .go file at any
	// depth; "*.go" matches only top-level .go files; "src/**/*.test.ts"
	// matches .test.ts files under any src subdirectory.
	Pattern string `json:"pattern" jsonschema:"description=Glob pattern (doublestar semantics; ** matches any number of dirs). Matched against the path relative to the search root."`
	// Path is the directory to search. Defaults to the agent cwd.
	Path string `json:"path,omitempty" jsonschema:"description=Directory to search (default: current directory)."`
	// Limit caps the number of paths returned. Zero means the default
	// (DefaultFindLimit); negative is rejected.
	Limit int `json:"limit,omitempty" jsonschema:"description=Maximum number of paths to return (default 1000).,minimum=0"`
}

// findTool implements the "find" built-in tool.
type findTool struct {
	ops FindOperations
}

// NewFindTool returns a find Tool backed by ops. The tool walks the search
// root, applies gitignore rules in-process, and matches the user's pattern
// via bmatcuk/doublestar. A nil ops defaults to OSFindOperations.
func NewFindTool(ops FindOperations) Tool {
	if ops == nil {
		ops = OSFindOperations{}
	}
	return &findTool{ops: ops}
}

// Name returns the tool's unique identifier.
func (t *findTool) Name() string { return "find" }

// Description returns the model-facing description of the tool's behavior.
func (t *findTool) Description() string {
	return fmt.Sprintf(
		"Find files matching a glob pattern under the current directory (or a given path). "+
			"Uses doublestar semantics: `**` matches any number of directories (including zero). "+
			"Respects .gitignore by default (also skips the .git directory). Output is one path "+
			"per line, sorted lexically, capped at %d results by default.",
		DefaultFindLimit,
	)
}

// Parameters returns the input JSON Schema (draft 2020-12) for the find tool.
func (t *findTool) Parameters() jsonschema.Schema {
	return ReflectSchema(&findArgs{})
}

// Execute validates args, resolves the search root, walks the tree while
// applying gitignore rules, and returns matching paths.
func (t *findTool) Execute(ctx context.Context, call ToolCall) (ToolResult, error) {
	if err := ctx.Err(); err != nil {
		return ToolResult{}, err
	}
	var args findArgs
	if bad := ParseArgs(call.Args, &args, "find"); bad != nil {
		return *bad, nil
	}
	if args.Pattern == "" {
		return NewErrorResult("find: missing required parameter \"pattern\""), nil
	}
	if args.Limit < 0 {
		return NewErrorResult(fmt.Sprintf("find: limit must be >= 0 (got %d)", args.Limit)), nil
	}
	if call.Cwd == "" {
		return NewErrorResult("find: missing cwd"), nil
	}

	// Validate the pattern up front via doublestar.Match against a literal
	// sentinel so a malformed pattern surfaces as "invalid pattern" rather
	// than a confusing "path not found" later.
	if _, err := doublestar.Match(args.Pattern, "x"); err != nil {
		return NewErrorResult(fmt.Errorf("find: %w: %v", ErrFindInvalidPattern, err).Error()), nil
	}

	searchPath := args.Path
	if searchPath == "" {
		searchPath = "."
	}
	absoluteRoot, err := ResolveWithinCwd(searchPath, call.Cwd)
	if err != nil {
		//nolint:nilerr // application error → ToolResult per tool.go contract
		return NewErrorResult("find: " + err.Error()), nil
	}
	isDir, err := t.ops.IsDirectory(absoluteRoot)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return NewErrorResult(fmt.Errorf("find: %w (%q)", ErrFindPathNotFound, args.Path).Error()), nil
		}
		return NewErrorResult(fmt.Sprintf("find: stat %q: %v", args.Path, err)), nil
	}
	if !isDir {
		return NewErrorResult(fmt.Sprintf("find: %q is not a directory", args.Path)), nil
	}

	limit := args.Limit
	if limit == 0 {
		limit = DefaultFindLimit
	}
	if limit < 1 {
		limit = 1
	}

	matches, walkErr := t.walkAndCollect(ctx, args.Pattern, absoluteRoot, limit)
	if walkErr != nil {
		if errors.Is(walkErr, context.Canceled) || errors.Is(walkErr, context.DeadlineExceeded) {
			return ToolResult{}, walkErr
		}
		return NewErrorResult(fmt.Sprintf("find: walk %q: %v", args.Path, walkErr)), nil
	}

	if len(matches) == 0 {
		return NewTextResult("No files found"), nil
	}

	out := formatFindResults(matches, absoluteRoot, call.Cwd)
	capped := TruncateHeadTail(out, DefaultMaxLines, DefaultMaxBytes)
	notices := buildFindNotices(len(matches), limit, out, capped)
	return NewTextResult(appendNotices(capped, notices)), nil
}

// walkAndCollect walks absoluteRoot, applies gitignore rules, matches each
// file against pattern, and returns matching absolute paths sorted lexically.
// At most `limit` matches are returned; further matches are dropped silently
// (the caller adds a notice).
func (t *findTool) walkAndCollect(ctx context.Context, pattern, absoluteRoot string, limit int) ([]string, error) {
	var matches []string
	ignoreStack := newIgnoreStack()
	walkErr := t.ops.WalkDir(absoluteRoot, func(p string, d fs.DirEntry, werr error) error {
		if werr != nil {
			return nil //nolint:nilerr // skip unreadable entries per filepath.WalkDir convention
		}
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		rel, rerr := filepath.Rel(absoluteRoot, p)
		if rerr != nil {
			rel = p
		}
		if d.IsDir() {
			base := d.Name()
			if base == ".git" {
				return fs.SkipDir
			}
			ignoreStack.push(p)
			if ignore, ok := tryReadGitignore(filepath.Join(p, ".gitignore")); ok {
				ignoreStack.add(ignore, p)
			}
			if ignoreStack.matchesDir(rel) {
				return fs.SkipDir
			}
			return nil
		}
		if !d.Type().IsRegular() {
			return nil
		}
		if ignoreStack.matchesFile(rel) {
			return nil
		}
		if matchFindPattern(pattern, rel) {
			matches = append(matches, p)
			if len(matches) >= limit {
				return errFindLimitReached
			}
		}
		return nil
	})
	if walkErr != nil && !errors.Is(walkErr, errFindLimitReached) {
		return nil, walkErr
	}
	sort.Strings(matches)
	return matches, nil
}

// errFindLimitReached is an internal sentinel used to short-circuit a walk
// once the caller's limit is hit. filepath.WalkDir accepts any error from
// the callback; returning fs.SkipDir would only skip one subtree, so we
// return a typed error and have the caller treat it as "stop, not a real
// failure".
var errFindLimitReached = errors.New("tools/find: limit reached")

// matchFindPattern reports whether relPath matches pattern under
// doublestar semantics. The pattern and path are normalized to forward
// slashes before matching so behavior is identical on POSIX and Windows.
func matchFindPattern(pattern, relPath string) bool {
	pattern = filepath.ToSlash(pattern)
	relPath = filepath.ToSlash(relPath)
	matched, err := doublestar.Match(pattern, relPath)
	return err == nil && matched
}

// formatFindResults renders matches as one path per line. Paths are
// displayed relative to cwd when possible (readability for the model);
// otherwise relative to the walk root; otherwise absolute.
func formatFindResults(matches []string, absoluteRoot, cwd string) string {
	var b strings.Builder
	for i, m := range matches {
		if i > 0 {
			b.WriteString("\n")
		}
		b.WriteString(displayPath(m, absoluteRoot, true, cwd))
	}
	return b.String()
}

// buildFindNotices assembles the `[...]` suffix appended to truncated find
// output. matchCount is the number of matches found; limit is the cap; raw
// is the pre-truncation output; capped is the post-truncation output.
func buildFindNotices(matchCount, limit int, raw, capped string) []string {
	var notices []string
	if matchCount >= limit {
		notices = append(notices, fmt.Sprintf(
			"%d paths limit reached. Use limit=%d for more, or refine pattern",
			limit, limit*2,
		))
	}
	if len(capped) < len(raw) {
		notices = append(notices, fmt.Sprintf("%d KiB output limit reached", DefaultMaxBytes/1024))
	}
	return notices
}

// RenderCall produces a TUI-friendly representation of the invocation.
// Format: `find /pattern/ in <path> [limit N]`.
func (t *findTool) RenderCall(args json.RawMessage, theme *Theme) string {
	var a findArgs
	_ = json.Unmarshal(args, &a)
	out := theme.Wrap(theme.Primary, "find") + " " + theme.Wrap(theme.Accent, "/"+a.Pattern+"/")
	target := a.Path
	if target == "" {
		target = "."
	}
	out += " " + theme.Wrap(theme.Muted, "in "+target)
	if a.Limit > 0 {
		out += " " + theme.Wrap(theme.Muted, fmt.Sprintf("limit %d", a.Limit))
	}
	return out
}

// RenderResult produces a TUI-friendly representation of the result.
func (t *findTool) RenderResult(result ToolResult, theme *Theme) string {
	prefix := ""
	if result.IsError {
		prefix = theme.Wrap(theme.Error, "error: ")
	}
	return prefix + renderContentBlocks(result.Content, theme)
}
