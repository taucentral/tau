package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"

	"github.com/invopop/jsonschema"
	gitignore "github.com/sabhiram/go-gitignore"
)

// GrepMaxLineLength is the per-line character cap applied to grep output
// before it reaches the LLM. Lines longer than this are truncated with an
// ellipsis marker so a single pathological match cannot blow the context
// budget. Matches pi's GREP_MAX_LINE_LENGTH.
const GrepMaxLineLength = 2000

// DefaultGrepLimit is the default cap on the number of matches returned by
// a single grep invocation. Matches pi's DEFAULT_LIMIT.
const DefaultGrepLimit = 100

// ErrGrepPathNotFound is returned by Grep when the target path does not
// exist.
var ErrGrepPathNotFound = errors.New("tools/grep: path not found")

// ErrGrepInvalidPattern is returned by Grep when the pattern fails to
// compile as a regular expression.
var ErrGrepInvalidPattern = errors.New("tools/grep: invalid pattern")

// GrepOperations abstracts the I/O surface of the grep tool. The default
// implementation uses the real OS and shells out to ripgrep when available;
// tests inject fake implementations to avoid filesystem and process side
// effects.
type GrepOperations interface {
	// IsDirectory returns true if absolutePath exists and is a directory.
	// Returns os.ErrNotExist (wrapped) if the path is missing.
	IsDirectory(absolutePath string) (bool, error)
	// ReadFile returns the full content of the file at absolutePath.
	ReadFile(absolutePath string) ([]byte, error)
	// WalkDir walks the file tree rooted at root, invoking fn for each
	// file and directory. Implementations MUST honor gitignore-style
	// external filtering (the caller applies gitignore internally); this
	// is just a directory traversal primitive.
	WalkDir(root string, fn fs.WalkDirFunc) error
	// LookPath searches for the named executable on PATH. Returns the
	// full path on success, or an error matching exec.ErrNotFound on
	// failure. Used to detect whether ripgrep is installed.
	LookPath(name string) (string, error)
}

// OSGrepOperations is the default GrepOperations backed by the real OS.
type OSGrepOperations struct{}

// IsDirectory delegates to os.Stat and reports IsDir.
func (OSGrepOperations) IsDirectory(p string) (bool, error) {
	fi, err := os.Stat(p)
	if err != nil {
		return false, err
	}
	return fi.IsDir(), nil
}

// ReadFile delegates to os.ReadFile.
func (OSGrepOperations) ReadFile(p string) ([]byte, error) { return os.ReadFile(p) }

// WalkDir delegates to filepath.WalkDir.
func (OSGrepOperations) WalkDir(root string, fn fs.WalkDirFunc) error {
	return filepath.WalkDir(root, fn)
}

// LookPath delegates to exec.LookPath.
func (OSGrepOperations) LookPath(name string) (string, error) { return exec.LookPath(name) }

// grepArgs is the input schema for the grep tool.
type grepArgs struct {
	// Pattern is the search pattern. Interpreted as a regular expression
	// by default; pass Literal=true to do a literal substring search.
	Pattern string `json:"pattern" jsonschema:"description=Search pattern (regex or literal string)."`
	// Path is the directory or file to search. Defaults to the agent cwd.
	Path string `json:"path,omitempty" jsonschema:"description=Directory or file to search (default: current directory)."`
	// Glob filters files by glob pattern (e.g. "*.go"). When empty, all
	// files are considered.
	Glob string `json:"glob,omitempty" jsonschema:"description=Filter files by glob pattern, e.g. '*.go' or '**/*_test.go'."`
	// IgnoreCase enables case-insensitive matching.
	IgnoreCase bool `json:"ignoreCase,omitempty" jsonschema:"description=Case-insensitive search (default false)."`
	// Literal treats Pattern as a literal substring rather than a regex.
	Literal bool `json:"literal,omitempty" jsonschema:"description=Treat pattern as literal string instead of regex (default false)."`
	// Context is the number of lines to show before and after each match.
	// Zero (the default) shows only the match line.
	Context int `json:"context,omitempty" jsonschema:"description=Number of context lines before and after each match (default 0).,minimum=0"`
	// Limit caps the number of matches returned. Zero means the default
	// (DefaultGrepLimit); negative is rejected.
	Limit int `json:"limit,omitempty" jsonschema:"description=Maximum number of matches to return (default 100).,minimum=0"`
}

// grepTool implements the "grep" built-in tool.
type grepTool struct {
	ops GrepOperations
}

// NewGrepTool returns a grep Tool backed by ops. When ops.LookPath("rg")
// succeeds, Execute shells out to ripgrep (which honors .gitignore by
// default). Otherwise a pure-Go matcher walks the tree and applies the
// same gitignore rules in-process. A nil ops defaults to OSGrepOperations.
func NewGrepTool(ops GrepOperations) Tool {
	if ops == nil {
		ops = OSGrepOperations{}
	}
	return &grepTool{ops: ops}
}

// Name returns the tool's unique identifier.
func (t *grepTool) Name() string { return "grep" }

// Description returns the model-facing description of the tool's behavior.
func (t *grepTool) Description() string {
	return fmt.Sprintf(
		"Search file contents for a pattern. Returns matching lines with file paths and "+
			"line numbers in the form `path:line: content`. Respects .gitignore. Output is "+
			"truncated to %d matches or %d KiB (whichever is hit first). Long lines are "+
			"truncated to %d chars. Uses ripgrep when available; falls back to a pure-Go "+
			"matcher otherwise. Pass context=N to include N lines of context around each "+
			"match (context lines use `path-line- content` separators).",
		DefaultGrepLimit, DefaultMaxBytes/1024, GrepMaxLineLength,
	)
}

// Parameters returns the input JSON Schema (draft 2020-12) for the grep tool.
func (t *grepTool) Parameters() jsonschema.Schema {
	return ReflectSchema(&grepArgs{})
}

// Execute validates args, resolves the search root, and dispatches to the
// ripgrep backend (preferred) or the pure-Go fallback.
func (t *grepTool) Execute(ctx context.Context, call ToolCall) (ToolResult, error) {
	if err := ctx.Err(); err != nil {
		return ToolResult{}, err
	}
	var args grepArgs
	if bad := ParseArgs(call.Args, &args, "grep"); bad != nil {
		return *bad, nil
	}
	if args.Pattern == "" {
		return NewErrorResult("grep: missing required parameter \"pattern\""), nil
	}
	if args.Context < 0 {
		return NewErrorResult(fmt.Sprintf("grep: context must be >= 0 (got %d)", args.Context)), nil
	}
	if args.Limit < 0 {
		return NewErrorResult(fmt.Sprintf("grep: limit must be >= 0 (got %d)", args.Limit)), nil
	}
	if call.Cwd == "" {
		return NewErrorResult("grep: missing cwd"), nil
	}

	searchPath := args.Path
	if searchPath == "" {
		searchPath = "."
	}

	limit := args.Limit
	if limit == 0 {
		limit = DefaultGrepLimit
	}
	if limit < 1 {
		limit = 1
	}

	// Compile the pattern up front so both backends fail fast on a bad
	// regex. A literal search compiles as a quoted metacharacter-free regex.
	// This check runs before path validation so a typo'd regex surfaces as
	// "invalid pattern" rather than a confusing "path not found".
	patternStr := args.Pattern
	if args.Literal {
		patternStr = regexp.QuoteMeta(patternStr)
	}
	flags := ""
	if args.IgnoreCase {
		flags = "(?i)"
	}
	re, err := regexp.Compile(flags + patternStr)
	if err != nil {
		return NewErrorResult(fmt.Errorf("grep: %w: %v", ErrGrepInvalidPattern, err).Error()), nil
	}

	absoluteRoot, err := ResolveWithinCwd(searchPath, call.Cwd)
	if err != nil {
		//nolint:nilerr // application error → ToolResult per tool.go contract
		return NewErrorResult("grep: " + err.Error()), nil
	}
	isDir, err := t.ops.IsDirectory(absoluteRoot)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return NewErrorResult(fmt.Errorf("grep: %w (%q)", ErrGrepPathNotFound, args.Path).Error()), nil
		}
		return NewErrorResult(fmt.Sprintf("grep: stat %q: %v", args.Path, err)), nil
	}

	// Prefer ripgrep when available.
	if rgPath, lerr := t.ops.LookPath("rg"); lerr == nil && rgPath != "" {
		return t.runWithRipgrep(ctx, rgPath, args, absoluteRoot, isDir, call.Cwd, limit)
	}

	// Fall back to pure-Go matcher.
	return t.runPureGo(ctx, re, args, absoluteRoot, isDir, call.Cwd, limit)
}

// runWithRipgrep invokes rg with the user's arguments and parses its JSON
// output stream. ripgrep honors .gitignore natively; we do not duplicate
// that logic here.
func (t *grepTool) runWithRipgrep(ctx context.Context, rgPath string, args grepArgs, absoluteRoot string, isDir bool, cwd string, limit int) (ToolResult, error) {
	if err := ctx.Err(); err != nil {
		return ToolResult{}, err
	}
	rgArgs := []string{
		"--json", "--line-number", "--color=never", "--hidden",
	}
	if args.IgnoreCase {
		rgArgs = append(rgArgs, "--ignore-case")
	}
	if args.Literal {
		rgArgs = append(rgArgs, "--fixed-strings")
	}
	if args.Glob != "" {
		rgArgs = append(rgArgs, "--glob", args.Glob)
	}
	if args.Context > 0 {
		rgArgs = append(rgArgs, "--context", fmt.Sprintf("%d", args.Context))
	}
	rgArgs = append(rgArgs, "--max-count", fmt.Sprintf("%d", limit))
	rgArgs = append(rgArgs, "--", args.Pattern, absoluteRoot)

	cmd := exec.CommandContext(ctx, rgPath, rgArgs...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	cmd.SysProcAttr = sysProcAttrForKill()
	runErr := cmd.Run()
	if ctxErr := ctx.Err(); ctxErr != nil {
		return ToolResult{}, ctxErr
	}
	if runErr != nil {
		// ripgrep exit code 1 = no matches; 0 = matches found; other = error.
		var ee *exec.ExitError
		if errors.As(runErr, &ee) && ee.ExitCode() == 1 {
			return NewTextResult("No matches found"), nil
		}
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = runErr.Error()
		}
		return NewErrorResult(fmt.Sprintf("grep: ripgrep failed: %s", msg)), nil
	}

	matches := parseRgJSON(stdout.Bytes())
	if len(matches) == 0 {
		return NewTextResult("No matches found"), nil
	}

	// Read context lines for matches that need them (when args.Context > 0).
	fileCache := map[string][]string{}
	out := t.formatMatches(matches, args, absoluteRoot, isDir, cwd, fileCache, &linesTruncatedFlag{})
	truncated := TruncateHeadTail(out, DefaultMaxLines, DefaultMaxBytes)
	notices := buildGrepNotices(len(matches), limit, out, truncated, false)
	return NewTextResult(appendNotices(truncated, notices)), nil
}

// rgMatch is a parsed entry from rg --json output.
type rgMatch struct {
	filePath   string
	lineNumber int
	lineText   string
}

// parseRgJSON walks the rg --json stream and extracts match events. Lines
// that do not parse as JSON or are not match events are silently dropped.
func parseRgJSON(buf []byte) []rgMatch {
	var matches []rgMatch
	for _, line := range bytes.Split(buf, []byte("\n")) {
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var ev struct {
			Type string `json:"type"`
			Data struct {
				Path struct {
					Text string `json:"text"`
				} `json:"path"`
				LineNumber int `json:"line_number"`
				Lines      struct {
					Text string `json:"text"`
				} `json:"lines"`
			} `json:"data"`
		}
		if err := json.Unmarshal(line, &ev); err != nil {
			continue
		}
		if ev.Type != "match" {
			continue
		}
		if ev.Data.Path.Text == "" {
			continue
		}
		matches = append(matches, rgMatch{
			filePath:   ev.Data.Path.Text,
			lineNumber: ev.Data.LineNumber,
			lineText:   ev.Data.Lines.Text,
		})
	}
	return matches
}

// linesTruncatedFlag is a one-bit out-parameter that lets formatMatches
// signal that at least one line was truncated to GrepMaxLineLength.
type linesTruncatedFlag struct{ v bool }

// formatMatches renders matches in `path:line: content` form. When context
// is requested, context lines are emitted with `path-line- content`
// separators, matching ripgrep's own format. fileCache memoizes file reads
// across matches (context lookup can hit the same file repeatedly).
func (t *grepTool) formatMatches(matches []rgMatch, args grepArgs, absoluteRoot string, isDir bool, cwd string, fileCache map[string][]string, truncated *linesTruncatedFlag) string {
	var b strings.Builder
	for i, m := range matches {
		if i > 0 {
			b.WriteString("\n")
		}
		display := displayPath(m.filePath, absoluteRoot, isDir, cwd)
		if args.Context > 0 {
			lines, ok := fileCache[m.filePath]
			if !ok {
				data, err := t.ops.ReadFile(m.filePath)
				if err == nil {
					lines = splitLinesString(string(data))
				}
				fileCache[m.filePath] = lines
			}
			start := m.lineNumber - args.Context
			if start < 1 {
				start = 1
			}
			end := m.lineNumber + args.Context
			if end > len(lines) {
				end = len(lines)
			}
			first := true
			for ln := start; ln <= end; ln++ {
				var text string
				if ln-1 >= 0 && ln-1 < len(lines) {
					text = lines[ln-1]
				}
				text = truncateGrepLine(text, truncated)
				if first {
					first = false
				} else {
					b.WriteString("\n")
				}
				if ln == m.lineNumber {
					b.WriteString(fmt.Sprintf("%s:%d: %s", display, ln, text))
				} else {
					b.WriteString(fmt.Sprintf("%s-%d- %s", display, ln, text))
				}
			}
		} else {
			line := strings.TrimRight(m.lineText, "\n")
			line = truncateGrepLine(line, truncated)
			b.WriteString(fmt.Sprintf("%s:%d: %s", display, m.lineNumber, line))
		}
	}
	return b.String()
}

// displayPath renders a match file path relative to the search root when
// the root is a directory, or as a basename when the root is a single file.
// Absolute paths are made relative to cwd for readability when possible.
func displayPath(filePath, absoluteRoot string, isDir bool, cwd string) string {
	if isDir {
		rel, err := filepath.Rel(absoluteRoot, filePath)
		if err == nil && !strings.HasPrefix(rel, "..") {
			return filepath.ToSlash(rel)
		}
	}
	if rel, err := filepath.Rel(cwd, filePath); err == nil && !strings.HasPrefix(rel, "..") {
		return filepath.ToSlash(rel)
	}
	return filepath.ToSlash(filePath)
}

// truncateGrepLine caps a single line to GrepMaxLineLength characters. Sets
// truncated.v when a truncation occurred.
func truncateGrepLine(line string, truncated *linesTruncatedFlag) string {
	if len(line) <= GrepMaxLineLength {
		return line
	}
	truncated.v = true
	return line[:GrepMaxLineLength] + " …"
}

// runPureGo is the ripgrep-absent fallback. It walks the search root,
// applies gitignore rules in-process, matches each file line-by-line, and
// formats the results exactly like the ripgrep path.
func (t *grepTool) runPureGo(ctx context.Context, re *regexp.Regexp, args grepArgs, absoluteRoot string, isDir bool, cwd string, limit int) (ToolResult, error) {
	if err := ctx.Err(); err != nil {
		return ToolResult{}, err
	}

	// Collect matching files. When the root is a single file, skip the
	// walk and match it directly.
	var files []string
	ignoreStack := newIgnoreStack()
	if !isDir {
		files = append(files, absoluteRoot)
	} else {
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
				// Load any .gitignore at this directory.
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
			if args.Glob != "" && !globMatch(args.Glob, filepath.Base(p)) {
				return nil
			}
			files = append(files, p)
			return nil
		})
		if walkErr != nil {
			if errors.Is(walkErr, context.Canceled) || errors.Is(walkErr, context.DeadlineExceeded) {
				return ToolResult{}, walkErr
			}
			return NewErrorResult(fmt.Sprintf("grep: walk %q: %v", args.Path, walkErr)), nil
		}
	}

	// Sort files for deterministic output.
	sort.Strings(files)

	var matches []rgMatch
	fileCache := map[string][]string{}
	perFileLimit := limit
	for _, file := range files {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ToolResult{}, ctxErr
		}
		data, rerr := t.ops.ReadFile(file)
		if rerr != nil {
			continue
		}
		lines := splitLinesString(string(data))
		fileCache[file] = lines
		for i, line := range lines {
			if len(matches) >= limit {
				break
			}
			if re.MatchString(line) {
				matches = append(matches, rgMatch{
					filePath:   file,
					lineNumber: i + 1,
					lineText:   line,
				})
			}
		}
		if len(matches) >= perFileLimit {
			break
		}
	}

	if len(matches) == 0 {
		return NewTextResult("No matches found"), nil
	}

	truncated := &linesTruncatedFlag{}
	out := t.formatMatchesFromCache(matches, args, absoluteRoot, isDir, cwd, fileCache, truncated)
	capped := TruncateHeadTail(out, DefaultMaxLines, DefaultMaxBytes)
	notices := buildGrepNotices(len(matches), limit, out, capped, truncated.v)
	return NewTextResult(appendNotices(capped, notices)), nil
}

// formatMatchesFromCache is the pure-Go analogue of formatMatches. It uses
// the caller-populated fileCache so files are read at most once.
func (t *grepTool) formatMatchesFromCache(matches []rgMatch, args grepArgs, absoluteRoot string, isDir bool, cwd string, fileCache map[string][]string, truncated *linesTruncatedFlag) string {
	var b strings.Builder
	for i, m := range matches {
		if i > 0 {
			b.WriteString("\n")
		}
		display := displayPath(m.filePath, absoluteRoot, isDir, cwd)
		lines := fileCache[m.filePath]
		if args.Context > 0 {
			start := m.lineNumber - args.Context
			if start < 1 {
				start = 1
			}
			end := m.lineNumber + args.Context
			if end > len(lines) {
				end = len(lines)
			}
			first := true
			for ln := start; ln <= end; ln++ {
				var text string
				if ln-1 >= 0 && ln-1 < len(lines) {
					text = lines[ln-1]
				}
				text = truncateGrepLine(text, truncated)
				if first {
					first = false
				} else {
					b.WriteString("\n")
				}
				if ln == m.lineNumber {
					b.WriteString(fmt.Sprintf("%s:%d: %s", display, ln, text))
				} else {
					b.WriteString(fmt.Sprintf("%s-%d- %s", display, ln, text))
				}
			}
		} else {
			line := strings.TrimRight(m.lineText, "\n")
			line = truncateGrepLine(line, truncated)
			b.WriteString(fmt.Sprintf("%s:%d: %s", display, m.lineNumber, line))
		}
	}
	return b.String()
}

// buildGrepNotices assembles the `[...]` suffix appended to truncated grep
// output. matchCount is the number of matches found; limit is the cap; raw
// is the pre-truncation output; capped is the post-truncation output; some
// lines were individually truncated to GrepMaxLineLength iff lineTruncated.
func buildGrepNotices(matchCount, limit int, raw, capped string, lineTruncated bool) []string {
	var notices []string
	if matchCount >= limit {
		notices = append(notices, fmt.Sprintf(
			"%d matches limit reached. Use limit=%d for more, or refine pattern",
			limit, limit*2,
		))
	}
	if len(capped) < len(raw) {
		notices = append(notices, fmt.Sprintf("%d KiB output limit reached", DefaultMaxBytes/1024))
	}
	if lineTruncated {
		notices = append(notices, fmt.Sprintf(
			"some lines truncated to %d chars. Use read tool to see full lines",
			GrepMaxLineLength,
		))
	}
	return notices
}

// appendNotices joins a cap'd output with its notices block, or returns
// the output unchanged when there are no notices.
func appendNotices(out string, notices []string) string {
	if len(notices) == 0 {
		return out
	}
	return out + "\n\n[" + strings.Join(notices, ". ") + "]"
}

// splitLinesString is the string-typed analogue of splitLines. It splits on
// '\n' preserving the trailing newline on each line except possibly the
// last.
func splitLinesString(s string) []string {
	if s == "" {
		return nil
	}
	// Normalize CRLF to LF so Windows-authored files line up.
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	return strings.Split(s, "\n")
}

// globMatch reports whether name matches the given shell glob. Supports
// the common *, ?, and [...] metacharacters via filepath.Match. When the
// glob contains a path separator, the whole path is matched; otherwise
// only the basename.
func globMatch(glob, name string) bool {
	if strings.ContainsAny(glob, "/\\") {
		ok, err := filepath.Match(glob, name)
		return err == nil && ok
	}
	ok, err := filepath.Match(glob, filepath.Base(name))
	return err == nil && ok
}

// ignoreStack accumulates .gitignore files as a directory walk descends.
// push/pop are tied to directory depth so ignore patterns are scoped
// correctly.
type ignoreStack struct {
	entries []ignoreEntry
}

// ignoreEntry pairs a compiled ignore object with the directory whose
// children it governs.
type ignoreEntry struct {
	dir    string
	ignore *gitignore.GitIgnore
}

// newIgnoreStack returns an empty ignoreStack.
func newIgnoreStack() *ignoreStack { return &ignoreStack{} }

// push records that we are descending into dir. dir MUST be absolute.
func (s *ignoreStack) push(dir string) { s.entries = append(s.entries, ignoreEntry{dir: dir}) }

// add registers a compiled ignore whose patterns are interpreted relative
// to dir. add is called after push for the same directory.
func (s *ignoreStack) add(ig *gitignore.GitIgnore, dir string) {
	for i := range s.entries {
		if s.entries[i].dir == dir {
			s.entries[i].ignore = ig
			return
		}
	}
	s.entries = append(s.entries, ignoreEntry{dir: dir, ignore: ig})
}

// matchesFile reports whether rel (relative to the walk root) is ignored
// by any ignore in the stack.
func (s *ignoreStack) matchesFile(rel string) bool {
	for _, e := range s.entries {
		if e.ignore == nil {
			continue
		}
		if e.ignore.MatchesPath(filepath.ToSlash(rel)) {
			return true
		}
	}
	return false
}

// matchesDir is like matchesFile but for directory entries. gitignore
// patterns ending in `/` only match directories; sabhiram/go-gitignore
// already handles this in MatchesPath, so we delegate.
func (s *ignoreStack) matchesDir(rel string) bool {
	return s.matchesFile(rel)
}

// tryReadGitignore loads and compiles a .gitignore at path. Returns
// (ignore, true) on success; (nil, false) if the file is missing or
// unreadable. A malformed .gitignore yields (nil, false) — we never fail
// a search over a bad ignore file.
func tryReadGitignore(path string) (*gitignore.GitIgnore, bool) {
	ig, err := gitignore.CompileIgnoreFile(path)
	if err != nil {
		return nil, false
	}
	return ig, true
}

// RenderCall produces a TUI-friendly representation of the invocation.
// Format: `grep /pattern/ in <path> [(glob)] [limit N]`.
func (t *grepTool) RenderCall(args json.RawMessage, theme *Theme) string {
	var a grepArgs
	_ = json.Unmarshal(args, &a)
	out := theme.Wrap(theme.Primary, "grep") + " " + theme.Wrap(theme.Accent, "/"+a.Pattern+"/")
	target := a.Path
	if target == "" {
		target = "."
	}
	out += " " + theme.Wrap(theme.Muted, "in "+target)
	if a.Glob != "" {
		out += " " + theme.Wrap(theme.Muted, "("+a.Glob+")")
	}
	if a.Limit > 0 {
		out += " " + theme.Wrap(theme.Muted, fmt.Sprintf("limit %d", a.Limit))
	}
	if a.IgnoreCase {
		out += " " + theme.Wrap(theme.Muted, "[ignore-case]")
	}
	if a.Literal {
		out += " " + theme.Wrap(theme.Muted, "[literal]")
	}
	return out
}

// RenderResult produces a TUI-friendly representation of the result.
func (t *grepTool) RenderResult(result ToolResult, theme *Theme) string {
	prefix := ""
	if result.IsError {
		prefix = theme.Wrap(theme.Error, "error: ")
	}
	return prefix + renderContentBlocks(result.Content, theme)
}

// _ keeps sync referenced; ignoreStack push/pop are intentionally
// goroutine-safe via the higher-level tool Execute contract (grep has no
// shared mutable state per invocation, but we keep sync imported for
// future extensions).
var _ = sync.Mutex{}
