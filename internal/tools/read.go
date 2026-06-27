package tools

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/invopop/jsonschema"

	"github.com/coevin/tau/internal/llm"
)

// ReadOperations abstracts the I/O surface of the read tool. The default
// implementation uses the real OS; tests inject fake implementations to
// avoid filesystem side effects.
type ReadOperations interface {
	// ReadFile returns the raw bytes of the file at absolutePath.
	ReadFile(absolutePath string) ([]byte, error)
	// Access verifies the file is readable. Returns a descriptive error
	// (wrapping os.ErrNotExist, os.ErrPermission, etc.) if not.
	Access(absolutePath string) error
	// DetectImageMimeType returns the image MIME type (e.g. "image/png")
	// if the file at absolutePath is a supported image format, or "" if
	// it is not. A nil error with an empty string means "not an image";
	// the caller then falls through to the text/binary path.
	DetectImageMimeType(absolutePath string) (string, error)
}

// OSReadOperations is the default ReadOperations backed by the real
// filesystem.
type OSReadOperations struct{}

// ReadFile delegates to os.ReadFile.
func (OSReadOperations) ReadFile(p string) ([]byte, error) { return os.ReadFile(p) }

// Access opens the file for reading to verify it exists and is readable.
// The file is closed immediately; subsequent operations re-open it.
func (OSReadOperations) Access(p string) error {
	f, err := os.Open(p)
	if err != nil {
		return err
	}
	return f.Close()
}

// DetectImageMimeType reads the first 16 bytes and sniffs for known image
// signatures (PNG, JPEG, GIF, WebP). Returns "" for non-images or
// unsupported image types.
func (OSReadOperations) DetectImageMimeType(p string) (string, error) {
	f, err := os.Open(p)
	if err != nil {
		return "", err
	}
	defer f.Close()
	var sniff [16]byte
	n, _ := f.Read(sniff[:])
	return DetectImageMimeTypeFromBytes(sniff[:n]), nil
}

// readArgs is the input schema for the read tool.
type readArgs struct {
	Path string `json:"path" jsonschema:"description=Path to the file to read. May be relative to the agent cwd or absolute."`
	// Offset is the 1-indexed line number to start reading from. When
	// omitted, reading starts at line 1.
	Offset *int `json:"offset,omitempty" jsonschema:"description=Line number to start reading from (1-indexed).,minimum=1"`
	// Limit is the maximum number of lines to return starting from Offset
	// (or line 1 if Offset is unset). When omitted, the read extends to
	// EOF subject to the default truncation limits.
	Limit *int `json:"limit,omitempty" jsonschema:"description=Maximum number of lines to read.,minimum=0"`
}

// readTool implements the "read" built-in tool.
type readTool struct {
	ops ReadOperations
}

// NewReadTool returns a read Tool backed by the given operations. Pass
// OSReadOperations{} for production use, or a fake implementation in tests.
// A nil ops defaults to OSReadOperations.
func NewReadTool(ops ReadOperations) Tool {
	if ops == nil {
		ops = OSReadOperations{}
	}
	return &readTool{ops: ops}
}

// Name returns the tool's unique identifier.
func (t *readTool) Name() string { return "read" }

// Description returns the model-facing description of the tool's behavior,
// including the default truncation limits and the offset/limit continuation
// hint.
func (t *readTool) Description() string {
	return fmt.Sprintf(
		"Read the contents of a file. Supports text files and images (png, jpg, gif, webp). "+
			"Images are returned as image content blocks. Text output is truncated to %d "+
			"lines and %d KiB using head-and-tail truncation (preserving the first and last "+
			"portions of the file). Use the offset and limit parameters for large files; "+
			"continue with offset until the file is exhausted.",
		DefaultMaxLines, DefaultMaxBytes/1024,
	)
}

// Parameters returns the input JSON Schema (draft 2020-12) for the read tool.
func (t *readTool) Parameters() jsonschema.Schema {
	return ReflectSchema(&readArgs{})
}

// Execute reads the file at args.Path (resolved against call.Cwd) and returns
// either a TextContent result for text files, an ImageContent result for
// supported images, or an IsError result for binary non-image content,
// missing files, permission errors, and invalid arguments.
func (t *readTool) Execute(ctx context.Context, call ToolCall) (ToolResult, error) {
	if err := ctx.Err(); err != nil {
		return ToolResult{}, err
	}
	var args readArgs
	if bad := ParseArgs(call.Args, &args, "read"); bad != nil {
		return *bad, nil
	}
	if args.Path == "" {
		r := NewErrorResult("read: missing required parameter \"path\"")
		return r, nil
	}
	if args.Offset != nil && *args.Offset < 1 {
		r := NewErrorResult(fmt.Sprintf("read: offset must be >= 1, got %d", *args.Offset))
		return r, nil
	}
	if args.Limit != nil && *args.Limit < 0 {
		r := NewErrorResult(fmt.Sprintf("read: limit must be >= 0, got %d", *args.Limit))
		return r, nil
	}

	absolutePath, err := ResolveWithinCwd(args.Path, call.Cwd)
	if err != nil {
		//nolint:nilerr // application error → ToolResult per tool.go contract
		return NewErrorResult("read: " + err.Error()), nil
	}

	if err := t.ops.Access(absolutePath); err != nil {
		return NewErrorResult(fmt.Sprintf("read: cannot access %q: %v", args.Path, err)), nil
	}

	mimeType, err := t.ops.DetectImageMimeType(absolutePath)
	if err != nil {
		return NewErrorResult(fmt.Sprintf("read: mime detection failed for %q: %v", args.Path, err)), nil
	}
	if mimeType != "" {
		return t.readImage(absolutePath, args.Path, mimeType)
	}

	data, err := t.ops.ReadFile(absolutePath)
	if err != nil {
		return NewErrorResult(fmt.Sprintf("read: failed to read %q: %v", args.Path, err)), nil
	}
	if IsBinary(data) {
		return NewErrorResult(fmt.Sprintf(
			"read: %q appears to be a binary file (%d bytes). Use bash with `xxd`, "+
				"`od`, or `file` to inspect binary content.",
			args.Path, len(data),
		)), nil
	}
	return t.readText(data, args)
}

// readImage returns a TextContent note plus an ImageContent block carrying
// the file's bytes base64-encoded.
func (t *readTool) readImage(absolutePath, displayPath, mimeType string) (ToolResult, error) {
	data, err := t.ops.ReadFile(absolutePath)
	if err != nil {
		return NewErrorResult(fmt.Sprintf("read: failed to read image %q: %v", displayPath, err)), nil
	}
	encoded := base64.StdEncoding.EncodeToString(data)
	note := llm.TextContent{Text: fmt.Sprintf("Read image file [%s] (%d bytes)", mimeType, len(data))}
	img := llm.ImageContent{Data: encoded, MimeType: mimeType}
	return ToolResult{Content: []llm.ContentBlock{note, img}}, nil
}

// readText slices the text by offset/limit, then applies head-tail truncation
// and emits continuation hints for the model.
func (t *readTool) readText(data []byte, args readArgs) (ToolResult, error) {
	text := string(data)
	allLines := strings.Split(text, "\n")
	totalLines := len(allLines)

	// 1-indexed offset → 0-indexed array access.
	startLine := 0
	if args.Offset != nil {
		startLine = *args.Offset - 1
		if startLine >= totalLines {
			r := NewErrorResult(fmt.Sprintf(
				"read: offset %d is beyond end of file (%d lines)",
				*args.Offset, totalLines,
			))
			return r, nil
		}
	}
	startLineDisplay := startLine + 1

	// User-specified limit is honored first; otherwise read to EOF.
	endLine := totalLines
	userLimitedLines := -1
	if args.Limit != nil {
		endLine = startLine + *args.Limit
		if endLine > totalLines {
			endLine = totalLines
		}
		userLimitedLines = endLine - startLine
	}
	selected := strings.Join(allLines[startLine:endLine], "\n")

	// First-line-size check: if the first line of the window alone exceeds
	// the byte cap, no truncation strategy can show it usefully. Suggest
	// a bash fallback that pipes through head -c.
	firstLineLen := len(allLines[startLine])
	if firstLineLen > DefaultMaxBytes {
		return NewTextResult(fmt.Sprintf(
			"[Line %d is %d bytes, exceeds %d-byte limit. Use bash: sed -n '%dp' %s | head -c %d]",
			startLineDisplay, firstLineLen, DefaultMaxBytes, startLineDisplay, args.Path, DefaultMaxBytes,
		)), nil
	}

	truncated := TruncateHeadTail(selected, DefaultMaxLines, DefaultMaxBytes)
	if truncated == selected {
		// No truncation needed. If the user-specified limit stopped early,
		// surface a continuation hint.
		if userLimitedLines >= 0 && startLine+userLimitedLines < totalLines {
			remaining := totalLines - (startLine + userLimitedLines)
			nextOffset := startLine + userLimitedLines + 1
			return NewTextResult(fmt.Sprintf(
				"%s\n\n[%d more lines in file. Use offset=%d to continue.]",
				selected, remaining, nextOffset,
			)), nil
		}
		return NewTextResult(selected), nil
	}

	// Truncation occurred. Count post-truncation output lines and emit an
	// offset continuation hint.
	outputLines := strings.Count(truncated, "\n") + 1
	endLineDisplay := startLineDisplay + outputLines - 1
	nextOffset := endLineDisplay + 1
	return NewTextResult(fmt.Sprintf(
		"%s\n\n[Showing lines %d-%d of %d. Use offset=%d to continue.]",
		truncated, startLineDisplay, endLineDisplay, totalLines, nextOffset,
	)), nil
}

// RenderCall produces a TUI-friendly representation of the invocation.
// Format: `read <path>` or `read <path>:<start>-<end>` when offset/limit set.
func (t *readTool) RenderCall(args json.RawMessage, theme *Theme) string {
	var a readArgs
	_ = json.Unmarshal(args, &a)
	path := a.Path
	if path == "" {
		path = "?"
	}
	out := theme.Wrap(theme.Primary, "read") + " " + theme.Wrap(theme.Accent, path)
	if a.Offset != nil || a.Limit != nil {
		start := 1
		if a.Offset != nil {
			start = *a.Offset
		}
		if a.Limit != nil {
			out += theme.Wrap(theme.Warning, fmt.Sprintf(":%d-%d", start, start+*a.Limit-1))
		} else {
			out += theme.Wrap(theme.Warning, fmt.Sprintf(":%d+", start))
		}
	}
	return out
}

// RenderResult produces a TUI-friendly representation of the result. Errors
// are prefixed in the theme's Error color; image blocks are summarized as
// a muted placeholder so the TUI doesn't dump the base64 payload.
func (t *readTool) RenderResult(result ToolResult, theme *Theme) string {
	prefix := ""
	if result.IsError {
		prefix = theme.Wrap(theme.Error, "error: ")
	}
	return prefix + renderContentBlocks(result.Content, theme)
}

// renderContentBlocks stringifies ContentBlocks for TUI display.
func renderContentBlocks(blocks []llm.ContentBlock, theme *Theme) string {
	var b strings.Builder
	for i, blk := range blocks {
		if i > 0 {
			b.WriteString("\n")
		}
		switch v := blk.(type) {
		case llm.TextContent:
			b.WriteString(v.Text)
		case llm.ImageContent:
			b.WriteString(theme.Wrap(theme.Muted,
				fmt.Sprintf("[image: %s, %d base64 chars]", v.MimeType, len(v.Data))))
		default:
			b.WriteString(theme.Wrap(theme.Muted, fmt.Sprintf("[%T]", v)))
		}
	}
	return b.String()
}

// ResolveWithinCwd resolves a possibly-relative path against cwd and returns
// the cleaned absolute path. If path is already absolute, it is returned
// cleaned. An empty cwd with a relative path is an error.
func ResolveWithinCwd(path, cwd string) (string, error) {
	if path == "" {
		return "", fmt.Errorf("empty path")
	}
	if filepath.IsAbs(path) {
		return filepath.Clean(path), nil
	}
	if cwd == "" {
		return "", fmt.Errorf("relative path %q requires a non-empty cwd", path)
	}
	return filepath.Join(cwd, path), nil
}

// IsBinary returns true if data contains a NUL byte within its first 8 KiB.
// This matches git's heuristic: text files effectively never contain NUL,
// while binary files almost always do within the first few kilobytes.
func IsBinary(data []byte) bool {
	end := len(data)
	if end > 8192 {
		end = 8192
	}
	for _, b := range data[:end] {
		if b == 0 {
			return true
		}
	}
	return false
}

// DetectImageMimeTypeFromBytes sniffs the leading bytes of a file for known
// image signatures. Returns "" for non-images or unsupported image types.
// Supported: PNG (with IHDR sanity check), JPEG (excluding JPEG 2000),
// GIF (any version), WebP (RIFF/WEBP).
func DetectImageMimeTypeFromBytes(buf []byte) string {
	// JPEG: FF D8 FF, excluding FF D8 FF F7 (JPEG 2000).
	if len(buf) >= 4 && buf[0] == 0xff && buf[1] == 0xd8 && buf[2] == 0xff && buf[3] == 0xf7 {
		return ""
	}
	if len(buf) >= 3 && buf[0] == 0xff && buf[1] == 0xd8 && buf[2] == 0xff {
		return "image/jpeg"
	}
	// PNG: 89 50 4E 47 0D 0A 1A 0A, followed by an IHDR chunk of length 13.
	pngSig := []byte{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a}
	if len(buf) >= 16 && bytesEqual(buf, pngSig) &&
		buf[8] == 0 && buf[9] == 0 && buf[10] == 0 && buf[11] == 13 &&
		buf[12] == 'I' && buf[13] == 'H' && buf[14] == 'D' && buf[15] == 'R' {
		return "image/png"
	}
	// GIF: "GIF" prefix (GIF87a or GIF89a).
	if len(buf) >= 3 && buf[0] == 'G' && buf[1] == 'I' && buf[2] == 'F' {
		return "image/gif"
	}
	// WebP: "RIFF"...."WEBP".
	if len(buf) >= 12 && buf[0] == 'R' && buf[1] == 'I' && buf[2] == 'F' && buf[3] == 'F' &&
		buf[8] == 'W' && buf[9] == 'E' && buf[10] == 'B' && buf[11] == 'P' {
		return "image/webp"
	}
	return ""
}

// bytesEqual returns true if a starts with the exact bytes of b.
func bytesEqual(a, b []byte) bool {
	if len(a) < len(b) {
		return false
	}
	for i, v := range b {
		if a[i] != v {
			return false
		}
	}
	return true
}

// ReflectSchema builds a flat (no $ref) JSON Schema from a sample struct.
// The reflector-generated $schema, $id, and $defs keys are stripped so the
// result can be embedded directly into a provider tool definition without
// confusing draft-2020-12 validators.
//
// This is the primary constructor for built-in tools whose parameters are
// typed Go structs with json and jsonschema tags.
func ReflectSchema(sample any) jsonschema.Schema {
	r := new(jsonschema.Reflector)
	r.DoNotReference = true
	s := r.Reflect(sample)
	if s == nil {
		return jsonschema.Schema{Type: "object"}
	}
	s.Version = ""
	s.ID = ""
	s.Definitions = nil
	return *s
}

// ParseJSONSchema unmarshals a JSON-Schema document produced by a plugin
// into the jsonschema.Schema shape. Used by PluginTool.Parameters() to
// surface plugin-declared schemas in the same struct form built-in tools
// use. Returns a minimal object schema on parse failure rather than an
// error: the tool is still callable, the model just gets a degenerate
// schema; the caller surfaces the parse error via a diagnostic.
func ParseJSONSchema(raw string) jsonschema.Schema {
	if raw == "" {
		return jsonschema.Schema{Type: "object"}
	}
	var s jsonschema.Schema
	if err := json.Unmarshal([]byte(raw), &s); err != nil {
		return jsonschema.Schema{Type: "object"}
	}
	if s.Type == "" {
		s.Type = "object"
	}
	return s
}

// RenderContentBlocks is the exported form of renderContentBlocks for use
// by the plugins package's PluginTool wrapper. Defined here because all
// built-in tools reach for the same helper.
func RenderContentBlocks(blocks []llm.ContentBlock, theme *Theme) string {
	return renderContentBlocks(blocks, theme)
}
