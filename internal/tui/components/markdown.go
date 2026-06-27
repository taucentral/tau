// Package components implements the individual TUI building blocks.
//
// markdown.go wraps charmbracelet/glamour to render assistant messages
// as styled markdown with syntax-highlighted code blocks. Because
// glamour.TermRenderer is not safe for concurrent use, a sync.Pool of
// renderers is used so multiple goroutines can render in parallel.
package components

import (
	"sync"

	"github.com/charmbracelet/glamour"
)

// MarkdownRenderer wraps a pool of glamour renderers for a given theme
// and word-wrap width. The pool is necessary because glamour.TermRenderer
// is not safe for concurrent use, but the TUI may render multiple
// messages in parallel (e.g. during scrollback refill).
type MarkdownRenderer struct {
	theme Theme
	width int
	pool  sync.Pool
}

// NewMarkdownRenderer returns a renderer pool for the given theme and
// word-wrap width. Width should match the conversational viewport's
// inner width so code blocks wrap rather than truncate.
func NewMarkdownRenderer(theme Theme, width int) *MarkdownRenderer {
	mr := &MarkdownRenderer{
		theme: theme,
		width: width,
	}
	mr.pool = sync.Pool{
		New: func() any {
			return mr.newRenderer()
		},
	}
	return mr
}

// newRenderer creates a single glamour.TermRenderer configured for the
// current theme and width.
func (mr *MarkdownRenderer) newRenderer() *glamour.TermRenderer {
	style := "dark"
	if isLightTheme(mr.theme) {
		style = "light"
	}
	r, err := glamour.NewTermRenderer(
		glamour.WithStandardStyle(style),
		glamour.WithWordWrap(mr.width),
	)
	if err != nil {
		// glamour.NewTermRenderer only errors on invalid options,
		// which is a programming bug. Fall back to a default.
		r, _ = glamour.NewTermRenderer()
		return r
	}
	return r
}

// isLightTheme returns true when the theme name suggests a light
// background, which selects glamour's light style.
func isLightTheme(theme Theme) bool {
	return theme.Name == "light"
}

// Render renders a markdown string to styled terminal output. Safe for
// concurrent use.
func (mr *MarkdownRenderer) Render(markdown string) (string, error) {
	r := mr.pool.Get().(*glamour.TermRenderer)
	defer mr.pool.Put(r)
	out, err := r.Render(markdown)
	if err != nil {
		return markdown, err
	}
	return out, nil
}

// SetWidth updates the word-wrap width for subsequently allocated
// renderers. Existing renderers in the pool are not affected, but they
// will be replaced as they are recycled.
func (mr *MarkdownRenderer) SetWidth(width int) {
	mr.width = width
	mr.pool = sync.Pool{
		New: func() any {
			return mr.newRenderer()
		},
	}
}

// PlainRender renders markdown without any ANSI styling. Used for
// non-TTY fallback and for export to files.
func PlainRender(markdown string) (string, error) {
	r, err := glamour.NewTermRenderer(
		glamour.WithWordWrap(80),
		glamour.WithStandardStyle("notty"),
	)
	if err != nil {
		return markdown, err
	}
	return r.Render(markdown)
}
