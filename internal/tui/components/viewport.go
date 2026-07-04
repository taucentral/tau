// Package components — viewport.go — conversational stream viewer.
//
// The viewport is the main scrollable region that displays the
// conversation: user messages, assistant messages (rendered as
// markdown), and tool-execution accordions. Content accumulates as
// the turn progresses; the viewport auto-scrolls to the bottom when
// new content arrives unless the user has scrolled up (in which case
// a "new messages" indicator is shown).
//
// The scrollback buffer is capped at MaxScrollbackLines (default
// 10 000) to bound memory. When the cap is reached the oldest lines
// are discarded.
package components

import (
	"strings"

	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
)

// DefaultScrollbackLines is the maximum number of rendered lines kept
// in memory. Older content is evicted when the cap is reached.
const DefaultScrollbackLines = 10000

// Viewport is the conversational stream viewer model.
type Viewport struct {
	theme    Theme
	width    int
	height   int
	vp       viewport.Model
	md       *MarkdownRenderer
	elements []ViewportElement
	rendered strings.Builder
	// autoScroll is true when the viewport should jump to the bottom
	// on new content. Set to false when the user scrolls up.
	autoScroll bool
	// maxLines caps the rendered buffer.
	maxLines int
}

// ViewportElement is one block of content in the stream.
type ViewportElement interface {
	// Render produces the terminal string for this element, given the
	// available width and theme.
	Render(width int, theme Theme, md *MarkdownRenderer) string
}

// NewViewport returns a viewport at the given dimensions.
func NewViewport(theme Theme, width, height int) *Viewport {
	vp := viewport.New(width, height)
	vp.SetContent("")
	return &Viewport{
		theme:      theme,
		width:      width,
		height:     height,
		vp:         vp,
		md:         NewMarkdownRenderer(theme, width),
		autoScroll: true,
		maxLines:   DefaultScrollbackLines,
	}
}

// SetDimensions updates the viewport size and reflows content.
func (v *Viewport) SetDimensions(width, height int) {
	v.width = width
	v.height = height
	v.vp.Width = width
	v.vp.Height = height
	v.md.SetWidth(width)
	v.rerender()
}

// AppendElement adds a new element to the stream and re-renders.
func (v *Viewport) AppendElement(el ViewportElement) {
	v.elements = append(v.elements, el)
	v.rerender()
	if v.autoScroll {
		v.vp.GotoBottom()
	}
}

// AppendText is a convenience for adding a plain-text block.
func (v *Viewport) AppendText(role, text string) {
	v.AppendElement(TextElement{Role: role, Text: text})
}

// ElementCount returns the number of top-level elements currently in
// the scrollback buffer. Used by tests to verify that content was
// appended.
func (v *Viewport) ElementCount() int { return len(v.elements) }

// Clear empties the scrollback buffer. The state tree on disk is
// unaffected — Clear only resets the in-memory viewport so the user
// sees a clean slate. New appends start fresh.
func (v *Viewport) Clear() {
	v.elements = nil
	v.rendered.Reset()
	v.vp.SetContent("")
}

// SetAutoScroll controls whether the viewport follows new content.
func (v *Viewport) SetAutoScroll(auto bool) {
	v.autoScroll = auto
	if auto {
		v.vp.GotoBottom()
	}
}

// AutoScroll reports whether auto-scroll is on.
func (v *Viewport) AutoScroll() bool {
	return v.autoScroll
}

// ResetScroll snaps the viewport to the top-left origin and re-enables
// auto-scroll. Used by /clear's ErrContextReset handler so the user
// sees a clean slate with the scroll position matching the cleared
// content. Without this, the scroll offset would refer to entries that
// no longer exist, producing visual artifacts.
func (v *Viewport) ResetScroll() {
	v.vp.GotoTop()
	v.autoScroll = true
}

// ScrollUp moves the viewport up by one page.
func (v *Viewport) ScrollUp() {
	v.vp.ScrollUp(v.height)
	v.autoScroll = false
}

// ScrollDown moves the viewport down by one page.
func (v *Viewport) ScrollDown() {
	v.vp.ScrollDown(v.height)
	if v.vp.AtBottom() {
		v.autoScroll = true
	}
}

// Init implements tea.Model.
func (v *Viewport) Init() tea.Cmd { return nil }

// Update implements tea.Model.
func (v *Viewport) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd
	v.vp, cmd = v.vp.Update(msg)
	// If the user scrolled to the bottom, re-enable auto-scroll.
	if v.vp.AtBottom() {
		v.autoScroll = true
	}
	return v, cmd
}

// View implements tea.Model.
func (v *Viewport) View() string {
	return v.vp.View()
}

// rerender rebuilds the rendered buffer from all elements and applies
// it to the viewport, enforcing the scrollback cap.
func (v *Viewport) rerender() {
	v.rendered.Reset()
	for _, el := range v.elements {
		v.rendered.WriteString(el.Render(v.width, v.theme, v.md))
		v.rendered.WriteString("\n")
	}
	content := v.rendered.String()
	// Enforce scrollback cap by trimming the oldest lines.
	if v.maxLines > 0 {
		lines := strings.Count(content, "\n")
		if lines > v.maxLines {
			// Skip past the first (lines - maxLines) newlines.
			skip := lines - v.maxLines
			idx := 0
			for i := 0; i < skip; i++ {
				next := strings.IndexByte(content[idx:], '\n')
				if next < 0 {
					break
				}
				idx += next + 1
			}
			content = content[idx:]
		}
	}
	v.vp.SetContent(content)
}

// --- Element types ---

// TextElement is a plain text block (user or assistant message).
type TextElement struct {
	Role string // "user", "assistant", "system"
	Text string
}

// Render implements ViewportElement.
func (e TextElement) Render(width int, theme Theme, md *MarkdownRenderer) string {
	indent := theme.Indent()
	var style func(string) string
	switch e.Role {
	case "user":
		s := theme.UserStyle()
		style = func(text string) string { return s.Render(text) }
	case "assistant":
		// Assistant messages are rendered as markdown.
		rendered, err := md.Render(e.Text)
		if err != nil {
			rendered = e.Text
		}
		return indentLines(rendered, indent)
	default:
		s := theme.MutedStyle()
		style = func(text string) string { return s.Render(text) }
	}
	return indentLines(style(e.Text), indent)
}

// indentLines prefixes every line in s with indent.
func indentLines(s, indent string) string {
	if indent == "" {
		return s
	}
	lines := strings.Split(s, "\n")
	for i, line := range lines {
		lines[i] = indent + line
	}
	return strings.Join(lines, "\n")
}

// ToolElement is a tool execution block. It delegates rendering to the
// Accordion component.
type ToolElement struct {
	Name     string
	Args     string
	Result   string
	IsError  bool
	Running  bool
	Expanded bool
}

// Render implements ViewportElement.
func (e ToolElement) Render(width int, theme Theme, _ *MarkdownRenderer) string {
	acc := NewAccordion(theme, e.Name, e.Args)
	acc.Running = e.Running
	acc.Expanded = e.Expanded
	acc.SetResult(e.Result)
	acc.IsError = e.IsError
	acc.SetWidth(width)
	return acc.View()
}
