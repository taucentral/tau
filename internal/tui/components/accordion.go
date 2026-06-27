// Package components — accordion.go — expandable/collapsible tool block.
//
// Each tool call in the conversational stream is rendered as an
// accordion: a single-line summary when collapsed (tool name +
// truncated args + spinner/status icon) and a multi-line block when
// expanded (full args + full result).
package components

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// Accordion renders a single tool-execution block.
type Accordion struct {
	theme    Theme
	name     string
	args     string
	result   string
	width    int
	Running  bool
	IsError  bool
	Expanded bool
}

// NewAccordion returns a collapsed accordion for the given tool name
// and args.
func NewAccordion(theme Theme, name, args string) *Accordion {
	return &Accordion{
		theme: theme,
		name:  name,
		args:  args,
	}
}

// SetWidth sets the rendering width so the args can be truncated to fit.
func (a *Accordion) SetWidth(width int) {
	a.width = width
}

// Toggle flips the expanded state.
func (a *Accordion) Toggle() {
	a.Expanded = !a.Expanded
}

// SetResult sets the tool result text.
func (a *Accordion) SetResult(result string) {
	a.result = result
}

// View renders the accordion.
func (a *Accordion) View() string {
	var b strings.Builder

	// Status icon.
	var icon string
	switch {
	case a.Running:
		icon = "◐"
	case a.IsError:
		icon = "✗"
	default:
		icon = "✓"
	}

	// Coloured icon.
	iconStr := a.theme.SuccessStyle().Render(icon)
	if a.IsError {
		iconStr = a.theme.ErrorStyle().Render(icon)
	}
	if a.Running {
		iconStr = lipgloss.NewStyle().Foreground(a.theme.WarningColor).Render(icon)
	}

	// Tool name in tool colour.
	nameStr := a.theme.ToolStyle().Render(a.name)

	// Truncated args.
	argsStr := truncateArgs(a.args, a.width, a.name)

	// Expand/collapse indicator.
	expand := "▸"
	if a.Expanded {
		expand = "▾"
	}

	// Summary line: icon  tool_name  truncated_args  ▸
	fmt.Fprintf(&b, "%s %s %s %s", iconStr, nameStr, argsStr, expand)

	if !a.Expanded {
		return b.String()
	}

	// Expanded view: full args and result.
	b.WriteString("\n")
	indent := a.theme.Indent()
	muted := a.theme.MutedStyle()

	// Full args.
	b.WriteString(muted.Render(indent + "args: "))
	b.WriteString(a.args)
	b.WriteString("\n")

	// Result (or "running…" message).
	if a.Running {
		b.WriteString(muted.Render(indent + "running…"))
	} else if a.result != "" {
		b.WriteString(muted.Render(indent + "result: "))
		for _, line := range strings.Split(a.result, "\n") {
			b.WriteString(indent)
			b.WriteString(indent)
			b.WriteString(line)
			b.WriteString("\n")
		}
		// Remove the trailing newline added by the loop; the parent
		// viewport adds its own between elements.
		s := b.String()
		if strings.HasSuffix(s, "\n") {
			b.Reset()
			b.WriteString(strings.TrimSuffix(s, "\n"))
		}
	}

	return b.String()
}

// truncateArgs shortens args so the summary line fits within width.
func truncateArgs(args string, width int, toolName string) string {
	// Reserve space for icon, name, expand indicator, and padding.
	reserve := len(toolName) + 10
	maxLen := width - reserve
	if maxLen < 10 {
		maxLen = 10
	}
	// Collapse whitespace for a compact single-line view.
	flat := strings.ReplaceAll(args, "\n", " ")
	flat = strings.TrimSpace(flat)
	if len(flat) <= maxLen {
		return flat
	}
	return flat[:maxLen-3] + "…"
}
