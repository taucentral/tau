package tools

// Theme controls how RenderCall and RenderResult format their output. The
// tools package uses ANSI escape codes directly so it has no dependency on
// the TUI layer; Phase 10's richer Theme converts to this shape when
// invoking tool rendering.
//
// All color fields are full SGR sequences (e.g., "\x1b[31m" for red) so
// they can be concatenated into rendered strings without parsing. An empty
// field means "no styling" — output is plain text suitable for non-TTY
// contexts (testing, JSON mode, redirected output).
type Theme struct {
	// Primary is the main accent color (tool name in call rendering).
	Primary string
	// Secondary is a supporting color (argument names, metadata keys).
	Secondary string
	// Accent is a high-attention color (active spinners, leaf indicators).
	Accent string
	// Success marks a successful (non-error) tool result.
	Success string
	// Error marks an errored tool result.
	Error string
	// Warning marks truncated output, partial results.
	Warning string
	// Muted is for low-priority text (timestamps, sizes).
	Muted string
	// Reset is the SGR reset sequence. Defaults to "\x1b[0m"; an empty
	// value is treated as the empty string for non-TTY themes.
	Reset string
	// Indent is the per-level indentation string used in rendered output.
	// Defaults to two spaces.
	Indent string
}

// PlainTheme is the zero-ANSI Theme used for non-TTY rendering (tests, JSON
// mode, redirected output). All style fields are empty; rendered text is
// plain.
func PlainTheme() *Theme {
	return &Theme{
		Indent: "  ",
	}
}

// ColorTheme is a basic dark-background Theme suitable for an interactive
// TTY. It uses the standard 16-color palette so it works on any terminal
// without truecolor support.
func ColorTheme() *Theme {
	return &Theme{
		Primary:   "\x1b[36m", // cyan
		Secondary: "\x1b[33m", // yellow
		Accent:    "\x1b[35m", // magenta
		Success:   "\x1b[32m", // green
		Error:     "\x1b[31m", // red
		Warning:   "\x1b[33m", // yellow
		Muted:     "\x1b[2m",  // dim
		Reset:     "\x1b[0m",
		Indent:    "  ",
	}
}

// Wrap returns s wrapped in the given style prefix and the theme's Reset
// suffix. If the style is empty, s is returned unchanged (no escape codes).
func (t *Theme) Wrap(style, s string) string {
	if style == "" {
		return s
	}
	if t.Reset == "" {
		return style + s + "\x1b[0m"
	}
	return style + s + t.Reset
}

// Indent returns the indentation string for the given depth.
func (t *Theme) IndentN(depth int) string {
	if t.Indent == "" {
		return ""
	}
	out := ""
	for i := 0; i < depth; i++ {
		out += t.Indent
	}
	return out
}
