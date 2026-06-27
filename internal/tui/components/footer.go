// Package components — footer.go — bottom status bar.
//
// The Footer packs session metadata (cwd, context fill, model, provider
// API, thinking level) into two lines at the bottom of the TUI. It
// replaces the narrower right-hand Sidebar and reclaims that width for
// the conversation viewport.
//
// Reference layout (pi):
//
//	third-party/pi/packages/coding-agent/src/modes/interactive/components/footer.ts:117-221
//
// tau deviates from pi by omitting git branch, session name, cache and
// cost columns, and cumulative token totals — those subsystems aren't
// implemented yet. The Footer is structured so each can be added later
// by extending the render branches below without re-architecting the
// component or its callers.
package components

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// FooterHeightLines is the number of screen lines View() emits. Callers
// (AppModel layout math) consult this constant rather than guessing.
const FooterHeightLines = 2

// Footer renders the two-line bottom status bar.
type Footer struct {
	theme         Theme
	width         int
	cwd           string
	model         string
	providerAPI   string
	thinkingLevel string
	tokensUsed    int
	tokensMax     int
}

// NewFooter returns a footer with the given theme and width. Use the
// Set* methods to populate fields; View renders exactly
// footerHeightLines lines.
func NewFooter(theme Theme, width int) *Footer {
	return &Footer{theme: theme, width: width}
}

// SetWidth updates the footer width. Called on terminal resize.
func (f *Footer) SetWidth(width int) { f.width = width }

// SetCwd sets the working directory shown on line 1. Render substitutes
// $HOME with "~" (see shortenCwd).
func (f *Footer) SetCwd(cwd string) { f.cwd = cwd }

// SetModel sets the model identifier shown on line 2.
func (f *Footer) SetModel(model string) { f.model = model }

// SetProviderAPI sets the API family tag (e.g. "anthropic", "openai")
// rendered in square brackets after the model. Empty suppresses the tag.
func (f *Footer) SetProviderAPI(api string) { f.providerAPI = api }

// SetThinkingLevel sets the thinking-effort label rendered after the
// model as "* <level>". Empty and "off" both suppress the suffix.
func (f *Footer) SetThinkingLevel(level string) { f.thinkingLevel = level }

// SetTokens updates the context-fill gauge. used/maxTokens are token counts;
// pass maxTokens==0 when the context window is unknown (the gauge renders "?").
func (f *Footer) SetTokens(used, maxTokens int) {
	f.tokensUsed = used
	f.tokensMax = maxTokens
}

// View renders the footer as two newline-joined lines.
func (f *Footer) View() string {
	return f.renderCwd() + "\n" + f.renderStatus()
}

// renderCwd renders line 1: the cwd with $HOME → "~" substitution.
func (f *Footer) renderCwd() string {
	return f.theme.MutedStyle().Render(shortenCwd(f.cwd))
}

// renderStatus renders line 2: tokens (left) + model info (right),
// joined with whitespace so the model info right-aligns to f.width.
func (f *Footer) renderStatus() string {
	left := f.renderTokens()
	right := f.renderModel()
	gap := f.width - lipgloss.Width(left) - lipgloss.Width(right)
	if gap < 1 {
		// Terminal too narrow to fit both sides; keep them separated
		// by a single space rather than overlapping.
		gap = 1
	}
	return left + strings.Repeat(" ", gap) + right
}

// renderTokens renders the context-fill gauge with color coding:
// warning above 90%, accent above 70%, muted otherwise. An unknown
// window (max==0) renders just "?".
func (f *Footer) renderTokens() string {
	if f.tokensMax <= 0 {
		return f.theme.MutedStyle().Render("?")
	}
	pct := float64(f.tokensUsed) / float64(f.tokensMax) * 100
	if pct > 100 {
		pct = 100
	}
	style := f.theme.MutedStyle()
	switch {
	case pct > 90:
		style = f.theme.WarningStyle()
	case pct > 70:
		style = f.theme.AccentStyle()
	}
	return style.Render(fmt.Sprintf("%.0f%% / %s", pct, formatTokenCount(f.tokensMax)))
}

// renderModel renders "<model>[ * <thinking>][ [<api>]]". The thinking
// suffix is suppressed when the level is empty or "off"; the API tag is
// suppressed when empty.
func (f *Footer) renderModel() string {
	var b strings.Builder
	b.WriteString(f.model)
	if f.thinkingLevel != "" && !strings.EqualFold(f.thinkingLevel, "off") {
		b.WriteString(" * ")
		b.WriteString(f.thinkingLevel)
	}
	if f.providerAPI != "" {
		b.WriteString(" [")
		b.WriteString(f.providerAPI)
		b.WriteString("]")
	}
	return f.theme.AssistantStyle().Render(b.String())
}

// shortenCwd substitutes a $HOME prefix with "~". Returns the input
// unchanged when $HOME is unset, empty, or not a prefix of cwd.
func shortenCwd(cwd string) string {
	if cwd == "" {
		return ""
	}
	home, err := os.UserHomeDir()
	if err == nil && home != "" {
		if cwd == home {
			return "~"
		}
		if strings.HasPrefix(cwd, home+"/") {
			return "~" + cwd[len(home):]
		}
	}
	return cwd
}

// formatTokenCount renders a token count compactly: "<1k" as plain
// digits, 1k–10k with one decimal, ≥10k as whole kilos.
func formatTokenCount(n int) string {
	if n < 1000 {
		return strconv.Itoa(n)
	}
	if n < 10000 {
		return fmt.Sprintf("%.1fk", float64(n)/1000)
	}
	return fmt.Sprintf("%dk", n/1000)
}
