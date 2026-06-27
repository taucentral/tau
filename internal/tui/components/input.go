// Package components — input.go — input textarea wrapper.
//
// The input component wraps bubbles/textarea for multi-line editing.
// It adds input history navigation (Up/Down) and slash-command
// autocomplete suggestions.
package components

import (
	"sort"
	"strings"

	"github.com/charmbracelet/bubbles/textarea"
	tea "github.com/charmbracelet/bubbletea"
)

// InputModel wraps a textarea with history and autocomplete.
type InputModel struct {
	theme      Theme
	ta         textarea.Model
	width      int
	height     int
	history    []string
	historyIdx int
	savedDraft string   // text saved when the user starts navigating history
	commands   []string // registered slash commands for autocomplete
	autoActive bool
	autoMatch  []string
	autoIndex  int
}

// NewInput returns an input model at the given dimensions.
func NewInput(theme Theme, width, height int) *InputModel {
	ta := textarea.New()
	ta.Placeholder = "Send a message… (Enter to submit, Shift+Enter for newline)"
	ta.Prompt = "│ "
	ta.ShowLineNumbers = false
	ta.SetWidth(width)
	ta.SetHeight(height)
	ta.CharLimit = 0 // unlimited
	return &InputModel{
		theme:  theme,
		ta:     ta,
		width:  width,
		height: height,
	}
}

// SetDimensions updates the textarea size.
func (m *InputModel) SetDimensions(width, height int) {
	m.width = width
	m.height = height
	m.ta.SetWidth(width)
	m.ta.SetHeight(height)
}

// Value returns the current text.
func (m *InputModel) Value() string { return m.ta.Value() }

// SetValue replaces the current text.
func (m *InputModel) SetValue(s string) { m.ta.SetValue(s) }

// Reset clears the textarea.
func (m *InputModel) Reset() {
	m.ta.Reset()
	m.autoActive = false
	m.autoMatch = nil
}

// Focus sets the input focus.
func (m *InputModel) Focus() tea.Cmd { return m.ta.Focus() }

// Blur removes the input focus.
func (m *InputModel) Blur() tea.Cmd { m.ta.Blur(); return nil }

// AppendHistory adds an input to the history.
func (m *InputModel) AppendHistory(input string) {
	input = strings.TrimSpace(input)
	if input == "" {
		return
	}
	// Don't add consecutive duplicates.
	if len(m.history) > 0 && m.history[len(m.history)-1] == input {
		return
	}
	m.history = append(m.history, input)
	m.historyIdx = len(m.history)
}

// HistoryPrev navigates to the previous history entry (Up arrow).
func (m *InputModel) HistoryPrev() {
	if len(m.history) == 0 {
		return
	}
	if m.historyIdx == len(m.history) {
		// Save the current draft so we can restore it on Down.
		m.savedDraft = m.ta.Value()
	}
	if m.historyIdx > 0 {
		m.historyIdx--
	}
	m.ta.SetValue(m.history[m.historyIdx])
	// Move cursor to end.
	m.ta.CursorEnd()
}

// HistoryNext navigates to the next history entry (Down arrow).
func (m *InputModel) HistoryNext() {
	if len(m.history) == 0 {
		return
	}
	if m.historyIdx < len(m.history) {
		m.historyIdx++
	}
	if m.historyIdx == len(m.history) {
		m.ta.SetValue(m.savedDraft)
		m.savedDraft = ""
	} else {
		m.ta.SetValue(m.history[m.historyIdx])
	}
	m.ta.CursorEnd()
}

// SetCommands registers the slash command names for autocomplete.
func (m *InputModel) SetCommands(names []string) {
	m.commands = make([]string, len(names))
	copy(m.commands, names)
	sort.Strings(m.commands)
}

// AutocompleteActive reports whether the autocomplete overlay is shown.
func (m *InputModel) AutocompleteActive() bool { return m.autoActive }

// AutocompleteStart computes matches for the current text and activates
// the overlay. Called when the user types a "/" prefix.
func (m *InputModel) AutocompleteStart() {
	text := m.ta.Value()
	if !strings.HasPrefix(text, "/") {
		m.autoActive = false
		return
	}
	m.autoMatch = nil
	for _, cmd := range m.commands {
		if strings.HasPrefix(cmd, text) {
			m.autoMatch = append(m.autoMatch, cmd)
		}
	}
	m.autoActive = len(m.autoMatch) > 0
	m.autoIndex = 0
}

// AutocompleteNext cycles to the next match. Does nothing if the
// overlay is not active.
func (m *InputModel) AutocompleteNext() {
	if !m.autoActive || len(m.autoMatch) == 0 {
		return
	}
	m.autoIndex = (m.autoIndex + 1) % len(m.autoMatch)
}

// AutocompleteAccept replaces the textarea with the selected match.
func (m *InputModel) AutocompleteAccept() {
	if !m.autoActive || len(m.autoMatch) == 0 {
		return
	}
	m.ta.SetValue(m.autoMatch[m.autoIndex] + " ")
	m.autoActive = false
	m.autoMatch = nil
}

// AutocompleteCancel dismisses the overlay.
func (m *InputModel) AutocompleteCancel() {
	m.autoActive = false
	m.autoMatch = nil
}

// Init implements tea.Model.
func (m *InputModel) Init() tea.Cmd { return nil }

// Update implements tea.Model.
func (m *InputModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd
	m.ta, cmd = m.ta.Update(msg)
	// Trigger autocomplete when the user types.
	if keyMsg, ok := msg.(tea.KeyMsg); ok {
		switch keyMsg.Type { //nolint:exhaustive // bubbletea defines dozens of KeyType variants; only Up/Down are meaningful here and the rest fall through to textarea's default handling.
		case tea.KeyUp:
			// Only navigate history if cursor is on the first line.
			if m.ta.Line() == 0 {
				m.HistoryPrev()
				return m, nil
			}
		case tea.KeyDown:
			if m.ta.Line()+1 >= m.ta.LineCount() {
				m.HistoryNext()
				return m, nil
			}
		}
		_ = keyMsg
	}
	return m, cmd
}

// View implements tea.Model.
func (m *InputModel) View() string {
	out := m.ta.View()
	if m.autoActive && len(m.autoMatch) > 0 {
		// Append the autocomplete overlay below the textarea.
		var b strings.Builder
		b.WriteString(out)
		b.WriteString("\n")
		muted := m.theme.MutedStyle()
		for i, match := range m.autoMatch {
			prefix := "  "
			if i == m.autoIndex {
				prefix = "▸ "
			}
			b.WriteString(muted.Render(prefix + match))
			b.WriteString("\n")
		}
		return b.String()
	}
	return out
}

// Textarea returns the underlying textarea model for direct manipulation.
func (m *InputModel) Textarea() *textarea.Model { return &m.ta }
