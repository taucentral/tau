// Package components implements tau's interactive terminal UI building
// blocks. Each component (viewport, footer, input, treeview, accordion,
// markdown renderer) is self-contained and renders via a shared Theme.
//
// Theme lives in this package rather than the parent tui package to avoid
// an import cycle (the parent tui package imports this one).
package components

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/charmbracelet/lipgloss"
)

// Theme holds the colours, indentation, and keyhint styles for the TUI.
// Every component reads from a *Theme so that swapping themes (dark ↔
// light, or a user-supplied custom JSON file) takes effect everywhere.
type Theme struct {
	// Name identifies the theme (e.g. "dark", "light"). Used in
	// diagnostics and saved settings.
	Name string `json:"name"`

	// Core palette.
	Background lipgloss.Color `json:"-"`
	Foreground lipgloss.Color `json:"-"`
	Primary    lipgloss.Color `json:"primary"`
	Secondary  lipgloss.Color `json:"secondary"`
	Accent     lipgloss.Color `json:"accent"`
	Muted      lipgloss.Color `json:"muted"`

	// Role-specific colours.
	UserMsg      lipgloss.Color `json:"userMsg"`
	AssistantMsg lipgloss.Color `json:"assistantMsg"`
	ToolName     lipgloss.Color `json:"toolName"`
	ErrorColor   lipgloss.Color `json:"error"`
	SuccessColor lipgloss.Color `json:"success"`
	WarningColor lipgloss.Color `json:"warning"`

	// Layout.
	Indentation int `json:"indentation"`

	// Border style for panes.
	BorderStyle lipgloss.Border `json:"-"`
}

// themeJSON is the on-disk representation. Colors are strings (ANSI
// hex or named) because lipgloss.Color does not marshal/unmarshal
// natively.
type themeJSON struct {
	Name         string `json:"name"`
	Primary      string `json:"primary"`
	Secondary    string `json:"secondary"`
	Accent       string `json:"accent"`
	Muted        string `json:"muted"`
	UserMsg      string `json:"userMsg"`
	AssistantMsg string `json:"assistantMsg"`
	ToolName     string `json:"toolName"`
	Error        string `json:"error"`
	Success      string `json:"success"`
	Warning      string `json:"warning"`
	Indentation  int    `json:"indentation"`
}

// DarkTheme is the default colour scheme for terminals with a dark
// background.
var DarkTheme = Theme{
	Name:         "dark",
	Background:   lipgloss.Color("#1a1a2e"),
	Foreground:   lipgloss.Color("#e0e0e0"),
	Primary:      lipgloss.Color("#7aa2f7"),
	Secondary:    lipgloss.Color("#bb9af7"),
	Accent:       lipgloss.Color("#e0af68"),
	Muted:        lipgloss.Color("#565f89"),
	UserMsg:      lipgloss.Color("#7dcfff"),
	AssistantMsg: lipgloss.Color("#c0caf5"),
	ToolName:     lipgloss.Color("#73daca"),
	ErrorColor:   lipgloss.Color("#f7768e"),
	SuccessColor: lipgloss.Color("#9ece6a"),
	WarningColor: lipgloss.Color("#e0af68"),
	Indentation:  2,
	BorderStyle:  lipgloss.RoundedBorder(),
}

// LightTheme is the default colour scheme for terminals with a light
// background.
var LightTheme = Theme{
	Name:         "light",
	Background:   lipgloss.Color("#f5f5f5"),
	Foreground:   lipgloss.Color("#343b58"),
	Primary:      lipgloss.Color("#34548a"),
	Secondary:    lipgloss.Color("#8c4351"),
	Accent:       lipgloss.Color("#8f6108"),
	Muted:        lipgloss.Color("#9699a3"),
	UserMsg:      lipgloss.Color("#0f4b6e"),
	AssistantMsg: lipgloss.Color("#343b58"),
	ToolName:     lipgloss.Color("#33635c"),
	ErrorColor:   lipgloss.Color("#8c4351"),
	SuccessColor: lipgloss.Color("#33635c"),
	WarningColor: lipgloss.Color("#8f6108"),
	Indentation:  2,
	BorderStyle:  lipgloss.RoundedBorder(),
}

// DefaultTheme returns the theme matching the given name. Falls back to
// DarkTheme when name is empty or unrecognised.
func DefaultTheme(name string) Theme {
	switch name {
	case "light":
		return LightTheme
	default:
		return DarkTheme
	}
}

// LoadTheme resolves a theme by name. If the name matches one of the
// built-in themes ("dark", "light"), the corresponding default is
// returned. Otherwise the themes directory under agentDir is searched
// for a JSON file named `<name>.json`.
//
// Returns an error only when a custom theme file exists but cannot be
// parsed. A missing custom file falls back to DarkTheme silently so
// the TUI can still launch.
func LoadTheme(agentDir, name string) (Theme, error) {
	if name == "" || name == "dark" {
		return DarkTheme, nil
	}
	if name == "light" {
		return LightTheme, nil
	}

	path := filepath.Join(agentDir, "themes", name+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return DarkTheme, nil
		}
		return Theme{}, fmt.Errorf("components.LoadTheme: read %s: %w", path, err)
	}

	var raw themeJSON
	if err := json.Unmarshal(data, &raw); err != nil {
		return Theme{}, fmt.Errorf("components.LoadTheme: parse %s: %w", path, err)
	}

	return themeFromJSON(raw), nil
}

// themeFromJSON converts the on-disk representation to a Theme.
func themeFromJSON(raw themeJSON) Theme {
	t := Theme{
		Name:         raw.Name,
		Primary:      lipgloss.Color(raw.Primary),
		Secondary:    lipgloss.Color(raw.Secondary),
		Accent:       lipgloss.Color(raw.Accent),
		Muted:        lipgloss.Color(raw.Muted),
		UserMsg:      lipgloss.Color(raw.UserMsg),
		AssistantMsg: lipgloss.Color(raw.AssistantMsg),
		ToolName:     lipgloss.Color(raw.ToolName),
		ErrorColor:   lipgloss.Color(raw.Error),
		SuccessColor: lipgloss.Color(raw.Success),
		WarningColor: lipgloss.Color(raw.Warning),
		Indentation:  raw.Indentation,
		BorderStyle:  lipgloss.RoundedBorder(),
	}
	if t.Name == "" {
		t.Name = "custom"
	}
	if t.Indentation <= 0 {
		t.Indentation = 2
	}
	return t
}

// UserStyle returns a lipgloss style for user messages.
func (t Theme) UserStyle() lipgloss.Style {
	return lipgloss.NewStyle().Foreground(t.UserMsg)
}

// AssistantStyle returns a lipgloss style for assistant messages.
func (t Theme) AssistantStyle() lipgloss.Style {
	return lipgloss.NewStyle().Foreground(t.AssistantMsg)
}

// ToolStyle returns a lipgloss style for tool names.
func (t Theme) ToolStyle() lipgloss.Style {
	return lipgloss.NewStyle().Foreground(t.ToolName)
}

// ErrorStyle returns a lipgloss style for error text.
func (t Theme) ErrorStyle() lipgloss.Style {
	return lipgloss.NewStyle().Foreground(t.ErrorColor)
}

// MutedStyle returns a lipgloss style for muted / secondary text.
func (t Theme) MutedStyle() lipgloss.Style {
	return lipgloss.NewStyle().Foreground(t.Muted)
}

// AccentStyle returns a lipgloss style for highlighted / active elements.
func (t Theme) AccentStyle() lipgloss.Style {
	return lipgloss.NewStyle().Foreground(t.Accent)
}

// SuccessStyle returns a lipgloss style for success / completed states.
func (t Theme) SuccessStyle() lipgloss.Style {
	return lipgloss.NewStyle().Foreground(t.SuccessColor)
}

// WarningStyle returns a lipgloss style for warning / diagnostic text.
func (t Theme) WarningStyle() lipgloss.Style {
	return lipgloss.NewStyle().Foreground(t.WarningColor)
}

// PrimaryStyle returns a lipgloss style for primary / focused elements.
func (t Theme) PrimaryStyle() lipgloss.Style {
	return lipgloss.NewStyle().Foreground(t.Primary).Bold(true)
}

// PaneStyle returns a lipgloss style for bordered panes.
func (t Theme) PaneStyle() lipgloss.Style {
	return lipgloss.NewStyle().
		Border(t.BorderStyle, true).
		BorderForeground(t.Muted)
}

// Indent returns a string of spaces matching the theme's indentation.
func (t Theme) Indent() string {
	if t.Indentation <= 0 {
		return ""
	}
	out := make([]byte, t.Indentation)
	for i := range out {
		out[i] = ' '
	}
	return string(out)
}
