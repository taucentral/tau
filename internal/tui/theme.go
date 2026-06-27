// Package tui implements tau's interactive terminal UI.
//
// This file re-exports the Theme type and helpers from the lower-level
// internal/tui/components package. Theme lives in components to avoid an
// import cycle (the parent tui package imports components, which needs
// the Theme type). External callers should keep using tui.Theme; the
// alias keeps the public surface stable.
package tui

import "github.com/coevin/tau/internal/tui/components"

// Theme is an alias for components.Theme so callers can refer to the
// type as tui.Theme without importing the components package directly.
type Theme = components.Theme

// DarkTheme is the default colour scheme for dark-background terminals.
var DarkTheme = components.DarkTheme

// LightTheme is the default colour scheme for light-background terminals.
var LightTheme = components.LightTheme

// DefaultTheme returns the built-in theme matching name. Falls back to
// DarkTheme when name is empty or unrecognised.
func DefaultTheme(name string) Theme { return components.DefaultTheme(name) }

// LoadTheme resolves a theme by name, falling back to DarkTheme when the
// built-in lookup or custom-file load fails silently. Returns an error
// only when a custom theme file exists but cannot be parsed.
func LoadTheme(agentDir, name string) (Theme, error) {
	return components.LoadTheme(agentDir, name)
}
