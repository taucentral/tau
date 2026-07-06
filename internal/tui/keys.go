// Package tui — keys.go — keybinding definitions and conflict detection.
//
// The TUI MUST NOT hardcode keybindings in component code. Each component reads
// from a *Keybindings struct produced by ResolveKeybindings, which merges
// DefaultKeybindings with user overrides found in Settings.Keybindings.
//
// Defaults mirror pi so users carry muscle memory across both tools:
//
//   - submit  → enter
//   - newline → shift+enter, ctrl+j (two keys because some terminals cannot
//     distinguish shift+enter from enter; ctrl+j (LF) always works)
//   - abort   → escape
//   - quit    → ctrl+d
//
// Reference:
//
//	third-party/pi/packages/tui/src/keybindings.ts:118     (tui.input.submit / newLine)
//	third-party/pi/packages/coding-agent/src/core/keybindings.ts:63-67 (app.interrupt/exit)
//
// Conflict detection follows pi's rule (keybindings.ts:171-185): only
// conflicts AMONG user-supplied overrides are reported. Default-vs-default
// overlaps (e.g., "enter" is both ActionSubmit and ActionTreeOpen) are
// intentional because the two actions fire in different focus contexts
// (textarea vs tree view); they are not surfaced as diagnostics.
package tui

import (
	"fmt"
	"sort"
	"strings"

	"github.com/taucentral/tau/internal/config"
)

// Action identifiers. These are the canonical names used in
// Settings.Keybindings and in DefaultKeybindings. Components reference these
// constants rather than raw strings so typos are caught at compile time.
const (
	ActionSubmit      = "submit"      // submit input textarea
	ActionNewline     = "newline"     // insert a newline in the textarea
	ActionQuit        = "quit"        // exit the TUI
	ActionAbort       = "abort"       // cancel the in-flight turn
	ActionScrollUp    = "scrollUp"    // scroll the viewport up
	ActionScrollDown  = "scrollDown"  // scroll the viewport down
	ActionExpand      = "expand"      // expand/collapse focused accordion
	ActionHistoryPrev = "historyPrev" // navigate to previous input in history
	ActionHistoryNext = "historyNext" // navigate to next input in history
	ActionTreeOpen    = "treeOpen"    // open/checkout selected tree entry
	ActionTreeUp      = "treeUp"      // move selection up in tree view
	ActionTreeDown    = "treeDown"    // move selection down in tree view
)

// Keybindings maps action names (the Action* constants above) to one or more
// key strings in bubbletea's format ("ctrl+enter", "esc", "up", etc.). A
// multi-key action fires when ANY of its keys is pressed.
type Keybindings map[string]config.Keybinding

// DefaultKeybindings returns the canonical key assignments, mirroring pi's
// defaults (see file doc). Components use these when no user override exists.
//
// Reference:
//
//	third-party/pi/packages/tui/src/keybindings.ts:118-133
//	third-party/pi/packages/coding-agent/src/core/keybindings.ts:63-67
func DefaultKeybindings() Keybindings {
	return Keybindings{
		ActionSubmit:      config.Keybinding{"enter"},
		ActionNewline:     config.Keybinding{"shift+enter", "ctrl+j"},
		ActionQuit:        config.Keybinding{"ctrl+d"},
		ActionAbort:       config.Keybinding{"escape"},
		ActionScrollUp:    config.Keybinding{"pgup"},
		ActionScrollDown:  config.Keybinding{"pgdown"},
		ActionExpand:      config.Keybinding{"tab"},
		ActionHistoryPrev: config.Keybinding{"up"},
		ActionHistoryNext: config.Keybinding{"down"},
		ActionTreeOpen:    config.Keybinding{"enter"},
		ActionTreeUp:      config.Keybinding{"shift+up"},
		ActionTreeDown:    config.Keybinding{"shift+down"},
	}
}

// ResolveKeybindings merges defaults with user-supplied overrides from
// settings. Unknown action names are dropped silently so typos in
// settings.json don't pollute the binding map. Empty key strings inside an
// override are dropped (the action falls back to its default).
//
// The returned Keybindings is a fresh map; mutating it does not affect the
// caller's settings or future calls to DefaultKeybindings.
func ResolveKeybindings(overrides map[string]config.Keybinding) Keybindings {
	out := DefaultKeybindings()
	def := DefaultKeybindings()
	for action, keys := range overrides {
		if _, isKnown := out[action]; !isKnown {
			continue
		}
		// Drop empty strings from the user-supplied list.
		filtered := make(config.Keybinding, 0, len(keys))
		for _, k := range keys {
			if strings.TrimSpace(k) != "" {
				filtered = append(filtered, k)
			}
		}
		if len(filtered) == 0 {
			// All entries were empty; keep the default.
			out[action] = def[action]
			continue
		}
		out[action] = filtered
	}
	return out
}

// Conflict pairs two actions that are bound to the same key.
type Conflict struct {
	Key     string
	ActionA string
	ActionB string
}

// String renders a Conflict for diagnostic output.
func (c Conflict) String() string {
	return fmt.Sprintf("keybind conflict: %q and %q are both bound to %q", c.ActionA, c.ActionB, c.Key)
}

// DetectConflicts scans the bindings for keys assigned to more than one
// action and returns one Conflict per overlapping pair, sorted by key. This
// is the low-level primitive; the TUI's startup diagnostic path uses
// DetectUserConflicts instead, since default-vs-default overlaps are
// intentional (different focus contexts) and should not be surfaced to users.
func DetectConflicts(kb Keybindings) []Conflict {
	// Group actions by their (normalized) key string.
	byKey := make(map[string][]string)
	for action, keys := range kb {
		for _, key := range keys {
			nk := normalizeKey(key)
			if nk == "" {
				continue
			}
			byKey[nk] = append(byKey[nk], action)
		}
	}
	return pairConflicts(byKey)
}

// DetectUserConflicts inspects only the user-supplied overrides for keys
// claimed by more than one action. Default-vs-default overlaps (e.g.,
// ActionSubmit and ActionTreeOpen both bound to "enter") are NOT reported
// here because they fire in different focus contexts and are intentional.
//
// This mirrors pi's KeybindingsManager, which only scans userBindings for
// conflicts (third-party/pi/packages/tui/src/keybindings.ts:171).
func DetectUserConflicts(overrides map[string]config.Keybinding) []Conflict {
	byKey := make(map[string][]string)
	for action, keys := range overrides {
		if _, isKnown := DefaultKeybindings()[action]; !isKnown {
			continue
		}
		for _, key := range keys {
			nk := normalizeKey(key)
			if nk == "" {
				continue
			}
			byKey[nk] = append(byKey[nk], action)
		}
	}
	return pairConflicts(byKey)
}

// pairConflicts builds Conflict pairs from a key→actions index. Pairs within
// each key are emitted in sorted action order; the result is sorted by key
// for deterministic output.
func pairConflicts(byKey map[string][]string) []Conflict {
	var conflicts []Conflict
	for key, actions := range byKey {
		if len(actions) < 2 {
			continue
		}
		sort.Strings(actions)
		// Deduplicate action names: the same action can appear twice if a
		// user mistakenly lists a key twice in one binding.
		seen := make(map[string]bool, len(actions))
		unique := actions[:0]
		for _, a := range actions {
			if !seen[a] {
				seen[a] = true
				unique = append(unique, a)
			}
		}
		for i := 0; i < len(unique); i++ {
			for j := i + 1; j < len(unique); j++ {
				conflicts = append(conflicts, Conflict{
					Key:     key,
					ActionA: unique[i],
					ActionB: unique[j],
				})
			}
		}
	}
	sort.Slice(conflicts, func(i, j int) bool {
		if conflicts[i].Key != conflicts[j].Key {
			return conflicts[i].Key < conflicts[j].Key
		}
		if conflicts[i].ActionA != conflicts[j].ActionA {
			return conflicts[i].ActionA < conflicts[j].ActionA
		}
		return conflicts[i].ActionB < conflicts[j].ActionB
	})
	return conflicts
}

// ResolveWithConflicts is a convenience that merges defaults + overrides and
// runs user-only conflict detection. Returns the resolved bindings and any
// conflicts found among the overrides.
func ResolveWithConflicts(overrides map[string]config.Keybinding) (Keybindings, []Conflict) {
	kb := ResolveKeybindings(overrides)
	return kb, DetectUserConflicts(overrides)
}

// Get returns the first key string for the named action. Used by UI surfaces
// that show one key per action (e.g., the help overlay). Falls back to the
// default's first key when the action is missing from kb. Returns "" for an
// unknown action.
func (kb Keybindings) Get(action string) string {
	if keys, ok := kb[action]; ok && len(keys) > 0 {
		return keys[0]
	}
	def := DefaultKeybindings()
	if keys, ok := def[action]; ok && len(keys) > 0 {
		return keys[0]
	}
	return ""
}

// Keys returns all keys bound to the named action. The returned slice is a
// copy; callers may mutate it freely. Falls back to defaults when the action
// is missing from kb.
func (kb Keybindings) Keys(action string) []string {
	if keys, ok := kb[action]; ok && len(keys) > 0 {
		out := make([]string, len(keys))
		copy(out, keys)
		return out
	}
	def := DefaultKeybindings()
	if keys, ok := def[action]; ok && len(keys) > 0 {
		out := make([]string, len(keys))
		copy(out, keys)
		return out
	}
	return nil
}

// Matches reports whether the given key string matches any of the action's
// bound keys. Both sides are normalized (trimmed, lowercased) before
// comparison, so "Ctrl+Enter" in settings matches "ctrl+enter" in bubbletea's
// KeyMsg.String().
func (kb Keybindings) Matches(action, key string) bool {
	want := normalizeKey(key)
	if want == "" {
		return false
	}
	for _, candidate := range kb.Keys(action) {
		if normalizeKey(candidate) == want {
			return true
		}
	}
	return false
}

// normalizeKey lowercases and trims the key string so comparisons are
// tolerant of user capitalisation in settings.json.
func normalizeKey(s string) string {
	return strings.ToLower(strings.TrimSpace(s))
}

// ConflictsAsDiagnostics converts a Conflict slice to human-readable
// diagnostic strings suitable for the startup diagnostic channel.
func ConflictsAsDiagnostics(conflicts []Conflict) []string {
	if len(conflicts) == 0 {
		return nil
	}
	out := make([]string, len(conflicts))
	for i, c := range conflicts {
		out[i] = c.String()
	}
	return out
}
