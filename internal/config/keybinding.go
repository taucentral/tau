// keybinding.go — flexible key-list type for Settings.Keybindings.
//
// Pi's KeybindingsConfig (third-party/pi/packages/tui/src/keybindings.ts:52)
// accepts each entry as `KeyId | KeyId[] | undefined`, so a single action can
// be bound to one key or several. The on-disk JSON therefore looks like either:
//
//	{ "keybindings": { "submit": "enter" } }
//	{ "keybindings": { "newLine": ["shift+enter", "ctrl+j"] } }
//
// Keybinding is the typed form of one entry: a non-empty slice of key strings,
// with a custom UnmarshalJSON that accepts the single-string form and promotes
// it to a one-element slice. MarshalJSON round-trips losslessly.
package config

import (
	"encoding/json"
	"fmt"
)

// Keybinding is one action's key list. The zero value is an empty (unset)
// binding; nil and empty are treated equivalently by callers.
type Keybinding []string

// UnmarshalJSON accepts either a JSON string ("ctrl+c") or a JSON array of
// strings (["ctrl+c", "ctrl+d"]). Any other shape is an error. JSON null
// leaves the destination at its zero value (treated as "unset"). Empty
// strings inside an array are kept; ResolveKeybindings in the tui package
// drops them.
func (k *Keybinding) UnmarshalJSON(data []byte) error {
	// JSON null → unset. encoding/json still calls our method for null when
	// the destination has a custom UnmarshalJSON, so we have to handle it
	// explicitly to avoid turning null into Keybinding{""} via the string
	// branch below.
	if string(data) == "null" {
		*k = nil
		return nil
	}

	// Try the single-string form first. Pi uses this for most actions.
	var single string
	if err := json.Unmarshal(data, &single); err == nil {
		*k = Keybinding{single}
		return nil
	}

	// Fall back to the array form. Re-decode as []string so type checking
	// is strict: a heterogeneous array like [1, "x"] is rejected.
	var multi []string
	if err := json.Unmarshal(data, &multi); err == nil {
		*k = Keybinding(multi)
		return nil
	}

	return fmt.Errorf("keybinding must be a string or array of strings, got %s", string(data))
}

// MarshalJSON serialises as the compact form: single-element bindings round-trip
// as a JSON string; multi-element bindings as a JSON array. This keeps the
// settings file readable for the common case.
func (k Keybinding) MarshalJSON() ([]byte, error) {
	switch len(k) {
	case 0:
		return []byte("null"), nil
	case 1:
		return json.Marshal(k[0])
	default:
		return json.Marshal([]string(k))
	}
}
