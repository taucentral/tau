package config

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestKeybinding_UnmarshalJSON_StringForm(t *testing.T) {
	var k Keybinding
	if err := json.Unmarshal([]byte(`"ctrl+c"`), &k); err != nil {
		t.Fatalf("unmarshal string form: %v", err)
	}
	if len(k) != 1 || k[0] != "ctrl+c" {
		t.Errorf("got %v, want [ctrl+c]", k)
	}
}

func TestKeybinding_UnmarshalJSON_ArrayForm(t *testing.T) {
	var k Keybinding
	if err := json.Unmarshal([]byte(`["shift+enter", "ctrl+j"]`), &k); err != nil {
		t.Fatalf("unmarshal array form: %v", err)
	}
	if len(k) != 2 || k[0] != "shift+enter" || k[1] != "ctrl+j" {
		t.Errorf("got %v, want [shift+enter ctrl+j]", k)
	}
}

func TestKeybinding_UnmarshalJSON_InvalidShapes(t *testing.T) {
	cases := []string{
		`42`,
		`true`,
		`["ctrl+c", 7]`,
		`{"key": "ctrl+c"}`,
	}
	for _, in := range cases {
		var k Keybinding
		err := json.Unmarshal([]byte(in), &k)
		if err == nil {
			t.Errorf("expected error for %s; got nil (result=%v)", in, k)
		}
	}
}

// TestKeybinding_UnmarshalJSON_NullIsNoop verifies that JSON null is accepted
// as a no-op: the destination stays at its zero value. This matches how
// encoding/json handles null for other custom types.
func TestKeybinding_UnmarshalJSON_NullIsNoop(t *testing.T) {
	var k Keybinding = Keybinding{"preexisting"}
	if err := json.Unmarshal([]byte(`null`), &k); err != nil {
		t.Fatalf("null should be accepted as a no-op: %v", err)
	}
	if len(k) != 0 {
		t.Errorf("null should leave the binding empty; got %v", k)
	}
}

func TestKeybinding_MarshalJSON_RoundTrip(t *testing.T) {
	cases := []Keybinding{
		{"enter"},
		{"shift+enter", "ctrl+j"},
	}
	for _, original := range cases {
		data, err := json.Marshal(original)
		if err != nil {
			t.Fatalf("marshal %v: %v", original, err)
		}
		// Single-element bindings marshal as a string; multi-element as array.
		switch len(original) {
		case 1:
			if !strings.HasPrefix(string(data), `"`) {
				t.Errorf("single-element binding %v should marshal as string; got %s", original, data)
			}
		default:
			if !strings.HasPrefix(string(data), `[`) {
				t.Errorf("multi-element binding %v should marshal as array; got %s", original, data)
			}
		}
		var roundTripped Keybinding
		if err := json.Unmarshal(data, &roundTripped); err != nil {
			t.Fatalf("unmarshal marshalled %s: %v", data, err)
		}
		if len(roundTripped) != len(original) {
			t.Errorf("round-trip length mismatch: got %v want %v", roundTripped, original)
		}
		for i := range original {
			if roundTripped[i] != original[i] {
				t.Errorf("round-trip[%d]: got %q want %q", i, roundTripped[i], original[i])
			}
		}
	}
}

func TestKeybinding_InSettingsRoundTrip(t *testing.T) {
	// Verify that the Settings struct's Keybindings field accepts both forms
	// via strict JSON decoding.
	src := `{
		"keybindings": {
			"submit": "enter",
			"newLine": ["shift+enter", "ctrl+j"]
		}
	}`
	var s Settings
	dec := json.NewDecoder(strings.NewReader(src))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&s); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got := s.Keybindings["submit"]; len(got) != 1 || got[0] != "enter" {
		t.Errorf("submit = %v, want [enter]", got)
	}
	if got := s.Keybindings["newLine"]; len(got) != 2 || got[0] != "shift+enter" || got[1] != "ctrl+j" {
		t.Errorf("newLine = %v, want [shift+enter ctrl+j]", got)
	}
}
