package tui

import (
	"sort"
	"testing"

	"github.com/taucentral/tau/internal/config"
)

func TestDefaultKeybindings_AllActionsPresent(t *testing.T) {
	kb := DefaultKeybindings()
	required := []string{
		ActionSubmit, ActionNewline, ActionQuit, ActionAbort,
		ActionScrollUp, ActionScrollDown, ActionExpand,
		ActionHistoryPrev, ActionHistoryNext,
		ActionTreeOpen, ActionTreeUp, ActionTreeDown,
	}
	for _, action := range required {
		if _, ok := kb[action]; !ok {
			t.Errorf("DefaultKeybindings missing action %q", action)
		}
	}
}

// TestDefaultKeybindings_MatchPi verifies tau's defaults mirror pi's
// coding-agent defaults. Reference:
//
//	third-party/pi/packages/tui/src/keybindings.ts:118-119
//	third-party/pi/packages/coding-agent/src/core/keybindings.ts:63-67
func TestDefaultKeybindings_MatchPi(t *testing.T) {
	kb := DefaultKeybindings()
	cases := []struct {
		action string
		want   config.Keybinding
	}{
		{ActionSubmit, config.Keybinding{"enter"}},
		{ActionNewline, config.Keybinding{"shift+enter", "ctrl+j"}},
		{ActionAbort, config.Keybinding{"escape"}},
		{ActionQuit, config.Keybinding{"ctrl+d"}},
	}
	for _, tt := range cases {
		t.Run(tt.action, func(t *testing.T) {
			got := kb[tt.action]
			if len(got) != len(tt.want) {
				t.Fatalf("%s = %v, want %v", tt.action, got, tt.want)
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("%s[%d] = %q, want %q", tt.action, i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestResolveKeybindings_OverridesApplied(t *testing.T) {
	overrides := map[string]config.Keybinding{
		ActionSubmit: {"ctrl+s"},
		ActionQuit:   {"ctrl+q"},
	}
	kb := ResolveKeybindings(overrides)
	if got := kb.Get(ActionSubmit); got != "ctrl+s" {
		t.Errorf("submit = %q, want 'ctrl+s'", got)
	}
	if got := kb.Get(ActionQuit); got != "ctrl+q" {
		t.Errorf("quit = %q, want 'ctrl+q'", got)
	}
	// Non-overridden actions keep defaults.
	def := DefaultKeybindings()
	if got, want := kb.Get(ActionAbort), def[ActionAbort][0]; got != want {
		t.Errorf("abort = %q, want default %q", got, want)
	}
}

// TestResolveKeybindings_MultiKeyOverride verifies that an override with
// multiple keys is preserved end-to-end.
func TestResolveKeybindings_MultiKeyOverride(t *testing.T) {
	overrides := map[string]config.Keybinding{
		ActionSubmit: {"ctrl+s", "ctrl+enter"},
	}
	kb := ResolveKeybindings(overrides)
	got := kb.Keys(ActionSubmit)
	if len(got) != 2 {
		t.Fatalf("submit override = %v, want 2 keys", got)
	}
	if got[0] != "ctrl+s" || got[1] != "ctrl+enter" {
		t.Errorf("submit override = %v, want [ctrl+s ctrl+enter]", got)
	}
	if !kb.Matches(ActionSubmit, "ctrl+s") {
		t.Error("ctrl+s should match submit")
	}
	if !kb.Matches(ActionSubmit, "ctrl+enter") {
		t.Error("ctrl+enter should match submit")
	}
	if kb.Matches(ActionSubmit, "enter") {
		t.Error("default 'enter' should no longer match submit after override")
	}
}

func TestResolveKeybindings_UnknownActionDropped(t *testing.T) {
	overrides := map[string]config.Keybinding{
		"bogusAction": {"ctrl+z"},
		ActionSubmit:  {"ctrl+j"},
	}
	kb := ResolveKeybindings(overrides)
	if _, ok := kb["bogusAction"]; ok {
		t.Error("unknown action should be dropped")
	}
	if got := kb.Get(ActionSubmit); got != "ctrl+j" {
		t.Errorf("submit = %q, want 'ctrl+j'", got)
	}
}

func TestResolveKeybindings_NilOverrides(t *testing.T) {
	kb := ResolveKeybindings(nil)
	def := DefaultKeybindings()
	for action, keys := range def {
		if got := kb.Get(action); got != keys[0] {
			t.Errorf("nil overrides: %s = %q, want %q", action, got, keys[0])
		}
	}
}

// TestResolveKeybindings_EmptyEntriesDropped verifies that empty strings
// inside an override are dropped, but valid entries in the same list survive.
func TestResolveKeybindings_EmptyEntriesDropped(t *testing.T) {
	overrides := map[string]config.Keybinding{
		ActionSubmit: {"", "ctrl+s", ""},
	}
	kb := ResolveKeybindings(overrides)
	got := kb.Keys(ActionSubmit)
	if len(got) != 1 || got[0] != "ctrl+s" {
		t.Errorf("submit keys = %v, want [ctrl+s]", got)
	}
}

// TestResolveKeybindings_AllEmptyEntriesFallsBackToDefault verifies that an
// override list containing only empty strings does not clear the binding —
// it falls back to the default.
func TestResolveKeybindings_AllEmptyEntriesFallsBackToDefault(t *testing.T) {
	overrides := map[string]config.Keybinding{
		ActionSubmit: {"", ""},
	}
	kb := ResolveKeybindings(overrides)
	def := DefaultKeybindings()
	if got, want := kb.Get(ActionSubmit), def[ActionSubmit][0]; got != want {
		t.Errorf("all-empty override: submit = %q, want default %q", got, want)
	}
}

func TestDetectConflicts_NoConflicts(t *testing.T) {
	kb := Keybindings{
		"a": config.Keybinding{"ctrl+a"},
		"b": config.Keybinding{"ctrl+b"},
		"c": config.Keybinding{"ctrl+c"},
	}
	if conflicts := DetectConflicts(kb); len(conflicts) != 0 {
		t.Errorf("expected 0 conflicts, got %d: %v", len(conflicts), conflicts)
	}
}

func TestDetectConflicts_FindsDuplicates(t *testing.T) {
	kb := Keybindings{
		"a": config.Keybinding{"ctrl+x"},
		"b": config.Keybinding{"ctrl+x"},
		"c": config.Keybinding{"ctrl+y"},
		"d": config.Keybinding{"ctrl+x"},
	}
	// Three actions share "ctrl+x": (a,b), (a,d), (b,d) = 3 pairs.
	conflicts := DetectConflicts(kb)
	if len(conflicts) != 3 {
		t.Errorf("expected 3 conflicts for 3 actions on same key, got %d: %v", len(conflicts), conflicts)
	}
	for _, c := range conflicts {
		if c.Key != "ctrl+x" {
			t.Errorf("conflict key = %q, want 'ctrl+x'", c.Key)
		}
	}
}

// TestDetectConflicts_HandlesMultiKeyActions verifies that conflict detection
// inspects every key in a multi-key binding.
func TestDetectConflicts_HandlesMultiKeyActions(t *testing.T) {
	kb := Keybindings{
		"a": config.Keybinding{"ctrl+x", "ctrl+y"},
		"b": config.Keybinding{"ctrl+y"},
	}
	conflicts := DetectConflicts(kb)
	// "ctrl+y" is shared by a and b; one pair.
	if len(conflicts) != 1 {
		t.Fatalf("expected 1 conflict, got %d: %v", len(conflicts), conflicts)
	}
	if conflicts[0].Key != "ctrl+y" {
		t.Errorf("conflict key = %q, want 'ctrl+y'", conflicts[0].Key)
	}
}

// TestDetectConflicts_DefaultsHaveSubmitAndTreeOpenOverlap documents the
// intentional default-vs-default overlap: ActionSubmit and ActionTreeOpen are
// both bound to "enter" because they fire in different focus contexts
// (textarea vs tree view). The overlap is detected by DetectConflicts but
// is NOT surfaced as a user diagnostic (see TestDetectUserConflicts_*).
func TestDetectConflicts_DefaultsHaveSubmitAndTreeOpenOverlap(t *testing.T) {
	conflicts := DetectConflicts(DefaultKeybindings())
	found := false
	for _, c := range conflicts {
		if c.Key == "enter" &&
			((c.ActionA == ActionSubmit && c.ActionB == ActionTreeOpen) ||
				(c.ActionA == ActionTreeOpen && c.ActionB == ActionSubmit)) {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected submit/treeOpen overlap on 'enter' in defaults; conflicts = %v", conflicts)
	}
}

// TestDetectUserConflicts_DefaultsAreClean verifies the property that matters
// for users: with no user overrides, no conflicts are reported. The
// submit/treeOpen overlap on "enter" is intentional and stays silent.
func TestDetectUserConflicts_DefaultsAreClean(t *testing.T) {
	if conflicts := DetectUserConflicts(nil); len(conflicts) != 0 {
		t.Errorf("nil overrides should produce no conflicts; got %v", conflicts)
	}
}

func TestDetectUserConflicts_ReportsUserOverlap(t *testing.T) {
	overrides := map[string]config.Keybinding{
		ActionSubmit: {"ctrl+j"},
		ActionAbort:  {"ctrl+j"}, // same as submit
	}
	conflicts := DetectUserConflicts(overrides)
	if len(conflicts) != 1 {
		t.Fatalf("expected 1 user conflict, got %d: %v", len(conflicts), conflicts)
	}
	c := conflicts[0]
	if c.Key != "ctrl+j" {
		t.Errorf("conflict key = %q, want 'ctrl+j'", c.Key)
	}
	if c.ActionA != ActionAbort && c.ActionB != ActionAbort {
		t.Errorf("expected abort in the pair; got %v", c)
	}
}

// TestDetectUserConflicts_IgnoresUnknownActions verifies that typos in user
// overrides do not produce spurious conflicts.
func TestDetectUserConflicts_IgnoresUnknownActions(t *testing.T) {
	overrides := map[string]config.Keybinding{
		"bogusAction": {"ctrl+j"},
		"alsoBogus":   {"ctrl+j"},
		ActionSubmit:  {"enter"},
	}
	if conflicts := DetectUserConflicts(overrides); len(conflicts) != 0 {
		t.Errorf("unknown actions should not produce conflicts; got %v", conflicts)
	}
}

func TestResolveWithConflicts_UserConflictPath(t *testing.T) {
	overrides := map[string]config.Keybinding{
		ActionSubmit: {"ctrl+j"},
		ActionAbort:  {"ctrl+j"}, // conflict with submit
	}
	kb, conflicts := ResolveWithConflicts(overrides)
	if got := kb.Get(ActionSubmit); got != "ctrl+j" {
		t.Errorf("submit = %q, want 'ctrl+j'", got)
	}
	if got := kb.Get(ActionAbort); got != "ctrl+j" {
		t.Errorf("abort = %q, want 'ctrl+j'", got)
	}
	found := false
	for _, c := range conflicts {
		if c.Key == "ctrl+j" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected a conflict on 'ctrl+j', got: %v", conflicts)
	}
}

// TestResolveWithConflicts_DefaultsAreSilent is the end-to-end smoke check
// for the bug originally reported by the user: a fresh TUI with no user
// overrides must produce no keybind-conflict diagnostics.
func TestResolveWithConflicts_DefaultsAreSilent(t *testing.T) {
	_, conflicts := ResolveWithConflicts(nil)
	if len(conflicts) != 0 {
		t.Errorf("fresh install must have no keybind conflicts; got %v", conflicts)
	}
}

func TestKeybindingsGet_MissingAction(t *testing.T) {
	kb := Keybindings{}
	def := DefaultKeybindings()
	got := kb.Get(ActionSubmit)
	if want := def[ActionSubmit][0]; got != want {
		t.Errorf("Get on empty kb: submit = %q, want default %q", got, want)
	}
}

func TestKeybindingsGet_UnknownAction(t *testing.T) {
	kb := DefaultKeybindings()
	if got := kb.Get("bogus"); got != "" {
		t.Errorf("Get on unknown action: got %q, want empty", got)
	}
}

func TestKeybindingsKeys_CopyNotAlias(t *testing.T) {
	kb := DefaultKeybindings()
	keys := kb.Keys(ActionNewline)
	if len(keys) == 0 {
		t.Fatal("expected at least one newline key")
	}
	original := keys[0]
	keys[0] = "mutated"
	// Mutating the returned slice must not affect kb.
	if kb[ActionNewline][0] != original {
		t.Errorf("Keys() returned a slice that aliases internal state; mutation leaked: %v", kb[ActionNewline])
	}
}

func TestKeybindingsMatches(t *testing.T) {
	kb := DefaultKeybindings()
	tests := []struct {
		action string
		key    string
		want   bool
	}{
		{ActionSubmit, "enter", true},
		{ActionSubmit, "Enter", true},   // case-insensitive
		{ActionSubmit, " enter ", true}, // trimmed
		{ActionSubmit, "ctrl+enter", false},
		{ActionNewline, "shift+enter", true},
		{ActionNewline, "ctrl+j", true},
		{ActionNewline, "enter", false},
		{ActionQuit, "ctrl+d", true},
		{ActionAbort, "escape", true},
		{ActionAbort, "esc", false},
	}
	for _, tt := range tests {
		t.Run(tt.action+"/"+tt.key, func(t *testing.T) {
			got := kb.Matches(tt.action, tt.key)
			if got != tt.want {
				t.Errorf("Matches(%q, %q) = %v, want %v", tt.action, tt.key, got, tt.want)
			}
		})
	}
}

func TestConflictsAsDiagnostics(t *testing.T) {
	conflicts := []Conflict{
		{Key: "ctrl+x", ActionA: "submit", ActionB: "abort"},
	}
	diags := ConflictsAsDiagnostics(conflicts)
	if len(diags) != 1 {
		t.Fatalf("expected 1 diagnostic, got %d", len(diags))
	}
	if diags[0] == "" {
		t.Error("diagnostic is empty")
	}
	if d := ConflictsAsDiagnostics(nil); d != nil {
		t.Errorf("nil conflicts: got %v, want nil", d)
	}
}

func TestResolveKeybindings_DoesNotMutateDefaults(t *testing.T) {
	original := DefaultKeybindings()[ActionSubmit]
	overrides := map[string]config.Keybinding{
		ActionSubmit: {"ctrl+s"},
	}
	_ = ResolveKeybindings(overrides)
	after := DefaultKeybindings()[ActionSubmit]
	if len(after) != len(original) || after[0] != original[0] {
		t.Errorf("ResolveKeybindings mutated DefaultKeybindings: submit = %v, want %v", after, original)
	}
}

func TestDetectConflicts_SortedByKey(t *testing.T) {
	kb := Keybindings{
		"a": config.Keybinding{"z"},
		"b": config.Keybinding{"z"},
		"c": config.Keybinding{"a"},
		"d": config.Keybinding{"a"},
	}
	conflicts := DetectConflicts(kb)
	keys := make([]string, len(conflicts))
	for i, c := range conflicts {
		keys[i] = c.Key
	}
	if !sort.StringsAreSorted(keys) {
		t.Errorf("conflicts not sorted by key: %v", keys)
	}
}
