package config

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestDefaultSettings_HasRequiredDefaults(t *testing.T) {
	s := DefaultSettings()
	if s.Compaction == nil || !*s.Compaction.Enabled {
		t.Errorf("Compaction.Enabled should default to true")
	}
	if s.Compaction == nil || s.Compaction.ReserveTokens == nil || *s.Compaction.ReserveTokens != 8192 {
		t.Errorf("Compaction.ReserveTokens should default to 8192")
	}
	if s.Retry == nil || s.Retry.MaxRetries == nil || *s.Retry.MaxRetries != 4 {
		t.Errorf("Retry.MaxRetries should default to 4")
	}
	if s.Retry == nil || s.Retry.Provider == nil || s.Retry.Provider.MaxRetryDelayMs == nil {
		t.Errorf("Retry.Provider.MaxRetryDelayMs should be set")
	} else if *s.Retry.Provider.MaxRetryDelayMs != 30000 {
		t.Errorf("Retry.Provider.MaxRetryDelayMs = %d, want 30000", *s.Retry.Provider.MaxRetryDelayMs)
	}
	if s.Terminal == nil || s.Terminal.ShowImages == nil || !*s.Terminal.ShowImages {
		t.Errorf("Terminal.ShowImages should default to true")
	}
	if s.Transport == nil || *s.Transport != TransportAuto {
		t.Errorf("Transport should default to auto")
	}
	if s.SteeringMode == nil || *s.SteeringMode != SteeringOneAtATime {
		t.Errorf("SteeringMode should default to one-at-a-time")
	}
	if s.DefaultThinkingLevel == nil || *s.DefaultThinkingLevel != ThinkingOff {
		t.Errorf("DefaultThinkingLevel should default to off")
	}
}

func TestDefaultSettings_NotMutatedAcrossCalls(t *testing.T) {
	a := DefaultSettings()
	if a.Compaction == nil || a.Compaction.ReserveTokens == nil {
		t.Fatal("missing defaults")
	}
	*a.Compaction.ReserveTokens = 9999

	b := DefaultSettings()
	if *b.Compaction.ReserveTokens != 8192 {
		t.Errorf("DefaultSettings() returned a value affected by prior mutation: got %d, want 8192",
			*b.Compaction.ReserveTokens)
	}
}

func TestSettings_JSONRoundTrip(t *testing.T) {
	s := DefaultSettings()
	data, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var got Settings
	if err := strictJSONDecode(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	// Spot-check a couple of fields.
	if got.Compaction == nil || got.Compaction.ReserveTokens == nil {
		t.Fatalf("Compaction.ReserveTokens missing after round trip")
	}
	if *got.Compaction.ReserveTokens != 8192 {
		t.Errorf("ReserveTokens = %d, want 8192", *got.Compaction.ReserveTokens)
	}
}

func TestSettings_UnknownFieldRejected(t *testing.T) {
	const bad = `{"modle":"typo"}` //nolint:misspell // the typo is the test's point
	var s Settings
	err := strictJSONDecode([]byte(bad), &s)
	if err == nil {
		t.Fatalf("expected schema violation for unknown field")
	}
	if !errors.Is(err, ErrSchemaViolation) {
		t.Errorf("err = %v, want ErrSchemaViolation", err)
	}
}

// TestSettings_ToolsUnknownFieldRejected verifies that strict JSON
// decoding rejects unknown fields inside the "tools" namespace,
// per task 5.3.
func TestSettings_ToolsUnknownFieldRejected(t *testing.T) {
	const bad = `{"tools":{"hydrationMode":"heuristic","bogusField":42}}`
	var s Settings
	err := strictJSONDecode([]byte(bad), &s)
	if err == nil {
		t.Fatalf("expected schema violation for unknown tools field")
	}
	if !errors.Is(err, ErrSchemaViolation) {
		t.Errorf("err = %v, want ErrSchemaViolation", err)
	}
}

// TestSettings_ToolsDefaultsApplied verifies the default Settings
// carries sensible Tools defaults (heuristic mode, window=5).
func TestSettings_ToolsDefaultsApplied(t *testing.T) {
	s := DefaultSettings()
	if s.Tools == nil {
		t.Fatalf("Tools should be non-nil in DefaultSettings")
	}
	if s.Tools.HydrationMode == nil || *s.Tools.HydrationMode != "heuristic" {
		t.Errorf("Tools.HydrationMode = %v, want \"heuristic\"", s.Tools.HydrationMode)
	}
	if s.Tools.RecentUseWindow == nil || *s.Tools.RecentUseWindow != 5 {
		t.Errorf("Tools.RecentUseWindow = %v, want 5", s.Tools.RecentUseWindow)
	}
}

// TestSettings_ToolsRoundTrip verifies ToolsSettings survives a
// marshal → strict-decode round trip.
func TestSettings_ToolsRoundTrip(t *testing.T) {
	mode := "off"
	window := 10
	always := []string{"read", "write"}
	s := Settings{
		Tools: &ToolsSettings{
			HydrationMode:   &mode,
			RecentUseWindow: &window,
			AlwaysRender:    always,
		},
	}
	data, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var back Settings
	if err := strictJSONDecode(data, &back); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if back.Tools == nil {
		t.Fatalf("Tools missing after round trip")
	}
	if back.Tools.HydrationMode == nil || *back.Tools.HydrationMode != "off" {
		t.Errorf("HydrationMode = %v, want \"off\"", back.Tools.HydrationMode)
	}
	if back.Tools.RecentUseWindow == nil || *back.Tools.RecentUseWindow != 10 {
		t.Errorf("RecentUseWindow = %v, want 10", back.Tools.RecentUseWindow)
	}
	if len(back.Tools.AlwaysRender) != 2 {
		t.Errorf("AlwaysRender len = %d, want 2", len(back.Tools.AlwaysRender))
	}
}

func TestSettings_InvalidThinkingLevel(t *testing.T) {
	bad := ThinkingLevel("bogus")
	s := Settings{DefaultThinkingLevel: &bad}
	if err := s.Validate(); !errors.Is(err, ErrSchemaViolation) {
		t.Errorf("Validate err = %v, want ErrSchemaViolation", err)
	}
}

func TestSettings_InvalidTransport(t *testing.T) {
	bad := TransportSetting("udp")
	s := Settings{Transport: &bad}
	if err := s.Validate(); !errors.Is(err, ErrSchemaViolation) {
		t.Errorf("Validate err = %v, want ErrSchemaViolation", err)
	}
}

func TestSettings_InvalidSteeringMode(t *testing.T) {
	bad := SteeringMode("two-at-a-time")
	s := Settings{SteeringMode: &bad}
	if err := s.Validate(); !errors.Is(err, ErrSchemaViolation) {
		t.Errorf("Validate err = %v, want ErrSchemaViolation", err)
	}
}

func TestSettings_ValidEnumPasses(t *testing.T) {
	lvl := ThinkingHigh
	tr := TransportWebsocket
	mode := SteeringAll
	s := Settings{
		DefaultThinkingLevel: &lvl,
		Transport:            &tr,
		SteeringMode:         &mode,
	}
	if err := s.Validate(); err != nil {
		t.Errorf("Validate err = %v, want nil", err)
	}
}

func TestSettings_OptionalFieldsZeroValuedWhenOmitted(t *testing.T) {
	const minimal = `{"defaultModel":"gpt-4o"}`
	var s Settings
	if err := strictJSONDecode([]byte(minimal), &s); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if s.DefaultModel == nil || *s.DefaultModel != "gpt-4o" {
		t.Errorf("DefaultModel = %v, want \"gpt-4o\"", s.DefaultModel)
	}
	if s.Compaction != nil {
		t.Errorf("Compaction should be nil when omitted, got %+v", s.Compaction)
	}
	if s.Retry != nil {
		t.Errorf("Retry should be nil when omitted")
	}
}

func TestSettings_DefaultProjectTrustEnum(t *testing.T) {
	cases := []string{"ask", "always", "never"}
	for _, c := range cases {
		v := c
		s := Settings{DefaultProjectTrust: &v}
		if err := s.Validate(); err != nil {
			t.Errorf("Validate(DefaultProjectTrust=%q) = %v, want nil", c, err)
		}
	}
	bad := "sometimes"
	s := Settings{DefaultProjectTrust: &bad}
	if err := s.Validate(); !errors.Is(err, ErrSchemaViolation) {
		t.Errorf("Validate err = %v, want ErrSchemaViolation", err)
	}
}

// Verify that an empty Settings has zero impact on the JSON tag mapping.
func TestSettings_EmptyHasNoOptionalFields(t *testing.T) {
	var s Settings
	data, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	got := string(data)
	if got != "{}" {
		t.Errorf("empty Settings marshals to %s, want {}", got)
	}
	// And contains no camelCase keys.
	if strings.Contains(got, ":") {
		t.Errorf("empty Settings should have no keys: %s", got)
	}
}

// TestPromptsSettings_DefaultsAreZero confirms that an unset
// PromptsSettings behaves identically to an explicit zero-valued
// one. This is the acceptance test for task 3.2: "WalkToRoot=false,
// MaxAncestorDepth=0 (unlimited), StopDir='' are the defaults".
//
// The runtime translates nil pointer fields to zero-value WalkOpts,
// so a nil *PromptsSettings and an &PromptsSettings{} must produce
// the same walk behavior. We check this via JSON round-trip: the
// empty struct marshals to "{}" and a DefaultSettings() has nil
// Prompts.
func TestPromptsSettings_DefaultsAreZero(t *testing.T) {
	// DefaultSettings should leave Prompts nil (nil == zero for
	// pointer-to-struct with all-zero fields).
	def := DefaultSettings()
	if def.Prompts != nil {
		t.Errorf("DefaultSettings().Prompts = %+v, want nil (nil == zero-value defaults)", def.Prompts)
	}

	// An explicit zero PromptsSettings should marshal to "{}" —
	// all fields are omitempty pointers.
	empty := &PromptsSettings{}
	data, err := json.Marshal(empty)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if string(data) != "{}" {
		t.Errorf("zero PromptsSettings marshals to %s, want {}", string(data))
	}

	// A populated PromptsSettings should round-trip all three fields.
	walk := true
	depth := 2
	stop := "/tmp/stop"
	populated := &PromptsSettings{
		WalkToRoot:       &walk,
		MaxAncestorDepth: &depth,
		StopDir:          &stop,
	}
	data, err = json.Marshal(populated)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var back PromptsSettings
	if err := strictJSONDecode(data, &back); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if back.WalkToRoot == nil || *back.WalkToRoot != walk {
		t.Errorf("WalkToRoot did not round-trip: got %+v", back.WalkToRoot)
	}
	if back.MaxAncestorDepth == nil || *back.MaxAncestorDepth != depth {
		t.Errorf("MaxAncestorDepth did not round-trip: got %+v", back.MaxAncestorDepth)
	}
	if back.StopDir == nil || *back.StopDir != stop {
		t.Errorf("StopDir did not round-trip: got %+v", back.StopDir)
	}
}
