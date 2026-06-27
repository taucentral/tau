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
