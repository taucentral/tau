package config

import (
	"errors"
	"strings"
	"testing"
)

func TestResolveValue_Empty(t *testing.T) {
	got, err := ResolveValue("")
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if got != "" {
		t.Errorf("got %q, want empty", got)
	}
}

func TestResolveValue_Literal(t *testing.T) {
	got, err := ResolveValue("hello world")
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if got != "hello world" {
		t.Errorf("got %q, want literal passthrough", got)
	}
}

func TestResolveValue_LiteralWithSpecialChars(t *testing.T) {
	// No $ at all → literal.
	got, _ := ResolveValue("sk-12345!@#")
	if got != "sk-12345!@#" {
		t.Errorf("got %q", got)
	}
}

func TestResolveValue_EnvVarSet(t *testing.T) {
	t.Setenv("TAU_TEST_RESOLVE", "resolved-value")
	got, err := ResolveValue("$TAU_TEST_RESOLVE")
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if got != "resolved-value" {
		t.Errorf("got %q, want \"resolved-value\"", got)
	}
}

func TestResolveValue_EnvVarUnset(t *testing.T) {
	// Explicitly unset by first setting to a value (which t.Setenv
	// restores at the end) — os.Unsetenv is needed to actually unset.
	// Easiest robust approach: pick a name that won't exist.
	got, err := ResolveValue("$TAU_DEFINITELY_NOT_SET_X1Y2Z3")
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if got != "" {
		t.Errorf("got %q, want empty for unset env var", got)
	}
}

func TestResolveValue_BracedEnvVar(t *testing.T) {
	t.Setenv("TAU_TEST_BRACED", "braced-val")
	got, _ := ResolveValue("${TAU_TEST_BRACED}")
	if got != "braced-val" {
		t.Errorf("got %q", got)
	}
}

func TestResolveValue_BracedUnknownName_Literal(t *testing.T) {
	// ${1bad} is not a valid env name; emit literally.
	got, _ := ResolveValue("${1bad}")
	if got != "${1bad}" {
		t.Errorf("got %q, want literal", got)
	}
}

func TestResolveValue_BracedUnterminated_Literal(t *testing.T) {
	got, _ := ResolveValue("${UNTERMINATED")
	if got != "${UNTERMINATED" {
		t.Errorf("got %q, want literal", got)
	}
}

func TestResolveValue_DollarDollarEscapesToLiteralDollar(t *testing.T) {
	got, _ := ResolveValue("$$VAR")
	if got != "$VAR" {
		t.Errorf("got %q, want \"$VAR\" (escape)", got)
	}
}

func TestResolveValue_DollarBangEscapesToLiteralBang(t *testing.T) {
	got, _ := ResolveValue("$!echo")
	if got != "!echo" {
		t.Errorf("got %q, want \"!echo\" (escape)", got)
	}
}

func TestResolveValue_TrailingDollar_Literal(t *testing.T) {
	got, _ := ResolveValue("foo$")
	if got != "foo$" {
		t.Errorf("got %q", got)
	}
}

func TestResolveValue_LoneDollar_Literal(t *testing.T) {
	got, _ := ResolveValue("$")
	if got != "$" {
		t.Errorf("got %q", got)
	}
}

func TestResolveValue_DollarFollowedByNonName_Literal(t *testing.T) {
	got, _ := ResolveValue("$123")
	if got != "$123" {
		t.Errorf("got %q, want \"$123\" (digits can't start env name)", got)
	}
}

func TestResolveValue_MixedTemplate(t *testing.T) {
	t.Setenv("TAU_A", "alpha")
	t.Setenv("TAU_B", "beta")
	got, _ := ResolveValue("pre-$TAU_A-mid-${TAU_B}-post")
	want := "pre-alpha-mid-beta-post"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestResolveValue_CommandSuccess(t *testing.T) {
	got, err := ResolveValue("!echo hello")
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if got != "hello" {
		t.Errorf("got %q, want \"hello\"", got)
	}
}

func TestResolveValue_CommandMultilineTrimsTrailingOnly(t *testing.T) {
	got, err := ResolveValue("!printf 'line1\\nline2\\n'")
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	want := "line1\nline2"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestResolveValue_CommandFailure(t *testing.T) {
	_, err := ResolveValue("!false")
	if err == nil {
		t.Fatalf("expected error for failing command")
	}
	if !errors.Is(err, ErrInvalidValue) {
		t.Errorf("err = %v, want ErrInvalidValue", err)
	}
	if !strings.Contains(err.Error(), "exited") {
		t.Errorf("err %q should mention exit", err.Error())
	}
}

func TestResolveValue_CommandEmptyAfterBang(t *testing.T) {
	_, err := ResolveValue("!  ")
	if err == nil {
		t.Fatalf("expected error for empty command")
	}
	if !errors.Is(err, ErrInvalidValue) {
		t.Errorf("err = %v, want ErrInvalidValue", err)
	}
}
