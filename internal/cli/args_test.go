package cli

import (
	"errors"
	"os"
	"strings"
	"testing"
)

func TestParseArgs_NoArgs(t *testing.T) {
	args, err := ParseArgs(nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(args.Positional) != 0 {
		t.Errorf("expected no positional args, got %v", args.Positional)
	}
}

func TestParseArgs_PositionalOnly(t *testing.T) {
	args, err := ParseArgs([]string{"hello"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(args.Positional) != 1 || args.Positional[0] != "hello" {
		t.Errorf("expected Positional=[hello], got %v", args.Positional)
	}
}

func TestParseArgs_FlagsBeforePositional(t *testing.T) {
	args, err := ParseArgs([]string{"--model", "gpt-4o", "--print", "hello"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if args.Model != "gpt-4o" {
		t.Errorf("Model = %q, want %q", args.Model, "gpt-4o")
	}
	if !args.Print {
		t.Errorf("Print = false, want true")
	}
	if len(args.Positional) != 1 || args.Positional[0] != "hello" {
		t.Errorf("Positional = %v, want [hello]", args.Positional)
	}
}

func TestParseArgs_LongFlagEqualsValue(t *testing.T) {
	args, err := ParseArgs([]string{"--model=gpt-4o", "--cwd=/tmp"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if args.Model != "gpt-4o" {
		t.Errorf("Model = %q", args.Model)
	}
	if args.Cwd != "/tmp" {
		t.Errorf("Cwd = %q", args.Cwd)
	}
}

func TestParseArgs_ShortFlags(t *testing.T) {
	args, err := ParseArgs([]string{"-m", "claude-opus-4-5", "-p", "anthropic", "-c", "/home"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if args.Model != "claude-opus-4-5" {
		t.Errorf("Model = %q", args.Model)
	}
	if args.Provider != "anthropic" {
		t.Errorf("Provider = %q", args.Provider)
	}
	if args.Cwd != "/home" {
		t.Errorf("Cwd = %q", args.Cwd)
	}
}

func TestParseArgs_ShortFlagAttachedValue(t *testing.T) {
	args, err := ParseArgs([]string{"-mclaude-opus-4-5"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if args.Model != "claude-opus-4-5" {
		t.Errorf("Model = %q", args.Model)
	}
}

func TestParseArgs_EndOfFlagsMarker(t *testing.T) {
	args, err := ParseArgs([]string{"--", "--print"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if args.Print {
		t.Errorf("Print should be false after --")
	}
	if len(args.Positional) != 1 || args.Positional[0] != "--print" {
		t.Errorf("Positional = %v, want [--print]", args.Positional)
	}
}

func TestParseArgs_JSONImpliesPrint(t *testing.T) {
	args, err := ParseArgs([]string{"--json"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !args.JSON {
		t.Errorf("JSON should be true")
	}
	if !args.Print {
		t.Errorf("Print should be implied by JSON")
	}
}

func TestParseArgs_BooleanFlags(t *testing.T) {
	cases := []struct {
		flag string
	}{
		{"--offline"}, {"--no-setup"}, {"--rpc"}, {"--continue"},
		{"--no-session"}, {"--fork"}, {"--list-models"},
		{"--version"}, {"--help"}, {"-h"},
	}
	for _, c := range cases {
		t.Run(c.flag, func(t *testing.T) {
			args, err := ParseArgs([]string{c.flag})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			// Just verify the flag is recognized without error.
			_ = args
		})
	}
}

func TestParseArgs_Subcommand(t *testing.T) {
	args, err := ParseArgs([]string{"config", "--path"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if args.Subcommand != "config" {
		t.Errorf("Subcommand = %q", args.Subcommand)
	}
	if len(args.SubcommandArgs) != 1 || args.SubcommandArgs[0] != "--path" {
		t.Errorf("SubcommandArgs = %v", args.SubcommandArgs)
	}
}

func TestParseArgs_UnknownFlagReturnsError(t *testing.T) {
	_, err := ParseArgs([]string{"--bogus"})
	if err == nil {
		t.Fatalf("expected error for unknown flag")
	}
	var pe *ParseError
	if !errors.As(err, &pe) {
		t.Fatalf("expected *ParseError, got %T (%v)", err, err)
	}
}

func TestParseArgs_FlagWithoutValue(t *testing.T) {
	_, err := ParseArgs([]string{"--model"})
	if err == nil {
		t.Fatalf("expected error for missing flag value")
	}
}

func TestParseArgs_ShortFlagWithoutValue(t *testing.T) {
	_, err := ParseArgs([]string{"-m"})
	if err == nil {
		t.Fatalf("expected error for missing short flag value")
	}
}

func TestParseError_Format(t *testing.T) {
	pe := &ParseError{Flag: "bogus", Msg: "unknown flag"}
	got := pe.Error()
	if !strings.Contains(got, "bogus") || !strings.Contains(got, "unknown flag") {
		t.Errorf("Error() = %q, expected to contain flag and message", got)
	}
}

func TestDeriveMode_ExplicitFlags(t *testing.T) {
	// We can't easily fake TTYs in a unit test; verify the explicit-flag paths
	// instead, which don't depend on TTY status.
	args := Args{RPC: true}
	if got := args.DeriveMode(nil, nil); got != "rpc" {
		t.Errorf("DeriveMode(RPC) = %q, want rpc", got)
	}
	args = Args{Print: true, JSON: true}
	if got := args.DeriveMode(nil, nil); got != "json" {
		t.Errorf("DeriveMode(Print+JSON) = %q, want json", got)
	}
	args = Args{Print: true}
	if got := args.DeriveMode(nil, nil); got != "print" {
		t.Errorf("DeriveMode(Print) = %q, want print", got)
	}
}

func TestDeriveMode_NonTTYFallback_Print(t *testing.T) {
	// stdin/stdout are *os.File pointers; passing nil for both is not a TTY
	// path. We open two *os.Files on os.DevNull which are guaranteed non-TTY.
	in, err := os.Open(os.DevNull)
	if err != nil {
		t.Fatalf("open DevNull: %v", err)
	}
	defer in.Close()
	out, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		t.Fatalf("open DevNull w: %v", err)
	}
	defer out.Close()
	args := Args{}
	if got := args.DeriveMode(in, out); got != "print" {
		t.Errorf("DeriveMode on non-TTY = %q, want print", got)
	}
}

func TestDeriveMode_PresetModeWins(t *testing.T) {
	// args.Mode is set by the caller (e.g., from a future env var or
	// settings directive); it overrides TTY auto-detect.
	args := Args{Mode: "interactive"}
	if got := args.DeriveMode(nil, nil); got != "interactive" {
		t.Errorf("DeriveMode(Mode=interactive, no TTY) = %q, want interactive", got)
	}
}

func TestExplicitMode(t *testing.T) {
	cases := []struct {
		args Args
		want string
	}{
		{Args{RPC: true}, "rpc"},
		{Args{Print: true, JSON: true}, "json"},
		{Args{Print: true}, "print"},
		{Args{}, ""},
		{Args{Mode: "interactive"}, ""}, // Mode is not an explicit flag
	}
	for _, c := range cases {
		if got := c.args.ExplicitMode(); got != c.want {
			t.Errorf("ExplicitMode(%+v) = %q, want %q", c.args, got, c.want)
		}
	}
}
