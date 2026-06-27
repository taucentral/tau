// Package cli implements tau's command-line interface: argument parsing,
// subcommand dispatch, and first-run setup.
//
// The parser is intentionally minimal (no cobra/urfave). It accepts:
//
//	tau [flags] [positional...]
//	tau [flags] -- positional...      (-- ends flag parsing)
//	tau <subcommand> [args...]        (config, update)
//
// Flag forms: --flag value | --flag=value | -f value | -fvalue
package cli

import (
	"fmt"
	"os"
	"strings"
)

// Args holds the parsed command-line arguments.
type Args struct {
	// BinaryVersion is the build-time version string. It is set by
	// cmd/tau/main.go and read by Dispatch when --version is requested.
	BinaryVersion string

	// Mode controls how tau runs. Values: "interactive", "print", "json", "rpc".
	// Auto-detected if empty (see Dispatch).
	Mode string

	// Positional is the list of non-flag arguments (the user prompt parts).
	Positional []string

	// Model overrides Settings.DefaultModel.
	Model string

	// Provider overrides Settings.DefaultProvider.
	Provider string

	// Thinking level: off|minimal|low|medium|high|xhigh.
	Thinking string

	// Tools is a glob pattern filtering the active tool set.
	Tools string

	// Cwd overrides the OS current working directory.
	Cwd string

	// Offline disables network-dependent features.
	Offline bool

	// Version requests --version output.
	Version bool

	// Help requests --help output.
	Help bool

	// NoSetup skips the first-run setup wizard.
	NoSetup bool

	// Print forces print mode (--print).
	Print bool

	// JSON forces print mode with JSON output (--json).
	JSON bool

	// RPC forces RPC mode (--rpc).
	RPC bool

	// Continue resumes the most recent session in cwd.
	Continue bool

	// NoSession runs without persisting state to disk.
	NoSession bool

	// Resume reopens a session by ID.
	Resume string

	// Fork creates a new session forked from the current one.
	Fork bool

	// Session opens a specific session file path.
	Session string

	// ListModels requests --list-models output.
	ListModels bool

	// Export writes the session transcript to the given path on exit.
	Export string

	// Subcommand is the first positional when it matches a known subcommand
	// (config, update). Empty when no subcommand was recognized.
	Subcommand string

	// SubcommandArgs is the remaining args after a recognized subcommand.
	SubcommandArgs []string
}

// ParseArgs parses argv into Args. It does not perform any I/O.
//
// Returns a typed *ParseError when argv is malformed so callers can exit 2
// (argument error) rather than 1 (runtime error).
func ParseArgs(argv []string) (Args, error) {
	args := Args{}

	// Recognize subcommands first. A subcommand consumes the rest of argv.
	if len(argv) > 0 {
		switch argv[0] {
		case "config", "update":
			args.Subcommand = argv[0]
			args.SubcommandArgs = argv[1:]
			return args, nil
		}
	}

	endOfFlags := false
	i := 0
	for i < len(argv) {
		tok := argv[i]
		if endOfFlags {
			args.Positional = append(args.Positional, tok)
			i++
			continue
		}
		switch {
		case tok == "--":
			endOfFlags = true
			i++
		case tok == "--help" || tok == "-h":
			args.Help = true
			i++
		case tok == "--version":
			args.Version = true
			i++
		case tok == "--offline":
			args.Offline = true
			i++
		case tok == "--no-setup":
			args.NoSetup = true
			i++
		case tok == "--print":
			args.Print = true
			i++
		case tok == "--json":
			args.JSON = true
			args.Print = true
			i++
		case tok == "--rpc":
			args.RPC = true
			i++
		case tok == "--continue":
			args.Continue = true
			i++
		case tok == "--no-session":
			args.NoSession = true
			i++
		case tok == "--fork":
			args.Fork = true
			i++
		case tok == "--list-models":
			args.ListModels = true
			i++
		case strings.HasPrefix(tok, "--"):
			// long flag with optional =value
			name, value, hasValue := splitLongFlag(tok[2:])
			if !hasValue {
				if i+1 >= len(argv) {
					return args, &ParseError{Flag: name, Msg: "missing value"}
				}
				value = argv[i+1]
				i += 2
			} else {
				i++
			}
			if err := assignLongFlag(&args, name, value); err != nil {
				return args, err
			}
		case strings.HasPrefix(tok, "-") && len(tok) > 1:
			// short flag (-f, -fvalue, -f value). Only the common ones are
			// supported here.
			name := tok[1:2]
			value := tok[2:]
			if value == "" {
				if i+1 >= len(argv) {
					return args, &ParseError{Flag: name, Msg: "missing value"}
				}
				value = argv[i+1]
				i += 2
			} else {
				i++
			}
			if err := assignShortFlag(&args, name, value); err != nil {
				return args, err
			}
		default:
			args.Positional = append(args.Positional, tok)
			i++
		}
	}
	return args, nil
}

// splitLongFlag splits "name=value" into ("name", "value", true) or
// ("name", "", false) when no = is present.
func splitLongFlag(s string) (string, string, bool) {
	if idx := strings.IndexByte(s, '='); idx >= 0 {
		return s[:idx], s[idx+1:], true
	}
	return s, "", false
}

func assignLongFlag(args *Args, name, value string) error {
	switch name {
	case "model":
		args.Model = value
	case "provider":
		args.Provider = value
	case "thinking":
		args.Thinking = value
	case "tools":
		args.Tools = value
	case "cwd":
		args.Cwd = value
	case "resume":
		args.Resume = value
	case "session":
		args.Session = value
	case "export":
		args.Export = value
	default:
		return &ParseError{Flag: name, Msg: "unknown flag"}
	}
	return nil
}

func assignShortFlag(args *Args, name, value string) error {
	switch name {
	case "m":
		args.Model = value
	case "p":
		args.Provider = value
	case "c":
		args.Cwd = value
	default:
		return &ParseError{Flag: name, Msg: "unknown short flag"}
	}
	return nil
}

// ParseError is returned by ParseArgs for malformed argv. Exit code should be 2.
type ParseError struct {
	Flag string
	Msg  string
}

func (e *ParseError) Error() string {
	return fmt.Sprintf("argument --%s: %s", e.Flag, e.Msg)
}

// ExplicitMode returns the mode selected by an explicit flag, or "" when
// no explicit selection was made. Resolution order:
//
//   - --rpc                  → "rpc"
//   - --json (implies print) → "json"
//   - --print                → "print"
//
// Empty return value means "auto-detect from TTY status" — see DeriveMode.
func (a Args) ExplicitMode() string {
	switch {
	case a.RPC:
		return "rpc"
	case a.JSON:
		return "json"
	case a.Print:
		return "print"
	}
	return ""
}

// DeriveMode returns the run mode selected by explicit flag, falling back
// to TTY detection per the modes spec. The stdin/stdout parameters are
// *os.File so callers can pass real TTYs or fakes in tests; pass nil to
// simulate a non-TTY (e.g., a pipe).
//
// Spec: "if --print or --json is set, use print mode; else if --rpc is
// set, use RPC mode; else if both stdin and stdout are TTYs, use
// interactive mode; else fall back to print mode. Explicit flags override
// auto-detection."
func (a Args) DeriveMode(stdin, stdout *os.File) string {
	if m := a.ExplicitMode(); m != "" {
		return m
	}
	// Pre-set Mode field wins over auto-detect.
	if a.Mode != "" {
		return a.Mode
	}
	in := isTTY(stdin)
	out := isTTY(stdout)
	if in && out {
		return "interactive"
	}
	return "print"
}
