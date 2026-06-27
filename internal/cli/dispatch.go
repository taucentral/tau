// Package cli — dispatch.go — selects a run mode based on Args + TTY detection
// and invokes the corresponding run function.
package cli

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/coevin/tau/internal/modes"
)

// ErrNotImplemented is returned by Dispatch for functionality that the parser
// and dispatcher recognize but that is not yet implemented. It is a typed
// sentinel so tests and callers can distinguish "not implemented" from
// runtime errors.
//
// Mode handlers, subcommand handlers, and metadata emitters are wired here
// intentionally so the parser and dispatcher are exercised by `tau --help`,
// `tau --version`, and the metadata paths. The mode handlers themselves are
// implemented in later phases.
var ErrNotImplemented = errors.New("not implemented")

// Dispatch selects and invokes the run mode based on Args and TTY detection.
// It is the main entrypoint called by cmd/tau/main.go after signal handling
// is set up.
//
// Dispatch is the only place that performs side effects on stdout/stderr for
// metadata commands (--help, --version). All other paths are deferred to
// handler functions.
func Dispatch(ctx context.Context, args Args) error {
	// Metadata short-circuits. These do not load config or sessions.
	if args.Help {
		printHelp(os.Stdout, args.BinaryVersion)
		return nil
	}
	if args.Version {
		fmt.Fprintf(os.Stdout, "tau %s\n", args.BinaryVersion)
		return nil
	}

	if args.Subcommand != "" {
		return dispatchSubcommand(ctx, args)
	}

	if args.ListModels {
		return listModels(ctx, args)
	}

	// First-run setup wizard. Only runs for session-bearing modes; the
	// metadata paths above (help/version/subcommands/list-models) skip
	// it because they don't load Settings or start a session.
	if err := maybeSetup(ctx, args); err != nil {
		return fmt.Errorf("setup: %w", err)
	}

	// Determine run mode. Explicit flags take precedence; otherwise auto-detect
	// from TTY status.
	mode := args.DeriveMode(os.Stdin, os.Stdout)

	switch mode {
	case "print", "json":
		return runPrint(ctx, args)
	case "rpc":
		return runRPC(ctx, args)
	case "interactive":
		return runInteractive(ctx, args)
	default:
		return fmt.Errorf("unknown mode %q", mode)
	}
}

// isTTY returns true if f is a terminal.
//
// On Unix we use the TCGETS ioctl (the same check used by `golang.org/x/term`
// and `github.com/mattn/go-isatty`); the simpler `ModeCharDevice` check used
// to be common but misclassifies /dev/null as a TTY on Linux. On Windows we
// fall back to the file-mode check, which is good enough for the auto-detect
// fallback.
func isTTY(f *os.File) bool {
	if f == nil {
		return false
	}
	return isTerminalFd(f.Fd())
}

// dispatchSubcommand handles `tau config` and `tau update`.
func dispatchSubcommand(ctx context.Context, args Args) error {
	switch args.Subcommand {
	case "config":
		return runConfigSubcommand(ctx, args)
	case "update":
		return runUpdateSubcommand(ctx, args)
	default:
		return fmt.Errorf("unknown subcommand %q", args.Subcommand)
	}
}

// printHelp emits a usage summary to w. Kept compact; the parser is the source
// of truth for what flags exist.
func printHelp(w io.Writer, version string) {
	fmt.Fprintf(w, `tau %s — native Go coding agent

Usage:
  tau [flags] [prompt]
  tau [flags] -- [prompt...]    (-- ends flag parsing)
  tau <subcommand> [args...]

Subcommands:
  config                        Open settings (use --path to print the path only)
  update                        Self-update

Modes:
  --print                       Non-interactive single-turn (text to stdout)
  --json                        Print mode with JSON output (implies --print)
  --rpc                         JSON-RPC over stdin/stdout (for IDE integrations)
  (default)                     Interactive TUI when stdin&&stdout are TTYs

Flags:
  --model <id>                  Override Settings.DefaultModel
  --provider <name>             Override Settings.DefaultProvider
  --thinking <level>            off|minimal|low|medium|high|xhigh
  --tools <glob>                Filter active tool set
  --cwd <path>                  Override working directory
  --offline                     Disable network-dependent features
  --resume <sessionID>          Resume a session by ID
  --continue                    Resume the most recent session in cwd
  --fork                        Fork the current session into a new one
  --session <path>              Open a specific session file
  --no-session                  Run without persisting state to disk
  --no-setup                    Skip first-run setup wizard
  --list-models                 List available models and exit
  --export <path>               Write the transcript to <path> on exit
  --version                     Print version and exit
  --help, -h                    Print this help and exit

Short flags: -m (model), -p (provider), -c (cwd)
`, version)
}

// runPrint handles `tau --print` / `tau --json`. It wires the session
// via the wire layer and delegates to modes.RunPrint, which performs
// exactly one agentic turn and writes the result to stdout.
func runPrint(ctx context.Context, args Args) error {
	wired, err := wireSession(ctx, args)
	if err != nil {
		return err
	}
	defer func() { _ = wired.Session.Shutdown(ctx) }()

	opts := modes.PrintOptions{
		Prompt:     strings.Join(args.Positional, " "),
		JSON:       args.JSON,
		ExportPath: args.Export,
		Stdout:     os.Stdout,
		Stderr:     os.Stderr,
	}
	if _, err := modes.RunPrint(ctx, opts, wired.Session); err != nil {
		return err
	}
	return nil
}

// runRPC handles `tau --rpc`. It wires the session and delegates to
// modes.RunRPC, which speaks JSON-RPC 2.0 over stdin/stdout until the
// client sends session/shutdown or closes stdin.
func runRPC(ctx context.Context, args Args) error {
	wired, err := wireSession(ctx, args)
	if err != nil {
		return err
	}
	defer func() { _ = wired.Session.Shutdown(ctx) }()

	opts := modes.RPCOptions{
		Stdin:  os.Stdin,
		Stdout: os.Stdout,
		Stderr: os.Stderr,
	}
	return modes.RunRPC(ctx, opts, wired.Session)
}

// runInteractive handles the default TUI mode. It wires the session
// and delegates to modes.RunInteractive, which boots the bubbletea
// program and runs until the user quits.
func runInteractive(ctx context.Context, args Args) error {
	wired, err := wireSession(ctx, args)
	if err != nil {
		return err
	}
	settings := wired.Runtime.Options.Settings
	themeName := ""
	if settings.Theme != nil {
		themeName = *settings.Theme
	}
	opts := modes.InteractiveOptions{
		Stdin:     os.Stdin,
		Stdout:    os.Stdout,
		Stderr:    os.Stderr,
		ThemeName: themeName,
		AgentDir:  wired.Runtime.ConfigDir,
	}
	if err := modes.RunInteractive(ctx, opts, wired.Session); err != nil {
		return err
	}
	return nil
}
