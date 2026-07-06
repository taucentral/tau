// Package main is the tau binary entrypoint.
//
// tau is a native Go coding agent that mirrors the pi TypeScript coding agent.
// The binary is built with CGO disabled (see Makefile, CGO_ENABLED=0) so the
// produced binary is statically linked with zero libc dependency on Linux.
//
// tau dispatches to one of three run modes based on flags and TTY detection:
//
//   - interactive (bubbletea TUI) when stdin/stdout are TTYs
//   - print      (single-turn, text or JSON) when --print/--json or non-TTY
//   - rpc        (JSON-RPC over stdin/stdout) when --rpc
//
// Process lifecycle:
//
//  1. Parse argv via internal/cli.
//  2. Install signal handlers for SIGINT and SIGTERM.
//  3. Dispatch to the selected run mode.
//  4. Context cancellation drives in-flight tool abort and plugin shutdown.
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/taucentral/tau/internal/cli"
)

// version is the binary version. It is overridden at build time via -ldflags.
var version = "0.0.0-dev"

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "tau:", err)
		os.Exit(1)
	}
}

// run parses argv, establishes cancellation, and dispatches to the selected
// mode. It is extracted from main() for testability.
func run(argv []string) error {
	args, err := cli.ParseArgs(argv)
	if err != nil {
		return err
	}
	args.BinaryVersion = version

	// Establish a root context cancelled by SIGINT/SIGTERM.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	return cli.Dispatch(ctx, args)
}
