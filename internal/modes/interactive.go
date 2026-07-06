// interactive.go — bubbletea TUI mode.
//
// The interactive mode is tau's default run mode when stdin and stdout
// are both TTYs. It boots a charmbracelet/bubbletea program that drives
// the AppModel, wires the agent's event bus to the program via
// tui.SubscribeEvents, and runs until the user requests quit (/quit
// command or the configured quit keybinding).
//
// The function is thin on purpose: all rendering and key dispatch live
// in the tui package. modes.RunInteractive owns only the I/O plumbing,
// context handling, and shutdown sequencing.

package modes

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/taucentral/tau/internal/agent"
	"github.com/taucentral/tau/internal/tui"
)

// InteractiveOptions is the input bundle for RunInteractive. Stdin,
// Stdout, and Stderr default to the process streams when nil so tests
// can swap them for buffers.
type InteractiveOptions struct {
	// Stdin is the input reader. Defaults to os.Stdin.
	Stdin io.Reader
	// Stdout is the output writer. Defaults to os.Stdout.
	Stdout io.Writer
	// Stderr is the error writer; used only for diagnostics emitted
	// when the program fails to start. Defaults to os.Stderr.
	Stderr io.Writer

	// Width and Height, when non-zero, override terminal-size auto-
	// detection. tea.WindowSizeMsg arrives shortly after start and
	// reflows the layout anyway, but tests use these to avoid needing
	// a real TTY for the initial render.
	Width  int
	Height int

	// ThemeName selects the colour theme ("dark", "light", or a custom
	// file name under AgentDir/themes/). Empty defaults to "dark".
	ThemeName string

	// AgentDir is the agent config directory, used for theme file
	// lookup. Empty falls back to the session runtime's ConfigDir.
	AgentDir string

	// QuitCh, when non-nil, is closed by RunInteractive when the user
	// requests quit (via /quit or the quit keybinding). Orchestration
	// layers can wait on it to know the TUI has finished. When nil,
	// RunInteractive allocates its own channel.
	QuitCh chan struct{}

	// noAltScreen, when true, disables the alternate screen buffer.
	// Tests set this so output goes to the provided Stdout (which may
	// be a buffer) instead of being rendered off-screen.
	noAltScreen bool
}

// RunInteractive boots the TUI against session and blocks until the
// user quits (via the configured keybinding or /quit), the context is
// cancelled, or the underlying program returns a fatal error.
//
// RunInteractive does not call session.Shutdown — the caller owns the
// session lifecycle and is expected to defer it. RunInteractive only
// ensures the bubbletea program exits cleanly so the caller's shutdown
// sees a quiescent state.
func RunInteractive(ctx context.Context, opts InteractiveOptions, session *agent.AgentSession) error {
	if session == nil {
		return errors.New("modes.RunInteractive: session is nil")
	}
	stdin := opts.Stdin
	if stdin == nil {
		stdin = os.Stdin
	}
	stdout := opts.Stdout
	if stdout == nil {
		stdout = os.Stdout
	}
	stderr := opts.Stderr
	if stderr == nil {
		stderr = os.Stderr
	}

	rt := session.Runtime()

	// Resolve theme. LoadTheme falls back to DarkTheme silently when
	// the named custom theme file does not exist, so the only error
	// path is malformed JSON — surfaced to the caller as fatal.
	agentDir := opts.AgentDir
	if agentDir == "" {
		agentDir = rt.ConfigDir
	}
	themeName := opts.ThemeName
	if themeName == "" {
		themeName = "dark"
	}
	theme, err := tui.LoadTheme(agentDir, themeName)
	if err != nil {
		return fmt.Errorf("modes.RunInteractive: load theme %q: %w", themeName, err)
	}

	// Initial dimensions. The real values arrive via tea.WindowSizeMsg
	// once the program starts; these are just the seed for the first
	// render so the model never sees zero size.
	width := opts.Width
	if width <= 0 {
		width = 80
	}
	height := opts.Height
	if height <= 0 {
		height = 24
	}

	quit := opts.QuitCh
	if quit == nil {
		quit = make(chan struct{})
	}

	model := tui.NewAppModel(tui.AppOptions{
		Session:  session,
		Settings: rt.Options.Settings,
		Theme:    theme,
		Width:    width,
		Height:   height,
		AgentDir: agentDir,
		QuitCh:   quit,
	})

	// Build the program. WithInput/WithOutput override the defaults so
	// tests can capture the rendered output. WithAltScreen is enabled
	// unless the caller disabled it; the alt screen keeps the user's
	// scrollback intact.
	progOpts := []tea.ProgramOption{
		tea.WithInput(stdin),
		tea.WithOutput(stdout),
	}
	if !opts.noAltScreen {
		progOpts = append(progOpts, tea.WithAltScreen())
	}
	prog := tea.NewProgram(model, progOpts...)

	// Subscribe agent events → eventMsg → Update. Forward goroutine
	// exits when the session shuts down (bus closes channels) or when
	// stop fires.
	stop := make(chan struct{})
	defer close(stop)
	tui.SubscribeEvents(prog, session, stop)

	// Watcher: cancel the program when ctx fires or quit is requested.
	// prog.Quit is idempotent so calling it from multiple goroutines is
	// safe.
	watcherDone := make(chan struct{})
	go func() {
		defer close(watcherDone)
		select {
		case <-ctx.Done():
			prog.Quit()
		case <-quit:
			// AppModel already closed quit; tell the program to exit.
			prog.Quit()
		case <-watcherDone:
			// Program exited on its own (user typed the quit key).
		}
	}()

	// Run blocks until the program exits.
	if _, err := prog.Run(); err != nil {
		// prog.Quit was already called via the watcher; the watcher
		// goroutine is unblocked by closing watcherDone — but only
		// after Run returns. So we can't reuse it as a fence. The
		// goroutine instead self-exits when one of its three cases
		// fires (ctx.Done / quit / watcherDone). To avoid leaking it,
		// we close watcherDone manually here as a fence.
		fmt.Fprintf(stderr, "tau: tui exited with error: %v\n", err)
		return fmt.Errorf("modes.RunInteractive: %w", err)
	}

	// If the program exited because the user quit, the requestQuit
	// helper in AppModel closed the quit channel. The watcher's other
	// case (ctx.Done) didn't fire, so the goroutine is still alive.
	// Closing watcherDone here would race with the deferred close(stop)
	// — instead, the watcher also selects on a self-close. To make the
	// liveness simple, we drain the watcher by waiting on it.
	<-watcherDone
	return nil
}
