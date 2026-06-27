// startup.go — first-run setup wizard and project trust prompt.
//
// Dispatch calls MaybeSetup before invoking any session-bearing run mode
// (print / rpc / interactive). The wizard:
//
//   - is a no-op when AgentDir already exists with a settings.json
//     (i.e. this is not a first run).
//   - is skipped silently when --no-setup is set.
//   - writes a default settings.json (no prompts) when stdin or stdout
//     is not a TTY, so headless / CI installs come up configured.
//   - prompts for theme / default model / analytics opt-in when both
//     stdin and stdout are TTYs.
//
// The wizard never blocks on input when the user pipes anything: a non-TTY
// stdin auto-skips even if --no-setup wasn't passed. This matches the cli
// spec scenario "First run in CI".
//
// All prompts use a bufio.Reader directly on os.Stdin so the wizard works
// with any line-buffered terminal; no bubbletea dependency at this layer.

package cli

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/coevin/tau/internal/config"
)

// setupAnswers captures the user's responses to the wizard prompts. Empty
// fields mean "use the default" — the writer maps each empty field to the
// appropriate Settings default.
type setupAnswers struct {
	Theme           string // "dark" | "light" | "" (default)
	DefaultModel    string // model id or "" (skip)
	EnableAnalytics bool
}

// maybeSetup is the entry point Dispatch calls before invoking a session-
// bearing run mode. It returns nil on success and a non-nil error only
// when writing the default config failed — the wizard never fails just
// because the user pressed Ctrl+C or typed garbage (those cases fall back
// to defaults silently).
//
// The function is environment-aware via TAU_CONFIG_DIR so tests can point
// at a temp dir.
func maybeSetup(_ context.Context, args Args) error {
	if args.NoSetup {
		return nil
	}
	agentDir, err := config.AgentDir()
	if err != nil {
		// We can't even resolve where the config would live; bail out
		// silently so a downstream run-mode handler surfaces the real
		// error when it actually needs config.
		return nil //nolint:nilerr // intentional silent bail; downstream surfaces the error
	}
	settingsPath := filepath.Join(agentDir, "settings.json")
	if fileExists(settingsPath) {
		// Returning nil here is the "not first run" path: settings.json
		// already exists, so the wizard would be redundant.
		return nil
	}

	// First run. Branch on TTY status.
	stdinTTY := isTTY(os.Stdin)
	stdoutTTY := isTTY(os.Stdout)

	if !stdinTTY || !stdoutTTY {
		return writeDefaultConfig(agentDir)
	}

	answers, err := promptSetup(os.Stdin, os.Stdout)
	if err != nil {
		// Treat prompt errors (e.g. Ctrl+C, malformed input) as "write
		// defaults" rather than fatal — the user can re-run setup later
		// by deleting settings.json.
		answers = setupAnswers{}
	}
	return writeSetupAnswers(agentDir, answers)
}

// fileExists reports whether path exists (any type — file or directory).
// Used by maybeSetup to detect first run.
func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// promptSetup drives the interactive wizard. Reads from stdin and writes
// prompts to stdout. Empty / unrecognized answers fall back to defaults
// so a user can press Enter through the wizard.
func promptSetup(in io.Reader, out io.Writer) (setupAnswers, error) {
	reader := bufio.NewReader(in)
	var answers setupAnswers

	// Theme.
	fmt.Fprintln(out, "\nWelcome to tau. Let's set up a few defaults.")
	fmt.Fprintln(out, "You can re-run setup by deleting ~/.config/tau/agent/settings.json.")
	fmt.Fprint(out, "\nTheme [dark/light] (default: dark): ")
	line, err := reader.ReadString('\n')
	if err != nil && !errors.Is(err, io.EOF) {
		return answers, err
	}
	answers.Theme = normalizeSetupChoice(line, "dark", "light", "dark")

	// Default model.
	fmt.Fprint(out, "Default model id (Enter to skip, e.g. claude-opus-4-5 or gpt-4o): ")
	line, err = reader.ReadString('\n')
	if err != nil && !errors.Is(err, io.EOF) {
		return answers, err
	}
	answers.DefaultModel = strings.TrimSpace(line)

	// Analytics opt-in (default off per spec).
	fmt.Fprint(out, "Enable anonymous analytics? [y/N] (default: N): ")
	line, err = reader.ReadString('\n')
	if err != nil && !errors.Is(err, io.EOF) {
		return answers, err
	}
	answers.EnableAnalytics = normalizeYesNo(line, false)

	fmt.Fprintln(out, "Setup complete. Configuration written to ~/.config/tau/agent/settings.json")
	return answers, nil
}

// normalizeSetupChoice lowercases the line and matches it against the
// allowed set. Returns the default if the input is empty or unrecognized.
func normalizeSetupChoice(line, opt1, opt2, defaultChoice string) string {
	choice := strings.ToLower(strings.TrimSpace(line))
	switch choice {
	case opt1:
		return opt1
	case opt2:
		return opt2
	case "":
		return defaultChoice
	}
	return defaultChoice
}

// normalizeYesNo returns true if the line starts with y or Y. Returns
// defaultVal for empty/unrecognized input.
func normalizeYesNo(line string, defaultVal bool) bool {
	choice := strings.ToLower(strings.TrimSpace(line))
	if choice == "" {
		return defaultVal
	}
	return strings.HasPrefix(choice, "y")
}

// writeDefaultConfig writes a zero-answer settings.json. Called for the
// non-TTY (CI / piped) path so subsequent runs see "configured" state.
func writeDefaultConfig(agentDir string) error {
	return writeSetupAnswers(agentDir, setupAnswers{})
}

// writeSetupAnswers turns answers into Settings mutations and writes
// them to <agentDir>/settings.json atomically. Empty fields fall back
// to the package defaults; we don't write literally-zero values.
func writeSetupAnswers(agentDir string, answers setupAnswers) error {
	if err := config.MkdirAll(agentDir); err != nil {
		return fmt.Errorf("setup: mkdir agent dir: %w", err)
	}
	storage, err := config.NewFileSettingsStorage(agentDir, agentDir, true)
	if err != nil {
		return fmt.Errorf("setup: open settings storage: %w", err)
	}
	defer storage.Close()
	ctx := context.Background()
	return storage.Save(ctx, config.ScopeGlobal, func(current config.Settings) config.Settings {
		s := current
		if s.Theme == nil && answers.Theme != "" {
			t := answers.Theme
			s.Theme = &t
		}
		if s.DefaultModel == nil && answers.DefaultModel != "" {
			m := answers.DefaultModel
			s.DefaultModel = &m
		}
		if s.EnableAnalytics == nil {
			v := answers.EnableAnalytics
			s.EnableAnalytics = &v
		}
		return s
	})
}
