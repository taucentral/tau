package cli

import (
	"bytes"
	"context"
	"errors"
	"os"
	"strings"
	"testing"
)

// rerunCtx is a cancelled-neutral context for tests that don't exercise
// cancellation.
var rerunCtx = context.Background()

func TestDispatch_HelpShortCircuit(t *testing.T) {
	// Capture stdout via swap; Dispatch writes to os.Stdout directly.
	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	os.Stdout = w
	defer func() { os.Stdout = old }()

	err = Dispatch(rerunCtx, Args{Help: true, BinaryVersion: "test-1"})
	w.Close()
	if err != nil {
		t.Fatalf("Dispatch(Help) returned error: %v", err)
	}
	var buf bytes.Buffer
	if _, copyErr := buf.ReadFrom(r); copyErr != nil {
		t.Fatalf("read pipe: %v", copyErr)
	}
	if !strings.Contains(buf.String(), "tau test-1") {
		t.Errorf("help output missing version string; got:\n%s", buf.String())
	}
}

func TestDispatch_VersionShortCircuit(t *testing.T) {
	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	os.Stdout = w
	defer func() { os.Stdout = old }()

	err = Dispatch(rerunCtx, Args{Version: true, BinaryVersion: "v9.9.9"})
	w.Close()
	if err != nil {
		t.Fatalf("Dispatch(Version) returned error: %v", err)
	}
	var buf bytes.Buffer
	if _, copyErr := buf.ReadFrom(r); copyErr != nil {
		t.Fatalf("read pipe: %v", copyErr)
	}
	if !strings.Contains(buf.String(), "tau v9.9.9") {
		t.Errorf("version output missing; got:\n%s", buf.String())
	}
}

func TestDispatch_SubcommandConfig_Succeeds(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("TAU_CONFIG_DIR", dir)
	// Capture stdout via temp file to avoid pipe races with the tabwriter.
	tmp, err := os.CreateTemp(dir, "out")
	if err != nil {
		t.Fatalf("temp: %v", err)
	}
	old := os.Stdout
	os.Stdout = tmp
	err = Dispatch(rerunCtx, Args{Subcommand: "config"})
	os.Stdout = old
	tmp.Close()
	if err != nil {
		t.Errorf("config subcommand: err = %v, want nil", err)
	}
}

func TestDispatch_SubcommandUpdate_Succeeds(t *testing.T) {
	// Update writes to stderr; no error is expected (v1 stub).
	old := os.Stderr
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	os.Stderr = w
	err = Dispatch(rerunCtx, Args{Subcommand: "update"})
	_ = w.Close()
	os.Stderr = old
	if err != nil {
		t.Errorf("update subcommand: err = %v, want nil", err)
	}
	var buf bytes.Buffer
	_, _ = buf.ReadFrom(r)
	if !strings.Contains(buf.String(), "package manager") {
		t.Errorf("update output missing expected message; got:\n%s", buf.String())
	}
}

func TestDispatch_SubcommandUnknown(t *testing.T) {
	err := Dispatch(rerunCtx, Args{Subcommand: "bogus"})
	if err == nil || errors.Is(err, ErrNotImplemented) {
		t.Errorf("unknown subcommand: err = %v, want a non-ErrNotImplemented error", err)
	}
}

func TestDispatch_ListModels_Succeeds(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("TAU_CONFIG_DIR", dir)
	tmp, err := os.CreateTemp(dir, "out")
	if err != nil {
		t.Fatalf("temp: %v", err)
	}
	old := os.Stdout
	os.Stdout = tmp
	err = Dispatch(rerunCtx, Args{ListModels: true})
	os.Stdout = old
	tmp.Close()
	if err != nil {
		t.Errorf("list-models: err = %v, want nil", err)
	}
}

// TestDispatch_RPCMode_NoModel_Errors verifies the wire layer surfaces
// ErrNoModel when --rpc is requested with no model configured. The
// protocol-level RPC round-trip is exercised in internal/modes tests
// (where the stdin/stdout pipes can be injected cleanly without racing
// on the process's real os.Stdin / os.Stdout globals).
func TestDispatch_RPCMode_NoModel_Errors(t *testing.T) {
	dir := t.TempDir()
	err := Dispatch(rerunCtx, Args{
		RPC: true,
		Cwd: dir,
	})
	if !errors.Is(err, ErrNoModel) {
		t.Errorf("Dispatch(rpc, no model): err = %v, want ErrNoModel", err)
	}
}

// TestDispatch_PrintMode_FauxModel exercises the full print-mode path:
// wire layer + modes.RunPrint + faux provider. Sets the prompt via
// Positional and the faux model via Model so the wire layer can build
// a session without real credentials.
func TestDispatch_PrintMode_FauxModel(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("TAU_FAUX_SCRIPT", "dispatch reply")
	tmp, err := os.CreateTemp(dir, "out")
	if err != nil {
		t.Fatalf("temp: %v", err)
	}
	old := os.Stdout
	os.Stdout = tmp
	err = Dispatch(rerunCtx, Args{
		Print:      true,
		Model:      "faux",
		Cwd:        dir,
		Positional: []string{"hi"},
	})
	os.Stdout = old
	tmp.Close()
	if err != nil {
		t.Fatalf("Dispatch(print): %v", err)
	}
	b, readErr := os.ReadFile(tmp.Name())
	if readErr != nil {
		t.Fatalf("read tmp: %v", readErr)
	}
	if !strings.Contains(string(b), "dispatch reply") {
		t.Errorf("stdout = %q, want substring 'dispatch reply'", string(b))
	}
}

// TestDispatch_PrintMode_NoModel_Errors verifies the wire layer surfaces
// ErrNoModel when no model is configured.
func TestDispatch_PrintMode_NoModel_Errors(t *testing.T) {
	dir := t.TempDir()
	err := Dispatch(rerunCtx, Args{
		Print:      true,
		Cwd:        dir,
		Positional: []string{"hi"},
	})
	if !errors.Is(err, ErrNoModel) {
		t.Errorf("Dispatch(print, no model): err = %v, want ErrNoModel", err)
	}
}

// TestDispatch_InteractiveMode_WiresSession verifies that explicit
// Mode="interactive" enters runInteractive, which calls wireSession.
// With no model configured, wireSession returns ErrNoModel — proving
// the dispatch path reaches the wire layer instead of an unimplemented
// stub.
func TestDispatch_InteractiveMode_WiresSession(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("TAU_CONFIG_DIR", dir)
	err := Dispatch(rerunCtx, Args{Mode: "interactive", Cwd: dir})
	if err == nil {
		t.Fatal("interactive mode with no model: err = nil, want ErrNoModel")
	}
	if errors.Is(err, ErrNotImplemented) {
		t.Errorf("interactive mode: err = %v, want it to NOT be ErrNotImplemented", err)
	}
}
