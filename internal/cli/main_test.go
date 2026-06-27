package cli

import (
	"os"
	"testing"
)

// TestMain isolates the cli package tests from the user's real
// ~/.config/tau directory. Without this, tests that land in
// maybeSetup or any path that resolves config.AgentDir() would write
// to the user's home directory on the first run that lacks a real
// settings.json. Individual tests can override with t.Setenv when
// they need a specific config dir layout.
func TestMain(m *testing.M) {
	tmp, err := os.MkdirTemp("", "tau-cli-testconfig")
	if err != nil {
		panic("cli test: cannot create temp dir: " + err.Error())
	}
	defer os.RemoveAll(tmp)
	os.Setenv("TAU_CONFIG_DIR", tmp)
	os.Exit(m.Run())
}
