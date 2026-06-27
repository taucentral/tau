//go:build windows

package tools

import (
	"os"
	"os/exec"
	"syscall"
)

// sysProcAttrForKill configures the child to run in its own process group
// via CREATE_NEW_PROCESS_GROUP so it can be killed as a unit. Windows-only.
func sysProcAttrForKill() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP}
}

// killProcessGroup kills the child process. Windows doesn't have native
// process-tree kill in user mode; we rely on cmd.Process.Kill() for the
// direct child. Detached descendants may survive.
func killProcessGroup(cmd *exec.Cmd) error {
	if cmd == nil || cmd.Process == nil {
		return nil
	}
	return cmd.Process.Kill()
}

// signalNameFromState extracts the signal/exit reason from a Windows
// process state. Windows uses exit codes, not signals, so this returns ""
// except in the case of a forced kill (where the exit code is non-zero).
func signalNameFromState(state *os.ProcessState) string {
	if state == nil {
		return ""
	}
	if !state.Exited() {
		return "KILL"
	}
	return ""
}
