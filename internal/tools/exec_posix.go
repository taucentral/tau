//go:build !windows

package tools

import (
	"os"
	"os/exec"
	"syscall"
)

// sysProcAttrForKill configures the child to run in its own process group
// so the entire tree can be killed on timeout/cancellation. POSIX-only.
func sysProcAttrForKill() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{Setpgid: true}
}

// killProcessGroup sends SIGKILL to the child's process group. Best-effort;
// errors are swallowed because the child may already be dead.
func killProcessGroup(cmd *exec.Cmd) error {
	if cmd == nil || cmd.Process == nil {
		return nil
	}
	// Negative PID kills the entire process group (child + descendants).
	pgid, err := syscall.Getpgid(cmd.Process.Pid)
	if err == nil {
		_ = syscall.Kill(-pgid, syscall.SIGKILL)
		return nil
	}
	// Fallback: kill just the child.
	return cmd.Process.Kill()
}

// signalNameFromState extracts the signal name (e.g. "KILL", "TERM") from
// a *os.ProcessState, or "" if the process exited normally.
func signalNameFromState(state *os.ProcessState) string {
	if state == nil {
		return ""
	}
	status, ok := state.Sys().(syscall.WaitStatus)
	if !ok {
		return ""
	}
	if !status.Signaled() {
		return ""
	}
	return status.Signal().String()
}
