//go:build windows

package cli

import "os"

// isTerminalFd returns true if fd refers to a console. Windows implementation
// uses the file-mode check; a more precise version would call
// GetConsoleMode, but the auto-detect fallback tolerates false positives
// here.
func isTerminalFd(fd uintptr) bool {
	// Best-effort on Windows; reuse the file-stat path.
	f := os.NewFile(fd, "")
	if f == nil {
		return false
	}
	stat, err := f.Stat()
	if err != nil {
		return false
	}
	return (stat.Mode() & os.ModeCharDevice) != 0
}
