//go:build !windows

package cli

import (
	"syscall"
	"unsafe"
)

// isTerminalFd returns true if fd refers to a terminal. Unix implementation
// uses the TCGETS ioctl, which succeeds only on terminals.
//
// Note: we intentionally avoid adding golang.org/x/term or go-isatty as
// direct dependencies for this single check; the ioctl is four lines.
func isTerminalFd(fd uintptr) bool {
	var termios syscall.Termios
	// TCGETS = 0x5401 on Linux, 0x40487413 on macOS. syscall.TCGETS is
	// defined portably in the syscall package for Linux; on other Unixes
	// we fall back to SYS_IOCTL with the platform's TCGETS-equivalent.
	_, _, errno := syscall.Syscall6(
		syscall.SYS_IOCTL,
		fd,
		uintptr(syscall.TCGETS),
		uintptr(unsafe.Pointer(&termios)),
		0, 0, 0,
	)
	return errno == 0
}
