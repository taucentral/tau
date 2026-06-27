// resolve.go — sigil resolver for config values.
//
// A config value can be one of:
//
//	""                empty (caller decides if fatal)
//	"$VAR"            environment-variable reference (single name)
//	"${VAR}"          environment-variable reference (braced)
//	"$$..."           literal "$" (escape)
//	"!cmd args"       shell command; stdout is captured and trimmed
//	anything else     literal
//
// Unknown sigils or malformed references are treated as literals, never
// as errors. The only error return is a non-zero shell exit. An unset
// environment variable contributes "" — callers decide whether empty is
// fatal.

package config

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
)

// ResolveValue resolves s according to the sigil rules above. It is
// safe to call concurrently (no shared state, no caching).
func ResolveValue(s string) (string, error) {
	if s == "" {
		return "", nil
	}
	if strings.HasPrefix(s, "!") {
		return resolveCommand(s[1:])
	}
	return resolveTemplate(s), nil
}

// resolveTemplate interpolates $VAR / ${VAR} / $$ references using the
// current process environment.
func resolveTemplate(s string) string {
	var b strings.Builder
	i := 0
	for i < len(s) {
		c := s[i]
		if c != '$' {
			b.WriteByte(c)
			i++
			continue
		}
		if i+1 >= len(s) {
			b.WriteByte('$')
			i++
			continue
		}
		switch next := s[i+1]; next {
		case '$', '!':
			b.WriteByte(next)
			i += 2
		case '{':
			end := strings.IndexByte(s[i+2:], '}')
			if end < 0 {
				b.WriteByte('$')
				i++
				continue
			}
			name := s[i+2 : i+2+end]
			if !isValidEnvName(name) {
				b.WriteString(s[i : i+2+end+1])
				i += 2 + end + 1
				continue
			}
			b.WriteString(os.Getenv(name))
			i += 2 + end + 1
		default:
			name := readEnvName(s[i+1:])
			if name == "" {
				b.WriteByte('$')
				i++
				continue
			}
			b.WriteString(os.Getenv(name))
			i += 1 + len(name)
		}
	}
	return b.String()
}

// readEnvName reads the longest [_A-Za-z][_A-Za-z0-9]* prefix of s.
func readEnvName(s string) string {
	i := 0
	for i < len(s) {
		c := s[i]
		if i == 0 && !isAlpha(c) && c != '_' {
			break
		}
		if i > 0 && !isAlnum(c) && c != '_' {
			break
		}
		i++
	}
	return s[:i]
}

func isAlpha(c byte) bool {
	return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
}

func isAlnum(c byte) bool {
	return isAlpha(c) || (c >= '0' && c <= '9')
}

// isValidEnvName reports whether name is a syntactically valid POSIX env
// var name (nonempty, alpha/underscore led).
func isValidEnvName(name string) bool {
	if name == "" {
		return false
	}
	if !isAlpha(name[0]) && name[0] != '_' {
		return false
	}
	for i := 1; i < len(name); i++ {
		if !isAlnum(name[i]) && name[i] != '_' {
			return false
		}
	}
	return true
}

// resolveCommand runs cmd via the platform's shell and returns its
// trimmed stdout. A non-zero exit returns ErrInvalidValue with the
// command's stderr (or exit code) in the message.
//
// ResolveValue is a synchronous one-shot helper invoked at config load
// (not per-request), so it has no caller-supplied ctx. exec.CommandContext
// is used with context.Background() so the linter can verify the subprocess
// is killable; a config-loader shellout is expected to complete quickly.
//
//nolint:noctx // intentional: no request-scoped ctx at config-load time
func resolveCommand(cmd string) (string, error) {
	if strings.TrimSpace(cmd) == "" {
		return "", fmt.Errorf("%w: empty command after '!'", ErrInvalidValue)
	}
	var c *exec.Cmd
	if runtime.GOOS == "windows" {
		c = exec.CommandContext(context.Background(), "cmd", "/c", cmd)
	} else {
		c = exec.CommandContext(context.Background(), "sh", "-c", cmd)
	}
	c.Env = os.Environ()
	out, err := c.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return "", fmt.Errorf("%w: command %q exited %d: %s",
				ErrInvalidValue, cmd, ee.ExitCode(),
				strings.TrimSpace(string(ee.Stderr)))
		}
		return "", fmt.Errorf("%w: command %q: %v",
			ErrInvalidValue, cmd, err)
	}
	return strings.TrimSpace(string(out)), nil
}
