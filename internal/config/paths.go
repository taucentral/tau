// paths.go — config directory resolution for tau.
//
// Resolution order matches the spec at
// openspec/changes/initial/specs/config/spec.md (Requirement: Config
// directory resolution):
//
//   ConfigDir: TAU_CONFIG_DIR → XDG_CONFIG_HOME/tau (POSIX) → os.UserConfigDir()/tau
//   AgentDir:  TAU_AGENT_DIR → <ConfigDir>/agent
//   SessionsDir(cwd): <AgentDir>/sessions/<EncodeCwd(cwd)>
//   PluginsDir: [<ConfigDir>/plugins, <cwd>/plugins]

package config

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// appName is the directory name under the user's config dir.
const appName = "tau"

// DirMode is the directory mode used for all config tree writes. Tighter
// than the 0755 default; matches pi's mode 0700 invariant.
const DirMode os.FileMode = 0o700

// FileMode is the mode used for secret-bearing config files (auth.json,
// trust.json). Settings.json and models.json use this too for consistency.
const FileMode os.FileMode = 0o600

// ConfigDir returns the root of tau's config tree.
//
// Resolution:
//  1. $TAU_CONFIG_DIR if set (absolute or relative to cwd; we trust the user).
//  2. $XDG_CONFIG_HOME/tau on POSIX when XDG_CONFIG_HOME is set.
//  3. os.UserConfigDir()/tau — cross-platform stdlib fallback that honors
//     %APPDATA% on Windows, ~/Library/Application Support on macOS, and
//     $XDG_CONFIG_HOME or ~/.config on Linux.
//
//nolint:revive // name is intentional: "config.ConfigDir" reads as "the config dir".
func ConfigDir() (string, error) {
	if v := os.Getenv("TAU_CONFIG_DIR"); v != "" {
		return v, nil
	}
	if runtime.GOOS != "windows" {
		if xdg := os.Getenv("XDG_CONFIG_HOME"); xdg != "" {
			if !filepath.IsAbs(xdg) {
				return "", errors.New("XDG_CONFIG_HOME is not absolute")
			}
			return filepath.Join(xdg, appName), nil
		}
	}
	base, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(base, appName), nil
}

// AgentDir returns the agent subdirectory. Resolution:
//
//	$TAU_AGENT_DIR if set → <ConfigDir>/agent
func AgentDir() (string, error) {
	if v := os.Getenv("TAU_AGENT_DIR"); v != "" {
		return v, nil
	}
	c, err := ConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(c, "agent"), nil
}

// SessionsDir returns the per-cwd session storage directory.
// The cwd is encoded via EncodeCwd so multiple cwds share the agent tree
// without colliding.
func SessionsDir(cwd string) (string, error) {
	if cwd == "" {
		return "", errors.New("cwd is required")
	}
	a, err := AgentDir()
	if err != nil {
		return "", err
	}
	abs, err := filepath.Abs(cwd)
	if err != nil {
		return "", err
	}
	return filepath.Join(a, "sessions", EncodeCwd(abs)), nil
}

// PluginsDir returns the ordered list of plugin scan roots. Project-scoped
// plugins shadow global ones when both define a plugin with the same name.
// The project directory is included even for untrusted cwds; callers must
// gate project plugin loading on trust themselves.
func PluginsDir(cwd string) ([]string, error) {
	c, err := ConfigDir()
	if err != nil {
		return nil, err
	}
	global := filepath.Join(c, "plugins")
	if cwd == "" {
		return []string{global}, nil
	}
	abs, err := filepath.Abs(cwd)
	if err != nil {
		return nil, err
	}
	project := filepath.Join(abs, ".tau", "plugins")
	return []string{global, project}, nil
}

// EncodeCwd encodes an absolute path as a single filesystem-safe directory
// name. "/" and "\" are replaced with "-", and the result always has
// exactly one leading "-" so the encoded form is distinguishable from a
// literal name and so two distinct paths never collide.
//
// Examples:
//
//	/home/user/project  ->  -home-user-project
//	C:\Users\Alice\proj  ->  -C-Users-Alice-proj
//	/                    ->  -
//
// The encoding is one-to-one within a single OS's path shape: every
// character that is a path separator on either platform is replaced, and
// the leading dash (which a POSIX absolute path contributes naturally) is
// normalized so Windows drive-prefixed paths get the same shape.
func EncodeCwd(cwd string) string {
	r := strings.NewReplacer("/", "-", "\\", "-")
	encoded := r.Replace(cwd)
	// Normalize: strip one natural leading dash (POSIX absolute paths),
	// then add exactly one. Paths that didn't start with a separator still
	// get the leading dash.
	encoded = strings.TrimPrefix(encoded, "-")
	return "-" + encoded
}

// MkdirAll creates path and any missing parents with mode 0700. It is a
// thin wrapper around os.MkdirAll that enforces the package's directory
// mode invariant.
func MkdirAll(path string) error {
	if err := os.MkdirAll(path, DirMode); err != nil {
		if errors.Is(err, os.ErrPermission) {
			return ErrPermission
		}
		return err
	}
	return nil
}
