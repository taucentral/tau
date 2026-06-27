// templates.go — load named prompt templates from disk.
//
// Templates live in either of two directories and are plain markdown
// (or text) files keyed by name:
//
//   - Global: <ConfigDir>/agent/prompts/<name>.md
//   - Project: <ProjectDir>/.tau/prompts/<name>.md
//
// Project shadows global: if both exist, the project copy wins. The
// loader does NOT merge; callers wanting both should Load them
// separately under different names.
//
// Template names are validated: only [a-z0-9._-] is allowed, and the
// name cannot start with a dot or contain path separators. This keeps
// the lookup filesystem-safe without a complicated allowlist.

package prompts

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ErrTemplateNotFound is returned when neither the project nor the
// global directory has a template with the given name.
var ErrTemplateNotFound = errors.New("prompts: template not found")

// ErrInvalidTemplateName is returned when the name contains characters
// outside the allowed set or would resolve outside the prompts
// directory.
var ErrInvalidTemplateName = errors.New("prompts: invalid template name")

// Loader resolves named prompt templates against the global and
// project template directories.
type Loader struct {
	// ConfigDir is the global config root (typically ~/.config/tau).
	// Templates are sought at <ConfigDir>/agent/prompts/. Empty
	// disables global lookups.
	ConfigDir string
	// ProjectDir is the project cwd. Templates are sought at
	// <ProjectDir>/.tau/prompts/. Empty disables project lookups.
	ProjectDir string
}

// NewLoader returns a Loader using the standard directories.
func NewLoader(configDir, projectDir string) *Loader {
	return &Loader{ConfigDir: configDir, ProjectDir: projectDir}
}

// Load returns the content of the named template. Project shadows
// global. Returns ErrTemplateNotFound when neither exists, or
// ErrInvalidTemplateName when the name fails validation.
//
// The content is returned as-is; the caller applies any substitutions
// (template variables, etc.) — the prompts package does not interpret
// template syntax.
func (l *Loader) Load(name string) (string, error) {
	if err := validateTemplateName(name); err != nil {
		return "", err
	}
	// Project first (shadows global).
	if l.ProjectDir != "" {
		if body, err := readTemplate(filepath.Join(l.ProjectDir, ".tau", "prompts", name+".md")); err == nil {
			return body, nil
		} else if !errors.Is(err, os.ErrNotExist) {
			return "", err
		}
	}
	// Global fallback.
	if l.ConfigDir != "" {
		if body, err := readTemplate(filepath.Join(l.ConfigDir, "agent", "prompts", name+".md")); err == nil {
			return body, nil
		} else if !errors.Is(err, os.ErrNotExist) {
			return "", err
		}
	}
	return "", ErrTemplateNotFound
}

// validateTemplateName enforces the naming rule: only [a-z0-9._-], no
// leading dot, no path separators, length 1..64. The rule is strict
// enough that filepath.Join cannot escape the prompts directory even
// on case-insensitive filesystems.
func validateTemplateName(name string) error {
	if len(name) == 0 || len(name) > 64 {
		return ErrInvalidTemplateName
	}
	if strings.HasPrefix(name, ".") {
		return ErrInvalidTemplateName
	}
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z':
		case r >= '0' && r <= '9':
		case r == '.' || r == '_' || r == '-':
		default:
			return ErrInvalidTemplateName
		}
	}
	return nil
}

// readTemplate reads one file. Returns os.ErrNotExist-wrapped error if
// missing, so callers can errors.Is against it.
func readTemplate(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// Available returns the names of available templates (without the .md
// suffix), with project names shadowing global ones. Sorted
// lexicographically. Returns nil if neither directory exists.
//
// Available is intended for discovery UIs (e.g., a /prompts slash
// command). It does NOT cache; callers should call once at startup.
func (l *Loader) Available() ([]string, error) {
	seen := map[string]bool{}
	for _, dir := range []string{
		filepath.Join(l.ConfigDir, "agent", "prompts"),
		filepath.Join(l.ProjectDir, ".tau", "prompts"),
	} {
		if dir == "" || dir == string(filepath.Separator) {
			continue
		}
		entries, err := os.ReadDir(dir)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			return nil, fmt.Errorf("prompts: list %s: %w", dir, err)
		}
		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}
			name := entry.Name()
			if !strings.HasSuffix(name, ".md") {
				continue
			}
			stem := strings.TrimSuffix(name, ".md")
			if validateTemplateName(stem) != nil {
				continue
			}
			seen[stem] = true
		}
	}
	if len(seen) == 0 {
		return nil, nil
	}
	out := make([]string, 0, len(seen))
	for k := range seen {
		out = append(out, k)
	}
	// Sort for stable output across filesystems.
	for i := 0; i < len(out); i++ {
		for j := i + 1; j < len(out); j++ {
			if out[j] < out[i] {
				out[i], out[j] = out[j], out[i]
			}
		}
	}
	return out, nil
}
