// Package config implements tau's configuration, paths, auth, trust, and
// model-registry layers.
//
// The package is intentionally UI-free: it exposes pure data types, file
// storage backends, and typed sentinel errors. Prompting for trust decisions
// and other interactive flows lives in the cli layer.
//
// All file writes use the temp-file-then-rename pattern for crash atomicity
// and gofrs/flock for cross-process serialization.
package config

import "errors"

// ErrFileNotFound is returned when a config file is missing. First-run is
// the common cause; callers decide whether the absence is fatal.
var ErrFileNotFound = errors.New("config file not found")

// ErrInvalidValue is returned when a value cannot be resolved: a malformed
// $ / ! sigil, an unset environment variable where one was required, or a
// shell command that exited non-zero.
var ErrInvalidValue = errors.New("invalid config value")

// ErrPermission is returned when a filesystem permission check or change
// fails (chmod, mkdir, write to a read-only location).
var ErrPermission = errors.New("permission denied")

// ErrSchemaViolation is returned when a config file parses as JSON but does
// not match the expected schema (unknown field, invalid enum value, wrong
// shape).
var ErrSchemaViolation = errors.New("schema violation")

// ErrUntrustedProject is returned when project-scoped settings access is
// attempted on a cwd that has not been marked trusted.
var ErrUntrustedProject = errors.New("project is not trusted")
