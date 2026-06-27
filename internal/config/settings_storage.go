// settings_storage.go — file-backed Settings storage with reflection-based
// deep merge.
//
// Two scopes are supported:
//   - global:  $AGENT_DIR/settings.json
//   - project: <cwd>/.tau/settings.json
//
// The effective Settings is global deep-merged with project (project wins
// per-leaf). Project scope is unreadable and unwritable when the cwd is
// untrusted (returns ErrUntrustedProject).
//
// Concurrency:
//   - gofrs/flock guards the read-modify-write window per scope file.
//   - Load uses a shared lock; Save uses an exclusive lock.
//   - Atomic writes use temp file + rename (D2.5).

package config

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sync"

	"github.com/gofrs/flock"
)

// SettingsScope names a settings file scope.
type SettingsScope string

const (
	ScopeGlobal  SettingsScope = "global"
	ScopeProject SettingsScope = "project"
)

// SettingsStorage is the read/write interface for settings. Save applies
// fn to the current state of the named scope and writes the result back
// atomically. Load returns the deep-merged effective settings.
type SettingsStorage interface {
	Load(ctx context.Context) (Settings, error)
	Save(ctx context.Context, scope SettingsScope, fn func(current Settings) Settings) error
	Close() error
}

// FileSettingsStorage is the production backend: two JSON files guarded by
// gofrs/flock, written atomically via temp-then-rename.
type FileSettingsStorage struct {
	globalPath  string
	projectPath string
	globalLock  *flock.Flock
	projectLock *flock.Flock
	trusted     bool
	mu          sync.Mutex // serializes in-process Save calls
}

// NewFileSettingsStorage creates a storage backend rooted at agentDir for
// global scope and cwd/.tau for project scope. trusted controls whether
// the project scope is readable/writable.
//
// The flock instances use a separate "<path>.lock" file rather than the
// settings file itself: gofrs/flock creates its target on first lock, and
// using the settings file itself would produce a zero-byte file that
// later parses as an EOF error.
func NewFileSettingsStorage(agentDir, cwd string, trusted bool) (*FileSettingsStorage, error) {
	gp := filepath.Join(agentDir, "settings.json")
	if cwd == "" {
		return nil, errors.New("cwd is required")
	}
	pp := filepath.Join(cwd, ".tau", "settings.json")
	return &FileSettingsStorage{
		globalPath:  gp,
		projectPath: pp,
		globalLock:  flock.New(gp + ".lock"),
		projectLock: flock.New(pp + ".lock"),
		trusted:     trusted,
	}, nil
}

// Load reads both scopes and returns the deep-merged effective settings.
// Missing files are not errors: they contribute zero-valued Settings.
func (s *FileSettingsStorage) Load(ctx context.Context) (Settings, error) {
	if err := ctx.Err(); err != nil {
		return Settings{}, err
	}
	global, err := s.loadScope(s.globalPath, s.globalLock)
	if err != nil {
		return Settings{}, fmt.Errorf("load global settings: %w", err)
	}
	if !s.trusted {
		return mergeSettings(DefaultSettings(), global), nil
	}
	project, err := s.loadScope(s.projectPath, s.projectLock)
	if err != nil {
		return Settings{}, fmt.Errorf("load project settings: %w", err)
	}
	return mergeSettings(mergeSettings(DefaultSettings(), global), project), nil
}

// loadScope reads one scope file. Missing or empty file → zero Settings,
// nil error.
func (s *FileSettingsStorage) loadScope(path string, lock *flock.Flock) (Settings, error) {
	if err := lock.RLock(); err == nil {
		defer func() { _ = lock.Unlock() }()
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return Settings{}, nil
		}
		return Settings{}, err
	}
	if len(bytes.TrimSpace(data)) == 0 {
		return Settings{}, nil
	}
	var out Settings
	if err := strictJSONDecode(data, &out); err != nil {
		return Settings{}, err
	}
	return out, nil
}

// Save reads the current scope file, applies fn, validates, and writes
// the result atomically. Project-scope Save on an untrusted cwd returns
// ErrUntrustedProject without touching the file.
func (s *FileSettingsStorage) Save(ctx context.Context, scope SettingsScope, fn func(Settings) Settings) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if scope == ScopeProject && !s.trusted {
		return ErrUntrustedProject
	}
	path, lock := s.pathsFor(scope)
	if path == "" {
		return fmt.Errorf("%w: unknown scope %q", ErrSchemaViolation, scope)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Ensure the directory exists so flock can create its lockfile.
	if err := MkdirAll(filepath.Dir(path)); err != nil {
		return err
	}

	if err := lock.Lock(); err != nil {
		return err
	}
	defer func() { _ = lock.Unlock() }()

	data, err := os.ReadFile(path)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	var current Settings
	if err == nil && len(bytes.TrimSpace(data)) > 0 {
		if err := strictJSONDecode(data, &current); err != nil {
			return err
		}
	}
	next := fn(current)
	if err := next.Validate(); err != nil {
		return err
	}
	return writeAtomic(path, next)
}

// Close releases any held resources. FileSettingsStorage does not hold
// resources between calls (file locks are per-operation), so Close is a
// no-op; it satisfies the SettingsStorage interface for callers that
// defer Close uniformly across implementations.
func (s *FileSettingsStorage) Close() error { return nil }

func (s *FileSettingsStorage) pathsFor(scope SettingsScope) (string, *flock.Flock) {
	switch scope {
	case ScopeGlobal:
		return s.globalPath, s.globalLock
	case ScopeProject:
		return s.projectPath, s.projectLock
	}
	return "", nil
}

// writeAtomic marshals v as JSON, writes it to path+".tmp", then renames
// over path. The parent directory is created with mode 0700 if missing.
func writeAtomic(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	dir := filepath.Dir(path)
	if err := MkdirAll(dir); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, FileMode); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

// InMemorySettingsStorage is the test backend. It does no I/O and holds
// the two scope Settings values in memory.
type InMemorySettingsStorage struct {
	mu      sync.Mutex
	global  Settings
	project Settings
	trusted bool
}

// NewInMemorySettingsStorage returns an in-memory storage with the given
// trust state.
func NewInMemorySettingsStorage(trusted bool) *InMemorySettingsStorage {
	return &InMemorySettingsStorage{trusted: trusted}
}

// Load returns the deep-merged effective settings.
func (s *InMemorySettingsStorage) Load(ctx context.Context) (Settings, error) {
	if err := ctx.Err(); err != nil {
		return Settings{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	eff := mergeSettings(DefaultSettings(), s.global)
	if s.trusted {
		eff = mergeSettings(eff, s.project)
	}
	return eff, nil
}

// Save applies fn to the current scope state and stores the result.
func (s *InMemorySettingsStorage) Save(ctx context.Context, scope SettingsScope, fn func(Settings) Settings) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if scope == ScopeProject && !s.trusted {
		return ErrUntrustedProject
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	switch scope {
	case ScopeGlobal:
		s.global = fn(s.global)
		return s.global.Validate()
	case ScopeProject:
		s.project = fn(s.project)
		return s.project.Validate()
	}
	return fmt.Errorf("%w: unknown scope %q", ErrSchemaViolation, scope)
}

// Close is a no-op.
func (s *InMemorySettingsStorage) Close() error { return nil }

// mergeSettings returns base deep-merged with override (override wins at
// leaves). The base value is not mutated; a fresh copy is returned.
func mergeSettings(base, override Settings) Settings {
	out := deepCopy(reflect.ValueOf(base))
	ov := deepCopy(reflect.ValueOf(override))
	deepMerge(out, ov)
	return out.Interface().(Settings)
}

// deepMerge recursively merges src into dst (mutates dst). See D2.4.
//
// Settings types only contain Ptr/Struct/Slice/Map/scalar; other reflect.Kinds
// (Chan, Func, Complex*, UnsafePointer, …) are unreachable here and would
// have no meaningful merge semantics anyway.
func deepMerge(dst, src reflect.Value) {
	if !dst.IsValid() || !src.IsValid() {
		return
	}
	switch dst.Kind() { //nolint:exhaustive // see comment above
	case reflect.Ptr:
		if src.IsNil() {
			return
		}
		if dst.IsNil() {
			dst.Set(deepCopy(src))
			return
		}
		deepMerge(dst.Elem(), src.Elem())
	case reflect.Struct:
		for i := 0; i < dst.NumField(); i++ {
			if !dst.Type().Field(i).IsExported() {
				continue
			}
			deepMerge(dst.Field(i), src.Field(i))
		}
	case reflect.Slice:
		if src.IsNil() {
			return
		}
		dst.Set(deepCopy(src))
	case reflect.Map:
		if src.IsNil() {
			return
		}
		if dst.IsNil() {
			dst.Set(reflect.MakeMapWithSize(dst.Type(), src.Len()))
		}
		iter := src.MapRange()
		for iter.Next() {
			// Per-key merge: src wins for new keys, src overwrites for
			// existing keys. Settings map fields hold scalars (int,
			// *string), so deep recursion into values is unnecessary.
			dst.SetMapIndex(iter.Key(), deepCopy(iter.Value()))
		}
	default:
		// Scalar: override wins.
		if src.IsValid() {
			dst.Set(src)
		}
	}
}

// deepCopy returns a deep copy of v. For scalar kinds it returns v as-is.
//
// See deepMerge: Settings types only use Ptr/Struct/Slice/Map/scalar.
func deepCopy(v reflect.Value) reflect.Value {
	if !v.IsValid() {
		return v
	}
	switch v.Kind() { //nolint:exhaustive // see deepMerge comment
	case reflect.Ptr:
		if v.IsNil() {
			return v
		}
		c := reflect.New(v.Elem().Type())
		c.Elem().Set(deepCopy(v.Elem()))
		return c
	case reflect.Struct:
		c := reflect.New(v.Type()).Elem()
		for i := 0; i < v.NumField(); i++ {
			if !v.Type().Field(i).IsExported() {
				continue
			}
			c.Field(i).Set(deepCopy(v.Field(i)))
		}
		return c
	case reflect.Slice:
		if v.IsNil() {
			return v
		}
		c := reflect.MakeSlice(v.Type(), v.Len(), v.Len())
		for i := 0; i < v.Len(); i++ {
			c.Index(i).Set(deepCopy(v.Index(i)))
		}
		return c
	case reflect.Map:
		if v.IsNil() {
			return v
		}
		c := reflect.MakeMapWithSize(v.Type(), v.Len())
		iter := v.MapRange()
		for iter.Next() {
			c.SetMapIndex(deepCopy(iter.Key()), deepCopy(iter.Value()))
		}
		return c
	default:
		return v
	}
}
