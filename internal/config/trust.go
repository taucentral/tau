// trust.go — file-backed project trust store.
//
// The file at $AGENT_DIR/trust.json is a flat map from absolute cwd path
// to trust status. tau consults this to decide whether to load
// project-scoped settings/plugins/.agents from a cwd.
//
// The actual prompting ("Trust this project? [y/n]") lives in the cli
// layer (Phase 9). The config package only persists decisions.
//
// File format:
//
//	{
//	  "/home/alice/proj":    "trusted",
//	  "/home/alice/scratch": "untrusted"
//	}

package config

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"github.com/gofrs/flock"
)

// TrustStatus is the persisted per-cwd decision.
type TrustStatus string

const (
	TrustTrusted   TrustStatus = "trusted"
	TrustUntrusted TrustStatus = "untrusted"
)

// validStatuses is the set of acceptable TrustStatus values.
var validStatuses = map[TrustStatus]bool{
	TrustTrusted:   true,
	TrustUntrusted: true,
}

// TrustStore is the read/write interface for project trust decisions.
type TrustStore interface {
	Load() error
	IsTrusted(cwd string) (bool, error)
	SetTrust(cwd string, status TrustStatus) error
	Save() error
}

// FileTrustStore persists trust decisions to a JSON file at path. The
// file mode is tightened to 0600 on every Save.
type FileTrustStore struct {
	path  string
	lock  *flock.Flock
	mu    sync.Mutex
	m     map[string]TrustStatus
	dirty bool
}

// NewFileTrustStore returns a store backed at path. The caller must call
// Load before any read; the methods do not auto-load so that callers can
// distinguish "first run" from "load failed".
func NewFileTrustStore(path string) *FileTrustStore {
	return &FileTrustStore{
		path: path,
		lock: flock.New(path + ".lock"),
		m:    make(map[string]TrustStatus),
	}
}

// Load reads the trust file once into memory. Missing file is not an
// error. Malformed JSON, schema-violating JSON, or a permissive file
// mode (any non-0600) is returned as an error.
func (s *FileTrustStore) Load() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.m = make(map[string]TrustStatus)
	s.dirty = false

	if err := MkdirAll(filepath.Dir(s.path)); err != nil {
		return err
	}
	if err := s.lock.RLock(); err == nil {
		defer func() { _ = s.lock.Unlock() }()
	}
	info, err := os.Stat(s.path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	if info.Mode().Perm() != FileMode {
		if err := os.Chmod(s.path, FileMode); err != nil {
			return err
		}
	}
	data, err := os.ReadFile(s.path)
	if err != nil {
		return err
	}
	if len(data) == 0 {
		return nil
	}
	if err := strictJSONDecode(data, &s.m); err != nil {
		return err
	}
	// Validate values.
	for k, v := range s.m {
		if !validStatuses[v] {
			return fmt.Errorf("%w: trust entry %q has invalid status %q",
				ErrSchemaViolation, k, v)
		}
	}
	return nil
}

// IsTrusted returns whether cwd is marked trusted. False is returned for
// unknown cwds (no entry). Invalid cwds (empty) are also false.
func (s *FileTrustStore) IsTrusted(cwd string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if cwd == "" {
		return false, nil
	}
	status, ok := s.m[filepath.Clean(cwd)]
	if !ok {
		return false, nil
	}
	return status == TrustTrusted, nil
}

// SetTrust adds or replaces the entry for cwd. An already-known cwd is
// overwritten. The change is staged in memory; Save persists it.
func (s *FileTrustStore) SetTrust(cwd string, status TrustStatus) error {
	if cwd == "" {
		return errors.New("cwd is required")
	}
	if !validStatuses[status] {
		return fmt.Errorf("%w: invalid trust status %q", ErrSchemaViolation, status)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.m[filepath.Clean(cwd)] = status
	s.dirty = true
	return nil
}

// Save writes the in-memory map to disk if it has changed since Load.
// Atomic write via temp + rename. File mode is 0600; directory mode is
// 0700.
func (s *FileTrustStore) Save() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.dirty {
		return nil
	}
	if err := s.lock.Lock(); err != nil {
		return err
	}
	defer func() { _ = s.lock.Unlock() }()

	// Build a sorted-keyed map for deterministic output.
	keys := make([]string, 0, len(s.m))
	for k := range s.m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	sorted := make(map[string]TrustStatus, len(keys))
	for _, k := range keys {
		sorted[k] = s.m[k]
	}
	if err := writeAtomic(s.path, sorted); err != nil {
		return err
	}
	s.dirty = false
	return nil
}
