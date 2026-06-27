// auth.go — file-backed auth.json storage for provider API keys.
//
// The file format mirrors pi's:
//
//	{
//	  "anthropic": {"apiKey": "sk-..."},
//	  "openai":    {"apiKey": "sk-..."}
//	}
//
// File mode is 0600; directory mode is 0700. On Load, if the file mode is
// more permissive than 0600, we tighten it and emit a single diagnostic
// via the injected Diagnostic callback. Subsequent loads in the same
// process do not re-emit.

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

// ProviderAuth is one provider's stored credentials. OAuth fields are
// reserved for future expansion; the spec only requires API keys.
type ProviderAuth struct {
	APIKey     string `json:"apiKey,omitempty"`
	OAuthToken string `json:"oauthToken,omitempty"`
}

// AuthStore is the read/write interface for provider credentials.
// Implementations must be safe for concurrent use.
type AuthStore interface {
	Get(provider string) (string, bool)
	Set(provider, apiKey string) error
	Delete(provider string) error
	Load() (map[string]ProviderAuth, error)
	Save(entries map[string]ProviderAuth) error
}

// DiagnosticFunc receives one-shot informational messages about the auth
// store's health (e.g., "tightened mode on auth.json from 0644 to 0600").
type DiagnosticFunc func(msg string)

// FileAuthStore is the production backend. It serializes via an in-process
// mutex; cross-process safety is provided by flock on a separate .lock
// file (flock creates its target on first lock, so using the auth file
// itself would inject a zero-byte file that would later parse as EOF).
type FileAuthStore struct {
	path      string
	lock      *flock.Flock
	mu        sync.Mutex
	cached    map[string]ProviderAuth
	loaded    bool
	tightened bool // already emitted the mode-tightened diagnostic
	loadErr   error
	diag      DiagnosticFunc
}

// NewFileAuthStore returns a store backed at path. The diag callback, if
// non-nil, is invoked once when the file's mode is found to be looser
// than 0600.
func NewFileAuthStore(path string, diag DiagnosticFunc) *FileAuthStore {
	return &FileAuthStore{
		path: path,
		lock: flock.New(path + ".lock"),
		diag: diag,
	}
}

// Get returns the API key for provider. The boolean is false if no key is
// stored under that name.
func (s *FileAuthStore) Get(provider string) (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.loaded {
		s.loadLocked()
	}
	if e, ok := s.cached[provider]; ok {
		return e.APIKey, e.APIKey != ""
	}
	return "", false
}

// Set writes apiKey for provider and persists immediately. Empty apiKey
// clears the entry.
func (s *FileAuthStore) Set(provider, apiKey string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.loaded {
		s.loadLocked()
	}
	if s.loadErr != nil {
		return s.loadErr
	}
	if apiKey == "" {
		delete(s.cached, provider)
	} else {
		s.cached[provider] = ProviderAuth{APIKey: apiKey}
	}
	return s.saveLocked()
}

// Delete removes the entry for provider. Deleting a missing provider is
// not an error.
func (s *FileAuthStore) Delete(provider string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.loaded {
		s.loadLocked()
	}
	if s.loadErr != nil {
		return s.loadErr
	}
	delete(s.cached, provider)
	return s.saveLocked()
}

// Load returns a snapshot of all stored credentials. The returned map is
// a copy; callers may mutate it freely.
func (s *FileAuthStore) Load() (map[string]ProviderAuth, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.loadLocked()
	if s.loadErr != nil {
		return nil, s.loadErr
	}
	out := make(map[string]ProviderAuth, len(s.cached))
	for k, v := range s.cached {
		out[k] = v
	}
	return out, nil
}

// Save replaces all stored credentials with entries and persists them.
func (s *FileAuthStore) Save(entries map[string]ProviderAuth) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cached = make(map[string]ProviderAuth, len(entries))
	for k, v := range entries {
		s.cached[k] = v
	}
	s.loaded = true
	s.loadErr = nil
	return s.saveLocked()
}

// loadLocked reads the file once into s.cached. Subsequent calls are
// no-ops. Missing file is not an error. Permissive file modes are
// tightened and reported via the diag callback.
//
// loadLocked holds s.mu; callers must hold it.
func (s *FileAuthStore) loadLocked() {
	s.loaded = true
	s.cached = make(map[string]ProviderAuth)
	s.loadErr = nil

	if err := MkdirAll(filepath.Dir(s.path)); err != nil {
		s.loadErr = err
		return
	}
	if err := s.lock.RLock(); err == nil {
		defer func() { _ = s.lock.Unlock() }()
	}
	info, err := os.Stat(s.path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return
		}
		s.loadErr = err
		return
	}
	if info.Mode().Perm() != FileMode && !s.tightened {
		if err := os.Chmod(s.path, FileMode); err == nil {
			s.tightened = true
			if s.diag != nil {
				s.diag(fmt.Sprintf("tightened %s mode from %o to %o",
					s.path, info.Mode().Perm(), FileMode))
			}
		}
	}
	data, err := os.ReadFile(s.path)
	if err != nil {
		s.loadErr = err
		return
	}
	if len(data) == 0 {
		return
	}
	if err := strictJSONDecode(data, &s.cached); err != nil {
		s.cached = make(map[string]ProviderAuth)
		s.loadErr = err
		return
	}
}

// saveLocked writes s.cached atomically under an exclusive lock.
//
// Caller holds s.mu. We re-acquire the flock for the actual write so
// other processes may share the lock; s.mu serializes this process.
func (s *FileAuthStore) saveLocked() error {
	if err := s.lock.Lock(); err != nil {
		return err
	}
	defer func() { _ = s.lock.Unlock() }()
	out := make(map[string]ProviderAuth, len(s.cached))
	keys := make([]string, 0, len(s.cached))
	for k, v := range s.cached {
		keys = append(keys, k)
		out[k] = v
	}
	sort.Strings(keys)
	// Re-build a sorted-keyed map so json.Marshal emits deterministic output.
	sorted := make(map[string]ProviderAuth, len(keys))
	for _, k := range keys {
		sorted[k] = out[k]
	}
	return writeAtomic(s.path, sorted)
}

// InMemoryAuthStore is the test backend.
type InMemoryAuthStore struct {
	mu sync.Mutex
	m  map[string]ProviderAuth
}

// NewInMemoryAuthStore returns an empty in-memory store.
func NewInMemoryAuthStore() *InMemoryAuthStore {
	return &InMemoryAuthStore{m: make(map[string]ProviderAuth)}
}

// Get returns the stored API key for provider, or ("", false).
func (s *InMemoryAuthStore) Get(provider string) (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e, ok := s.m[provider]; ok {
		return e.APIKey, e.APIKey != ""
	}
	return "", false
}

// Set writes apiKey for provider. Empty apiKey clears the entry.
func (s *InMemoryAuthStore) Set(provider, apiKey string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if apiKey == "" {
		delete(s.m, provider)
		return nil
	}
	s.m[provider] = ProviderAuth{APIKey: apiKey}
	return nil
}

// Delete removes the entry for provider.
func (s *InMemoryAuthStore) Delete(provider string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.m, provider)
	return nil
}

// Load returns a copy of all stored credentials.
func (s *InMemoryAuthStore) Load() (map[string]ProviderAuth, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make(map[string]ProviderAuth, len(s.m))
	for k, v := range s.m {
		out[k] = v
	}
	return out, nil
}

// Save replaces all stored credentials.
func (s *InMemoryAuthStore) Save(entries map[string]ProviderAuth) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.m = make(map[string]ProviderAuth, len(entries))
	for k, v := range entries {
		s.m[k] = v
	}
	return nil
}
