package state

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/taucentral/tau/internal/config"
	"github.com/taucentral/tau/internal/llm"
)

// boltManager is the production Manager: a single bbolt file per session.
// It owns one Store and caches the current LeafID for fast Append/Tree
// operations.
//
// Concurrency: the mutex guards closed and leafID. Store operations are
// concurrency-safe via bbolt MVCC, so the mutex is held only briefly to
// read/update the cached leafID — never across an entire Append.
type boltManager struct {
	store *Store
	cwd   string

	mu     sync.Mutex
	leafID string // cached from meta[leaf]; read via LeafID()
	closed bool
}

// CreateManager creates a new session file under cwd's sessions directory
// and returns a Manager rooted at a fresh SessionHeader entry. The new
// file's name encodes the creation timestamp and a random session ID.
//
// cwd MUST be non-empty; SessionID/Model/Provider in the header are
// persisted verbatim. Returns a wrapped error if the directory or file
// cannot be created.
func CreateManager(cwd string, header SessionHeaderPayload) (Manager, error) {
	dir, err := sessionsDirFor(cwd)
	if err != nil {
		return nil, err
	}
	path, sessionID, err := newSessionFilePath(dir)
	if err != nil {
		return nil, err
	}
	header.SessionID = sessionID
	header.Version = CurrentSchemaVersion
	header.Cwd = cwd
	if header.CreatedAt.IsZero() {
		header.CreatedAt = time.Now().UTC()
	}
	return openBoltManager(path, cwd, header)
}

// OpenManager opens an existing session file and returns a Manager. The
// session's existing SessionHeader entry becomes the root; meta[leaf]
// becomes the cached LeafID. Returns a wrapped error if the file is not a
// valid session (missing root, missing leaf, etc.).
func OpenManager(path, cwd string) (Manager, error) {
	store, err := OpenStore(path)
	if err != nil {
		return nil, err
	}
	leaf, err := store.LeafID()
	if err != nil {
		store.Close()
		return nil, err
	}
	mgr := &boltManager{
		store:  store,
		cwd:    cwd,
		leafID: leaf,
	}
	return mgr, nil
}

// openBoltManager is the shared constructor for CreateManager and tests:
// opens the store, writes the SessionHeader as the root entry, sets the
// leaf pointer to it.
func openBoltManager(path, cwd string, header SessionHeaderPayload) (*boltManager, error) {
	store, err := OpenStore(path)
	if err != nil {
		return nil, err
	}
	rootID, err := NewID(func(candidate string) bool {
		ok, err := store.Exists(candidate)
		// On error surface as collision so the caller sees it.
		return err != nil || ok
	})
	if err != nil {
		store.Close()
		return nil, err
	}
	root := Entry{
		ID:        rootID,
		ParentID:  "",
		Kind:      KindSessionHeader,
		Timestamp: time.Now().UTC(),
		Payload:   header,
	}
	if err := store.Append(root); err != nil {
		store.Close()
		return nil, fmt.Errorf("state: write session header: %w", err)
	}
	if err := store.SetLeaf(rootID); err != nil {
		store.Close()
		return nil, err
	}
	return &boltManager{
		store:  store,
		cwd:    cwd,
		leafID: rootID,
	}, nil
}

// Append assigns an ID, persists the entry as a child of the current leaf,
// advances the leaf pointer, and returns the new ID.
func (m *boltManager) Append(entry Entry) (string, error) {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return "", ErrManagerClosed
	}
	leaf := m.leafID
	m.mu.Unlock()

	id, err := NewID(func(candidate string) bool {
		ok, err := m.store.Exists(candidate)
		return err != nil || ok
	})
	if err != nil {
		return "", err
	}
	// kindOf derives the authoritative Kind from the payload type, so a
	// caller-supplied Kind mismatch is silently corrected.
	e := Entry{
		ID:        id,
		ParentID:  leaf,
		Kind:      kindOf(entry.Payload),
		Timestamp: time.Now().UTC(),
		Payload:   entry.Payload,
	}
	if err := m.store.Append(e); err != nil {
		return "", err
	}
	if err := m.store.SetLeaf(id); err != nil {
		return "", err
	}
	m.mu.Lock()
	m.leafID = id
	m.mu.Unlock()
	return id, nil
}

// AppendAt writes entry as a child of parentID and returns the assigned
// ID. Unlike Append, AppendAt does NOT advance the leaf pointer — the
// caller owns leaf movement via SetLeaf. Used by branchManager to write
// into a shared backing store without disturbing the parent's leaf, and
// by MergeState to integrate a branch's entries one at a time without
// disturbing the parent's leaf mid-walk.
//
// Returns ErrInvalidBranch when parentID is not in the store.
func (m *boltManager) AppendAt(entry Entry, parentID string) (string, error) {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return "", ErrManagerClosed
	}
	m.mu.Unlock()

	// Validate parentID exists in the store.
	if _, err := m.store.Get(parentID); err != nil {
		if errors.Is(err, ErrEntryNotFound) {
			return "", fmt.Errorf("%w: parent %q not in tree", ErrInvalidBranch, parentID)
		}
		return "", err
	}

	id, err := NewID(func(candidate string) bool {
		ok, err := m.store.Exists(candidate)
		return err != nil || ok
	})
	if err != nil {
		return "", err
	}
	e := Entry{
		ID:        id,
		ParentID:  parentID,
		Kind:      kindOf(entry.Payload),
		Timestamp: time.Now().UTC(),
		Payload:   entry.Payload,
	}
	if err := m.store.Append(e); err != nil {
		return "", err
	}
	// Intentionally do NOT call SetLeaf — that's the AppendAt contract.
	return id, nil
}

// Branch moves the leaf pointer to fromID. Future appends descend from
// fromID; the previously-active path remains in the tree.
func (m *boltManager) Branch(fromID string) error {
	return m.SetLeaf(fromID)
}

// BranchWithSummary is documented in the Manager interface. The bolt
// implementation defers to the agent loop (Phase 8) which composes the
// real flow from Branch + a BranchSummary entry. Returns
// ErrBranchWithSummaryUnsupported so callers can detect the handoff.
func (m *boltManager) BranchWithSummary(ctx context.Context, fromID string, client llm.LLMClient) (string, error) {
	return "", ErrBranchWithSummaryUnsupported
}

// CreateBranchedSession extracts the path root → leafID into a new session
// file with a fresh session ID, recording the parent via the new header's
// ParentSession field per state-tree spec scenario "CreateBranchedSession
// extracts a path".
func (m *boltManager) CreateBranchedSession(leafID string) (Manager, error) {
	tree, err := m.Tree()
	if err != nil {
		return nil, err
	}
	path, err := tree.Path(leafID)
	if err != nil {
		return nil, err
	}
	if len(path) == 0 || path[0].Kind != KindSessionHeader {
		return nil, fmt.Errorf("%w: path root is not a SessionHeader", ErrInvalidBranch)
	}
	origHeader, ok := path[0].Payload.(SessionHeaderPayload)
	if !ok {
		return nil, fmt.Errorf("%w: path root payload type %T", ErrInvalidBranch, path[0].Payload)
	}
	newHeader := SessionHeaderPayload{
		Cwd:           m.cwd,
		Model:         origHeader.Model,
		Provider:      origHeader.Provider,
		Version:       CurrentSchemaVersion,
		ParentSession: origHeader.SessionID,
		CreatedAt:     time.Now().UTC(),
	}
	child, err := CreateManager(m.cwd, newHeader)
	if err != nil {
		return nil, err
	}
	// Copy path[1:] (skip original root) via Append. Each Append advances
	// the new manager's leaf, so the chain reconstructs naturally.
	for _, e := range path[1:] {
		if _, err := child.Append(Entry{Kind: e.Kind, Payload: e.Payload}); err != nil {
			child.Close()
			return nil, err
		}
	}
	return child, nil
}

// ForkFrom opens sourcePath and forks it into a new session under cwd.
// Equivalent to source.CreateBranchedSession(source.LeafID()).
func (m *boltManager) ForkFrom(sourcePath string) (Manager, error) {
	src, err := OpenManager(sourcePath, m.cwd)
	if err != nil {
		return nil, err
	}
	defer src.Close()
	return src.CreateBranchedSession(src.LeafID())
}

// ContinueRecent opens the most recently modified session under cwd, or
// returns ErrNoSession when none exists.
func (m *boltManager) ContinueRecent(cwd string) (Manager, error) {
	sessions, err := m.List(cwd)
	if err != nil {
		return nil, err
	}
	if len(sessions) == 0 {
		return nil, ErrNoSession
	}
	// List returns newest-first by mtime.
	return OpenManager(sessions[0].Path, cwd)
}

// InMemory returns an in-memory Manager (no persistence) rooted at cwd.
// The returned Manager has an empty tree until entries are appended.
func (m *boltManager) InMemory(cwd string) Manager {
	return NewInMemoryManager(cwd)
}

// List returns SessionInfo for every session under cwd. Newest-first by
// mtime. Returns nil (not an error) when no sessions exist.
func (m *boltManager) List(cwd string) ([]SessionInfo, error) {
	dir, err := sessionsDirFor(cwd)
	if err != nil {
		return nil, err
	}
	out, err := listSessionsInDir(dir, cwd)
	if err != nil {
		return nil, err
	}
	sortSessionsByLastActiveDesc(out)
	return out, nil
}

// ListAll returns SessionInfo for every session across all cwds. Walks the
// sessions directory. Newest-first by mtime.
func (m *boltManager) ListAll() ([]SessionInfo, error) {
	dir, err := config.AgentDir()
	if err != nil {
		return nil, err
	}
	sessionsRoot := filepath.Join(dir, "sessions")
	entries, err := os.ReadDir(sessionsRoot)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var all []SessionInfo
	for _, ent := range entries {
		if !ent.IsDir() {
			continue
		}
		// The directory name is the encoded cwd; we pass it through as
		// SessionInfo.Cwd for now (decoding to a real path is a display
		// concern handled by the TUI).
		encoded := ent.Name()
		dirPath := filepath.Join(sessionsRoot, encoded)
		sessions, err := listSessionsInDir(dirPath, encoded)
		if err != nil {
			continue
		}
		all = append(all, sessions...)
	}
	sortSessionsByLastActiveDesc(all)
	return all, nil
}

// BuildContext walks leaf → root per state-tree spec and returns the
// assembled Context. See context.go for the algorithm.
func (m *boltManager) BuildContext(ctx context.Context) (Context, error) {
	m.mu.Lock()
	leaf := m.leafID
	m.mu.Unlock()
	return buildContext(m.store, leaf)
}

// LeafID returns the current leaf entry ID.
func (m *boltManager) LeafID() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.leafID
}

// SetLeaf moves the leaf pointer to id. Returns ErrInvalidBranch when id
// is not in the store.
func (m *boltManager) SetLeaf(id string) error {
	if _, err := m.store.Get(id); err != nil {
		if errors.Is(err, ErrEntryNotFound) {
			return fmt.Errorf("%w: %q not in tree", ErrInvalidBranch, id)
		}
		return err
	}
	if err := m.store.SetLeaf(id); err != nil {
		return err
	}
	m.mu.Lock()
	m.leafID = id
	m.mu.Unlock()
	return nil
}

// Tree loads every entry and returns a validated Tree.
func (m *boltManager) Tree() (*Tree, error) {
	entries, err := m.store.All()
	if err != nil {
		return nil, err
	}
	return NewTree(entries)
}

// Close releases the underlying store. Idempotent.
func (m *boltManager) Close() error {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return nil
	}
	m.closed = true
	m.mu.Unlock()
	return m.store.Close()
}
