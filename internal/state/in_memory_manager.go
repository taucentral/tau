package state

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/coevin/tau/internal/llm"
)

// inMemoryManager is the no-persistence Manager: all entries live in a
// slice guarded by a mutex. Used by tests and by InMemory() factory.
//
// The tree is reconstructed on every Tree()/BuildContext() call from the
// in-memory entry slice, so the cost is O(N) per call. Acceptable for
// ephemeral sessions; production sessions use boltManager.
type inMemoryManager struct {
	cwd string

	mu      sync.Mutex
	entries []Entry // append-only; IDs are unique within this slice
	leafID  string
	closed  bool
}

// NewInMemoryManager returns a Manager that does not persist. The returned
// manager is seeded with a SessionHeader entry as the root so callers can
// immediately Append without first writing a header themselves.
func NewInMemoryManager(cwd string) Manager {
	m := &inMemoryManager{cwd: cwd}
	// Seed the root SessionHeader so the tree invariant (exactly one
	// root, kind SessionHeader) holds. The root ID is generated via the
	// same collision-checked path as any other Append.
	header := SessionHeaderPayload{
		SessionID: "",
		Cwd:       cwd,
		Version:   CurrentSchemaVersion,
		CreatedAt: time.Now().UTC(),
	}
	// Inline the seeding (cannot call Append because Append would
	// require the first entry to be a SessionHeader — which it is — but
	// Append also reads m.leafID which is "" pre-seed, exactly what we
	// want for the root's ParentID).
	rootID, err := NewID(func(string) bool { return false })
	if err != nil {
		// NewID with an always-false collision check cannot fail under
		// crypto/rand; surface as a panic so the caller sees a loud bug.
		panic("state: in-memory seed: " + err.Error())
	}
	header.SessionID = rootID
	m.entries = append(m.entries, Entry{
		ID:        rootID,
		ParentID:  "",
		Kind:      KindSessionHeader,
		Timestamp: time.Now().UTC(),
		Payload:   header,
	})
	m.leafID = rootID
	return m
}

// Append assigns an ID, appends the entry as a child of the current leaf,
// and advances the leaf pointer. The root SessionHeader is seeded at
// construction (NewInMemoryManager), so callers always Append children.
func (m *inMemoryManager) Append(entry Entry) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return "", ErrManagerClosed
	}
	id, err := NewID(func(candidate string) bool {
		for _, e := range m.entries {
			if e.ID == candidate {
				return true
			}
		}
		return false
	})
	if err != nil {
		return "", err
	}
	e := Entry{
		ID:        id,
		ParentID:  m.leafID,
		Kind:      kindOf(entry.Payload),
		Timestamp: time.Now().UTC(),
		Payload:   entry.Payload,
	}
	m.entries = append(m.entries, e)
	m.leafID = id
	return id, nil
}

// AppendAt writes entry as a child of parentID without advancing the
// leaf pointer. Used by branchManager and MergeState to integrate entries
// into a shared tree without disturbing the active leaf.
//
// Returns ErrInvalidBranch when parentID is not in the entries slice.
func (m *inMemoryManager) AppendAt(entry Entry, parentID string) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return "", ErrManagerClosed
	}
	// Validate parentID in entries.
	found := false
	for _, e := range m.entries {
		if e.ID == parentID {
			found = true
			break
		}
	}
	if !found {
		return "", fmt.Errorf("%w: parent %q not in tree", ErrInvalidBranch, parentID)
	}
	id, err := NewID(func(candidate string) bool {
		for _, e := range m.entries {
			if e.ID == candidate {
				return true
			}
		}
		return false
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
	m.entries = append(m.entries, e)
	// Intentionally do NOT advance m.leafID — that's the AppendAt contract.
	return id, nil
}

// Branch moves the leaf pointer to fromID.
func (m *inMemoryManager) Branch(fromID string) error {
	return m.SetLeaf(fromID)
}

// BranchWithSummary defers to the agent loop per the cross-phase contract.
func (m *inMemoryManager) BranchWithSummary(ctx context.Context, fromID string, client llm.LLMClient) (string, error) {
	return "", ErrBranchWithSummaryUnsupported
}

// CreateBranchedSession extracts the path root → leafID into a new
// in-memory manager.
func (m *inMemoryManager) CreateBranchedSession(leafID string) (Manager, error) {
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
	origHeader, _ := path[0].Payload.(SessionHeaderPayload)
	child := NewInMemoryManager(m.cwd).(*inMemoryManager)
	newHeader := SessionHeaderPayload{
		Cwd:           m.cwd,
		Model:         origHeader.Model,
		Provider:      origHeader.Provider,
		Version:       CurrentSchemaVersion,
		ParentSession: origHeader.SessionID,
		CreatedAt:     time.Now().UTC(),
	}
	if _, err := child.Append(Entry{Kind: KindSessionHeader, Payload: newHeader}); err != nil {
		return nil, err
	}
	for _, e := range path[1:] {
		if _, err := child.Append(Entry{Kind: e.Kind, Payload: e.Payload}); err != nil {
			return nil, err
		}
	}
	return child, nil
}

// ForkFrom delegates to OpenManager on the source path then CreateBranchedSession.
func (m *inMemoryManager) ForkFrom(sourcePath string) (Manager, error) {
	src, err := OpenManager(sourcePath, m.cwd)
	if err != nil {
		return nil, err
	}
	defer src.Close()
	return src.CreateBranchedSession(src.LeafID())
}

// ContinueRecent returns ErrNoSession for in-memory managers (no backing
// store means no "recent" session to continue).
func (m *inMemoryManager) ContinueRecent(cwd string) (Manager, error) {
	return nil, ErrNoSession
}

// InMemory returns another in-memory manager for the given cwd.
func (m *inMemoryManager) InMemory(cwd string) Manager {
	return NewInMemoryManager(cwd)
}

// List delegates to a boltManager-style scan via config.Paths. In-memory
// managers don't have a cwd-scoped backing store, so we route through the
// shared listSessionsInDir helper for consistency with the Manager
// interface contract.
func (m *inMemoryManager) List(cwd string) ([]SessionInfo, error) {
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

// ListAll mirrors boltManager.ListAll via the shared helpers.
func (m *inMemoryManager) ListAll() ([]SessionInfo, error) {
	tmp := &boltManager{cwd: m.cwd}
	return tmp.ListAll()
}

// BuildContext delegates to the shared algorithm in context.go.
func (m *inMemoryManager) BuildContext(ctx context.Context) (Context, error) {
	m.mu.Lock()
	leaf := m.leafID
	entries := make([]Entry, len(m.entries))
	copy(entries, m.entries)
	m.mu.Unlock()
	if leaf == "" {
		return Context{}, nil
	}
	tree, err := NewTree(entries)
	if err != nil {
		return Context{}, err
	}
	return buildContextFromTree(tree, leaf)
}

// LeafID returns the current leaf entry ID.
func (m *inMemoryManager) LeafID() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.leafID
}

// SetLeaf moves the leaf pointer to id.
func (m *inMemoryManager) SetLeaf(id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, e := range m.entries {
		if e.ID == id {
			m.leafID = id
			return nil
		}
	}
	return fmt.Errorf("%w: %q not in tree", ErrInvalidBranch, id)
}

// Tree reconstructs a Tree from the in-memory entry slice.
func (m *inMemoryManager) Tree() (*Tree, error) {
	m.mu.Lock()
	entries := make([]Entry, len(m.entries))
	copy(entries, m.entries)
	m.mu.Unlock()
	return NewTree(entries)
}

// Close marks the manager closed; subsequent operations return
// ErrManagerClosed. In-memory state is discarded.
func (m *inMemoryManager) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil
	}
	m.closed = true
	m.entries = nil
	m.leafID = ""
	return nil
}
