package state

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/taucentral/tau/internal/llm"
)

// branchManager shares the parent Manager's READ-side store and maintains a
// private shadow buffer for this branch's writes. Reads (Tree, BuildContext)
// merge parent store + shadow; writes (Append) go to the shadow only.
// MergeState integrates the shadow into the parent via AppendAt;
// MergePolicyNone discards the shadow — no orphans, no cleanup.
//
// Design rationale (state-tree spec "branches share a tree structure"):
// the literal realization of a "branch on the parent's tree" is a shared
// backing store with independent leaf pointers. The shadow buffer avoids
// orphan entries: writes never touch the parent's store, so discarding a
// branch (MergePolicyNone) leaves zero residue.
//
// Concurrency: the mutex guards shadow, leafID, and closed. The parent
// Manager is concurrency-safe, so parent.Tree() calls run without holding
// the branch mutex (the returned Tree is a read-only snapshot). Parent ID
// cache is populated once at construction and is stable for the branch's
// lifetime — orchestration semantics freeze the parent's tree while
// children run (the parent's leaf only advances via MergeState AFTER the
// child completes).
type branchManager struct {
	parent    Manager
	cwd       string
	branchAt  string // parent's leaf ID at branch time — parent of first shadow entry
	parentIDs map[string]struct{}

	mu     sync.Mutex
	shadow []Entry // this branch's appends, parented on branchAt or prior shadow entries
	leafID string  // this branch's leaf (branchAt or a shadow entry ID)
	closed bool
}

// NewBranchManager creates a Manager whose reads see the parent's tree
// plus a private shadow buffer, and whose writes (Append) go to the shadow
// only. The branch's initial leaf is the parent's current leaf; the first
// Append hangs the new entry off that leaf.
//
// The parent's tree is treated as read-only for the lifetime of the
// branch: parent.Tree() is called on read and at construction (to cache
// parent IDs for collision checking), never for mutation. MergeState
// integrates the shadow into the parent via AppendAt at merge time.
func NewBranchManager(parent Manager, cwd string) Manager {
	branchAt := parent.LeafID()
	parentIDs := map[string]struct{}{}
	if ptree, err := parent.Tree(); err == nil {
		for _, id := range ptree.IDs() {
			parentIDs[id] = struct{}{}
		}
	}
	return &branchManager{
		parent:    parent,
		cwd:       cwd,
		branchAt:  branchAt,
		parentIDs: parentIDs,
		leafID:    branchAt,
	}
}

// Append writes a new entry as a child of the current leaf and advances
// the leaf pointer to the new entry. The entry goes to the shadow buffer;
// the parent's store is untouched.
func (m *branchManager) Append(entry Entry) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return "", ErrManagerClosed
	}
	parent := m.leafID
	id, err := NewID(func(candidate string) bool {
		if _, ok := m.parentIDs[candidate]; ok {
			return true
		}
		for _, e := range m.shadow {
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
		ParentID:  parent,
		Kind:      kindOf(entry.Payload),
		Timestamp: time.Now().UTC(),
		Payload:   entry.Payload,
	}
	m.shadow = append(m.shadow, e)
	m.leafID = id
	return id, nil
}

// AppendAt writes entry as a child of parentID without advancing the leaf
// pointer. parentID must resolve in the parent's tree or the shadow.
func (m *branchManager) AppendAt(entry Entry, parentID string) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return "", ErrManagerClosed
	}
	if !m.idExistsLocked(parentID) {
		return "", fmt.Errorf("%w: parent %q not in tree", ErrInvalidBranch, parentID)
	}
	id, err := NewID(func(candidate string) bool {
		if _, ok := m.parentIDs[candidate]; ok {
			return true
		}
		for _, e := range m.shadow {
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
	m.shadow = append(m.shadow, e)
	return id, nil
}

// idExistsLocked reports whether id resolves in the parent cache or the
// shadow. Called with m.mu held.
func (m *branchManager) idExistsLocked(id string) bool {
	if _, ok := m.parentIDs[id]; ok {
		return true
	}
	for _, e := range m.shadow {
		if e.ID == id {
			return true
		}
	}
	return false
}

// Branch moves the leaf pointer to fromID. fromID must resolve in the
// parent's tree or the shadow.
func (m *branchManager) Branch(fromID string) error {
	return m.SetLeaf(fromID)
}

// BranchWithSummary defers to the agent loop per the cross-phase contract.
// The branchManager itself never summarizes; the agent composes Branch +
// BranchSummary entries.
func (m *branchManager) BranchWithSummary(ctx context.Context, fromID string, client llm.LLMClient) (string, error) {
	return "", ErrBranchWithSummaryUnsupported
}

// CreateBranchedSession extracts the path root → leafID from the combined
// (parent + shadow) tree into a new persistent session file. The new
// session inherits Cwd and records the parent via ParentSession.
func (m *branchManager) CreateBranchedSession(leafID string) (Manager, error) {
	combined, err := m.combinedEntries()
	if err != nil {
		return nil, err
	}
	tree, err := NewTree(combined)
	if err != nil {
		return nil, err
	}
	return createBranchedSessionFromTree(tree, leafID, m.cwd)
}

// ForkFrom opens sourcePath and forks it into a new session under cwd.
func (m *branchManager) ForkFrom(sourcePath string) (Manager, error) {
	src, err := OpenManager(sourcePath, m.cwd)
	if err != nil {
		return nil, err
	}
	defer src.Close()
	return src.CreateBranchedSession(src.LeafID())
}

// ContinueRecent delegates to the parent (the branch shares the parent's
// sessions directory).
func (m *branchManager) ContinueRecent(cwd string) (Manager, error) {
	return m.parent.ContinueRecent(cwd)
}

// InMemory returns a fresh in-memory Manager for cwd.
func (m *branchManager) InMemory(cwd string) Manager {
	return NewInMemoryManager(cwd)
}

// List delegates to the parent.
func (m *branchManager) List(cwd string) ([]SessionInfo, error) {
	return m.parent.List(cwd)
}

// ListAll delegates to the parent.
func (m *branchManager) ListAll() ([]SessionInfo, error) {
	return m.parent.ListAll()
}

// combinedEntries returns the parent's entries concatenated with the
// shadow. The shadow entries' ParentIDs resolve either to parent entries
// (the first shadow entry parents on branchAt) or to earlier shadow
// entries, so the combined slice forms a valid single-root tree.
func (m *branchManager) combinedEntries() ([]Entry, error) {
	m.mu.Lock()
	shadow := append([]Entry(nil), m.shadow...)
	m.mu.Unlock()
	return m.combinedFromShadow(shadow)
}

// combinedFromShadow fetches the parent's entries and appends the supplied
// shadow snapshot. Split out so callers can snapshot shadow under the lock
// and build the combined set outside it.
func (m *branchManager) combinedFromShadow(shadow []Entry) ([]Entry, error) {
	ptree, err := m.parent.Tree()
	if err != nil {
		return nil, err
	}
	ids := ptree.IDs()
	out := make([]Entry, 0, len(ids)+len(shadow))
	for _, id := range ids {
		e, _ := ptree.Get(id)
		out = append(out, e)
	}
	out = append(out, shadow...)
	return out, nil
}

// BuildContext walks leaf → root over the combined tree per state-tree
// spec "BuildContext walk".
func (m *branchManager) BuildContext(ctx context.Context) (Context, error) {
	m.mu.Lock()
	leaf := m.leafID
	shadow := append([]Entry(nil), m.shadow...)
	m.mu.Unlock()
	combined, err := m.combinedFromShadow(shadow)
	if err != nil {
		return Context{}, err
	}
	tree, err := NewTree(combined)
	if err != nil {
		return Context{}, err
	}
	return buildContextFromTree(tree, leaf)
}

// LeafID returns the current leaf entry ID.
func (m *branchManager) LeafID() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.leafID
}

// SetLeaf moves the leaf pointer to id. id must resolve in the parent's
// tree or the shadow.
func (m *branchManager) SetLeaf(id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return ErrManagerClosed
	}
	if !m.idExistsLocked(id) {
		return fmt.Errorf("%w: %q not in tree", ErrInvalidBranch, id)
	}
	m.leafID = id
	return nil
}

// Tree reconstructs a validated Tree from the parent's entries plus the
// shadow.
func (m *branchManager) Tree() (*Tree, error) {
	combined, err := m.combinedEntries()
	if err != nil {
		return nil, err
	}
	return NewTree(combined)
}

// Close marks the branch closed; subsequent operations return
// ErrManagerClosed. The shadow is discarded. The parent Manager is NOT
// closed — the embedder (or the parent session) owns the parent's
// lifecycle.
func (m *branchManager) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil
	}
	m.closed = true
	m.shadow = nil
	m.leafID = ""
	return nil
}

// BranchShadow returns the shadow entries of a branch manager created by
// NewBranchManager, or (nil, false) when mgr is not a branch manager
// (e.g., a boltManager or inMemoryManager). Used by agent.MergeState to
// integrate branch writes into the parent via AppendAt. The returned
// slice is a defensive copy; callers may mutate it freely.
func BranchShadow(mgr Manager) ([]Entry, bool) {
	if bm, ok := mgr.(*branchManager); ok {
		bm.mu.Lock()
		defer bm.mu.Unlock()
		out := make([]Entry, len(bm.shadow))
		copy(out, bm.shadow)
		return out, true
	}
	return nil, false
}

// IsBranch reports whether mgr is a branch manager (i.e., was created by
// NewBranchManager). Used by agent.MergeState to distinguish branch-based
// children (shadow-integration path) from legacy in-memory children
// (copy-then-append fallback).
func IsBranch(mgr Manager) bool {
	_, ok := mgr.(*branchManager)
	return ok
}

// IsClosed reports whether mgr has been closed via Close. Type-switches on
// the concrete implementation known to this package. Returns false for
// unknown types (custom Manager implementations from embedders). Used by
// agent.MergeState to detect closed/deallocated child state per the
// orchestration contract.
func IsClosed(mgr Manager) bool {
	switch m := mgr.(type) {
	case *boltManager:
		m.mu.Lock()
		defer m.mu.Unlock()
		return m.closed
	case *inMemoryManager:
		m.mu.Lock()
		defer m.mu.Unlock()
		return m.closed
	case *lazyManager:
		m.mu.Lock()
		defer m.mu.Unlock()
		return m.closed
	case *branchManager:
		m.mu.Lock()
		defer m.mu.Unlock()
		return m.closed
	}
	return false
}
