package state

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/coevin/tau/internal/llm"
)

// lazyManager wraps a buffer plus an eventual boltManager. It implements
// the state-tree spec "Lazy file creation" requirement: no .bolt file is
// written until the first Append of an assistant message. If the manager
// is closed before that, no file appears on disk.
//
// Lifecycle:
//
//  1. Create(cwd, header) returns a *lazyManager with an empty buffer.
//  2. Append adds entries to the buffer; LeafID advances; no file is
//     touched on disk.
//  3. The first Append whose payload is a Message with Role=assistant
//     triggers flushLocked: CreateManager writes the .bolt file with the
//     SessionHeader as root, the buffer is batch-appended (re-parented to
//     the new root), and the lazy manager switches to "delegating mode".
//  4. Subsequent operations go straight to the underlying boltManager.
//
// Concurrency: the mutex guards buffer, leafID, bolt, and closed. The
// bolt field is read under the mutex to decide between buffer-mode and
// delegating-mode; the bolt operations themselves run outside the lock
// because boltManager is already concurrency-safe.
type lazyManager struct {
	cwd    string
	header SessionHeaderPayload

	mu     sync.Mutex
	buffer []Entry      // entries appended pre-flush
	leafID string       // current leaf (advanced by Append / Branch)
	bolt   *boltManager // non-nil after flush
	closed bool
}

// Create returns a Manager that defers .bolt file creation until the first
// Append of an assistant message. Per state-tree spec scenario "Aborted
// before first response": if Close is called before any assistant message,
// no file appears on disk.
//
// The SessionHeader root is materialized in the buffer at Create time so
// that Tree(), LeafID(), and AppendAt() all work before flush. This is
// essential for the orchestration seam: MergeState calls AppendAt on a
// parent that may never have had Run (and thus never had Append) called
// directly. Without a materialized root, there is nothing to parent onto.
// flushLocked skips this synthetic root (CreateManager writes its own).
func Create(cwd string, header SessionHeaderPayload) (Manager, error) {
	if cwd == "" {
		return nil, errors.New("state: Create requires non-empty cwd")
	}
	header.Cwd = cwd
	header.Version = CurrentSchemaVersion
	if header.CreatedAt.IsZero() {
		header.CreatedAt = time.Now().UTC()
	}
	rootID, err := NewID(func(string) bool { return false })
	if err != nil {
		return nil, err
	}
	root := Entry{
		ID:        rootID,
		ParentID:  "",
		Kind:      KindSessionHeader,
		Timestamp: header.CreatedAt,
		Payload:   header,
	}
	return &lazyManager{
		cwd:    cwd,
		header: header,
		buffer: []Entry{root},
		leafID: rootID,
	}, nil
}

// Append buffers the entry (pre-flush) or delegates to bolt (post-flush).
// Pre-flush, an assistant Message payload triggers flushLocked.
func (m *lazyManager) Append(entry Entry) (string, error) {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return "", ErrManagerClosed
	}
	if m.bolt != nil {
		m.mu.Unlock()
		return m.bolt.Append(entry)
	}

	id, err := NewID(func(candidate string) bool {
		for _, e := range m.buffer {
			if e.ID == candidate {
				return true
			}
		}
		return false
	})
	if err != nil {
		m.mu.Unlock()
		return "", err
	}
	e := Entry{
		ID:        id,
		ParentID:  m.leafID,
		Kind:      kindOf(entry.Payload),
		Timestamp: time.Now().UTC(),
		Payload:   entry.Payload,
	}
	m.buffer = append(m.buffer, e)
	m.leafID = id

	// Flush on first assistant message. We hold the mutex across the
	// flush so concurrent Appends serialize.
	flushErr := error(nil)
	if isAssistantMessage(e) {
		flushErr = m.flushLocked()
	}
	m.mu.Unlock()
	if flushErr != nil {
		return "", flushErr
	}
	return id, nil
}

// flushLocked creates the .bolt file via CreateManager, then batch-writes
// the buffer (minus the synthetic root) re-parented onto the new bolt root.
// Called with m.mu held.
func (m *lazyManager) flushLocked() error {
	mgr, err := CreateManager(m.cwd, m.header)
	if err != nil {
		return err
	}
	bolt, ok := mgr.(*boltManager)
	if !ok {
		// CreateManager returns *boltManager today; this guard is
		// defensive against future refactors that swap the concrete
		// type. Surface as a typed error so the caller can diagnose.
		mgr.Close()
		return fmt.Errorf("state: lazy flush: CreateManager returned %T, want *boltManager", mgr)
	}

	// The freshly-created bolt SessionHeader's ID is the real root.
	rootID := bolt.leafID

	// The buffer starts with a synthetic SessionHeader root (materialized
	// at Create time). Skip it — CreateManager already wrote its own —
	// and re-parent any entries that reference the synthetic root onto
	// the bolt root.
	var batch []Entry
	if len(m.buffer) > 0 && m.buffer[0].Kind == KindSessionHeader {
		syntheticRootID := m.buffer[0].ID
		batch = make([]Entry, 0, len(m.buffer)-1)
		for _, e := range m.buffer[1:] {
			if e.ParentID == syntheticRootID {
				e.ParentID = rootID
			}
			batch = append(batch, e)
		}
	} else {
		// Defensive: buffer without a leading SessionHeader root
		// (legacy / manually constructed). Use the old re-parent logic.
		batch = make([]Entry, len(m.buffer))
		copy(batch, m.buffer)
		if len(batch) > 0 && batch[0].ParentID == "" {
			batch[0].ParentID = rootID
		}
	}

	if err := bolt.store.AppendBatch(batch); err != nil {
		bolt.Close()
		return fmt.Errorf("state: lazy flush: batch write: %w", err)
	}

	// Translate leaf: the synthetic root's ID doesn't exist in the bolt
	// store. If m.leafID still points at the synthetic root (no Append
	// or AppendAt has advanced it past the root), use the bolt root's ID.
	// AppendAt does NOT advance m.leafID, so also check for "" (legacy).
	leafForBolt := m.leafID
	if len(m.buffer) > 0 && m.buffer[0].Kind == KindSessionHeader && leafForBolt == m.buffer[0].ID {
		leafForBolt = rootID
	} else if leafForBolt == "" {
		if len(batch) > 0 {
			leafForBolt = batch[len(batch)-1].ID
		} else {
			leafForBolt = rootID
		}
	}
	if err := bolt.store.SetLeaf(leafForBolt); err != nil {
		bolt.Close()
		return fmt.Errorf("state: lazy flush: set leaf: %w", err)
	}
	bolt.leafID = leafForBolt
	m.leafID = leafForBolt // keep lazy leaf in sync with bolt leaf

	m.bolt = bolt
	m.buffer = nil
	return nil
}

// isAssistantMessage reports whether e is a Message entry whose Role is
// "assistant". Used to decide flush trigger.
func isAssistantMessage(e Entry) bool {
	if e.Kind != KindMessage {
		return false
	}
	mp, ok := e.Payload.(MessagePayload)
	if !ok {
		return false
	}
	return mp.Role == llm.RoleAssistant
}

// AppendAt writes entry as a child of parentID without advancing the
// leaf pointer. Pre-flush, the entry goes to the buffer; an assistant
// Message payload triggers flushLocked (which batch-writes the buffer to
// the new .bolt file without advancing the leaf). Post-flush, this
// delegates to boltManager.AppendAt.
//
// Returns ErrInvalidBranch when parentID is not in the buffer (pre-flush)
// or the store (post-flush).
func (m *lazyManager) AppendAt(entry Entry, parentID string) (string, error) {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return "", ErrManagerClosed
	}
	if m.bolt != nil {
		bolt := m.bolt
		m.mu.Unlock()
		return bolt.AppendAt(entry, parentID)
	}

	// Pre-flush: validate parentID in buffer.
	// Special case: when the buffer is empty (no root materialized yet —
	// e.g. MergeState integrating child entries into a parent that has
	// never had Append called directly), accept parentID == "" as
	// appending to the implicit root. The entry becomes the first entry
	// in the buffer's local tree; flushLocked will re-parent it onto the
	// bolt root when flush is triggered.
	if len(m.buffer) == 0 && parentID == "" {
		// Allow: the entry becomes the buffer's first entry.
	} else {
		found := false
		for _, e := range m.buffer {
			if e.ID == parentID {
				found = true
				break
			}
		}
		if !found {
			m.mu.Unlock()
			return "", fmt.Errorf("%w: parent %q not in tree", ErrInvalidBranch, parentID)
		}
	}

	id, err := NewID(func(candidate string) bool {
		for _, e := range m.buffer {
			if e.ID == candidate {
				return true
			}
		}
		return false
	})
	if err != nil {
		m.mu.Unlock()
		return "", err
	}
	e := Entry{
		ID:        id,
		ParentID:  parentID,
		Kind:      kindOf(entry.Payload),
		Timestamp: time.Now().UTC(),
		Payload:   entry.Payload,
	}
	m.buffer = append(m.buffer, e)
	// Intentionally do NOT advance m.leafID — that's the AppendAt contract.

	// Flush on first assistant message. flushLocked batch-writes the
	// buffer (which now includes this entry) to the new .bolt file, then
	// sets bolt.leafID = m.leafID. Because AppendAt did NOT advance
	// m.leafID, bolt's leaf stays at the prior leaf — which is correct.
	flushErr := error(nil)
	if isAssistantMessage(e) {
		flushErr = m.flushLocked()
	}
	m.mu.Unlock()
	if flushErr != nil {
		return "", flushErr
	}
	return id, nil
}

// Branch moves the leaf pointer to fromID. Pre-flush, the new leaf must
// be in the buffer.
func (m *lazyManager) Branch(fromID string) error {
	return m.SetLeaf(fromID)
}

// BranchWithSummary defers to the agent loop per the cross-phase contract.
func (m *lazyManager) BranchWithSummary(ctx context.Context, fromID string, client llm.LLMClient) (string, error) {
	m.mu.Lock()
	bolt := m.bolt
	m.mu.Unlock()
	if bolt != nil {
		return bolt.BranchWithSummary(ctx, fromID, client)
	}
	return "", ErrBranchWithSummaryUnsupported
}

// CreateBranchedSession extracts a path into a new session.
func (m *lazyManager) CreateBranchedSession(leafID string) (Manager, error) {
	m.mu.Lock()
	bolt := m.bolt
	m.mu.Unlock()
	if bolt != nil {
		return bolt.CreateBranchedSession(leafID)
	}
	// Pre-flush: build a Tree from the buffer and extract the path.
	tree, err := m.bufferTreeLocked()
	if err != nil {
		return nil, err
	}
	return createBranchedSessionFromTree(tree, leafID, m.cwd)
}

// ForkFrom opens sourcePath and forks it into a new session under cwd.
// Pre-flush we have nothing on disk yet, so this delegates to the source
// manager's CreateBranchedSession.
func (m *lazyManager) ForkFrom(sourcePath string) (Manager, error) {
	src, err := OpenManager(sourcePath, m.cwd)
	if err != nil {
		return nil, err
	}
	defer src.Close()
	return src.CreateBranchedSession(src.LeafID())
}

// ContinueRecent opens the most recently modified session under cwd.
func (m *lazyManager) ContinueRecent(cwd string) (Manager, error) {
	return (&boltManager{cwd: cwd}).ContinueRecent(cwd)
}

// InMemory returns an in-memory Manager.
func (m *lazyManager) InMemory(cwd string) Manager {
	return NewInMemoryManager(cwd)
}

// List returns SessionInfo for sessions under cwd.
func (m *lazyManager) List(cwd string) ([]SessionInfo, error) {
	return (&boltManager{cwd: cwd}).List(cwd)
}

// ListAll returns SessionInfo for sessions across all cwds.
func (m *lazyManager) ListAll() ([]SessionInfo, error) {
	return (&boltManager{cwd: m.cwd}).ListAll()
}

// BuildContext walks leaf → root per spec.
func (m *lazyManager) BuildContext(ctx context.Context) (Context, error) {
	m.mu.Lock()
	bolt := m.bolt
	leaf := m.leafID
	buffer := append([]Entry(nil), m.buffer...)
	m.mu.Unlock()
	if bolt != nil {
		return bolt.BuildContext(ctx)
	}
	if leaf == "" {
		return Context{}, nil
	}
	tree, err := NewTree(buffer)
	if err != nil {
		return Context{}, err
	}
	return buildContextFromTree(tree, leaf)
}

// LeafID returns the current leaf entry ID. Post-flush this delegates to
// the boltManager so the cached value stays in sync with the store.
func (m *lazyManager) LeafID() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.bolt != nil {
		return m.bolt.leafID
	}
	return m.leafID
}

// SetLeaf moves the leaf pointer to id. Pre-flush, id must be in the
// buffer; post-flush, id must be in the store.
func (m *lazyManager) SetLeaf(id string) error {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return ErrManagerClosed
	}
	if m.bolt != nil {
		bolt := m.bolt
		m.mu.Unlock()
		return bolt.SetLeaf(id)
	}
	// Pre-flush: scan buffer.
	for _, e := range m.buffer {
		if e.ID == id {
			m.leafID = id
			m.mu.Unlock()
			return nil
		}
	}
	m.mu.Unlock()
	return fmt.Errorf("%w: %q not in tree", ErrInvalidBranch, id)
}

// Tree returns a Tree over the buffer (pre-flush) or store (post-flush).
func (m *lazyManager) Tree() (*Tree, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.bolt != nil {
		return m.bolt.Tree()
	}
	return m.bufferTreeLocked()
}

// bufferTreeLocked builds a Tree from m.buffer + a synthetic root derived
// from m.header. Called with m.mu held.
//
// The synthetic root is necessary because the buffer's first entry has
// ParentID="" (it would become the buffer's local root). Tree requires
// exactly one such entry. If the buffer is empty, an empty Tree is
// returned (which NewTree rejects — callers must handle the empty case).
func (m *lazyManager) bufferTreeLocked() (*Tree, error) {
	if len(m.buffer) == 0 {
		return nil, fmt.Errorf("%w: lazy manager has no entries yet", ErrTreeInvalid)
	}
	return NewTree(m.buffer)
}

// Close releases resources. Pre-flush, this is a no-op (no backing file).
// Post-flush, it closes the underlying boltManager.
func (m *lazyManager) Close() error {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return nil
	}
	m.closed = true
	bolt := m.bolt
	m.mu.Unlock()
	if bolt != nil {
		return bolt.Close()
	}
	return nil
}

// createBranchedSessionFromTree is a helper used by lazyManager's
// CreateBranchedSession pre-flush path. It extracts path root → leafID
// into a new boltManager (created via CreateManager).
func createBranchedSessionFromTree(tree *Tree, leafID, cwd string) (Manager, error) {
	path, err := tree.Path(leafID)
	if err != nil {
		return nil, err
	}
	if len(path) == 0 || path[0].Kind != KindSessionHeader {
		// Pre-flush the buffer may not include a SessionHeader; in that
		// case synthesize a header from the first entry's data.
		return nil, fmt.Errorf("%w: cannot branch: path does not start with SessionHeader", ErrInvalidBranch)
	}
	origHeader, _ := path[0].Payload.(SessionHeaderPayload)
	newHeader := SessionHeaderPayload{
		Cwd:           cwd,
		Model:         origHeader.Model,
		Provider:      origHeader.Provider,
		Version:       CurrentSchemaVersion,
		ParentSession: origHeader.SessionID,
		CreatedAt:     time.Now().UTC(),
	}
	child, err := CreateManager(cwd, newHeader)
	if err != nil {
		return nil, err
	}
	for _, e := range path[1:] {
		if _, err := child.Append(Entry{Kind: e.Kind, Payload: e.Payload}); err != nil {
			child.Close()
			return nil, err
		}
	}
	return child, nil
}
