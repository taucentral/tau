package state

import (
	"errors"
	"fmt"
	"sort"
)

// Tree is an in-memory index over a session's entries, built once from a
// flat entry slice (typically loaded from bbolt at Open time). It validates
// structural invariants at construction: exactly one root, no cycles, every
// non-root ParentID resolves to an existing entry.
//
// The index is read-only — mutations go through Manager.Append which
// rewrites the store and produces a new Tree on next read. This keeps the
// Tree safe for concurrent readers (e.g., a long-running BuildContext walk
// is not blocked by an append that produces a new Tree for later reads).
type Tree struct {
	entries  map[string]Entry    // ID → Entry
	children map[string][]string // ParentID → child IDs in insertion order
	rootID   string              // entry ID with empty ParentID
}

// ErrTreeInvalid wraps any structural validation failure so callers can
// distinguish corruption from a simple missing-entry lookup.
var ErrTreeInvalid = errors.New("state: invalid tree")

// NewTree builds a Tree from a flat entry slice. It returns a wrapped
// ErrTreeInvalid when:
//   - the slice is empty,
//   - no entry has an empty ParentID (no root),
//   - more than one entry has an empty ParentID (multiple roots),
//   - any non-root entry's ParentID does not resolve to another entry,
//   - following ParentID backward from any entry cycles without reaching root.
//
// NewTree accepts entries in any order; the children index is built in
// insertion order so /tree displays append-ordered siblings.
func NewTree(entries []Entry) (*Tree, error) {
	if len(entries) == 0 {
		return nil, fmt.Errorf("%w: empty entry set", ErrTreeInvalid)
	}
	t := &Tree{
		entries:  make(map[string]Entry, len(entries)),
		children: make(map[string][]string, len(entries)),
	}
	// Detect duplicate IDs and record the root candidate.
	dup := make(map[string]struct{}, len(entries))
	var roots []string
	for _, e := range entries {
		if _, ok := dup[e.ID]; ok {
			return nil, fmt.Errorf("%w: duplicate entry ID %q", ErrTreeInvalid, e.ID)
		}
		dup[e.ID] = struct{}{}
		t.entries[e.ID] = e
		if e.ParentID == "" {
			roots = append(roots, e.ID)
		}
	}
	switch len(roots) {
	case 0:
		return nil, fmt.Errorf("%w: no root entry (all entries have a non-empty ParentID)", ErrTreeInvalid)
	case 1:
		t.rootID = roots[0]
	default:
		return nil, fmt.Errorf("%w: multiple roots (%v)", ErrTreeInvalid, roots)
	}
	// Build children index in insertion order and verify every non-root
	// ParentID resolves.
	for _, e := range entries {
		if e.ParentID == "" {
			continue
		}
		if _, ok := t.entries[e.ParentID]; !ok {
			return nil, fmt.Errorf("%w: entry %q references missing parent %q", ErrTreeInvalid, e.ID, e.ParentID)
		}
		t.children[e.ParentID] = append(t.children[e.ParentID], e.ID)
	}
	// Cycle detection: walk backward from every entry to root. Any revisit
	// before reaching root means a cycle.
	for _, e := range entries {
		if err := t.assertReachesRoot(e.ID); err != nil {
			return nil, err
		}
	}
	return t, nil
}

// assertReachesRoot follows ParentID from id up to root. It returns a wrapped
// ErrTreeInvalid if a cycle is detected or the walk leaves the tree.
func (t *Tree) assertReachesRoot(id string) error {
	visited := make(map[string]struct{})
	cur := id
	for {
		e, ok := t.entries[cur]
		if !ok {
			return fmt.Errorf("%w: walk from %q left the tree at %q", ErrTreeInvalid, id, cur)
		}
		if e.ParentID == "" {
			// Reached root.
			return nil
		}
		if _, seen := visited[cur]; seen {
			return fmt.Errorf("%w: cycle detected walking from %q (revisited %q)", ErrTreeInvalid, id, cur)
		}
		visited[cur] = struct{}{}
		cur = e.ParentID
	}
}

// Root returns the root entry (the one entry with empty ParentID).
func (t *Tree) Root() Entry {
	return t.entries[t.rootID]
}

// RootID returns the root entry's ID.
func (t *Tree) RootID() string {
	return t.rootID
}

// Get returns the entry with the given ID and a found flag.
func (t *Tree) Get(id string) (Entry, bool) {
	e, ok := t.entries[id]
	return e, ok
}

// Len returns the number of entries in the tree.
func (t *Tree) Len() int {
	return len(t.entries)
}

// IDs returns all entry IDs in stable ascending order. Useful for collision
// lookups during NewID generation and for tests asserting exact membership.
func (t *Tree) IDs() []string {
	out := make([]string, 0, len(t.entries))
	for id := range t.entries {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

// Children returns the entries whose ParentID is the given id, in insertion
// order. Returns nil if id has no children. The caller can sort the result
// by Timestamp if display order requires it.
func (t *Tree) Children(id string) []Entry {
	ids := t.children[id]
	if len(ids) == 0 {
		return nil
	}
	out := make([]Entry, 0, len(ids))
	for _, cid := range ids {
		out = append(out, t.entries[cid])
	}
	return out
}

// WalkFromLeaf returns entries from leaf to root inclusive, in walk order
// (leaf first, root last). It is the natural direction for BuildContext
// which collects until it hits a compaction boundary or the root.
//
// Returns a wrapped ErrTreeInvalid if leafID is unknown or the parent chain
// is broken (should not happen for trees built via NewTree which already
// validates this).
func (t *Tree) WalkFromLeaf(leafID string) ([]Entry, error) {
	e, ok := t.entries[leafID]
	if !ok {
		return nil, fmt.Errorf("%w: leaf %q not in tree", ErrTreeInvalid, leafID)
	}
	out := []Entry{e}
	cur := e.ParentID
	for cur != "" {
		parent, ok := t.entries[cur]
		if !ok {
			return nil, fmt.Errorf("%w: parent %q missing during walk from %q", ErrTreeInvalid, cur, leafID)
		}
		out = append(out, parent)
		cur = parent.ParentID
	}
	return out, nil
}

// Path returns entries from root to leaf inclusive, in display order (root
// first, leaf last). It is the reverse of WalkFromLeaf. Used by
// CreateBranchedSession to extract a linear subsequence.
func (t *Tree) Path(leafID string) ([]Entry, error) {
	walk, err := t.WalkFromLeaf(leafID)
	if err != nil {
		return nil, err
	}
	// Reverse in place.
	for i, j := 0, len(walk)-1; i < j; i, j = i+1, j-1 {
		walk[i], walk[j] = walk[j], walk[i]
	}
	return walk, nil
}

// Ancestors returns the IDs of all entries strictly between id (exclusive)
// and root (inclusive) in walk order (id's parent first, root last). Used by
// compaction tests that need to enumerate "everything older than X".
func (t *Tree) Ancestors(id string) ([]string, error) {
	e, ok := t.entries[id]
	if !ok {
		return nil, fmt.Errorf("%w: %q not in tree", ErrTreeInvalid, id)
	}
	var out []string
	cur := e.ParentID
	for cur != "" {
		out = append(out, cur)
		parent, ok := t.entries[cur]
		if !ok {
			return nil, fmt.Errorf("%w: parent %q missing during walk from %q", ErrTreeInvalid, cur, id)
		}
		cur = parent.ParentID
	}
	return out, nil
}
