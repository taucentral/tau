package state

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/coevin/tau/internal/config"
	"github.com/coevin/tau/internal/llm"
)

// Context is the output of BuildContext: a list of messages ready to send
// to the LLM plus optional system prompt blocks. The Messages slice is in
// root → leaf order and has compaction/branch-summary synthetic messages
// prepended per the state-tree spec rules.
type Context struct {
	// Messages is the conversation history in root → leaf order.
	Messages []llm.Message
	// System is reserved for future system-prompt assembly; today it is
	// empty because the agent loop builds the system prompt separately.
	System []llm.ContentBlock
}

// SessionInfo describes a session on disk for List/ListAll displays.
type SessionInfo struct {
	// SessionID is the SessionHeader payload's SessionID, derived from the
	// file name when the header has not been read.
	SessionID string
	// Path is the absolute path to the .bolt file.
	Path string
	// Cwd is the working directory the session was created in.
	Cwd string
	// CreatedAt is the SessionHeader timestamp; zero if not yet read.
	CreatedAt time.Time
	// LastActive is the file's mtime; useful for "ContinueRecent" selection.
	LastActive time.Time
	// LeafID is the meta[leaf] value at scan time; empty if not set.
	LeafID string
	// ParentSession is the SessionHeader's ParentSession; empty for roots.
	ParentSession string
}

// Manager is the persistence and tree-management interface per state-tree
// spec "Manager interface". A Manager owns one session's tree and mediates
// all access. Implementations MUST be safe for concurrent use.
//
// The interface is large because it covers session-lifecycle (Create, Open,
// List), tree mutation (Append, Branch), and read-side (BuildContext, Tree)
// operations. A single implementation (boltManager) backs most calls; an
// in-memory implementation is provided for tests and ephemeral sessions.
type Manager interface {
	// Append writes a new entry as a child of the current leaf and
	// advances the leaf pointer to the new entry. Returns the assigned
	// ID. The caller supplies Kind and Payload; the Manager assigns ID,
	// ParentID, and Timestamp (any caller-supplied values are ignored).
	Append(entry Entry) (string, error)

	// Branch moves the leaf pointer to fromID without writing a new
	// entry. Future appends descend from fromID; the previously-active
	// path remains in the tree. Returns ErrInvalidBranch if fromID is
	// not in the tree.
	Branch(fromID string) error

	// BranchWithSummary summarizes the path from the current leaf back
	// to fromID using client, appends a BranchSummary entry as a child
	// of fromID, then calls Branch(fromID). Returns the new BranchSummary
	// entry's ID. Used by /branch when the user wants the abandoned
	// branch's context preserved as a summary.
	BranchWithSummary(ctx context.Context, fromID string, client llm.LLMClient) (string, error)

	// CreateBranchedSession extracts the path root → leafID into a new
	// session file with a fresh session ID, recording the parent via
	// the new header's ParentSession field.
	CreateBranchedSession(leafID string) (Manager, error)

	// ForkFrom opens the session at sourcePath and forks it into a new
	// session under cwd. The forked session inherits the parent's leaf
	// path.
	ForkFrom(sourcePath string) (Manager, error)

	// ContinueRecent opens the most recently active session for cwd, or
	// returns ErrNoSession when none exists.
	ContinueRecent(cwd string) (Manager, error)

	// InMemory returns an in-memory Manager (no persistence) rooted at
	// cwd. Used by tests and by ephemeral sessions that explicitly opt
	// out of persistence.
	InMemory(cwd string) Manager

	// List returns SessionInfo for every session under cwd (encoded).
	List(cwd string) ([]SessionInfo, error)

	// ListAll returns SessionInfo for every session across all cwds.
	ListAll() ([]SessionInfo, error)

	// BuildContext walks leaf → root per state-tree spec "BuildContext
	// walk", applying compaction and BranchSummary rules, and returns
	// the assembled Context.
	BuildContext(ctx context.Context) (Context, error)

	// LeafID returns the current leaf entry ID.
	LeafID() string

	// SetLeaf moves the leaf pointer to id. Returns ErrInvalidBranch
	// when id is not in the tree.
	SetLeaf(id string) error

	// Tree loads every entry and returns a validated Tree.
	Tree() (*Tree, error)

	// Close releases resources. Idempotent.
	Close() error
}

// ErrInvalidBranch is returned by Branch/SetLeaf when the requested entry
// ID is not in the tree.
var ErrInvalidBranch = errors.New("state: invalid branch target")

// ErrNoSession is returned by ContinueRecent when no session exists for the
// given cwd.
var ErrNoSession = errors.New("state: no session found")

// ErrManagerClosed is returned by any Manager method called after Close.
var ErrManagerClosed = errors.New("state: manager is closed")

// ErrBranchWithSummaryUnsupported is returned by BranchWithSummary on
// managers that have not been wired to an LLM client. The boltManager and
// inMemoryManager implementations both surface this; the agent loop in
// Phase 8 wraps the lower-level Branch + BranchSummary entry primitives to
// provide the real flow.
//
// Tests assert this sentinel so the cross-phase contract is documented.
var ErrBranchWithSummaryUnsupported = errors.New("state: BranchWithSummary not wired; the agent loop composes it from Branch + BranchSummary entries")

// sessionFileNameMangle separates the timestamp and uuid portions of a
// session .bolt filename so SessionInfo.CreatedAt and SessionID can be
// populated without reading the file. Returns ok=false when the name does
// not match the expected "<timestamp>_<uuid>.bolt" shape.
func sessionFileNameMangle(name string) (timestamp string, uuid string, ok bool) {
	base := strings.TrimSuffix(name, ".bolt")
	if base == name {
		return "", "", false
	}
	idx := strings.Index(base, "_")
	if idx <= 0 || idx == len(base)-1 {
		return "", "", false
	}
	return base[:idx], base[idx+1:], true
}

//--- shared helpers used by both boltManager and inMemoryManager ---

// listSessionsInDir scans dir for *.bolt files and returns SessionInfo for
// each, populating Path/Cwd/LastActive/SessionID from filename + fileinfo
// without opening the file. Used by List(cwd) and as the inner loop of
// ListAll.
func listSessionsInDir(dir, cwd string) ([]SessionInfo, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var out []SessionInfo
	for _, ent := range entries {
		name := ent.Name()
		if !strings.HasSuffix(name, ".bolt") {
			continue
		}
		full := filepath.Join(dir, name)
		info, err := ent.Info()
		if err != nil {
			continue
		}
		si := SessionInfo{
			Path:       full,
			Cwd:        cwd,
			LastActive: info.ModTime(),
		}
		if ts, uuid, ok := sessionFileNameMangle(name); ok {
			si.SessionID = uuid
			if parsed, err := time.Parse(sessionFileTimestampLayout, ts); err == nil {
				si.CreatedAt = parsed
			}
		}
		out = append(out, si)
	}
	return out, nil
}

// sessionFileTimestampLayout matches the layout used by newSessionFilePath.
// Uses an explicit numeric layout so RFC3330-style timestamps parse without
// locale drift.
const sessionFileTimestampLayout = "20060102T150405Z"

// sortSessionsByLastActiveDesc orders sessions newest-first by mtime.
// ContinueRecent picks the head; List displays use the same order.
func sortSessionsByLastActiveDesc(sessions []SessionInfo) {
	sort.SliceStable(sessions, func(i, j int) bool {
		return sessions[i].LastActive.After(sessions[j].LastActive)
	})
}

//--- shared helpers for file-based session discovery ---

// newSessionFilePath constructs "<dir>/<timestamp>_<uuid>.bolt" where
// timestamp is UTC in sessionFileTimestampLayout and uuid is 8 hex chars
// from crypto/rand. Used by Create and CreateBranchedSession.
func newSessionFilePath(dir string) (string, string, error) {
	id, err := NewID(func(string) bool { return false })
	if err != nil {
		return "", "", fmt.Errorf("state: generate session id: %w", err)
	}
	ts := time.Now().UTC().Format(sessionFileTimestampLayout)
	name := ts + "_" + id + ".bolt"
	return filepath.Join(dir, name), id, nil
}

// sessionsDirFor resolves the sessions directory for cwd, creating it if
// missing. Wraps config.SessionsDir with mkdir semantics.
func sessionsDirFor(cwd string) (string, error) {
	dir, err := config.SessionsDir(cwd)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		return "", fmt.Errorf("state: mkdir %s: %w", dir, err)
	}
	return dir, nil
}
