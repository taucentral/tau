package compaction

import (
	"sort"

	"github.com/taucentral/tau/internal/state"
)

// DefaultMaxRecentFileReads is the cap on distinct file paths whose most-recent
// read entry is protected from compaction. The cap is per session.
const DefaultMaxRecentFileReads = 5

// ProtectionConfig controls which entries the compaction pipeline must not
// archive.
type ProtectionConfig struct {
	// MaxRecentFileReads caps the number of distinct file paths whose
	// most-recent read entry is protected. Zero means DefaultMaxRecentFileReads.
	MaxRecentFileReads int

	// FileReadTools names the tools treated as file reads. Empty means
	// DefaultFileReadTools.
	FileReadTools []string

	// FileWriteTools names the tools treated as file modifications. Empty
	// means DefaultFileWriteTools.
	FileWriteTools []string
}

// ProtectionList is the set of entry IDs that MUST NOT be archived by the
// sliding-window stage. The cut-point walker extends the kept region to
// include every protected entry.
type ProtectionList struct {
	entries map[string]struct{}
}

// Contains reports whether id is protected.
func (p ProtectionList) Contains(id string) bool {
	_, ok := p.entries[id]
	return ok
}

// Len returns the number of protected entries.
func (p ProtectionList) Len() int {
	return len(p.entries)
}

// IDs returns the protected entry IDs in sorted order for deterministic output.
func (p ProtectionList) IDs() []string {
	out := make([]string, 0, len(p.entries))
	for id := range p.entries {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

// BuildProtectionList walks entries (any order) and marks entry IDs that
// MUST survive compaction:
//
//   - SessionInfo entries (kind: SessionInfo) — per spec.
//   - Label entries (kind: Label) — per spec.
//   - The most-recent read entry per distinct file path, capped at
//     cfg.MaxRecentFileReads (distinct paths selected newest-first).
//
// System prompt and AGENTS.md/CLAUDE.md are NOT entries in the state tree;
// they are prepended at request time by the agent loop and are therefore
// inherently immune to compaction. This function does not need to know
// about them.
func BuildProtectionList(entries []state.Entry, cfg ProtectionConfig) ProtectionList {
	maxRecent := cfg.MaxRecentFileReads
	if maxRecent <= 0 {
		maxRecent = DefaultMaxRecentFileReads
	}
	readTools := cfg.FileReadTools
	if len(readTools) == 0 {
		readTools = DefaultFileReadTools
	}
	writeTools := cfg.FileWriteTools
	if len(writeTools) == 0 {
		writeTools = DefaultFileWriteTools
	}

	protected := make(map[string]struct{})

	// SessionInfo and Label entries are always protected.
	for _, e := range entries {
		switch e.Kind {
		case state.KindSessionInfo, state.KindLabel:
			protected[e.ID] = struct{}{}
		case state.KindSessionHeader, state.KindMessage, state.KindThinkingLevelChange,
			state.KindModelChange, state.KindCompaction, state.KindBranchSummary,
			state.KindCustom, state.KindCustomMessage, state.KindClearMarker:
			// Other entry kinds are protected via file-tracking and the
			// cut-point logic; they're not unconditionally pinned here.
		}
	}

	// For file reads: take the most-recent ToolUse per distinct file path,
	// then keep the top-N paths by recency. The entry holding the ToolUse
	// is the one we protect.
	tracking := ExtractFileTracking(entries, readTools, writeTools)

	// Build per-path "most recent read" list. entries are already sorted
	// oldest-first; iterate in reverse to get newest-first per path.
	perPath := make(map[string]FileOperation)
	for _, op := range tracking.Reads {
		existing, ok := perPath[op.Path]
		if !ok || op.Timestamp.After(existing.Timestamp) {
			perPath[op.Path] = op
		}
	}

	// Order paths newest-first by their most-recent read timestamp.
	pathsByRecency := make([]string, 0, len(perPath))
	for path, op := range perPath {
		pathsByRecency = append(pathsByRecency, path)
		_ = op
	}
	sort.SliceStable(pathsByRecency, func(i, j int) bool {
		oi := perPath[pathsByRecency[i]]
		oj := perPath[pathsByRecency[j]]
		if oi.Timestamp.Equal(oj.Timestamp) {
			return oi.EntryID < oj.EntryID
		}
		return oi.Timestamp.After(oj.Timestamp)
	})

	// Cap at maxRecent and protect the holding entries.
	if len(pathsByRecency) > maxRecent {
		pathsByRecency = pathsByRecency[:maxRecent]
	}
	for _, path := range pathsByRecency {
		protected[perPath[path].EntryID] = struct{}{}
	}

	return ProtectionList{entries: protected}
}
