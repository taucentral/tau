package compaction

import (
	"fmt"

	"github.com/taucentral/tau/internal/state"
)

// SlidingResult describes the Compaction entry written by Archive.
type SlidingResult struct {
	// CompactionEntryID is the ID of the newly-appended Compaction entry.
	CompactionEntryID string
	// FirstKeptEntryID is the oldest entry that remains in the LLM context
	// after this compaction. Stored on the Compaction entry's payload.
	FirstKeptEntryID string
	// ArchivedCount is the number of entries in walk older than the cut
	// (i.e., entries that are now excluded from context).
	ArchivedCount int
}

// Archive writes a Compaction entry onto mgr that records summary and the
// oldest-kept boundary. The Compaction entry becomes the new leaf; future
// BuildContext calls honor FirstKeptEntryID by excluding older entries from
// the assembled context (see state.buildContextFromTree).
//
// walk is the leaf → root walk used by FindCutPoint; cutIdx is the index in
// walk of the oldest entry to keep. Entries walk[cutIdx+1:] are considered
// archived (they remain in the tree for /tree and /checkout but are excluded
// from BuildContext output).
//
// Archive does not delete or rewrite anything: archived entries stay in the
// state tree per spec scenario "Archived entries still visible in tree".
func Archive(mgr state.Manager, walk []state.Entry, cutIdx int, summary string) (SlidingResult, error) {
	if cutIdx < 0 || cutIdx >= len(walk) {
		return SlidingResult{}, fmt.Errorf("compaction: invalid cut index %d (walk len %d)", cutIdx, len(walk))
	}
	firstKeptID := walk[cutIdx].ID
	payload := state.CompactionPayload{
		Summary:          summary,
		FirstKeptEntryID: firstKeptID,
	}
	id, err := mgr.Append(state.Entry{Kind: state.KindCompaction, Payload: payload})
	if err != nil {
		return SlidingResult{}, fmt.Errorf("compaction: append Compaction entry: %w", err)
	}
	return SlidingResult{
		CompactionEntryID: id,
		FirstKeptEntryID:  firstKeptID,
		ArchivedCount:     len(walk) - cutIdx - 1,
	}, nil
}
