package compaction

import (
	"context"
	"fmt"
	"log"

	"github.com/coevin/tau/internal/llm/tokencounter"
	"github.com/coevin/tau/internal/state"
)

// DefaultReserveTokens is the per-turn token reserve applied when
// Settings.Compaction.ReserveTokens is nil. Matches the compaction spec
// ("reserveTokens defaults to 8192").
const DefaultReserveTokens = 8192

// DefaultKeepRecentTokens is the floor on the kept region applied when
// Settings.Compaction.KeepRecentTokens is nil. Matches pi's
// keepRecentTokens default at
// third-party/pi/packages/coding-agent/src/core/settings-manager.ts:771-781
// and the compaction spec ("keepRecentTokens defaults to 20000").
const DefaultKeepRecentTokens = 20000

// Result describes the outcome of a MaybeCompact call.
type Result struct {
	// Compacted is true when the pipeline ran and wrote a Compaction entry.
	// False means the context was within budget and nothing was written.
	Compacted bool `json:"compacted"`
	// Reason is a short human-readable string explaining why compaction ran
	// or was skipped. Examples: "below threshold", "compacted N entries".
	Reason string `json:"reason,omitempty"`
	// ArchivedCount is the number of entries excluded from future contexts.
	ArchivedCount int `json:"archivedCount,omitempty"`
	// CompactionEntryID is the ID of the new Compaction entry, or "" when
	// Compacted is false.
	CompactionEntryID string `json:"compactionEntryId,omitempty"`
	// FirstKeptEntryID is the oldest entry that remains in context, or ""
	// when Compacted is false.
	FirstKeptEntryID string `json:"firstKeptEntryId,omitempty"`
	// Summary is the structured-summary text stored on the Compaction entry.
	// Empty when Compacted is false.
	Summary string `json:"summary,omitempty"`
	// PreCompactionTokens is the BPE-accurate token count of the context
	// before compaction ran. Always populated.
	PreCompactionTokens int `json:"preCompactionTokens"`
	// PostCompactionTokens is the BPE-accurate token count of the context
	// after compaction (or unchanged when skipped).
	PostCompactionTokens int `json:"postCompactionTokens"`
}

// Compactor orchestrates the compaction pipeline per spec. The zero value
// is not usable; construct via NewCompactor.
type Compactor struct {
	Counter       tokencounter.TokenCounter
	ReserveTokens int
	// keepRecent is the floor on the kept region: FindCutPoint walks
	// backward from the leaf and returns the closest eligible cut at or
	// before the position where the BPE-accurate accumulator crosses
	// keepRecent. Matches pi's keepRecentTokens; see cutpoint.go.
	keepRecent    int
	Protection    ProtectionConfig
	Summarizer    *Summarizer
}

// NewCompactor returns a Compactor with the standard defaults applied.
//
// reserveTokens of zero means DefaultReserveTokens. keepRecentTokens of
// zero means DefaultKeepRecentTokens. keepRecentTokens is clamped to
// min(keepRecentTokens, contextWindow/2) — a floor exceeding half the
// context window would let the kept region dominate the budget and push
// the next turn over the model's hard limit. The clamp prevents a
// misconfigured KeepRecentTokens (e.g., 500000 set when the field was a
// no-op) from effectively disabling compaction.
//
// contextWindow SHOULD reflect the active model's true budget. Callers
// MAY pass 0 when they want no clamping (treated as +∞); this is the
// path used by tests that construct a Compactor without a real model.
func NewCompactor(counter tokencounter.TokenCounter, summarizer *Summarizer, reserveTokens, keepRecentTokens, contextWindow int) *Compactor {
	if reserveTokens <= 0 {
		reserveTokens = DefaultReserveTokens
	}
	if keepRecentTokens <= 0 {
		keepRecentTokens = DefaultKeepRecentTokens
	}
	if contextWindow > 0 {
		half := contextWindow / 2
		if half > 0 && keepRecentTokens > half {
			log.Printf("compaction: clamped keepRecentTokens from %d to %d (contextWindow/2)", keepRecentTokens, half)
			keepRecentTokens = half
		}
	}
	return &Compactor{
		Counter:       counter,
		ReserveTokens: reserveTokens,
		keepRecent:    keepRecentTokens,
		Protection:    ProtectionConfig{},
		Summarizer:    summarizer,
	}
}

// MaybeCompact runs the compaction pipeline if and only if the current
// context token count exceeds (contextWindow - ReserveTokens). model and
// contextWindow describe the active model's budget. The method is safe to
// call every turn; below-threshold is a cheap no-op.
//
// The pipeline:
//  1. Build leaf → root walk via mgr.Tree().
//  2. Compute total tokens. Skip if under threshold.
//  3. Build ProtectionList.
//  4. FindCutPoint.
//  5. Extract FileTracking from the would-be-archived region.
//  6. Summarize via s.Summarizer (uses previous Compaction entry if any).
//  7. Archive via sliding.Archive.
func (c *Compactor) MaybeCompact(
	ctx context.Context,
	mgr state.Manager,
	model string,
	contextWindow int,
) (Result, error) {
	if c.Counter == nil {
		return Result{}, fmt.Errorf("compaction: Counter is nil")
	}
	if mgr == nil {
		return Result{}, fmt.Errorf("compaction: manager is nil")
	}

	leafID := mgr.LeafID()
	if leafID == "" {
		return Result{Reason: "empty session"}, nil
	}

	tree, err := mgr.Tree()
	if err != nil {
		return Result{}, fmt.Errorf("compaction: load tree: %w", err)
	}
	walk, err := tree.WalkFromLeaf(leafID)
	if err != nil {
		return Result{}, fmt.Errorf("compaction: walk tree: %w", err)
	}
	if len(walk) == 0 {
		return Result{Reason: "empty walk"}, nil
	}
	if len(walk) < 2 {
		// Only a root (e.g., a freshly-created session with just a
		// SessionHeader) → nothing to archive regardless of budget math.
		return Result{Reason: "nothing to compact"}, nil
	}

	totalTokens := 0
	for _, e := range walk {
		totalTokens += countEntryTokens(e, c.Counter, model)
	}

	threshold := contextWindow - c.ReserveTokens
	if totalTokens <= threshold {
		return Result{
			Reason:               "below threshold",
			PreCompactionTokens:  totalTokens,
			PostCompactionTokens: totalTokens,
		}, nil
	}

	if c.Summarizer == nil {
		return Result{}, fmt.Errorf("compaction: Summarizer is nil (required for above-threshold compaction)")
	}

	protected := BuildProtectionList(walk, c.Protection)
	// FindCutPoint uses keepRecent as the FLOOR on the kept region (matching
	// pi's keepRecentTokens), not as a ceiling. The trigger threshold above
	// already used ReserveTokens to decide whether to compact at all; the
	// floor is independent of that decision and only governs how much we
	// keep. See cutpoint.go and design.md D1.1 / D1.4.
	cutIdx := FindCutPoint(walk, protected, c.Counter, model, c.keepRecent)
	if cutIdx == NoCutNeeded {
		// FindCutPoint never returns NoCutNeeded in the current
		// implementation (it always picks at least walk[0]); retained
		// for forward compatibility. Treat as below-threshold.
		return Result{
			Reason:               "no eligible cut point",
			PreCompactionTokens:  totalTokens,
			PostCompactionTokens: totalTokens,
		}, nil
	}
	if cutIdx >= len(walk)-1 {
		// Cut is at the root — nothing would be archived. Skip.
		return Result{
			Reason:               "cut at root; nothing to archive",
			PreCompactionTokens:  totalTokens,
			PostCompactionTokens: totalTokens,
		}, nil
	}

	// archived region = walk[cutIdx+1:], in leaf→root order. Reverse for
	// chronological (root→leaf) order for the summarizer.
	archived := make([]state.Entry, 0, len(walk)-cutIdx-1)
	for i := len(walk) - 1; i > cutIdx; i-- {
		archived = append(archived, walk[i])
	}

	// File tracking covers the archived region (so the summary captures the
	// file operations being dropped from context).
	tracking := ExtractFileTracking(archived, c.Protection.FileReadTools, c.Protection.FileWriteTools)

	// Look up any previous Compaction entry so we use the update prompt.
	previousSummary := findPreviousSummary(walk, cutIdx)

	summary, err := c.Summarizer.Summarize(ctx, previousSummary, archived, tracking)
	if err != nil {
		return Result{}, fmt.Errorf("compaction: summarize: %w", err)
	}

	result, err := Archive(mgr, walk, cutIdx, summary)
	if err != nil {
		return Result{}, fmt.Errorf("compaction: archive: %w", err)
	}

	postTokens := 0
	for i := 0; i <= cutIdx; i++ {
		postTokens += countEntryTokens(walk[i], c.Counter, model)
	}
	// Include the new Compaction entry's tokens too (it contributes a
	// synthetic user message in BuildContext). Approximate by counting the
	// summary text.
	postTokens += c.Counter.Count(model, summary) + 4

	return Result{
		Compacted:            true,
		Reason:               fmt.Sprintf("compacted %d entries", result.ArchivedCount),
		ArchivedCount:        result.ArchivedCount,
		CompactionEntryID:    result.CompactionEntryID,
		FirstKeptEntryID:     result.FirstKeptEntryID,
		Summary:              summary,
		PreCompactionTokens:  totalTokens,
		PostCompactionTokens: postTokens,
	}, nil
}

// findPreviousSummary locates the most-recent Compaction entry anywhere in
// the walk (kept OR archived) and returns its summary text. Returns "" when
// no prior compaction exists.
//
// Per spec scenario "Iterative summary update", the LLM MUST receive the
// previous summary plus the new entries being archived. Even when the prior
// Compaction entry falls in the archived region, its summary is the
// information we want to carry forward — the new summary replaces the old
// one in the post-compaction context regardless of where the cut sits.
//
// cutIdx is the index of the last kept entry; it is taken as a hint only.
// The search scans the entire walk because the previous summary is the
// canonical carry-forward artifact whether the originating Compaction entry
// survives in the kept region or is archived. walk is leaf → root; the
// FIRST Compaction entry encountered scanning from index 0 is the most
// recent one.
func findPreviousSummary(walk []state.Entry, cutIdx int) string {
	_ = cutIdx // search whole walk per spec; see doc comment
	for i := 0; i < len(walk); i++ {
		if walk[i].Kind != state.KindCompaction {
			continue
		}
		cp, ok := walk[i].Payload.(state.CompactionPayload)
		if ok {
			return cp.Summary
		}
	}
	return ""
}
