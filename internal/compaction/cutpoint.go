package compaction

import (
	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/llm/tokencounter"
	"github.com/coevin/tau/internal/state"
)

// NoCutNeeded signals that the walk fits within the target budget and no
// compaction is required. FindCutPoint returns this as a sentinel index.
const NoCutNeeded = -1

// FindCutPoint returns the index of the OLDEST entry that remains in
// context after compaction. Entries older than the cut are archived by the
// sliding-window stage.
//
// keepRecentTokens is the FLOOR on the kept region (matching pi's
// keepRecentTokens). The algorithm:
//
//  1. Pre-compute eligible cut points (ascending index = oldest first).
//  2. Default cutIndex = cutPoints[0] (the OLDEST eligible). This is the
//     force-keep minimum when the floor is never crossed: compaction
//     occurs only at the natural session boundary.
//  3. Walk BACKWARD from leaf (walk[0]) toward root (walk[N-1]),
//     accumulating BPE-accurate tokens.
//  4. When `accumulated >= keepRecentTokens`, find the closest eligible
//     cut point at-or-newer than the crossing entry (i.e., the smallest
//     cutPoint index `c` such that `c >= i`) and use that as the cut.
//  5. After the floor cut: extend to include any protected entries that
//     fall in the would-be archived region (protected entries cannot be
//     archived).
//
// Floor enforcement and protected-entry extension both pull entries from
// the archived region into the kept region. They compose; both effects
// apply.
//
// Reference: third-party/pi/packages/coding-agent/src/core/compaction/compaction.ts:387-449
//
// Eligibility (per spec): an entry is an eligible cut boundary iff it is
// a Message kind entry that is NOT a tool-result-only user message AND
// NOT a user message whose immediately-younger neighbor is an assistant
// message with ToolUse blocks. Non-message entries (SessionHeader, Label,
// etc.) are not eligible cut points but can still be included in the kept
// region.
//
// walk MUST be in leaf → root order (as produced by state.Tree.WalkFromLeaf).
// keepRecentTokens is the FLOOR; ReserveTokens (the trigger threshold) is
// handled by MaybeCompact and does NOT influence this function (design.md
// D1.4).
func FindCutPoint(
	walk []state.Entry,
	protected ProtectionList,
	counter tokencounter.TokenCounter,
	model string,
	keepRecentTokens int,
) int {
	if len(walk) == 0 {
		return NoCutNeeded
	}
	tokens := make([]int, len(walk))
	for i, e := range walk {
		tokens[i] = countEntryTokens(e, counter, model)
	}

	// Pre-compute eligible cut points in ascending index order (oldest
	// first, since walk[N-1] is the root). Mirrors pi's findValidCutPoints
	// at compaction.ts:300-338.
	cutPoints := make([]int, 0, len(walk))
	for i := range walk {
		if isEligibleCut(walk[i], walk, i) {
			cutPoints = append(cutPoints, i)
		}
	}
	if len(cutPoints) == 0 {
		// No eligible cut anywhere. Force-keep walk[0] (the leaf) so we
		// don't archive the entire tree; the next compaction will revisit.
		// This matches pi's `cutPoints.length === 0` branch at
		// compaction.ts:395-397.
		return 0
	}

	// Default: keep from the OLDEST eligible cut. In tau's walk layout,
	// walk[0] is the leaf (newest) and walk[N-1] is the root (oldest), so
	// the OLDEST eligible has the HIGHEST index — cutPoints[len-1]. (Pi's
	// array layout is the reverse: pi's cutPoints[0] is the oldest. See
	// compaction.ts:401.)
	//
	// This default is the force-keep minimum when the floor is never
	// crossed — compaction occurs only at the natural session boundary.
	cutIndex := cutPoints[len(cutPoints)-1]

	// Walk forward in array index (i=0→N-1), which is leaf → root (newest
	// → oldest) in walk order, accumulating tokens. When the floor is
	// crossed, find the closest eligible cut at-or-newer than the crossing
	// entry. In tau's layout, "newer than i" = "closer to leaf" = "index
	// <= i". So we want the LARGEST cutPoint <= i. (Pi walks backward in
	// its layout where entries[startIndex] is oldest; tau's walk direction
	// is the reverse, so the comparison reverses too. See compaction.ts:403-422.)
	running := 0
	for i := 0; i < len(walk); i++ {
		running += tokens[i]
		if running < keepRecentTokens {
			continue
		}
		// Floor crossed at index i. Find the largest cutPoint <= i.
		for j := len(cutPoints) - 1; j >= 0; j-- {
			if cutPoints[j] <= i {
				cutIndex = cutPoints[j]
				break
			}
		}
		break
	}

	// Protected entries that fall in the would-be archived region must be
	// pulled into the kept region. Walk from cutIndex forward (toward root)
	// and extend the cut to the deepest protected entry.
	cut := cutIndex
	for j := cutIndex + 1; j < len(walk); j++ {
		if protected.Contains(walk[j].ID) {
			cut = j
		}
	}
	return cut
}

// countEntryTokens returns the BPE token count of an entry's conversational
// representation. Non-message entries contribute a small constant so their
// presence is accounted for without overstating cost.
func countEntryTokens(e state.Entry, counter tokencounter.TokenCounter, model string) int {
	if e.Kind == state.KindMessage {
		mp, ok := e.Payload.(state.MessagePayload)
		if !ok {
			return 1
		}
		return counter.CountMessages(model, []llm.Message{mp.AsMessage()})
	}
	if e.Kind == state.KindCustomMessage {
		// Custom messages appear in the LLM context; count their raw content.
		cmp, ok := e.Payload.(state.CustomMessagePayload)
		if !ok {
			return 1
		}
		return counter.Count(model, string(cmp.Content)) + 4
	}
	// SessionHeader, Compaction, BranchSummary, Label, SessionInfo, Custom,
	// ThinkingLevelChange, ModelChange do not appear in the LLM context.
	// Count them as zero so they don't influence the budget.
	return 0
}

// isEligibleCut reports whether walk[idx] can serve as the OLDEST KEPT entry
// of a compaction cut. See FindCutPoint for the eligibility rules.
//
// "walk" is the leaf → root walk; idx-1 is the next-younger entry (closer
// to the leaf). The "following assistant message" in spec terms is the
// next-younger neighbor because that's the entry that chronologically
// follows walk[idx].
func isEligibleCut(e state.Entry, walk []state.Entry, idx int) bool {
	if e.Kind != state.KindMessage {
		return false
	}
	mp, ok := e.Payload.(state.MessagePayload)
	if !ok {
		return false
	}
	switch mp.Role {
	case llm.RoleSystem, llm.RoleTool:
		// System prompts and tool-role messages are never valid cut points
		// (system belongs to the always-kept prefix; tool messages are
		// handled indirectly via isToolResultMessage on user-role entries).
		return false
	case llm.RoleUser:
		// Tool-result-only user messages are not eligible (cutting here
		// would orphan the matching ToolUse that's newer in the walk).
		if isToolResultMessage(mp) {
			return false
		}
		// "User message that precedes a ToolUse in the following assistant
		// message": the following message is walk[idx-1] (younger). If it's
		// an assistant message with ToolUse blocks, this user message is
		// not eligible.
		if idx > 0 {
			younger := walk[idx-1]
			if younger.Kind == state.KindMessage {
				ymp, ok := younger.Payload.(state.MessagePayload)
				if ok && ymp.Role == llm.RoleAssistant && hasToolUse(ymp) {
					return false
				}
			}
		}
		return true
	case llm.RoleAssistant:
		return true
	default:
		// system/tool roles: not normal cut points.
		return false
	}
}

// isToolResultMessage reports whether mp is a user-role message consisting
// solely of ToolResult blocks (i.e., a tool-call response).
func isToolResultMessage(mp state.MessagePayload) bool {
	if mp.Role != llm.RoleUser || len(mp.Content) == 0 {
		return false
	}
	for _, b := range mp.Content {
		if _, ok := b.(llm.ToolResult); !ok {
			return false
		}
	}
	return true
}

// hasToolUse reports whether mp carries any ToolUse block.
func hasToolUse(mp state.MessagePayload) bool {
	for _, b := range mp.Content {
		if _, ok := b.(llm.ToolUse); ok {
			return true
		}
	}
	return false
}
