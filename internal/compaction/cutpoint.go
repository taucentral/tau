package compaction

import (
	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/llm/tokencounter"
	"github.com/coevin/tau/internal/state"
)

// NoCutNeeded signals that the walk fits within the target budget and no
// compaction is required. FindCutPoint returns this as a sentinel index.
const NoCutNeeded = -1

// FindCutPoint walks backward from walk[0] (the leaf) toward walk[len-1]
// (the root), accumulating token counts, and returns the index of the
// OLDEST entry that remains in context. Entries older than the cut are
// archived by the sliding-window stage.
//
// Returns NoCutNeeded when the total token count of walk is already at or
// below targetKept.
//
// Algorithm (per spec "Cut-point detection"):
//
//  1. Compute the per-entry token count via counter.
//  2. If total <= targetKept, no compaction is needed.
//  3. Walk leaf → root, accumulating kept tokens. Stop adding entries as
//     soon as the next entry would push the total over targetKept.
//  4. Among the kept region, the deepest ELIGIBLE entry is the ideal cut.
//  5. Extend the cut to include any protected entries that fall in the
//     would-be archived region (protected entries cannot be archived).
//
// Eligibility (per spec): an entry is an eligible cut boundary iff it is a
// Message kind entry that is NOT a tool-result-only user message AND NOT a
// user message whose immediately-younger neighbor is an assistant message
// with ToolUse blocks. Non-message entries (SessionHeader, Label, etc.)
// are not eligible cut points but can still be included in the kept region.
//
// walk MUST be in leaf → root order (as produced by state.Tree.WalkFromLeaf).
func FindCutPoint(
	walk []state.Entry,
	protected ProtectionList,
	counter tokencounter.TokenCounter,
	model string,
	targetKept int,
) int {
	if len(walk) == 0 {
		return NoCutNeeded
	}
	tokens := make([]int, len(walk))
	total := 0
	for i, e := range walk {
		tokens[i] = countEntryTokens(e, counter, model)
		total += tokens[i]
	}
	if total <= targetKept {
		return NoCutNeeded
	}

	running := 0
	idealCut := -1
	for i, e := range walk {
		if running+tokens[i] > targetKept {
			break
		}
		running += tokens[i]
		if isEligibleCut(e, walk, i) {
			idealCut = i
		}
	}
	if idealCut < 0 {
		// Could not fit even the leaf in budget (small targetKept or a
		// very large leaf entry). Force-keep walk[0] so we don't archive
		// the entire tree; the next compaction will revisit.
		idealCut = 0
	}

	// Protected entries that fall in the would-be archived region must be
	// pulled into the kept region. Walk from idealCut forward (toward root)
	// and extend the cut to the deepest protected entry.
	cut := idealCut
	for j := idealCut + 1; j < len(walk); j++ {
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
