package state

import (
	"fmt"
	"strings"

	"github.com/taucentral/tau/internal/llm"
)

// buildContext is the bbolt-backed entry point: it loads every entry from
// the store, builds a Tree, then delegates to buildContextFromTree. The
// separation lets inMemoryManager share the algorithm without going
// through bbolt.
func buildContext(s *Store, leafID string) (Context, error) {
	entries, err := s.All()
	if err != nil {
		return Context{}, err
	}
	tree, err := NewTree(entries)
	if err != nil {
		return Context{}, err
	}
	return buildContextFromTree(tree, leafID)
}

// buildContextFromTree implements the state-tree spec "BuildContext walk":
//
//  1. Walk leaf → root via Tree.WalkFromLeaf.
//  2. ClearMarker cutoff: if a ClearMarker entry is encountered, truncate
//     the walk to entries newer than it. The marker is a hard barrier;
//     nothing older survives. (selectKept applies this before any other
//     cutoff.)
//  3. If a Compaction entry is encountered, keep entries from leaf down
//     to and including FirstKeptEntryID; drop everything older. The
//     Compaction entry itself contributes its Summary as a synthetic user
//     message and is NOT included as itself.
//  4. BranchSummary entries are stripped from the output and their Summary
//     text injected as synthetic user messages.
//  5. SessionHeader, ThinkingLevelChange, ModelChange, Label, SessionInfo,
//     and Custom kinds are skipped (not part of LLM context).
//  6. Message and CustomMessage entries convert to llm.Message.
//  7. The result is reversed to root → leaf order.
//  8. Synthetic messages (compaction summary, then branch summaries) are
//     prepended so the LLM sees them first.
//  9. Pair integrity: if any ToolResult block in the kept region
//     references a ToolUse that is NOT in the kept region, the walk
//     extends backward to include the matching ToolUse's assistant
//     message (state-tree spec scenario "Never split user/tool pairs").
func buildContextFromTree(tree *Tree, leafID string) (Context, error) {
	walk, err := tree.WalkFromLeaf(leafID)
	if err != nil {
		return Context{}, err
	}

	kept, summaryText, branchSummaries, err := selectKept(walk)
	if err != nil {
		return Context{}, err
	}

	// Reverse to root → leaf order for output.
	for i, j := 0, len(kept)-1; i < j; i, j = i+1, j-1 {
		kept[i], kept[j] = kept[j], kept[i]
	}

	msgs := entriesToMessages(kept)
	msgs = prependSynthetic(msgs, summaryText, branchSummaries)

	return Context{Messages: msgs}, nil
}

// selectKept applies the ClearMarker cutoff, compaction cutoff, and
// pair-integrity rule to the leaf → root walk. Returns:
//   - kept: the entries to include (leaf → root order, NOT reversed)
//   - compactionSummary: text for the synthetic message, "" if no compaction
//   - branchSummaries: collected BranchSummary text (in leaf→root order)
//
// Cutoff order:
//  1. ClearMarker (hard barrier; nothing older survives).
//  2. Compaction (kept region is leaf..FirstKeptEntryID).
//  3. Pair integrity extends kept backward for orphaned ToolResults.
func selectKept(walk []Entry) (kept []Entry, compactionSummary string, branchSummaries []string, err error) {
	// ClearMarker cutoff: hard barrier. The marker is appended by /clear
	// as a child of the then-current leaf. When walking leaf→root, the
	// first ClearMarker encountered (if any) is the most-recent one;
	// entries older than it are invisible to the model. Truncating walk
	// here means the rest of the function (Compaction scan, BranchSummary
	// strip, extendForToolPairs) operates on the post-clear slice.
	clearMarkerIdx := -1
	for i, e := range walk {
		if e.Kind == KindClearMarker {
			clearMarkerIdx = i
			break
		}
	}
	if clearMarkerIdx >= 0 {
		walk = walk[:clearMarkerIdx]
	}

	// Locate the most-recent compaction entry in the walk.
	compactionIdx := -1
	var compactionPayload CompactionPayload
	for i, e := range walk {
		if e.Kind == KindCompaction {
			compactionIdx = i
			compactionPayload, _ = e.Payload.(CompactionPayload)
			break
		}
	}

	if compactionIdx < 0 {
		// No compaction: keep everything.
		kept = walk
	} else {
		// Find FirstKeptEntryID's position in the walk (it's older than
		// the Compaction entry, so search walk[compactionIdx+1:]).
		cutoff := -1
		for j := compactionIdx + 1; j < len(walk); j++ {
			if walk[j].ID == compactionPayload.FirstKeptEntryID {
				cutoff = j
				break
			}
		}
		if cutoff < 0 {
			return nil, "", nil, fmt.Errorf(
				"state: BuildContext: FirstKeptEntryID %q not in leaf→root walk",
				compactionPayload.FirstKeptEntryID,
			)
		}
		// Keep walk[0..cutoff], skipping the Compaction entry at
		// compactionIdx. The Compaction entry contributes summaryText.
		for i := 0; i <= cutoff; i++ {
			if i == compactionIdx {
				continue
			}
			kept = append(kept, walk[i])
		}
		compactionSummary = compactionPayload.Summary
	}

	// Strip BranchSummary entries from kept; collect their text.
	stripped := kept[:0]
	for _, e := range kept {
		if e.Kind == KindBranchSummary {
			bp, _ := e.Payload.(BranchSummaryPayload)
			branchSummaries = append(branchSummaries, bp.Summary)
			continue
		}
		stripped = append(stripped, e)
	}
	kept = stripped

	// Pair-integrity: extend kept backward (toward root) until every
	// ToolResult's matching ToolUse is included.
	kept = extendForToolPairs(walk, kept)

	return kept, compactionSummary, branchSummaries, nil
}

// extendForToolPairs ensures every ToolResult block in kept has its
// matching ToolUse present. If a ToolResult references a ToolUse that is
// in walk but not in kept, the matching entry (and any ancestors between
// the kept boundary and the match) are added to kept.
//
// "walk" is the full leaf → root walk; "kept" is the post-cutoff subset.
// We work in walk indices.
func extendForToolPairs(walk []Entry, kept []Entry) []Entry {
	if len(kept) == 0 {
		return kept
	}
	// Build a set of kept entry IDs and collect all ToolUse IDs in kept.
	keptSet := make(map[string]struct{}, len(kept))
	toolUsesInKept := make(map[string]struct{})
	for _, e := range kept {
		keptSet[e.ID] = struct{}{}
		if e.Kind == KindMessage {
			mp, _ := e.Payload.(MessagePayload)
			for _, b := range mp.Content {
				if tu, ok := b.(llm.ToolUse); ok {
					toolUsesInKept[tu.ID] = struct{}{}
				}
			}
		}
	}
	// Find ToolResults in kept whose ToolUse ID is missing.
	missingToolUseIDs := make(map[string]struct{})
	for _, e := range kept {
		if e.Kind != KindMessage {
			continue
		}
		mp, _ := e.Payload.(MessagePayload)
		for _, b := range mp.Content {
			tr, ok := b.(llm.ToolResult)
			if !ok {
				continue
			}
			if _, hasUse := toolUsesInKept[tr.ToolUseID]; !hasUse {
				missingToolUseIDs[tr.ToolUseID] = struct{}{}
			}
		}
	}
	if len(missingToolUseIDs) == 0 {
		return kept
	}
	// Walk leaf → root; for each entry NOT in kept, check if it holds a
	// missing ToolUse. If yes, add it (and every older entry down to it)
	// to kept. Loop because adding new ToolResults may surface new pairs.
	for {
		added := false
		// Find the deepest (closest to root) missing ToolUse's enclosing
		// entry. Walking from root side (end of walk slice) lets us add
		// a contiguous range including the entry.
		deepestIdx := -1
		for i := len(walk) - 1; i >= 0; i-- {
			if _, isKept := keptSet[walk[i].ID]; isKept {
				continue
			}
			if walk[i].Kind != KindMessage {
				continue
			}
			mp, _ := walk[i].Payload.(MessagePayload)
			for _, b := range mp.Content {
				tu, ok := b.(llm.ToolUse)
				if !ok {
					continue
				}
				if _, missing := missingToolUseIDs[tu.ID]; missing {
					deepestIdx = i
					break
				}
			}
			if deepestIdx >= 0 {
				break
			}
		}
		if deepestIdx < 0 {
			break
		}
		// Add walk[deepestIdx] plus every entry between deepestIdx and
		// the next-younger kept entry. This preserves tree continuity in
		// the output (otherwise we'd skip a parent, breaking Tree
		// reconstruction downstream).
		for j := deepestIdx; j >= 0; j-- {
			if _, isKept := keptSet[walk[j].ID]; isKept {
				break
			}
			kept = append(kept, walk[j])
			keptSet[walk[j].ID] = struct{}{}
			// If this entry has ToolUse blocks, remove from missing.
			if walk[j].Kind == KindMessage {
				mp, _ := walk[j].Payload.(MessagePayload)
				for _, b := range mp.Content {
					if tu, ok := b.(llm.ToolUse); ok {
						delete(missingToolUseIDs, tu.ID)
					}
				}
			}
			added = true
		}
		if !added {
			break
		}
	}
	return kept
}

// entriesToMessages converts kept entries (in any order) to llm.Message
// values. The caller MUST have already filtered out non-conversational
// kinds; this helper also silently skips SessionHeader/Compaction/etc as a
// defensive measure.
func entriesToMessages(entries []Entry) []llm.Message {
	var out []llm.Message
	for _, e := range entries {
		switch e.Kind {
		case KindMessage:
			mp, _ := e.Payload.(MessagePayload)
			out = append(out, mp.AsMessage())
		case KindCustomMessage:
			cmp, _ := e.Payload.(CustomMessagePayload)
			out = append(out, llm.Message{
				Role:    llm.Role(cmp.Role),
				Content: []llm.ContentBlock{llm.TextContent{Text: string(cmp.Content)}},
			})
		case KindSessionHeader, KindCompaction, KindBranchSummary,
			KindThinkingLevelChange, KindModelChange, KindLabel,
			KindSessionInfo, KindCustom, KindClearMarker:
			// Skip: non-conversational kinds (already stripped by the caller
			// in normal flow; this branch is a defensive measure).
		default:
			// Forward-compat: silently ignore unknown kinds the caller may
			// introduce in a future schema revision.
		}
	}
	return out
}

// prependSynthetic injects the compaction summary and branch summaries as
// synthetic user messages at the start of the output. Compaction summary
// is innermost (closest to the conversation); branch summaries appear
// before it (they describe the parent branch).
//
// The text wraps each summary in a "[...]" prefix so the model can tell
// synthetic messages apart from user input.
func prependSynthetic(msgs []llm.Message, compactionSummary string, branchSummaries []string) []llm.Message {
	var synthetic []llm.Message
	if compactionSummary != "" {
		synthetic = append(synthetic, llm.Message{
			Role: llm.RoleUser,
			Content: []llm.ContentBlock{llm.TextContent{
				Text: "[Compacted earlier conversation]\n" + strings.TrimSpace(compactionSummary),
			}},
		})
	}
	// branchSummaries is in leaf→root order. To present parent-branch
	// context in chronological order, prepend them reversed.
	for i := len(branchSummaries) - 1; i >= 0; i-- {
		synthetic = append(synthetic, llm.Message{
			Role: llm.RoleUser,
			Content: []llm.ContentBlock{llm.TextContent{
				Text: "[Parent branch summary]\n" + strings.TrimSpace(branchSummaries[i]),
			}},
		})
	}
	if len(synthetic) == 0 {
		return msgs
	}
	return append(synthetic, msgs...)
}
