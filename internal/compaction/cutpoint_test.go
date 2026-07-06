package compaction

import (
	"testing"
	"time"

	"github.com/taucentral/tau/internal/llm"
	"github.com/taucentral/tau/internal/state"
)

// Backward-walk floor semantic (design.md D1.1, D1.2):
// FindCutPoint walks leaf → root (walk[0] → walk[N-1]) accumulating tokens,
// and returns the index of the OLDEST entry that remains in context. The
// walk stops when running >= keepRecentTokens AND at least one eligible cut
// has been seen; the cut is the closest eligible cut at or before the
// position where the accumulator crosses the floor.
//
// Per design.md D1.4, ReserveTokens is trigger-only (handled by MaybeCompact)
// and does NOT influence FindCutPoint. The last parameter to FindCutPoint is
// keepRecentTokens (the floor), NOT a ceiling.

// TestFindCutPoint_EmptyWalk: empty walk → NoCutNeeded.
func TestFindCutPoint_EmptyWalk(t *testing.T) {
	got := FindCutPoint(nil, ProtectionList{}, detCounter{charsPerToken: 1}, "test-model", 100)
	if got != NoCutNeeded {
		t.Errorf("empty walk = %d, want NoCutNeeded (%d)", got, NoCutNeeded)
	}
}

// Test 4.1: floor crossed mid-walk. The walker keeps the entries from the
// leaf up to (and including) the eligible cut at-or-newer than the crossing
// point. Entries older than the cut are archived.
func TestFindCutPoint_FloorCrossedMidWalk(t *testing.T) {
	// Walk (leaf→root): m4(0), m3(1), m2(2), m1(3), header(4). Each msg = 54 tokens.
	// Floor = 110. cutPoints = [0,1,2,3] (all eligible).
	// Default cutIndex = 3 (oldest eligible). Loop walks i=0→4:
	//   i=0: running=54.   <110, continue.
	//   i=1: running=108.  <110, continue.
	//   i=2: running=162.  >=110! Largest cutPoint<=2 is 2. cutIndex=2. break.
	// Returned cutIndex = 2 (m2 is the oldest kept). Kept region = walk[0..2]
	// = m4, m3, m2 = 162 tokens ≥ floor 110. m1 and header are archived.
	walk := []state.Entry{
		mkMessage("m4", "m3", "user", strings50("a"), nowAt(4*time.Second)),
		mkMessage("m3", "m2", "user", strings50("b"), nowAt(3*time.Second)),
		mkMessage("m2", "m1", "user", strings50("c"), nowAt(2*time.Second)),
		mkMessage("m1", "h", "user", strings50("d"), nowAt(1*time.Second)),
		mkSessionHeader("h"),
	}
	got := FindCutPoint(walk, ProtectionList{}, detCounter{charsPerToken: 1}, "test-model", 110)
	if got != 2 {
		t.Errorf("cut = %d, want 2 (m2 — floor crossed at m2; m2 is the at-or-newer eligible cut)", got)
	}
}

// Test 4.2: floor below natural cut. A small floor is crossed soon, cutting
// near the leaf and archiving most of the walk. Proves the floor is a FLOOR
// (minimum kept), not a ceiling (maximum kept).
func TestFindCutPoint_FloorBelowNaturalCut(t *testing.T) {
	// Same walk as 4.1 (4 msgs × 54 tokens = 216 total). Floor = 55.
	//   i=0: running=54.  <55, continue.
	//   i=1: running=108. >=55! Largest cutPoint<=1 is 1. cutIndex=1. break.
	// Returned cutIndex = 1 (m3 is the oldest kept). Kept = m4+m3 = 108 ≥ 55.
	walk := []state.Entry{
		mkMessage("m4", "m3", "user", strings50("a"), nowAt(4*time.Second)),
		mkMessage("m3", "m2", "user", strings50("b"), nowAt(3*time.Second)),
		mkMessage("m2", "m1", "user", strings50("c"), nowAt(2*time.Second)),
		mkMessage("m1", "h", "user", strings50("d"), nowAt(1*time.Second)),
		mkSessionHeader("h"),
	}
	got := FindCutPoint(walk, ProtectionList{}, detCounter{charsPerToken: 1}, "test-model", 55)
	if got != 1 {
		t.Errorf("cut = %d, want 1 (m3 — small floor crossed early, archive m2/m1/header)", got)
	}
}

// Test 4.3: floor of zero. Walker crosses the floor on the very first
// entry (the leaf) and cuts there, archiving everything older.
func TestFindCutPoint_FloorZero_KeepsOnlyLeaf(t *testing.T) {
	// Floor = 0. Default cutIndex=3.
	//   i=0: running=54. >=0! Largest cutPoint<=0 is 0. cutIndex=0. break.
	// Returned cutIndex = 0 (leaf only). Everything older is archived.
	walk := []state.Entry{
		mkMessage("m4", "m3", "user", strings50("a"), nowAt(4*time.Second)),
		mkMessage("m3", "m2", "user", strings50("b"), nowAt(3*time.Second)),
		mkMessage("m2", "m1", "user", strings50("c"), nowAt(2*time.Second)),
		mkMessage("m1", "h", "user", strings50("d"), nowAt(1*time.Second)),
		mkSessionHeader("h"),
	}
	got := FindCutPoint(walk, ProtectionList{}, detCounter{charsPerToken: 1}, "test-model", 0)
	if got != 0 {
		t.Errorf("cut = %d, want 0 (floor 0 crosses immediately at leaf; archive everything older)", got)
	}
}

// Test 4.4: floor larger than total walk. Walker exhausts without crossing
// the floor; the cut is the OLDEST eligible entry (default), so compaction
// archives nothing beyond the natural session boundary (the SessionHeader).
func TestFindCutPoint_FloorLargerThanTotal_OldestEligible(t *testing.T) {
	// 4 messages × 54 tokens = 216 total. Floor = 10_000 (much larger).
	// Walker accumulates everything: running=216 at end. Never crosses 10_000.
	// Final cutIndex = default = cutPoints[len-1] = 3 (m1, oldest eligible).
	// Archived: just the header.
	walk := []state.Entry{
		mkMessage("m4", "m3", "user", strings50("a"), nowAt(4*time.Second)),
		mkMessage("m3", "m2", "user", strings50("b"), nowAt(3*time.Second)),
		mkMessage("m2", "m1", "user", strings50("c"), nowAt(2*time.Second)),
		mkMessage("m1", "h", "user", strings50("d"), nowAt(1*time.Second)),
		mkSessionHeader("h"),
	}
	got := FindCutPoint(walk, ProtectionList{}, detCounter{charsPerToken: 1}, "test-model", 10_000)
	if got != 3 {
		t.Errorf("cut = %d, want 3 (m1, oldest eligible — floor exceeded total, force-keep all eligible)", got)
	}
}

// Test 4.5: floor with protected entry in the would-be archived region.
// The protected-entries extension runs AFTER the floor cut and pulls
// protected entries into the kept region.
func TestFindCutPoint_FloorWithProtectedEntry(t *testing.T) {
	// Walk: [m4(6), m3(6), m2(6), p1(0 protected), m1(6), header(0)].
	// Floor = 12. Walk backward:
	//   i=5 header (0):  running=0,  not eligible, continue
	//   i=4 m1 (6):      running=6,  eligible → cutIndex=4. 6<12 continue
	//   i=3 p1 (0):      running=6,  not eligible (SessionInfo), continue. 6<12 continue
	//   i=2 m2 (6):      running=12, eligible → cutIndex=2. 12>=12, break.
	// Initial cutIndex = 2 (m2). Protected scan: j=3 (p1) is protected → cut=3.
	// j=4 (m1) not protected. Final cut = 3.
	walk := []state.Entry{
		mkMessage("m4", "m3", "user", "aa", nowAt(4*time.Second)),
		mkMessage("m3", "m2", "user", "bb", nowAt(3*time.Second)),
		mkMessage("m2", "m1", "user", "cc", nowAt(2*time.Second)),
		mkSessionInfo("p1", "m1", "k", "v", nowAt(1*time.Second)), // protected
		mkMessage("m1", "h", "user", "dd", nowAt(0)),
		mkSessionHeader("h"),
	}
	protected := ProtectionList{entries: map[string]struct{}{"p1": {}}}
	got := FindCutPoint(walk, protected, detCounter{charsPerToken: 1}, "test-model", 12)
	if got != 3 {
		t.Errorf("cut = %d, want 3 (m2 floor-cut, extended to 3 by protected p1)", got)
	}
}

// Test 4.6: ToolUse/ToolResult pairing preserved. The tool-result-only user
// message is NOT an eligible cut. Cutting at it would orphan the matching
// ToolUse that's newer in the walk.
func TestFindCutPoint_ToolResultPairNotEligible(t *testing.T) {
	// Sequence (leaf → root): [a2(assistant, text, 54), u1(user, ToolResult, 9), a1(assistant, ToolUse, 9), header].
	// Floor = 60. Walk backward:
	//   i=3 header (0):  running=0,  not eligible, continue
	//   i=2 a1 (9):      running=9,  eligible (assistant) → cutIndex=2. 9<60 continue
	//   i=1 u1 (9):      running=18, NOT eligible (tool-result-only). 18<60 continue
	//   i=0 a2 (54):     running=72, eligible (assistant) → cutIndex=0. 72>=60, break.
	// Returned cutIndex = 0.
	tuAssist := state.Entry{
		ID:        "a1",
		ParentID:  "h",
		Kind:      state.KindMessage,
		Timestamp: nowAt(1 * time.Second),
		Payload: state.MessagePayload{
			Role: llm.RoleAssistant,
			Content: []llm.ContentBlock{
				llm.ToolUse{ID: "tu1", Name: "read", Input: []byte(`{"path":"/x"}`)},
			},
		},
	}
	trUser := state.Entry{
		ID:        "u1",
		ParentID:  "a1",
		Kind:      state.KindMessage,
		Timestamp: nowAt(2 * time.Second),
		Payload: state.MessagePayload{
			Role: llm.RoleUser,
			Content: []llm.ContentBlock{
				llm.ToolResult{ToolUseID: "tu1", Content: []llm.ContentBlock{llm.TextContent{Text: "result"}}},
			},
		},
	}
	leafAssist := mkMessage("a2", "u1", "assistant", strings50("z"), nowAt(3*time.Second))
	walk := []state.Entry{
		leafAssist,
		trUser,
		tuAssist,
		mkSessionHeader("h"),
	}
	got := FindCutPoint(walk, ProtectionList{}, detCounter{charsPerToken: 1}, "test-model", 60)
	if got != 0 {
		t.Errorf("cut = %d, want 0 (tool-result-only user is not eligible; cut at leaf)", got)
	}
}

// TestFindCutPoint_NonMessageNotEligibleAsBoundary: non-message entries
// (Label, SessionInfo, etc.) are never eligible cut points but can still be
// included in the kept region.
func TestFindCutPoint_NonMessageNotEligibleAsBoundary(t *testing.T) {
	// Walk (leaf→root): m2(0,user,5), l1(1,Label,0), m1(2,user,5), header(3,0).
	// cutPoints = [0, 2] (Labels not eligible). Default cutIndex = 2.
	// Loop i=0→3:
	//   i=0: running=5.  <8, continue.
	//   i=1: running=5.  (Label adds 0). <8, continue.
	//   i=2: running=10. >=8! Largest cutPoint<=2 is 2 (m1). cutIndex=2. break.
	// Returned cutIndex = 2 (m1 is oldest kept). Kept = m2, l1, m1.
	walk := []state.Entry{
		mkMessage("m2", "l1", "user", "x", nowAt(2*time.Second)),
		mkLabel("l1", "m1", "tag", nowAt(1*time.Second)),
		mkMessage("m1", "h", "user", "y", nowAt(0)),
		mkSessionHeader("h"),
	}
	got := FindCutPoint(walk, ProtectionList{}, detCounter{charsPerToken: 1}, "test-model", 8)
	if got != 2 {
		t.Errorf("cut = %d, want 2 (Label not eligible; m1 is the closest at-or-newer eligible to floor crossing)", got)
	}
}

// TestFindCutPoint_UserBeforeToolUseAssistantNotEligible: a user message
// whose immediately-younger neighbor is an assistant message with ToolUse
// blocks is not eligible (cutting there would orphan the ToolUse from its
// preceding user intent).
func TestFindCutPoint_UserBeforeToolUseAssistantNotEligible(t *testing.T) {
	// Sequence (leaf → root): u2(0,user,54), a1(1,assistant+ToolUse,9), u1(2,user,9), header(3).
	// u1 is NOT eligible (younger neighbor a1 is assistant+ToolUse).
	// u2 eligible (no younger neighbor). a1 eligible (assistant).
	// cutPoints = [0, 1]. Default cutIndex = 1.
	// Loop:
	//   i=0: running=54. <60, continue.
	//   i=1: running=63. >=60! Largest cutPoint<=1 is 1 (a1). cutIndex=1. break.
	// Returned cutIndex = 1 (a1 is oldest kept). Kept = u2, a1. u1 archived.
	tuAssist := state.Entry{
		ID:        "a1",
		ParentID:  "u1",
		Kind:      state.KindMessage,
		Timestamp: nowAt(2 * time.Second),
		Payload: state.MessagePayload{
			Role: llm.RoleAssistant,
			Content: []llm.ContentBlock{
				llm.TextContent{Text: "x"},
				llm.ToolUse{ID: "tu1", Name: "read", Input: []byte(`{"path":"/x"}`)},
			},
		},
	}
	userPrecede := mkMessage("u1", "h2", "user", "hello", nowAt(1*time.Second))
	walk := []state.Entry{
		mkMessage("u2", "a1", "user", strings50("z"), nowAt(3*time.Second)),
		tuAssist,
		userPrecede,
		mkSessionHeader("h2"),
	}
	got := FindCutPoint(walk, ProtectionList{}, detCounter{charsPerToken: 1}, "test-model", 60)
	if got != 1 {
		t.Errorf("cut = %d, want 1 (a1 — u1 not eligible; a1 is the at-or-newer eligible to floor crossing)", got)
	}
}

// TestFindCutPoint_SystemMessageNotEligible: system-role messages are never
// valid cut points (system belongs to the always-kept prefix).
func TestFindCutPoint_SystemMessageNotEligible(t *testing.T) {
	// Walk: [u1(user,5), s1(system,9), header(0)]. Floor = 5.
	// Walk backward:
	//   i=2 header (0): running=0,  not eligible, continue
	//   i=1 s1 (9):     running=9,  NOT eligible (system). 9>=5 but cutIndex=-1, continue
	//   i=0 u1 (5):     running=14, eligible → cutIndex=0. 14>=5 && cutIndex>=0, break.
	// Returned cutIndex = 0.
	walk := []state.Entry{
		mkMessage("u1", "s1", "user", "x", nowAt(2*time.Second)),
		mkMessage("s1", "h", "system", "sysprompt", nowAt(1*time.Second)),
		mkSessionHeader("h"),
	}
	got := FindCutPoint(walk, ProtectionList{}, detCounter{charsPerToken: 1}, "test-model", 5)
	if got != 0 {
		t.Errorf("cut = %d, want 0 (system not eligible; floor forces cut at leaf)", got)
	}
}

// TestFindCutPoint_ForceKeepLeafWhenNoEligible: when no eligible cut is
// seen at all, the walker force-keeps walk[0] (the leaf) so we don't
// archive the entire tree. The next compaction will revisit.
func TestFindCutPoint_ForceKeepLeafWhenNoEligible(t *testing.T) {
	// Walk with ONLY non-message entries and one system message — no
	// eligible cuts anywhere. Floor = 100.
	walk := []state.Entry{
		mkMessage("s1", "l1", "system", "sys", nowAt(1*time.Second)),
		mkLabel("l1", "h", "tag", nowAt(0)),
		mkSessionHeader("h"),
	}
	got := FindCutPoint(walk, ProtectionList{}, detCounter{charsPerToken: 1}, "test-model", 100)
	if got != 0 {
		t.Errorf("cut = %d, want 0 (force-keep leaf when no eligible cut)", got)
	}
}

// TestIsEligibleCut_AssistantMessage: assistant text message is eligible.
func TestIsEligibleCut_AssistantMessage(t *testing.T) {
	e := state.Entry{
		Kind:    state.KindMessage,
		Payload: state.MessagePayload{Role: llm.RoleAssistant, Content: []llm.ContentBlock{llm.TextContent{Text: "x"}}},
	}
	if !isEligibleCut(e, nil, 0) {
		t.Errorf("assistant text message should be eligible")
	}
}

// TestIsEligibleCut_LabelNotEligible: Label is not a Message kind.
func TestIsEligibleCut_LabelNotEligible(t *testing.T) {
	e := mkLabel("l", "", "x", nowAt(0))
	if isEligibleCut(e, nil, 0) {
		t.Errorf("Label should not be eligible")
	}
}

// TestHasToolUse_TrueFalse: hasToolUse correctly detects ToolUse blocks.
func TestHasToolUse_TrueFalse(t *testing.T) {
	with := state.MessagePayload{
		Role:    llm.RoleAssistant,
		Content: []llm.ContentBlock{llm.ToolUse{ID: "tu1", Name: "read"}},
	}
	if !hasToolUse(with) {
		t.Errorf("hasToolUse should be true for assistant+ToolUse")
	}
	without := state.MessagePayload{
		Role:    llm.RoleAssistant,
		Content: []llm.ContentBlock{llm.TextContent{Text: "x"}},
	}
	if hasToolUse(without) {
		t.Errorf("hasToolUse should be false for plain assistant text")
	}
}

// TestIsToolResultMessage: detects user-role messages consisting solely of
// ToolResult blocks.
func TestIsToolResultMessage(t *testing.T) {
	pure := state.MessagePayload{
		Role: llm.RoleUser,
		Content: []llm.ContentBlock{
			llm.ToolResult{ToolUseID: "tu1", Content: []llm.ContentBlock{llm.TextContent{Text: "x"}}},
		},
	}
	if !isToolResultMessage(pure) {
		t.Errorf("pure tool-result user message should be detected")
	}
	mixed := state.MessagePayload{
		Role: llm.RoleUser,
		Content: []llm.ContentBlock{
			llm.TextContent{Text: "note"},
			llm.ToolResult{ToolUseID: "tu1"},
		},
	}
	if isToolResultMessage(mixed) {
		t.Errorf("mixed text+tool-result message should NOT be detected as pure result")
	}
	empty := state.MessagePayload{Role: llm.RoleUser}
	if isToolResultMessage(empty) {
		t.Errorf("empty user message should not be tool-result")
	}
}

// strings50 returns a string of exactly 50 chars by repeating the given
// filler char.
func strings50(fill string) string {
	out := make([]byte, 0, 50)
	for len(out) < 50 {
		out = append(out, fill...)
	}
	return string(out[:50])
}
