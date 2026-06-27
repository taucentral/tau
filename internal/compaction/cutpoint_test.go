package compaction

import (
	"testing"
	"time"

	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/state"
)

func TestFindCutPoint_EmptyWalk(t *testing.T) {
	got := FindCutPoint(nil, ProtectionList{}, detCounter{charsPerToken: 1}, "test-model", 100)
	if got != NoCutNeeded {
		t.Errorf("empty walk = %d, want NoCutNeeded (%d)", got, NoCutNeeded)
	}
}

func TestFindCutPoint_BelowThreshold(t *testing.T) {
	// walk[0] is the leaf; walk[len-1] is the root.
	// Each text block is 1 token (counter: 1 char per token, +4 framing).
	// Total = 2 messages × (1 char + 4 framing) = 10 tokens. Budget 100.
	walk := []state.Entry{
		mkMessage("m2", "m1", "assistant", "y", nowAt(2*time.Second)),
		mkMessage("m1", "h", "user", "x", nowAt(1*time.Second)),
		mkSessionHeader("h"),
	}
	got := FindCutPoint(walk, ProtectionList{}, detCounter{charsPerToken: 1}, "test-model", 100)
	if got != NoCutNeeded {
		t.Errorf("below threshold = %d, want NoCutNeeded (%d)", got, NoCutNeeded)
	}
}

func TestFindCutPoint_CutsToBudget(t *testing.T) {
	// 4 user messages of 50 chars each = 4 × (50 + 4 framing) = 216 tokens.
	// Budget = 110. Walking leaf→root, we accumulate until >110. The
	// oldest message that fits in budget is the cut.
	walk := []state.Entry{
		mkMessage("m4", "m3", "user", strings50("a"), nowAt(4*time.Second)),
		mkMessage("m3", "m2", "user", strings50("b"), nowAt(3*time.Second)),
		mkMessage("m2", "m1", "user", strings50("c"), nowAt(2*time.Second)),
		mkMessage("m1", "h", "user", strings50("d"), nowAt(1*time.Second)),
		mkSessionHeader("h"),
	}
	// tokens: m4=54, m3=54, m2=54, m1=54, header=0 (non-message).
	// Total=216. Budget=110. Walk: m4(54)+m3(54)=108 <= 110; m2 would
	// push to 162 > 110, stop. So m3 (index 1) is the deepest eligible
	// cut that fits. m1, m2, header are archived.
	got := FindCutPoint(walk, ProtectionList{}, detCounter{charsPerToken: 1}, "test-model", 110)
	if got != 1 {
		t.Errorf("cut = %d, want 1 (m3)", got)
	}
}

func TestFindCutPoint_NeverSplitToolResultPair(t *testing.T) {
	// Assistant with ToolUse → User with ToolResult → Assistant with ToolUse
	// (leaf). The walker must not cut on the user-with-ToolResult message.
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
	// Each message has 5 (nonTextBlockCost for ToolUse/ToolResult) + 4 framing
	// = 9 tokens for the tool pair; the leaf has 50 + 4 = 54.
	// Total ≈ 9 + 9 + 54 = 72.
	// Budget 60: walker keeps leaf (54), then trUser would push to 63 > 60,
	// so it stops at index 0. But trUser is not eligible as a cut anyway
	// (it's tool-result-only). tuAssist would be eligible but it doesn't
	// fit. So idealCut=0 (force-keep leaf). No protected entries. The
	// walker returns 0.
	got := FindCutPoint(walk, ProtectionList{}, detCounter{charsPerToken: 1}, "test-model", 60)
	if got != 0 {
		t.Errorf("cut = %d, want 0 (force-keep leaf)", got)
	}
}

func TestFindCutPoint_ProtectedEntriesExtendCut(t *testing.T) {
	walk := []state.Entry{
		mkMessage("m4", "m3", "user", "aa", nowAt(4*time.Second)),
		mkMessage("m3", "m2", "user", "bb", nowAt(3*time.Second)),
		mkMessage("m2", "m1", "user", "cc", nowAt(2*time.Second)),
		mkSessionInfo("p1", "m1", "k", "v", nowAt(1*time.Second)), // protected
		mkMessage("m1", "h", "user", "dd", nowAt(0)),
		mkSessionHeader("h"),
	}
	// tokens: m4=6, m3=6, m2=6, p1=0 (non-message), m1=6, header=0.
	// Total = 24. Budget = 18. Walk: m4(6)+m3(6)+m2(6)=18 <= 18; m1
	// would push to 24 > 18, stop. idealCut = index 2 (m2). But p1
	// (index 3) is protected and falls in the archived region — extend
	// cut to 3.
	protected := ProtectionList{entries: map[string]struct{}{"p1": {}}}
	got := FindCutPoint(walk, protected, detCounter{charsPerToken: 1}, "test-model", 18)
	if got != 3 {
		t.Errorf("cut = %d, want 3 (extended to protect p1)", got)
	}
}

func TestFindCutPoint_NonMessageNotEligibleAsBoundary(t *testing.T) {
	// Even if a non-message entry fits the budget, it cannot be the cut
	// boundary; the walker should pick the deepest message boundary.
	walk := []state.Entry{
		mkMessage("m2", "l1", "user", "x", nowAt(2*time.Second)),
		mkLabel("l1", "m1", "tag", nowAt(1*time.Second)),
		mkMessage("m1", "h", "user", "y", nowAt(0)),
		mkSessionHeader("h"),
	}
	// tokens: m2=5, l1=0, m1=5, header=0. Total=10. Budget=8.
	// Walk: m2(5) fits; l1(0) fits (index 1); m1(5) would push to 10>8.
	// l1 (index 1) is non-eligible (Label), so idealCut stays at index 0.
	// No protected entries → cut = 0.
	got := FindCutPoint(walk, ProtectionList{}, detCounter{charsPerToken: 1}, "test-model", 8)
	if got != 0 {
		t.Errorf("cut = %d, want 0 (Label not eligible)", got)
	}
}

func TestFindCutPoint_UserBeforeToolUseAssistantNotEligible(t *testing.T) {
	// Sequence: [user (text), assistant (tool_use)] → user (text) (leaf).
	// The first user message precedes an assistant ToolUse, so it's not
	// eligible as a cut. The walker should skip it.
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
		// leaf
		mkMessage("u2", "a1", "user", strings50("z"), nowAt(3*time.Second)),
		tuAssist,
		userPrecede,
		mkSessionHeader("h2"),
	}
	// tokens: u2=54, a1=9 (5+4), u1=9, header=0. Total ≈ 72.
	// Budget=60: u2(54) fits; a1(9) would push to 63 > 60, stop.
	// idealCut = 0 (u2 is eligible; a1 doesn't fit). u1 is not reached.
	// But even if it were: u1 is non-eligible because the immediately-
	// younger neighbor (a1) is assistant with ToolUse.
	got := FindCutPoint(walk, ProtectionList{}, detCounter{charsPerToken: 1}, "test-model", 60)
	if got != 0 {
		t.Errorf("cut = %d, want 0", got)
	}
}

func TestFindCutPoint_SystemMessageNotEligible(t *testing.T) {
	walk := []state.Entry{
		mkMessage("u1", "s1", "user", "x", nowAt(2*time.Second)),
		mkMessage("s1", "h", "system", "sysprompt", nowAt(1*time.Second)),
		mkSessionHeader("h"),
	}
	// Budget 5 < 9 (u1 tokens). u1 doesn't fit, idealCut=0 (force-keep).
	// Even with bigger budget, s1 is system role → not eligible.
	got := FindCutPoint(walk, ProtectionList{}, detCounter{charsPerToken: 1}, "test-model", 5)
	if got != 0 {
		t.Errorf("cut = %d, want 0 (force-keep leaf)", got)
	}
}

func TestFindCutPoint_ForceKeepLeafWhenNothingFits(t *testing.T) {
	// The leaf alone exceeds the budget; walker must still force-keep it
	// (return 0) to avoid archiving everything.
	walk := []state.Entry{
		mkMessage("m1", "h", "user", strings50("huge"), nowAt(0)),
		mkSessionHeader("h"),
	}
	got := FindCutPoint(walk, ProtectionList{}, detCounter{charsPerToken: 1}, "test-model", 1)
	if got != 0 {
		t.Errorf("cut = %d, want 0 (force-keep leaf)", got)
	}
}

func TestFindCutPoint_CutAtRoot_NoArchiving(t *testing.T) {
	// Whole walk fits → NoCutNeeded. Caller (compactor) handles this.
	walk := []state.Entry{
		mkMessage("m1", "h", "user", "x", nowAt(0)),
		mkSessionHeader("h"),
	}
	got := FindCutPoint(walk, ProtectionList{}, detCounter{charsPerToken: 1}, "test-model", 100)
	if got != NoCutNeeded {
		t.Errorf("cut = %d, want NoCutNeeded (%d)", got, NoCutNeeded)
	}
}

func TestIsEligibleCut_AssistantMessage(t *testing.T) {
	e := state.Entry{
		Kind:    state.KindMessage,
		Payload: state.MessagePayload{Role: llm.RoleAssistant, Content: []llm.ContentBlock{llm.TextContent{Text: "x"}}},
	}
	if !isEligibleCut(e, nil, 0) {
		t.Errorf("assistant text message should be eligible")
	}
}

func TestIsEligibleCut_LabelNotEligible(t *testing.T) {
	e := mkLabel("l", "", "x", nowAt(0))
	if isEligibleCut(e, nil, 0) {
		t.Errorf("Label should not be eligible")
	}
}

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
