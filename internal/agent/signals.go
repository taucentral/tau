// signals.go — per-turn signal extraction for lazy-tool hydration.
//
// The agent loop derives TurnSignals from the state tree and the
// session settings before each request. These helpers walk the
// message history to extract the latest user text and the set of
// recently-called tool names. The registry consults these values
// (plus mode/always-render/window from settings) when evaluating
// LazyHeadlessTool hydration triggers.

package agent

import (
	"sort"

	"github.com/taucentral/tau/internal/llm"
	"github.com/taucentral/tau/internal/tools"
)

// buildTurnSignals assembles a tools.TurnSignals value from the
// request's message history and the session's Tools settings. The
// agent calls this once per request before invoking Registry.Schemas.
//
// Settings bridging: internal/tools cannot import internal/config
// (import cycle), so the agent layer copies ToolsSettings fields
// into TurnSignals here. Nil settings yield the zero-value defaults
// (HydrationModeHeuristic, RecentUseWindow=5).
func buildTurnSignals(msgs []llm.Message, toolsSettings toolsSettingsView) tools.TurnSignals {
	mode := tools.HydrationMode(toolsSettings.HydrationMode)
	if mode == "" {
		mode = tools.HydrationModeHeuristic
	}
	window := toolsSettings.RecentUseWindow
	if window <= 0 {
		window = 5
	}
	return tools.TurnSignals{
		UserMessage:     latestUserMessage(msgs),
		RecentToolCalls: recentToolNames(msgs, window),
		Mode:            mode,
		AlwaysRender:    toolsSettings.AlwaysRender,
		RecentUseWindow: window,
	}
}

// toolsSettingsView is a local view of config.ToolsSettings. Keeping
// it as an unexported struct here avoids importing internal/config
// from the agent layer's signal-extraction helper; the caller copies
// fields from config.ToolsSettings into this view.
type toolsSettingsView struct {
	HydrationMode   string
	AlwaysRender    []string
	RecentUseWindow int
}

// latestUserMessage returns the text of the most recent user-role
// message whose content includes a non-empty TextContent block. Tool-
// result messages (also Role=user) are skipped because their content
// is ToolResult blocks, not TextContent.
//
// Returns "" when no user text exists in the message slice.
func latestUserMessage(msgs []llm.Message) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		m := msgs[i]
		if m.Role != llm.RoleUser {
			continue
		}
		for _, b := range m.Content {
			if tc, ok := b.(llm.TextContent); ok && tc.Text != "" {
				return tc.Text
			}
		}
	}
	return ""
}

// recentToolNames collects the distinct tool names that appear in
// ToolUse blocks within the last `window` assistant messages that
// contain tool calls. Returns a sorted slice suitable for set lookup
// by the registry.
//
// The window bounds the lookback to roughly `window` turns of tool
// activity. window <= 0 returns nil (no recency matching).
func recentToolNames(msgs []llm.Message, window int) []string {
	if window <= 0 {
		return nil
	}
	seen := make(map[string]bool)
	out := make([]string, 0)
	turnsSeen := 0
	for i := len(msgs) - 1; i >= 0; i-- {
		m := msgs[i]
		if m.Role != llm.RoleAssistant {
			continue
		}
		hasToolUse := false
		for _, b := range m.Content {
			if tu, ok := b.(llm.ToolUse); ok {
				hasToolUse = true
				if !seen[tu.Name] {
					seen[tu.Name] = true
					out = append(out, tu.Name)
				}
			}
		}
		if hasToolUse {
			turnsSeen++
			if turnsSeen >= window {
				break
			}
		}
	}
	sort.Strings(out)
	return out
}
