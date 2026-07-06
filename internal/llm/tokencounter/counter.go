// Package tokencounter provides model-aware token counting for the llm layer.
//
// The default TokenCounter (returned by New) uses real BPE encoding via
// github.com/pkoukk/tiktoken-go for known model prefixes and falls back to
// a chars/4 heuristic (HeuristicCounter) for unknown models. The fallback
// path logs a single warning per unknown model so callers can identify
// models that should be added to the registry.
//
// Counts returned by this package are best-effort estimates used for
// budgeting (e.g., context-window reservation and compaction triggers).
// Authoritative counts come from the provider's Usage field on the Final
// delta; this package exists so callers can budget before a request is sent.
package tokencounter

import "github.com/taucentral/tau/internal/llm"

// TokenCounter counts tokens for a given model.
//
// Implementations must be safe for concurrent use.
type TokenCounter interface {
	// Count returns the token count for a single text string. Returns 0
	// for the empty string.
	Count(model, text string) int

	// CountMessages returns the total token count for a slice of messages,
	// including per-message framing overhead (role tag, delimiters) and a
	// small fixed cost per non-text content block.
	CountMessages(model string, msgs []llm.Message) int
}

// perMessageOverhead is the per-message framing cost in tokens. The value 4
// matches the widely cited approximation from OpenAI's cookbook
// (<https://github.com/openai/openai-cookbook>) for ChatML message framing.
const perMessageOverhead = 4

// nonTextBlockCost is the approximate token cost of a non-text content block
// (image, tool_use, tool_result, thinking). The exact per-provider cost
// depends on framing and is not modelled here; the value is intentionally
// small so totals stay within provider-reported usage to within a few percent
// for typical agent conversations.
const nonTextBlockCost = 5

// countMessageText sums Count over every TextContent block in a message and
// adds nonTextBlockCost for every other block type. Callers pass in a
// text-counting function (the implementation-specific Count).
func countMessageText(m llm.Message, countText func(string) int) int {
	total := 0
	for _, b := range m.Content {
		if tc, ok := b.(llm.TextContent); ok {
			total += countText(tc.Text)
			continue
		}
		total += nonTextBlockCost
	}
	return total
}
