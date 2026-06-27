package tokencounter

import "github.com/coevin/tau/internal/llm"

// HeuristicCounter estimates tokens as len(text)/4. This matches the
// well-known "1 token ≈ 4 characters of English text" approximation and is
// the fallback path used by the default counter when no BPE encoding is
// registered for a model. It is exposed publicly so callers that
// intentionally want a fast, network-free estimate can use it directly.
type HeuristicCounter struct{}

// Count returns max(1, len(text)/4) for any non-empty input. The minimum
// of 1 prevents callers from accidentally treating short non-empty strings
// as zero-cost.
func (HeuristicCounter) Count(model, text string) int {
	if text == "" {
		return 0
	}
	n := len(text) / 4
	if n == 0 {
		n = 1
	}
	return n
}

// CountMessages sums Count over every TextContent block, adds
// nonTextBlockCost for every other block type, and adds perMessageOverhead
// per message.
func (HeuristicCounter) CountMessages(model string, msgs []llm.Message) int {
	h := HeuristicCounter{}
	total := 0
	for _, m := range msgs {
		total += countMessageText(m, func(s string) int { return h.Count(model, s) })
		total += perMessageOverhead
	}
	return total
}
