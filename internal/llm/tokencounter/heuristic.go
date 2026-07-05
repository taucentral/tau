package tokencounter

import (
	"math"
	"unicode"

	"github.com/coevin/tau/internal/llm"
)

// HeuristicCounter is a fast, network-free token estimator used as the
// fallback path when no BPE encoding is registered for a model. It walks
// the input runes once, classifies each by Unicode script, and sums
// per-script fractional token contributions.
//
// The estimator is calibrated to err slightly high — the safe direction
// for budgeting, since overestimating triggers compaction a turn early
// (cheap) while underestimating blows the context window (fails the
// request). For ASCII-only input the result is identical to the legacy
// len(text)/4 formula, preserving the behavior external test stubs
// (runtime_test.go) rely on.
//
// HeuristicCounter is exported so callers that intentionally want a fast,
// network-free estimate can use it directly.
type HeuristicCounter struct{}

// Count returns the script-aware token estimate for text. For any
// non-empty input it returns at least 1. The model parameter is accepted
// for interface conformance but ignored — per-model calibration (e.g.
// tuning ratios against a known vocabulary size) is a separate opt-in
// layer not implemented here.
func (HeuristicCounter) Count(model, text string) int {
	_ = model // accepted for interface conformance; intentionally unused
	if text == "" {
		return 0
	}
	var tokens float64
	for _, r := range text {
		tokens += tokenWeightForRune(r)
	}
	n := int(math.Floor(tokens))
	if n == 0 {
		n = 1
	}
	return n
}

// tokenWeightForRune returns the approximate fractional tokens a single
// rune contributes under the script-aware heuristic.
//
// Per-script ratios (chars/token):
//   - ASCII (Latin, digits, punctuation, control): 4.0 — preserves legacy
//     len(text)/4 behavior exactly.
//   - CJK (Han, Hiragana, Katakana, Hangul): 1.5 — Chinese/Japanese/Korean
//     vocabularies are far denser per character than Latin script.
//   - Emoji / extended pictographic: 1.0 — pictographics are typically a
//     single token in modern vocabularies.
//   - Everything else (Cyrillic, Greek, Arabic, Hebrew, Devanagari, Thai,
//     other): 2.5 — middle ground; most non-CJK scripts cluster around
//     2-3 chars/token in modern BPE vocabularies.
//
// floor(weighted sum) reproduces integer division semantics: for ASCII
// (weight 0.25/rune), floor(runeCount * 0.25) == len(text)/4.
func tokenWeightForRune(r rune) float64 {
	switch {
	case r < 0x80:
		return 0.25 // ASCII: 1/4.0
	case isCJK(r):
		return 1.0 / 1.5
	case isEmoji(r):
		return 1.0
	default:
		return 1.0 / 2.5
	}
}

// isCJK reports whether r belongs to a Chinese/Japanese/Korean script.
func isCJK(r rune) bool {
	return unicode.Is(unicode.Han, r) ||
		unicode.Is(unicode.Hiragana, r) ||
		unicode.Is(unicode.Katakana, r) ||
		unicode.Is(unicode.Hangul, r)
}

// isEmoji reports whether r is likely a pictographic / emoji code point.
// The check covers the common emoji blocks; variation selectors and
// combining marks are intentionally excluded — they are rare as
// standalone runes and counting them as their own token would slightly
// overcount.
func isEmoji(r rune) bool {
	return (r >= 0x1F300 && r <= 0x1FAFF) || // Emoji & supplemental pictographic
		(r >= 0x2600 && r <= 0x27BF) || // Misc symbols, dingbats
		(r >= 0x1F1E6 && r <= 0x1F1FF) // Regional indicators (flags)
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
