package tokencounter

import (
	"log/slog"
	"sort"
	"strings"
	"sync"

	tiktoken "github.com/pkoukk/tiktoken-go"

	"github.com/taucentral/tau/internal/llm"
)

// modelPrefixRegistry maps a lowercased model prefix to the tiktoken
// encoding name that best approximates its tokenizer. Matching is
// longest-prefix-first (see encodingForModel).
//
// The registry intentionally includes prefix entries for non-OpenAI
// providers (e.g., claude-*) using cl100k_base as a pragmatic approximation
// because Anthropic does not publish a tokenizer. Counts for those models
// are accurate to within a few percent of provider-reported usage in
// practice — good enough for budgeting, never used as the source of truth
// (the Final delta carries authoritative Usage).
var modelPrefixRegistry = map[string]string{
	// OpenAI — o200k_base (current generation).
	"gpt-4o":           "o200k_base",
	"gpt-4.1":          "o200k_base",
	"gpt-4.5":          "o200k_base",
	"gpt-5":            "o200k_base",
	"o1":               "o200k_base",
	"o3":               "o200k_base",
	"o4":               "o200k_base",
	"chatgpt-4o":       "o200k_base",
	"text-embedding-3": "o200k_base",
	// OpenAI — cl100k_base (previous generation, still widely deployed).
	"gpt-4":                  "cl100k_base",
	"gpt-3.5":                "cl100k_base",
	"gpt-35":                 "cl100k_base", // Azure deployment naming
	"text-embedding-ada-002": "cl100k_base",
	// OpenAI — legacy.
	"text-davinci-003": "p50k_base",
	"text-davinci-002": "p50k_base",
	"davinci":          "r50k_base",
	"text-curie":       "r50k_base",
	// Anthropic — cl100k_base approximation (no public tokenizer).
	"claude-3":       "cl100k_base",
	"claude-2":       "cl100k_base",
	"claude-instant": "cl100k_base",
}

// sortedPrefixes is computed once at init time so encodingForModel can
// longest-prefix-match without re-sorting on every call.
var sortedPrefixes = func() []string {
	out := make([]string, 0, len(modelPrefixRegistry))
	for k := range modelPrefixRegistry {
		out = append(out, k)
	}
	sort.Slice(out, func(i, j int) bool {
		return len(out[i]) > len(out[j])
	})
	return out
}()

// encodingForModel returns the tiktoken encoding name for a model
// identifier using longest-prefix matching, or "" if no prefix matches.
func encodingForModel(model string) string {
	lower := strings.ToLower(model)
	for _, prefix := range sortedPrefixes {
		if strings.HasPrefix(lower, prefix) {
			return modelPrefixRegistry[prefix]
		}
	}
	return ""
}

// tiktokenCounter is the default TokenCounter. It uses real BPE for models
// with a registered encoding and a chars/4 heuristic otherwise. Unknown
// models trigger one warning log entry each (per process) so missing
// registry entries are discoverable in production logs.
type tiktokenCounter struct {
	fallback TokenCounter

	// encMu guards encoders.
	encMu sync.Mutex
	// encoders caches encoding-name → *tiktoken.Tiktoken. Loading a BPE
	// table downloads ~1MB on first use; caching keeps subsequent calls
	// fast and offline.
	encoders map[string]*tiktoken.Tiktoken

	// warned records models for which the fallback warning has already
	// been emitted. We do not need to remember the encoding name; only
	// whether the warning fired.
	warned sync.Map
}

// New returns the default TokenCounter: BPE via tiktoken-go for known
// models with a chars/4 heuristic fallback.
func New() TokenCounter {
	return &tiktokenCounter{
		fallback: HeuristicCounter{},
		encoders: make(map[string]*tiktoken.Tiktoken),
	}
}

// Count returns the BPE token count for text under the model's registered
// encoding, or the chars/4 heuristic when no encoding is registered or the
// encoding fails to load.
func (c *tiktokenCounter) Count(model, text string) int {
	if text == "" {
		return 0
	}
	name := encodingForModel(model)
	if name == "" {
		c.warnFallback(model)
		return c.fallback.Count(model, text)
	}
	enc, err := c.encoder(name)
	if err != nil {
		// Encoder failed to load (e.g., BPE download failed). Degrade to
		// heuristic rather than returning zero or panicking.
		c.warnFallback(model)
		return c.fallback.Count(model, text)
	}
	// EncodeOrdinary never panics on special tokens (unlike Encode, which
	// panics if the input contains a disallowed special token).
	return len(enc.EncodeOrdinary(text))
}

// CountMessages sums Count over every TextContent block, adds
// nonTextBlockCost for every other block type, and adds perMessageOverhead
// per message.
func (c *tiktokenCounter) CountMessages(model string, msgs []llm.Message) int {
	total := 0
	for _, m := range msgs {
		total += countMessageText(m, func(s string) int { return c.Count(model, s) })
		total += perMessageOverhead
	}
	return total
}

// encoder returns the cached Tiktoken for an encoding name, loading it on
// first use. Concurrent calls for the same name serialize on encMu; this
// is fine because loads only happen once per process.
func (c *tiktokenCounter) encoder(name string) (*tiktoken.Tiktoken, error) {
	c.encMu.Lock()
	defer c.encMu.Unlock()
	if t, ok := c.encoders[name]; ok {
		return t, nil
	}
	t, err := tiktoken.GetEncoding(name)
	if err != nil {
		return nil, err
	}
	c.encoders[name] = t
	return t, nil
}

// warnFallback emits a one-time-per-model warning indicating the counter
// is using the chars/4 heuristic. Subsequent calls for the same model are
// silent.
func (c *tiktokenCounter) warnFallback(model string) {
	if _, loaded := c.warned.LoadOrStore(model, struct{}{}); loaded {
		return
	}
	slog.Warn("tokencounter: no BPE encoding registered for model; using script-aware heuristic",
		"model", model,
		"hint", "token counts will be approximate; add the model to modelPrefixRegistry if a known encoding exists",
	)
}
