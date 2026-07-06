package tokencounter

import (
	"bytes"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"testing"

	"github.com/taucentral/tau/internal/llm"
)

func TestHeuristic_Basic(t *testing.T) {
	h := HeuristicCounter{}
	if got := h.Count("any-model", ""); got != 0 {
		t.Errorf("empty = %d, want 0", got)
	}
	if got := h.Count("any-model", "ab"); got != 1 {
		t.Errorf("ab = %d, want 1 (min-clamped)", got)
	}
	if got := h.Count("any-model", "abcdefgh"); got != 2 {
		t.Errorf("abcdefgh = %d, want 2", got)
	}
}

func TestHeuristic_CountMessages(t *testing.T) {
	h := HeuristicCounter{}
	msgs := []llm.Message{
		{Role: llm.RoleUser, Content: []llm.ContentBlock{
			llm.TextContent{Text: "abcdefgh"},                  // 2
			llm.ImageContent{Data: "x", MimeType: "image/png"}, // 5 (non-text)
		}},
	}
	// 2 (text) + 5 (image) + 4 (per-message overhead) = 11
	if got := h.CountMessages("m", msgs); got != 11 {
		t.Errorf("got %d, want 11", got)
	}
}

func TestEncodingForModel_KnownPrefixes(t *testing.T) {
	cases := map[string]string{
		"gpt-4o":            "o200k_base",
		"gpt-4o-2024-11-20": "o200k_base",
		"gpt-4":             "cl100k_base",
		"gpt-4-0613":        "cl100k_base",
		"gpt-3.5-turbo":     "cl100k_base",
		"gpt-35-turbo":      "cl100k_base", // Azure naming
		"text-davinci-003":  "p50k_base",
		"claude-3-5-sonnet": "cl100k_base",
		"claude-2.1":        "cl100k_base",
	}
	for model, want := range cases {
		if got := encodingForModel(model); got != want {
			t.Errorf("encodingForModel(%q) = %q, want %q", model, got, want)
		}
	}
}

func TestEncodingForModel_LongestPrefixWins(t *testing.T) {
	// gpt-4o (o200k_base) is more specific than gpt-4 (cl100k_base).
	if got := encodingForModel("gpt-4o"); got != "o200k_base" {
		t.Errorf("gpt-4o = %q, want o200k_base (longest-prefix)", got)
	}
}

func TestEncodingForModel_Unknown(t *testing.T) {
	if got := encodingForModel("some-future-model"); got != "" {
		t.Errorf("got %q, want empty for unknown model", got)
	}
}

func TestTiktoken_KnownModelBPE(t *testing.T) {
	c := New()
	// "hello world" is 2 tokens under cl100k_base and o200k_base.
	got := c.Count("gpt-4o", "hello world")
	if got < 1 || got > 5 {
		t.Errorf("Count(gpt-4o, hello world) = %d, want 1..5 (BPE)", got)
	}
	// Longer text scales; rough invariant: token count is much smaller
	// than character count for English text.
	long := strings.Repeat("the quick brown fox ", 50)
	if got, want := c.Count("gpt-4", long), len(long)/4; got >= want {
		t.Errorf("Count(gpt-4, long English) = %d, want < %d (heuristic ceiling)", got, want)
	}
}

// TestTiktoken_BPEAccuracyAgainstKnownCounts verifies the counter produces
// exact counts for strings whose tokenization is published and stable across
// cl100k_base / o200k_base. This is the Phase 3.18 "BPE accuracy" gate.
func TestTiktoken_BPEAccuracyAgainstKnownCounts(t *testing.T) {
	c := New()
	cases := []struct {
		model string
		text  string
		want  int
	}{
		{"gpt-4o", "hello world", 2},
		{"gpt-4", "hello world", 2},
		{"gpt-4o", "Hello, world!", 4},
		{"gpt-4", "Hello, world!", 4},
		{"gpt-4o", "The quick brown fox jumps over the lazy dog.", 10},
		{"gpt-4", "The quick brown fox jumps over the lazy dog.", 10},
		{"gpt-3.5-turbo", "The quick brown fox jumps over the lazy dog.", 10},
		{"claude-3-5-sonnet", "The quick brown fox jumps over the lazy dog.", 10},
		// 50 repetitions of "the " plus the implicit trailing "the" token.
		{"gpt-4o", strings.Repeat("the ", 50), 51},
	}
	for _, tc := range cases {
		if got := c.Count(tc.model, tc.text); got != tc.want {
			t.Errorf("Count(%q, %q) = %d, want %d", tc.model, tc.text, got, tc.want)
		}
	}
}

func TestTiktoken_UnknownModelFallback(t *testing.T) {
	c := New()
	// Capture slog warnings to verify the one-time-per-model rule.
	var buf bytes.Buffer
	prev := slog.Default()
	defer slog.SetDefault(prev)
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))

	text := "abcdefghij" // 10 chars → 2 tokens via chars/4
	got := c.Count("totally-unknown-model-xyz", text)
	if want := 2; got != want {
		t.Errorf("fallback Count = %d, want %d", got, want)
	}
	// Call again for the same model — warning should NOT fire twice.
	_ = c.Count("totally-unknown-model-xyz", text)
	_ = c.Count("totally-unknown-model-xyz", "more text")

	logs := buf.String()
	if got := strings.Count(logs, "level=WARN"); got != 1 {
		t.Errorf("warn count = %d, want exactly 1\nlogs:\n%s", got, logs)
	}
	if !strings.Contains(logs, "totally-unknown-model-xyz") {
		t.Errorf("warning does not name the model\nlogs:\n%s", logs)
	}

	// A different unknown model triggers its own warning.
	_ = c.Count("another-unknown-model", "x")
	if got := strings.Count(buf.String(), "level=WARN"); got != 2 {
		t.Errorf("warn count after second model = %d, want 2", got)
	}
}

func TestTiktoken_EmptyString(t *testing.T) {
	c := New()
	if got := c.Count("gpt-4o", ""); got != 0 {
		t.Errorf("empty Count = %d, want 0", got)
	}
}

func TestTiktoken_CountMessages(t *testing.T) {
	c := New()
	body := "The quick brown fox jumps over the lazy dog and then keeps running through the field."
	msgs := []llm.Message{
		{Role: llm.RoleUser, Content: []llm.ContentBlock{
			llm.TextContent{Text: body},
		}},
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
			llm.TextContent{Text: body},
			llm.ToolUse{ID: "t1", Name: "noop", Input: []byte("{}")},
		}},
	}
	// BPE counts English text more tightly than chars/4 (English packs many
	// common words into single tokens). With sufficiently long text, the
	// tiktoken count should be strictly less than the heuristic ceiling.
	heuristic := HeuristicCounter{}.CountMessages("gpt-4o", msgs)
	got := c.CountMessages("gpt-4o", msgs)
	if got >= heuristic {
		t.Errorf("BPE CountMessages = %d, heuristic = %d; expected BPE < heuristic for English text", got, heuristic)
	}
}

func TestTiktoken_ConcurrentSafe(t *testing.T) {
	c := New()
	const goroutines = 16
	const iters = 50
	var wg sync.WaitGroup
	wg.Add(goroutines)
	errs := make(chan error, goroutines)
	for i := 0; i < goroutines; i++ {
		go func(seed int) {
			defer wg.Done()
			defer func() {
				if r := recover(); r != nil {
					errs <- fmt.Errorf("panic: %v", r)
				}
			}()
			model := "gpt-4o"
			if seed%2 == 0 {
				model = "gpt-3.5-turbo"
			}
			for j := 0; j < iters; j++ {
				_ = c.Count(model, "the quick brown fox jumps over the lazy dog")
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("concurrent: %v", err)
	}
}

// TestHeuristic_ASCII_MatchesOldBehavior is the explicit regression guard
// for the ASCII path. External test stubs (runtime_test.go) depend on
// HeuristicCounter.Count returning max(1, len(text)/4) for ASCII input.
// The script-aware rewrite must preserve that exactly.
func TestHeuristic_ASCII_MatchesOldBehavior(t *testing.T) {
	h := HeuristicCounter{}
	cases := []string{
		"",
		"a",
		"ab",
		"abc",
		"abcd",
		"abcde",
		"abcdef",
		"abcdefg",
		"abcdefgh",
		"abcdefghi",
		"hello world, this is a test string",
		strings.Repeat("a", 100),
	}
	for _, text := range cases {
		want := len(text) / 4
		if want == 0 && text != "" {
			want = 1
		}
		got := h.Count("any-model", text)
		if got != want {
			t.Errorf("Count(%q) = %d, want %d (legacy len/4 formula)", text, got, want)
		}
	}
}

// TestHeuristic_ScriptAware verifies the script-aware buckets produce
// reasonable estimates for non-ASCII content. Assertions use loose bounds
// rather than exact numbers — the ratios are heuristic estimates that may
// be recalibrated, and tight assertions would make the tests brittle.
func TestHeuristic_ScriptAware(t *testing.T) {
	h := HeuristicCounter{}
	cases := []struct {
		name      string
		text      string
		minTokens int // inclusive lower bound
		maxTokens int // inclusive upper bound
	}{
		// Pure ASCII baseline — exact match to legacy formula.
		{"ascii_short", "hello", 1, 2}, // 5*0.25=1.25 → floor 1
		{"ascii_long", "hello world this is a test", 6, 7}, // 26*0.25=6.5 → floor 6

		// CJK — 1.5 chars/token. 4 chars → ~2.67 → floor 2.
		{"chinese_short", "你好世界", 2, 4},
		// 13 chars → ~8.67 → floor 8.
		{"chinese_long", "你好世界，这是一个测试字符串", 8, 13},

		// Japanese (Han + Hiragana mix) — same bucket.
		{"japanese_mixed", "こんにちは世界", 4, 7}, // 7 chars → ~4.67 → floor 4

		// Korean (Hangul + ASCII space).
		{"korean", "안녕하세요 세계", 4, 7}, // 6 Hangul + 1 space → ~4.25 → floor 4

		// Other scripts — 2.5 chars/token.
		{"cyrillic", "Привет мир", 3, 6}, // 9 Cyrillic + 1 space → ~3.85 → floor 3
		{"arabic", "مرحبا بالعالم", 4, 8}, // 12 Arabic + 1 space → ~5.05 → floor 5

		// Emoji — 1.0 chars/token (1 token per emoji).
		{"emoji_pure", "😀😁😂🤣😃", 5, 5},

		// Mixed ASCII + CJK (realistic: code with Chinese comment).
		// "def hello " (10 ASCII → 2.5) + "你好世界" (4 CJK → 2.67) = 5.17 → floor 5.
		{"mixed_ascii_cjk", "def hello 你好世界", 4, 8},

		// Single-rune min-clamp cases.
		{"single_cjk", "你", 1, 1},   // 0.667 → floor 0 → clamp 1
		{"single_emoji", "😀", 1, 1}, // 1.0 → floor 1
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := h.Count("any-model", tc.text)
			if got < tc.minTokens || got > tc.maxTokens {
				t.Errorf("Count(%q) = %d, want in [%d, %d]", tc.text, got, tc.minTokens, tc.maxTokens)
			}
		})
	}
}

// TestHeuristic_ModelParamIgnored confirms the model parameter has no
// effect on the count. This is the documented contract — per-model
// calibration is a separate opt-in layer.
func TestHeuristic_ModelParamIgnored(t *testing.T) {
	h := HeuristicCounter{}
	text := "hello 你好 world 世界"
	first := h.Count("glm-5.2", text)
	second := h.Count("some-other-model", text)
	if first != second {
		t.Errorf("model param changed count: glm-5.2=%d, some-other-model=%d (must be identical)", first, second)
	}
}
