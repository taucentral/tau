// convenience_test.go — verifies DefaultSettings, BuiltinTools,
// NewFauxProvider, and the type-alias identity contract.
//
// Per task 6.3:
//   (a) BuiltinTools() returns tools named read/bash/edit/write/grep/find/ls.
//   (b) An embedder passing NewFauxProvider("hello") as LLMClient produces
//       the canned response on Run with zero network I/O.
//   (c) tau.Message and llm.Message are the same compile-time type.

package tau

import (
	"context"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/taucentral/tau/internal/config"
	"github.com/taucentral/tau/internal/llm"
)

func TestBuiltinToolsReturnsAllSeven(t *testing.T) {
	tools := BuiltinTools()
	if len(tools) != 7 {
		t.Fatalf("BuiltinTools() returned %d tools, want 7", len(tools))
	}
	names := make([]string, 0, len(tools))
	for _, tl := range tools {
		names = append(names, tl.Name())
	}
	sort.Strings(names)

	want := []string{"bash", "edit", "find", "grep", "ls", "read", "write"}
	for i, w := range want {
		if i >= len(names) || names[i] != w {
			t.Errorf("BuiltinTools() names[%d] = %q, want %q (full list: %v)", i, names, w, names)
		}
	}
}

func TestBuiltinToolsReturnsFreshSlice(t *testing.T) {
	first := BuiltinTools()
	if len(first) != 7 {
		t.Fatalf("BuiltinTools() returned %d, want 7", len(first))
	}
	// Mutate the returned slice.
	first[0] = nil
	first = append(first, nil)

	second := BuiltinTools()
	if len(second) != 7 {
		t.Errorf("BuiltinTools() after mutation returned %d, want 7 (mutation leaked)", len(second))
	}
	for _, tl := range second {
		if tl == nil {
			t.Errorf("BuiltinTools() returned a nil entry after mutation of prior result")
		}
	}
}

func TestNewFauxProviderProducesCannedResponse(t *testing.T) {
	client := NewFauxProvider("hello world")

	opts := Options{
		Cwd:           t.TempDir(),
		Model:         "faux",
		LLMClient:     client,
		Tools:         []HeadlessTool{},
		Settings:      config.DefaultSettings(),
		ContextWindow: 200000,
	}
	// Tools must contain at least one entry; use a single read tool to
	// satisfy the constructor.
	opts.Tools = BuiltinTools()[:1]

	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	events := sess.SubscribeTopics(TopicMessageUpdate)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := sess.Run(ctx, "anything"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Drain MessageUpdate events; collect any text that flows through. The
	// faux provider emits a single TextDelta with the canned response.
	var collected string
	deadline := time.After(2 * time.Second)
drain:
	for {
		select {
		case evt, ok := <-events:
			if !ok {
				break drain
			}
			if mu, ok := evt.(MessageUpdateEvent); ok {
				if td, ok := mu.Delta.(TextDelta); ok {
					collected += td.Text
				}
			}
		case <-deadline:
			break drain
		}
	}
	if collected == "" {
		t.Fatal("no TextDelta events received; the faux provider's canned response did not flow through the bus")
	}
	// The faux provider's response is "hello world" — the collected text
	// must contain it (possibly surrounded by other rendering artefacts
	// if the bus carries display-formatted versions; we only check the
	// raw delta path which should be exactly the canned string).
	if !contains([]string{collected}, "hello world") && collected != "hello world" {
		t.Errorf("collected TextDelta text = %q, want to contain %q", collected, "hello world")
	}
}

func TestNewFauxProviderEmptyReturnsDefaultResponse(t *testing.T) {
	// When called with no responses, returns a working client that emits
	// the default faux response. We don't assert on the exact text — that
	// is internal to fauxprovider; we only assert non-nil.
	client := NewFauxProvider()
	if client == nil {
		t.Fatal("NewFauxProvider() returned nil")
	}
}

// compile-time type-identity assertions. If any of these stops compiling,
// the SDK has accidentally wrapped an internal type instead of aliasing
// it. See spec §sdk-public-api "Type aliases" / Scenario: Alias identity.
var (
	_ = func(m Message) llm.Message { return m }
	_ = func(m llm.Message) Message { return m }

	_ = func(c ContentBlock) llm.ContentBlock { return c }
	_ = func(c llm.ContentBlock) ContentBlock { return c }

	_ = func(u ToolUse) llm.ToolUse { return u }
	_ = func(u llm.ToolUse) ToolUse { return u }

	_ = func(r Role) llm.Role { return r }
	_ = func(r llm.Role) Role { return r }

	_ = func(s Settings) config.Settings { return s }
	_ = func(s config.Settings) Settings { return s }
)

func TestTypeAliasIdentityCompiles(t *testing.T) {
	// The var declarations above are the test — they only compile when
	// the SDK type aliases resolve to the same underlying type. This
	// dummy assertion exists so the test runner counts the file.
	if t == nil {
		t.Fatal("unreachable")
	}
}

// ensure no leftover env leakage breaks the test process.
func init() {
	_ = os.Unsetenv("TAU_FAUX_SCRIPT")
}
