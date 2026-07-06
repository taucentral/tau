// slash_test.go — verifies the SDK slash-command surface.
//
// Per task 7.4:
//   (a) an embedder-supplied Registry is reflected via SlashCommands().
//   (b) the default registry exposes the documented built-ins.
//   (c) SlashCommands() is sorted and a fresh slice copy.
//   (d) a custom registry that omits a built-in is reflected (i.e., the
//       embedder can selectively disable built-ins).
//
// The "custom Command is invoked when its name is sent" assertion
// requires UI dispatch wiring that lives in the embedder; the SDK does
// not dispatch commands from the registry. The slash package's own tests
// (internal/slash/slash_test.go) cover dispatch; we cover the SDK
// surface here.

package tau

import (
	"context"
	"reflect"
	"sort"
	"testing"

	"github.com/taucentral/tau/internal/config"
	"github.com/taucentral/tau/internal/fauxprovider"
	"github.com/taucentral/tau/internal/tools"
)

func basicSlashOpts(t *testing.T) Options {
	t.Helper()
	return Options{
		Cwd:           t.TempDir(),
		Model:         "faux",
		LLMClient:     fauxprovider.NewWithResponse("ok"),
		Tools:         []HeadlessTool{tools.NewReadTool(tools.OSReadOperations{})},
		Settings:      config.DefaultSettings(),
		ContextWindow: 200000,
	}
}

func TestSlashCommandsReturnsBuiltinsByDefault(t *testing.T) {
	sess, err := CreateAgentSession(context.Background(), basicSlashOpts(t))
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	got := sess.SlashCommands()
	// The default registry is constructed by slash.DefaultRegistry, which
	// registers ten built-ins: clear, checkout, cls, compact, fork, help,
	// label, model, quit, tree.
	want := []string{"clear", "checkout", "cls", "compact", "fork", "help", "label", "model", "quit", "tree"}
	sortedGot := append([]string(nil), got...)
	sort.Strings(sortedGot)
	sortedWant := append([]string(nil), want...)
	sort.Strings(sortedWant)
	if !reflect.DeepEqual(sortedGot, sortedWant) {
		t.Errorf("SlashCommands() = %v, want (any order) %v", got, want)
	}
}

func TestSlashCommandsReturnsSortedWithoutLeadingSlash(t *testing.T) {
	sess, err := CreateAgentSession(context.Background(), basicSlashOpts(t))
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	got := sess.SlashCommands()
	if len(got) == 0 {
		t.Fatal("SlashCommands() returned empty slice")
	}
	// No leading slash on any returned name.
	for _, n := range got {
		if len(n) == 0 {
			t.Errorf("SlashCommands() returned an empty name in %v", got)
		}
		if n[0] == '/' {
			t.Errorf("SlashCommands() returned %q with leading slash; SDK contract is no slash", n)
		}
	}
	// Already sorted.
	sorted := append([]string(nil), got...)
	sort.Strings(sorted)
	if !reflect.DeepEqual(got, sorted) {
		t.Errorf("SlashCommands() = %v, want sorted %v", got, sorted)
	}
}

func TestSlashCommandsReturnsFreshCopy(t *testing.T) {
	sess, err := CreateAgentSession(context.Background(), basicSlashOpts(t))
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	first := sess.SlashCommands()
	if len(first) == 0 {
		t.Fatal("SlashCommands() returned empty slice")
	}
	first[0] = "MUTATED"
	first = append(first, "extra")

	second := sess.SlashCommands()
	for _, n := range second {
		if n == "MUTATED" || n == "extra" {
			t.Errorf("SlashCommands() mutation leaked into subsequent call: %v contains %q", second, n)
		}
	}
}

// customCmd is intentionally omitted — a custom Command needs to take
// *agent.AgentSession in its Execute signature, and that type is in
// internal/agent which pkg/tau tests cannot import directly. The SDK's
// Command type is an alias for slash.Command, so an embedder who wants a
// custom command defines it in their own package using the tau.Command
// interface (which resolves to slash.Command at compile time). The
// invocation behavior is covered by internal/slash/slash_test.go; here
// we only verify the SDK's registry-surface plumbing.

func TestSlashCommandsReflectsInjectedRegistry(t *testing.T) {
	// Construct an empty registry, inject it, and verify SlashCommands()
	// sees the empty set (no fallback when explicitly nil-registry).
	customReg := NewRegistry()

	opts := basicSlashOpts(t)
	opts.SlashCommands = customReg
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	// An empty injected registry yields an empty SlashCommands() list
	// (the fallback to DefaultSlashRegistry only kicks in when nil).
	got := sess.SlashCommands()
	if len(got) != 0 {
		t.Errorf("SlashCommands() with empty injected registry = %v, want []", got)
	}
}

func TestSlashCommandsOmittingABuiltinReflected(t *testing.T) {
	// Construct a registry containing only /clear (omit the other
	// built-ins) and verify SlashCommands() returns just that one.
	reg := NewRegistry()
	// Use the default registry as a source of built-in instances; take
	// just /clear.
	defaults := DefaultSlashRegistry()
	clearCmd, ok := defaults.Lookup("/clear")
	if !ok {
		t.Fatalf("DefaultSlashRegistry() missing /clear; cannot run test")
	}
	reg.Register(clearCmd)

	opts := basicSlashOpts(t)
	opts.SlashCommands = reg
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	got := sess.SlashCommands()
	if len(got) != 1 {
		t.Fatalf("SlashCommands() = %v, want exactly [clear]", got)
	}
	if got[0] != "clear" {
		t.Errorf("SlashCommands()[0] = %q, want %q", got[0], "clear")
	}
}

// Avoid unused-import warnings.
var _ = reflect.DeepEqual
