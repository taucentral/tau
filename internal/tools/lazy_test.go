// lazy_test.go — tests for LazyHeadlessTool hydration triggers.
//
// Covers the nine trigger-matrix scenarios from tasks.md §6.1, the
// concurrency assertion from §6.2, and the eager-only contract
// assertion from §6.3. Stubs exercise both the eager (HeadlessTool
// only) and lazy (HeadlessTool + LazyHeadlessTool) paths through
// Registry.Schemas.

package tools

import (
	"context"
	"sort"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/coevin/tau/internal/llm"
	"github.com/invopop/jsonschema"
)

// ---- Stub tools ----

// eagerStub implements HeadlessTool only. Used to verify the eager
// fast path always renders regardless of TurnSignals.
type eagerStub struct {
	name string
	desc string
}

func (e *eagerStub) Name() string                  { return e.name }
func (e *eagerStub) Description() string           { return e.desc }
func (e *eagerStub) Parameters() jsonschema.Schema { return jsonschema.Schema{Type: "object"} }
func (e *eagerStub) Execute(ctx context.Context, call ToolCall) (ToolResult, error) {
	return NewTextResult("eager:" + e.name), nil
}

// lazyStub implements both HeadlessTool and LazyHeadlessTool. The
// hydrates counter tracks Hydrate invocations so the first-call miss
// test can verify the runtime hydrated on a miss.
type lazyStub struct {
	name     string
	desc     string
	tag      ToolTag
	schema   jsonschema.Schema
	hydrates int32 // atomic; incremented per Hydrate call
}

func (l *lazyStub) Name() string                  { return l.name }
func (l *lazyStub) Description() string           { return l.desc }
func (l *lazyStub) Parameters() jsonschema.Schema { return l.schema }
func (l *lazyStub) Execute(ctx context.Context, call ToolCall) (ToolResult, error) {
	return NewTextResult("lazy:" + l.name), nil
}
func (l *lazyStub) Tag() ToolTag { return l.tag }
func (l *lazyStub) Hydrate(ctx context.Context) (jsonschema.Schema, error) {
	atomic.AddInt32(&l.hydrates, 1)
	return l.schema, nil
}

func (l *lazyStub) hydrateCount() int {
	return int(atomic.LoadInt32(&l.hydrates))
}

// baseSignals returns a TurnSignals value with sensible defaults for
// tests: heuristic mode, window=5, empty user message and recency.
func baseSignals() TurnSignals {
	return TurnSignals{
		Mode:            HydrationModeHeuristic,
		RecentUseWindow: 5,
	}
}

// schemaNames extracts the Name field from each ToolSchema for
// easy assertion. Returns a sorted slice.
func schemaNames(schemas []llm.ToolSchema) []string {
	out := make([]string, 0, len(schemas))
	for _, s := range schemas {
		out = append(out, s.Name)
	}
	sort.Strings(out)
	return out
}

// ---- Task 6.1: Trigger-matrix tests ----

// Case 1: registry with only HeadlessTool instances — all render.
func TestLazy_AllEagerRendersAll(t *testing.T) {
	r := NewRegistry()
	r.MustRegister(&eagerStub{name: "alpha", desc: "a"})
	r.MustRegister(&eagerStub{name: "beta", desc: "b"})

	schemas, err := r.Schemas(context.Background(), baseSignals())
	if err != nil {
		t.Fatalf("Schemas: %v", err)
	}
	if len(schemas) != 2 {
		t.Fatalf("expected 2 schemas, got %d", len(schemas))
	}
	names := schemaNames(schemas)
	if names[0] != "alpha" || names[1] != "beta" {
		t.Errorf("got %v, want [alpha beta]", names)
	}
}

// Case 2: one LazyHeadlessTool with AlwaysRender=true — renders every turn.
func TestLazy_AlwaysRenderTag(t *testing.T) {
	r := NewRegistry()
	ls := &lazyStub{
		name:   "always-tool",
		tag:    ToolTag{AlwaysRender: true},
		schema: jsonschema.Schema{Type: "object"},
	}
	r.MustRegister(ls)

	// Turn 1: no user message, no recency — AlwaysRender forces render.
	schemas, err := r.Schemas(context.Background(), baseSignals())
	if err != nil {
		t.Fatalf("Schemas turn 1: %v", err)
	}
	if len(schemas) != 1 || schemas[0].Name != "always-tool" {
		t.Fatalf("turn 1: expected [always-tool], got %+v", schemas)
	}

	// Turn 2: still renders.
	schemas, err = r.Schemas(context.Background(), baseSignals())
	if err != nil {
		t.Fatalf("Schemas turn 2: %v", err)
	}
	if len(schemas) != 1 {
		t.Fatalf("turn 2: expected 1 schema, got %d", len(schemas))
	}
}

// Case 3: lazy tool used 2 turns ago, RecentUseWindow=5 — renders.
// Simulated by including the tool name in signals.RecentToolCalls.
func TestLazy_RecentUseWithinWindow(t *testing.T) {
	r := NewRegistry()
	ls := &lazyStub{
		name:   "recent-tool",
		tag:    ToolTag{Intent: "unrelated"}, // no intent match
		schema: jsonschema.Schema{Type: "object"},
	}
	r.MustRegister(ls)

	sig := baseSignals()
	sig.RecentToolCalls = []string{"recent-tool"} // used 2 turns ago, within window=5

	schemas, err := r.Schemas(context.Background(), sig)
	if err != nil {
		t.Fatalf("Schemas: %v", err)
	}
	if len(schemas) != 1 || schemas[0].Name != "recent-tool" {
		t.Fatalf("expected [recent-tool], got %+v", schemas)
	}
}

// Case 4: lazy tool used 10 turns ago, RecentUseWindow=5 — hidden.
// Simulated by NOT including the tool name in signals.RecentToolCalls.
func TestLazy_RecentUseOutsideWindow(t *testing.T) {
	r := NewRegistry()
	ls := &lazyStub{
		name:   "stale-tool",
		tag:    ToolTag{Intent: "unrelated"},
		schema: jsonschema.Schema{Type: "object"},
	}
	r.MustRegister(ls)

	// signals carry an empty RecentToolCalls (tool used too long ago).
	sig := baseSignals()

	schemas, err := r.Schemas(context.Background(), sig)
	if err != nil {
		t.Fatalf("Schemas: %v", err)
	}
	if len(schemas) != 0 {
		t.Fatalf("expected 0 schemas (hidden), got %d: %+v", len(schemas), schemas)
	}
}

// Case 5: lazy tool whose Intent keyword appears in the user message — renders.
func TestLazy_IntentKeywordMatch(t *testing.T) {
	r := NewRegistry()
	ls := &lazyStub{
		name:   "search-tool",
		tag:    ToolTag{Intent: "search"},
		schema: jsonschema.Schema{Type: "object"},
	}
	r.MustRegister(ls)

	sig := baseSignals()
	sig.UserMessage = "please search the codebase for references"

	schemas, err := r.Schemas(context.Background(), sig)
	if err != nil {
		t.Fatalf("Schemas: %v", err)
	}
	if len(schemas) != 1 || schemas[0].Name != "search-tool" {
		t.Fatalf("expected [search-tool], got %+v", schemas)
	}
}

// Case 5b: intent match is case-insensitive.
func TestLazy_IntentKeywordCaseInsensitive(t *testing.T) {
	r := NewRegistry()
	ls := &lazyStub{
		name:   "search-tool",
		tag:    ToolTag{Intent: "Search"},
		schema: jsonschema.Schema{Type: "object"},
	}
	r.MustRegister(ls)

	sig := baseSignals()
	sig.UserMessage = "please SEARCH the codebase"

	schemas, err := r.Schemas(context.Background(), sig)
	if err != nil {
		t.Fatalf("Schemas: %v", err)
	}
	if len(schemas) != 1 {
		t.Fatalf("expected 1 schema, got %d", len(schemas))
	}
}

// Case 6: lazy tool with no match — hidden.
func TestLazy_NoMatchHidden(t *testing.T) {
	r := NewRegistry()
	ls := &lazyStub{
		name:   "niche-tool",
		tag:    ToolTag{Intent: "deploy-kubernetes"},
		schema: jsonschema.Schema{Type: "object"},
	}
	r.MustRegister(ls)

	sig := baseSignals()
	sig.UserMessage = "what is the weather today" // no intent match

	hidden := r.HiddenTools(context.Background(), sig)
	if len(hidden) != 1 || hidden[0] != "niche-tool" {
		t.Errorf("HiddenTools = %v, want [niche-tool]", hidden)
	}

	schemas, err := r.Schemas(context.Background(), sig)
	if err != nil {
		t.Fatalf("Schemas: %v", err)
	}
	if len(schemas) != 0 {
		t.Errorf("expected 0 schemas, got %d", len(schemas))
	}
}

// Case 7: first-call miss — model calls a hidden lazy tool.
// The registry hides the tool from Schemas output; the runtime's
// first-call miss path then hydrates directly. This test verifies
// the registry primitives (LastRendered excludes hidden tools, and
// Hydrate is callable). Full runtime integration is tested in the
// agent package.
func TestLazy_FirstCallMissRegistryPrimitives(t *testing.T) {
	r := NewRegistry()
	ls := &lazyStub{
		name:   "miss-tool",
		tag:    ToolTag{Intent: "never-matches"},
		schema: jsonschema.Schema{Type: "object"},
	}
	r.MustRegister(ls)

	sig := baseSignals()
	sig.UserMessage = "unrelated message"

	// Schemas hides the tool.
	schemas, err := r.Schemas(context.Background(), sig)
	if err != nil {
		t.Fatalf("Schemas: %v", err)
	}
	if len(schemas) != 0 {
		t.Fatalf("expected 0 schemas, got %d", len(schemas))
	}

	// LastRendered does NOT include the hidden tool.
	rendered := r.LastRendered()
	if rendered["miss-tool"] {
		t.Errorf("LastRendered should not include miss-tool")
	}

	// The runtime's first-call miss path calls Hydrate directly when
	// !rendered[name]. Verify Hydrate works and increments the counter.
	if ls.hydrateCount() != 0 {
		t.Errorf("hydrateCount before miss = %d, want 0", ls.hydrateCount())
	}
	_, err = ls.Hydrate(context.Background())
	if err != nil {
		t.Fatalf("Hydrate: %v", err)
	}
	if ls.hydrateCount() != 1 {
		t.Errorf("hydrateCount after miss = %d, want 1", ls.hydrateCount())
	}
}

// Case 8: mix of eager and lazy tools — eager always render, lazy follow triggers.
func TestLazy_MixedEagerAndLazy(t *testing.T) {
	r := NewRegistry()
	r.MustRegister(&eagerStub{name: "eager-a", desc: "a"})
	r.MustRegister(&eagerStub{name: "eager-b", desc: "b"})
	r.MustRegister(&lazyStub{
		name:   "lazy-matched",
		tag:    ToolTag{Intent: "search"},
		schema: jsonschema.Schema{Type: "object"},
	})
	r.MustRegister(&lazyStub{
		name:   "lazy-hidden",
		tag:    ToolTag{Intent: "deploy"},
		schema: jsonschema.Schema{Type: "object"},
	})

	sig := baseSignals()
	sig.UserMessage = "please search for files"

	schemas, err := r.Schemas(context.Background(), sig)
	if err != nil {
		t.Fatalf("Schemas: %v", err)
	}
	// Expect: eager-a, eager-b, lazy-matched (3). lazy-hidden omitted.
	if len(schemas) != 3 {
		t.Fatalf("expected 3 schemas, got %d: %+v", len(schemas), schemas)
	}
	names := schemaNames(schemas)
	want := []string{"eager-a", "eager-b", "lazy-matched"}
	for i, w := range want {
		if names[i] != w {
			t.Errorf("names[%d] = %q, want %q", i, names[i], w)
		}
	}
}

// Case 9: HydrationMode=model_declared — for v1, behaves like Off
// (everything renders) since the selection round-trip is not yet
// implemented. See tool.go's HydrationModeModelDeclared doc comment.
func TestLazy_ModelDeclaredRendersAll(t *testing.T) {
	r := NewRegistry()
	r.MustRegister(&eagerStub{name: "eager-a", desc: "a"})
	r.MustRegister(&lazyStub{
		name:   "lazy-a",
		tag:    ToolTag{Intent: "never-matches"},
		schema: jsonschema.Schema{Type: "object"},
	})
	r.MustRegister(&lazyStub{
		name:   "lazy-b",
		tag:    ToolTag{Intent: "also-no-match"},
		schema: jsonschema.Schema{Type: "object"},
	})

	sig := baseSignals()
	sig.Mode = HydrationModeModelDeclared
	sig.UserMessage = "nothing relevant"

	schemas, err := r.Schemas(context.Background(), sig)
	if err != nil {
		t.Fatalf("Schemas: %v", err)
	}
	// model_declared behaves like Off for v1: all render.
	if len(schemas) != 3 {
		t.Fatalf("expected 3 schemas (model_declared renders all), got %d: %+v", len(schemas), schemas)
	}
}

// HydrationModeOff renders all lazy tools.
func TestLazy_OffModeRendersAll(t *testing.T) {
	r := NewRegistry()
	r.MustRegister(&lazyStub{
		name:   "lazy-a",
		tag:    ToolTag{Intent: "never-matches"},
		schema: jsonschema.Schema{Type: "object"},
	})

	sig := baseSignals()
	sig.Mode = HydrationModeOff

	schemas, err := r.Schemas(context.Background(), sig)
	if err != nil {
		t.Fatalf("Schemas: %v", err)
	}
	if len(schemas) != 1 {
		t.Fatalf("expected 1 schema (off mode), got %d", len(schemas))
	}
}

// Settings.Tools.AlwaysRender override forces a lazy tool to render.
func TestLazy_SettingsAlwaysRender(t *testing.T) {
	r := NewRegistry()
	r.MustRegister(&lazyStub{
		name:   "forced-tool",
		tag:    ToolTag{Intent: "never-matches"},
		schema: jsonschema.Schema{Type: "object"},
	})

	sig := baseSignals()
	sig.AlwaysRender = []string{"forced-tool"}
	sig.UserMessage = "unrelated"

	schemas, err := r.Schemas(context.Background(), sig)
	if err != nil {
		t.Fatalf("Schemas: %v", err)
	}
	if len(schemas) != 1 || schemas[0].Name != "forced-tool" {
		t.Fatalf("expected [forced-tool], got %+v", schemas)
	}
}

// Case 3b: lazy tool recently used (via RecordToolUse incremental
// tracker), no signals.RecentToolCalls — renders via trigger #4.
// This verifies the registry's internal recentUse map IS consulted
// by the heuristic, not just maintained as dead code.
func TestLazy_RecentUseViaRecordToolUse(t *testing.T) {
	r := NewRegistry()
	ls := &lazyStub{
		name:   "tracked-tool",
		tag:    ToolTag{Intent: "unrelated"}, // no intent match
		schema: jsonschema.Schema{Type: "object"},
	}
	r.MustRegister(ls)

	// Record a recent use — recentUse["tracked-tool"] = 0.
	r.RecordToolUse("tracked-tool")

	// signals carry an EMPTY RecentToolCalls (the state-tree path
	// didn't populate it). The heuristic should still render via
	// the incremental recentUse map (trigger #4).
	sig := baseSignals()

	schemas, err := r.Schemas(context.Background(), sig)
	if err != nil {
		t.Fatalf("Schemas: %v", err)
	}
	if len(schemas) != 1 || schemas[0].Name != "tracked-tool" {
		t.Fatalf("expected [tracked-tool] via recentUse trigger, got %+v", schemas)
	}
}

// Case 4b: lazy tool whose RecordToolUse entry is outside the window
// — hidden via trigger #4 (recentUse[name] > RecentUseWindow).
func TestLazy_RecentUseRecordToolUseOutsideWindow(t *testing.T) {
	r := NewRegistry()
	ls := &lazyStub{
		name:   "stale-tracked",
		tag:    ToolTag{Intent: "unrelated"},
		schema: jsonschema.Schema{Type: "object"},
	}
	r.MustRegister(ls)

	// Record a use, then bump the counter past the window by
	// calling RecordToolUse on OTHER tools enough times.
	r.RecordToolUse("stale-tracked") // recentUse["stale-tracked"] = 0
	for i := 0; i < 6; i++ {
		r.RecordToolUse("other-tool")
	}
	// Now recentUse["stale-tracked"] = 6, window = 5 → outside.

	sig := baseSignals() // RecentUseWindow = 5

	schemas, err := r.Schemas(context.Background(), sig)
	if err != nil {
		t.Fatalf("Schemas: %v", err)
	}
	if len(schemas) != 0 {
		t.Fatalf("expected 0 schemas (stale via recentUse), got %d: %+v", len(schemas), schemas)
	}
}

// RecordToolUse maintains the recent-use map correctly.
func TestLazy_RecordToolUse(t *testing.T) {
	r := NewRegistry()
	r.MustRegister(&lazyStub{name: "a", schema: jsonschema.Schema{Type: "object"}})
	r.MustRegister(&lazyStub{name: "b", schema: jsonschema.Schema{Type: "object"}})

	// Initial state: empty recent-use map.
	m := r.RecentUseMap()
	if len(m) != 0 {
		t.Fatalf("initial RecentUseMap = %v, want empty", m)
	}

	// Use tool "a": resets a to 0.
	r.RecordToolUse("a")
	m = r.RecentUseMap()
	if m["a"] != 0 {
		t.Errorf("after RecordToolUse(a): a=%d, want 0", m["a"])
	}

	// Use tool "b": resets b to 0, increments a to 1.
	r.RecordToolUse("b")
	m = r.RecentUseMap()
	if m["a"] != 1 {
		t.Errorf("after RecordToolUse(b): a=%d, want 1", m["a"])
	}
	if m["b"] != 0 {
		t.Errorf("after RecordToolUse(b): b=%d, want 0", m["b"])
	}

	// Use "a" again: resets a to 0, increments b to 1.
	r.RecordToolUse("a")
	m = r.RecentUseMap()
	if m["a"] != 0 {
		t.Errorf("after RecordToolUse(a) again: a=%d, want 0", m["a"])
	}
	if m["b"] != 1 {
		t.Errorf("after RecordToolUse(a) again: b=%d, want 1", m["b"])
	}
}

// ---- Task 6.2: Concurrency test ----

// TestLazy_SchemasConcurrentNoRace runs two Schemas calls in parallel
// on a registry with mixed eager+lazy tools. Under -race, this verifies
// the registry's locking is correct.
func TestLazy_SchemasConcurrentNoRace(t *testing.T) {
	r := NewRegistry()
	for i := 0; i < 10; i++ {
		r.MustRegister(&eagerStub{name: "eager-" + string(rune('a'+i)), desc: "e"})
		r.MustRegister(&lazyStub{
			name:   "lazy-" + string(rune('a'+i)),
			tag:    ToolTag{Intent: "search"},
			schema: jsonschema.Schema{Type: "object"},
		})
	}

	sig := baseSignals()
	sig.UserMessage = "search something"
	sig.RecentToolCalls = []string{"lazy-a", "lazy-b"}

	var wg sync.WaitGroup
	var s1, s2 []llm.ToolSchema
	var err1, err2 error
	wg.Add(2)
	go func() {
		defer wg.Done()
		s1, err1 = r.Schemas(context.Background(), sig)
	}()
	go func() {
		defer wg.Done()
		s2, err2 = r.Schemas(context.Background(), sig)
	}()
	wg.Wait()

	if err1 != nil {
		t.Fatalf("goroutine 1: %v", err1)
	}
	if err2 != nil {
		t.Fatalf("goroutine 2: %v", err2)
	}
	n1 := schemaNames(s1)
	n2 := schemaNames(s2)
	if len(n1) != len(n2) {
		t.Fatalf("output length mismatch: %d vs %d", len(n1), len(n2))
	}
	for i := range n1 {
		if n1[i] != n2[i] {
			t.Fatalf("output mismatch at %d: %q vs %q", i, n1[i], n2[i])
		}
	}
}

// ---- Task 6.3: Contract assertion (eager-only equivalence) ----

// TestLazy_EagerOnlyMatchesOldSchemas verifies that a registry with
// only eager HeadlessTool instances produces identical Schemas output
// to the pre-lazy-registration behavior (the old no-arg Schemas()).
// The old behavior is simulated by a legacy function that calls
// Parameters() directly on each tool in sorted order.
func TestLazy_EagerOnlyMatchesOldSchemas(t *testing.T) {
	r := NewRegistry()
	stubs := []*eagerStub{
		{name: "read", desc: "read a file"},
		{name: "write", desc: "write a file"},
		{name: "bash", desc: "run a command"},
		{name: "glob", desc: "find files"},
	}
	for _, tl := range stubs {
		r.MustRegister(tl)
	}

	// New behavior: Schemas with TurnSignals.
	sig := TurnSignals{Mode: HydrationModeHeuristic}
	gotSchemas, err := r.Schemas(context.Background(), sig)
	if err != nil {
		t.Fatalf("Schemas: %v", err)
	}

	// Legacy behavior: iterate tools in sorted order, call Parameters().
	wantSchemas := legacySchemas(r)

	if len(gotSchemas) != len(wantSchemas) {
		t.Fatalf("length mismatch: got %d, want %d", len(gotSchemas), len(wantSchemas))
	}
	for i := range gotSchemas {
		if gotSchemas[i].Name != wantSchemas[i].Name {
			t.Errorf("[%d] Name: got %q, want %q", i, gotSchemas[i].Name, wantSchemas[i].Name)
		}
		if gotSchemas[i].Description != wantSchemas[i].Description {
			t.Errorf("[%d] Description: got %q, want %q", i, gotSchemas[i].Description, wantSchemas[i].Description)
		}
		// Parameters are raw JSON; compare as strings.
		gotParams := string(gotSchemas[i].Parameters)
		wantParams := string(wantSchemas[i].Parameters)
		if gotParams != wantParams {
			t.Errorf("[%d] Parameters: got %s, want %s", i, gotParams, wantParams)
		}
	}
}

// legacySchemas simulates the pre-lazy-registration Schemas() behavior:
// iterate tools in sorted order, call Parameters(), marshal to JSON.
func legacySchemas(r *Registry) []llm.ToolSchema {
	names := r.Names()
	out := make([]llm.ToolSchema, 0, len(names))
	for _, name := range names {
		t, _ := r.Lookup(name)
		schema := t.Parameters()
		raw, _ := schema.MarshalJSON()
		out = append(out, llm.ToolSchema{
			Name:        t.Name(),
			Description: t.Description(),
			Parameters:  raw,
		})
	}
	return out
}
