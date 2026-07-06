package tools

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/taucentral/tau/internal/llm"
)

// ErrUnknownTool is returned by Lookup when no tool with the given name is
// registered. The agent loop catches this and synthesizes a ToolResult
// with IsError=true so the model sees a clear error rather than a generic
// "internal error".
var ErrUnknownTool = errors.New("tools: unknown tool")

// ErrDuplicateTool is returned by Register when another tool with the same
// name is already registered. First-registration wins so plugin-load order
// is deterministic.
var ErrDuplicateTool = errors.New("tools: duplicate tool name")

// Registry maps tool names to HeadlessTool implementations. Lookup is O(1)
// via a map; Schemas() returns the schemas for tools that should render
// this turn (eager tools always render; LazyHeadlessTool instances render
// when their hydration triggers match) sorted by name for deterministic
// LLM request construction.
//
// The Registry stores HeadlessTool rather than Tool so a headless embedder
// (one who does not implement the TUI rendering methods) can register and
// execute tools without satisfying the larger Tool interface. The TUI
// type-asserts to Tool when it needs to render.
//
// Lazy tool hydration: when a registered tool satisfies LazyHeadlessTool,
// the runtime type-asserts in Schemas and evaluates the tag-based
// hydration triggers from TurnSignals. Tools that do not satisfy
// LazyHeadlessTool always render eagerly. The runtime calls RecordToolUse
// after each successful Execute so the recent-use map stays current.
//
// The zero value is NOT usable; construct via NewRegistry.
type Registry struct {
	mu    sync.RWMutex
	tools map[string]HeadlessTool

	// recentUse tracks turns-since-last-use per tool name. Incremented
	// across every RecordToolUse call; the entry for the just-used tool
	// resets to 0. Absent means "never used this session".
	recentUse map[string]int

	// lastRendered tracks which tool names rendered in the most recent
	// Schemas call. Used by HiddenTools for observability and by the
	// runtime's first-call miss detection (see agent.runOneTool).
	lastRendered map[string]bool
}

// NewRegistry returns an empty Registry.
func NewRegistry() *Registry {
	return &Registry{
		tools:        make(map[string]HeadlessTool),
		recentUse:    make(map[string]int),
		lastRendered: make(map[string]bool),
	}
}

// Register adds a tool. Returns ErrDuplicateTool if a tool with the same
// name is already registered.
func (r *Registry) Register(t HeadlessTool) error {
	if t == nil {
		return errors.New("tools: Register(nil)")
	}
	if t.Name() == "" {
		return errors.New("tools: tool has empty Name")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.tools[t.Name()]; exists {
		return fmt.Errorf("%w: %q", ErrDuplicateTool, t.Name())
	}
	r.tools[t.Name()] = t
	return nil
}

// MustRegister is a convenience wrapper that panics on registration error.
// Intended for use in init() blocks where failure indicates a programming
// bug (duplicate name, nil tool).
func (r *Registry) MustRegister(t HeadlessTool) {
	if err := r.Register(t); err != nil {
		panic(err)
	}
}

// Lookup returns the tool registered under name, or ErrUnknownTool.
func (r *Registry) Lookup(name string) (HeadlessTool, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	t, ok := r.tools[name]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnknownTool, name)
	}
	return t, nil
}

// Names returns the registered tool names sorted alphabetically.
func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]string, 0, len(r.tools))
	for name := range r.tools {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// Schemas returns the llm.ToolSchema for every tool that should render
// this turn, sorted by tool name. The agent loop embeds this in every
// LLM request.
//
// Eager tools (HeadlessTool that do not satisfy LazyHeadlessTool) always
// render via their Parameters() method. Lazy tools render when a
// hydration trigger matches; non-matching lazy tools are omitted from
// the output entirely (see design.md D1.4). The runtime's first-call
// miss fallback catches tool calls for hidden lazy tools (see
// agent.runOneTool).
//
// The first-call miss path uses LastRendered to determine whether a
// ToolUse for a lazy tool constitutes a miss.
func (r *Registry) Schemas(ctx context.Context, signals TurnSignals) ([]llm.ToolSchema, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	names := make([]string, 0, len(r.tools))
	for name := range r.tools {
		names = append(names, name)
	}
	sort.Strings(names)

	out := make([]llm.ToolSchema, 0, len(names))
	rendered := make(map[string]bool, len(names))
	alwaysSet := stringSet(signals.AlwaysRender)
	recentSet := stringSet(signals.RecentToolCalls)

	for _, name := range names {
		t := r.tools[name]

		// Type-assert against LazyHeadlessTool. The ok=false path is
		// the eager fast path: render via Parameters() exactly as the
		// pre-lazy-registration Schemas() did.
		lazy, isLazy := t.(LazyHeadlessTool)
		if !isLazy {
			out = append(out, eagerSchema(t))
			rendered[name] = true
			continue
		}

		// Lazy path: evaluate triggers in defined order.
		if !shouldRenderLazy(name, lazy, signals, alwaysSet, recentSet, r.recentUse) {
			// Hidden this turn; do not include in output.
			continue
		}

		// Trigger matched; render via Hydrate.
		schema, err := lazy.Hydrate(ctx)
		if err != nil {
			return nil, fmt.Errorf("tools: hydrate %q: %w", name, err)
		}
		raw, err := schema.MarshalJSON()
		if err != nil {
			// Schema.MarshalJSON only fails on a structurally invalid
			// schema; surface as a programming bug rather than
			// silently sending an empty schema.
			return nil, fmt.Errorf("tools: hydrate %q: marshal schema: %w", name, err)
		}
		out = append(out, llm.ToolSchema{
			Name:        t.Name(),
			Description: t.Description(),
			Parameters:  raw,
		})
		rendered[name] = true
	}

	r.lastRendered = rendered
	return out, nil
}

// LastRendered returns the set of tool names that rendered in the most
// recent Schemas(ctx, signals) call. The runtime uses this in runOneTool
// to detect first-call misses (a lazy tool the model called even though
// its schema was not rendered this turn). Tests use it for observability.
//
// The returned map is a defensive copy; callers may mutate freely.
func (r *Registry) LastRendered() map[string]bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make(map[string]bool, len(r.lastRendered))
	for k, v := range r.lastRendered {
		out[k] = v
	}
	return out
}

// RecordToolUse updates the recent-use map. The runtime calls this after
// each successful tool Execute. Tools not called recently accumulate
// turns-since-last-use; tools never called are absent from the map.
//
// Per design.md D1.8, recency is per-session and does not persist across
// sessions. The map is keyed by tool name and tracks turns since last
// use: each RecordToolUse call increments every existing entry, then
// resets the just-used tool's entry to 0.
func (r *Registry) RecordToolUse(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.recentUse == nil {
		r.recentUse = make(map[string]int)
	}
	for k := range r.recentUse {
		r.recentUse[k]++
	}
	r.recentUse[name] = 0
}

// HiddenTools returns the sorted list of registered LazyHeadlessTool
// names that the heuristic would HIDE this turn. Useful for observability
// (e.g. logging which tools were omitted from Request.Tools) and for
// tests exercising the trigger matrix.
//
// This method re-evaluates the triggers against the supplied signals
// without calling Hydrate. It is safe to call concurrently with Schemas.
func (r *Registry) HiddenTools(ctx context.Context, signals TurnSignals) []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	alwaysSet := stringSet(signals.AlwaysRender)
	recentSet := stringSet(signals.RecentToolCalls)
	out := make([]string, 0)
	for name, t := range r.tools {
		lazy, ok := t.(LazyHeadlessTool)
		if !ok {
			continue
		}
		if !shouldRenderLazy(name, lazy, signals, alwaysSet, recentSet, r.recentUse) {
			out = append(out, name)
		}
	}
	sort.Strings(out)
	return out
}

// RecentUseMap returns a defensive copy of the recent-use map. The map
// is maintained by RecordToolUse (called by the runtime after each
// successful Execute) and consulted by the lazy heuristic as one of the
// recency triggers. Test-only helper.
func (r *Registry) RecentUseMap() map[string]int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make(map[string]int, len(r.recentUse))
	for k, v := range r.recentUse {
		out[k] = v
	}
	return out
}

// shouldRenderLazy evaluates the hydration triggers in defined order:
//  1. tag.AlwaysRender == true
//  2. tool name in signals.AlwaysRender
//  3. tool name in signals.RecentToolCalls (state-tree-derived recency)
//  4. registry's internal recentUse[name] <= signals.RecentUseWindow
//     (incremental recency maintained via RecordToolUse)
//  5. signals.Mode is HydrationModeOff or HydrationModeModelDeclared
//     (both force eager rendering: Off bypasses the heuristic, and
//     ModelDeclared's selection round-trip is not yet implemented so
//     for v1 it renders everything — see tool.go's doc comment)
//  6. tag.Intent substring-match against signals.UserMessage (case-insensitive)
//
// Returns true if any trigger matches; false otherwise.
//
// Triggers 3 and 4 are dual recency trackers: RecentToolCalls is derived
// per turn from the state tree by the agent layer; recentUse is maintained
// incrementally by RecordToolUse. Either match is sufficient.
func shouldRenderLazy(name string, lazy LazyHeadlessTool, signals TurnSignals, alwaysSet, recentSet map[string]bool, recentUse map[string]int) bool {
	tag := lazy.Tag()
	if tag.AlwaysRender {
		return true
	}
	if alwaysSet[name] {
		return true
	}
	if recentSet[name] {
		return true
	}
	// Consult the registry's incremental recency tracker. A tool that
	// was used within the last RecentUseWindow turns renders. Absent
	// means "never used this session" → does not match.
	if turns, ok := recentUse[name]; ok {
		if signals.RecentUseWindow > 0 && turns <= signals.RecentUseWindow {
			return true
		}
	}
	if signals.Mode == HydrationModeOff || signals.Mode == HydrationModeModelDeclared {
		return true
	}
	if tag.Intent != "" && signals.UserMessage != "" {
		if strings.Contains(
			strings.ToLower(signals.UserMessage),
			strings.ToLower(tag.Intent),
		) {
			return true
		}
	}
	return false
}

// eagerSchema renders an eager HeadlessTool's schema via Parameters(),
// matching the pre-lazy-registration behavior. Kept as a helper so the
// eager path stays visually separate from the lazy path.
func eagerSchema(t HeadlessTool) llm.ToolSchema {
	schema := t.Parameters()
	raw, err := schema.MarshalJSON()
	if err != nil {
		// Schema.MarshalJSON only fails on a structurally invalid
		// schema, which is a programming bug. Surface it as an
		// empty schema so the LLM sees "no parameters" rather than
		// a broken request.
		raw = []byte(`{"type":"object","properties":{}}`)
	}
	return llm.ToolSchema{
		Name:        t.Name(),
		Description: t.Description(),
		Parameters:  raw,
	}
}

// stringSet converts a slice of strings into a set for fast lookup.
// nil input returns an empty set.
func stringSet(in []string) map[string]bool {
	out := make(map[string]bool, len(in))
	for _, s := range in {
		out[s] = true
	}
	return out
}

// Len returns the number of registered tools.
func (r *Registry) Len() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.tools)
}
