package tools

import (
	"errors"
	"fmt"
	"sort"
	"sync"

	"github.com/coevin/tau/internal/llm"
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
// via a map; Schemas() returns all schemas sorted by name for deterministic
// LLM request construction.
//
// The Registry stores HeadlessTool rather than Tool so a headless embedder
// (one who does not implement the TUI rendering methods) can register and
// execute tools without satisfying the larger Tool interface. The TUI
// type-asserts to Tool when it needs to render.
//
// The zero value is NOT usable; construct via NewRegistry.
type Registry struct {
	mu    sync.RWMutex
	tools map[string]HeadlessTool
}

// NewRegistry returns an empty Registry.
func NewRegistry() *Registry {
	return &Registry{tools: make(map[string]HeadlessTool)}
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

// Schemas returns the llm.ToolSchema for every registered tool, sorted by
// tool name. The agent loop embeds this in every LLM request.
func (r *Registry) Schemas() []llm.ToolSchema {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.tools))
	for name := range r.tools {
		names = append(names, name)
	}
	sort.Strings(names)
	out := make([]llm.ToolSchema, 0, len(names))
	for _, name := range names {
		t := r.tools[name]
		schema := t.Parameters()
		// jsonschema.Schema marshals cleanly to JSON; the agent loop
		// passes raw bytes to providers.
		raw, err := schema.MarshalJSON()
		if err != nil {
			// Schema.MarshalJSON only fails on a structurally invalid
			// schema, which is a programming bug. Surface it as an
			// empty schema so the LLM sees "no parameters" rather than
			// a broken request.
			raw = []byte(`{"type":"object","properties":{}}`)
		}
		out = append(out, llm.ToolSchema{
			Name:        t.Name(),
			Description: t.Description(),
			Parameters:  raw,
		})
	}
	return out
}

// Len returns the number of registered tools.
func (r *Registry) Len() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.tools)
}
