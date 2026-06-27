// provider.go — public SDK provider registry.
//
// tau deviates from pi by using Go's compile-time `init()` registry for
// LLM providers rather than Node's runtime `require()` discovery. The
// registry decouples "how do I add a provider" from "edit tau's source":
// an embedder calls RegisterProvider("acme", factory) and the resulting
// client is immediately available via LookupProvider("acme").
//
// Built-in providers (anthropic, openai) self-register via init() in
// provider_builtins.go under build tag `provider_builtins` (on by default).
// Embedders that want to strip built-ins (e.g., a binary shipping only a
// custom provider) build with `-tags provider_builtins=off`.
//
// The registry is process-wide and safe for concurrent use. Duplicate
// registrations are rejected with ErrProviderAlreadyRegistered so a
// plugin-loaded provider cannot silently shadow a built-in.

package tau

import (
	"errors"
	"fmt"
	"net/http"
	"sort"
	"sync"

	"github.com/coevin/tau/internal/llm"
)

// ProviderOptions is the SDK-level bundle passed to a ProviderFactory.
// It carries every field the built-in anthropic/openai factories know
// how to consume. Custom providers may pick a subset; unknown fields are
// ignored.
type ProviderOptions struct {
	// APIKey is the resolved credential. Required for built-ins.
	APIKey string

	// BaseURL overrides the provider's default API endpoint. Useful for
	// OpenAI-compatible providers (DeepSeek, Ollama, LM Studio,
	// OpenRouter, Groq) and for pointing at a local proxy.
	BaseURL string

	// Headers are appended to every request after the auth headers.
	Headers map[string]string

	// Transport overrides the underlying *http.Client. If nil, the
	// provider uses http.DefaultClient. Callers SHOULD configure
	// timeouts via this field for production use.
	Transport *http.Client
}

// ProviderFactory constructs an LLMClient from ProviderOptions. Every
// registered provider supplies one. The factory MUST validate the
// options (e.g., reject an empty APIKey when one is required) and return
// a descriptive error.
type ProviderFactory func(opts ProviderOptions) (LLMClient, error)

// ErrProviderAlreadyRegistered is returned by RegisterProvider when
// another factory is already registered under the same name. The
// existing registration is preserved.
var ErrProviderAlreadyRegistered = errors.New("tau: provider already registered")

// ErrProviderNotFound is returned by LookupProvider,
// NewAnthropicClient, and NewOpenAIClient when no factory is registered
// under the requested name. The most common cause is building with
// `-tags provider_builtins=off` without manually registering the
// provider first.
var ErrProviderNotFound = errors.New("tau: provider not found")

// providerRegistry is the process-wide map of provider name → factory.
// Built-ins self-register via init(); embedders call RegisterProvider at
// any time. The RWMutex guards both the map and the names cache.
type providerRegistry struct {
	mu     sync.RWMutex
	facts  map[string]ProviderFactory
	names  []string // sorted, kept in sync with facts
	dirty  bool
}

var globalProviderRegistry = &providerRegistry{
	facts: make(map[string]ProviderFactory),
}

// RegisterProvider adds factory under name. The first caller wins:
// subsequent calls for the same name return ErrProviderAlreadyRegistered
// and leave the original registration intact. RegisterProvider is safe
// for concurrent use and may be called from init() blocks or at runtime.
func RegisterProvider(name string, factory ProviderFactory) error {
	if name == "" {
		return errors.New("tau: provider name is required")
	}
	if factory == nil {
		return errors.New("tau: provider factory is nil")
	}
	globalProviderRegistry.mu.Lock()
	defer globalProviderRegistry.mu.Unlock()
	if _, exists := globalProviderRegistry.facts[name]; exists {
		return fmt.Errorf("%w: %q", ErrProviderAlreadyRegistered, name)
	}
	globalProviderRegistry.facts[name] = factory
	globalProviderRegistry.dirty = true
	return nil
}

// MustRegisterProvider is the convenience wrapper that panics on
// registration error. Intended for use in init() blocks where failure
// indicates a programming bug.
func MustRegisterProvider(name string, factory ProviderFactory) {
	if err := RegisterProvider(name, factory); err != nil {
		panic(err)
	}
}

// LookupProvider returns the factory registered under name, or
// ErrProviderNotFound.
func LookupProvider(name string) (ProviderFactory, error) {
	globalProviderRegistry.mu.RLock()
	defer globalProviderRegistry.mu.RUnlock()
	f, ok := globalProviderRegistry.facts[name]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrProviderNotFound, name)
	}
	return f, nil
}

// Providers returns the registered provider names, sorted
// lexicographically. The returned slice is a fresh copy; callers may
// mutate freely.
func Providers() []string {
	globalProviderRegistry.mu.Lock()
	defer globalProviderRegistry.mu.Unlock()
	if globalProviderRegistry.dirty {
		names := make([]string, 0, len(globalProviderRegistry.facts))
		for n := range globalProviderRegistry.facts {
			names = append(names, n)
		}
		sort.Strings(names)
		globalProviderRegistry.names = names
		globalProviderRegistry.dirty = false
	}
	out := make([]string, len(globalProviderRegistry.names))
	copy(out, globalProviderRegistry.names)
	return out
}

// resolveProvider is the internal helper used by NewAnthropicClient and
// NewOpenAIClient. It looks up the factory and invokes it with opts.
func resolveProvider(name string, opts ProviderOptions) (llm.LLMClient, error) {
	f, err := LookupProvider(name)
	if err != nil {
		return nil, err
	}
	return f(opts)
}
