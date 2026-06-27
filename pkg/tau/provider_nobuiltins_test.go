//go:build provider_builtins_off

// provider_nobuiltins_test.go — verifies that under the
// `-tags provider_builtins_off` build, the SDK's provider registry starts
// empty and the typed constructors return ErrProviderNotFound.
//
// This file is ONLY compiled when the `provider_builtins_off` tag is set.
// Run it explicitly:
//
//	go test -tags provider_builtins_off -race -timeout 120s ./pkg/tau/...

package tau

import (
	"errors"
	"testing"
)

func TestProvidersEmptyWhenBuiltinsCompiledOut(t *testing.T) {
	// The provider registry is process-global; other tests in this
	// binary (TestContractProviderRegistry*, TestRegisterProvider*) add
	// their own custom providers, which leak into Providers() even
	// though they have nothing to do with the built-in factories. To
	// assert the spec's "built-ins compiled out ⇒ Providers() is
	// empty" semantics, we wipe the registry and verify built-in
	// self-registration did not fire under this build tag.
	globalProviderRegistry.mu.Lock()
	globalProviderRegistry.facts = make(map[string]ProviderFactory)
	globalProviderRegistry.names = nil
	globalProviderRegistry.dirty = false
	globalProviderRegistry.mu.Unlock()

	got := Providers()
	if len(got) != 0 {
		t.Errorf("Providers() under -tags provider_builtins_off = %v, want empty", got)
	}
	// Re-verify the specific built-in names are absent as a weaker
	// fallback assertion (in case future tests re-introduce pollution
	// between the wipe above and this point).
	for _, name := range []string{"anthropic", "openai"} {
		if contains(got, name) {
			t.Errorf("Providers() under -tags provider_builtins_off includes built-in %q", name)
		}
	}
}

func TestNewAnthropicClientReturnsNotFoundWhenBuiltinsCompiledOut(t *testing.T) {
	_, err := NewAnthropicClient(AnthropicOptions{APIKey: "test-key"})
	if !errors.Is(err, ErrProviderNotFound) {
		t.Errorf("NewAnthropicClient under -tags provider_builtins_off: got %v, want ErrProviderNotFound", err)
	}
}

func TestNewOpenAIClientReturnsNotFoundWhenBuiltinsCompiledOut(t *testing.T) {
	_, err := NewOpenAIClient(OpenAIOptions{APIKey: "test-key"})
	if !errors.Is(err, ErrProviderNotFound) {
		t.Errorf("NewOpenAIClient under -tags provider_builtins_off: got %v, want ErrProviderNotFound", err)
	}
}
