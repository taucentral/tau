//go:build !provider_builtins_off

// contract_builtins_test.go — contract assertions that depend on the
// built-in provider factories being registered. Compiled in by default;
// suppressed by `-tags provider_builtins_off`. The companion assertions
// for the builtins-off configuration live in provider_nobuiltins_test.go
// and contract_test.go (which is build-tag-agnostic).

package tau

import (
	"testing"
)

// TestContractProviderRegistryProvidersIncludesBuiltins is the
// contract-test assertion that the default build self-registers the
// anthropic and openai factories. It is gated because the built-in
// factories are compiled out under `-tags provider_builtins_off`; in
// that build TestProvidersEmptyWhenBuiltinsCompiledOut covers the
// inverse assertion.
func TestContractProviderRegistryProvidersIncludesBuiltins(t *testing.T) {
	// Under the default build, anthropic and openai are registered.
	for _, want := range []string{"anthropic", "openai"} {
		if !contains(Providers(), want) {
			t.Errorf("Providers() = %v, missing %q", Providers(), want)
		}
	}
}
