// provider_test.go — verifies the SDK provider registry.
//
// Covers (per task 5.5):
//   (a) Providers() includes anthropic/openai by default.
//   (b) -tags provider_builtins=off yields empty Providers() — see
//       provider_nobuiltins_test.go which is compiled under the inverted
//       tag.
//   (c) custom provider registration + lookup.
//   (d) duplicate registration returns ErrProviderAlreadyRegistered and
//       leaves the original intact.
//   (e) concurrent registration under -race is safe.
//   (f) NewAnthropicClient returns ErrProviderNotFound when the built-in
//       is disabled and not manually registered — see
//       provider_nobuiltins_test.go.
//   (g) ResolveAuth walks the explicit→auth.json→env→sigil chain.

package tau

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
	"testing"

	"github.com/coevin/tau/internal/config"
	"github.com/coevin/tau/internal/llm"
)

// fauxLLMClient is the minimal llm.LLMClient implementation used as a
// factory return value by registry tests. It does not need to actually
// stream — only satisfy the interface so the factory's return compiles.
type fauxLLMClient struct{}

func (fauxLLMClient) Stream(ctx context.Context, _ llm.Request) (<-chan llm.Delta, error) {
	ch := make(chan llm.Delta)
	close(ch)
	return ch, nil
}

func TestRegisterAndLookupCustomProvider(t *testing.T) {
	// Register under a unique name so we don't collide with built-ins or
	// other test cases. Use t.Name() to namespace.
	name := "custom-" + sanitize(t.Name())
	factory := func(opts ProviderOptions) (LLMClient, error) {
		if opts.APIKey == "" {
			return nil, errors.New("test: APIKey required")
		}
		return fauxLLMClient{}, nil
	}
	if err := RegisterProvider(name, factory); err != nil {
		t.Fatalf("RegisterProvider: %v", err)
	}

	got, err := LookupProvider(name)
	if err != nil {
		t.Fatalf("LookupProvider: %v", err)
	}
	if got == nil {
		t.Fatal("LookupProvider returned nil factory")
	}
	// Invoke the factory to confirm it round-trips.
	cli, err := got(ProviderOptions{APIKey: "k"})
	if err != nil {
		t.Fatalf("factory invocation: %v", err)
	}
	if cli == nil {
		t.Fatal("factory returned nil client")
	}

	// And confirm it shows up in Providers().
	if !contains(Providers(), name) {
		t.Errorf("Providers() = %v, missing %q", Providers(), name)
	}
}

func TestDuplicateRegistrationReturnsSentinelAndPreservesOriginal(t *testing.T) {
	name := "dup-" + sanitize(t.Name())
	original := func(ProviderOptions) (LLMClient, error) {
		return fauxLLMClient{}, nil
	}
	if err := RegisterProvider(name, original); err != nil {
		t.Fatalf("initial RegisterProvider: %v", err)
	}

	// Second registration under the same name must fail with the sentinel.
	second := func(ProviderOptions) (LLMClient, error) {
		return nil, errors.New("should never be called")
	}
	err := RegisterProvider(name, second)
	if !errors.Is(err, ErrProviderAlreadyRegistered) {
		t.Fatalf("RegisterProvider duplicate: got %v, want ErrProviderAlreadyRegistered", err)
	}

	// The original factory must still be the one registered.
	got, err := LookupProvider(name)
	if err != nil {
		t.Fatalf("LookupProvider after duplicate: %v", err)
	}
	// Verify by behaviour: invoking returns fauxLLMClient{} without
	// error; the second factory would have errored.
	cli, err := got(ProviderOptions{APIKey: "k"})
	if err != nil {
		t.Errorf("factory invocation after duplicate: got error %v, want the original factory preserved", err)
	}
	if cli == nil {
		t.Error("factory invocation returned nil after duplicate registration")
	}
}

func TestLookupUnknownProviderReturnsSentinel(t *testing.T) {
	_, err := LookupProvider("does-not-exist-" + sanitize(t.Name()))
	if !errors.Is(err, ErrProviderNotFound) {
		t.Errorf("LookupProvider unknown: got %v, want ErrProviderNotFound", err)
	}
}

func TestRegisterProviderValidation(t *testing.T) {
	if err := RegisterProvider("", func(ProviderOptions) (LLMClient, error) { return nil, nil }); err == nil {
		t.Error("RegisterProvider('') got nil error, want one")
	}
	if err := RegisterProvider("empty-factory-"+sanitize(t.Name()), nil); err == nil {
		t.Error("RegisterProvider(nil factory) got nil error, want one")
	}
}

func TestConcurrentRegisterProviderIsRaceFree(t *testing.T) {
	// Hammer RegisterProvider with unique names from many goroutines.
	// Run with -race to detect data races on the underlying map.
	const n = 50
	var wg sync.WaitGroup
	wg.Add(n)
	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			name := fmt.Sprintf("race-%d-%s", i, sanitize(t.Name()))
			factory := func(ProviderOptions) (LLMClient, error) { return fauxLLMClient{}, nil }
			if err := RegisterProvider(name, factory); err != nil {
				errs <- err
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("concurrent RegisterProvider: %v", err)
	}

	// All n names should be present.
	names := Providers()
	count := 0
	for i := 0; i < n; i++ {
		needle := fmt.Sprintf("race-%d-%s", i, sanitize(t.Name()))
		if contains(names, needle) {
			count++
		}
	}
	if count != n {
		t.Errorf("after concurrent registration: found %d/%d names", count, n)
	}
}

func TestProvidersReturnsFreshCopy(t *testing.T) {
	first := Providers()
	if len(first) == 0 {
		t.Fatal("Providers() returned empty slice")
	}
	// Mutate the returned slice.
	first[0] = "MUTATED"
	first = append(first, "extra")

	second := Providers()
	for _, n := range second {
		if n == "MUTATED" || n == "extra" {
			t.Errorf("Providers() mutation leaked: %v contains %q", second, n)
		}
	}
}

// --- NewAnthropicClient / NewOpenAIClient -----------------------------------
//
// The "happy path" tests for these constructors live in
// provider_builtins_test.go (gated to the default build), because they
// depend on the built-in factories being registered. The negative-path
// tests live in provider_nobuiltins_test.go (gated to the off build).

// --- ResolveAuth ------------------------------------------------------------

func TestResolveAuthExplicitWins(t *testing.T) {
	// Set up env + auth.json with DIFFERENT values; explicit must win.
	t.Setenv("ANTHROPIC_API_KEY", "env-value")
	auth := config.NewInMemoryAuthStore()
	_ = auth.Set("anthropic", "auth-json-value")

	got, source, err := ResolveAuth("anthropic", "explicit-value", auth)
	if err != nil {
		t.Fatalf("ResolveAuth: %v", err)
	}
	if got != "explicit-value" {
		t.Errorf("ResolveAuth explicit: got %q, want explicit-value (source=%s)", got, source)
	}
	if source != "explicit" {
		t.Errorf("ResolveAuth source: got %q, want explicit", source)
	}
}

func TestResolveAuthAuthJSONWhenNoExplicit(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "env-value")
	auth := config.NewInMemoryAuthStore()
	_ = auth.Set("anthropic", "auth-json-value")

	got, source, err := ResolveAuth("anthropic", "", auth)
	if err != nil {
		t.Fatalf("ResolveAuth: %v", err)
	}
	if got != "auth-json-value" {
		t.Errorf("ResolveAuth auth.json: got %q, want auth-json-value (source=%s)", got, source)
	}
	if source != "auth.json" {
		t.Errorf("ResolveAuth source: got %q, want auth.json", source)
	}
}

func TestResolveAuthEnvWhenNoExplicitOrAuthJSON(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "env-value")
	auth := config.NewInMemoryAuthStore() // empty

	got, source, err := ResolveAuth("anthropic", "", auth)
	if err != nil {
		t.Fatalf("ResolveAuth: %v", err)
	}
	if got != "env-value" {
		t.Errorf("ResolveAuth env: got %q, want env-value (source=%s)", got, source)
	}
	if source != "env" {
		t.Errorf("ResolveAuth source: got %q, want env", source)
	}
}

func TestResolveAuthSigilResolvesEnv(t *testing.T) {
	t.Setenv("MY_TEST_SIGIL_KEY", "sigil-resolved")
	auth := config.NewInMemoryAuthStore()

	got, source, err := ResolveAuth("anthropic", "$MY_TEST_SIGIL_KEY", auth)
	if err != nil {
		t.Fatalf("ResolveAuth sigil: %v", err)
	}
	if got != "sigil-resolved" {
		t.Errorf("ResolveAuth sigil: got %q, want sigil-resolved (source=%s)", got, source)
	}
	if source != "sigil" {
		t.Errorf("ResolveAuth source: got %q, want sigil", source)
	}
}

func TestResolveAuthErrorsWhenExhausted(t *testing.T) {
	// No explicit, no auth.json, no env. The OPENAI_API_KEY env var may
	// already be set in the test environment, so we explicitly clear it.
	os.Unsetenv("OPENAI_API_KEY")
	auth := config.NewInMemoryAuthStore()

	_, _, err := ResolveAuth("openai", "", auth)
	if err == nil {
		t.Error("ResolveAuth exhausted: got nil error, want one")
	}
}

func TestResolveAuthUnknownProvider(t *testing.T) {
	_, _, err := ResolveAuth("does-not-exist", "", nil)
	if !errors.Is(err, ErrProviderNotFound) {
		t.Errorf("ResolveAuth unknown provider: got %v, want ErrProviderNotFound", err)
	}
}

// sanitize strips run-id characters from a test name so the result is a
// valid element of the provider name space (alphanumeric + dash).
func sanitize(s string) string {
	out := make([]byte, 0, len(s))
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			out = append(out, byte(r))
		default:
			out = append(out, '-')
		}
	}
	return string(out)
}
