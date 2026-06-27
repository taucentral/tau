//go:build !provider_builtins_off

// provider_builtins_test.go — verifies the built-in anthropic / openai
// factories self-register under the default build.
//
// This file is compiled in by default. The mirror tests for the
// `-tags provider_builtins_off` build live in provider_nobuiltins_test.go.

package tau

import (
	"testing"
)

func TestProvidersIncludesBuiltins(t *testing.T) {
	names := Providers()
	// Default build has anthropic and openai registered via init().
	want := []string{"anthropic", "openai"}
	for _, w := range want {
		if !contains(names, w) {
			t.Errorf("Providers() = %v, missing %q", names, w)
		}
	}
}

func TestNewAnthropicClientSucceedsWithAPIKey(t *testing.T) {
	// Built-in anthropic factory is registered under the default build.
	cli, err := NewAnthropicClient(AnthropicOptions{APIKey: "test-key"})
	if err != nil {
		t.Fatalf("NewAnthropicClient: %v", err)
	}
	if cli == nil {
		t.Fatal("NewAnthropicClient returned nil client")
	}
}

func TestNewAnthropicClientRejectsEmptyAPIKey(t *testing.T) {
	// Built-in factory rejects empty APIKey via anthropic.New.
	_, err := NewAnthropicClient(AnthropicOptions{APIKey: ""})
	if err == nil {
		t.Error("NewAnthropicClient('') got nil error, want one from anthropic.New")
	}
}

func TestNewOpenAIClientSucceedsWithAPIKey(t *testing.T) {
	cli, err := NewOpenAIClient(OpenAIOptions{APIKey: "test-key"})
	if err != nil {
		t.Fatalf("NewOpenAIClient: %v", err)
	}
	if cli == nil {
		t.Fatal("NewOpenAIClient returned nil client")
	}
}
