//go:build !provider_builtins_off

// provider_builtins.go — built-in LLM provider factories.
//
// This file is compiled in by default. It registers the anthropic and
// openai providers into the SDK's process-wide provider registry via an
// init() block, so a freshly-imported tau module is ready to call
// NewAnthropicClient / NewOpenAIClient without any further setup.
//
// Embedders that want to strip the built-ins (e.g., a binary that ships
// only a custom provider) build with `-tags provider_builtins_off`. In
// that mode Providers() returns an empty slice until the embedder calls
// RegisterProvider explicitly.
//
// tau deviates from the spec's prose ("-tags provider_builtins=off") by
// using the inverted-off tag `provider_builtins_off` instead. Go's
// boolean build-tag evaluation does not allow a positive tag (e.g.,
// `provider_builtins`) to be "on by default"; only the negated form
// achieves that. The user-facing disable invocation is therefore
// `-tags provider_builtins_off`, not `-tags provider_builtins=off`.
//
// Each factory maps the SDK-level ProviderOptions onto the internal
// provider's Options struct and delegates to its New constructor. The
// resulting *Client satisfies llm.LLMClient, which LLMClient aliases.

package tau

import (
	"github.com/taucentral/tau/internal/llm/provider/anthropic"
	"github.com/taucentral/tau/internal/llm/provider/openai"
)

func init() {
	MustRegisterProvider("anthropic", func(opts ProviderOptions) (LLMClient, error) {
		ao := anthropic.Options{
			APIKey:     opts.APIKey,
			BaseURL:    opts.BaseURL,
			HTTPClient: opts.Transport,
			Headers:    opts.Headers,
		}
		return anthropic.New(ao)
	})
	MustRegisterProvider("openai", func(opts ProviderOptions) (LLMClient, error) {
		oo := openai.Options{
			APIKey:     opts.APIKey,
			BaseURL:    opts.BaseURL,
			HTTPClient: opts.Transport,
			Headers:    opts.Headers,
		}
		return openai.New(oo)
	})
}
