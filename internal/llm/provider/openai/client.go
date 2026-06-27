// client.go — OpenAI Chat Completions provider implementation of llm.LLMClient.
//
// The same client serves OpenAI proper and OpenAI-compatible providers
// (DeepSeek, Ollama, LM Studio, OpenRouter, Groq) via a configurable BaseURL.

package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/coevin/tau/internal/config"
	"github.com/coevin/tau/internal/llm"
)

// EnvVar is the environment variable consulted by ResolveAuth.
const EnvVar = "OPENAI_API_KEY"

// Options configures a Client.
type Options struct {
	APIKey      string
	BaseURL     string // default "https://api.openai.com/v1"
	HTTPClient  *http.Client
	Headers     map[string]string
	RetryPolicy llm.RetryPolicy
}

// Client implements llm.LLMClient for the OpenAI Chat Completions API.
type Client struct {
	apiKey      string
	baseURL     string
	http        *http.Client
	headers     map[string]string
	retryPolicy llm.RetryPolicy
}

// Compile-time check.
var _ llm.LLMClient = (*Client)(nil)

// New returns a Client ready to issue requests.
func New(opts Options) (*Client, error) {
	if opts.APIKey == "" {
		return nil, fmt.Errorf("openai: APIKey is required (use ResolveAuth to resolve)")
	}
	if opts.BaseURL == "" {
		opts.BaseURL = DefaultBaseURL
	}
	if opts.HTTPClient == nil {
		opts.HTTPClient = http.DefaultClient
	}
	if err := validateBaseURL(opts.BaseURL); err != nil {
		return nil, err
	}
	rp := opts.RetryPolicy
	if rp.MaxRetries == 0 && rp.BaseDelay == 0 && rp.MaxDelay == 0 {
		rp = llm.DefaultRetryPolicy()
	}
	return &Client{
		apiKey:      opts.APIKey,
		baseURL:     opts.BaseURL,
		http:        opts.HTTPClient,
		headers:     opts.Headers,
		retryPolicy: rp,
	}, nil
}

// validateBaseURL rejects obviously malformed base URLs.
func validateBaseURL(s string) error {
	if !strings.HasPrefix(s, "http://") && !strings.HasPrefix(s, "https://") {
		return fmt.Errorf("openai: BaseURL must start with http:// or https://, got %q", s)
	}
	return nil
}

// Stream issues a streaming completion request.
func (c *Client) Stream(ctx context.Context, req llm.Request) (<-chan llm.Delta, error) {
	body, err := marshalRequest(req)
	if err != nil {
		return nil, err
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}

	url := c.baseURL + "/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	for k, v := range c.headers {
		httpReq.Header.Set(k, v)
	}

	//
	//nolint:bodyclose // resp.Body is closed by the streaming goroutine below
	// (see defer on resp.Body.Close); llm.Do returns nil resp on error per
	// its contract at retry.go:96, so the success path is the only one that
	// owns a Body. The bodyclose linter cannot trace goroutine-local defers.
	resp, _, err := llm.Do(ctx, c.retryPolicy, func(ctx context.Context) (*http.Response, error) {
		httpReq.Body = io.NopCloser(bytes.NewReader(payload))
		return c.http.Do(httpReq)
	})
	if err != nil {
		return nil, translateErr(err)
	}

	ch := make(chan llm.Delta, 16)
	go func() {
		defer close(ch)
		defer func() {
			_ = resp.Body.Close()
		}()
		parser := newStreamParser()
		err := llm.ReadSSE(resp.Body, func(ev llm.SSEEvent) error {
			deltas, perr := parser.handle(ev)
			if perr != nil {
				return perr
			}
			for _, d := range deltas {
				select {
				case <-ctx.Done():
					return ctx.Err()
				case ch <- d:
				}
			}
			return nil
		})
		if err != nil {
			select {
			case <-ctx.Done():
				return
			case ch <- llm.Final{StopReason: llm.StopReasonError, Err: err}:
			}
		}
	}()

	return ch, nil
}

// translateErr wraps non-retryable HTTP errors with helpful context.
func translateErr(err error) error {
	var se *llm.HTTPStatusError
	if errors.As(err, &se) {
		return fmt.Errorf("openai: HTTP %d: %w (auth may be invalid; check API key)", se.Status, se)
	}
	return fmt.Errorf("openai: %w", err)
}

// AuthSource identifies where ResolveAuth found the credential.
type AuthSource string

const (
	AuthSourceExplicit AuthSource = "explicit"
	AuthSourceAuthJSON AuthSource = "auth.json"
	AuthSourceEnv      AuthSource = "env"
	AuthSourceSigil    AuthSource = "sigil"
)

// AuthResult carries the resolved API key and where it came from.
type AuthResult struct {
	APIKey string
	Source AuthSource
}

// ResolveAuth implements the four-step auth resolution chain for the OpenAI
// provider. The providerName argument lets callers reuse this for
// OpenAI-compatible providers (e.g., "deepseek", "groq").
func ResolveAuth(providerName, envVarName, explicit string, auth config.AuthStore) (AuthResult, error) {
	// Step 1: explicit (may be a sigil).
	if explicit != "" {
		if strings.HasPrefix(explicit, "$") || strings.HasPrefix(explicit, "!") {
			resolved, err := config.ResolveValue(explicit)
			if err != nil {
				return AuthResult{}, fmt.Errorf("openai: resolve sigil: %w", err)
			}
			if resolved != "" {
				return AuthResult{APIKey: resolved, Source: AuthSourceSigil}, nil
			}
		}
		if explicit != "" {
			return AuthResult{APIKey: explicit, Source: AuthSourceExplicit}, nil
		}
	}
	// Step 2: auth.json.
	if auth != nil {
		if key, ok := auth.Get(providerName); ok && key != "" {
			return AuthResult{APIKey: key, Source: AuthSourceAuthJSON}, nil
		}
	}
	// Step 3: env var.
	if envVarName != "" {
		if v := os.Getenv(envVarName); v != "" {
			return AuthResult{APIKey: v, Source: AuthSourceEnv}, nil
		}
	}
	if envVarName == "" {
		envVarName = EnvVar
	}
	return AuthResult{}, fmt.Errorf("openai: no credential found in explicit value, auth.json, or %s", envVarName)
}
