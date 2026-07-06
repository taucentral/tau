// client.go — Anthropic Messages provider implementation of llm.LLMClient.
//
// Auth resolution chain (per llm-client spec):
//   1. Explicit APIKey on the provider/model definition (passed in Options.APIKey)
//   2. auth.json entry for the provider (resolved by caller via AuthStore)
//   3. Provider-specific env var (ANTHROPIC_API_KEY)
//   4. $ENV_VAR or !shell command string (resolved by config.ResolveValue)
//
// The Client takes an already-resolved API key. The ResolveAuth helper
// implements the chain for callers (e.g., the Phase 8 runtime).

package anthropic

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

	"github.com/taucentral/tau/internal/config"
	"github.com/taucentral/tau/internal/llm"
)

// ProviderID is the canonical name for this provider.
const ProviderID = "anthropic"

// EnvVar is the environment variable consulted by ResolveAuth.
const EnvVar = "ANTHROPIC_API_KEY"

// Options configures a Client. All fields are optional except APIKey.
type Options struct {
	// APIKey is the resolved credential. Required.
	APIKey string
	// BaseURL defaults to DefaultBaseURL ("https://api.anthropic.com").
	BaseURL string
	// HTTPClient overrides the default *http.Client. If nil, http.DefaultClient
	// is used. Callers SHOULD configure timeouts via this field.
	HTTPClient *http.Client
	// Headers are appended to every request after the auth headers.
	Headers map[string]string
	// RetryPolicy controls the retry behavior for the initial POST. The SSE
	// stream itself is not retried — providers can't resume a stream.
	RetryPolicy llm.RetryPolicy
}

// Client implements llm.LLMClient for the Anthropic Messages API.
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
		return nil, fmt.Errorf("anthropic: APIKey is required (use ResolveAuth to resolve)")
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

// validateBaseURL rejects obviously malformed base URLs early.
func validateBaseURL(s string) error {
	if !strings.HasPrefix(s, "http://") && !strings.HasPrefix(s, "https://") {
		return fmt.Errorf("anthropic: BaseURL must start with http:// or https://, got %q", s)
	}
	return nil
}

// Stream issues a streaming completion request. The returned channel emits
// llm.Delta values per the protocol documented in llm/client.go.
func (c *Client) Stream(ctx context.Context, req llm.Request) (<-chan llm.Delta, error) {
	// Marshal the request body.
	body, err := marshalRequest(req)
	if err != nil {
		return nil, err
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("anthropic: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/messages", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	httpReq.Header.Set("x-api-key", c.apiKey)
	httpReq.Header.Set("anthropic-version", APIVersion)
	for k, v := range c.headers {
		httpReq.Header.Set(k, v)
	}

	// Issue the request with retries.
	//
	//nolint:bodyclose // resp.Body is closed by the streaming goroutine below
	// (see defer on resp.Body.Close); llm.Do returns nil resp on error per
	// its contract at retry.go:96, so the success path is the only one that
	// owns a Body. The bodyclose linter cannot trace goroutine-local defers.
	resp, _, err := llm.Do(ctx, c.retryPolicy, func(ctx context.Context) (*http.Response, error) {
		// Rewind the body for each retry.
		httpReq.Body = io.NopCloser(bytes.NewReader(payload))
		return c.http.Do(httpReq)
	})
	if err != nil {
		// Translate non-retryable HTTP errors to a Final delta via the
		// channel — callers don't want a hard error if the model returned
		// a 4xx. The streaming protocol says: if Stream returns non-nil
		// err, no channel. We follow that contract; callers check err.
		return nil, translateErr(err)
	}

	// Stream the SSE body on a goroutine.
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
			// Inject a Final with the error if no Final was emitted yet.
			select {
			case <-ctx.Done():
				return
			case ch <- llm.Final{StopReason: llm.StopReasonError, Err: err}:
			}
		}
	}()

	return ch, nil
}

// translateErr wraps non-retryable HTTP errors with helpful context. The
// underlying *llm.HTTPStatusError is preserved for errors.As so callers can
// inspect the status code.
func translateErr(err error) error {
	var se *llm.HTTPStatusError
	if errors.As(err, &se) {
		return fmt.Errorf("anthropic: HTTP %d: %w (auth may be invalid; check API key)", se.Status, se)
	}
	return fmt.Errorf("anthropic: %w", err)
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

// ResolveAuth implements the four-step auth resolution chain for the
// Anthropic provider. Inputs:
//
//   - explicit: the APIKey from the provider/model definition (may be a
//     sigil string starting with $ or !).
//   - auth: the AuthStore for auth.json lookup (nil skips step 2).
//
// Returns AuthResult with the first non-empty value, or an error if all
// four steps produced no credential.
func ResolveAuth(explicit string, auth config.AuthStore) (AuthResult, error) {
	// Step 1: explicit.
	if explicit != "" {
		// May be a sigil — resolve it first.
		if strings.HasPrefix(explicit, "$") || strings.HasPrefix(explicit, "!") {
			resolved, err := config.ResolveValue(explicit)
			if err != nil {
				return AuthResult{}, fmt.Errorf("anthropic: resolve sigil: %w", err)
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
		if key, ok := auth.Get(ProviderID); ok && key != "" {
			return AuthResult{APIKey: key, Source: AuthSourceAuthJSON}, nil
		}
	}
	// Step 3: env var.
	if v := os.Getenv(EnvVar); v != "" {
		return AuthResult{APIKey: v, Source: AuthSourceEnv}, nil
	}
	return AuthResult{}, fmt.Errorf("anthropic: no credential found in explicit value, auth.json, or %s", EnvVar)
}
