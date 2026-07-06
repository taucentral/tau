package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/taucentral/tau/internal/config"
	"github.com/taucentral/tau/internal/llm"
)

func TestNew_RequiresAPIKey(t *testing.T) {
	_, err := New(Options{})
	if err == nil {
		t.Errorf("expected error for missing APIKey")
	}
}

func TestNew_DefaultBaseURL(t *testing.T) {
	c, err := New(Options{APIKey: "k"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if c.baseURL != DefaultBaseURL {
		t.Errorf("baseURL = %q, want %q", c.baseURL, DefaultBaseURL)
	}
}

func TestNew_RejectsBadBaseURL(t *testing.T) {
	_, err := New(Options{APIKey: "k", BaseURL: "ftp://bad"})
	if err == nil {
		t.Errorf("expected error for non-http base URL")
	}
}

func TestNew_HeadersPropagated(t *testing.T) {
	c, _ := New(Options{
		APIKey:  "k",
		Headers: map[string]string{"X-Custom": "v"},
	})
	if c.headers["X-Custom"] != "v" {
		t.Errorf("custom header not propagated")
	}
}

func TestStream_AgainstMockServer_TextOnly(t *testing.T) {
	// Stand up a server that streams a known-good Anthropic SSE response.
	sse := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"claude-opus-4-5","usage":{"input_tokens":5,"output_tokens":0}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify auth header.
		if r.Header.Get("x-api-key") != "test-key" {
			t.Errorf("x-api-key = %q", r.Header.Get("x-api-key"))
		}
		if r.Header.Get("anthropic-version") != APIVersion {
			t.Errorf("anthropic-version = %q", r.Header.Get("anthropic-version"))
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, sse)
	}))
	defer server.Close()

	c, err := New(Options{
		APIKey:      "test-key",
		BaseURL:     server.URL,
		HTTPClient:  server.Client(),
		RetryPolicy: llm.RetryPolicy{MaxRetries: 0},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	maxTok := 100
	ch, err := c.Stream(context.Background(), llm.Request{
		Model:     "claude-opus-4-5",
		MaxTokens: &maxTok,
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	msg, err := llm.CollectStream(context.Background(), ch, "claude-opus-4-5", ProviderID)
	if err != nil {
		t.Fatalf("CollectStream: %v", err)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("Content len = %d", len(msg.Content))
	}
	tc, ok := msg.Content[0].(llm.TextContent)
	if !ok {
		t.Fatalf("Content[0] type = %T", msg.Content[0])
	}
	if tc.Text != "hi" {
		t.Errorf("Text = %q", tc.Text)
	}
	if msg.Usage == nil || msg.Usage.Output != 1 {
		t.Errorf("Usage = %+v", msg.Usage)
	}
}

func TestStream_ServerError4xxReturnsErr(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		fmt.Fprint(w, `{"error":{"type":"authentication_error","message":"invalid api key"}}`)
	}))
	defer server.Close()

	c, _ := New(Options{
		APIKey:      "k",
		BaseURL:     server.URL,
		HTTPClient:  server.Client(),
		RetryPolicy: llm.RetryPolicy{MaxRetries: 0},
	})
	_, err := c.Stream(context.Background(), llm.Request{Model: "m"})
	if err == nil {
		t.Fatal("expected error for 401")
	}
	var se *llm.HTTPStatusError
	if !errors.As(err, &se) {
		// translateErr wraps with fmt.Errorf, but errors.As should still
		// unwrap to HTTPStatusError since translateErr uses %w formatting.
		t.Errorf("err type = %T, want *llm.HTTPStatusError wrapper", err)
	}
}

func TestStream_ServerError5xxRetries(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls < 2 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		// Quick terminal event to drain.
		fmt.Fprint(w, strings.Join([]string{
			`event: message_start`,
			`data: {"type":"message_start","message":{"id":"m","model":"m","usage":{"input_tokens":1,"output_tokens":0}}}`,
			``,
			`event: message_delta`,
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}`,
			``,
			`event: message_stop`,
			`data: {"type":"message_stop"}`,
			``,
		}, "\n"))
	}))
	defer server.Close()

	c, _ := New(Options{
		APIKey:  "k",
		BaseURL: server.URL,
		HTTPClient: &http.Client{
			Timeout: 5 * time.Second,
		},
		RetryPolicy: llm.RetryPolicy{MaxRetries: 2, BaseDelay: time.Millisecond, MaxDelay: 5 * time.Millisecond},
	})
	ch, err := c.Stream(context.Background(), llm.Request{Model: "m"})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	for range ch {
		_ = ch // revive:empty-block — drain
	}
	if calls < 2 {
		t.Errorf("calls = %d, want >= 2", calls)
	}
}

func TestStream_RequestBodyShape(t *testing.T) {
	var capturedBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, strings.Join([]string{
			`event: message_start`,
			`data: {"type":"message_start","message":{"id":"m","model":"m","usage":{"input_tokens":1,"output_tokens":0}}}`,
			``,
			`event: message_delta`,
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}`,
			``,
			`event: message_stop`,
			`data: {"type":"message_stop"}`,
			``,
		}, "\n"))
	}))
	defer server.Close()

	c, _ := New(Options{
		APIKey:      "k",
		BaseURL:     server.URL,
		HTTPClient:  server.Client(),
		RetryPolicy: llm.RetryPolicy{MaxRetries: 0},
	})
	maxTok := 4096
	ch, _ := c.Stream(context.Background(), llm.Request{
		Model:     "claude-opus-4-5",
		MaxTokens: &maxTok,
		System:    []llm.ContentBlock{llm.TextContent{Text: "Be nice."}},
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.TextContent{Text: "hi"}}}},
	})
	for range ch {
		_ = ch // revive:empty-block — drain
	}

	var parsed map[string]any
	if err := json.Unmarshal(capturedBody, &parsed); err != nil {
		t.Fatalf("body not valid JSON: %v", err)
	}
	if parsed["model"] != "claude-opus-4-5" {
		t.Errorf("model = %v", parsed["model"])
	}
	if parsed["stream"] != true {
		t.Errorf("stream = %v", parsed["stream"])
	}
	if parsed["max_tokens"] != float64(4096) {
		t.Errorf("max_tokens = %v", parsed["max_tokens"])
	}
}

func TestResolveAuth_ExplicitLiteral(t *testing.T) {
	res, err := ResolveAuth("sk-literal", nil)
	if err != nil {
		t.Fatalf("ResolveAuth: %v", err)
	}
	if res.APIKey != "sk-literal" || res.Source != AuthSourceExplicit {
		t.Errorf("res = %+v", res)
	}
}

func TestResolveAuth_AuthJSONWins(t *testing.T) {
	auth := &fakeAuthStore{entries: map[string]string{"anthropic": "sk-from-authjson"}}
	res, err := ResolveAuth("", auth)
	if err != nil {
		t.Fatalf("ResolveAuth: %v", err)
	}
	if res.APIKey != "sk-from-authjson" || res.Source != AuthSourceAuthJSON {
		t.Errorf("res = %+v", res)
	}
}

func TestResolveAuth_EnvFallback(t *testing.T) {
	t.Setenv(EnvVar, "sk-from-env")
	res, err := ResolveAuth("", nil)
	if err != nil {
		t.Fatalf("ResolveAuth: %v", err)
	}
	if res.APIKey != "sk-from-env" || res.Source != AuthSourceEnv {
		t.Errorf("res = %+v", res)
	}
}

func TestResolveAuth_SigilValue(t *testing.T) {
	t.Setenv("TAU_TEST_KEY", "sk-from-sigil-env")
	res, err := ResolveAuth("$TAU_TEST_KEY", nil)
	if err != nil {
		t.Fatalf("ResolveAuth: %v", err)
	}
	if res.APIKey != "sk-from-sigil-env" || res.Source != AuthSourceSigil {
		t.Errorf("res = %+v", res)
	}
}

func TestResolveAuth_NoCredential(t *testing.T) {
	t.Setenv(EnvVar, "")
	_, err := ResolveAuth("", nil)
	if err == nil {
		t.Errorf("expected error when no credential found")
	}
}

func TestResolveAuth_ExplicitWinsOverAuthJSON(t *testing.T) {
	auth := &fakeAuthStore{entries: map[string]string{"anthropic": "sk-from-authjson"}}
	res, _ := ResolveAuth("sk-explicit", auth)
	if res.APIKey != "sk-explicit" || res.Source != AuthSourceExplicit {
		t.Errorf("res = %+v, want explicit", res)
	}
}

// fakeAuthStore is a minimal AuthStore for tests.
type fakeAuthStore struct {
	entries map[string]string
}

func (f *fakeAuthStore) Get(provider string) (string, bool) {
	v, ok := f.entries[provider]
	return v, ok
}
func (f *fakeAuthStore) Set(provider, apiKey string) error {
	f.entries[provider] = apiKey
	return nil
}
func (f *fakeAuthStore) Delete(provider string) error {
	delete(f.entries, provider)
	return nil
}
func (f *fakeAuthStore) Load() (map[string]config.ProviderAuth, error) {
	out := make(map[string]config.ProviderAuth, len(f.entries))
	for k, v := range f.entries {
		out[k] = config.ProviderAuth{APIKey: v}
	}
	return out, nil
}
func (f *fakeAuthStore) Save(entries map[string]config.ProviderAuth) error {
	f.entries = make(map[string]string, len(entries))
	for k, v := range entries {
		f.entries[k] = v.APIKey
	}
	return nil
}

// Compile-time check.
var _ config.AuthStore = (*fakeAuthStore)(nil)
