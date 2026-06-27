package openai

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coevin/tau/internal/config"
	"github.com/coevin/tau/internal/llm"
)

func TestNew_RequiresAPIKey(t *testing.T) {
	_, err := New(Options{})
	if err == nil {
		t.Errorf("expected error for missing APIKey")
	}
}

func TestNew_DefaultBaseURL(t *testing.T) {
	c, _ := New(Options{APIKey: "k"})
	if c.baseURL != DefaultBaseURL {
		t.Errorf("baseURL = %q, want %q", c.baseURL, DefaultBaseURL)
	}
}

func TestNew_RejectsBadBaseURL(t *testing.T) {
	_, err := New(Options{APIKey: "k", BaseURL: "ftp://x"})
	if err == nil {
		t.Errorf("expected error for non-http base URL")
	}
}

func TestStream_AgainstMockServer_TextOnly(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		``,
		`data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q", got)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, sse)
	}))
	defer server.Close()

	c, _ := New(Options{
		APIKey:      "test-key",
		BaseURL:     server.URL,
		HTTPClient:  server.Client(),
		RetryPolicy: llm.RetryPolicy{MaxRetries: 0},
	})
	ch, err := c.Stream(context.Background(), llm.Request{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	msg, err := llm.CollectStream(context.Background(), ch, "gpt-4o", ProviderID)
	if err != nil {
		t.Fatalf("CollectStream: %v", err)
	}
	tc, ok := msg.Content[0].(llm.TextContent)
	if !ok || tc.Text != "hi" {
		t.Errorf("Content[0] = %+v", msg.Content[0])
	}
	if msg.Usage == nil || msg.Usage.Output != 1 {
		t.Errorf("Usage = %+v", msg.Usage)
	}
}

func TestStream_AgainstMockServer_ToolCall(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"id":"c","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read","arguments":""}}]},"finish_reason":null}]}`,
		``,
		`data: {"id":"c","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":\"/x\"}"}}]},"finish_reason":null}]}`,
		``,
		`data: {"id":"c","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, sse)
	}))
	defer server.Close()

	c, _ := New(Options{
		APIKey:      "k",
		BaseURL:     server.URL,
		HTTPClient:  server.Client(),
		RetryPolicy: llm.RetryPolicy{MaxRetries: 0},
	})
	ch, _ := c.Stream(context.Background(), llm.Request{Model: "gpt-4o"})
	msg, err := llm.CollectStream(context.Background(), ch, "gpt-4o", ProviderID)
	if err != nil {
		t.Fatalf("CollectStream: %v", err)
	}
	tu, ok := msg.Content[0].(llm.ToolUse)
	if !ok {
		t.Fatalf("Content[0] = %T", msg.Content[0])
	}
	if tu.ID != "call_1" || tu.Name != "read" {
		t.Errorf("ToolUse = %+v", tu)
	}
}

func TestStream_401ReturnsErr(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		fmt.Fprint(w, `{"error":{"message":"invalid api key"}}`)
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
		t.Fatal("expected error")
	}
	var se *llm.HTTPStatusError
	if !errors.As(err, &se) {
		t.Errorf("err type = %T, want *llm.HTTPStatusError wrapper", err)
	}
}

func TestStream_503Retries(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls < 2 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, strings.Join([]string{
			`data: {"id":"c","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			``,
			`data: [DONE]`,
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

func TestResolveAuth_Explicit(t *testing.T) {
	res, err := ResolveAuth("openai", EnvVar, "sk-explicit", nil)
	if err != nil {
		t.Fatalf("ResolveAuth: %v", err)
	}
	if res.APIKey != "sk-explicit" || res.Source != AuthSourceExplicit {
		t.Errorf("res = %+v", res)
	}
}

func TestResolveAuth_AuthJSON(t *testing.T) {
	auth := &fakeAuthStore{entries: map[string]string{"openai": "sk-authjson"}}
	res, err := ResolveAuth("openai", EnvVar, "", auth)
	if err != nil {
		t.Fatalf("ResolveAuth: %v", err)
	}
	if res.APIKey != "sk-authjson" || res.Source != AuthSourceAuthJSON {
		t.Errorf("res = %+v", res)
	}
}

func TestResolveAuth_Env(t *testing.T) {
	t.Setenv(EnvVar, "sk-env")
	res, err := ResolveAuth("openai", EnvVar, "", nil)
	if err != nil {
		t.Fatalf("ResolveAuth: %v", err)
	}
	if res.APIKey != "sk-env" || res.Source != AuthSourceEnv {
		t.Errorf("res = %+v", res)
	}
}

func TestResolveAuth_CustomEnvVar(t *testing.T) {
	// OpenAI-compatible providers use custom env vars.
	t.Setenv("DEEPSEEK_API_KEY", "sk-deepseek")
	res, err := ResolveAuth("deepseek", "DEEPSEEK_API_KEY", "", nil)
	if err != nil {
		t.Fatalf("ResolveAuth: %v", err)
	}
	if res.APIKey != "sk-deepseek" {
		t.Errorf("APIKey = %q", res.APIKey)
	}
}

func TestResolveAuth_NoCredential(t *testing.T) {
	t.Setenv(EnvVar, "")
	_, err := ResolveAuth("openai", EnvVar, "", nil)
	if err == nil {
		t.Errorf("expected error")
	}
}

func TestResolveAuth_SigilValue(t *testing.T) {
	t.Setenv("TAU_OAI_KEY", "sk-sigil")
	res, err := ResolveAuth("openai", EnvVar, "$TAU_OAI_KEY", nil)
	if err != nil {
		t.Fatalf("ResolveAuth: %v", err)
	}
	if res.APIKey != "sk-sigil" || res.Source != AuthSourceSigil {
		t.Errorf("res = %+v", res)
	}
}

type fakeAuthStore struct {
	entries map[string]string
}

func (f *fakeAuthStore) Get(provider string) (string, bool) {
	v, ok := f.entries[provider]
	return v, ok
}
func (f *fakeAuthStore) Set(provider, key string) error {
	f.entries[provider] = key
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

var _ config.AuthStore = (*fakeAuthStore)(nil)
