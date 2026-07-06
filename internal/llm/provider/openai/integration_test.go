package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/taucentral/tau/internal/llm"
)

// TestIntegration_ToolResultFlow verifies a multi-turn OpenAI request that
// contains an assistant tool_call message followed by a tool-result message
// is marshaled to the OpenAI Chat Completions wire shape (role=tool with
// tool_call_id) and the subsequent streamed response is assembled correctly.
// Covers the Phase 3.17 "tool-result" fixture for the OpenAI provider.
func TestIntegration_ToolResultFlow(t *testing.T) {
	var capturedBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, strings.Join([]string{
			`data: {"id":"chatcmpl_x","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
			``,
			`data: {"id":"chatcmpl_x","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"the file has 42 lines"},"finish_reason":null}]}`,
			``,
			`data: {"id":"chatcmpl_x","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			``,
			`data: {"id":"chatcmpl_x","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":20,"completion_tokens":6,"total_tokens":26}}`,
			``,
			`data: [DONE]`,
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
	ch, err := c.Stream(context.Background(), llm.Request{
		Model: "gpt-4o",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.TextContent{Text: "count lines in /etc/hosts"}}},
			{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
				llm.ToolUse{ID: "call_1", Name: "bash", Input: json.RawMessage(`{"cmd":"wc -l /etc/hosts"}`)},
			}},
			{Role: llm.RoleTool, ToolCallID: "call_1", ToolName: "bash", Content: []llm.ContentBlock{
				llm.TextContent{Text: "42 /etc/hosts"},
			}},
		},
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	msg, err := llm.CollectStream(context.Background(), ch, "gpt-4o", ProviderID)
	if err != nil {
		t.Fatalf("CollectStream: %v", err)
	}

	// Verify wire body shape: the tool result becomes a top-level message
	// with role=tool and tool_call_id=call_1.
	var wire struct {
		Messages []struct {
			Role       string          `json:"role"`
			Content    json.RawMessage `json:"content"`
			ToolCallID string          `json:"tool_call_id,omitempty"`
			Name       string          `json:"name,omitempty"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(capturedBody, &wire); err != nil {
		t.Fatalf("unmarshal wire body: %v", err)
	}
	if len(wire.Messages) != 3 {
		t.Fatalf("wire messages len = %d", len(wire.Messages))
	}
	tr := wire.Messages[2]
	if tr.Role != "tool" {
		t.Errorf("tool result role = %q, want tool", tr.Role)
	}
	if tr.ToolCallID != "call_1" {
		t.Errorf("tool_call_id = %q, want call_1", tr.ToolCallID)
	}
	// Content for role=tool is a plain JSON string.
	var s string
	if err := json.Unmarshal(tr.Content, &s); err != nil {
		t.Errorf("tool content unmarshal: %v (raw=%s)", err, tr.Content)
	}
	if s != "42 /etc/hosts" {
		t.Errorf("tool content = %q, want 42 /etc/hosts", s)
	}

	// Verify response was assembled.
	tc, ok := msg.Content[0].(llm.TextContent)
	if !ok || !strings.Contains(tc.Text, "42") {
		t.Errorf("Content[0] = %+v", msg.Content[0])
	}
	if msg.Usage == nil || msg.Usage.Input != 20 || msg.Usage.Output != 6 {
		t.Errorf("Usage = %+v", msg.Usage)
	}
}

// TestIntegration_DeltaOrderingAndFinalTermination verifies delta ordering
// for a text+tool_call mixed stream: every TextDelta and ToolCallDelta
// precedes the UsageDelta, which precedes the Final, and exactly one Final
// is emitted and is the last delta. Covers the Phase 3.17 "delta ordering
// and Final termination" gate for the OpenAI provider.
func TestIntegration_DeltaOrderingAndFinalTermination(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"id":"c","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"calling"},"finish_reason":null}]}`,
		``,
		`data: {"id":"c","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"bash","arguments":""}}]},"finish_reason":null}]}`,
		``,
		`data: {"id":"c","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"cmd\":\"ls\"}"}}]},"finish_reason":null}]}`,
		``,
		`data: {"id":"c","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		``,
		`data: {"id":"c","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":4,"completion_tokens":6,"total_tokens":10}}`,
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
	ch, err := c.Stream(context.Background(), llm.Request{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	var deltas []llm.Delta
	for d := range ch {
		deltas = append(deltas, d)
	}
	if len(deltas) == 0 {
		t.Fatalf("no deltas received")
	}

	// Locate index of each kind and assert ordering: text < tool < usage < final.
	idxOf := func(target string) int {
		for i, d := range deltas {
			switch d.(type) {
			case llm.TextDelta:
				if target == "text" {
					return i
				}
			case llm.ToolCallDelta:
				if target == "tool" {
					return i
				}
			case llm.UsageDelta:
				if target == "usage" {
					return i
				}
			case llm.Final:
				if target == "final" {
					return i
				}
			}
		}
		return -1
	}
	textI, toolI, usageI, finalI := idxOf("text"), idxOf("tool"), idxOf("usage"), idxOf("final")
	if textI < 0 || toolI < 0 || usageI < 0 || finalI < 0 {
		t.Fatalf("missing kinds: text=%d tool=%d usage=%d final=%d", textI, toolI, usageI, finalI)
	}
	if textI >= toolI {
		t.Errorf("text (%d) not before tool (%d)", textI, toolI)
	}
	if toolI >= usageI {
		t.Errorf("tool (%d) not before usage (%d)", toolI, usageI)
	}
	if usageI >= finalI {
		t.Errorf("usage (%d) not before final (%d)", usageI, finalI)
	}
	if finalI != len(deltas)-1 {
		t.Errorf("Final at index %d, want last (%d)", finalI, len(deltas)-1)
	}
	finalCount := 0
	for _, d := range deltas {
		if _, ok := d.(llm.Final); ok {
			finalCount++
		}
	}
	if finalCount != 1 {
		t.Errorf("Final count = %d, want 1", finalCount)
	}
}
