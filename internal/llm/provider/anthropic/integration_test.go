package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/coevin/tau/internal/llm"
)

// TestIntegration_ToolResultFlow verifies a multi-turn request containing a
// tool_use assistant message followed by a tool_result user message is
// marshaled to Anthropic's wire shape and the subsequent streamed response is
// assembled correctly. This covers the Phase 3.17 "tool-result" fixture.
func TestIntegration_ToolResultFlow(t *testing.T) {
	var capturedBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, strings.Join([]string{
			`event: message_start`,
			`data: {"type":"message_start","message":{"id":"msg_2","model":"claude-opus-4-5","usage":{"input_tokens":12,"output_tokens":0}}}`,
			``,
			`event: content_block_start`,
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			``,
			`event: content_block_delta`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"the file has 42 lines"}}`,
			``,
			`event: content_block_stop`,
			`data: {"type":"content_block_stop","index":0}`,
			``,
			`event: message_delta`,
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}`,
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
	maxTok := 200
	ch, err := c.Stream(context.Background(), llm.Request{
		Model:     "claude-opus-4-5",
		MaxTokens: &maxTok,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.TextContent{Text: "count lines in /etc/hosts"}}},
			{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
				llm.ToolUse{ID: "toolu_1", Name: "bash", Input: json.RawMessage(`{"cmd":"wc -l /etc/hosts"}`)},
			}},
			{Role: llm.RoleTool, ToolCallID: "toolu_1", ToolName: "bash", Content: []llm.ContentBlock{
				llm.ToolResult{
					ToolUseID: "toolu_1",
					Content:   []llm.ContentBlock{llm.TextContent{Text: "42 /etc/hosts"}},
				},
			}},
		},
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	msg, err := llm.CollectStream(context.Background(), ch, "claude-opus-4-5", ProviderID)
	if err != nil {
		t.Fatalf("CollectStream: %v", err)
	}

	// Verify the wire body: tool role becomes a user-role message whose
	// content is a tool_result block referencing toolu_1.
	var wire struct {
		Messages []struct {
			Role    string            `json:"role"`
			Content []json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(capturedBody, &wire); err != nil {
		t.Fatalf("unmarshal wire body: %v", err)
	}
	if len(wire.Messages) != 3 {
		t.Fatalf("wire messages len = %d", len(wire.Messages))
	}
	if wire.Messages[2].Role != "user" {
		t.Errorf("tool result wire role = %q, want user", wire.Messages[2].Role)
	}
	if len(wire.Messages[2].Content) != 1 {
		t.Fatalf("tool result content len = %d", len(wire.Messages[2].Content))
	}
	var probe struct {
		Type      string            `json:"type"`
		ToolUseID string            `json:"tool_use_id"`
		Content   []json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(wire.Messages[2].Content[0], &probe); err != nil {
		t.Fatalf("unmarshal tool_result block: %v", err)
	}
	if probe.Type != "tool_result" || probe.ToolUseID != "toolu_1" {
		t.Errorf("tool_result probe = %+v", probe)
	}
	// The tool_result content array contains a single text block whose
	// text carries the tool output.
	if len(probe.Content) != 1 {
		t.Fatalf("tool_result inner content len = %d", len(probe.Content))
	}
	var inner struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(probe.Content[0], &inner); err != nil {
		t.Fatalf("unmarshal inner text block: %v", err)
	}
	if inner.Type != "text" || inner.Text != "42 /etc/hosts" {
		t.Errorf("inner text block = %+v", inner)
	}

	// Verify the response was assembled.
	tc, ok := msg.Content[0].(llm.TextContent)
	if !ok || !strings.Contains(tc.Text, "42") {
		t.Errorf("Content[0] = %+v", msg.Content[0])
	}
	if msg.Usage == nil || msg.Usage.Input != 12 || msg.Usage.Output != 7 {
		t.Errorf("Usage = %+v", msg.Usage)
	}
}

// TestIntegration_DeltaOrderingAndFinalTermination verifies delta ordering
// for a text+tool_use mixed response: every TextDelta and ToolCallDelta
// precedes the UsageDelta, which precedes the Final, and exactly one Final
// is emitted. This is the Phase 3.17 "delta ordering and Final termination"
// gate.
func TestIntegration_DeltaOrderingAndFinalTermination(t *testing.T) {
	sse := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_x","model":"claude-opus-4-5","usage":{"input_tokens":3,"output_tokens":0}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"calling"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_x","name":"bash","input":{}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_input":"{\"cmd\":\"ls\"}"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":1}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
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
	maxTok := 200
	ch, err := c.Stream(context.Background(), llm.Request{Model: "claude-opus-4-5", MaxTokens: &maxTok})
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
		t.Fatalf("missing kinds: text=%d tool=%d usage=%d final=%d (deltas=%v)", textI, toolI, usageI, finalI, deltas)
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
	// Final must be the LAST delta.
	if finalI != len(deltas)-1 {
		t.Errorf("Final at index %d, want last (%d)", finalI, len(deltas)-1)
	}
	// Exactly one Final.
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
