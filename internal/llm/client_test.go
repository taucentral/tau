package llm

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
)

func TestDelta_SealedInterface(t *testing.T) {
	// Compile-time: only types in this package implement isDelta().
	var _ Delta = TextDelta{}
	var _ Delta = ThinkingDelta{}
	var _ Delta = ToolCallDelta{}
	var _ Delta = UsageDelta{}
	var _ Delta = Final{}
}

func TestToolSchema_RoundTrip(t *testing.T) {
	schema := ToolSchema{
		Name:        "read",
		Description: "Read a file",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`),
	}
	data, err := json.Marshal(schema)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var got ToolSchema
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if got.Name != schema.Name || got.Description != schema.Description {
		t.Errorf("round-trip mismatch: %+v", got)
	}
	if string(got.Parameters) != string(schema.Parameters) {
		t.Errorf("Parameters = %s, want %s", got.Parameters, schema.Parameters)
	}
}

func TestTransport_Constants(t *testing.T) {
	cases := []struct {
		got, want Transport
	}{
		{TransportAuto, "auto"},
		{TransportSSE, "sse"},
		{TransportWebSocket, "websocket"},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("constant = %q, want %q", c.got, c.want)
		}
	}
}

func TestIsAbort(t *testing.T) {
	if !IsAbort(context.Canceled) {
		t.Errorf("IsAbort(context.Canceled) = false")
	}
	if !IsAbort(context.DeadlineExceeded) {
		t.Errorf("IsAbort(context.DeadlineExceeded) = false")
	}
	if IsAbort(errors.New("other")) {
		t.Errorf("IsAbort(other error) = true")
	}
	// Wrapped cancelation should also be detected.
	wrapped := errors.Join(context.Canceled, errors.New("wrapper"))
	if !IsAbort(wrapped) {
		t.Errorf("IsAbort(wrapped canceled) = false")
	}
}

func TestRequest_ZeroValues(t *testing.T) {
	// A zero Request should be a valid value; pointers must be nil so
	// providers can distinguish unset from zero.
	var req Request
	if req.ThinkingBudget != nil {
		t.Errorf("ThinkingBudget should be nil by default")
	}
	if req.MaxTokens != nil {
		t.Errorf("MaxTokens should be nil by default")
	}
	if req.Temperature != nil {
		t.Errorf("Temperature should be nil by default")
	}
	if req.Model != "" {
		t.Errorf("Model should be empty by default")
	}
}

// fauxClient is a minimal LLMClient for tests in higher-level packages.
// It replays a scripted sequence of Delta values.
type fauxClient struct {
	deltas []Delta
	err    error
}

func (f *fauxClient) Stream(ctx context.Context, req Request) (<-chan Delta, error) {
	if f.err != nil {
		return nil, f.err
	}
	ch := make(chan Delta, len(f.deltas))
	go func() {
		defer close(ch)
		for _, d := range f.deltas {
			select {
			case <-ctx.Done():
				return
			case ch <- d:
			}
		}
	}()
	return ch, nil
}

// Compile-time check: fauxClient satisfies LLMClient.
var _ LLMClient = (*fauxClient)(nil)

func TestFauxClient_SatisfiesLLMClient(t *testing.T) {
	// The compile-time check above is sufficient; this test exists so
	// go test surfaces regressions if the interface changes.
	fc := &fauxClient{
		deltas: []Delta{
			TextDelta{Text: "hi"},
			UsageDelta{InputTokens: 5, OutputTokens: 2},
			Final{StopReason: StopReasonEndTurn},
		},
	}
	ch, err := fc.Stream(context.Background(), Request{})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	var count int
	for range ch {
		count++
	}
	if count != 3 {
		t.Errorf("got %d deltas, want 3", count)
	}
}

func TestFauxClient_AbortsOnCtxCancel(t *testing.T) {
	fc := &fauxClient{
		deltas: []Delta{
			TextDelta{Text: "x"},
			TextDelta{Text: "y"},
		},
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel before reading
	ch, _ := fc.Stream(ctx, Request{})
	// Channel should still close promptly.
	for range ch {
		_ = ch // revive:empty-block — intentional drain loop
	}
}
