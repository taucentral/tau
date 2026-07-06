package fauxprovider

import (
	"context"
	"strings"
	"testing"

	"github.com/taucentral/tau/internal/llm"
)

func TestClient_DefaultResponse(t *testing.T) {
	c := New()
	if got := c.Response(); got != DefaultResponse {
		t.Errorf("Response() = %q, want %q", got, DefaultResponse)
	}
}

func TestClient_TAU_FAUX_SCRIPT_EnvOverrides(t *testing.T) {
	t.Setenv("TAU_FAUX_SCRIPT", "custom response")
	c := New()
	if got := c.Response(); got != "custom response" {
		t.Errorf("Response() = %q, want %q", got, "custom response")
	}
}

func TestClient_TAU_FAUX_SCRIPT_EmptyFallsBackToDefault(t *testing.T) {
	t.Setenv("TAU_FAUX_SCRIPT", "   ")
	c := New()
	if got := c.Response(); got != DefaultResponse {
		t.Errorf("Response() = %q, want default", got)
	}
}

func TestClient_Stream_EmitsTextThenFinal(t *testing.T) {
	c := NewWithResponse("hi there")
	ch, err := c.Stream(context.Background(), llm.Request{Model: "faux"})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	var deltas []llm.Delta
	for d := range ch {
		deltas = append(deltas, d)
	}
	if len(deltas) != 2 {
		t.Fatalf("expected 2 deltas, got %d: %+v", len(deltas), deltas)
	}
	td, ok := deltas[0].(llm.TextDelta)
	if !ok {
		t.Fatalf("delta[0] type = %T, want TextDelta", deltas[0])
	}
	if td.Text != "hi there" {
		t.Errorf("delta[0].Text = %q, want %q", td.Text, "hi there")
	}
	f, ok := deltas[1].(llm.Final)
	if !ok {
		t.Fatalf("delta[1] type = %T, want Final", deltas[1])
	}
	if f.StopReason != llm.StopReasonEndTurn {
		t.Errorf("delta[1].StopReason = %q, want %q", f.StopReason, llm.StopReasonEndTurn)
	}
}

func TestClient_Stream_RecordsRequests(t *testing.T) {
	c := NewWithResponse("ok")
	_, _ = c.Stream(context.Background(), llm.Request{Model: "faux-1"})
	_, _ = c.Stream(context.Background(), llm.Request{Model: "faux-2"})
	got := c.RecordedRequests()
	if len(got) != 2 {
		t.Fatalf("expected 2 recorded requests, got %d", len(got))
	}
	if got[0].Model != "faux-1" || got[1].Model != "faux-2" {
		t.Errorf("recorded models = %q, %q", got[0].Model, got[1].Model)
	}
}

func TestClient_Stream_HonoursCancelledContext(t *testing.T) {
	c := NewWithResponse("never emitted")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := c.Stream(ctx, llm.Request{})
	if err == nil {
		t.Fatalf("expected error on cancelled ctx")
	}
	if !strings.Contains(err.Error(), "context canceled") {
		t.Errorf("expected context.Canceled in error, got: %v", err)
	}
}

func TestClient_Stream_ResponseIsExactString(t *testing.T) {
	// Sanity check: the test asserts the response is exactly the string
	// configured, with no trailing newline added by the provider. The
	// print-mode handler adds its own trailing newline.
	c := NewWithResponse("ok")
	ch, _ := c.Stream(context.Background(), llm.Request{})
	for d := range ch {
		if td, ok := d.(llm.TextDelta); ok {
			if td.Text != "ok" {
				t.Errorf("TextDelta.Text = %q, want %q", td.Text, "ok")
			}
		}
	}
}
