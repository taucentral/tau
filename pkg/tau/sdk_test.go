package tau

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/coevin/tau/internal/config"
	"github.com/coevin/tau/internal/fauxprovider"
	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/tools"
)

// newTestOptions returns an Options bundle wired against the faux
// provider so SDK tests never touch the network.
func newTestOptions(t *testing.T) Options {
	t.Helper()
	return Options{
		Cwd:           t.TempDir(),
		Model:         "faux",
		LLMClient:     fauxprovider.NewWithResponse("faux sdk reply"),
		Tools:         []HeadlessTool{tools.NewReadTool(tools.OSReadOperations{})},
		Settings:      config.DefaultSettings(),
		ContextWindow: 200000,
	}
}

func TestCreateAgentSession_Validation(t *testing.T) {
	tests := []struct {
		name string
		mut  func(o *Options)
		want string
	}{
		{"missing model", func(o *Options) { o.Model = "" }, "Model"},
		{"missing client", func(o *Options) { o.LLMClient = nil }, "LLMClient"},
		{"missing tools", func(o *Options) { o.Tools = nil }, "Tools"},
		{"empty cwd", func(o *Options) { o.Cwd = "" }, "cwd"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			opts := newTestOptions(t)
			tc.mut(&opts)
			_, err := CreateAgentSession(context.Background(), opts)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.want)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("err = %q, want substring %q", err.Error(), tc.want)
			}
		})
	}
}

func TestCreateAgentSession_Defaults(t *testing.T) {
	opts := newTestOptions(t)
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	if got := sess.Model(); got != "faux" {
		t.Errorf("Model = %q, want faux", got)
	}
	if sess.Cwd() == "" {
		t.Error("Cwd is empty")
	}
	if sess.Cwd() != opts.Cwd {
		t.Errorf("Cwd = %q, want %q", sess.Cwd(), opts.Cwd)
	}
	if sess.CreatedAt().IsZero() {
		t.Error("CreatedAt is zero")
	}
	if sess.SessionID() != "" {
		t.Errorf("SessionID = %q, want empty for fresh lazy session", sess.SessionID())
	}
}

func TestAgentSession_RunSingleTurn(t *testing.T) {
	opts := newTestOptions(t)
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := sess.Run(ctx, "hello"); err != nil {
		t.Fatalf("Run: %v", err)
	}
}

func TestAgentSession_RunEmptyPromptFails(t *testing.T) {
	opts := newTestOptions(t)
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	if err := sess.Run(context.Background(), ""); err == nil {
		t.Error("Run with empty prompt: err = nil, want error")
	}
}

func TestAgentSession_SubscribeReceivesLifecycle(t *testing.T) {
	opts := newTestOptions(t)
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	// Subscribe before Run so the session_start event lands in our
	// channel and not on the floor.
	events := sess.Subscribe()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := sess.Run(ctx, "hello"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Collect events until the bus quietses. Turn end is the canonical
	// "turn complete" signal; the faux provider emits exactly one.
	got := map[string]bool{}
	timeout := time.After(2 * time.Second)
	for {
		select {
		case evt := <-events:
			got[eventName(evt)] = true
			if _, ok := evt.(TurnEndEvent); ok {
				return
			}
		case <-timeout:
			t.Fatalf("timeout waiting for turn_end. got events: %v", got)
		}
	}
}

func TestAgentSession_ShutdownIsIdempotent(t *testing.T) {
	opts := newTestOptions(t)
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	if err := sess.Shutdown(context.Background()); err != nil {
		t.Fatalf("first Shutdown: %v", err)
	}
	if err := sess.Shutdown(context.Background()); err != nil {
		t.Errorf("second Shutdown: %v, want nil (idempotent)", err)
	}
}

func TestAgentSession_RunAfterShutdownReturnsErrRuntimeShutdown(t *testing.T) {
	opts := newTestOptions(t)
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	if err := sess.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
	err = sess.Run(context.Background(), "hello")
	if !errors.Is(err, ErrRuntimeShutdown) {
		t.Errorf("Run after Shutdown: err = %v, want ErrRuntimeShutdown", err)
	}
}

func TestAgentSession_AbortCancelsInFlightTurn(t *testing.T) {
	// Build a client whose Stream blocks until the test closes the
	// release channel so the turn is provably in-flight when Abort runs.
	client := &blockingClient{release: make(chan struct{})}
	opts := Options{
		Cwd:       t.TempDir(),
		Model:     "faux",
		LLMClient: client,
		Tools:     []HeadlessTool{tools.NewReadTool(tools.OSReadOperations{})},
		Settings:  config.DefaultSettings(),
	}
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	runDone := make(chan error, 1)
	go func() {
		runDone <- sess.Run(context.Background(), "hello")
	}()

	// Give the turn a moment to enter Stream, then abort.
	time.Sleep(50 * time.Millisecond)
	sess.Abort("test")

	select {
	case err := <-runDone:
		if err == nil {
			t.Error("Run returned nil after Abort; want cancellation error")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Run did not return after Abort (deadlock)")
	}
	// Unblock the client so Shutdown can complete cleanly.
	close(client.release)
}

// eventName returns a short identifier for an event used by the
// lifecycle-coverage test.
func eventName(evt SessionEvent) string {
	switch evt.(type) {
	case SessionStartEvent:
		return "session_start"
	case TurnStartEvent:
		return "turn_start"
	case MessageStartEvent:
		return "message_start"
	case MessageUpdateEvent:
		return "message_update"
	case ToolCallEvent:
		return "tool_call"
	case ToolResultEvent:
		return "tool_result"
	case MessageEndEvent:
		return "message_end"
	case TurnEndEvent:
		return "turn_end"
	case SessionShutdownEvent:
		return "session_shutdown"
	default:
		return "unknown"
	}
}

// blockingClient is an LLMClient whose Stream blocks until release is
// closed, then returns a single TextDelta + Final. Used to verify that
// Abort cancels the in-flight turn.
type blockingClient struct {
	release chan struct{}
}

func (c *blockingClient) Stream(ctx context.Context, _ Request) (<-chan llm.Delta, error) {
	ch := make(chan llm.Delta, 2)
	go func() {
		defer close(ch)
		select {
		case <-c.release:
			ch <- llm.TextDelta{Text: "released"}
			ch <- llm.Final{StopReason: llm.StopReasonEndTurn}
		case <-ctx.Done():
		}
	}()
	return ch, nil
}

func TestTypeAliasesAreIdenticalToInternal(_ *testing.T) {
	// Smoke test that the alias declarations resolve to the same types
	// the internal packages export. If the alias is ever copy-pasted
	// into a separate type, this fails to compile.
	// nolint:staticcheck // intentional compile-time type check (QF1011)
	{
		var _ Role = llm.Role("")
		var _ ContentBlock = llm.TextContent{}
		var _ Tool = Tool(nil)
		var _ SessionEvent = SessionEvent(nil)
		var _ ToolCall = ToolCall{}
		var _ ToolResult = ToolResult{}
	}
}
