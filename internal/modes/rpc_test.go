package modes

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coevin/tau/internal/agent"
	"github.com/coevin/tau/internal/config"
	"github.com/coevin/tau/internal/fauxprovider"
	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/tools"
)

// TestRunRPC_NilSessionReturnsError verifies the guard.
func TestRunRPC_NilSessionReturnsError(t *testing.T) {
	err := RunRPC(context.Background(), RPCOptions{
		Stdin:  strings.NewReader(""),
		Stdout: io.Discard,
	}, nil)
	if err == nil || !strings.Contains(err.Error(), "session is nil") {
		t.Errorf("RunRPC(nil): err = %v, want 'session is nil'", err)
	}
}

// TestRunRPC_SessionStartResponds verifies a session/start request
// produces a JSON-RPC response with sessionId, model, and cwd.
func TestRunRPC_SessionStartResponds(t *testing.T) {
	h := newRPCTestHarness(t)
	defer h.close()

	h.writeLine(`{"jsonrpc":"2.0","id":1,"method":"session/start"}`)
	h.writeLine(`{"jsonrpc":"2.0","id":2,"method":"session/shutdown"}`)
	h.closeInput()

	if err := h.run(time.Second); err != nil {
		t.Fatalf("RunRPC: %v", err)
	}
	out := h.frames()
	if len(out) != 2 {
		t.Fatalf("expected 2 frames, got %d: %v", len(out), out)
	}
	var startResp map[string]any
	if err := json.Unmarshal(out[0], &startResp); err != nil {
		t.Fatalf("start response not JSON: %v", err)
	}
	if startResp["jsonrpc"] != "2.0" {
		t.Errorf("jsonrpc = %v, want 2.0", startResp["jsonrpc"])
	}
	result, ok := startResp["result"].(map[string]any)
	if !ok {
		t.Fatalf("start response missing result: %v", startResp)
	}
	for _, k := range []string{"sessionId", "model", "cwd"} {
		if _, ok := result[k]; !ok {
			t.Errorf("start result missing %q: %v", k, result)
		}
	}
}

// TestRunRPC_ShutdownReturnsOK verifies session/shutdown returns a
// success response.
func TestRunRPC_ShutdownReturnsOK(t *testing.T) {
	h := newRPCTestHarness(t)
	defer h.close()

	h.writeLine(`{"jsonrpc":"2.0","id":1,"method":"session/shutdown"}`)
	h.closeInput()
	if err := h.run(time.Second); err != nil {
		t.Fatalf("RunRPC: %v", err)
	}
	out := h.frames()
	if len(out) != 1 {
		t.Fatalf("expected 1 frame, got %d: %v", len(out), out)
	}
	var resp map[string]any
	if err := json.Unmarshal(out[0], &resp); err != nil {
		t.Fatalf("response not JSON: %v", err)
	}
	if resp["result"] == nil {
		t.Errorf("result is nil: %v", resp)
	}
}

// TestRunRPC_UnknownMethodReturnsError verifies -32601 for unknown
// methods.
func TestRunRPC_UnknownMethodReturnsError(t *testing.T) {
	h := newRPCTestHarness(t)
	defer h.close()

	h.writeLine(`{"jsonrpc":"2.0","id":1,"method":"session/bogus"}`)
	h.writeLine(`{"jsonrpc":"2.0","id":2,"method":"session/shutdown"}`)
	h.closeInput()
	if err := h.run(time.Second); err != nil {
		t.Fatalf("RunRPC: %v", err)
	}
	out := h.frames()
	var firstResp map[string]any
	if err := json.Unmarshal(out[0], &firstResp); err != nil {
		t.Fatalf("first response not JSON: %v", err)
	}
	errObj, ok := firstResp["error"].(map[string]any)
	if !ok {
		t.Fatalf("expected error object, got: %v", firstResp)
	}
	if code, _ := errObj["code"].(float64); code != -32601 {
		t.Errorf("error.code = %v, want -32601", errObj["code"])
	}
}

// TestRunRPC_MalformedJSONReturnsParseError verifies -32700.
func TestRunRPC_MalformedJSONReturnsParseError(t *testing.T) {
	h := newRPCTestHarness(t)
	defer h.close()

	h.writeLine(`not-json`)
	h.writeLine(`{"jsonrpc":"2.0","id":1,"method":"session/shutdown"}`)
	h.closeInput()
	if err := h.run(time.Second); err != nil {
		t.Fatalf("RunRPC: %v", err)
	}
	out := h.frames()
	var firstResp map[string]any
	if err := json.Unmarshal(out[0], &firstResp); err != nil {
		t.Fatalf("first response not JSON: %v", err)
	}
	errObj, ok := firstResp["error"].(map[string]any)
	if !ok {
		t.Fatalf("expected error object, got: %v", firstResp)
	}
	if code, _ := errObj["code"].(float64); code != -32700 {
		t.Errorf("error.code = %v, want -32700", errObj["code"])
	}
}

// TestRunRPC_SendMessageStreamsNotifications verifies session/sendMessage
// runs a turn, emits messageDelta + turnEnd notifications, and returns
// a success response.
func TestRunRPC_SendMessageStreamsNotifications(t *testing.T) {
	h := newRPCTestHarness(t)
	defer h.close()

	h.writeLine(`{"jsonrpc":"2.0","id":1,"method":"session/sendMessage","params":{"prompt":"hi"}}`)
	h.writeLine(`{"jsonrpc":"2.0","id":2,"method":"session/shutdown"}`)
	h.closeInput()
	if err := h.run(2 * time.Second); err != nil {
		t.Fatalf("RunRPC: %v", err)
	}
	out := h.frames()
	if len(out) < 2 {
		t.Fatalf("expected at least 2 frames, got %d: %v", len(out), out)
	}

	// Verify at least one messageDelta and one turnEnd notification.
	var sawDelta, sawTurnEnd bool
	for _, raw := range out {
		var m map[string]any
		if err := json.Unmarshal(raw, &m); err != nil {
			continue
		}
		method, _ := m["method"].(string)
		switch method {
		case "notifications/messageDelta":
			sawDelta = true
		case "notifications/turnEnd":
			sawTurnEnd = true
		}
	}
	if !sawDelta {
		t.Errorf("missing notifications/messageDelta; got %d frames", len(out))
	}
	if !sawTurnEnd {
		t.Errorf("missing notifications/turnEnd; got %d frames", len(out))
	}
}

// TestRunRPC_AbortWithoutActiveTurnIsNoop verifies session/abort
// succeeds even when no turn is running.
func TestRunRPC_AbortWithoutActiveTurnIsNoop(t *testing.T) {
	h := newRPCTestHarness(t)
	defer h.close()

	h.writeLine(`{"jsonrpc":"2.0","id":1,"method":"session/abort"}`)
	h.writeLine(`{"jsonrpc":"2.0","id":2,"method":"session/shutdown"}`)
	h.closeInput()
	if err := h.run(time.Second); err != nil {
		t.Fatalf("RunRPC: %v", err)
	}
	out := h.frames()
	var abortResp map[string]any
	if err := json.Unmarshal(out[0], &abortResp); err != nil {
		t.Fatalf("abort response not JSON: %v", err)
	}
	if abortResp["result"] == nil {
		t.Errorf("abort result is nil: %v", abortResp)
	}
}

// TestRunRPC_ListToolsReturnsNames verifies session/listTools returns
// the registered tool names.
func TestRunRPC_ListToolsReturnsNames(t *testing.T) {
	h := newRPCTestHarness(t)
	defer h.close()

	h.writeLine(`{"jsonrpc":"2.0","id":1,"method":"session/listTools"}`)
	h.writeLine(`{"jsonrpc":"2.0","id":2,"method":"session/shutdown"}`)
	h.closeInput()
	if err := h.run(time.Second); err != nil {
		t.Fatalf("RunRPC: %v", err)
	}
	out := h.frames()
	var listResp map[string]any
	if err := json.Unmarshal(out[0], &listResp); err != nil {
		t.Fatalf("listTools response not JSON: %v", err)
	}
	result, _ := listResp["result"].(map[string]any)
	toolsList, _ := result["tools"].([]any)
	if len(toolsList) == 0 {
		t.Errorf("expected at least one tool; got: %v", result)
	}
	// "read" should be in the list since the test fixture wires it.
	sawRead := false
	for _, n := range toolsList {
		if s, ok := n.(string); ok && s == "read" {
			sawRead = true
		}
	}
	if !sawRead {
		t.Errorf("expected 'read' in tools list: %v", toolsList)
	}
}

// TestRunRPC_SendMessageMissingPromptErrors verifies -32602 when
// params lacks a prompt field.
func TestRunRPC_SendMessageMissingPromptErrors(t *testing.T) {
	h := newRPCTestHarness(t)
	defer h.close()

	h.writeLine(`{"jsonrpc":"2.0","id":1,"method":"session/sendMessage","params":{}}`)
	h.writeLine(`{"jsonrpc":"2.0","id":2,"method":"session/shutdown"}`)
	h.closeInput()
	if err := h.run(time.Second); err != nil {
		t.Fatalf("RunRPC: %v", err)
	}
	out := h.frames()
	var sendResp map[string]any
	if err := json.Unmarshal(out[0], &sendResp); err != nil {
		t.Fatalf("response not JSON: %v", err)
	}
	errObj, ok := sendResp["error"].(map[string]any)
	if !ok {
		t.Fatalf("expected error object, got: %v", sendResp)
	}
	if code, _ := errObj["code"].(float64); code != -32602 {
		t.Errorf("error.code = %v, want -32602", errObj["code"])
	}
}

// TestRunRPC_EOFTriggersCleanExit verifies RunRPC returns nil when the
// client closes stdin without sending shutdown.
func TestRunRPC_EOFTriggersCleanExit(t *testing.T) {
	h := newRPCTestHarness(t)
	defer h.close()

	h.closeInput()
	if err := h.run(time.Second); err != nil {
		t.Errorf("RunRPC on EOF: err = %v, want nil", err)
	}
}

// TestRunRPC_BlankLinesAreIgnored verifies empty lines on stdin do not
// produce responses.
func TestRunRPC_BlankLinesAreIgnored(t *testing.T) {
	h := newRPCTestHarness(t)
	defer h.close()

	h.writeLine(`   `)
	h.writeLine(``)
	h.writeLine(`{"jsonrpc":"2.0","id":1,"method":"session/shutdown"}`)
	h.closeInput()
	if err := h.run(time.Second); err != nil {
		t.Fatalf("RunRPC: %v", err)
	}
	out := h.frames()
	if len(out) != 1 {
		t.Errorf("expected 1 frame, got %d: %v", len(out), out)
	}
}

// TestRunRPC_NotificationFromClientIsDropped verifies requests with no
// id (JSON-RPC notifications) are silently accepted and not responded to.
func TestRunRPC_NotificationFromClientIsDropped(t *testing.T) {
	h := newRPCTestHarness(t)
	defer h.close()

	h.writeLine(`{"jsonrpc":"2.0","method":"client/telemetry","params":{"foo":"bar"}}`)
	h.writeLine(`{"jsonrpc":"2.0","id":1,"method":"session/shutdown"}`)
	h.closeInput()
	if err := h.run(time.Second); err != nil {
		t.Fatalf("RunRPC: %v", err)
	}
	out := h.frames()
	if len(out) != 1 {
		t.Errorf("expected 1 frame (only the shutdown ack), got %d: %v", len(out), out)
	}
}

// TestRunRPC_CancelledContextReturns verifies ctx cancellation
// propagates to RunRPC's return value even when the read is blocked
// waiting for client input.
//
// The test uses an io.Pipe so ReadBytes blocks until cancel fires —
// the realistic model for "stdin is a pipe waiting for the next frame
// when the context is cancelled." A bytes.Reader over an empty buffer
// was used here previously; it returned EOF before cancel could be
// observed, making the test nondeterministic (the goroutine could
// finish and return nil before cancel() ran).
func TestRunRPC_CancelledContextReturns(t *testing.T) {
	h := newRPCTestHarness(t)
	defer h.close()

	rIn, wIn := io.Pipe()
	defer rIn.Close()
	defer wIn.Close()

	ctx, cancel := context.WithCancel(context.Background())
	doneCh := make(chan error, 1)
	go func() {
		doneCh <- RunRPC(ctx, RPCOptions{
			Stdin:  rIn,
			Stdout: h.lockingWriter(),
		}, h.sess)
	}()
	cancel()
	select {
	case err := <-doneCh:
		if err != context.Canceled {
			t.Errorf("RunRPC on cancel: err = %v, want context.Canceled", err)
		}
	case <-time.After(time.Second):
		t.Fatalf("RunRPC did not return after ctx cancel")
	}
}

// TestRunRPC_ApproveAcknowledges verifies session/approve returns
// success even though the v1 agent loop auto-approves all calls.
func TestRunRPC_ApproveAcknowledges(t *testing.T) {
	h := newRPCTestHarness(t)
	defer h.close()

	h.writeLine(`{"jsonrpc":"2.0","id":1,"method":"session/approve","params":{"id":"tu_1","approved":true}}`)
	h.writeLine(`{"jsonrpc":"2.0","id":2,"method":"session/shutdown"}`)
	h.closeInput()
	if err := h.run(time.Second); err != nil {
		t.Fatalf("RunRPC: %v", err)
	}
	out := h.frames()
	var approveResp map[string]any
	if err := json.Unmarshal(out[0], &approveResp); err != nil {
		t.Fatalf("response not JSON: %v", err)
	}
	if approveResp["result"] == nil {
		t.Errorf("approve result is nil: %v", approveResp)
	}
}

// TestRunRPC_AbortMidTurnCancels verifies session/abort cancels the
// in-flight turn. Uses a blocking client that never completes on its
// own; without abort, RunRPC would hang. The test uses real io.Pipe so
// the abort request arrives after the turn starts.
func TestRunRPC_AbortMidTurnCancels(t *testing.T) {
	blocking := newBlockingClient()
	sess := newSessionWithClient(t, blocking)

	rIn, wIn := io.Pipe()
	out := &bytes.Buffer{}
	var mu sync.Mutex
	stdout := &mutexWriter{mu: &mu, buf: out}

	doneCh := make(chan error, 1)
	go func() {
		doneCh <- RunRPC(context.Background(), RPCOptions{
			Stdin:  rIn,
			Stdout: stdout,
		}, sess)
	}()

	writeLn(wIn, `{"jsonrpc":"2.0","id":1,"method":"session/sendMessage","params":{"prompt":"block"}}`)
	// Tiny delay so the request latches turnActive before abort.
	time.Sleep(50 * time.Millisecond)
	writeLn(wIn, `{"jsonrpc":"2.0","id":2,"method":"session/abort"}`)
	// Release so the blocking client doesn't deadlock when ctx cancels.
	blocking.release()
	writeLn(wIn, `{"jsonrpc":"2.0","id":3,"method":"session/shutdown"}`)
	wIn.Close()

	select {
	case err := <-doneCh:
		if err != nil {
			t.Fatalf("RunRPC: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatalf("RunRPC did not return")
	}

	frames := splitFrames(out.Bytes())
	if len(frames) < 3 {
		t.Errorf("expected at least 3 frames (send, abort, shutdown), got %d: %v", len(frames), frames)
	}
}

// writeLn writes s + "\n" to w. Used by tests that drive an io.Pipe.
func writeLn(w io.Writer, s string) {
	_, _ = w.Write([]byte(s + "\n"))
}

// splitFrames splits a newline-delimited buffer into byte slices,
// dropping empty/whitespace-only lines.
func splitFrames(b []byte) [][]byte {
	var out [][]byte
	for _, line := range bytes.Split(b, []byte("\n")) {
		if len(bytes.TrimSpace(line)) > 0 {
			out = append(out, line)
		}
	}
	return out
}

// --- harness ---

// rpcTestHarness owns a real *agent.AgentSession plus an input buffer
// (strings.Reader) and output buffer (bytes.Buffer under a mutex). Tests
// queue JSON-RPC lines via writeLine, then call run() which feeds the
// whole buffer to RunRPC and waits for it to return.
type rpcTestHarness struct {
	sess  *agent.AgentSession
	inbuf *bytes.Buffer
	buf   *bytes.Buffer
	mu    sync.Mutex
}

func newRPCTestHarness(t *testing.T) *rpcTestHarness {
	t.Helper()
	client := fauxprovider.NewWithResponse("rpc response")
	opts := agent.SessionOptions{
		Model:         "faux",
		Settings:      config.DefaultSettings(),
		LLMClient:     client,
		Tools:         []tools.HeadlessTool{tools.NewReadTool(tools.OSReadOperations{})},
		ContextWindow: 200000,
	}
	rt, err := agent.CreateAgentSessionRuntime(context.Background(), t.TempDir(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSessionRuntime: %v", err)
	}
	sess := agent.NewAgentSession(rt)
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	return &rpcTestHarness{
		sess:  sess,
		inbuf: &bytes.Buffer{},
		buf:   &bytes.Buffer{},
	}
}

// writeLine appends one line of input to the in-buffer. No I/O happens
// until run() is called.
func (h *rpcTestHarness) writeLine(s string) {
	h.inbuf.WriteString(s)
	h.inbuf.WriteByte('\n')
}

// closeInput is a no-op kept for API stability; the in-buffer is
// consumed atomically by run().
func (h *rpcTestHarness) closeInput() {}

// run feeds the in-buffer to RunRPC and blocks until it returns or the
// timeout elapses.
func (h *rpcTestHarness) run(timeout time.Duration) error {
	opts := RPCOptions{
		Stdin:  bytes.NewReader(h.inbuf.Bytes()),
		Stdout: h.lockingWriter(),
	}
	done := make(chan error, 1)
	go func() {
		done <- RunRPC(context.Background(), opts, h.sess)
	}()
	select {
	case err := <-done:
		return err
	case <-time.After(timeout):
		return io.ErrNoProgress
	}
}

// frames returns the captured output split into individual JSON objects
// (one per line). Trims blank trailing lines.
func (h *rpcTestHarness) frames() [][]byte {
	h.mu.Lock()
	defer h.mu.Unlock()
	var out [][]byte
	for _, line := range bytes.Split(h.buf.Bytes(), []byte("\n")) {
		if len(bytes.TrimSpace(line)) > 0 {
			out = append(out, line)
		}
	}
	return out
}

// lockingWriter returns a writer that serializes writes via h.mu.
// RunRPC writes one JSON object per line; the mutex keeps whole-object
// writes atomic so the test can split on newlines safely.
func (h *rpcTestHarness) lockingWriter() io.Writer {
	return &mutexWriter{mu: &h.mu, buf: h.buf}
}

// close is a no-op; t.Cleanup handles session shutdown.
func (h *rpcTestHarness) close() {}

// mutexWriter adapts a (*sync.Mutex, *bytes.Buffer) pair to io.Writer.
type mutexWriter struct {
	mu  *sync.Mutex
	buf *bytes.Buffer
}

func (w *mutexWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.buf.Write(p)
}

// newSessionWithClient wires a fresh session against the given client.
// Used by tests that need a custom client (e.g., the blocking client).
func newSessionWithClient(t *testing.T, client llm.LLMClient) *agent.AgentSession {
	t.Helper()
	opts := agent.SessionOptions{
		Model:         "faux",
		Settings:      config.DefaultSettings(),
		LLMClient:     client,
		Tools:         []tools.HeadlessTool{tools.NewReadTool(tools.OSReadOperations{})},
		ContextWindow: 200000,
	}
	rt, err := agent.CreateAgentSessionRuntime(context.Background(), t.TempDir(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSessionRuntime: %v", err)
	}
	sess := agent.NewAgentSession(rt)
	t.Cleanup(func() { sess.Shutdown(context.Background()) })
	return sess
}

// blockingClient is a deterministic LLMClient that blocks Stream until
// release is called. Used to test concurrent sendMessage rejection.
type blockingClient struct {
	once     sync.Once
	released chan struct{}
}

func newBlockingClient() *blockingClient {
	return &blockingClient{released: make(chan struct{})}
}

func (b *blockingClient) release() {
	b.once.Do(func() { close(b.released) })
}

func (b *blockingClient) Stream(ctx context.Context, _ llm.Request) (<-chan llm.Delta, error) {
	ch := make(chan llm.Delta, 2)
	go func() {
		defer close(ch)
		select {
		case <-ctx.Done():
			return
		case <-b.released:
		}
		select {
		case <-ctx.Done():
			return
		case ch <- llm.TextDelta{Text: "released"}:
		}
		select {
		case <-ctx.Done():
			return
		case ch <- llm.Final{StopReason: llm.StopReasonEndTurn}:
		}
	}()
	return ch, nil
}

// guard against accidental real os.Stdout pollution if a future edit
// removes the lockingWriter.
var _ = os.Stdout
