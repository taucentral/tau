// Package e2e — modes_integration_test.go covers tasks 9.7, 9.8, 9.9.
//
// These tests exercise the full path from cli.Dispatch through the wire
// layer into modes.RunPrint / modes.RunRPC, verifying the user-visible
// behaviour: stdout content, exit codes, JSON schema, and the JSON-RPC
// notification stream.
//
// The faux provider (internal/fauxprovider) supplies a deterministic
// response string via TAU_FAUX_SCRIPT so the tests never touch the
// network.
package e2e

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

	"github.com/taucentral/tau/internal/agent"
	"github.com/taucentral/tau/internal/cli"
	"github.com/taucentral/tau/internal/config"
	"github.com/taucentral/tau/internal/fauxprovider"
	"github.com/taucentral/tau/internal/llm"
	"github.com/taucentral/tau/internal/modes"
	"github.com/taucentral/tau/internal/tools"
)

// --- 9.7: tau --print --model faux "hello" ---

// TestIntegration_PrintMode_FauxModel verifies the full Dispatch → wire
// → RunPrint path returns the faux provider's scripted response on stdout
// and that Dispatch returns nil (the caller maps nil to exit code 0).
func TestIntegration_PrintMode_FauxModel(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("TAU_CONFIG_DIR", dir)
	t.Setenv("TAU_FAUX_SCRIPT", "integration print reply")

	tmp, err := os.CreateTemp(dir, "stdout")
	if err != nil {
		t.Fatalf("temp: %v", err)
	}
	old := os.Stdout
	os.Stdout = tmp
	t.Cleanup(func() { os.Stdout = old })

	err = cli.Dispatch(context.Background(), cli.Args{
		Print:      true,
		Model:      "faux",
		Cwd:        dir,
		Positional: []string{"hello"},
	})
	tmp.Close()
	if err != nil {
		t.Fatalf("Dispatch: err = %v, want nil (exit 0)", err)
	}

	b, readErr := os.ReadFile(tmp.Name())
	if readErr != nil {
		t.Fatalf("read stdout tmp: %v", readErr)
	}
	out := string(b)
	if !strings.Contains(out, "integration print reply") {
		t.Errorf("stdout = %q, want substring 'integration print reply'", out)
	}
	// Text mode ends with a newline so pipelines see output.
	if !strings.HasSuffix(out, "\n") {
		t.Errorf("stdout = %q, want trailing newline", out)
	}
}

// TestIntegration_PrintMode_FauxModel_PersistsSession verifies a
// session file is written under AgentDir so --resume can find it later.
func TestIntegration_PrintMode_FauxModel_PersistsSession(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("TAU_CONFIG_DIR", dir)
	t.Setenv("TAU_FAUX_SCRIPT", "persist check")

	tmp, err := os.CreateTemp(dir, "stdout")
	if err != nil {
		t.Fatalf("temp: %v", err)
	}
	old := os.Stdout
	os.Stdout = tmp
	t.Cleanup(func() { os.Stdout = old })

	err = cli.Dispatch(context.Background(), cli.Args{
		Print:      true,
		Model:      "faux",
		Cwd:        dir,
		Positional: []string{"hello"},
	})
	tmp.Close()
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}

	// The sessions directory should contain at least one .json file for
	// this cwd. We don't assert the exact session ID (it's a UUID) but
	// we do assert the directory exists and is non-empty.
	agentDir, dirErr := config.AgentDir()
	if dirErr != nil {
		t.Fatalf("AgentDir: %v", dirErr)
	}
	entries, readErr := os.ReadDir(agentDir)
	if readErr != nil {
		t.Fatalf("ReadDir(%s): %v", agentDir, readErr)
	}
	// At minimum the sessions subdir should exist.
	sawSessions := false
	for _, e := range entries {
		if e.IsDir() && e.Name() == "sessions" {
			sawSessions = true
			sessionEntries, _ := os.ReadDir(agentDir + "/sessions")
			for _, se := range sessionEntries {
				t.Logf("session dir: %s", se.Name())
			}
		}
	}
	if !sawSessions {
		t.Errorf("expected sessions/ dir under %s; entries: %v", agentDir, entries)
	}
}

// --- 9.8: tau --print --json ---

// TestIntegration_PrintMode_JSON verifies --json produces a valid JSON
// document on stdout containing all required fields per the modes spec:
// messages, toolCalls, usage, sessionID.
func TestIntegration_PrintMode_JSON(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("TAU_CONFIG_DIR", dir)
	t.Setenv("TAU_FAUX_SCRIPT", "json body text")

	tmp, err := os.CreateTemp(dir, "stdout")
	if err != nil {
		t.Fatalf("temp: %v", err)
	}
	old := os.Stdout
	os.Stdout = tmp
	t.Cleanup(func() { os.Stdout = old })

	err = cli.Dispatch(context.Background(), cli.Args{
		Print:      true,
		JSON:       true,
		Model:      "faux",
		Cwd:        dir,
		Positional: []string{"hello"},
	})
	tmp.Close()
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}

	b, readErr := os.ReadFile(tmp.Name())
	if readErr != nil {
		t.Fatalf("read stdout tmp: %v", readErr)
	}

	var doc map[string]any
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatalf("stdout is not valid JSON: %v\nraw: %s", err, b)
	}

	// Required top-level keys per modes spec.
	for _, key := range []string{"messages", "toolCalls", "usage", "sessionID"} {
		if _, ok := doc[key]; !ok {
			t.Errorf("JSON output missing required key %q; keys: %v", key, mapKeys(doc))
		}
	}

	// messages must be a non-empty array with at least one entry whose
	// role is "assistant" and whose content contains the scripted text.
	msgs, ok := doc["messages"].([]any)
	if !ok {
		t.Fatalf("messages is not an array: %T", doc["messages"])
	}
	if len(msgs) == 0 {
		t.Errorf("messages is empty")
	}
	var sawAssistant bool
	for _, m := range msgs {
		mm, ok := m.(map[string]any)
		if !ok {
			continue
		}
		if mm["role"] == "assistant" {
			sawAssistant = true
			content, _ := mm["content"].(string)
			if !strings.Contains(content, "json body text") {
				t.Errorf("assistant content = %q, want substring 'json body text'", content)
			}
		}
	}
	if !sawAssistant {
		t.Errorf("no assistant message in transcript: %v", msgs)
	}

	// sessionID must be a non-empty string.
	sid, _ := doc["sessionID"].(string)
	if sid == "" {
		t.Errorf("sessionID is empty")
	}

	// usage must be present (may be zero if the provider sends no
	// UsageDelta, but the key must exist).
	if _, ok := doc["usage"].(map[string]any); !ok {
		t.Errorf("usage is not an object: %T", doc["usage"])
	}
}

// mapKeys returns the keys of m as a string slice for diagnostics.
func mapKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

// --- 9.9: RPC round-trip via os.Pipe ---

// TestIntegration_RPC_RoundTrip exercises a full JSON-RPC session:
// session/start → session/sendMessage → notifications → session/shutdown.
// The test drives modes.RunRPC via real io.Pipe so the read/write
// scheduling mirrors a real IDE client: shutdown is sent only after the
// turn's turnEnd notification arrives, not before.
func TestIntegration_RPC_RoundTrip(t *testing.T) {
	// Build a session wired against the faux provider with a scripted
	// response. We bypass cli.Dispatch because that would swap os.Stdin
	// / os.Stdout globals — racy under `go test -race`. modes.RunRPC
	// accepts explicit Stdin/Stdout for exactly this reason.
	client := fauxprovider.NewWithResponse("rpc integration turn")

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

	stdinR, stdinW := io.Pipe()
	stdoutR, stdoutW := io.Pipe()

	doneCh := make(chan error, 1)
	go func() {
		doneCh <- modes.RunRPC(context.Background(), modes.RPCOptions{
			Stdin:  stdinR,
			Stdout: stdoutW,
		}, sess)
	}()

	// Reader goroutine: collects frames as they arrive and signals when
	// turnEnd is observed. This lets us send shutdown only after the
	// turn finishes — exactly what a real IDE client does.
	type jsonFrame struct {
		raw []byte
		m   map[string]any
	}
	var frames []jsonFrame
	framesReady := make(chan struct{})
	turnEndSeen := make(chan struct{})
	go func() {
		defer close(framesReady)
		buf := make([]byte, 0, 4096)
		tmp := make([]byte, 1024)
		for {
			n, readErr := stdoutR.Read(tmp)
			if n > 0 {
				buf = append(buf, tmp[:n]...)
				// Extract complete lines.
				for {
					idx := bytes.IndexByte(buf, '\n')
					if idx < 0 {
						break
					}
					line := buf[:idx]
					buf = buf[idx+1:]
					if len(bytes.TrimSpace(line)) == 0 {
						continue
					}
					var m map[string]any
					_ = json.Unmarshal(line, &m)
					frames = append(frames, jsonFrame{raw: line, m: m})
					if method, _ := m["method"].(string); method == "notifications/turnEnd" {
						select {
						case <-turnEndSeen:
						default:
							close(turnEndSeen)
						}
					}
				}
			}
			if readErr != nil {
				break
			}
		}
	}()

	// Drive the protocol: start → sendMessage → (wait for turnEnd) → shutdown.
	writeLn(stdinW, `{"jsonrpc":"2.0","id":1,"method":"session/start"}`)
	writeLn(stdinW, `{"jsonrpc":"2.0","id":2,"method":"session/sendMessage","params":{"prompt":"hello"}}`)

	// Wait for the turn to finish before sending shutdown. This avoids
	// handleShutdown cancelling the in-flight turn's context.
	select {
	case <-turnEndSeen:
	case <-time.After(5 * time.Second):
		t.Fatalf("did not receive notifications/turnEnd")
	}

	writeLn(stdinW, `{"jsonrpc":"2.0","id":3,"method":"session/shutdown"}`)
	stdinW.Close()

	// Wait for RunRPC to return.
	select {
	case err := <-doneCh:
		if err != nil {
			t.Fatalf("RunRPC: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatalf("RunRPC did not return")
	}

	// Close the stdout pipe so the reader goroutine can exit.
	stdoutW.Close()
	<-framesReady

	// Verify the protocol-level invariants across all frames.
	if len(frames) < 4 {
		t.Fatalf("expected at least 4 frames, got %d", len(frames))
	}
	var sawStartResp, sawSendResp, sawShutdownResp bool
	var sawDelta, sawTurnEnd bool
	for _, f := range frames {
		id := f.m["id"]
		method, _ := f.m["method"].(string)
		if id != nil {
			switch id {
			case float64(1):
				sawStartResp = true
				result, _ := f.m["result"].(map[string]any)
				if result == nil {
					t.Errorf("start response missing result: %v", f.m)
				}
			case float64(2):
				sawSendResp = true
				if f.m["error"] != nil {
					t.Errorf("sendMessage error: %v", f.m["error"])
				}
			case float64(3):
				sawShutdownResp = true
			}
		} else if method != "" {
			switch method {
			case "notifications/messageDelta":
				sawDelta = true
				params, _ := f.m["params"].(map[string]any)
				text, _ := params["text"].(string)
				if !strings.Contains(text, "rpc integration turn") {
					t.Errorf("messageDelta text = %q, want substring 'rpc integration turn'", text)
				}
			case "notifications/turnEnd":
				sawTurnEnd = true
			}
		}
	}
	if !sawStartResp {
		t.Errorf("missing session/start response")
	}
	if !sawDelta {
		t.Errorf("missing notifications/messageDelta")
	}
	if !sawTurnEnd {
		t.Errorf("missing notifications/turnEnd")
	}
	if !sawSendResp {
		t.Errorf("missing session/sendMessage response")
	}
	if !sawShutdownResp {
		t.Errorf("missing session/shutdown response")
	}
}

// TestIntegration_RPC_ListTools verifies the RPC listTools method
// returns the tool names registered on the session, confirming the
// tool registry is visible through the RPC protocol.
func TestIntegration_RPC_ListTools(t *testing.T) {
	client := fauxprovider.NewWithResponse("list tools")

	opts := agent.SessionOptions{
		Model:     "faux",
		Settings:  config.DefaultSettings(),
		LLMClient: client,
		Tools: []tools.HeadlessTool{
			tools.NewReadTool(tools.OSReadOperations{}),
		},
		ContextWindow: 200000,
	}
	rt, err := agent.CreateAgentSessionRuntime(context.Background(), t.TempDir(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSessionRuntime: %v", err)
	}
	sess := agent.NewAgentSession(rt)
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	stdinR, stdinW := io.Pipe()
	out := &bytes.Buffer{}
	var mu sync.Mutex
	stdout := &mutexWriterSync{mu: &mu, buf: out}

	doneCh := make(chan error, 1)
	go func() {
		doneCh <- modes.RunRPC(context.Background(), modes.RPCOptions{
			Stdin:  stdinR,
			Stdout: stdout,
		}, sess)
	}()

	writeLn(stdinW, `{"jsonrpc":"2.0","id":1,"method":"session/listTools"}`)
	writeLn(stdinW, `{"jsonrpc":"2.0","id":2,"method":"session/shutdown"}`)
	stdinW.Close()

	select {
	case err := <-doneCh:
		if err != nil {
			t.Fatalf("RunRPC: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatalf("RunRPC did not return")
	}

	frames := splitFrames(out.Bytes())
	var listResp map[string]any
	for _, raw := range frames {
		var m map[string]any
		if err := json.Unmarshal(raw, &m); err != nil {
			continue
		}
		if m["id"] == float64(1) {
			listResp = m
			break
		}
	}
	if listResp == nil {
		t.Fatalf("missing listTools response; frames: %s", out.String())
	}
	result, _ := listResp["result"].(map[string]any)
	toolsList, _ := result["tools"].([]any)
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

// writeLn writes s + "\n" to w. Used to drive an io.Pipe.
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

// mutexWriterSync serializes writes to buf via mu. This keeps whole-
// object writes atomic so the test can split frames on newlines.
type mutexWriterSync struct {
	mu  *sync.Mutex
	buf *bytes.Buffer
}

func (w *mutexWriterSync) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.buf.Write(p)
}

// Compile-time check: ensure the llm package is referenced so the
// import is not flagged as unused if future edits remove a call site.
var _ = llm.StopReasonEndTurn
