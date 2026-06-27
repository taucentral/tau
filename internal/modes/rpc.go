// rpc.go — JSON-RPC 2.0 server over stdin/stdout.
//
// The RPC mode is designed for IDE and editor integrations. It speaks
// JSON-RPC 2.0 over stdin/stdout: one JSON object per line on input
// (requests from the client) and one JSON object per line on output
// (responses and notifications from the server).
//
// Methods (see openspec/changes/initial/specs/modes/spec.md):
//
//   - session/start        initialize the session
//   - session/sendMessage  send a user prompt; triggers one agentic turn
//   - session/abort        cancel the in-flight turn
//   - session/listTools    list registered tool names
//   - session/shutdown     clean shutdown
//   - session/approve      approve a pending tool call (reserved for future
//                          approval-aware tools; the v1 agent loop does
//                          not block on approvals)
//
// Notifications (server → client, no id):
//
//   - notifications/messageDelta    incremental assistant text
//   - notifications/toolCall        tool invocation started
//   - notifications/toolResult      tool completed
//   - notifications/turnEnd         turn finished
//   - notifications/approvalRequired  approval needed (reserved)
//
// Each notification's params carry a "type" discriminator for future
// extensibility; the v1 shapes are documented on each handler below.

package modes

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/coevin/tau/internal/agent"
	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/state"
)

// RPCOptions is the input bundle for RunRPC. Stdin/Stdout/Stderr default
// to the process streams when nil.
type RPCOptions struct {
	Stdin  io.Reader
	Stdout io.Writer
	Stderr io.Writer
}

// RunRPC reads JSON-RPC requests from opts.Stdin until the client sends
// session/shutdown, stdin reaches EOF, or ctx is cancelled. Responses
// and notifications are written to opts.Stdout as newline-delimited
// JSON objects.
//
// RunRPC returns nil on clean shutdown, ctx.Err() when cancelled, or a
// descriptive error on internal failures (e.g., illegal state). The
// caller is responsible for session.Shutdown after RunRPC returns.
func RunRPC(ctx context.Context, opts RPCOptions, session *agent.AgentSession) error {
	if session == nil {
		return errors.New("modes.RunRPC: session is nil")
	}
	stdin := opts.Stdin
	if stdin == nil {
		stdin = os.Stdin
	}
	stdout := opts.Stdout
	if stdout == nil {
		stdout = os.Stdout
	}
	stderr := opts.Stderr
	if stderr == nil {
		stderr = os.Stderr
	}
	_ = stderr // reserved for diagnostic logging

	srv := newRPCServer(session, stdout)
	defer srv.stop()

	// Main loop: read a line, dispatch, repeat. A line may be longer
	// than bufio's default token cap; Scanner is avoided for that
	// reason. The reader here blocks until input is available or the
	// context fires.
	reader := bufio.NewReader(stdin)
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		line, readErr := reader.ReadBytes('\n')
		// Trim trailing whitespace; blank lines are skipped.
		trimmed := trimSpaceBytes(line)
		if len(trimmed) > 0 {
			if err := srv.handleLine(ctx, trimmed); err != nil {
				if errors.Is(err, errShutdown) {
					return nil
				}
				return err
			}
		}
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				return nil
			}
			return fmt.Errorf("rpc: read stdin: %w", readErr)
		}
	}
}

// errShutdown is the sentinel returned by handleLine when the client
// requested session/shutdown. RunRPC translates it to a nil return.
var errShutdown = errors.New("rpc: shutdown requested")

// rpcServer owns the per-session JSON-RPC machinery: the stdout mutex,
// the event-forwarding goroutine, and the in-flight turn cancellation.
type rpcServer struct {
	session *agent.AgentSession
	out     io.Writer
	mu      sync.Mutex // guards writes to out

	// turnMu guards turn state. Only one turn may be active at a time.
	// handleSendMessage acquires turnActive before running; handleAbort
	// and handleShutdown cancel via turnCancel without acquiring the
	// active-turn slot.
	turnMu       sync.Mutex
	turnCancel   context.CancelFunc
	turnActive   bool
	shuttingDown bool

	// turnWG is non-zero while a turn goroutine is running.
	// handleShutdown waits on it before responding so we don't exit
	// mid-turn.
	turnWG sync.WaitGroup
}

// newRPCServer returns a fresh server bound to the given session.
func newRPCServer(session *agent.AgentSession, out io.Writer) *rpcServer {
	return &rpcServer{session: session, out: out}
}

// stop tears down any in-flight turn.
func (s *rpcServer) stop() {
	s.turnMu.Lock()
	if s.turnCancel != nil {
		s.turnCancel()
		s.turnCancel = nil
	}
	s.turnMu.Unlock()
}

// handleLine parses one JSON-RPC message and dispatches it. Returns
// errShutdown when the client requested shutdown.
func (s *rpcServer) handleLine(ctx context.Context, line []byte) error {
	var req rpcRequest
	if err := json.Unmarshal(line, &req); err != nil {
		s.writeResponse(rpcResponse{
			JSONRPC: "2.0",
			ID:      nil,
			Error:   &rpcError{Code: -32700, Message: "parse error: " + err.Error()},
		})
		return nil //nolint:nilerr // JSON-RPC servers reply with an error object, not a Go error
	}
	if req.JSONRPC == "" {
		req.JSONRPC = "2.0" // tolerant; require nothing but the body
	}
	// A request with no id is a notification; the server does not
	// respond to notifications. The current client→server protocol has
	// no defined notifications, so we silently drop.
	if req.ID == nil {
		return nil
	}

	switch req.Method {
	case "session/start":
		return s.handleStart(req)
	case "session/sendMessage":
		// Dispatch in a goroutine so the main loop can read the next
		// request — specifically session/abort — while the turn runs.
		// handleShutdown waits on turnWG before responding.
		s.turnWG.Add(1)
		go func() {
			defer s.turnWG.Done()
			s.handleSendMessage(ctx, req)
		}()
		return nil
	case "session/abort":
		return s.handleAbort(req)
	case "session/listTools":
		return s.handleListTools(req)
	case "session/approve":
		return s.handleApprove(req)
	case "session/shutdown":
		return s.handleShutdown(req)
	default:
		s.writeResponse(rpcResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error:   &rpcError{Code: -32601, Message: "method not found: " + req.Method},
		})
		return nil
	}
}

// handleStart acknowledges session/start. The session is already wired
// by the time RunRPC is invoked; this method exists so IDEs can fetch
// the session identity for display.
func (s *rpcServer) handleStart(req rpcRequest) error {
	rt := s.session.Runtime()
	sid := rt.SessionID
	if sid == "" {
		// Fall back to the header. Same behavior as print mode.
		if tree, err := rt.State.Tree(); err == nil {
			if hdr, ok := tree.Root().Payload.(state.SessionHeaderPayload); ok {
				sid = hdr.SessionID
			}
		}
	}
	s.writeResponseOK(req.ID, map[string]any{
		"sessionId": sid,
		"model":     rt.Options.Model,
		"cwd":       rt.Cwd,
	})
	return nil
}

// handleSendMessage runs one agentic turn in a goroutine, forwarding
// events as notifications. The response is sent after the turn completes
// (or the turn is cancelled by session/abort / session/shutdown).
//
// handleSendMessage is dispatched in its own goroutine by handleLine so
// the main RunRPC loop can keep reading stdin for session/abort.
func (s *rpcServer) handleSendMessage(ctx context.Context, req rpcRequest) {
	// Reject concurrent sends. (With the goroutine dispatch model,
	// a second sendMessage would race for turnActive; the loser gets
	// rejected here.)
	s.turnMu.Lock()
	if s.turnActive {
		s.turnMu.Unlock()
		s.writeError(req.ID, -32603, "turn already in progress")
		return
	}
	s.turnActive = true
	s.turnMu.Unlock()

	// Extract the prompt from params.
	var params struct {
		Prompt string `json:"prompt"`
	}
	if len(req.Params) > 0 {
		if err := json.Unmarshal(req.Params, &params); err != nil {
			s.finishTurn()
			s.writeError(req.ID, -32602, "invalid params: "+err.Error())
			return
		}
	}
	if params.Prompt == "" {
		s.finishTurn()
		s.writeError(req.ID, -32602, "invalid params: prompt is required")
		return
	}

	// Subscribe before launching the turn. The bus does not support
	// per-subscriber unsubscribe (channels close only on bus Shutdown),
	// so we stop the forwarding goroutine via a stop channel and drain
	// any in-flight events after Run returns.
	bus := s.session.Runtime().EventBus
	eventsCh := bus.Subscribe()
	eventsDone := make(chan struct{})
	stop := make(chan struct{})
	go func() {
		defer close(eventsDone)
		for {
			select {
			case evt, ok := <-eventsCh:
				if !ok {
					return
				}
				s.forwardEvent(evt)
			case <-stop:
				// Drain any remaining events that were already
				// in the channel when stop fired.
				for {
					select {
					case evt, ok := <-eventsCh:
						if !ok {
							return
						}
						s.forwardEvent(evt)
					default:
						return
					}
				}
			}
		}
	}()

	turnCtx, cancel := context.WithCancel(ctx)
	s.turnMu.Lock()
	s.turnCancel = cancel
	s.turnMu.Unlock()

	// Run synchronously; the JSON-RPC response is sent after the turn
	// completes (or is cancelled). The notification goroutine drains
	// events in parallel.
	runErr := s.session.Run(turnCtx, params.Prompt)
	cancel()

	// Signal the forwarding goroutine to exit, then wait for it to
	// finish draining any in-flight events.
	close(stop)
	<-eventsDone

	s.finishTurn()

	if runErr != nil {
		s.writeError(req.ID, -32603, "turn failed: "+runErr.Error())
		return
	}
	s.writeResponseOK(req.ID, map[string]any{"ok": true})
}

// finishTurn clears the in-flight turn state under the turn mutex.
func (s *rpcServer) finishTurn() {
	s.turnMu.Lock()
	s.turnCancel = nil
	s.turnActive = false
	s.turnMu.Unlock()
}

// handleAbort cancels the in-flight turn (if any).
func (s *rpcServer) handleAbort(req rpcRequest) error {
	s.turnMu.Lock()
	cancel := s.turnCancel
	s.turnMu.Unlock()
	if cancel != nil {
		cancel()
	}
	s.writeResponseOK(req.ID, map[string]any{"ok": true})
	return nil
}

// handleListTools returns the names of registered tools.
func (s *rpcServer) handleListTools(req rpcRequest) error {
	rt := s.session.Runtime()
	names := rt.Registry.Names()
	s.writeResponseOK(req.ID, map[string]any{"tools": names})
	return nil
}

// handleApprove is reserved for future approval-aware tools. The v1
// agent loop auto-approves all tool calls, so this returns success
// with no side effects.
func (s *rpcServer) handleApprove(req rpcRequest) error {
	s.writeResponseOK(req.ID, map[string]any{"ok": true, "note": "v1 auto-approves all tool calls"})
	return nil
}

// handleShutdown acknowledges shutdown and signals RunRPC to exit. It
// cancels any in-flight turn and waits for the turn goroutine to
// finish before writing the response, so the client sees a clean
// "turn cancelled" notification before the shutdown ack.
func (s *rpcServer) handleShutdown(req rpcRequest) error {
	s.turnMu.Lock()
	s.shuttingDown = true
	if s.turnCancel != nil {
		s.turnCancel()
		s.turnCancel = nil
	}
	s.turnMu.Unlock()
	// Wait for any in-flight turn goroutine to finish so we don't
	// race on stdout writes during exit.
	s.turnWG.Wait()
	s.writeResponseOK(req.ID, map[string]any{"ok": true})
	return errShutdown
}

// forwardEvent converts an agent.Event to the matching notification
// and writes it. Unknown events are dropped.
func (s *rpcServer) forwardEvent(evt agent.Event) {
	switch e := evt.(type) {
	case agent.MessageUpdateEvent:
		switch d := e.Delta.(type) {
		case llm.TextDelta:
			s.writeNotification("notifications/messageDelta", map[string]any{
				"text": d.Text,
			})
		case llm.ToolCallDelta:
			// First-delta for a content index carries name+id; we
			// forward the full call in the dedicated toolCall
			// notification instead of fragmenting the args JSON.
			if d.Name != "" {
				s.writeNotification("notifications/toolCall", map[string]any{
					"id":   d.ID,
					"name": d.Name,
				})
			}
		}
	case agent.ToolCallEvent:
		// The ToolCallEvent carries the full parsed call. Emit once
		// with the structured args so clients can render an inline
		// card before the result arrives.
		s.writeNotification("notifications/toolCall", map[string]any{
			"id":   e.Call.ID,
			"name": e.Call.Name,
			"args": e.Call.Args,
		})
	case agent.ToolResultEvent:
		s.writeNotification("notifications/toolResult", map[string]any{
			"id":     e.ToolID,
			"result": extractText(e.Result.Content),
			"error":  e.Result.IsError,
		})
	case agent.TurnEndEvent:
		s.writeNotification("notifications/turnEnd", map[string]any{
			"turn":     e.Turn,
			"finished": e.Finished,
		})
	}
}

// writeResponseOK emits a JSON-RPC success response.
func (s *rpcServer) writeResponseOK(id json.RawMessage, result any) {
	raw, err := json.Marshal(result)
	if err != nil {
		s.writeResponse(rpcResponse{
			JSONRPC: "2.0",
			ID:      id,
			Error:   &rpcError{Code: -32603, Message: "marshal result: " + err.Error()},
		})
		return
	}
	s.writeResponse(rpcResponse{
		JSONRPC: "2.0",
		ID:      id,
		Result:  raw,
	})
}

// writeError emits a JSON-RPC error response.
func (s *rpcServer) writeError(id json.RawMessage, code int, msg string) {
	s.writeResponse(rpcResponse{
		JSONRPC: "2.0",
		ID:      id,
		Error:   &rpcError{Code: code, Message: msg},
	})
}

// writeNotification emits a server→client notification (no id).
func (s *rpcServer) writeNotification(method string, params any) {
	raw, err := json.Marshal(params)
	if err != nil {
		return
	}
	s.writeResponse(rpcResponse{
		JSONRPC: "2.0",
		Method:  method,
		Params:  raw,
	})
}

// writeResponse is the single stdout writer. All output goes through
// here so the mutex guarantees whole-object atomicity.
func (s *rpcServer) writeResponse(resp rpcResponse) {
	b, err := json.Marshal(resp)
	if err != nil {
		return
	}
	b = append(b, '\n')
	s.mu.Lock()
	defer s.mu.Unlock()
	_, _ = s.out.Write(b)
}

// trimSpaceBytes returns s with leading and trailing ASCII whitespace
// removed. Avoids the strings package for a hot path.
func trimSpaceBytes(s []byte) []byte {
	start, end := 0, len(s)
	for start < end && isASCIISpace(s[start]) {
		start++
	}
	for end > start && isASCIISpace(s[end-1]) {
		end--
	}
	return s[start:end]
}

// isASCIISpace reports whether b is space, tab, CR, or LF.
func isASCIISpace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\r' || b == '\n'
}
