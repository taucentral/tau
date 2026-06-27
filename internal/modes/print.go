// Package modes implements tau's three run modes: print, rpc, interactive.
//
// Each mode handler takes a wired *agent.AgentSession and a small options
// bundle (the prompt string, JSON flag, etc.) and writes its output to
// stdout/stderr. The modes package does not parse argv or load Settings —
// the cli layer does that and hands the wired session over.
//
// The modes package is the only place that performs stdout/stderr writes
// for run-mode paths; the cli layer is in charge of metadata subcommands
// (--help, --version) and the first-run setup wizard.
package modes

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/coevin/tau/internal/agent"
	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/state"
)

// PrintOptions is the input bundle to RunPrint. The cli layer fills it
// from cli.Args (kept as a separate type so the modes package doesn't
// depend on cli — that would be a circular import).
type PrintOptions struct {
	// Prompt is the user's input string (the joined positional args).
	Prompt string

	// JSON requests structured output. When false, RunPrint writes only
	// the final assistant text to stdout.
	JSON bool

	// ExportPath, when non-empty, writes the transcript to the named
	// file in addition to stdout. Useful for piping.
	ExportPath string

	// Stdout and Stderr let tests capture output without swapping the
	// process's real os.Stdout. When nil, RunPrint uses os.Stdout /
	// os.Stderr directly.
	Stdout io.Writer
	Stderr io.Writer
}

// PrintResult captures the data RunPrint used to produce output. Tests
// inspect it directly instead of scraping stdout. The struct is also the
// shape of the JSON output when PrintOptions.JSON is true.
type PrintResult struct {
	// Messages is the full transcript (user + assistant + tool-result
	// messages) in root → leaf order.
	Messages []PrintMessage `json:"messages"`

	// ToolCalls is the ordered list of tool invocations and their
	// outcomes. Empty when the model made no tool calls.
	ToolCalls []PrintToolCall `json:"toolCalls"`

	// Usage totals from the model's Final deltas. Empty when no usage
	// information was reported.
	Usage PrintUsage `json:"usage"`

	// SessionID is the session identifier, suitable for `tau --resume
	// <id>`. May be empty for ephemeral (in-memory) sessions.
	SessionID string `json:"sessionID"`

	// Text is the final assistant message's concatenated text content.
	// Empty when the assistant produced no text (e.g., a tool-only
	// response — rare but valid).
	Text string `json:"-"`
}

// PrintMessage is one row in the transcript. Content is the rendered
// text of the message (text blocks concatenated; tool blocks represented
// as "[tool_use name=... id=...]" for readability).
type PrintMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// PrintToolCall is one tool invocation in the trace.
type PrintToolCall struct {
	ID     string          `json:"id"`
	Name   string          `json:"name"`
	Args   json.RawMessage `json:"args,omitempty"`
	Result string          `json:"result"`
	Error  bool            `json:"error,omitempty"`
}

// PrintUsage is the per-turn token totals. Fields are zero when the
// provider didn't report them.
type PrintUsage struct {
	InputTokens  int `json:"inputTokens"`
	OutputTokens int `json:"outputTokens"`
	CacheRead    int `json:"cacheRead,omitempty"`
	CacheWrite   int `json:"cacheWrite,omitempty"`
}

// RunPrint executes one agentic turn against session and writes the
// result to opts.Stdout (or os.Stdout when nil). Returns nil on success;
// non-nil errors are surfaced to the caller, which the cli layer renders
// to stderr with exit code 1.
//
// RunPrint blocks until session.Run returns or ctx is cancelled. The
// caller is responsible for session.Shutdown after RunPrint returns.
func RunPrint(ctx context.Context, opts PrintOptions, session *agent.AgentSession) (*PrintResult, error) {
	if session == nil {
		return nil, errors.New("modes.RunPrint: session is nil")
	}
	if opts.Prompt == "" {
		return nil, errors.New("modes.RunPrint: prompt is empty")
	}
	stdout := opts.Stdout
	if stdout == nil {
		stdout = os.Stdout
	}
	stderr := opts.Stderr
	if stderr == nil {
		stderr = os.Stderr
	}
	_ = stderr // reserved for future per-token progress reporting

	// Subscribe to events to capture tool calls and usage. Subscribe
	// BEFORE Run so we don't miss the first event. The bus does not
	// support per-subscriber unsubscribe; we stop the collector via a
	// done channel after Run returns and drain any in-flight events.
	bus := session.Runtime().EventBus
	eventsCh := bus.Subscribe()
	collector := newEventCollector()
	collectorDone := make(chan struct{})
	stop := make(chan struct{})
	go func() {
		defer close(collectorDone)
		for {
			select {
			case evt, ok := <-eventsCh:
				if !ok {
					return
				}
				collector.handle(evt)
			case <-stop:
				// Drain any remaining events that were already
				// in the channel when stop fired.
				for {
					select {
					case evt, ok := <-eventsCh:
						if !ok {
							return
						}
						collector.handle(evt)
					default:
						return
					}
				}
			}
		}
	}()

	// Run one agentic turn. Blocks until the model returns Final with no
	// pending tool calls, or ctx is cancelled.
	runErr := session.Run(ctx, opts.Prompt)

	// Signal the collector to exit, then wait for it to finish draining.
	close(stop)
	<-collectorDone

	if runErr != nil {
		return nil, runErr
	}

	result, err := buildPrintResult(session, collector)
	if err != nil {
		return nil, err
	}

	// Write the output.
	if err := writePrintOutput(stdout, opts, result); err != nil {
		return nil, err
	}

	if opts.ExportPath != "" {
		if err := writeExportFile(opts.ExportPath, opts, result); err != nil {
			return nil, fmt.Errorf("write export %q: %w", opts.ExportPath, err)
		}
	}
	return result, nil
}

// buildPrintResult walks the session's state tree and the captured events
// to assemble a PrintResult.
func buildPrintResult(session *agent.AgentSession, coll *eventCollector) (*PrintResult, error) {
	rt := session.Runtime()
	tree, err := rt.State.Tree()
	if err != nil {
		return nil, fmt.Errorf("read state tree: %w", err)
	}
	leaf := rt.State.LeafID()
	if leaf == "" {
		leaf = tree.RootID()
	}
	path, err := tree.Path(leaf)
	if err != nil {
		return nil, fmt.Errorf("walk state tree: %w", err)
	}
	result := &PrintResult{
		SessionID: rt.SessionID,
		Usage:     coll.usage,
		ToolCalls: coll.toolCalls,
	}
	// Fall back to the session header's SessionID when the runtime
	// wasn't given an explicit one. The factory-created manager
	// assigns a fresh UUID at Create time.
	if result.SessionID == "" {
		if hdr, ok := tree.Root().Payload.(state.SessionHeaderPayload); ok {
			result.SessionID = hdr.SessionID
		}
	}
	for _, node := range path {
		if node.Kind != state.KindMessage {
			continue
		}
		mp, ok := node.Payload.(state.MessagePayload)
		if !ok {
			continue
		}
		msg := PrintMessage{Role: string(mp.Role)}
		textParts := []string{}
		for _, block := range mp.Content {
			switch b := block.(type) {
			case llm.TextContent:
				textParts = append(textParts, b.Text)
			case llm.ToolUse:
				textParts = append(textParts, fmt.Sprintf("[tool_use name=%s id=%s]", b.Name, b.ID))
			case llm.ToolResult:
				// ToolResult content is rendered into the user-role
				// message that wraps it; the trace's ToolCalls slice
				// already captures the structured result.
				if b.IsError {
					textParts = append(textParts, "[tool_result error]")
				} else {
					textParts = append(textParts, "[tool_result]")
				}
			}
		}
		// Join with newline so multi-block messages stay readable.
		for i, p := range textParts {
			if i > 0 {
				msg.Content += "\n"
			}
			msg.Content += p
		}
		result.Messages = append(result.Messages, msg)
	}

	// The "final assistant text" is the text of the last assistant
	// message in the path. Walk backward to find it.
	for i := len(result.Messages) - 1; i >= 0; i-- {
		if result.Messages[i].Role == string(llm.RoleAssistant) {
			// Strip any tool_use markers — we only want the text body.
			result.Text = stripToolUseMarkers(result.Messages[i].Content)
			break
		}
	}
	return result, nil
}

// writePrintOutput writes either the JSON document or the plain text to w.
func writePrintOutput(w io.Writer, opts PrintOptions, result *PrintResult) error {
	if opts.JSON {
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		if err := enc.Encode(result); err != nil {
			return fmt.Errorf("encode json: %w", err)
		}
		return nil
	}
	// Text mode: final assistant text + trailing newline. Empty text
	// still gets a newline so shell pipelines see *something*.
	if _, err := fmt.Fprintln(w, result.Text); err != nil {
		return fmt.Errorf("write stdout: %w", err)
	}
	return nil
}

// writeExportFile writes the same content as writePrintOutput to the
// named file. Truncates if it exists.
func writeExportFile(path string, opts PrintOptions, result *PrintResult) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	return writePrintOutput(f, opts, result)
}

// stripToolUseMarkers removes the "[tool_use name=... id=...]" lines that
// buildPrintResult inserts into the assistant message body. The text
// mode writes only the textual portion of the assistant response.
func stripToolUseMarkers(s string) string {
	out := []string{}
	// Split on newline and drop lines that look like tool markers.
	for _, line := range splitLines(s) {
		if isToolUseMarker(line) || isToolResultMarker(line) {
			continue
		}
		out = append(out, line)
	}
	// Rejoin with newline; trim trailing whitespace.
	joined := ""
	for i, l := range out {
		if i > 0 {
			joined += "\n"
		}
		joined += l
	}
	return joined
}

// isToolUseMarker reports whether s starts with the marker prefix
// inserted by buildPrintResult for tool_use blocks.
func isToolUseMarker(s string) bool {
	return startsWith(s, "[tool_use ")
}

// isToolResultMarker reports whether s starts with the marker prefix
// inserted for tool_result blocks.
func isToolResultMarker(s string) bool {
	return startsWith(s, "[tool_result")
}

// startsWith is a small wrapper around strings.HasPrefix so the file
// doesn't need to import the strings package solely for two checks.
func startsWith(s, prefix string) bool {
	if len(s) < len(prefix) {
		return false
	}
	return s[:len(prefix)] == prefix
}

// splitLines breaks s into lines without a trailing empty entry when s
// ends with a newline. Used by stripToolUseMarkers.
func splitLines(s string) []string {
	if s == "" {
		return nil
	}
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}

// --- event collector ---

// eventCollector accumulates tool-call events and usage data from the
// session's EventBus while Run is executing. It is safe for use from a
// single goroutine that drains the subscribe channel.
type eventCollector struct {
	toolCalls []PrintToolCall
	usage     PrintUsage
}

// newEventCollector returns an empty collector.
func newEventCollector() *eventCollector {
	return &eventCollector{}
}

// extractText concatenates TextContent blocks from a ContentBlock slice
// into a single string. Non-text blocks are skipped. Used to render a
// plain-text view of a ToolResult for the JSON trace.
func extractText(blocks []llm.ContentBlock) string {
	out := ""
	for i, b := range blocks {
		if t, ok := b.(llm.TextContent); ok {
			if i > 0 && out != "" {
				out += "\n"
			}
			out += t.Text
		}
	}
	return out
}

// handle type-switches on evt and updates the collector's state.
func (c *eventCollector) handle(evt agent.Event) {
	switch e := evt.(type) {
	case agent.ToolCallEvent:
		// Append a pending tool call; the result event fills it in.
		c.toolCalls = append(c.toolCalls, PrintToolCall{
			ID:   e.Call.ID,
			Name: e.Call.Name,
			Args: e.Call.Args,
		})
	case agent.ToolResultEvent:
		// Find the matching call and fill in its result text.
		for i := len(c.toolCalls) - 1; i >= 0; i-- {
			if c.toolCalls[i].ID == e.ToolID {
				c.toolCalls[i].Result = extractText(e.Result.Content)
				c.toolCalls[i].Error = e.Result.IsError
				break
			}
		}
	case agent.MessageUpdateEvent:
		// Accumulate usage from UsageDelta deltas. Providers emit
		// exactly one UsageDelta per stream, after the last content
		// delta and before Final.
		if u, ok := e.Delta.(llm.UsageDelta); ok {
			c.usage.InputTokens += u.InputTokens
			c.usage.OutputTokens += u.OutputTokens
			c.usage.CacheRead += u.CacheReadTokens
			c.usage.CacheWrite += u.CacheWriteTokens
		}
	}
}
