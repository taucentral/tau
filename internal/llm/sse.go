// sse.go — minimal SSE reader shared by all providers.
//
// The Server-Sent Events stream format is line-based:
//
//   - Lines beginning with ":" are comments (heartbeats).
//   - "event: foo" sets the event type for the next data block.
//   - "data: ..." appends to the current data buffer (newline-separated).
//   - A blank line dispatches the current event.
//
// This reader does NOT support multi-line "data:" continuations the way
// browsers do — providers we care about (Anthropic, OpenAI) emit each
// event as a single "data:" line followed by a blank line.

package llm

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"strings"
)

// SSEEvent is one decoded Server-Sent Event.
type SSEEvent struct {
	// Type is the value of the "event:" line, or "message" if omitted
	// (the SSE default per the spec).
	Type string
	// Data is the concatenated payload (one line per "data:" line).
	Data string
}

// ErrSSEClosed is returned by ReadSSE when the underlying reader returns
// io.EOF without a terminal blank line.
var ErrSSEClosed = errors.New("sse: stream closed without final event")

// ReadSSE scans r as an SSE stream and emits events to fn. Returns nil on a
// clean EOF (the stream ended with a complete event followed by EOF). The
// caller MUST close r after ReadSSE returns.
//
// fn is invoked for each complete event. If fn returns an error, ReadSSE
// stops and returns that error.
func ReadSSE(r io.Reader, fn func(SSEEvent) error) error {
	scanner := bufio.NewScanner(r)
	// Anthropic/OpenAI events can carry large JSON payloads (especially
	// content_block_start with tool_use schemas). Lift the per-line buffer.
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

	var (
		eventType string
		dataLines []string
	)
	dispatch := func() error {
		if len(dataLines) == 0 {
			// Empty event (just a blank line).
			eventType = ""
			return nil
		}
		ev := SSEEvent{
			Type: eventType,
			Data: strings.Join(dataLines, "\n"),
		}
		if ev.Type == "" {
			ev.Type = "message"
		}
		if err := fn(ev); err != nil {
			return err
		}
		eventType = ""
		dataLines = nil
		return nil
	}

	for scanner.Scan() {
		line := scanner.Text()
		// Strip trailing \r in case of CRLF line endings.
		line = strings.TrimRight(line, "\r")

		if line == "" {
			if err := dispatch(); err != nil {
				return err
			}
			continue
		}
		if strings.HasPrefix(line, ":") {
			// Comment / heartbeat.
			continue
		}
		if strings.HasPrefix(line, "event:") {
			eventType = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			continue
		}
		if strings.HasPrefix(line, "data:") {
			dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			continue
		}
		// Other fields (id:, retry:) are ignored.
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("sse: scan: %w", err)
	}
	// Flush any pending event after EOF.
	if len(dataLines) > 0 || eventType != "" {
		return dispatch()
	}
	return nil
}
