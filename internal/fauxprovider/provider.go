// Package fauxprovider is a built-in deterministic LLMClient used by the
// CLI's print/rpc integration tests and by `tau --model faux` invocations.
//
// The faux provider returns a fixed script per Stream call so tests can
// assert the binary's end-to-end behavior without hitting a real API. It
// is wired by the CLI's wire layer when args.Model == "faux" (or when
// the test harness explicitly injects it).
//
// The script is taken from the TAU_FAUX_SCRIPT environment variable when
// set; otherwise it defaults to DefaultResponse. TAU_FAUX_SCRIPT is a
// newline-delimited list of "text:<chunk>" entries — one TextDelta per
// line — followed by an implicit Final. This keeps the format trivial
// for tests to construct.
package fauxprovider

import (
	"context"
	"errors"
	"os"
	"strings"
	"sync"

	"github.com/coevin/tau/internal/llm"
)

// DefaultResponse is the canned message returned when TAU_FAUX_SCRIPT is
// not set. Tests can verify `tau --print --model faux "hello"` returns
// exactly this string.
const DefaultResponse = "Hello from the faux provider."

// Client is a deterministic LLMClient. Every Stream call returns the same
// script: a single TextDelta containing the configured response, followed
// by a Final with StopReason "end_turn".
//
// Requests are recorded for test assertions.
type Client struct {
	mu       sync.Mutex
	response string
	requests []llm.Request
}

// New returns a Client configured from the environment. If TAU_FAUX_SCRIPT
// is unset or empty, the response is DefaultResponse.
func New() *Client {
	resp := strings.TrimSpace(os.Getenv("TAU_FAUX_SCRIPT"))
	if resp == "" {
		resp = DefaultResponse
	}
	return &Client{response: resp}
}

// NewWithResponse returns a Client with an explicit response. Useful when
// the caller wants to avoid touching the process environment.
func NewWithResponse(response string) *Client {
	return &Client{response: response}
}

// Stream emits the canned script. Each call appends req to the recorded
// requests slice. The deltas are emitted on a buffered channel so the
// consumer's loop never blocks on stream start.
func (c *Client) Stream(ctx context.Context, req llm.Request) (<-chan llm.Delta, error) {
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	c.mu.Lock()
	c.requests = append(c.requests, req)
	resp := c.response
	c.mu.Unlock()

	ch := make(chan llm.Delta, 2)
	go func() {
		defer close(ch)
		select {
		case <-ctx.Done():
			return
		case ch <- llm.TextDelta{Text: resp}:
		}
		select {
		case <-ctx.Done():
			return
		case ch <- llm.Final{StopReason: llm.StopReasonEndTurn}:
		}
	}()
	return ch, nil
}

// Response returns the canned response the Client will emit on the next
// Stream call.
func (c *Client) Response() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.response
}

// RecordedRequests returns a copy of the requests the Client has seen.
// Useful for tests asserting the request payload reached the provider.
func (c *Client) RecordedRequests() []llm.Request {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]llm.Request, len(c.requests))
	copy(out, c.requests)
	return out
}

// ErrFauxProviderAborted is returned by Stream when ctx is cancelled
// before the first delta is emitted. Tests use it to verify the abort
// path without panicking the provider.
var ErrFauxProviderAborted = errors.New("fauxprovider: stream aborted before first delta")

// Compile-time check: Client satisfies llm.LLMClient.
var _ llm.LLMClient = (*Client)(nil)
