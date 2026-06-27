// Package e2e contains deterministic end-to-end tests for the agent loop.
//
// faux_provider.go provides a scripted LLMClient that replays a pre-set
// sequence of Delta values per call to Stream. Tests construct a script
// (a slice of Delta slices, one per expected turn iteration) and pass
// the faux client into SessionOptions. The faux client:
//
//   - Records the requests it received for later assertion.
//   - Returns a buffered channel pre-populated with the next script's
//     deltas so consumers don't race on stream start.
//   - Honours ctx cancellation by closing the channel early.
//   - Panics if a Stream call arrives when no scripts remain (a test
//     setup bug, not a runtime failure).
//
// The faux client is safe for concurrent use within a single test
// (multiple Stream calls serialize via a mutex).
package e2e

import (
	"context"
	"errors"
	"sync"

	"github.com/coevin/tau/internal/llm"
)

// FauxProvider is a deterministic LLMClient for end-to-end agent tests.
// Each call to Stream consumes the next entry in Scripts. The entry's
// deltas are delivered on a buffered channel.
//
// Construct via NewFauxProvider (or literal &FauxProvider{Scripts: ...}).
// After the test runs, Requests has one entry per Stream call.
type FauxProvider struct {
	mu sync.Mutex

	// Scripts is the per-call delta sequences. Scripts[0] is emitted on
	// the first Stream call, Scripts[1] on the second, etc. Tests with
	// multi-iteration turns set up one script per iteration.
	Scripts [][]llm.Delta

	// Requests records each Request passed to Stream, in call order.
	// Safe to read after the test's Run call returns.
	Requests []llm.Request

	// ErrStreamStart, when non-nil, is returned immediately by Stream
	// instead of producing a channel. Used to test failure modes.
	ErrStreamStart error

	// idx tracks the next script index. Mutable under mu.
	idx int
}

// NewFauxProvider returns a FauxProvider with the given scripts.
func NewFauxProvider(scripts ...[]llm.Delta) *FauxProvider {
	return &FauxProvider{Scripts: scripts}
}

// Stream returns a buffered channel pre-populated with the next script's
// deltas. The caller's ctx is honoured: if cancelled while the consumer
// is still draining, the consumer (the agent loop) returns ctx.Err().
//
// Stream is safe for concurrent use: scripts are consumed atomically.
// Returns ErrStreamStart instead of a channel when that field is set.
func (f *FauxProvider) Stream(ctx context.Context, req llm.Request) (<-chan llm.Delta, error) {
	f.mu.Lock()
	if f.ErrStreamStart != nil {
		f.mu.Unlock()
		return nil, f.ErrStreamStart
	}
	if f.idx >= len(f.Scripts) {
		f.mu.Unlock()
		return nil, errors.New("e2e.FauxProvider: no script left; the test's expected turn count is wrong")
	}
	script := f.Scripts[f.idx]
	f.idx++
	f.Requests = append(f.Requests, req)
	f.mu.Unlock()

	// Buffered so the producer never blocks even if the consumer is
	// slow. Deltas are emitted with a short select to honour ctx.
	ch := make(chan llm.Delta, len(script))
	go func() {
		defer close(ch)
		for _, d := range script {
			select {
			case <-ctx.Done():
				return
			case ch <- d:
			}
		}
	}()
	return ch, nil
}

// Calls returns the number of Stream calls so far. Safe for concurrent
// use. Useful for asserting the turn-loop's iteration count.
func (f *FauxProvider) Calls() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.idx
}

// RecordedRequests returns a copy of the Requests slice captured so
// far. Useful for post-turn assertions about tool schemas, system
// prompt, and conversation content.
func (f *FauxProvider) RecordedRequests() []llm.Request {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]llm.Request, len(f.Requests))
	copy(out, f.Requests)
	return out
}

// Compile-time check: FauxProvider satisfies llm.LLMClient.
var _ llm.LLMClient = (*FauxProvider)(nil)
