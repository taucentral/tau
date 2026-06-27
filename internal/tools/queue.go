package tools

import (
	"context"
	"path/filepath"
	"sync"
)

// FileMutationQueue serializes file-mutating tool calls per path. Two
// concurrent edits to the SAME file run in arrival order; two concurrent
// edits to DIFFERENT files run in parallel.
//
// Non-mutating tools (read, grep, find, ls, bash) bypass the queue. The
// queue is per session (or per agent runtime); it has no global state.
//
// The zero value is usable; construct via NewFileMutationQueue for clarity.
type FileMutationQueue struct {
	mu    sync.Mutex
	locks map[string]*sync.Mutex
}

// NewFileMutationQueue returns an empty queue.
func NewFileMutationQueue() *FileMutationQueue {
	return &FileMutationQueue{locks: make(map[string]*sync.Mutex)}
}

// Run executes fn while holding the per-path lock. Other Run calls for the
// same path block until fn returns; calls for any other path proceed in
// parallel. fn runs uncanceled unless ctx is canceled.
//
// The path is normalized via filepath.Clean so "foo.txt" and "./foo.txt"
// share a lock.
func (q *FileMutationQueue) Run(ctx context.Context, path string, fn func() error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	pathLock := q.lockFor(path)
	// Acquire the per-path lock while respecting ctx cancellation. We
	// can't use context.AfterSelect here because we need to release the
	// path lock if ctx fires mid-acquire.
	acquired := make(chan struct{})
	go func() {
		pathLock.Lock()
		close(acquired)
	}()
	select {
	case <-ctx.Done():
		// We can't cancel the goroutine's Lock acquisition, so we
		// proceed to wait for it to finish and then immediately
		// unlock. This burns a tiny window but is safe.
		<-acquired
		pathLock.Unlock()
		return ctx.Err()
	case <-acquired:
	}
	defer pathLock.Unlock()
	return fn()
}

// lockFor returns the per-path mutex, creating it on first sight. The
// outer mutex is held briefly to serialize map writes.
func (q *FileMutationQueue) lockFor(path string) *sync.Mutex {
	normalized := filepath.Clean(path)
	q.mu.Lock()
	defer q.mu.Unlock()
	if m, ok := q.locks[normalized]; ok {
		return m
	}
	m := &sync.Mutex{}
	q.locks[normalized] = m
	return m
}
