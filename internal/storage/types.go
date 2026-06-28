// types.go — canonical Store interface and Entry/Query types.
//
// This package owns the storage type definitions so internal packages
// (agent, runtime) can consume them without importing pkg/tau. The
// SDK re-exports every type and sentinel as aliases: pkg/tau.Store
// IS internal/storage.Store, pkg/tau.Entry IS internal/storage.Entry,
// and so on. Identity is preserved.
//
// Reference: docs/input/context/plugin-support/whitepaper.md §3.4.

package storage

import (
	"context"
	"errors"
	"time"
)

// Entry is a single unit of stored context. The ID is the only field
// the runtime treats as a primary key — two Puts with the same ID
// overwrite. Embedders choose ID conventions (UUIDs, slugs, content
// hashes); the runtime does not generate IDs.
type Entry struct {
	// ID is the stable identifier for the entry. Required on Put;
	// FileStore (and any future backend) refuses empty IDs.
	ID string

	// Text is the entry body. Stored verbatim by FileStore; the
	// reference backend renders it as the markdown body under the
	// YAML frontmatter. Embedders using a structured backend may
	// serialise structured data here.
	Text string

	// Tags are free-form labels used by Query.TagsQuery. FileStore
	// matches entries whose Tags include every queried tag (AND).
	Tags []string

	// Embedding is an optional dense vector. FileStore does not
	// compute embeddings and returns ErrUnsupportedQuery when
	// Query.EmbeddingQuery is set; vector-aware backends use it
	// for semantic similarity queries.
	Embedding []float32

	// Timestamp is the entry's logical time. FileStore sorts
	// results by file mtime descending and applies SinceQuery as
	// a Timestamp filter.
	Timestamp time.Time

	// Source is the entry's provenance — typically the session id
	// or user id that produced it. FileStore round-trips it
	// verbatim; the runtime does not interpret it.
	Source string
}

// Query selects entries from a Store. Zero-value fields are ignored;
// at least one field SHOULD be set (a zero Query matches everything
// up to Limit). When multiple fields are set, the match is the AND
// of every non-zero field.
type Query struct {
	// KeywordQuery matches entries whose Text contains the
	// case-insensitive substring. FileStore performs the search
	// in Go; vector-aware backends may ignore this field.
	KeywordQuery string

	// EmbeddingQuery is a dense vector for semantic similarity.
	// FileStore returns ErrUnsupportedQuery when this field is
	// non-empty; vector-aware backends return entries ordered
	// by similarity to the query vector.
	EmbeddingQuery []float32

	// TagsQuery matches entries whose Tags include every queried
	// tag (AND semantics). Order does not matter; duplicates in
	// TagsQuery are deduplicated by the implementation.
	TagsQuery []string

	// SinceQuery filters by Timestamp: only entries whose
	// Timestamp is at or after SinceQuery are returned.
	SinceQuery time.Time

	// Limit caps the result count. Zero means "use the backend's
	// default limit"; FileStore's default is 100. A negative
	// Limit is treated as zero.
	Limit int
}

// Store is the cross-session context interface implemented by every
// storage backend. The agent runtime accepts a caller-supplied Store
// via SessionOptions.Store; when nil, the runtime runs without
// storage features.
//
// Lifecycle contract:
//
//   - The runtime does NOT call Close on a store supplied via
//     SessionOptions.Store. The embedder owns the injected store's
//     lifecycle. This matches the asymmetry documented on
//     SessionOptions.Store: unlike StateManager (which has a
//     runtime-created default that the runtime closes on Shutdown),
//     Store has no default — nil means "no store" — so there is
//     nothing for the runtime to close.
//
// Concurrency contract:
//
//   - Implementations MUST be safe for concurrent use. The agent
//     loop MAY issue Query calls from a middleware goroutine while
//     another goroutine Puts; readers never block writers and vice
//     versa (FileStore achieves this via per-Put file locks).
type Store interface {
	// Put writes (or overwrites) a single entry. Returns a typed
	// error (ErrStoreReadOnly, ErrStoreClosed) for the documented
	// failure modes; other errors surface whatever the underlying
	// backend produced.
	Put(ctx context.Context, e Entry) error

	// Query returns entries matching q. The returned slice is
	// owned by the caller; mutations do not affect the store.
	// Returns ErrUnsupportedQuery when the backend cannot satisfy
	// the query shape (e.g., EmbeddingQuery against FileStore).
	Query(ctx context.Context, q Query) ([]Entry, error)

	// Close releases any held resources. Safe to call multiple
	// times. After Close, Put and Query return ErrStoreClosed.
	// The runtime does NOT call Close on stores supplied via
	// SessionOptions.Store; the embedder owns the store's lifecycle.
	Close() error
}

// ErrStoreClosed is returned by Put and Query after Close has been
// called. The runtime never closes a store it did not create; this
// sentinel is for embedders managing their own store lifecycles.
var ErrStoreClosed = errors.New("storage: store is closed")

// ErrStoreReadOnly is returned by Put when the store was constructed
// in read-only mode. FileStore does not use this sentinel today; it
// is exported so embedders building a read-only backend (e.g., a
// snapshot mounted from a tarball) have a stable error identity.
var ErrStoreReadOnly = errors.New("storage: store is read-only")

// ErrUnsupportedQuery is returned by Query when the backend cannot
// satisfy the requested query shape. The canonical trigger is
// Query.EmbeddingQuery against FileStore, which does not compute
// embeddings.
var ErrUnsupportedQuery = errors.New("storage: unsupported query")

// DefaultLimit is the result cap applied by FileStore when Query.Limit
// is zero. Embedders MAY override by setting Query.Limit explicitly.
const DefaultLimit = 100
