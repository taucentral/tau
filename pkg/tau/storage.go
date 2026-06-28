// storage.go — public SDK aliases for the cross-session storage seam.
//
// Every type and sentinel in this file is an alias for the canonical
// definition in internal/storage. Identity is preserved: pkg/tau.Store
// IS internal/storage.Store, pkg/tau.Entry IS internal/storage.Entry,
// and errors.Is(err, tau.ErrUnsupportedQuery) works against errors
// produced by internal code because the values are the same.
//
// The Store interface is the SDK's extension point for cross-session
// context: an embedder can write entries (decisions, facts, summaries)
// during one turn and query them in subsequent turns — even from a
// different session. The reference backend is the file-backed FileStore
// (re-exported via NewFileStore); embedders who want embeddings,
// sqlite, or vector DB support implement Store themselves and inject
// via Options.Store.
//
// Reference: docs/input/context/plugin-support/whitepaper.md §3.4.

package tau

import (
	"github.com/coevin/tau/internal/storage"
)

// Entry is a single unit of stored context. The ID is the only field
// the runtime treats as a primary key. See internal/storage.Entry for
// the field-by-field documentation.
type Entry = storage.Entry

// Query selects entries from a Store. Zero-value fields are ignored;
// the match is the AND of every non-zero field. See
// internal/storage.Query for the field-by-field documentation.
type Query = storage.Query

// Store is the cross-session context interface implemented by every
// storage backend. See internal/storage.Store for the lifecycle and
// concurrency contracts.
type Store = storage.Store

// ErrStoreClosed is returned by Put and Query after Close has been
// called. The runtime never closes a store it did not create; this
// sentinel is for embedders managing their own store lifecycles.
//
// Re-exported from internal/storage so errors.Is identity holds
// against errors produced by any backend.
var ErrStoreClosed = storage.ErrStoreClosed

// ErrStoreReadOnly is returned by Put when the store was constructed
// in read-only mode.
//
// Re-exported from internal/storage.
var ErrStoreReadOnly = storage.ErrStoreReadOnly

// ErrUnsupportedQuery is returned by Query when the backend cannot
// satisfy the requested query shape. The canonical trigger is
// Query.EmbeddingQuery against FileStore.
//
// Re-exported from internal/storage so errors.Is identity holds
// against errors produced by any backend.
var ErrUnsupportedQuery = storage.ErrUnsupportedQuery
