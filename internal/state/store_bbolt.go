package state

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"go.etcd.io/bbolt"
)

// Bucket names per state-tree spec ("buckets `entries` ... and `meta`").
const (
	bucketEntries = "entries"
	bucketMeta    = "meta"
)

// Meta keys per state-tree spec ("meta (keyed by `leaf`, `info`, `labels`)").
const (
	metaKeyLeaf   = "leaf"
	metaKeyInfo   = "info"
	metaKeyLabels = "labels"
)

// storeFileMode matches the security posture of trust.json/auth.json (0600):
// session contents may include user code, secrets in tool output, etc., so
// the file is owner-read/write only.
const storeFileMode = 0600

// openTimeout caps how long OpenStore waits to acquire the bbolt file lock.
// A shorter timeout surfaces "another process holds the lock" as a quick
// error rather than an indefinite hang.
const openTimeout = 5 * time.Second

// ErrStoreClosed is returned when any Store method is called after Close.
// Close nils the underlying bbolt handle, so post-close operations fail
// fast instead of panicking inside bbolt.
var ErrStoreClosed = errors.New("state: store is closed")

// ErrEntryNotFound is returned by Store.Get when the requested entry ID is
// not in the store.
var ErrEntryNotFound = errors.New("state: entry not found")

// Store wraps a bbolt DB for a single session. All operations are
// concurrency-safe via bbolt's MVCC; readers get a consistent snapshot even
// while a writer is mid-commit (per state-tree spec scenario "Concurrent
// reader").
//
// The store layer knows nothing about Leaves, Trees, or context building —
// those are Manager concerns. Store handles primitives: read/write entries,
// read/write meta keys, atomic batch appends.
type Store struct {
	db *bbolt.DB
}

// OpenStore opens (or creates) the bbolt file at path. The file is created
// with mode 0600. Returns a wrapped error if path exists but is not a valid
// bbolt file, or if another process holds the file lock.
func OpenStore(path string) (*Store, error) {
	db, err := bbolt.Open(path, storeFileMode, &bbolt.Options{Timeout: openTimeout})
	if err != nil {
		return nil, fmt.Errorf("state: open %s: %w", path, err)
	}
	s := &Store{db: db}
	if err := s.initBuckets(); err != nil {
		db.Close()
		return nil, err
	}
	return s, nil
}

// initBuckets creates the entries and meta buckets idempotently.
func (s *Store) initBuckets() error {
	return s.db.Update(func(tx *bbolt.Tx) error {
		if _, err := tx.CreateBucketIfNotExists([]byte(bucketEntries)); err != nil {
			return fmt.Errorf("state: create %s bucket: %w", bucketEntries, err)
		}
		if _, err := tx.CreateBucketIfNotExists([]byte(bucketMeta)); err != nil {
			return fmt.Errorf("state: create %s bucket: %w", bucketMeta, err)
		}
		return nil
	})
}

// Close closes the underlying bbolt DB. After Close, all operations return
// ErrStoreClosed. Close is idempotent.
func (s *Store) Close() error {
	if s.db == nil {
		return nil
	}
	err := s.db.Close()
	s.db = nil
	return err
}

// Path returns the on-disk file path. Empty when the store is closed.
func (s *Store) Path() string {
	if s.db == nil {
		return ""
	}
	return s.db.Path()
}

// Append writes a single entry inside one Update transaction. Returns a
// wrapped error if the entry ID is empty, already exists, or fails to
// marshal.
func (s *Store) Append(entry Entry) error {
	if s.db == nil {
		return ErrStoreClosed
	}
	return s.db.Update(func(tx *bbolt.Tx) error {
		return appendEntry(tx, entry)
	})
}

// AppendBatch writes multiple entries inside a single Update transaction.
// Atomicity guarantee per state-tree spec scenario "Atomic append": either
// all entries commit or none do.
func (s *Store) AppendBatch(entries []Entry) error {
	if s.db == nil {
		return ErrStoreClosed
	}
	return s.db.Update(func(tx *bbolt.Tx) error {
		for _, e := range entries {
			if err := appendEntry(tx, e); err != nil {
				return err
			}
		}
		return nil
	})
}

// appendEntry is the per-entry write used by both Append and AppendBatch.
// It runs inside the caller's transaction so batch appends stay atomic.
func appendEntry(tx *bbolt.Tx, e Entry) error {
	b := tx.Bucket([]byte(bucketEntries))
	if b == nil {
		return fmt.Errorf("state: %s bucket missing", bucketEntries)
	}
	if e.ID == "" {
		return errors.New("state: entry has empty ID")
	}
	if b.Get([]byte(e.ID)) != nil {
		return fmt.Errorf("state: entry ID %q already exists", e.ID)
	}
	data, err := json.Marshal(e)
	if err != nil {
		return fmt.Errorf("state: marshal entry %q: %w", e.ID, err)
	}
	return b.Put([]byte(e.ID), data)
}

// Get returns the entry with the given ID. Returns ErrEntryNotFound when
// the ID is not present in the store.
func (s *Store) Get(id string) (Entry, error) {
	if s.db == nil {
		return Entry{}, ErrStoreClosed
	}
	var entry Entry
	err := s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte(bucketEntries))
		if b == nil {
			return fmt.Errorf("state: %s bucket missing", bucketEntries)
		}
		data := b.Get([]byte(id))
		if data == nil {
			return ErrEntryNotFound
		}
		if err := json.Unmarshal(data, &entry); err != nil {
			return fmt.Errorf("state: unmarshal entry %q: %w", id, err)
		}
		return nil
	})
	return entry, err
}

// All returns every entry in the store. Order is bbolt's key-order which
// callers must NOT rely on; pass the result to NewTree which sorts and
// validates structure.
func (s *Store) All() ([]Entry, error) {
	if s.db == nil {
		return nil, ErrStoreClosed
	}
	var out []Entry
	err := s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte(bucketEntries))
		if b == nil {
			return fmt.Errorf("state: %s bucket missing", bucketEntries)
		}
		return b.ForEach(func(k, v []byte) error {
			var e Entry
			if err := json.Unmarshal(v, &e); err != nil {
				return fmt.Errorf("state: unmarshal entry %q: %w", string(k), err)
			}
			out = append(out, e)
			return nil
		})
	})
	return out, err
}

// Exists reports whether the given entry ID is already in the store. Used
// by idgen.NewID as its collision callback.
func (s *Store) Exists(id string) (bool, error) {
	if s.db == nil {
		return false, ErrStoreClosed
	}
	exists := false
	err := s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte(bucketEntries))
		if b == nil {
			return fmt.Errorf("state: %s bucket missing", bucketEntries)
		}
		exists = b.Get([]byte(id)) != nil
		return nil
	})
	return exists, err
}

// SetLeaf writes the leaf entry ID to meta.
func (s *Store) SetLeaf(id string) error {
	return s.setMeta(metaKeyLeaf, []byte(id))
}

// LeafID reads the leaf entry ID from meta. Returns "" when no leaf is set
// (e.g., on a freshly-initialized store before any append).
func (s *Store) LeafID() (string, error) {
	v, err := s.getMeta(metaKeyLeaf)
	if err != nil {
		return "", err
	}
	return string(v), nil
}

// SetInfo writes the session-info map to meta as JSON. The map stores
// session-level metadata (e.g., exit reason, run duration) that does not
// belong on any single entry.
func (s *Store) SetInfo(info map[string]string) error {
	data, err := json.Marshal(info)
	if err != nil {
		return fmt.Errorf("state: marshal info: %w", err)
	}
	return s.setMeta(metaKeyInfo, data)
}

// Info reads the session-info map from meta. Returns nil when no info is
// set.
func (s *Store) Info() (map[string]string, error) {
	v, err := s.getMeta(metaKeyInfo)
	if err != nil {
		return nil, err
	}
	if len(v) == 0 {
		return nil, nil
	}
	var info map[string]string
	if err := json.Unmarshal(v, &info); err != nil {
		return nil, fmt.Errorf("state: unmarshal info: %w", err)
	}
	return info, nil
}

// SetLabels writes the label slice to meta as JSON.
func (s *Store) SetLabels(labels []string) error {
	data, err := json.Marshal(labels)
	if err != nil {
		return fmt.Errorf("state: marshal labels: %w", err)
	}
	return s.setMeta(metaKeyLabels, data)
}

// Labels reads the label slice from meta. Returns nil when no labels are
// set.
func (s *Store) Labels() ([]string, error) {
	v, err := s.getMeta(metaKeyLabels)
	if err != nil {
		return nil, err
	}
	if len(v) == 0 {
		return nil, nil
	}
	var labels []string
	if err := json.Unmarshal(v, &labels); err != nil {
		return nil, fmt.Errorf("state: unmarshal labels: %w", err)
	}
	return labels, nil
}

// setMeta writes a single meta key inside a Update transaction.
func (s *Store) setMeta(key string, value []byte) error {
	if s.db == nil {
		return ErrStoreClosed
	}
	return s.db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte(bucketMeta))
		if b == nil {
			return fmt.Errorf("state: %s bucket missing", bucketMeta)
		}
		return b.Put([]byte(key), value)
	})
}

// getMeta reads a single meta key inside a View transaction. The returned
// slice is a copy; bbolt reuses its mmap buffer across calls so the caller
// can hold the slice beyond the transaction.
func (s *Store) getMeta(key string) ([]byte, error) {
	if s.db == nil {
		return nil, ErrStoreClosed
	}
	var out []byte
	err := s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte(bucketMeta))
		if b == nil {
			return fmt.Errorf("state: %s bucket missing", bucketMeta)
		}
		if v := b.Get([]byte(key)); v != nil {
			out = append([]byte(nil), v...)
		}
		return nil
	})
	return out, err
}

// Update exposes a raw bbolt Update transaction for callers (notably
// migrations) that need to rewrite the entries bucket atomically.
func (s *Store) Update(fn func(tx *bbolt.Tx) error) error {
	if s.db == nil {
		return ErrStoreClosed
	}
	return s.db.Update(fn)
}

// View exposes a raw bbolt View transaction for callers that need
// consistent multi-read atomicity (notably BuildContext which reads every
// entry it needs inside one snapshot).
func (s *Store) View(fn func(tx *bbolt.Tx) error) error {
	if s.db == nil {
		return ErrStoreClosed
	}
	return s.db.View(fn)
}
