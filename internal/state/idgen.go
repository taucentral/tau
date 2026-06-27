package state

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
)

// idLen is the number of hex characters in an entry ID. The spec mandates
// 8 hex characters, giving 2^32 = 4.3 billion possible IDs per session
// (collision probability per generation ≈ 1 in 4 billion on a 1-entry
// session; birthday bound reached at ~65k entries).
const idLen = 8

// maxIDRetries caps collision retries per the spec ("fails after 8 retries").
const maxIDRetries = 8

// ErrIDCollision is returned by NewID when 8 consecutive retries all
// generated IDs that already exist in the session. With idLen=8 this is
// astronomically unlikely (would require >65k entries in a single session
// to even reach the birthday bound) but we surface it as a typed sentinel
// rather than panicking.
var ErrIDCollision = errors.New("state: id collision after 8 retries")

// idBytes is the byte count that encodes to idLen hex characters.
var idBytes = idLen / 2

// NewID generates a random 8-hex-character ID. The exists callback reports
// whether a candidate ID is already in use; the generator retries with a
// fresh random ID until exists returns false, up to maxIDRetries times.
//
// exists is called at most maxIDRetries+1 times. For bbolt-backed stores
// the implementation should do a bucket Get inside the callback.
func NewID(exists func(string) bool) (string, error) {
	buf := make([]byte, idBytes)
	for attempt := 0; attempt < maxIDRetries; attempt++ {
		if _, err := rand.Read(buf); err != nil {
			return "", err
		}
		id := hex.EncodeToString(buf)
		if !exists(id) {
			return id, nil
		}
	}
	return "", ErrIDCollision
}
