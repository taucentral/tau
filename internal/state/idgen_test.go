package state

import (
	"strings"
	"testing"
)

func TestNewID_FormatAndLength(t *testing.T) {
	id, err := NewID(func(string) bool { return false })
	if err != nil {
		t.Fatalf("NewID: %v", err)
	}
	if len(id) != idLen {
		t.Errorf("len = %d, want %d", len(id), idLen)
	}
	for _, r := range id {
		if !strings.ContainsRune("0123456789abcdef", r) {
			t.Errorf("non-hex char %q in %q", r, id)
		}
	}
}

func TestNewID_AvoidsCollisions(t *testing.T) {
	taken := []string{"aaaaaaaa", "bbbbbbbb"}
	exists := func(id string) bool {
		for _, existing := range taken {
			if existing == id {
				return true
			}
		}
		return false
	}
	for i := 0; i < 50; i++ {
		id, err := NewID(exists)
		if err != nil {
			t.Fatalf("iter %d: NewID: %v", i, err)
		}
		for _, existing := range taken {
			if id == existing {
				t.Fatalf("iter %d: generated colliding id %q", i, id)
			}
		}
		taken = append(taken, id)
	}
}

func TestNewID_ErrorsAfterMaxRetries(t *testing.T) {
	// Always-collide callback forces the generator to exhaust retries.
	_, err := NewID(func(string) bool { return true })
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	if err != ErrIDCollision {
		t.Errorf("err = %v, want ErrIDCollision", err)
	}
}

func TestNewID_Distribution(t *testing.T) {
	// Sanity check: 1000 generated IDs should yield close to 1000 unique IDs.
	// (Duplicates are astronomically unlikely with 8 hex chars and 1000 IDs.)
	seen := make(map[string]struct{}, 1000)
	for i := 0; i < 1000; i++ {
		id, err := NewID(func(string) bool { return false })
		if err != nil {
			t.Fatalf("iter %d: %v", i, err)
		}
		seen[id] = struct{}{}
	}
	if len(seen) < 990 {
		t.Errorf("got %d unique IDs out of 1000; want >= 990", len(seen))
	}
}
