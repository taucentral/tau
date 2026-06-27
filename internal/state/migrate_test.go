package state

import (
	"errors"
	"strings"
	"testing"
)

// snapshotRegistry copies the global migration registry so tests can mutate
// it without leaking state to other tests. Tests in this package share the
// process-wide registry, and RegisterMigration panics on duplicate From
// versions, so every test that touches the registry must defer a restore.
func snapshotRegistry() map[int]Migration {
	migrationMu.RLock()
	defer migrationMu.RUnlock()
	out := make(map[int]Migration, len(migrationRegistry))
	for k, v := range migrationRegistry {
		out[k] = v
	}
	return out
}

func restoreRegistry(snap map[int]Migration) {
	migrationMu.Lock()
	defer migrationMu.Unlock()
	migrationRegistry = snap
}

// withFreshRegistry runs fn with an empty registry, restoring the previous
// state on return. Use this for tests that register migrations so they do
// not interfere with each other or with the production registry.
func withFreshRegistry(t *testing.T, fn func()) {
	t.Helper()
	snap := snapshotRegistry()
	migrationMu.Lock()
	migrationRegistry = map[int]Migration{}
	migrationMu.Unlock()
	defer restoreRegistry(snap)
	fn()
}

func TestRegisterMigration_AddsToRegistry(t *testing.T) {
	withFreshRegistry(t, func() {
		m := Migration{
			From: 0,
			To:   1,
			Apply: func([]Entry) ([]Entry, error) {
				return nil, nil
			},
		}
		RegisterMigration(m)
		got := registeredMigrations()
		if len(got) != 1 {
			t.Fatalf("len = %d, want 1", len(got))
		}
		if got[0].From != 0 || got[0].To != 1 {
			t.Errorf("got %+v, want From=0 To=1", got[0])
		}
	})
}

func TestRegisterMigration_PanicsOnDuplicate(t *testing.T) {
	withFreshRegistry(t, func() {
		m := Migration{From: 0, To: 1, Apply: func(e []Entry) ([]Entry, error) { return e, nil }}
		RegisterMigration(m)
		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("expected panic on duplicate registration, got none")
			}
			s, _ := r.(string)
			if !strings.Contains(s, "version 0") {
				t.Errorf("panic = %v, want mention of version 0", r)
			}
		}()
		RegisterMigration(Migration{From: 0, To: 2, Apply: func(e []Entry) ([]Entry, error) { return e, nil }})
	})
}

func TestMigrateSession_AlreadyAtCurrent_NoOp(t *testing.T) {
	withFreshRegistry(t, func() {
		entries := []Entry{{ID: "abc12345", Kind: KindSessionHeader}}
		out, ver, err := MigrateSession(CurrentSchemaVersion, entries)
		if err != nil {
			t.Fatalf("MigrateSession: %v", err)
		}
		if ver != CurrentSchemaVersion {
			t.Errorf("ver = %d, want %d", ver, CurrentSchemaVersion)
		}
		if len(out) != 1 || out[0].ID != "abc12345" {
			t.Errorf("entries mutated: %+v", out)
		}
	})
}

func TestMigrateSession_RunsForwardChain(t *testing.T) {
	withFreshRegistry(t, func() {
		// Build a 0→1 chain where Apply appends a marker entry so we can
		// observe that it ran exactly once.
		var calls []int
		RegisterMigration(Migration{
			From: 0,
			To:   1,
			Apply: func(entries []Entry) ([]Entry, error) {
				calls = append(calls, 0)
				return append(entries, Entry{ID: "marker00", Kind: KindLabel}), nil
			},
		})

		entries := []Entry{{ID: "root0000", Kind: KindSessionHeader}}
		out, ver, err := MigrateSession(0, entries)
		if err != nil {
			t.Fatalf("MigrateSession: %v", err)
		}
		if ver != 1 {
			t.Errorf("ver = %d, want 1", ver)
		}
		if len(calls) != 1 {
			t.Errorf("Apply calls = %d, want 1", len(calls))
		}
		if len(out) != 2 || out[1].ID != "marker00" {
			t.Errorf("out = %+v, want appended marker", out)
		}
	})
}

func TestMigrateSession_ErrorsWhenMigrationMissing(t *testing.T) {
	withFreshRegistry(t, func() {
		// No migrations registered but header claims version 0, so the
		// loop looks for "from 0" and finds nothing.
		_, ver, err := MigrateSession(0, nil)
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if !strings.Contains(err.Error(), "missing migration from version 0") {
			t.Errorf("err = %v, want 'missing migration from version 0'", err)
		}
		if ver != 0 {
			t.Errorf("ver = %d, want 0 (unchanged on error)", ver)
		}
	})
}

func TestMigrateSession_ErrorsWhenApplyFails(t *testing.T) {
	withFreshRegistry(t, func() {
		applyErr := errors.New("disk on fire")
		RegisterMigration(Migration{
			From:  0,
			To:    1,
			Apply: func([]Entry) ([]Entry, error) { return nil, applyErr },
		})
		_, ver, err := MigrateSession(0, nil)
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if !errors.Is(err, applyErr) {
			t.Errorf("err = %v, want wrap of %v", err, applyErr)
		}
		if !strings.Contains(err.Error(), "v0→v1") {
			t.Errorf("err = %v, want 'v0→v1' label", err)
		}
		if ver != 0 {
			t.Errorf("ver = %d, want 0 (unchanged on Apply error)", ver)
		}
	})
}

func TestMigrateSession_ErrSchemaDowngrade(t *testing.T) {
	withFreshRegistry(t, func() {
		// No migrations registered → latestRegisteredVersion == CurrentSchemaVersion == 1.
		// A session claiming version 5 must be refused.
		const have = 5
		_, ver, err := MigrateSession(have, nil)
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		var sd *ErrSchemaDowngrade
		if !errors.As(err, &sd) {
			t.Fatalf("err = %v, want *ErrSchemaDowngrade", err)
		}
		if sd.Have != have {
			t.Errorf("Have = %d, want %d", sd.Have, have)
		}
		if sd.Latest != CurrentSchemaVersion {
			t.Errorf("Latest = %d, want %d", sd.Latest, CurrentSchemaVersion)
		}
		if ver != have {
			t.Errorf("ver = %d, want %d (unchanged on downgrade)", ver, have)
		}
		if !strings.Contains(sd.Error(), "downgrade refused") {
			t.Errorf("Error() = %q, want 'downgrade refused'", sd.Error())
		}
	})
}

func TestMigrateSession_MultiHopChain(t *testing.T) {
	withFreshRegistry(t, func() {
		// Build a 0→1→2 chain. CurrentSchemaVersion is a const (1), so the
		// loop stops after reaching version 1 even if a 1→2 migration is
		// registered. We verify the stop boundary by observing Apply calls.
		var calls []int
		RegisterMigration(Migration{
			From: 0, To: 1,
			Apply: func(e []Entry) ([]Entry, error) { calls = append(calls, 0); return e, nil },
		})
		RegisterMigration(Migration{
			From: 1, To: 2,
			Apply: func(e []Entry) ([]Entry, error) { calls = append(calls, 1); return e, nil },
		})
		_, ver, err := MigrateSession(0, nil)
		if err != nil {
			t.Fatalf("MigrateSession: %v", err)
		}
		if ver != CurrentSchemaVersion {
			t.Errorf("ver = %d, want %d (loop stops at CurrentSchemaVersion)", ver, CurrentSchemaVersion)
		}
		// Only the 0→1 migration should have run because the loop halts at
		// CurrentSchemaVersion regardless of what else is registered.
		if len(calls) != 1 || calls[0] != 0 {
			t.Errorf("calls = %v, want [0]", calls)
		}
	})
}

func TestRegisteredMigrations_SortedByFrom(t *testing.T) {
	withFreshRegistry(t, func() {
		// Register out of order; reader must return sorted ascending.
		RegisterMigration(Migration{From: 2, To: 3, Apply: func(e []Entry) ([]Entry, error) { return e, nil }})
		RegisterMigration(Migration{From: 0, To: 1, Apply: func(e []Entry) ([]Entry, error) { return e, nil }})
		RegisterMigration(Migration{From: 1, To: 2, Apply: func(e []Entry) ([]Entry, error) { return e, nil }})
		got := registeredMigrations()
		if len(got) != 3 {
			t.Fatalf("len = %d, want 3", len(got))
		}
		for i := 0; i < len(got)-1; i++ {
			if got[i].From >= got[i+1].From {
				t.Errorf("index %d→%d not ascending: %d >= %d", i, i+1, got[i].From, got[i+1].From)
			}
		}
	})
}
