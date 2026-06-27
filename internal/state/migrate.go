package state

import (
	"fmt"
	"sort"
	"sync"
)

// Migration is a single forward migration step. Each migration upgrades a
// session from FromVersion to ToVersion by transforming the entry stream
// in place. Migrations run inside a single bbolt.Update transaction so a
// crash leaves the session on its old version, ready to retry on next open.
type Migration struct {
	From int
	To   int
	// Apply transforms the entry stream. It receives entries ordered by
	// timestamp ascending (root first) and returns the migrated entries,
	// which the manager rewrites in place.
	Apply func(entries []Entry) ([]Entry, error)
}

// migrationRegistry holds registered migrations keyed by From version.
var (
	migrationMu       sync.RWMutex
	migrationRegistry = map[int]Migration{}
)

// RegisterMigration adds a migration to the registry. RegisterMigration
// panics on duplicate From versions to make registration-order mistakes
// loud at init time.
func RegisterMigration(m Migration) {
	migrationMu.Lock()
	defer migrationMu.Unlock()
	if _, exists := migrationRegistry[m.From]; exists {
		panic(fmt.Sprintf("state: duplicate migration from version %d", m.From))
	}
	migrationRegistry[m.From] = m
}

// latestRegisteredVersion returns the highest To version across all
// registered migrations, or CurrentSchemaVersion if no migrations exist.
// Used to detect downgrade attempts (a session claiming a version newer
// than what this build knows how to handle).
func latestRegisteredVersion() int {
	migrationMu.RLock()
	defer migrationMu.RUnlock()
	highest := CurrentSchemaVersion
	for _, m := range migrationRegistry {
		if m.To > highest {
			highest = m.To
		}
	}
	return highest
}

// ErrSchemaDowngrade is returned by Open when a session's header Version
// exceeds the latest version this build can produce. The spec mandates
// "the system MUST NOT silently downgrade".
type ErrSchemaDowngrade struct {
	Have   int
	Latest int
}

func (e *ErrSchemaDowngrade) Error() string {
	return fmt.Sprintf("state: session version %d exceeds latest known version %d (downgrade refused)", e.Have, e.Latest)
}

// MigrateSession runs forward migrations on the given entries until they
// reach CurrentSchemaVersion, or returns ErrSchemaDowngrade when the
// session's current version is ahead of what this build knows.
//
// headerVersion is the SessionHeader payload's Version field. Returns the
// migrated entries and the new version (CurrentSchemaVersion when fully
// migrated).
func MigrateSession(headerVersion int, entries []Entry) ([]Entry, int, error) {
	if headerVersion > latestRegisteredVersion() {
		return nil, headerVersion, &ErrSchemaDowngrade{Have: headerVersion, Latest: latestRegisteredVersion()}
	}
	current := headerVersion
	for current < CurrentSchemaVersion {
		migrationMu.RLock()
		m, ok := migrationRegistry[current]
		migrationMu.RUnlock()
		if !ok {
			return nil, current, fmt.Errorf("state: missing migration from version %d", current)
		}
		out, err := m.Apply(entries)
		if err != nil {
			return nil, current, fmt.Errorf("state: migration v%d→v%d: %w", m.From, m.To, err)
		}
		entries = out
		current = m.To
	}
	return entries, current, nil
}

// registeredMigrations returns a sorted snapshot of the registry for
// inspection. Test-only.
func registeredMigrations() []Migration {
	migrationMu.RLock()
	defer migrationMu.RUnlock()
	out := make([]Migration, 0, len(migrationRegistry))
	for _, m := range migrationRegistry {
		out = append(out, m)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].From < out[j].From })
	return out
}
