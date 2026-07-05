package tau

import (
	"context"
	"reflect"
	"sync"
	"testing"
	"time"
)

// TestIdentity_RoundTrip verifies the happy-path set/get pair: every
// field placed via WithIdentity is recovered verbatim from
// IdentityFromContext, including reference-typed fields.
func TestIdentity_RoundTrip(t *testing.T) {
	want := Identity{
		UserID:     "alice",
		Roles:      []string{"admin", "ops"},
		Tenant:     "acme",
		Attributes: map[string]string{"clearance_level": "5", "caller_type": "human"},
	}
	ctx := WithIdentity(context.Background(), want)
	got := IdentityFromContext(ctx)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("round-trip mismatch:\n got = %#v\nwant = %#v", got, want)
	}
}

// TestIdentity_BareContextReturnsZero verifies that IdentityFromContext
// on a ctx that never carried identity returns the zero value without
// panicking.
func TestIdentity_BareContextReturnsZero(t *testing.T) {
	got := IdentityFromContext(context.Background())
	if !reflect.DeepEqual(got, Identity{}) {
		t.Fatalf("bare ctx: got %#v, want zero Identity{}", got)
	}
}

// TestIdentity_ZeroValueIsAnonymousBaseline is the sub-test explicitly
// documenting the "anonymous baseline" semantic from design.md D1.7.
// The zero value compares equal to a freshly-constructed Identity{},
// which is the value an interceptor reads when no WithIdentity has
// wrapped the ctx. Whether anonymous is permitted is the embedder's
// policy decision (deny-by-default recommended in the cookbook).
func TestIdentity_ZeroValueIsAnonymousBaseline(t *testing.T) {
	t.Run("zero equals freshly constructed", func(t *testing.T) {
		var a Identity
		b := Identity{}
		if !reflect.DeepEqual(a, b) {
			t.Fatalf("zero mismatch: a = %#v, b = %#v", a, b)
		}
	})
	t.Run("zero fields are empty", func(t *testing.T) {
		z := Identity{}
		if z.UserID != "" || z.Tenant != "" || z.Roles != nil || z.Attributes != nil {
			t.Fatalf("zero Identity has non-empty fields: %#v", z)
		}
	})
}

// TestIdentity_UnrelatedContextKeyNoCollision verifies that an unrelated
// context.WithValue key does not produce a false-positive identity. The
// unexported identityKey struct type guarantees uniqueness by package
// isolation.
func TestIdentity_UnrelatedContextKeyNoCollision(t *testing.T) {
	ctx := context.WithValue(context.Background(), "someOtherKey", 42)
	ctx = context.WithValue(ctx, myCustomKey{}, "still not identity")
	got := IdentityFromContext(ctx)
	if !reflect.DeepEqual(got, Identity{}) {
		t.Fatalf("colliding keys: got non-zero identity %#v", got)
	}
}

type myCustomKey struct{}

// TestIdentity_SurvivesNestedContextWraps verifies that identity set on
// an outer ctx survives both context.WithCancel and context.WithTimeout
// (and by extension any other standard derivation).
func TestIdentity_SurvivesNestedContextWraps(t *testing.T) {
	want := Identity{UserID: "carol", Roles: []string{"viewer"}}
	root := WithIdentity(context.Background(), want)

	t.Run("WithCancel preserves identity", func(t *testing.T) {
		inner, cancel := context.WithCancel(root)
		defer cancel()
		if got := IdentityFromContext(inner); !reflect.DeepEqual(got, want) {
			t.Fatalf("WithCancel dropped identity: got %#v, want %#v", got, want)
		}
	})

	t.Run("WithTimeout preserves identity", func(t *testing.T) {
		inner, cancel := context.WithTimeout(root, time.Minute)
		defer cancel()
		if got := IdentityFromContext(inner); !reflect.DeepEqual(got, want) {
			t.Fatalf("WithTimeout dropped identity: got %#v, want %#v", got, want)
		}
	})
}

// TestIdentity_ConcurrentReadsNoRace verifies that IdentityFromContext
// is safe to call from many goroutines against the same ctx. Run under
// `-race` to catch any data race in the accessor.
func TestIdentity_ConcurrentReadsNoRace(t *testing.T) {
	want := Identity{UserID: "concurrent", Roles: []string{"load"}}
	ctx := WithIdentity(context.Background(), want)

	const n = 64
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			if got := IdentityFromContext(ctx); !reflect.DeepEqual(got, want) {
				t.Errorf("concurrent read mismatch: got %#v, want %#v", got, want)
			}
		}()
	}
	wg.Wait()
}
