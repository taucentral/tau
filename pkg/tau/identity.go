// identity.go — public SDK helper for attribute-based access control (ABAC).
//
// tau's middleware seam (see middleware.go) is the canonical hook for ABAC:
// an embedder-supplied ToolInterceptor reads identity from ctx in
// BeforeToolCall and gates each tool call. This helper establishes the
// canonical Identity shape and ctx-key convention so plugins compose —
// audit, governance, and policy interceptors all read the same type via
// IdentityFromContext.
//
// The runtime stays identity-agnostic: no internal/agent file inspects
// identity. Enforcement is the embedder's ToolInterceptor. The zero value
// of Identity represents an anonymous baseline; the interceptor decides
// whether anonymous is permitted (deny-by-default recommended in the SDK
// cookbook recipe at docs/sdk/cookbook.md).

package tau

import "context"

// identityKey is an unexported ctx-key type, per Go convention
// (https://pkg.go.dev/context#WithValue). Unexported struct types are
// globally unique: no other package can construct identityKey{}, so the
// key cannot collide with anything else a caller may place in ctx.
// WithIdentity and IdentityFromContext are the only entry points.
type identityKey struct{}

// Identity is the canonical caller-identity payload that an embedder
// threads through ctx for downstream ToolInterceptors to read. tau does
// NOT interpret any of these fields; embedders fill what their policy
// needs and read it back in their own interceptors. The shape is the
// smallest common baseline that supports the documented ABAC patterns
// (userID-based audit, role-based gating, tenant isolation) without
// fragmenting the convention.
//
// Treat an Identity placed in ctx as immutable. The Roles slice and
// Attributes map are reference types; mutating them after WithIdentity
// is visible to every reader of the ctx. To change identity mid-session,
// construct a new Identity and re-wrap the ctx.
//
// Use the Attributes map for anything beyond the typed fields:
// "clearance_level": "5", "caller_type": "ci_bot",
// "auth_method": "oidc", etc. Nested records are out of scope; encode
// them as JSON or wrap your own identity type if you need structure.
type Identity struct {
	UserID     string            // human-readable identifier; "" means anonymous
	Roles      []string          // role tags the policy layer consults
	Tenant     string            // multi-tenant isolation key
	Attributes map[string]string // free-form escape hatch
}

// WithIdentity returns a new ctx carrying id via context.WithValue under
// the unexported identityKey{}. The original ctx is unchanged. Use this
// at the boundary where the embedder has established identity (e.g. the
// HTTP handler that just verified the bearer token) before constructing
// or running a session.
//
// Per-session scope: call WithIdentity once before sess.Run.
// Per-turn scope (multi-tenant daemons): re-wrap ctx before each Run
// with the identity of the incoming request. Both work because the ctx
// keyed by this helper is what the runtime threads through every hook.
func WithIdentity(ctx context.Context, id Identity) context.Context {
	return context.WithValue(ctx, identityKey{}, id)
}

// IdentityFromContext returns the Identity carried by ctx, or the zero
// value (UserID == "", Roles == nil, Tenant == "", Attributes == nil)
// when ctx has no identity set.
//
// The zero value represents an **anonymous** baseline, not a policy
// decision. The interceptor decides whether anonymous is permitted;
// the SDK cookbook recipe documents deny-by-default as the recommended
// posture. Embedders who want permit-anonymous-by-default opt in
// explicitly in their ToolInterceptor.
//
// Anonymous identity is conservative data, not an enforcement posture:
// tau core never inspects the zero value, so a deployment can adopt
// whatever policy fits without fighting the runtime.
func IdentityFromContext(ctx context.Context) Identity {
	if v, ok := ctx.Value(identityKey{}).(Identity); ok {
		return v
	}
	return Identity{}
}
