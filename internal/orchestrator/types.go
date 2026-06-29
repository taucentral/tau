// types.go — canonical orchestration types owned by internal/orchestrator.
//
// The SDK aliases these via type aliases in pkg/tau/orchestration.go:
//
//   pkg/tau.OrchestrationSpec IS internal/orchestrator.OrchestrationSpec
//   pkg/tau.PhaseSpec        IS internal/orchestrator.PhaseSpec
//   pkg/tau.MergePolicy      IS internal/orchestrator.MergePolicy
//   pkg/tau.MergeSpec        IS internal/orchestrator.MergeSpec
//   pkg/tau.ConflictReport   IS internal/agent.ConflictReportShell
//
// Identity is preserved across the package boundary. errors.As works
// because ConflictReport IS ConflictReportShell.

package orchestrator

// OrchestrationSpec is the input to Orchestrator.Run.
type OrchestrationSpec struct {
	Phases         []PhaseSpec
	MergePolicy    MergePolicy
	ParentEventBus bool
}

// PhaseSpec describes one phase of an orchestration.
type PhaseSpec struct {
	// Name is the phase's stable identifier.
	Name string
	// Prompt is the user input passed to AgentSession.Run.
	Prompt string
	// Options carries the per-phase agent.SessionOptions. The SDK
	// wrapper converts the embedder-supplied tau.Options to the
	// internal SessionOptions shape before calling Run; the field
	// is typed as any so this package does not import internal/agent
	// for the SessionOptions struct (it does import agent for the
	// SessionOptions type via sequential.go's method signatures, but
	// the PhaseSpec.Options field stays loose to keep godoc readable).
	//
	// When nil/zero, the orchestrator inherits the parent's
	// Tools, Settings, and Middleware.
	Options any
	// DependsOn names phases that must complete before this phase.
	DependsOn []string
}

// MergePolicy selects how a child session's state-tree branch is
// reconciled with the parent's state during MergeState.
type MergePolicy int

const (
	MergePolicyNone MergePolicy = iota
	MergePolicyAppend
	MergePolicyReplay
)

// MergeSpec is the input to AgentSession.MergeState. The
// ConflictCallback field is an agent.ConflictReportShell callback so
// the runtime can invoke it without import cycles.
//
// Phase is copied into the ConflictReport on conflict so callers can
// attribute the conflict to the running phase. Populated by the
// orchestrator (SequentialOrchestrator passes the phase's Name).
type MergeSpec struct {
	Policy           MergePolicy
	Phase            string
	ConflictCallback func(*ConflictReport) error
}
