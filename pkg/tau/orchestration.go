// orchestration.go — multi-session orchestration seam.
//
// tau today is single-session: one AgentSession per process invocation.
// The orchestration seam lets an embedder (or a parent session) spawn
// additional sessions, dispatch work to them, and merge their state
// trees back into the parent. Three reference patterns are anticipated
// (whitepaper §3.3):
//
//   - Sequential phase — one session per phase, output of phase N
//     becomes input of phase N+1 (the OpenSpec workflow).
//   - Parallel fan-out — parent decomposes work, spawns N children
//     for independent subtasks, fans in results.
//   - Adversarial review — primary produces, secondary critiques,
//     primary revises.
//
// This change ships the seam itself plus SequentialOrchestrator as the
// reference implementation. Parallel fan-out and adversarial review
// orchestrators are follow-on changes.
//
// Lifecycle contract:
//
//   - The runtime NEVER calls Shutdown on child sessions spawned via
//     AgentSession.Spawn. The parent (or the embedder) owns each
//     child's lifecycle. A parent shutting down SHALL NOT
//     automatically shut down its children; children may outlive the
//     parent (adversarial review patterns often have the parent
//     complete before the critic finishes).
//   - The runtime SHALL NOT call any lifecycle method on an
//     orchestrator supplied via Options.Orchestrator. The embedder
//     owns the orchestrator's lifecycle.
//
// Canonical type definitions live in internal/orchestrator/types.go;
// this file re-exports them as type aliases so the SDK's
// tau.OrchestrationSpec IS internal/orchestrator.OrchestrationSpec,
// tau.PhaseSpec IS internal/orchestrator.PhaseSpec, etc. Identity is
// preserved. The runtime therefore accepts SDK values without
// conversion.
//
// The Orchestrator interface itself is declared here because its Run
// method returns <-chan SessionEvent (which is <-chan agent.Event via
// alias); internal/orchestrator already imports agent, so concrete
// implementations in that package satisfy this interface via Go's
// structural typing.
//
// Reference: docs/input/context/plugin-support/whitepaper.md §3.3.

package tau

import (
	"context"

	"github.com/coevin/tau/internal/agent"
	"github.com/coevin/tau/internal/orchestrator"
)

// Orchestrator drives multi-session workflows in the agent's own
// process. Run executes the supplied spec to completion, surfacing
// events from every child session on the returned channel. The channel
// closes when all phases complete OR when ctx is cancelled; in the
// latter case the returned error satisfies errors.Is(err,
// ErrOrchestrationAborted) and wraps ctx.Err().
//
// Implementations live in internal/orchestrator (the reference
// SequentialOrchestrator) and are re-exported via
// NewSequentialOrchestrator. Embedders implementing their own
// orchestrator import only pkg/tau.
type Orchestrator interface {
	// Run executes the orchestration spec. The returned channel is
	// buffered; events from every phase's child session are
	// multiplexed in arrival order. The channel closes when all
	// phases complete (err == nil) or when ctx is cancelled (err
	// satisfies errors.Is(err, ErrOrchestrationAborted)).
	//
	// Run SHALL NOT be safe for concurrent invocation from multiple
	// goroutines on the same Orchestrator. Run MAY be called again
	// after a previous Run returns, provided the orchestrator's
	// internal state permits it; SequentialOrchestrator does not
	// reuse across Run calls.
	Run(ctx context.Context, spec OrchestrationSpec) (<-chan SessionEvent, error)

	// Err returns the first non-nil phase error observed during the
	// most recent Run, or nil when all phases completed cleanly.
	// Context-cancellation errors are NOT returned here (they are
	// implied by the channel closing); only "real" phase failures
	// (spawn errors, child Run errors, merge errors) are surfaced.
	//
	// Callers SHOULD drain the Run channel to completion, then call
	// Err. Err is safe to call concurrently with Run.
	Err() error
}

// OrchestrationSpec is the input to Orchestrator.Run. It names the
// phases (for sequential orchestration) or sub-tasks (for fan-out),
// their inputs, model/tool budgets, and the merge policy applied
// after each phase completes.
//
// Alias of internal/orchestrator.OrchestrationSpec.
type OrchestrationSpec = orchestrator.OrchestrationSpec

// PhaseSpec describes one phase of an orchestration. A phase maps to
// exactly one Spawn + one Run + (optionally) one MergeState call.
//
// Alias of internal/orchestrator.PhaseSpec.
type PhaseSpec = orchestrator.PhaseSpec

// MergePolicy selects how a child session's state-tree branch is
// reconciled with the parent's state during MergeState.
//
// Alias of internal/orchestrator.MergePolicy.
type MergePolicy = orchestrator.MergePolicy

const (
	// MergePolicyNone discards the child's state after the phase
	// completes. Use for read-only inspection phases (audit,
	// summarize) whose state should not affect the parent.
	MergePolicyNone MergePolicy = orchestrator.MergePolicyNone

	// MergePolicyAppend adds the child's state-tree branch as a new
	// leaf on the parent's state tree. Conflict detection does NOT
	// run; the child's writes are accepted verbatim. Use for
	// independent subtasks whose results live alongside the parent's
	// history (fan-out merges).
	MergePolicyAppend MergePolicy = orchestrator.MergePolicyAppend

	// MergePolicyReplay replays the child's tool calls against the
	// parent's current state tree. When a replayed tool call would
	// conflict with the parent's state (e.g. both edited the same
	// file), MergeState returns ErrOrchestrationConflict wrapping a
	// *ConflictReport accessible via errors.As. Use for sequential
	// workflows where later phases depend on earlier ones' mutations.
	MergePolicyReplay MergePolicy = orchestrator.MergePolicyReplay
)

// MergeSpec is the input to AgentSession.MergeState. It selects a
// MergePolicy and an optional conflict callback invoked when
// MergePolicyReplay detects a conflict before the typed error is
// returned.
//
// Alias of internal/orchestrator.MergeSpec.
type MergeSpec = orchestrator.MergeSpec

// ConflictReport carries structured information about a replay-merge
// conflict. MergeState returns it wrapped in ErrOrchestrationConflict;
// callers recover it via errors.As(err, &report) where report is a
// *ConflictReport.
//
// Alias of internal/agent.ConflictReportShell.
type ConflictReport = agent.ConflictReportShell

// ErrOrchestrationConflict is returned by MergeState when
// MergePolicyReplay detects a conflict between the child's tool calls
// and the parent's state. The error wraps a *ConflictReport accessible
// via errors.As.
//
// This is a var re-export of internal/agent.ErrOrchestrationConflict;
// identity is preserved across the package boundary.
var ErrOrchestrationConflict = agent.ErrOrchestrationConflict

// ErrOrchestrationAborted is returned by Orchestrator.Run when the
// context is cancelled before all phases complete, OR when the spec
// contains a dependency cycle. In the cancellation case the error
// wraps ctx.Err(); in the cycle case it is unadorned. Callers
// distinguish the two by checking errors.Is(err, context.Canceled)
// or errors.Is(err, context.DeadlineExceeded).
//
// This is a var re-export of internal/agent.ErrOrchestrationAborted.
var ErrOrchestrationAborted = agent.ErrOrchestrationAborted

// ErrOrchestratorClosed is returned by MergeState when the child
// session's state-tree branch has been deallocated (e.g. the child's
// state manager was closed AND its backing file removed). The
// reference state manager preserves branches across Close, so this
// sentinel is reserved for embedders building ephemeral backends.
//
// This is a var re-export of internal/agent.ErrOrchestratorClosed.
var ErrOrchestratorClosed = agent.ErrOrchestratorClosed

// ErrNoOrchestrator is returned by Spawn and MergeState when the
// parent session was constructed without an orchestrator configured.
// It is distinct from ErrRuntimeShutdown (which signals terminal
// state) so callers can distinguish "orchestration not configured"
// from "session is shutting down". Contract tests accept either
// ErrNoOrchestrator OR ErrRuntimeShutdown per the spec's "OR a typed
// equivalent" language.
//
// This is a var re-export of internal/agent.ErrNoOrchestrator.
var ErrNoOrchestrator = agent.ErrNoOrchestrator
