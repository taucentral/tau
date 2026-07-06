// spawn.go — Spawn and MergeState methods on *AgentSession.
//
// These methods implement the in-process orchestration seam defined by
// pkg/tau/orchestration.go. The runtime does NOT auto-invoke them;
// embedders (or orchestrator implementations in internal/orchestrator)
// drive Spawn / MergeState from outside the agent loop.
//
// Lifecycle contract:
//
//   - Children spawned via Spawn are NOT shut down by the parent's
//     Shutdown. The embedder owns each child's lifecycle.
//   - The orchestrator injected via SessionOptions.Orchestrator is
//     likewise NEVER invoked by the runtime; the embedder drives it.
//
// State-tree semantics (shared-store branching):
//
//   - Each child receives a *state.branchManager that shares the
//     parent's read-side store and maintains a private shadow buffer
//     for writes. The child's reads (Tree, BuildContext) see the
//     parent's entries plus its own shadow; its writes go to the shadow
//     only. The parent's leaf pointer is untouched by child writes.
//   - MergeState integrates the shadow into the parent via AppendAt
//     (leaf-preserving append), then advances the parent's leaf.
//     MergePolicyNone discards the shadow — zero orphan entries because
//     the shadow never touched the parent's store.
//   - For non-branch children (test fakes using inMemoryManager
//     directly), MergeState falls back to orderedEntries + AppendAt.
//
// Reference: docs/input/context/plugin-support/whitepaper.md §3.3.

package agent

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/taucentral/tau/internal/state"
)

// ErrNoOrchestrator is returned by Spawn and MergeState when the
// parent session was constructed without an orchestrator configured.
// It is distinct from ErrRuntimeShutdown (which signals terminal
// state) so callers can distinguish "orchestration not configured"
// from "session is shutting down".
//
// tau deviates from the spec's suggestion of using ErrRuntimeShutdown
// for the no-orchestrator case: conflating the two makes embedder
// error handling ambiguous. A typed sentinel per failure mode is more
// honest. pkg/tau re-exports this as tau.ErrNoOrchestrator and the
// SDK's contract test accepts either ErrNoOrchestrator OR
// ErrRuntimeShutdown to honour the spec's "OR a typed equivalent"
// language.
var ErrNoOrchestrator = errors.New("agent: orchestrator not configured")

// spawnChildMu guards the per-parent spawn counter map. The map is
// keyed by *AgentSession pointer; entries are never cleaned up because
// parent session pointers are expected to be long-lived and few.
var (
	spawnChildMu      sync.Mutex
	spawnChildCounter = map[*AgentSession]int{}
)

// Spawn constructs a child AgentSession with a state-tree branch on the
// parent's tree and (optionally) inherited LLMClient/Model. The child is
// a fully functional *AgentSession; its Run, Shutdown, Subscribe, etc.
// methods behave identically to a top-level session.
//
// Spawn returns ErrNoOrchestrator when the parent has no orchestrator
// configured (Options.Orchestrator is nil), and ErrRuntimeShutdown
// when the parent's Shutdown has completed.
//
// State model: the child receives a *state.branchManager rooted at the
// parent's current leaf. The child's reads see the parent's history
// (root → branchAt) plus its own shadow writes. Writes by the child do
// NOT appear in the parent's view until MergeState is called.
func (s *AgentSession) Spawn(ctx context.Context, opts SessionOptions) (*AgentSession, error) {
	if s.rt == nil {
		return nil, ErrRuntimeShutdown
	}
	// Check terminal state first — a shutdown parent cannot spawn.
	s.rt.shutdownMu.Lock()
	shutdownDone := s.rt.shutdownDone
	s.rt.shutdownMu.Unlock()
	if shutdownDone {
		return nil, ErrRuntimeShutdown
	}

	// Orchestrator gate. Without one, the session is single-session.
	if s.rt.Options.Orchestrator == nil {
		return nil, ErrNoOrchestrator
	}

	// Inherit parent's LLMClient / Model when the caller did not
	// override them. Tools and Settings are required on opts.
	if opts.LLMClient == nil {
		opts.LLMClient = s.rt.Options.LLMClient
	}
	if opts.Model == "" {
		opts.Model = s.rt.Options.Model
	}

	// Each child gets a branch manager sharing the parent's store.
	// The branch's initial leaf is the parent's current leaf; the
	// child's reads see the parent's root → leaf path plus its own
	// shadow, and its writes go to the shadow only.
	opts.StateManager = state.NewBranchManager(s.rt.State, s.rt.Cwd)

	// Mint a unique session id for the child.
	spawnChildMu.Lock()
	spawnChildCounter[s]++
	childIdx := spawnChildCounter[s]
	spawnChildMu.Unlock()
	if opts.SessionID == "" {
		opts.SessionID = fmt.Sprintf("%s-child-%d-%d", s.rt.SessionID, childIdx, time.Now().UnixNano())
	}

	childRT, err := CreateAgentSessionRuntime(ctx, s.rt.Cwd, opts)
	if err != nil {
		return nil, err
	}
	return NewAgentSession(childRT), nil
}

// MergeState reconciles a child session's state-tree branch into the
// parent according to spec.Policy.
//
// Returns:
//   - nil on success.
//   - ErrNoOrchestrator — parent has no orchestrator configured.
//   - ErrOrchestratorClosed — child's state-tree pointer is nil or
//     the child's state manager has been closed.
//   - ErrOrchestrationConflict (MergePolicyReplay only) — a replayed
//     file-mutating tool call conflicts with the parent's current
//     state. The wrapped *ConflictReportShell carries Phase, File, and
//     LineRange; access via errors.As.
//
// MergePolicyNoneShell discards the child's shadow (no parent mutation,
// zero orphans). MergePolicyAppendShell threads the child's shadow onto
// the parent's tree via AppendAt, then advances the parent's leaf.
// MergePolicyReplayShell replays file-mutating tool calls against the
// parent's current write-set and reports the first conflict via
// ErrOrchestrationConflict with a populated *ConflictReportShell.
//
// On conflict the parent's state is unchanged (preceding replayed
// entries are NOT rolled back; this is documented in the SDK doc —
// callers who need atomicity should snapshot before calling).
func (s *AgentSession) MergeState(_ context.Context, child *AgentSession, spec MergeSpecShell) error {
	if s.rt == nil {
		return ErrRuntimeShutdown
	}
	if s.rt.Options.Orchestrator == nil {
		return ErrNoOrchestrator
	}
	if child == nil || child.rt == nil || child.rt.State == nil {
		return ErrOrchestratorClosed
	}
	// Detect closed/deallocated child state. Per the orchestration
	// contract, MergeState on a closed child returns
	// ErrOrchestratorClosed so callers can distinguish "child done"
	// from "conflict".
	if state.IsClosed(child.rt.State) {
		return ErrOrchestratorClosed
	}

	switch spec.Policy {
	case MergePolicyNoneShell:
		// Discard child's shadow / state. No parent mutation.
		return nil

	case MergePolicyAppendShell:
		return s.mergeAppend(child)

	case MergePolicyReplayShell:
		return s.mergeReplay(child, spec)

	default:
		return fmt.Errorf("agent.MergeState: unknown policy %d", spec.Policy)
	}
}

// mergeAppend threads the child's entries onto the parent's tree via
// AppendAt (leaf-preserving), then advances the parent's leaf to the
// final appended ID. Conflict detection does NOT run.
func (s *AgentSession) mergeAppend(child *AgentSession) error {
	entries, err := childEntriesForMerge(child.rt.State)
	if err != nil {
		return fmt.Errorf("agent.MergeState: load child entries: %w", err)
	}
	if len(entries) == 0 {
		return nil
	}
	parentLeaf := s.rt.State.LeafID()
	for _, e := range entries {
		id, err := s.rt.State.AppendAt(
			state.Entry{Kind: e.Kind, Payload: e.Payload},
			parentLeaf,
		)
		if err != nil {
			return fmt.Errorf("agent.MergeState: append entry: %w", err)
		}
		parentLeaf = id
	}
	// Advance the parent's leaf to the final appended entry.
	if err := s.rt.State.SetLeaf(parentLeaf); err != nil {
		return fmt.Errorf("agent.MergeState: advance leaf: %w", err)
	}
	return nil
}

// mergeReplay replays the child's write-tool-calls against the parent's
// current write-set. The FIRST conflicting file-mutating tool call
// produces an ErrOrchestrationConflict wrapping a *ConflictReportShell
// populated with Phase (from spec), File, and LineRange. Non-conflicting
// entries are applied via AppendAt.
func (s *AgentSession) mergeReplay(child *AgentSession, spec MergeSpecShell) error {
	entries, err := childEntriesForMerge(child.rt.State)
	if err != nil {
		return fmt.Errorf("agent.MergeState: load child entries: %w", err)
	}
	parentFiles := parentWriteFiles(s.rt.State)
	cwd := s.rt.Cwd

	parentLeaf := s.rt.State.LeafID()
	applied := false
	for _, e := range entries {
		// Check this entry's write ops for conflicts against the
		// parent's current write-set.
		for _, op := range childWriteOps([]state.Entry{e}) {
			report := detectConflict(op, parentFiles, spec.Phase, cwd)
			if report == nil {
				continue
			}
			// Conflict: offer the callback a chance to resolve.
			if spec.ConflictCallback != nil {
				cbErr := spec.ConflictCallback(report)
				if cbErr == nil {
					// Callback resolved: break inner loop, apply entry.
					break
				}
				if !errors.Is(cbErr, errAbortMerge) {
					return cbErr
				}
			}
			return fmt.Errorf("%w: %w", ErrOrchestrationConflict, report)
		}
		// No conflict (or callback resolved): apply.
		id, err := s.rt.State.AppendAt(
			state.Entry{Kind: e.Kind, Payload: e.Payload},
			parentLeaf,
		)
		if err != nil {
			return fmt.Errorf("agent.MergeState: append entry: %w", err)
		}
		parentLeaf = id
		applied = true
	}
	if applied {
		if err := s.rt.State.SetLeaf(parentLeaf); err != nil {
			return fmt.Errorf("agent.MergeState: advance leaf: %w", err)
		}
	}
	return nil
}

// childEntriesForMerge returns the child's own entries for MergeState
// integration. For branch managers (the common case after Spawn), this
// is the shadow buffer. For legacy in-memory managers (test fakes), it
// falls back to the root→leaf path excluding the SessionHeader root.
func childEntriesForMerge(mgr state.Manager) ([]state.Entry, error) {
	if shadow, ok := state.BranchShadow(mgr); ok {
		return shadow, nil
	}
	return orderedEntries(mgr)
}

// errAbortMerge is the sentinel returned by ConflictCallback to signal
// "abort the merge with ErrOrchestrationConflict". It is internal; the
// public surface is the callback's error return.
var errAbortMerge = errors.New("agent: merge aborted by conflict callback")

// MergeSpecShell is the internal/agent view of pkg/tau.MergeSpec. The
// SDK adapts the public struct to this shape before calling MergeState.
type MergeSpecShell struct {
	Policy           MergePolicyShell
	Phase            string
	ConflictCallback func(*ConflictReportShell) error
}

// MergePolicyShell is the internal/agent view of pkg/tau.MergePolicy.
// Order matches the public iota so the SDK can convert by literal
// value.
type MergePolicyShell int

const (
	// MergePolicyNoneShell discards child state after the phase completes.
	MergePolicyNoneShell MergePolicyShell = iota
	// MergePolicyAppendShell adds child state as a new branch on the parent.
	MergePolicyAppendShell
	// MergePolicyReplayShell replays child tool calls against the parent.
	MergePolicyReplayShell
)

// ConflictReportShell is the internal/agent view of pkg/tau.ConflictReport.
// The SDK type ConflictReport IS this type (type alias) so errors.As
// works across the SDK ↔ runtime boundary.
type ConflictReportShell struct {
	// Phase names the phase whose replayed tool call conflicted with
	// the parent's state. Populated from MergeSpecShell.Phase (set by
	// the orchestrator to the running phase's Name). Empty when the
	// conflict was not produced by an orchestrator-driven merge.
	Phase string

	// File is the path of the conflicting file (as it appears in the
	// child's tool call input, before cwd resolution). Empty for
	// non-file conflicts.
	File string

	// LineRange is the conflicting line range in File, [first, last]
	// inclusive, 1-indexed. For edit calls, derived from old_string's
	// first occurrence in the parent's current on-disk file. For write
	// calls, [1, newline_count(content)+1]. [0,0] when the range
	// cannot be determined (e.g. patch, or old_string not found).
	LineRange [2]int
}

// Error implements the error interface so ConflictReportShell can be
// chained via fmt.Errorf("%w: %w", sentinel, report) and recovered by
// callers via errors.As.
func (c *ConflictReportShell) Error() string {
	if c == nil {
		return "agent: orchestration conflict"
	}
	return fmt.Sprintf("agent: orchestration conflict in phase %q on %s lines %d-%d",
		c.Phase, c.File, c.LineRange[0], c.LineRange[1])
}

// orderedEntries returns the child's tree entries in root→leaf order.
// Returns an empty slice for an empty tree. The root SessionHeader
// entry is EXCLUDED — appending it to the parent would corrupt the
// parent's tree (which already has its own root). Used as the fallback
// for non-branch children (test fakes using inMemoryManager directly).
func orderedEntries(mgr state.Manager) ([]state.Entry, error) {
	tree, err := mgr.Tree()
	if err != nil {
		// Empty tree (no entries) returns ErrTreeInvalid; treat as
		// "nothing to merge".
		return nil, nil
	}
	leaf := mgr.LeafID()
	if leaf == "" {
		// Defensive: manager with no leaf has nothing to merge.
		return nil, nil
	}
	walk, err := tree.Path(leaf)
	if err != nil {
		return nil, err
	}
	// Drop the root (SessionHeader) — the parent already has its own.
	if len(walk) > 0 {
		walk = walk[1:]
	}
	return walk, nil
}
