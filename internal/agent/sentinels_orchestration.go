// sentinels_orchestration.go — orchestration-related sentinels owned
// by internal/agent so Spawn/MergeState can return them without
// importing pkg/tau.
//
// pkg/tau aliases these via var re-exports:
//
//   pkg/tau.ErrOrchestrationConflict  = agent.ErrOrchestrationConflict
//   pkg/tau.ErrOrchestrationAborted   = agent.ErrOrchestrationAborted
//   pkg/tau.ErrOrchestratorClosed     = agent.ErrOrchestratorClosed
//   pkg/tau.ErrNoOrchestrator         = agent.ErrNoOrchestrator
//
// Identity is preserved: errors.Is(err, tau.ErrOrchestrationConflict)
// works across the package boundary because the two names refer to the
// same variable.

package agent

import "errors"

// ErrOrchestrationConflict is returned by MergeState when
// MergePolicyReplay detects a conflict between the child's tool calls
// and the parent's state. The error wraps a *ConflictReportShell
// accessible via errors.As.
var ErrOrchestrationConflict = errors.New("agent: orchestration merge conflict")

// ErrOrchestrationAborted is returned by Orchestrator.Run on context
// cancellation or dependency cycle. The orchestrator implementations
// in internal/orchestrator return this directly; pkg/tau re-exports
// it so embedders can errors.Is against it.
var ErrOrchestrationAborted = errors.New("agent: orchestration aborted")

// ErrOrchestratorClosed is returned by MergeState when the child's
// state-tree branch has been deallocated. The reference state manager
// preserves branches across Close, so this sentinel rarely fires in
// practice; it exists for embedders building ephemeral backends.
var ErrOrchestratorClosed = errors.New("agent: orchestrator or state closed")
