// sequential.go — SequentialOrchestrator, the reference implementation
// of the tau.Orchestrator interface.
//
// SequentialOrchestrator drives phases in dependency order. Phases with
// no inter-phase DependsOn edges within the same dependency level run
// CONCURRENTLY; levels themselves execute strictly in order so phase B
// (DependsOn A) starts only after phase A's level completes.
//
// Each phase spawns a child session via parent.Spawn, runs the phase
// prompt to completion, drains the child's events onto the multiplexed
// channel returned by Run, and (when MergePolicy is not None) calls
// parent.MergeState before the child is shut down. The child's Run
// error is surfaced via Err() after the channel closes.
//
// Cycle detection: Run topologically sorts phases by DependsOn; a cycle
// causes Run to return ErrOrchestrationAborted with the cycle named,
// before any phase starts.
//
// Cancellation: when the ctx passed to Run is cancelled, every in-flight
// child is signalled via child.Abort, Run waits for each child's Run
// goroutine to return, closes the event channel, and the stored error
// satisfies errors.Is(ErrOrchestrationAborted) wrapping ctx.Err().
//
// Reference: docs/input/context/plugin-support/whitepaper.md §3.3.

package orchestrator

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sync"

	"github.com/taucentral/tau/internal/agent"
)

// ConflictReport is a type alias for agent.ConflictReportShell so the
// MergeSpec ConflictCallback type signature matches across the package
// boundary. pkg/tau.ConflictReport IS this type.
type ConflictReport = agent.ConflictReportShell

// SequentialOrchestrator executes phases in dependency order, with
// independent phases within the same dependency level running
// concurrently. Construct via NewSequentialOrchestrator.
//
// The orchestrator stores the first phase error (non-cancel) in firstErr
// so callers can retrieve it via Err() after the Run channel closes.
type SequentialOrchestrator struct {
	parent *agent.AgentSession

	errMu    sync.Mutex
	firstErr error
}

// NewSequentialOrchestrator returns an orchestrator that drives phases
// against the supplied parent session. The parent MUST have been
// constructed with Options.Orchestrator set (otherwise parent.Spawn
// will return ErrNoOrchestrator at Run time).
//
// The returned orchestrator does NOT take ownership of the parent;
// the caller owns the parent's lifecycle and MUST Shutdown it
// explicitly.
func NewSequentialOrchestrator(parent *agent.AgentSession) *SequentialOrchestrator {
	return &SequentialOrchestrator{parent: parent}
}

// Run executes the phases in spec. See the type doc for semantics.
// Implements the tau.Orchestrator interface (Run + Err).
func (o *SequentialOrchestrator) Run(ctx context.Context, spec OrchestrationSpec) (<-chan agent.Event, error) {
	if o == nil || o.parent == nil {
		return nil, agent.ErrNoOrchestrator
	}

	// Group phases into dependency levels. Phases within a level run
	// concurrently; levels run strictly in order.
	levels, err := dependencyLevels(spec.Phases)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", agent.ErrOrchestrationAborted, err)
	}

	// Buffered channel sized to hold a reasonable burst per phase.
	out := make(chan agent.Event, 64)

	// Re-check ctx before starting; bail early if cancelled.
	if err := ctx.Err(); err != nil {
		close(out)
		return nil, fmt.Errorf("%w: %v", agent.ErrOrchestrationAborted, err)
	}

	go func() {
		defer close(out)
		o.runLevels(ctx, levels, spec, out)
	}()

	return out, nil
}

// Err returns the first non-nil phase error observed during Run, or nil
// when all phases completed cleanly. Context-cancellation errors are
// NOT stored here (they are surfaced via the Run return channel's close
// + ctx.Err); only "real" phase failures (spawn errors, run errors,
// merge errors) are stored.
//
// Callers SHOULD drain the Run channel to completion, then call Err.
// Err is safe to call concurrently with Run.
func (o *SequentialOrchestrator) Err() error {
	o.errMu.Lock()
	defer o.errMu.Unlock()
	return o.firstErr
}

// setErr stores the first non-nil error. Subsequent calls are no-ops.
func (o *SequentialOrchestrator) setErr(err error) {
	if err == nil {
		return
	}
	o.errMu.Lock()
	defer o.errMu.Unlock()
	if o.firstErr == nil {
		o.firstErr = err
	}
}

// runLevels executes each dependency level. Within a level, all phases
// run concurrently; between levels, strict ordering is enforced. On the
// first phase error or context cancellation, subsequent levels do NOT
// start and in-flight children are signalled via child.Abort.
func (o *SequentialOrchestrator) runLevels(ctx context.Context, levels [][]PhaseSpec, spec OrchestrationSpec, out chan<- agent.Event) {
	var wg sync.WaitGroup
	var inFlight sync.Map // key: *agent.AgentSession, value: struct{}

	// Derive a cancellable context so a phase error can abort remaining
	// phases in the current level AND prevent subsequent levels.
	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	// ctx-watcher: on runCtx.Done(), call Abort on every in-flight
	// child so their Run goroutines unwind promptly.
	watcherDone := make(chan struct{})
	go func() {
		<-runCtx.Done()
		inFlight.Range(func(key, _ any) bool {
			if ch, ok := key.(*agent.AgentSession); ok {
				ch.Abort("orchestration cancelled")
			}
			return true
		})
		close(watcherDone)
	}()

	for _, level := range levels {
		// Stop if ctx cancelled or a prior level errored.
		if runCtx.Err() != nil {
			break
		}
		if err := o.Err(); err != nil {
			break
		}

		for _, phase := range level {
			wg.Add(1)
			go func(phase PhaseSpec) {
				defer wg.Done()
				if err := o.runOnePhase(runCtx, phase, spec, out, &inFlight); err != nil {
					o.setErr(err)
					cancel() // abort remaining phases in this + subsequent levels
				}
			}(phase)
		}
		wg.Wait() // all phases in this level complete before next level
	}

	// Ensure the watcher has finished calling Abort on every remaining
	// in-flight child before we return; the caller's deferred close(out)
	// must not race with an in-flight forwarder writing to out.
	cancel()
	<-watcherDone
}

// runOnePhase spawns a child, runs it, optionally merges, and forwards
// the child's events to out. Returns an error to abort the orchestration.
//
// inFlight tracks the child for the ctx-watcher's Abort pass. The child
// is removed from inFlight when runOnePhase returns.
func (o *SequentialOrchestrator) runOnePhase(ctx context.Context, phase PhaseSpec, spec OrchestrationSpec, out chan<- agent.Event, inFlight *sync.Map) error {
	so := o.adaptPhaseOptions(phase.Options)

	child, err := o.parent.Spawn(ctx, so)
	if err != nil {
		return fmt.Errorf("phase %q: spawn: %w", phase.Name, err)
	}
	inFlight.Store(child, struct{}{})
	defer inFlight.Delete(child)

	// Subscribe to the child's events BEFORE Run starts so we don't
	// miss SessionStart.
	sub := child.Runtime().EventBus.Subscribe()

	// Run the child to completion in a goroutine. runErrCh is
	// buffered so the Run goroutine never blocks.
	runErrCh := make(chan error, 1)
	go func() {
		runErrCh <- child.Run(ctx, phase.Prompt)
	}()

	// Forward events as they arrive. The forwarder exits when either
	// sub closes (child shut down) or Run completes (runErrCh
	// receives). The run error value is captured in runErr for
	// surfacing via the returned error.
	var runErr error
	doneSig := make(chan struct{})
	go func() {
		defer close(doneSig)
		for {
			select {
			case evt, ok := <-sub:
				if !ok {
					return
				}
				select {
				case out <- evt:
				case <-ctx.Done():
				}
				// D4: when ParentEventBus is set, forward every
				// child event to the parent's event bus so the
				// parent's existing subscribers observe child
				// progress without ranging on the orchestrator
				// channel.
				if spec.ParentEventBus {
					o.parent.Runtime().EventBus.Publish(evt)
				}
			case err := <-runErrCh:
				runErr = err
				return
			}
		}
	}()

	<-doneSig

	// Surface the child's Run error. Per task 4.2, the Run error MUST
	// not be silently discarded. A non-nil, non-cancel error aborts
	// the orchestration.
	if runErr != nil &&
		!errors.Is(runErr, context.Canceled) &&
		!errors.Is(runErr, context.DeadlineExceeded) {
		_ = child.Shutdown(ctx)
		return fmt.Errorf("phase %q: run: %w", phase.Name, runErr)
	}

	// Merge if the spec asks for it. D8: pass phase.Name as the
	// MergeSpec.Phase so conflict reports attribute correctly.
	if spec.MergePolicy != MergePolicyNone {
		mergeSpec := agent.MergeSpecShell{
			Policy: agent.MergePolicyShell(spec.MergePolicy),
			Phase:  phase.Name,
		}
		if err := o.parent.MergeState(ctx, child, mergeSpec); err != nil {
			_ = child.Shutdown(ctx)
			return fmt.Errorf("phase %q: merge: %w", phase.Name, err)
		}
	}

	_ = child.Shutdown(ctx)
	return nil
}

// adaptPhaseOptions converts the any-typed Options payload on a
// PhaseSpec into the agent.SessionOptions shape. Each field is
// inherited individually from the parent when the phase's value is
// zero-valued:
//
//   - Tools: inherited when len==0.
//   - Settings: inherited when it equals the zero config.Settings.
//   - Middleware: inherited when it equals the zero MiddlewareSet.
//
// LLMClient and Model are inherited by Spawn (not here).
func (o *SequentialOrchestrator) adaptPhaseOptions(raw any) agent.SessionOptions {
	parentOpts := o.parent.Runtime().Options
	if so, ok := raw.(agent.SessionOptions); ok {
		if len(so.Tools) == 0 {
			so.Tools = parentOpts.BuiltinTools
		}
		if reflect.DeepEqual(so.Settings, parentOpts.Settings) {
			// Phase explicitly set Settings equal to the parent's;
			// nothing to inherit.
		} else if isZeroSettings(so.Settings) {
			so.Settings = parentOpts.Settings
		}
		if reflect.DeepEqual(so.Middleware, agent.MiddlewareSet{}) {
			so.Middleware = parentOpts.Middleware
		}
		return so
	}
	// No phase-supplied options: inherit everything.
	return agent.SessionOptions{
		Tools:      parentOpts.BuiltinTools,
		Settings:   parentOpts.Settings,
		Middleware: parentOpts.Middleware,
	}
}

// isZeroSettings reports whether s is the zero value of its type. Uses
// reflect so this file does not need to import config just for the
// zero-value literal.
func isZeroSettings(s any) bool {
	if s == nil {
		return true
	}
	return reflect.DeepEqual(s, reflect.Zero(reflect.TypeOf(s)).Interface())
}

// dependencyLevels groups phases into dependency levels for concurrent
// execution. Level 0 contains all phases with no DependsOn; level N
// contains phases whose dependencies are all in levels < N. Phases
// within a level may run concurrently; levels themselves are strictly
// ordered.
//
// A dependency cycle returns an error naming the phases in the cycle.
func dependencyLevels(phases []PhaseSpec) ([][]PhaseSpec, error) {
	if len(phases) == 0 {
		return nil, nil
	}
	// Build name → index. Reject duplicate names.
	idx := map[string]int{}
	for i, p := range phases {
		if _, dup := idx[p.Name]; dup {
			return nil, fmt.Errorf("duplicate phase name %q", p.Name)
		}
		idx[p.Name] = i
	}
	// Validate DependsOn references exist.
	for _, p := range phases {
		for _, dep := range p.DependsOn {
			if _, ok := idx[dep]; !ok {
				return nil, fmt.Errorf("phase %q depends on unknown phase %q", p.Name, dep)
			}
		}
	}
	// Kahn's algorithm, level-by-level. Each iteration picks ALL nodes
	// with indeg==0 as one level, then decrements their dependents.
	indeg := make([]int, len(phases))
	for _, p := range phases {
		indeg[idx[p.Name]] = len(p.DependsOn)
	}
	adj := make([][]int, len(phases)) // dep → dependents
	for _, p := range phases {
		for _, dep := range p.DependsOn {
			adj[idx[dep]] = append(adj[idx[dep]], idx[p.Name])
		}
	}
	var levels [][]PhaseSpec
	var current []int
	for i, d := range indeg {
		if d == 0 {
			current = append(current, i)
		}
	}
	for len(current) > 0 {
		var level []PhaseSpec
		var next []int
		for _, n := range current {
			level = append(level, phases[n])
			for _, dep := range adj[n] {
				indeg[dep]--
				if indeg[dep] == 0 {
					next = append(next, dep)
				}
			}
		}
		levels = append(levels, level)
		current = next
	}
	// Cycle detection: if not all phases placed, remaining have a cycle.
	placed := 0
	for _, lvl := range levels {
		placed += len(lvl)
	}
	if placed != len(phases) {
		var cycle []string
		for i, d := range indeg {
			if d > 0 {
				cycle = append(cycle, phases[i].Name)
			}
		}
		return nil, fmt.Errorf("dependency cycle among phases: %v", cycle)
	}
	return levels, nil
}
