// contract_test.go — public SDK contract test.
//
// This file is the copyable contract that any consumer of pkg/tau can
// copy into their own package to verify the SDK surface they depend on
// has not regressed. It exercises the canonical happy-path and
// negative-path scenarios documented in specs/sdk-public-api/spec.md.
//
// Precedent: hashicorp/go-plugin ships a similar contract in
// `plugin/contract_test.go` so downstream consumers can pin the API
// surface they target. The pattern is: copy the file into your test
// suite, swap the faux fixtures (fauxprovider, in-memory state, dummy
// HeadlessTool) for your real implementations, and run.
//
// Each section below (Construction, Lifecycle, Events, Tools, State,
// Errors, Provider Registry, Slash Commands) covers one requirement
// group from the spec. Skim the section headers to find the
// substitution point for a given capability.

package tau

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/invopop/jsonschema"

	"github.com/coevin/tau/internal/config"
	"github.com/coevin/tau/internal/fauxprovider"
	"github.com/coevin/tau/internal/slash"
	"github.com/coevin/tau/internal/tools"
)

// contractOpts returns an Options bundle populated with every field set
// to a working value. Contract tests that exercise a specific failure
// mode should clone this and zero the field under test.
func contractOpts(t *testing.T) Options {
	t.Helper()
	return Options{
		Cwd:           t.TempDir(),
		Model:         "contract-test-model",
		LLMClient:     fauxprovider.NewWithResponse("ok"),
		Tools:         []HeadlessTool{tools.NewReadTool(tools.OSReadOperations{})},
		Settings:      config.DefaultSettings(),
		ContextWindow: 200000,
		StateManager:  NewInMemoryManager(t.TempDir()),
	}
}

// --- Construction ----------------------------------------------------------

func TestContractConstructionWithEveryField(t *testing.T) {
	opts := contractOpts(t)
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession with full Options: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	if got := sess.Model(); got != opts.Model {
		t.Errorf("Model() = %q, want %q", got, opts.Model)
	}
	if got := sess.Cwd(); got != opts.Cwd {
		t.Errorf("Cwd() = %q, want %q", got, opts.Cwd)
	}
	if sess.CreatedAt().IsZero() {
		t.Error("CreatedAt() returned zero time")
	}
}

func TestContractConstructionRejectsMissingModel(t *testing.T) {
	opts := contractOpts(t)
	opts.Model = ""
	_, err := CreateAgentSession(context.Background(), opts)
	if err == nil {
		t.Error("CreateAgentSession with empty Model: got nil error, want one")
	}
}

func TestContractConstructionRejectsMissingLLMClient(t *testing.T) {
	opts := contractOpts(t)
	opts.LLMClient = nil
	_, err := CreateAgentSession(context.Background(), opts)
	if err == nil {
		t.Error("CreateAgentSession with nil LLMClient: got nil error, want one")
	}
}

func TestContractConstructionRejectsMissingTools(t *testing.T) {
	opts := contractOpts(t)
	opts.Tools = nil
	_, err := CreateAgentSession(context.Background(), opts)
	if err == nil {
		t.Error("CreateAgentSession with nil Tools: got nil error, want one")
	}
}

// --- Lifecycle -------------------------------------------------------------

func TestContractRunSuccessPath(t *testing.T) {
	opts := contractOpts(t)
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := sess.Run(ctx, "hello"); err != nil {
		t.Errorf("Run on a happy-path context: %v", err)
	}
}

func TestContractRunOnCanceledContextReturnsCtxErr(t *testing.T) {
	opts := contractOpts(t)
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel before Run
	if err := sess.Run(ctx, "hello"); err == nil {
		t.Error("Run on canceled context: got nil error, want ctx.Err()")
	}
}

func TestContractRunAfterShutdownReturnsErrRuntimeShutdown(t *testing.T) {
	opts := contractOpts(t)
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	if err := sess.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
	err = sess.Run(context.Background(), "hello")
	if !errors.Is(err, ErrRuntimeShutdown) {
		t.Errorf("Run after Shutdown: got %v, want ErrRuntimeShutdown", err)
	}
}

func TestContractShutdownIsIdempotent(t *testing.T) {
	opts := contractOpts(t)
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	if err := sess.Shutdown(context.Background()); err != nil {
		t.Fatalf("first Shutdown: %v", err)
	}
	if err := sess.Shutdown(context.Background()); err != nil {
		t.Errorf("second Shutdown: %v", err)
	}
	// A third call should still be a no-op.
	if err := sess.Shutdown(context.Background()); err != nil {
		t.Errorf("third Shutdown: %v", err)
	}
}

// --- Events ----------------------------------------------------------------

func TestContractEventsSubscribeAllTopics(t *testing.T) {
	opts := contractOpts(t)
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	ch := sess.Subscribe()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := sess.Run(ctx, "hello"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Drain events; we expect at least the lifecycle topics.
	var seen []string
drain:
	for {
		select {
		case evt, ok := <-ch:
			if !ok {
				break drain
			}
			seen = append(seen, string(evt.Topic()))
		case <-time.After(2 * time.Second):
			break drain
		}
	}
	wantTopics := []Topic{
		TopicSessionStart, TopicTurnStart, TopicMessageStart,
		TopicMessageUpdate, TopicMessageEnd, TopicTurnEnd,
	}
	for _, want := range wantTopics {
		if !contains(seen, string(want)) {
			t.Errorf("Subscribe() missed topic %q; saw %v", want, seen)
		}
	}
}

func TestContractEventsSubscribeTopicsFiltered(t *testing.T) {
	opts := contractOpts(t)
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	ch := sess.SubscribeTopics(TopicTurnStart, TopicTurnEnd)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := sess.Run(ctx, "hello"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	var seen []string
drain:
	for {
		select {
		case evt, ok := <-ch:
			if !ok {
				break drain
			}
			seen = append(seen, string(evt.Topic()))
		case <-time.After(2 * time.Second):
			break drain
		}
	}
	for _, got := range seen {
		if got != string(TopicTurnStart) && got != string(TopicTurnEnd) {
			t.Errorf("filtered Subscribe got %q, want only turn_start/turn_end", got)
		}
	}
	if !contains(seen, string(TopicTurnStart)) {
		t.Errorf("filtered Subscribe missed turn_start; got %v", seen)
	}
	if !contains(seen, string(TopicTurnEnd)) {
		t.Errorf("filtered Subscribe missed turn_end; got %v", seen)
	}
}

// --- Tools -----------------------------------------------------------------

func TestContractToolsReturnsSortedAndFresh(t *testing.T) {
	opts := contractOpts(t)
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	first := sess.Tools()
	if len(first) == 0 {
		t.Fatal("Tools() returned empty slice")
	}
	for i := 1; i < len(first); i++ {
		if first[i-1] > first[i] {
			t.Errorf("Tools() not sorted: %v", first)
		}
	}

	// Mutation must not leak.
	first[0] = "MUTATED"
	second := sess.Tools()
	for _, n := range second {
		if n == "MUTATED" {
			t.Errorf("Tools() mutation leaked: %v contains MUTATED", second)
		}
	}
}

// recordingHeadlessTool is a custom HeadlessTool that records the args
// it received on Execute and returns a fixed text result. It exercises
// the public-API contract that an embedder-supplied HeadlessTool is
// dispatched by the agent loop on a matching ToolCallEvent.
//
// The methods map 1:1 onto tau.HeadlessTool (= tools.HeadlessTool):
//   - Name, Description: metadata
//   - Parameters: jsonschema.Schema from github.com/invopop/jsonschema
//     (an external public package any embedder implementing a custom
//     tool already depends on)
//   - Execute(ctx, tau.ToolCall) (tau.ToolResult, error)
//
// tau.ToolCall and tau.ToolResult are type aliases for the internal
// tools types, so an embedder constructs and inspects them entirely
// through the SDK surface.
type recordingHeadlessTool struct {
	mu       sync.Mutex
	invoked  bool
	gotArgs  string
	gotCwd   string
}

func (h *recordingHeadlessTool) Name() string { return "contract-custom" }
func (h *recordingHeadlessTool) Description() string {
	return "custom headless tool for contract test"
}
func (h *recordingHeadlessTool) Parameters() jsonschema.Schema {
	return jsonschema.Schema{Type: "object"}
}
func (h *recordingHeadlessTool) Execute(ctx context.Context, call ToolCall) (ToolResult, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.invoked = true
	h.gotArgs = string(call.Args)
	h.gotCwd = call.Cwd
	return NewTextResult("contract-custom-output"), nil
}

// customToolFaux is a faux LLMClient that on the first Stream emits a
// ToolCallDelta naming the custom tool, then a Final with stop_reason
// tool_use. On the second Stream (the agent loop re-streams with the
// tool result appended) it emits a plain text + end_turn. This drives
// the agent loop through the full tool-call → tool-result cycle for an
// embedder-supplied HeadlessTool.
type customToolFaux struct {
	mu    sync.Mutex
	calls int
}

func (f *customToolFaux) Stream(ctx context.Context, _ Request) (<-chan Delta, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	f.mu.Lock()
	f.calls++
	idx := f.calls
	f.mu.Unlock()
	ch := make(chan Delta, 4)
	go func() {
		defer close(ch)
		if idx == 1 {
			ch <- ToolCallDelta{ID: "tu1", Name: "contract-custom", PartialInput: `"hello"`}
			ch <- Final{StopReason: StopReasonToolUse}
			return
		}
		ch <- TextDelta{Text: "after tool"}
		ch <- Final{StopReason: StopReasonEndTurn}
	}()
	return ch, nil
}

// TestContractCustomHeadlessToolInvoked asserts that a custom
// HeadlessTool passed via Options.Tools is:
//
//   - registered in the session's tool registry (visible via Tools())
//   - dispatched by the agent loop when a provider-side ToolCallDelta
//     names it
//   - observed on the event bus as a TopicToolCall followed by a
//     TopicToolResult
//   - actually executed (its Execute was invoked with the model's
//     arguments)
//
// This covers task 8.1(d) of the spec.
func TestContractCustomHeadlessToolInvoked(t *testing.T) {
	tool := &recordingHeadlessTool{}
	opts := contractOpts(t)
	opts.LLMClient = &customToolFaux{}
	opts.Tools = []HeadlessTool{tool}

	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	// The custom tool must appear in Tools().
	if !contains(sliceStrings(sess.Tools()), "contract-custom") {
		t.Fatalf("Tools() = %v, missing contract-custom", sess.Tools())
	}

	// Subscribe to the tool_call + tool_result topics BEFORE Run so we
	// do not miss the (early) events.
	filtered := sess.SubscribeTopics(TopicToolCall, TopicToolResult)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := sess.Run(ctx, "use the custom tool"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Drain the filtered channel.
	var seenTopics []string
drain:
	for {
		select {
		case evt, ok := <-filtered:
			if !ok {
				break drain
			}
			seenTopics = append(seenTopics, string(evt.Topic()))
		case <-time.After(2 * time.Second):
			break drain
		}
	}

	if !contains(seenTopics, string(TopicToolCall)) {
		t.Errorf("filtered subscriber missed TopicToolCall; got %v", seenTopics)
	}
	if !contains(seenTopics, string(TopicToolResult)) {
		t.Errorf("filtered subscriber missed TopicToolResult; got %v", seenTopics)
	}

	// The tool must have actually run.
	tool.mu.Lock()
	invoked := tool.invoked
	gotArgs := tool.gotArgs
	tool.mu.Unlock()
	if !invoked {
		t.Fatal("custom HeadlessTool.Execute was not invoked by the agent loop")
	}
	if gotArgs != `"hello"` {
		t.Errorf("custom tool args = %q, want %q", gotArgs, `"hello"`)
	}
}

// sliceStrings is a local helper that converts a string slice to a
// bare []string for contains() comparisons. (sess.Tools() already
// returns []string; this is here so the call reads cleanly.)
func sliceStrings(in []string) []string { return in }

func TestContractProviderRegistryRejectsDuplicates(t *testing.T) {
	name := "contract-dup-" + sanitize(t.Name())
	factory := func(ProviderOptions) (LLMClient, error) { return fauxLLMClient{}, nil }
	if err := RegisterProvider(name, factory); err != nil {
		t.Fatalf("initial RegisterProvider: %v", err)
	}
	err := RegisterProvider(name, factory)
	if !errors.Is(err, ErrProviderAlreadyRegistered) {
		t.Errorf("duplicate RegisterProvider: got %v, want ErrProviderAlreadyRegistered", err)
	}
}

func TestContractProviderRegistryLookupRoundtrips(t *testing.T) {
	name := "contract-lookup-" + sanitize(t.Name())
	if err := RegisterProvider(name, func(ProviderOptions) (LLMClient, error) {
		return fauxLLMClient{}, nil
	}); err != nil {
		t.Fatalf("RegisterProvider: %v", err)
	}
	got, err := LookupProvider(name)
	if err != nil {
		t.Fatalf("LookupProvider: %v", err)
	}
	if got == nil {
		t.Fatal("LookupProvider returned nil factory")
	}
}

// TestContractProviderRegistryProvidersIncludesBuiltins is gated to the
// default build (built-in factories compiled in) and lives in
// contract_builtins_test.go. The mirror assertion for the
// `-tags provider_builtins_off` build is TestProvidersEmptyWhenBuiltinsCompiledOut
// in provider_nobuiltins_test.go.

// --- Slash commands --------------------------------------------------------

func TestContractSlashCommandsIncludesBuiltinsByDefault(t *testing.T) {
	opts := contractOpts(t)
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	got := sess.SlashCommands()
	for _, want := range []string{"help", "clear", "compact", "model", "quit"} {
		if !contains(got, want) {
			t.Errorf("SlashCommands() = %v, missing %q", got, want)
		}
	}
}

func TestContractSlashCommandsInjectedRegistryOverrides(t *testing.T) {
	opts := contractOpts(t)
	empty := NewRegistry()
	opts.SlashCommands = empty
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	got := sess.SlashCommands()
	if len(got) != 0 {
		t.Errorf("SlashCommands() with empty injected registry = %v, want []", got)
	}
}

// --- State -----------------------------------------------------------------

func TestContractStateInMemoryDoesNotTouchDisk(t *testing.T) {
	cwd := t.TempDir()
	mgr := NewInMemoryManager(cwd)
	opts := contractOpts(t)
	opts.Cwd = cwd
	opts.StateManager = mgr

	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := sess.Run(ctx, "hello"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	matches, err := filepath.Glob(filepath.Join(cwd, "*.bolt"))
	if err != nil {
		t.Fatalf("glob: %v", err)
	}
	if len(matches) > 0 {
		t.Errorf("found %d .bolt files under cwd; in-memory manager must not persist: %v", len(matches), matches)
	}
}

// --- Errors ----------------------------------------------------------------

// TestContractSentinelErrorsAreTyped verifies every exported SDK
// sentinel is usable with errors.Is. Sentinels split into two groups:
//
//  1. SDK-triggerable: the documented trigger can be exercised through
//     SDK-level operations alone (Run, Shutdown, LookupProvider,
//     RegisterProvider). For these we exercise the real trigger and
//     assert errors.Is.
//
//  2. Re-export only: the sentinel is an alias for an internal-package
//     sentinel whose trigger path requires an internal type
//     (*agent.AgentSession for slash dispatch; the tool Registry for
//     ErrUnknownTool / ErrToolAlreadyRegistered). The SDK does not
//     expose those operations, so the trigger cannot be exercised
//     from SDK-level code. For these we assert the SDK alias IS the
//     same value as the internal sentinel (pointer identity), which
//     proves errors.Is(err, tau.ErrXxx) is true exactly when
//     errors.Is(err, internal.ErrXxx) is true. The internal trigger
//     paths themselves are covered by internal/slash/slash_test.go
//     (TestRegistry_Execute_UnknownCommand, _NotASlashCommand,
//     TestQuitCommand, TestClearCommand, TestTreeCommand) and
//     internal/tools/tool_split_test.go (TestRegistryHeadlessSentinels).
func TestContractSentinelErrorsAreTyped(t *testing.T) {
	// --- Group 1: SDK-triggerable sentinels ---

	// ErrRuntimeShutdown — Run after Shutdown returns it.
	opts := contractOpts(t)
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	if err := sess.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
	if err := sess.Run(context.Background(), "x"); !errors.Is(err, ErrRuntimeShutdown) {
		t.Errorf("Run after Shutdown: got %v, want errors.Is(ErrRuntimeShutdown)", err)
	}

	// ErrProviderNotFound — LookupProvider on unknown name returns it.
	_, err = LookupProvider("contract-not-registered-" + sanitize(t.Name()))
	if !errors.Is(err, ErrProviderNotFound) {
		t.Errorf("LookupProvider unknown: got %v, want errors.Is(ErrProviderNotFound)", err)
	}

	// ErrProviderAlreadyRegistered — duplicate RegisterProvider returns it.
	name := "contract-sentinel-dup-" + sanitize(t.Name())
	_ = RegisterProvider(name, func(ProviderOptions) (LLMClient, error) { return fauxLLMClient{}, nil })
	err = RegisterProvider(name, func(ProviderOptions) (LLMClient, error) { return fauxLLMClient{}, nil })
	if !errors.Is(err, ErrProviderAlreadyRegistered) {
		t.Errorf("duplicate RegisterProvider: got %v, want errors.Is(ErrProviderAlreadyRegistered)", err)
	}

	// --- Group 2: re-exported sentinels (identity check) ---
	//
	// For each: the SDK alias must be the SAME error value as the
	// internal sentinel whose trigger is tested internally. Pointer
	// equality (==) implies errors.Is equivalence.

	// ErrUnknownTool = tools.ErrUnknownTool; trigger tested in
	// internal/tools/tool_split_test.go TestRegistryHeadlessSentinels
	// (Lookup of an unregistered name).
	if ErrUnknownTool != tools.ErrUnknownTool {
		t.Errorf("ErrUnknownTool identity mismatch: SDK sentinel is not tools.ErrUnknownTool")
	}
	if ErrUnknownTool == nil {
		t.Error("ErrUnknownTool is nil")
	}

	// ErrToolAlreadyRegistered = tools.ErrDuplicateTool; trigger tested
	// in internal/tools/tool_split_test.go TestRegistryHeadlessSentinels
	// (duplicate Register).
	if ErrToolAlreadyRegistered != tools.ErrDuplicateTool {
		t.Errorf("ErrToolAlreadyRegistered identity mismatch: SDK sentinel is not tools.ErrDuplicateTool")
	}
	if ErrToolAlreadyRegistered == nil {
		t.Error("ErrToolAlreadyRegistered is nil")
	}

	// ErrUnknownCommand = slash.ErrUnknownCommand; trigger tested in
	// internal/slash/slash_test.go TestRegistry_Execute_UnknownCommand.
	if ErrUnknownCommand != slash.ErrUnknownCommand {
		t.Errorf("ErrUnknownCommand identity mismatch: SDK sentinel is not slash.ErrUnknownCommand")
	}
	if ErrUnknownCommand == nil {
		t.Error("ErrUnknownCommand is nil")
	}

	// ErrNotASlashCommand = slash.ErrNotASlashCommand; trigger tested
	// in internal/slash/slash_test.go TestRegistry_Execute_NotASlashCommand.
	if ErrNotASlashCommand != slash.ErrNotASlashCommand {
		t.Errorf("ErrNotASlashCommand identity mismatch")
	}
	if ErrNotASlashCommand == nil {
		t.Error("ErrNotASlashCommand is nil")
	}

	// ErrQuitRequested = slash.ErrQuitRequested; trigger tested in
	// internal/slash/slash_test.go TestQuitCommand (/quit dispatch).
	if ErrQuitRequested != slash.ErrQuitRequested {
		t.Errorf("ErrQuitRequested identity mismatch")
	}
	if ErrQuitRequested == nil {
		t.Error("ErrQuitRequested is nil")
	}

	// ErrClearViewport = slash.ErrClearViewport; trigger tested in
	// internal/slash/slash_test.go TestClearCommand (/clear dispatch).
	if ErrClearViewport != slash.ErrClearViewport {
		t.Errorf("ErrClearViewport identity mismatch")
	}
	if ErrClearViewport == nil {
		t.Error("ErrClearViewport is nil")
	}

	// ErrShowTree = slash.ErrShowTree; trigger tested in
	// internal/slash/slash_test.go TestTreeCommand (/tree dispatch).
	if ErrShowTree != slash.ErrShowTree {
		t.Errorf("ErrShowTree identity mismatch")
	}
	if ErrShowTree == nil {
		t.Error("ErrShowTree is nil")
	}
}

// keep strings imported (used in future extensions of this file).
var _ = strings.Contains

// --- Middleware -----------------------------------------------------------
//
// Contract coverage for the request-middleware capability spec. Every
// requirement / scenario in specs/request-middleware/spec.md has a
// corresponding assertion here. Swap the fixtures (fauxprovider, the
// recording middleware adapters) for your real types when copying.

// recordingMutator captures each MutateRequest invocation for assertions.
type recordingMutator struct {
	calls atomic.Int32
	mu    sync.Mutex
	reqs  []Request
	err   error // returned on every call when non-nil
}

func (m *recordingMutator) MutateRequest(ctx context.Context, req *Request) error {
	m.calls.Add(1)
	m.mu.Lock()
	m.reqs = append(m.reqs, *req)
	err := m.err
	m.mu.Unlock()
	return err
}

func (m *recordingMutator) callCount() int32 { return m.calls.Load() }

// recordingObserver captures each ObserveResponse invocation.
type recordingObserver struct {
	calls atomic.Int32
	mu    sync.Mutex
	pairs []observedPair
	err   error
}

type observedPair struct {
	Req Request
	Res Response
}

func (o *recordingObserver) ObserveResponse(ctx context.Context, req *Request, resp *Response) error {
	o.calls.Add(1)
	o.mu.Lock()
	o.pairs = append(o.pairs, observedPair{Req: *req, Res: *resp})
	err := o.err
	o.mu.Unlock()
	return err
}

func (o *recordingObserver) callCount() int32 { return o.calls.Load() }

// shortCircuitInterceptor returns a fixed ToolResult from BeforeToolCall
// and records every AfterToolCall.
type shortCircuitInterceptor struct {
	before    *ToolResult // non-nil => short-circuit with this result
	beforeErr error

	afterMu  sync.Mutex
	afters   []ToolResult
	afterErr error
}

func (i *shortCircuitInterceptor) BeforeToolCall(ctx context.Context, call ToolCall) (*ToolResult, error) {
	return i.before, i.beforeErr
}

func (i *shortCircuitInterceptor) AfterToolCall(ctx context.Context, call ToolCall, result ToolResult) error {
	i.afterMu.Lock()
	i.afters = append(i.afters, result)
	i.afterMu.Unlock()
	return i.afterErr
}

func (i *shortCircuitInterceptor) afterResults() []ToolResult {
	i.afterMu.Lock()
	defer i.afterMu.Unlock()
	out := make([]ToolResult, len(i.afters))
	copy(out, i.afters)
	return out
}

// countingTool counts its Execute invocations. Satisfies HeadlessTool.
type countingTool struct {
	name  string
	calls atomic.Int32
}

func (t *countingTool) Name() string                  { return t.name }
func (t *countingTool) Description() string           { return "counts Execute calls" }
func (t *countingTool) Parameters() jsonschema.Schema { return jsonschema.Schema{} }
func (t *countingTool) Execute(ctx context.Context, call ToolCall) (ToolResult, error) {
	t.calls.Add(1)
	return NewTextResult(t.name + ":executed"), nil
}

func (t *countingTool) callCount() int32 { return t.calls.Load() }

// resultText extracts the text of the first TextContent block from a
// ToolResult. Returns "" when the result has no text block. Contract
// tests use this to assert short-circuit results without reaching into
// internal/types.
func resultText(r ToolResult) string {
	for _, b := range r.Content {
		if tc, ok := b.(TextContent); ok {
			return tc.Text
		}
	}
	return ""
}

// toolCallScript returns an LLMClient that emits one tool-call delta
// (the named tool) followed by a Final with StopReasonToolUse on the
// FIRST call, then a plain text delta + Final with StopReasonEndTurn on
// every subsequent call. This drives tool-interceptor contract tests
// through the public SDK: the agent loop dispatches the named tool once,
// re-dispatches, gets the closing EndTurn, and the turn completes
// without an infinite loop.
type toolCallScript struct {
	toolName string
	calls    atomic.Int32
}

func (s *toolCallScript) Stream(ctx context.Context, req Request) (<-chan Delta, error) {
	n := s.calls.Add(1)
	ch := make(chan Delta, 2)
	go func() {
		defer close(ch)
		if n == 1 {
			ch <- ToolCallDelta{ContentIndex: 0, ID: "tu_contract", Name: s.toolName, PartialInput: "{}"}
			ch <- Final{StopReason: StopReasonToolUse}
			return
		}
		ch <- TextDelta{ContentIndex: 0, Text: "done"}
		ch <- Final{StopReason: StopReasonEndTurn}
	}()
	return ch, nil
}

// TestContractMiddlewareRequestMutatorInvoked: register a recording
// RequestMutator that mutates req.Tools, run a turn with a faux provider,
// and assert the provider observed the mutated request.
func TestContractMiddlewareRequestMutatorInvoked(t *testing.T) {
	recordingClient := fauxprovider.NewWithResponse("ok")
	mutator := &recordingMutator{}
	opts := contractOpts(t)
	opts.LLMClient = recordingClient
	opts.Middleware = []any{mutator}
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession with mutator: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	// Mutator clears Tools on every call so we can observe the mutation
	// reaching the provider.
	mutator.err = nil
	goMutator := &reqToolsClearingMutator{inner: mutator}
	opts2 := contractOpts(t)
	opts2.LLMClient = recordingClient
	opts2.Middleware = []any{goMutator}
	sess2, err := CreateAgentSession(context.Background(), opts2)
	if err != nil {
		t.Fatalf("CreateAgentSession with clearing mutator: %v", err)
	}
	t.Cleanup(func() { sess2.Shutdown(context.Background()) })

	if err := sess2.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if got := goMutator.calls.Load(); got != 1 {
		t.Errorf("mutator invoked %d times, want 1", got)
	}
	reqs := recordingClient.RecordedRequests()
	if len(reqs) == 0 {
		t.Fatal("no requests recorded by the provider")
	}
	if len(reqs[len(reqs)-1].Tools) != 0 {
		t.Errorf("mutator did not clear Tools; provider saw %d entries", len(reqs[len(reqs)-1].Tools))
	}
}

// reqToolsClearingMutator wraps recordingMutator and clears req.Tools.
type reqToolsClearingMutator struct {
	inner *recordingMutator
	calls atomic.Int32
}

func (m *reqToolsClearingMutator) MutateRequest(ctx context.Context, req *Request) error {
	m.calls.Add(1)
	req.Tools = nil
	return m.inner.MutateRequest(ctx, req)
}

// TestContractMiddlewareResponseObserverInvoked: register a recording
// ResponseObserver, run a turn, assert it was called once with the
// request and the response.
func TestContractMiddlewareResponseObserverInvoked(t *testing.T) {
	observer := &recordingObserver{}
	opts := contractOpts(t)
	opts.Middleware = []any{observer}
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession with observer: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	if err := sess.Run(context.Background(), "hi"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if got := observer.callCount(); got != 1 {
		t.Errorf("observer invoked %d times, want 1", got)
	}
}

// TestContractMiddlewareToolInterceptorShortCircuit: register a
// ToolInterceptor returning NewTextResult("intercepted") from
// BeforeToolCall; register a countingTool; run a turn that triggers
// the tool; assert the tool's Execute was NOT called and the turn
// result reflects "intercepted".
func TestContractMiddlewareToolInterceptorShortCircuit(t *testing.T) {
	intercepted := NewTextResult("intercepted")
	interceptor := &shortCircuitInterceptor{before: &intercepted}
	tool := &countingTool{name: "counter"}

	opts := contractOpts(t)
	opts.LLMClient = &toolCallScript{toolName: "counter"}
	opts.Tools = []HeadlessTool{tool}
	opts.Middleware = []any{interceptor}
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession with interceptor: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	if err := sess.Run(context.Background(), "call counter"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if got := tool.callCount(); got != 0 {
		t.Errorf("underlying tool Execute called %d times, want 0 (short-circuit)", got)
	}
	afters := interceptor.afterResults()
	if len(afters) != 1 {
		t.Errorf("AfterToolCall invoked %d times, want 1", len(afters))
	}
	if len(afters) == 1 {
		got := resultText(afters[0])
		if got != "intercepted" {
			t.Errorf("AfterToolCall result text = %q, want %q", got, "intercepted")
		}
	}
}

// TestContractMiddlewareOrderingPreserved: register two RequestMutators
// (A then B) where B asserts it ran after A by checking shared state.
func TestContractMiddlewareOrderingPreserved(t *testing.T) {
	shared := &orderingState{}
	mutA := &orderingMutator{name: "A", shared: shared}
	mutB := &orderingMutator{name: "B", shared: shared, requireSawA: true}

	opts := contractOpts(t)
	opts.Middleware = []any{mutA, mutB}
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession with two mutators: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	if err := sess.Run(context.Background(), "ordering"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if mutB.calls.Load() != 1 || mutA.calls.Load() != 1 {
		t.Errorf("mutator invocations A=%d B=%d, want 1 and 1", mutA.calls.Load(), mutB.calls.Load())
	}
}

type orderingState struct {
	mu   sync.Mutex
	seen []string
}

func (s *orderingState) record(name string) {
	s.mu.Lock()
	s.seen = append(s.seen, name)
	s.mu.Unlock()
}

func (s *orderingState) has(name string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, n := range s.seen {
		if n == name {
			return true
		}
	}
	return false
}

type orderingMutator struct {
	name        string
	calls       atomic.Int32
	shared      *orderingState
	requireSawA bool
}

func (m *orderingMutator) MutateRequest(ctx context.Context, req *Request) error {
	m.calls.Add(1)
	if m.requireSawA && !m.shared.has("A") {
		return errors.New("orderingMutator B: did not see A first")
	}
	m.shared.record(m.name)
	return nil
}

// TestContractMiddlewareUnknownTypeRejected: pass Options.Middleware{42}
// to CreateAgentSession; assert the error satisfies
// errors.Is(err, ErrUnknownMiddlewareType); assert no session is
// allocated.
func TestContractMiddlewareUnknownTypeRejected(t *testing.T) {
	opts := contractOpts(t)
	opts.Middleware = []any{42}
	sess, err := CreateAgentSession(context.Background(), opts)
	if err == nil {
		t.Cleanup(func() { sess.Shutdown(context.Background()) })
		t.Fatal("CreateAgentSession with int middleware: got nil error, want one")
	}
	if !errors.Is(err, ErrUnknownMiddlewareType) {
		t.Errorf("err = %v, want errors.Is(err, ErrUnknownMiddlewareType)", err)
	}
	if sess != nil {
		t.Error("session allocated despite rejection")
	}
}

// TestContractMiddlewareErrorPropagation: a RequestMutator returning a
// sentinel error aborts Run; a ResponseObserver returning a sentinel
// error does NOT abort Run (it is logged instead).
func TestContractMiddlewareErrorPropagation(t *testing.T) {
	t.Run("mutator error aborts", func(t *testing.T) {
		sentinel := errors.New("mutator-abort")
		mutator := &recordingMutator{err: sentinel}
		opts := contractOpts(t)
		opts.Middleware = []any{mutator}
		sess, err := CreateAgentSession(context.Background(), opts)
		if err != nil {
			t.Fatalf("CreateAgentSession: %v", err)
		}
		t.Cleanup(func() { sess.Shutdown(context.Background()) })

		runErr := sess.Run(context.Background(), "go")
		if !errors.Is(runErr, sentinel) {
			t.Errorf("Run err = %v, want errors.Is(err, sentinel)", runErr)
		}
	})

	t.Run("observer error does not abort", func(t *testing.T) {
		sentinel := errors.New("observer-log")
		observer := &recordingObserver{err: sentinel}
		opts := contractOpts(t)
		opts.Middleware = []any{observer}
		sess, err := CreateAgentSession(context.Background(), opts)
		if err != nil {
			t.Fatalf("CreateAgentSession: %v", err)
		}
		t.Cleanup(func() { sess.Shutdown(context.Background()) })

		runErr := sess.Run(context.Background(), "go")
		if runErr != nil {
			t.Errorf("Run err = %v, want nil (observer error is logged, not propagated)", runErr)
		}
		if observer.callCount() != 1 {
			t.Errorf("observer invoked %d times, want 1 (error did not block remaining observers)", observer.callCount())
		}
	})
}

// TestContractMiddlewareSentinelIdentity: ErrUnknownMiddlewareType
// satisfies the standard errors.Is identity contract.
func TestContractMiddlewareSentinelIdentity(t *testing.T) {
	if !errors.Is(ErrUnknownMiddlewareType, ErrUnknownMiddlewareType) {
		t.Error("errors.Is(sentinel, sentinel) = false, want true")
	}
	if ErrUnknownMiddlewareType == nil {
		t.Error("ErrUnknownMiddlewareType is nil")
	}
}
