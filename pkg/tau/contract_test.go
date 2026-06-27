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
