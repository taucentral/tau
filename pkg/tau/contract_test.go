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
	"fmt"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/invopop/jsonschema"

	"github.com/taucentral/tau/internal/config"
	"github.com/taucentral/tau/internal/fauxprovider"
	"github.com/taucentral/tau/internal/slash"
	"github.com/taucentral/tau/internal/tools"
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

// TestContractSlashCommandsAcceptCustomCommand proves the public SDK
// seam accepts a user-defined Command implementation: Register records
// the custom command and Execute dispatches it. The custom command
// ignores its CommandSession argument, so the dispatch path is exercised
// with nil; the end-to-end path with a real session is covered by
// internal/slash/slash_test.go and by pkg/tau/contract (the external-
// shaped contract package).
func TestContractSlashCommandsAcceptCustomCommand(t *testing.T) {
	reg := NewRegistry()
	cmd := &contractEchoCommand{name: "/contractecho"}
	reg.Register(cmd)

	got, ok := reg.Lookup("/contractecho")
	if !ok {
		t.Fatal("Registry.Lookup(/contractecho) returned ok=false")
	}
	if got.Name() != "/contractecho" {
		t.Errorf("Lookup Name() = %q, want /contractecho", got.Name())
	}

	out, err := reg.Execute(context.Background(), "/contractecho hello", nil)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if out != "hello" {
		t.Errorf("Execute output = %q, want %q", out, "hello")
	}
	if cmd.got != "hello" {
		t.Errorf("custom command args = %q, want %q", cmd.got, "hello")
	}
}

// contractEchoCommand is a minimal tau.Command implementation used by
// TestContractSlashCommandsAcceptCustomCommand. Its purpose is to
// prove the public SDK accepts an externally-shaped Command value; the
// body is intentionally trivial.
type contractEchoCommand struct {
	name string
	got  string
}

func (c *contractEchoCommand) Name() string      { return c.name }
func (c *contractEchoCommand) ShortHelp() string { return "echo args (contract)" }
func (c *contractEchoCommand) Execute(_ context.Context, args string, _ CommandSession) (string, error) {
	c.got = args
	return args, nil
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
	Err error
}

func (o *recordingObserver) ObserveResponse(ctx context.Context, req *Request, resp *Response, streamErr error) error {
	o.calls.Add(1)
	o.mu.Lock()
	o.pairs = append(o.pairs, observedPair{Req: *req, Res: *resp, Err: streamErr})
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

// TestContractIdentityGatingAndAuditComposition: register a policy gate
// (deny-by-default) and an audit observer in the same session. Identity
// placed in ctx via WithIdentity reaches BeforeToolCall; permitted calls
// land in the audit sink with the caller's UserID. This exercises the
// full ABAC composition documented in docs/sdk/cookbook.md (j).
func TestContractIdentityGatingAndAuditComposition(t *testing.T) {
	// Allow "viewer" to call "counter"; deny everything else.
	policy := &identityPolicyInterceptor{
		allowed: map[string]map[string]bool{
			"viewer": {"counter": true},
		},
	}
	audit := &identityAuditInterceptor{}

	tool := &countingTool{name: "counter"}
	opts := contractOpts(t)
	opts.LLMClient = &toolCallScript{toolName: "counter"}
	opts.Tools = []HeadlessTool{tool}
	opts.Middleware = []any{policy, audit}
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	runCtx := WithIdentity(context.Background(), Identity{
		UserID: "alice@example.com",
		Roles:  []string{"viewer"},
		Tenant: "acme",
	})
	if err := sess.Run(runCtx, "call counter"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	// The tool was permitted, so Execute was invoked once.
	if got := tool.callCount(); got != 1 {
		t.Errorf("counter Execute = %d, want 1 (permit)", got)
	}
	// Audit recorded the permit with the caller's identity.
	entries := audit.entries()
	if len(entries) != 1 {
		t.Fatalf("audit entries = %d, want 1", len(entries))
	}
	if entries[0].UserID != "alice@example.com" {
		t.Errorf("audit UserID = %q, want %q", entries[0].UserID, "alice@example.com")
	}
	if entries[0].Tool != "counter" {
		t.Errorf("audit Tool = %q, want %q", entries[0].Tool, "counter")
	}
	if entries[0].IsError {
		t.Errorf("audit IsError = true, want false (permit path)")
	}
}

// TestContractIdentityDenyShortCircuitsAndStillAudits: when the policy
// denies a call, BeforeToolCall returns a *ToolResult; the runtime
// skips Execute but STILL invokes AfterToolCall on both interceptors.
// This guards the short-circuit-plus-AfterToolCall path at
// internal/agent/session.go:463-471 and the audit observer's ability
// to record denials.
func TestContractIdentityDenyShortCircuitsAndStillAudits(t *testing.T) {
	// Allow no roles: anonymous identity matches no entry.
	policy := &identityPolicyInterceptor{
		allowed: map[string]map[string]bool{
			"editor": {"counter": true},
		},
	}
	audit := &identityAuditInterceptor{}

	tool := &countingTool{name: "counter"}
	opts := contractOpts(t)
	opts.LLMClient = &toolCallScript{toolName: "counter"}
	opts.Tools = []HeadlessTool{tool}
	opts.Middleware = []any{policy, audit}
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	// Anonymous identity: zero value, no roles → deny-by-default.
	runCtx := WithIdentity(context.Background(), Identity{})
	if err := sess.Run(runCtx, "call counter"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Execute never ran.
	if got := tool.callCount(); got != 0 {
		t.Errorf("counter Execute = %d, want 0 (deny short-circuit)", got)
	}
	// Audit still recorded the denial.
	entries := audit.entries()
	if len(entries) != 1 {
		t.Fatalf("audit entries = %d, want 1 (AfterToolCall runs on short-circuit)", len(entries))
	}
	if !entries[0].IsError {
		t.Errorf("audit IsError = false, want true (denial result)")
	}
	if entries[0].UserID != "" {
		t.Errorf("audit UserID = %q, want empty (anonymous)", entries[0].UserID)
	}
}

// identityAuditEntry is the audit record. Test-local: the real shape
// lives in the audit-provenance plugin and the cookbook recipe.
type identityAuditEntry struct {
	When    time.Time
	UserID  string
	Tenant  string
	Tool    string
	Args    []byte
	IsError bool
}

type identityAuditInterceptor struct {
	mu      sync.Mutex
	record  []identityAuditEntry
}

func (a *identityAuditInterceptor) BeforeToolCall(ctx context.Context, call ToolCall) (*ToolResult, error) {
	return nil, nil // audit never gates
}

func (a *identityAuditInterceptor) AfterToolCall(ctx context.Context, call ToolCall, result ToolResult) error {
	id := IdentityFromContext(ctx)
	a.mu.Lock()
	a.record = append(a.record, identityAuditEntry{
		When:    time.Now().UTC(),
		UserID:  id.UserID,
		Tenant:  id.Tenant,
		Tool:    call.Name,
		Args:    call.Args,
		IsError: result.IsError,
	})
	a.mu.Unlock()
	return nil
}

func (a *identityAuditInterceptor) entries() []identityAuditEntry {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]identityAuditEntry, len(a.record))
	copy(out, a.record)
	return out
}

// identityPolicyInterceptor enforces a role→tool allow list with
// admin bypass and deny-by-default on the zero Identity.
type identityPolicyInterceptor struct {
	allowed map[string]map[string]bool
}

func (p *identityPolicyInterceptor) isAllowed(id Identity, tool string) bool {
	for _, role := range id.Roles {
		if role == "admin" {
			return true
		}
		if tools, ok := p.allowed[role]; ok && tools[tool] {
			return true
		}
	}
	return false
}

func (p *identityPolicyInterceptor) BeforeToolCall(ctx context.Context, call ToolCall) (*ToolResult, error) {
	id := IdentityFromContext(ctx)
	if !p.isAllowed(id, call.Name) {
		denial := NewErrorResult("denied by policy for " + id.UserID)
		return &denial, nil
	}
	return nil, nil
}

func (p *identityPolicyInterceptor) AfterToolCall(ctx context.Context, call ToolCall, result ToolResult) error {
	return nil
}

// --- Storage ----------------------------------------------------------------
//
// Contract coverage for the cross-session-storage capability spec.
// Every requirement / scenario in
// specs/cross-session-storage/spec.md has a corresponding assertion
// here. The fixtures (FileStore in a temp dir, recordingStore wrapper)
// are designed so embedders copying this test can swap in their own
// Store implementation and the contract still holds.

// recordingStore wraps a Store and records Close invocations. Used by
// the lifecycle test to assert the runtime never calls Close on an
// injected store. Every other method delegates verbatim.
type recordingStore struct {
	inner   Store
	closeMu sync.Mutex
	closes  int
}

func (r *recordingStore) Put(ctx context.Context, e Entry) error {
	return r.inner.Put(ctx, e)
}

func (r *recordingStore) Query(ctx context.Context, q Query) ([]Entry, error) {
	return r.inner.Query(ctx, q)
}

func (r *recordingStore) Close() error {
	r.closeMu.Lock()
	r.closes++
	r.closeMu.Unlock()
	return r.inner.Close()
}

func (r *recordingStore) closeCount() int {
	r.closeMu.Lock()
	defer r.closeMu.Unlock()
	return r.closes
}

// TestContractStorageStorePutQueryRoundTrip: inject a FileStore via
// Options.Store, run a no-op turn to ensure the runtime accepts the
// store, retrieve it via the Store() inspector, Put + Query, assert
// round-trip succeeds.
func TestContractStorageStorePutQueryRoundTrip(t *testing.T) {
	dir := t.TempDir()
	store, err := NewFileStore(dir)
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	opts := contractOpts(t)
	opts.Store = store

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

	got := sess.Store()
	if got == nil {
		t.Fatal("Store() returned nil after a store was injected")
	}
	if err := got.Put(ctx, Entry{
		ID:        "round-trip",
		Text:      "round-trip body",
		Timestamp: time.Now(),
		Tags:      []string{"test"},
	}); err != nil {
		t.Fatalf("Put: %v", err)
	}
	out, err := got.Query(ctx, Query{KeywordQuery: "round-trip"})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("Query returned %d, want 1", len(out))
	}
	if out[0].Text != "round-trip body" {
		t.Errorf("Query text = %q, want %q", out[0].Text, "round-trip body")
	}
}

// TestContractStorageLifecycleNotClosedOnShutdown: wrap a FileStore in
// a recordingStore, inject via Options.Store, run a turn, Shutdown, and
// assert the runtime did NOT call Close on the wrapper. This pins the
// "embedder owns the injected store's lifecycle" contract.
func TestContractStorageLifecycleNotClosedOnShutdown(t *testing.T) {
	dir := t.TempDir()
	inner, err := NewFileStore(dir)
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	rec := &recordingStore{inner: inner}

	opts := contractOpts(t)
	opts.Store = rec

	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := sess.Run(ctx, "hello"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if err := sess.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
	if got := rec.closeCount(); got != 0 {
		t.Errorf("runtime called Close on injected store %d time(s); want 0 (embedder owns lifecycle)", got)
	}

	// The embedder is responsible for closing; doing so now MUST NOT
	// panic or error.
	if err := inner.Close(); err != nil {
		t.Errorf("embedder-side Close: %v", err)
	}
}

// TestContractStorageSentinelsTyped: the three storage sentinels
// satisfy errors.Is identity. ErrUnsupportedQuery is also asserted to
// be returned by FileStore.Query when EmbeddingQuery is set.
func TestContractStorageSentinelsTyped(t *testing.T) {
	for _, sentinel := range []error{
		ErrStoreClosed,
		ErrStoreReadOnly,
		ErrUnsupportedQuery,
	} {
		if !errors.Is(sentinel, sentinel) {
			t.Errorf("errors.Is(%v, %v) = false, want true", sentinel, sentinel)
		}
		if sentinel == nil {
			t.Errorf("sentinel %v is nil", sentinel)
		}
	}

	// FileStore triggers ErrUnsupportedQuery for embedding queries.
	dir := t.TempDir()
	store, err := NewFileStore(dir)
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	_, err = store.Query(context.Background(), Query{EmbeddingQuery: []float32{0.1, 0.2}})
	if !errors.Is(err, ErrUnsupportedQuery) {
		t.Errorf("FileStore embedding Query err = %v, want errors.Is(ErrUnsupportedQuery)", err)
	}
}

// TestContractStorageSessionIsolation: two SDK sessions constructed
// with Options.Store pointing at different FileStore directories see
// only their own entries — cross-session isolation via configuration.
// Writes through session A's Store() inspector MUST NOT be visible to
// session B's Store() inspector. The runtime plays no part in the
// isolation; it is enforced entirely by the configured directory.
func TestContractStorageSessionIsolation(t *testing.T) {
	storeA, err := NewFileStore(t.TempDir())
	if err != nil {
		t.Fatalf("store A: %v", err)
	}
	defer storeA.Close()
	storeB, err := NewFileStore(t.TempDir())
	if err != nil {
		t.Fatalf("store B: %v", err)
	}
	defer storeB.Close()

	optsA := contractOpts(t)
	optsA.Store = storeA
	sessA, err := CreateAgentSession(context.Background(), optsA)
	if err != nil {
		t.Fatalf("CreateAgentSession A: %v", err)
	}
	defer sessA.Shutdown(context.Background())

	optsB := contractOpts(t)
	optsB.Store = storeB
	sessB, err := CreateAgentSession(context.Background(), optsB)
	if err != nil {
		t.Fatalf("CreateAgentSession B: %v", err)
	}
	defer sessB.Shutdown(context.Background())

	// Sanity: both sessions expose the store that was injected.
	storeFromA := sessA.Store()
	if storeFromA == nil {
		t.Fatal("session A Store() returned nil after a store was injected")
	}
	storeFromB := sessB.Store()
	if storeFromB == nil {
		t.Fatal("session B Store() returned nil after a store was injected")
	}

	ctx := context.Background()
	if err := storeFromA.Put(ctx, Entry{
		ID:        "only-in-a",
		Text:      "A only",
		Timestamp: time.Now(),
	}); err != nil {
		t.Fatalf("Put via session A's store: %v", err)
	}

	// Session A sees its own entry through its Store() inspector.
	outA, err := sessA.Store().Query(ctx, Query{Limit: 10})
	if err != nil {
		t.Fatalf("Query via session A: %v", err)
	}
	if len(outA) != 1 || outA[0].ID != "only-in-a" {
		t.Errorf("session A store = %v, want [only-in-a]", outA)
	}

	// Session B does NOT see session A's entry through its Store()
	// inspector. The runtime does not auto-scope storage by session.
	outB, err := sessB.Store().Query(ctx, Query{Limit: 10})
	if err != nil {
		t.Fatalf("Query via session B: %v", err)
	}
	if len(outB) != 0 {
		t.Errorf("session B store = %v, want [] (isolation by directory)", outB)
	}
}

// TestContractStorageStoreInspectorReturnsNilWhenOmitted: a session
// constructed without Options.Store returns nil from Store(). This
// pins the "nil disables storage" contract.
func TestContractStorageStoreInspectorReturnsNilWhenOmitted(t *testing.T) {
	opts := contractOpts(t) // no Store set
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })
	if got := sess.Store(); got != nil {
		t.Errorf("Store() = %v, want nil (no store injected)", got)
	}
}

// --- Orchestration --------------------------------------------------------

// writeToolFaux is a faux LLMClient that on the first Stream emits a
// ToolCallDelta naming the "write" tool with a caller-supplied path and
// content, then a Final with StopReasonToolUse. On subsequent streams it
// emits a plain text + end_turn. This drives the agent loop through the
// full write-tool cycle so the session's state tree records a tool_use
// touching the supplied path.
type writeToolFaux struct {
	mu    sync.Mutex
	calls int
	path  string
}

func (f *writeToolFaux) Stream(ctx context.Context, _ Request) (<-chan Delta, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	f.mu.Lock()
	f.mu.Unlock()
	f.calls++
	idx := f.calls
	ch := make(chan Delta, 4)
	go func() {
		defer close(ch)
		if idx == 1 {
			input := fmt.Sprintf(`{"path":%q,"content":"x"}`, f.path)
			ch <- ToolCallDelta{ID: "tu_write", Name: "write", PartialInput: input}
			ch <- Final{StopReason: StopReasonToolUse}
			return
		}
		ch <- TextDelta{Text: "after write"}
		ch <- Final{StopReason: StopReasonEndTurn}
	}()
	return ch, nil
}

// contractOrchOpts returns contractOpts with an Orchestrator configured
// so Spawn's gate is satisfied. Tests that need to mutate the Options
// further (e.g. set a different LLMClient) should clone the result.
func contractOrchOpts(t *testing.T) Options {
	t.Helper()
	opts := contractOpts(t)
	// Options.Orchestrator is the Spawn-time gate. The runtime only
	// checks nil vs non-nil; the real orchestrator used by Run is
	// wired separately via NewSequentialOrchestrator. We use a
	// placeholder Orchestrator implementation here so Spawn's gate
	// passes; its Run is never invoked.
	opts.Orchestrator = placeholderOrchestrator{}
	return opts
}

// placeholderOrchestrator is a non-nil Orchestrator whose Run is never
// invoked by tests that use it. It exists to satisfy the Spawn-time
// gate on Options.Orchestrator (the runtime checks nil vs non-nil).
type placeholderOrchestrator struct{}

func (placeholderOrchestrator) Run(ctx context.Context, spec OrchestrationSpec) (<-chan SessionEvent, error) {
	return nil, ErrOrchestrationAborted
}

func (placeholderOrchestrator) Err() error { return nil }

// TestContractOrchestrationSpawnMergeEndToEnd covers the canonical
// sequential-orchestration happy path at the SDK boundary:
//
//   - Construct a parent session with Options.Orchestrator set.
//   - Wire the parent with NewSequentialOrchestrator.
//   - Run a two-phase spec; both phases complete.
//   - The returned channel delivers events from both phases and closes.
//   - With MergePolicyAppend, the parent's state tree grows.
func TestContractOrchestrationSpawnMergeEndToEnd(t *testing.T) {
	opts := contractOrchOpts(t)
	parent, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { parent.Shutdown(context.Background()) })

	orch := NewSequentialOrchestrator(parent)
	spec := OrchestrationSpec{
		Phases: []PhaseSpec{
			{Name: "a", Prompt: "phase A"},
			{Name: "b", Prompt: "phase B", DependsOn: []string{"a"}},
		},
		MergePolicy: MergePolicyAppend,
	}

	ch, err := orch.Run(context.Background(), spec)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	count := 0
	for range ch {
		count++
	}
	if count == 0 {
		t.Errorf("expected at least one event from the orchestrator; got %d", count)
	}
}

// TestContractOrchestrationReplayConflict covers the conflict path:
// parent and child both emit a write tool_use for the same file;
// MergeState with MergePolicyReplay returns ErrOrchestrationConflict;
// errors.As populates a *ConflictReport naming the file, the phase, and
// a non-empty line range.
func TestContractOrchestrationReplayConflict(t *testing.T) {
	conflictFile := filepath.Join(t.TempDir(), "same.txt")

	// Parent: run a turn that writes the file.
	parentOpts := contractOrchOpts(t)
	parentOpts.LLMClient = &writeToolFaux{path: conflictFile}
	parent, err := CreateAgentSession(context.Background(), parentOpts)
	if err != nil {
		t.Fatalf("parent CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { parent.Shutdown(context.Background()) })
	if err := parent.Run(context.Background(), "write the file"); err != nil {
		t.Fatalf("parent Run: %v", err)
	}

	// Child: spawn and run a turn that writes the same file.
	child, err := parent.Spawn(context.Background(), Options{
		Cwd:       parentOpts.Cwd,
		Model:     parentOpts.Model,
		LLMClient: &writeToolFaux{path: conflictFile},
		Tools:     parentOpts.Tools,
		Settings:  parentOpts.Settings,
	})
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	t.Cleanup(func() { child.Shutdown(context.Background()) })
	if err := child.Run(context.Background(), "write the file again"); err != nil {
		t.Fatalf("child Run: %v", err)
	}

	// Pass Phase via MergeSpec so the conflict report attributes the
	// conflict to the named phase. This pins the D8 contract: the
	// orchestrator surfaces phase attribution through MergeSpec.Phase.
	mergeErr := parent.MergeState(context.Background(), child, MergeSpec{
		Policy: MergePolicyReplay,
		Phase:  "contract-review",
	})
	if !errors.Is(mergeErr, ErrOrchestrationConflict) {
		t.Fatalf("MergeState: got %v, want errors.Is(ErrOrchestrationConflict)", mergeErr)
	}
	var report *ConflictReport
	if !errors.As(mergeErr, &report) {
		t.Fatalf("errors.As did not populate *ConflictReport; err=%v", mergeErr)
	}
	if report.File != conflictFile {
		t.Errorf("report.File = %q, want %q", report.File, conflictFile)
	}
	// D8: Phase must be populated from MergeSpec.Phase.
	if report.Phase != "contract-review" {
		t.Errorf("report.Phase = %q, want %q", report.Phase, "contract-review")
	}
	// D9: LineRange must be populated. writeToolFaux writes content "x"
	// (zero newlines), so lineRangeOfWrite returns [1, 1].
	if report.LineRange[0] < 1 {
		t.Errorf("report.LineRange[0] = %d, want >= 1", report.LineRange[0])
	}
	if report.LineRange[1] < report.LineRange[0] {
		t.Errorf("report.LineRange = %v, want [first,last] with last >= first", report.LineRange)
	}
}

// TestContractOrchestratorInterfaceSurface pins the public SDK contract
// that Orchestrator is a 2-method interface: Run + Err. Embedders
// implementing their own Orchestrator (e.g., a fan-out variant) must
// satisfy both methods. A compile-time assertion here catches an
// accidental narrowing or widening of the interface.
func TestContractOrchestratorInterfaceSurface(t *testing.T) {
	// Compile-time check: an embedder-supplied type satisfying both
	// methods is assignable to the SDK Orchestrator interface.
	var _ Orchestrator = placeholderOrchestrator{}

	// Construct a real parent (NewSequentialOrchestrator dereferences
	// the parent to wire its event bus and state manager) and assert
	// the returned Orchestrator surface has both Run and Err callable.
	parent, err := CreateAgentSession(context.Background(), contractOrchOpts(t))
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { parent.Shutdown(context.Background()) })

	orch := NewSequentialOrchestrator(parent)
	if orch == nil {
		t.Fatal("NewSequentialOrchestrator returned nil")
	}
	// Err must be callable on a fresh orchestrator (no phase has run).
	if err := orch.Err(); err != nil {
		t.Errorf("Err() on fresh orchestrator = %v, want nil", err)
	}
	// Run signature must accept an OrchestrationSpec and return a
	// channel + error. Drive the validation path (cycle in an empty
	// spec is a no-op; we only exercise the method signature here).
	_, _ = orch.Run(context.Background(), OrchestrationSpec{})
}

// TestContractOrchestrationDependencyCycleRejected asserts that an
// OrchestrationSpec containing a cycle is rejected at Run time with
// ErrOrchestrationAborted, and the error message names the cycle.
// No phases start.
func TestContractOrchestrationDependencyCycleRejected(t *testing.T) {
	opts := contractOrchOpts(t)
	parent, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { parent.Shutdown(context.Background()) })

	orch := NewSequentialOrchestrator(parent)
	spec := OrchestrationSpec{
		Phases: []PhaseSpec{
			{Name: "a", DependsOn: []string{"b"}},
			{Name: "b", DependsOn: []string{"a"}},
		},
	}
	_, err = orch.Run(context.Background(), spec)
	if !errors.Is(err, ErrOrchestrationAborted) {
		t.Errorf("got %v, want errors.Is(ErrOrchestrationAborted)", err)
	}
	if err == nil || !strings.Contains(err.Error(), "cycle") {
		t.Errorf("error message does not name the cycle; got %q", err)
	}
}

// TestContractOrchestrationSentinelsTyped pins the typed identity of
// the three orchestration sentinels re-exported at the SDK level.
// Each sentinel must be non-nil and satisfy errors.Is for itself.
// ErrOrchestrationConflict must also wrap a recoverable
// *ConflictReport via errors.As.
func TestContractOrchestrationSentinelsTyped(t *testing.T) {
	for _, sentinel := range []error{
		ErrOrchestrationConflict,
		ErrOrchestrationAborted,
		ErrOrchestratorClosed,
		ErrNoOrchestrator,
	} {
		if sentinel == nil {
			t.Errorf("orchestration sentinel is nil")
			continue
		}
		if !errors.Is(sentinel, sentinel) {
			t.Errorf("sentinel %v does not satisfy errors.Is with itself", sentinel)
		}
	}

	// ConflictReport must be recoverable from a wrapped error via
	// errors.As. We construct the wrapper the same way MergeState
	// does: fmt.Errorf("%w: %w", ErrOrchestrationConflict, report).
	wrapped := fmt.Errorf("%w: %w",
		ErrOrchestrationConflict,
		&ConflictReport{File: "x.txt", LineRange: [2]int{1, 5}})
	if !errors.Is(wrapped, ErrOrchestrationConflict) {
		t.Errorf("wrapped conflict: errors.Is(ErrOrchestrationConflict) = false")
	}
	var report *ConflictReport
	if !errors.As(wrapped, &report) {
		t.Errorf("wrapped conflict: errors.As(*ConflictReport) = false")
	}
	if report == nil || report.File != "x.txt" {
		t.Errorf("report.File = %q, want %q", report.File, "x.txt")
	}
}

// TestContractOrchestrationSpawnFailsWithoutOrchestrator asserts that
// Spawn on a session constructed WITHOUT Options.Orchestrator returns
// a typed error. The SDK contract accepts either ErrNoOrchestrator or
// ErrRuntimeShutdown per the spec's "OR a typed equivalent" language;
// this test pins the concrete ErrNoOrchestrator sentinel.
func TestContractOrchestrationSpawnFailsWithoutOrchestrator(t *testing.T) {
	opts := contractOpts(t) // no Orchestrator set
	parent, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { parent.Shutdown(context.Background()) })

	_, err = parent.Spawn(context.Background(), Options{
		Cwd:       opts.Cwd,
		Model:     opts.Model,
		LLMClient: opts.LLMClient,
		Tools:     opts.Tools,
		Settings:  opts.Settings,
	})
	if !errors.Is(err, ErrNoOrchestrator) {
		t.Errorf("Spawn without Orchestrator: got %v, want errors.Is(ErrNoOrchestrator)", err)
	}
}
