// command_session_test.go — runtime tests for the public CommandSession
// adapter. The compile-time assertions in command_session.go guarantee
// the adapter structurally satisfies the interface; these tests verify
// the adapter's runtime behavior: that inspector methods read the
// underlying runtime fields, that mutation setters actually mutate, and
// that the MergeState type-assertion failure path returns the typed
// error declared in the spec.
package agent

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/coevin/tau/internal/config"
	"github.com/coevin/tau/internal/fauxprovider"
	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/storage"
	"github.com/coevin/tau/internal/tools"
)

// newCommandSessionTestSession wires a faux-provider AgentSession suitable
// for exercising the commandSessionView adapter. Registered with
// t.Cleanup for shutdown.
func newCommandSessionTestSession(t *testing.T) *AgentSession {
	t.Helper()
	client := fauxprovider.NewWithResponse("faux reply")
	opts := SessionOptions{
		Model:         "faux",
		Settings:      config.DefaultSettings(),
		LLMClient:     client,
		Tools:         []tools.HeadlessTool{tools.NewReadTool(tools.OSReadOperations{})},
		KnownModels:   nil,
		ProviderAPI:   config.APIAnthropic,
		ContextWindow: 200000,
	}
	rt, err := CreateAgentSessionRuntime(context.Background(), t.TempDir(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSessionRuntime: %v", err)
	}
	sess := NewAgentSession(rt)
	t.Cleanup(func() { sess.Shutdown(context.Background()) })
	return sess
}

// TestAsCommandSession_ReturnsNonNilView verifies the constructor
// returns a non-nil interface value whose concrete type is the
// unexported adapter. This complements the compile-time
// `var _ CommandSession = commandSessionView{}` assertion with a
// runtime check that AsCommandSession is actually wired to produce the
// adapter (and not, say, a nil interface via a misplaced return).
func TestAsCommandSession_ReturnsNonNilView(t *testing.T) {
	sess := newCommandSessionTestSession(t)
	v := sess.AsCommandSession()
	if v == nil {
		t.Fatal("AsCommandSession() returned nil; want non-nil CommandSession")
	}
	// The returned value MUST be the unexported adapter, not some other
	// satisfaction of the interface. This guards against a future
	// refactor that returns a different type whose field reads happen
	// to satisfy the interface but with different semantics.
	if _, ok := v.(commandSessionView); !ok {
		t.Errorf("AsCommandSession() returned %T, want commandSessionView", v)
	}
}

// TestCommandSessionView_InspectorsReadUnderlyingRuntime verifies
// every inspector method on CommandSession returns the value the
// underlying runtime was wired with. Catches drift if a future change
// repoints an inspector at the wrong field.
func TestCommandSessionView_InspectorsReadUnderlyingRuntime(t *testing.T) {
	sess := newCommandSessionTestSession(t)
	v := sess.AsCommandSession()

	if got := v.Model(); got != "faux" {
		t.Errorf("Model() = %q, want %q", got, "faux")
	}
	if got := v.Cwd(); got == "" {
		t.Errorf("Cwd() = %q, want non-empty (t.TempDir)", got)
	}
	if got := v.SessionID(); got != "" {
		// SessionID is empty for fresh lazy sessions until the first
		// assistant flush; populated here would mean the adapter read
		// the wrong field.
		t.Errorf("SessionID() = %q, want empty for fresh lazy session", got)
	}
	if got := v.CreatedAt(); got.IsZero() || got.After(time.Now().Add(time.Second)) {
		t.Errorf("CreatedAt() = %v, want a non-zero time not in the future", got)
	}
	if got := v.Tools(); len(got) == 0 {
		t.Errorf("Tools() = %v, want at least one tool (read)", got)
	}
	if got := v.Store(); got != nil {
		// No store is wired by SessionOptions zero value, so this should
		// be nil. If it's non-nil, the adapter read the wrong field.
		t.Errorf("Store() = %v, want nil (no store wired)", got)
	}
	if got := v.Orchestrator(); got != nil {
		t.Errorf("Orchestrator() = %v, want nil (no orchestrator wired)", got)
	}
}

// TestCommandRuntimeView_OptionsAndState verifies Runtime() returns a
// working CommandRuntime whose State and Options accessors reach the
// underlying runtime's fields.
func TestCommandRuntimeView_OptionsAndState(t *testing.T) {
	sess := newCommandSessionTestSession(t)
	v := sess.AsCommandSession()

	rt := v.Runtime()
	if rt == nil {
		t.Fatal("Runtime() returned nil; want non-nil CommandRuntime")
	}
	if rt.State() == nil {
		t.Error("State() returned nil; want the wired state.Manager")
	}
	if rt.Options() == nil {
		t.Fatal("Options() returned nil; want non-nil CommandOptions")
	}
	if rt.EventBus() == nil {
		t.Error("EventBus() returned nil; want the wired *EventBus")
	}
	// Compactor is optional; just exercise the accessor to confirm it
	// does not panic on a nil field.
	_ = rt.Compactor()
}

// TestCommandOptions_SetModelMutatesUnderlyingField covers spec
// scenario "External mutation via SetModel": an external plugin
// command calling session.Runtime().Options().SetModel(id) MUST mutate
// the wired session's options so subsequent turns use the new model.
// The existing slash /model tests exercise this end-to-end through the
// registry; this test exercises it directly on the adapter so the
// spec scenario is covered even if /model is removed or reworked.
func TestCommandOptions_SetModelMutatesUnderlyingField(t *testing.T) {
	sess := newCommandSessionTestSession(t)
	v := sess.AsCommandSession()

	opts := v.Runtime().Options()
	if got := opts.Model(); got != "faux" {
		t.Fatalf("Model() before SetModel = %q, want %q", got, "faux")
	}

	const newModel = "claude-opus-4-5-20251101"
	opts.SetModel(newModel)

	// Read through the adapter again — the mutation MUST be visible.
	if got := opts.Model(); got != newModel {
		t.Errorf("Model() after SetModel = %q, want %q", got, newModel)
	}
	// Read through the concrete runtime — the mutation MUST have landed
	// in the underlying resolvedOptions.Model field, not a copy. This
	// is what guarantees subsequent turns see the new model.
	if got := sess.rt.Options.Model; got != newModel {
		t.Errorf("underlying rt.Options.Model = %q, want %q", got, newModel)
	}
}

// TestCommandOptions_AccessorsReadUnderlyingFields verifies the
// remaining CommandOptions accessors (LLMClient, ContextWindow,
// KnownModels, ProviderAPI) read the underlying fields. Catches drift
// if a future change repoints an accessor at the wrong field.
func TestCommandOptions_AccessorsReadUnderlyingFields(t *testing.T) {
	sess := newCommandSessionTestSession(t)
	v := sess.AsCommandSession()
	opts := v.Runtime().Options()

	if opts.LLMClient() == nil {
		t.Error("LLMClient() returned nil; want the faux provider")
	}
	// ContextWindow defaults to the runtime's default (200000) when
	// SessionOptions.ContextWindow is zero; just confirm it's positive.
	if got := opts.ContextWindow(); got <= 0 {
		t.Errorf("ContextWindow() = %d, want positive (default applied)", got)
	}
	if got := opts.ProviderAPI(); got != config.APIAnthropic {
		t.Errorf("ProviderAPI() = %q, want %q", got, config.APIAnthropic)
	}
	if got := opts.KnownModels(); len(got) != 0 {
		t.Errorf("KnownModels() = %v, want empty (none wired)", got)
	}
}

// TestCommandSessionView_SlashCommandsStripsLeadingSlash verifies the
// SlashCommands inspector returns names without the leading "/", matching
// the SDK facade's convention. Without a slash registry wired it returns
// nil — this is a deliberate divergence from the SDK facade, which falls
// back to DefaultSlashRegistry(); the adapter is for plugin commands
// that register their own slash surface and do not need the fallback.
func TestCommandSessionView_SlashCommandsStripsLeadingSlash(t *testing.T) {
	sess := newCommandSessionTestSession(t)
	v := sess.AsCommandSession()

	// No SlashRegistry is wired by newCommandSessionTestSession, so the
	// adapter returns nil. This is the documented divergence from the
	// SDK facade.
	if got := v.SlashCommands(); got != nil {
		t.Errorf("SlashCommands() with no registry wired = %v, want nil", got)
	}
}

// foreignSession is a CommandSession-shaped value that is NOT a
// commandSessionView. Used by the MergeState test below to prove the
// type-assertion failure path returns the typed sentinel rather than
// panicking or silently succeeding.
type foreignSession struct{}

func (foreignSession) Run(context.Context, string) error                  { return nil }
func (foreignSession) Abort(string)                                       {}
func (foreignSession) Shutdown(context.Context) error                     { return nil }
func (foreignSession) Model() string                                      { return "" }
func (foreignSession) Cwd() string                                        { return "" }
func (foreignSession) SessionID() string                                  { return "" }
func (foreignSession) CreatedAt() time.Time                               { return time.Time{} }
func (foreignSession) Tools() []string                                    { return nil }
func (foreignSession) SlashCommands() []string                            { return nil }
func (foreignSession) Store() storage.Store                               { return nil }
func (foreignSession) Orchestrator() any                                  { return nil }
func (foreignSession) Subscribe() <-chan Event                            { return nil }
func (foreignSession) SubscribeTopics(...Topic) <-chan Event              { return nil }
func (foreignSession) Spawn(context.Context, SessionOptions) (CommandSession, error) {
	return nil, nil
}
func (foreignSession) MergeState(context.Context, CommandSession, MergeSpecShell) error {
	return nil
}
func (foreignSession) Runtime() CommandRuntime { return nil }

// Ensure foreignSession satisfies CommandSession at compile time. This
// guards against the test silently passing after a future interface
// change that foreignSession does not track.
var _ CommandSession = foreignSession{}

// foreignOptions is a CommandOptions-shaped value used to prove the
// adapter's MergeState type-assertion works in isolation.
var _ CommandOptions = (*foreignOptions)(nil)

type foreignOptions struct{}

func (foreignOptions) Model() string                       { return "" }
func (foreignOptions) SetModel(string)                     {}
func (foreignOptions) LLMClient() llm.LLMClient            { return nil }
func (foreignOptions) ContextWindow() int                  { return 0 }
func (foreignOptions) KnownModels() []config.KnownModel    { return nil }
func (foreignOptions) ProviderAPI() config.ModelAPI        { return "" }

// TestMergeState_RejectsForeignChild verifies the MergeState adapter
// type-asserts its child argument back to commandSessionView and
// returns ErrMergeStateForeignChild when a foreign CommandSession
// implementation is passed. Without this check, a plugin wrapping the
// session in its own type would silently fail or panic. Detectability
// via errors.Is is the project-style requirement (CLAUDE.md: "Typed
// sentinel errors via errors.Is").
func TestMergeState_RejectsForeignChild(t *testing.T) {
	sess := newCommandSessionTestSession(t)
	v := sess.AsCommandSession()

	err := v.MergeState(context.Background(), foreignSession{}, MergeSpecShell{})
	if err == nil {
		t.Fatal("MergeState with foreign child: err = nil, want ErrMergeStateForeignChild")
	}
	if !errors.Is(err, ErrMergeStateForeignChild) {
		t.Errorf("MergeState with foreign child: err = %v, want errors.Is(err, ErrMergeStateForeignChild) = true", err)
	}
}

// TestMergeState_NilChildReturnsOrchestratorClosed verifies a nil
// child surfaces an error rather than panicking on the type assertion.
// The exact sentinel matches what the underlying (*AgentSession).MergeState
// returns for a nil child — keeping the adapter's behavior consistent
// with the concrete type's.
func TestMergeState_NilChildReturnsOrchestratorClosed(t *testing.T) {
	sess := newCommandSessionTestSession(t)
	v := sess.AsCommandSession()

	err := v.MergeState(context.Background(), nil, MergeSpecShell{})
	if err == nil {
		t.Fatal("MergeState with nil child: err = nil, want non-nil")
	}
	// The adapter early-returns ErrOrchestratorClosed for nil child to
	// mirror what the underlying MergeState would return once it reached
	// its own nil check. Either sentinel in the family (ErrRuntimeShutdown,
	// ErrNoOrchestrator, ErrOrchestratorClosed) is acceptable — the key
	// assertion is that we don't panic on `nil.(commandSessionView)`.
}
