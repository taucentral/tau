package slash

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/coevin/tau/internal/agent"
	"github.com/coevin/tau/internal/config"
	"github.com/coevin/tau/internal/fauxprovider"
	"github.com/coevin/tau/internal/tools"
)

// newTestSession wires an AgentSession against the faux provider so
// slash tests don't touch the network. The session is registered with
// t.Cleanup for shutdown. KnownModels and ProviderAPI are left empty
// to mirror the real faux-provider wiring (see cli/wire.go).
func newTestSession(t *testing.T) *agent.AgentSession {
	t.Helper()
	client := fauxprovider.NewWithResponse("faux reply")
	opts := agent.SessionOptions{
		Model:     "faux",
		Settings:  config.DefaultSettings(),
		LLMClient: client,
		Tools:     []tools.HeadlessTool{tools.NewReadTool(tools.OSReadOperations{})},
	}
	rt, err := agent.CreateAgentSessionRuntime(context.Background(), t.TempDir(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSessionRuntime: %v", err)
	}
	sess := agent.NewAgentSession(rt)
	t.Cleanup(func() { sess.Shutdown(context.Background()) })
	return sess
}

// newTestSessionWithModels wires a faux-provider session whose runtime
// carries a populated KnownModels list. Used by the /model tests to
// exercise validation, listing, and cross-API refusal without standing
// up a real LLM client. The session's active Model is always
// "claude-opus" with ProviderAPI=anthropic — tests that exercise
// switches build their KnownModels around that starting point.
func newTestSessionWithModels(t *testing.T, known []config.KnownModel) *agent.AgentSession {
	t.Helper()
	client := fauxprovider.NewWithResponse("faux reply")
	opts := agent.SessionOptions{
		Model:       "claude-opus",
		Settings:    config.DefaultSettings(),
		LLMClient:   client,
		Tools:       []tools.HeadlessTool{tools.NewReadTool(tools.OSReadOperations{})},
		KnownModels: known,
		ProviderAPI: config.APIAnthropic,
	}
	rt, err := agent.CreateAgentSessionRuntime(context.Background(), t.TempDir(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSessionRuntime: %v", err)
	}
	sess := agent.NewAgentSession(rt)
	t.Cleanup(func() { sess.Shutdown(context.Background()) })
	return sess
}

// --- Parse ---

func TestParse(t *testing.T) {
	tests := []struct {
		input string
		name  string
		args  string
		ok    bool
	}{
		{"hello", "", "", false},
		{"/", "", "", false},
		{"  /", "", "", false},
		{"/quit", "/quit", "", true},
		{"  /quit  ", "/quit", "", true},
		{"/label  my label", "/label", "my label", true},
		{"/model\tfaux-1", "/model", "faux-1", true},
		{"/checkout   e12345  ", "/checkout", "e12345  ", true},
		{"plain text", "", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			name, args, ok := Parse(tt.input)
			if ok != tt.ok {
				t.Errorf("ok = %v, want %v", ok, tt.ok)
			}
			if name != tt.name {
				t.Errorf("name = %q, want %q", name, tt.name)
			}
			if args != tt.args {
				t.Errorf("args = %q, want %q", args, tt.args)
			}
		})
	}
}

// --- Registry ---

func TestRegistry_RegisterAndLookup(t *testing.T) {
	r := NewRegistry()
	r.Register(newQuitCommand())
	if _, ok := r.Lookup("/quit"); !ok {
		t.Error("Lookup /quit returned false after Register")
	}
	if _, ok := r.Lookup("/nosuch"); ok {
		t.Error("Lookup /nosuch returned true; want false")
	}
}

func TestRegistry_Names(t *testing.T) {
	r := DefaultRegistry()
	names := r.Names()
	want := []string{"/checkout", "/clear", "/cls", "/compact", "/fork", "/help", "/label", "/model", "/quit", "/tree"}
	if len(names) != len(want) {
		t.Fatalf("Names count = %d, want %d (%v)", len(names), len(want), names)
	}
	for i, w := range want {
		if names[i] != w {
			t.Errorf("Names[%d] = %q, want %q", i, names[i], w)
		}
	}
}

func TestRegistry_Execute_UnknownCommand(t *testing.T) {
	r := NewRegistry()
	_, err := r.Execute(context.Background(), "/nosuch", newTestSession(t).AsCommandSession())
	if !errors.Is(err, ErrUnknownCommand) {
		t.Errorf("err = %v, want ErrUnknownCommand", err)
	}
}

func TestRegistry_Execute_NotASlashCommand(t *testing.T) {
	r := NewRegistry()
	_, err := r.Execute(context.Background(), "hello", nil)
	if !errors.Is(err, ErrNotASlashCommand) {
		t.Errorf("err = %v, want ErrNotASlashCommand", err)
	}
}

// --- Individual commands ---

func TestQuitCommand(t *testing.T) {
	r := DefaultRegistry()
	_, err := r.Execute(context.Background(), "/quit", newTestSession(t).AsCommandSession())
	if !errors.Is(err, ErrQuitRequested) {
		t.Errorf("/quit: err = %v, want ErrQuitRequested", err)
	}
}

func TestClearCommand(t *testing.T) {
	r := DefaultRegistry()
	sess := newTestSession(t)
	_, err := r.Execute(context.Background(), "/clear", sess.AsCommandSession())
	if !errors.Is(err, ErrContextReset) {
		t.Errorf("/clear: err = %v, want ErrContextReset", err)
	}
}

func TestClsCommand(t *testing.T) {
	r := DefaultRegistry()
	_, err := r.Execute(context.Background(), "/cls", newTestSession(t).AsCommandSession())
	if !errors.Is(err, ErrClearScreen) {
		t.Errorf("/cls: err = %v, want ErrClearScreen", err)
	}
}

// TestClearViewportAliasCompatibility verifies that the deprecated
// ErrClearViewport alias still matches the same error value as
// ErrClearScreen, so any caller matching on the old name keeps working.
func TestClearViewportAliasCompatibility(t *testing.T) {
	if !errors.Is(ErrClearScreen, ErrClearViewport) {
		t.Error("errors.Is(ErrClearScreen, ErrClearViewport) = false; alias broken")
	}
	if !errors.Is(ErrClearViewport, ErrClearScreen) {
		t.Error("errors.Is(ErrClearViewport, ErrClearScreen) = false; alias broken")
	}
}

func TestTreeCommand(t *testing.T) {
	r := DefaultRegistry()
	_, err := r.Execute(context.Background(), "/tree", newTestSession(t).AsCommandSession())
	if !errors.Is(err, ErrShowTree) {
		t.Errorf("/tree: err = %v, want ErrShowTree", err)
	}
}

func TestModelCommand_PrintCurrent_FauxNoRegistry(t *testing.T) {
	r := DefaultRegistry()
	sess := newTestSession(t)
	out, err := r.Execute(context.Background(), "/model", sess.AsCommandSession())
	if err != nil {
		t.Fatalf("/model: %v", err)
	}
	if !strings.Contains(out, "faux") {
		t.Errorf("/model output = %q, want substring 'faux'", out)
	}
	if !strings.Contains(out, "no models.json configured") {
		t.Errorf("/model output should hint at missing models.json: %q", out)
	}
}

// TestModelCommand_SwitchesWithoutValidationWhenNoRegistry covers the
// "no models.json configured" branch: the switch is applied but the
// response must be honest about not having validated the id. This
// replaces the pre-fix behaviour where /model silently accepted any
// string and claimed a clean switch.
func TestModelCommand_SwitchesWithoutValidationWhenNoRegistry(t *testing.T) {
	r := DefaultRegistry()
	sess := newTestSession(t)
	out, err := r.Execute(context.Background(), "/model new-model-id", sess.AsCommandSession())
	if err != nil {
		t.Fatalf("/model switch: %v", err)
	}
	if !strings.Contains(out, "new-model-id") {
		t.Errorf("output = %q, want 'new-model-id'", out)
	}
	if !strings.Contains(out, "not validated") {
		t.Errorf("output should warn the id was not validated: %q", out)
	}
	if sess.Runtime().Options.Model != "new-model-id" {
		t.Errorf("after switch: Model = %q", sess.Runtime().Options.Model)
	}
}

func TestModelCommand_AlreadyActive_NoStateChange(t *testing.T) {
	r := DefaultRegistry()
	known := []config.KnownModel{
		{Provider: "ant", Model: config.ModelDefinition{ID: "claude-opus", API: config.APIAnthropic}},
	}
	sess := newTestSessionWithModels(t, known)

	out, err := r.Execute(context.Background(), "/model claude-opus", sess.AsCommandSession())
	if err != nil {
		t.Fatalf("/model already-active: %v", err)
	}
	if !strings.Contains(out, "already active") {
		t.Errorf("output = %q, want 'already active'", out)
	}
	if sess.Runtime().Options.Model != "claude-opus" {
		t.Errorf("Model changed unexpectedly: %q", sess.Runtime().Options.Model)
	}
}

func TestModelCommand_NoArgs_ListsKnownModels(t *testing.T) {
	r := DefaultRegistry()
	known := []config.KnownModel{
		{Provider: "ant", Model: config.ModelDefinition{ID: "claude-opus", API: config.APIAnthropic}},
		{Provider: "oai", Model: config.ModelDefinition{ID: "gpt-4o", API: config.APIOpenAI}},
	}
	sess := newTestSessionWithModels(t, known)

	out, err := r.Execute(context.Background(), "/model", sess.AsCommandSession())
	if err != nil {
		t.Fatalf("/model: %v", err)
	}
	if !strings.Contains(out, "claude-opus") || !strings.Contains(out, "gpt-4o") {
		t.Errorf("listing missing models: %q", out)
	}
	if !strings.Contains(out, "[anthropic]") || !strings.Contains(out, "[openai]") {
		t.Errorf("listing missing api tags: %q", out)
	}
	// The active entry must be marked.
	markLines := strings.Split(out, "\n")
	var sawActiveMarker bool
	for _, line := range markLines {
		if strings.Contains(line, "claude-opus") && strings.Contains(line, "←") {
			sawActiveMarker = true
		}
	}
	if !sawActiveMarker {
		t.Errorf("active model not marked with ←: %q", out)
	}
}

func TestModelCommand_SwitchesToKnownModel_SameAPI(t *testing.T) {
	r := DefaultRegistry()
	known := []config.KnownModel{
		{Provider: "ant", Model: config.ModelDefinition{ID: "claude-opus", API: config.APIAnthropic}},
		{Provider: "ant", Model: config.ModelDefinition{ID: "claude-sonnet", API: config.APIAnthropic}},
	}
	sess := newTestSessionWithModels(t, known)

	out, err := r.Execute(context.Background(), "/model claude-sonnet", sess.AsCommandSession())
	if err != nil {
		t.Fatalf("/model same-api switch: %v", err)
	}
	if !strings.Contains(out, "claude-opus → claude-sonnet") {
		t.Errorf("output = %q, want transition message", out)
	}
	if strings.Contains(out, "not validated") {
		// Sanity: same-API switch should NOT carry the unvalidated warning.
		t.Errorf("same-API switch should not warn unvalidated: %q", out)
	}
	if sess.Runtime().Options.Model != "claude-sonnet" {
		t.Errorf("Model = %q, want claude-sonnet", sess.Runtime().Options.Model)
	}
}

func TestModelCommand_RefusesUnknownModel_ListsAvailable(t *testing.T) {
	r := DefaultRegistry()
	known := []config.KnownModel{
		{Provider: "ant", Model: config.ModelDefinition{ID: "claude-opus", API: config.APIAnthropic}},
	}
	sess := newTestSessionWithModels(t, known)

	_, err := r.Execute(context.Background(), "/model bogus-model", sess.AsCommandSession())
	if err == nil {
		t.Fatal("/model with unknown id should error when KnownModels is populated")
	}
	if !strings.Contains(err.Error(), "not in models.json") {
		t.Errorf("err = %q, want 'not in models.json'", err)
	}
	if !strings.Contains(err.Error(), "claude-opus") {
		t.Errorf("err should list available models: %q", err)
	}
	if sess.Runtime().Options.Model != "claude-opus" {
		t.Errorf("Model changed on refused switch: %q", sess.Runtime().Options.Model)
	}
}

func TestModelCommand_RefusesCrossAPI(t *testing.T) {
	r := DefaultRegistry()
	known := []config.KnownModel{
		{Provider: "ant", Model: config.ModelDefinition{ID: "claude-opus", API: config.APIAnthropic}},
		{Provider: "oai", Model: config.ModelDefinition{ID: "gpt-4o", API: config.APIOpenAI}},
	}
	sess := newTestSessionWithModels(t, known)

	_, err := r.Execute(context.Background(), "/model gpt-4o", sess.AsCommandSession())
	if err == nil {
		t.Fatal("/model cross-API switch should error")
	}
	if !strings.Contains(err.Error(), "uses API") || !strings.Contains(err.Error(), "Restart tau") {
		t.Errorf("err = %q, want cross-API refusal with restart hint", err)
	}
	if sess.Runtime().Options.Model != "claude-opus" {
		t.Errorf("Model changed on refused switch: %q", sess.Runtime().Options.Model)
	}
}

func TestModelCommand_CaseInsensitiveMatch(t *testing.T) {
	r := DefaultRegistry()
	known := []config.KnownModel{
		{Provider: "ant", Model: config.ModelDefinition{ID: "claude-opus", API: config.APIAnthropic}},
		{Provider: "ant", Model: config.ModelDefinition{ID: "claude-sonnet", API: config.APIAnthropic}},
	}
	sess := newTestSessionWithModels(t, known)

	out, err := r.Execute(context.Background(), "/model CLAUDE-SONNET", sess.AsCommandSession())
	if err != nil {
		t.Fatalf("/model case-insensitive: %v", err)
	}
	if sess.Runtime().Options.Model != "claude-sonnet" {
		t.Errorf("Model = %q, want claude-sonnet (lowercased registry form)", sess.Runtime().Options.Model)
	}
	if !strings.Contains(out, "claude-sonnet") {
		t.Errorf("output = %q, want transition target", out)
	}
}

func TestCheckoutCommand_NoEntryID(t *testing.T) {
	r := DefaultRegistry()
	_, err := r.Execute(context.Background(), "/checkout", newTestSession(t).AsCommandSession())
	if err == nil {
		t.Error("/checkout with no arg: err = nil, want usage error")
	}
}

func TestCheckoutCommand_UnknownEntry(t *testing.T) {
	r := DefaultRegistry()
	_, err := r.Execute(context.Background(), "/checkout nonexistent", newTestSession(t).AsCommandSession())
	if err == nil {
		t.Error("/checkout unknown entry: err = nil, want error")
	}
}

func TestLabelCommand_NoText(t *testing.T) {
	r := DefaultRegistry()
	_, err := r.Execute(context.Background(), "/label", newTestSession(t).AsCommandSession())
	if err == nil {
		t.Error("/label with no text: err = nil, want usage error")
	}
}

func TestLabelCommand_AppendsLabel(t *testing.T) {
	r := DefaultRegistry()
	sess := newTestSession(t)
	out, err := r.Execute(context.Background(), "/label my label text", sess.AsCommandSession())
	if err != nil {
		t.Fatalf("/label: %v", err)
	}
	if !strings.Contains(out, "my label text") {
		t.Errorf("output = %q, want substring 'my label text'", out)
	}
}

func TestForkCommand_EmptySession(t *testing.T) {
	r := DefaultRegistry()
	_, err := r.Execute(context.Background(), "/fork", newTestSession(t).AsCommandSession())
	if err == nil {
		t.Error("/fork on empty session: err = nil, want error")
	}
}

func TestCompactCommand_PrintsNoCompactionNeeded(t *testing.T) {
	r := DefaultRegistry()
	sess := newTestSession(t)
	out, err := r.Execute(context.Background(), "/compact", sess.AsCommandSession())
	if err != nil {
		t.Fatalf("/compact: %v", err)
	}
	// Fresh empty session → compactor reports empty.
	if !strings.Contains(out, "no compaction") && !strings.Contains(out, "empty") {
		t.Errorf("/compact output = %q, want substring 'no compaction' or 'empty'", out)
	}
}

func TestHelpCommand_ListsAllCommands(t *testing.T) {
	r := DefaultRegistry()
	sess := newTestSession(t)
	out, err := r.Execute(context.Background(), "/help", sess.AsCommandSession())
	if err != nil {
		t.Fatalf("/help: %v", err)
	}
	for _, name := range r.Names() {
		if !strings.Contains(out, name) {
			t.Errorf("/help output missing %q: %q", name, out)
		}
	}
}

// --- Nil-session guards ---

func TestCommands_RejectNilSession(t *testing.T) {
	r := DefaultRegistry()
	for _, name := range r.Names() {
		// /quit, /cls, /tree, and /help do not require a session —
		// they return immediately or only consult the registry.
		// /clear DOES require a session now (it appends a ClearMarker).
		switch name {
		case "/quit", "/cls", "/tree", "/help":
			continue
		}
		t.Run(name, func(t *testing.T) {
			_, err := r.Execute(context.Background(), name, nil)
			if err == nil {
				t.Errorf("%s with nil session: err = nil, want error", name)
			}
		})
	}
}

// --- Custom command invocation (task 7.4a) ---

// recordingCommand is a custom Command that records its invocation args
// for assertion. It demonstrates that an embedder-supplied Registry
// containing a custom Command dispatches through Registry.Execute when
// the matching /<name> directive is sent.
type recordingCommand struct {
	name string
	got  string
}

func (c *recordingCommand) Name() string      { return c.name }
func (c *recordingCommand) ShortHelp() string { return "custom command for testing" }
func (c *recordingCommand) Execute(ctx context.Context, args string, session agent.CommandSession) (string, error) {
	c.got = args
	return "custom-command-output", nil
}

// TestRegistry_ExecutesCustomCommand verifies that a custom Command
// registered into a Registry is invoked when its /<name> directive is
// sent through Registry.Execute. This covers task 7.4(a): "an
// embedder-supplied Registry containing a custom Command is invoked
// when the agent loop receives a matching /custom directive."
func TestRegistry_ExecutesCustomCommand(t *testing.T) {
	reg := NewRegistry()
	cmd := &recordingCommand{name: "/custom"}
	reg.Register(cmd)
	// Also register the default /help so the registry is not empty
	// around it; this proves custom + built-in commands coexist.
	help := &helpCommand{}
	reg.Register(help)
	help.registry = reg

	sess := newTestSession(t)
	out, err := reg.Execute(context.Background(), "/custom arg-value", sess.AsCommandSession())
	if err != nil {
		t.Fatalf("/custom: %v", err)
	}
	if out != "custom-command-output" {
		t.Errorf("/custom output = %q, want %q", out, "custom-command-output")
	}
	if cmd.got != "arg-value" {
		t.Errorf("custom command received args = %q, want %q", cmd.got, "arg-value")
	}
}

// TestRegistry_CustomCommandWithNoArgs covers the zero-argument case:
// "/custom" with no trailing text must invoke Execute with args="".
func TestRegistry_CustomCommandWithNoArgs(t *testing.T) {
	reg := NewRegistry()
	cmd := &recordingCommand{name: "/custom"}
	reg.Register(cmd)

	sess := newTestSession(t)
	if _, err := reg.Execute(context.Background(), "/custom", sess.AsCommandSession()); err != nil {
		t.Fatalf("/custom (no args): %v", err)
	}
	if cmd.got != "" {
		t.Errorf("custom command args = %q, want empty", cmd.got)
	}
}

// TestRegistry_CustomCommandCoexistsWithBuiltins verifies that
// registering a custom command alongside the built-in set does not
// disturb dispatch of either. This addresses the spec's "embedder can
// selectively disable built-ins by constructing a registry from an
// explicit subset" + custom-additive intent.
func TestRegistry_CustomCommandCoexistsWithBuiltins(t *testing.T) {
	reg := DefaultRegistry()
	cmd := &recordingCommand{name: "/custom"}
	reg.Register(cmd)

	sess := newTestSession(t)
	// Built-in still dispatches.
	if _, err := reg.Execute(context.Background(), "/quit", sess.AsCommandSession()); err != nil {
		if !errors.Is(err, ErrQuitRequested) {
			t.Errorf("/quit after custom register: err = %v, want ErrQuitRequested", err)
		}
	}
	// Custom dispatches.
	if _, err := reg.Execute(context.Background(), "/custom hi", sess.AsCommandSession()); err != nil {
		t.Errorf("/custom after built-in register: %v", err)
	}
	if cmd.got != "hi" {
		t.Errorf("custom command args = %q, want %q", cmd.got, "hi")
	}
}
