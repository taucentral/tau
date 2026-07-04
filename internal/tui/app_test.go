package tui

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/coevin/tau/internal/agent"
	"github.com/coevin/tau/internal/config"
	"github.com/coevin/tau/internal/fauxprovider"
	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/tools"
	"github.com/coevin/tau/internal/tui/components"
)

// newTestApp constructs an AppModel wired against the faux provider so
// tests can drive it without network access. The returned session is
// registered with t.Cleanup so it shuts down when the test finishes.
func newTestApp(t *testing.T, width, height int) *AppModel {
	t.Helper()
	client := fauxprovider.NewWithResponse("faux assistant reply")

	opts := agent.SessionOptions{
		Model:         "faux",
		Settings:      config.DefaultSettings(),
		LLMClient:     client,
		Tools:         []tools.HeadlessTool{tools.NewReadTool(tools.OSReadOperations{})},
		ContextWindow: 200000,
	}
	rt, err := agent.CreateAgentSessionRuntime(context.Background(), t.TempDir(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSessionRuntime: %v", err)
	}
	sess := agent.NewAgentSession(rt)
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	return NewAppModel(AppOptions{
		Session:  sess,
		Settings: config.DefaultSettings(),
		Theme:    DarkTheme,
		Width:    width,
		Height:   height,
	})
}

func TestNewAppModel_Defaults(t *testing.T) {
	m := newTestApp(t, 0, 0)
	if m.width != 80 || m.height != 24 {
		t.Errorf("dimensions = %dx%d, want 80x24", m.width, m.height)
	}
	if m.theme.Name != "dark" {
		t.Errorf("theme.Name = %q, want 'dark'", m.theme.Name)
	}
	if m.inputHeight != 3 {
		t.Errorf("inputHeight = %d, want 3", m.inputHeight)
	}
	if m.footerHeight != components.FooterHeightLines {
		t.Errorf("footerHeight = %d, want %d", m.footerHeight, components.FooterHeightLines)
	}
	if m.footer == nil {
		t.Error("footer is nil after construction")
	}
	if len(m.slashCommands) == 0 {
		t.Error("slashCommands is empty")
	}
}

func TestAppModel_InitReturnsCmd(t *testing.T) {
	m := newTestApp(t, 80, 24)
	cmd := m.Init()
	if cmd == nil {
		t.Error("Init returned nil cmd; want focus command")
	}
}

func TestAppModel_ViewNonEmpty(t *testing.T) {
	m := newTestApp(t, 80, 24)
	out := m.View()
	if out == "" {
		t.Error("View returned empty string")
	}
}

func TestAppModel_WindowSizeMsg(t *testing.T) {
	m := newTestApp(t, 80, 24)
	model, cmd := m.Update(tea.WindowSizeMsg{Width: 120, Height: 40})
	if cmd != nil {
		t.Errorf("WindowSizeMsg cmd = %v, want nil", cmd)
	}
	app, ok := model.(*AppModel)
	if !ok {
		t.Fatalf("model = %T, want *AppModel", model)
	}
	if app.width != 120 || app.height != 40 {
		t.Errorf("after resize: %dx%d, want 120x40", app.width, app.height)
	}
}

func TestAppModel_HelpSlashCommand(t *testing.T) {
	m := newTestApp(t, 80, 24)
	m.handleSlashCommand("/help")
	if m.viewport.ElementCount() == 0 {
		t.Error("/help did not append to viewport")
	}
}

func TestAppModel_ClearSlashCommand(t *testing.T) {
	m := newTestApp(t, 80, 24)
	// Seed the viewport so /clear has something to remove.
	m.viewport.AppendText("user", "hello")
	m.viewport.AppendText("assistant", "world")
	if m.viewport.ElementCount() != 2 {
		t.Fatalf("seed: ElementCount = %d, want 2", m.viewport.ElementCount())
	}
	leafBefore := m.session.Runtime().State.LeafID()

	m.input.SetValue("ignored")
	m.handleSlashCommand("/clear")
	if m.input.Value() != "" {
		t.Errorf("after /clear: Value = %q, want empty", m.input.Value())
	}
	// /clear clears the viewport then appends the success message, so
	// exactly one element (the recovery hint) survives.
	if m.viewport.ElementCount() != 1 {
		t.Fatalf("after /clear: ElementCount = %d, want 1 (success message)", m.viewport.ElementCount())
	}
	// The success message must mention both the archived leaf and
	// /checkout so the user knows how to recover the prior context.
	view := m.View()
	if !strings.Contains(view, "Cleared") {
		t.Errorf("after /clear: view missing 'Cleared': %q", view)
	}
	if !strings.Contains(view, "/checkout") {
		t.Errorf("after /clear: view missing '/checkout' hint: %q", view)
	}
	if !strings.Contains(view, leafBefore) {
		t.Errorf("after /clear: view missing archived leaf id %q: %q", leafBefore, view)
	}
	// The leaf pointer advanced (ClearMarker appended as child).
	if leafAfter := m.session.Runtime().State.LeafID(); leafAfter == leafBefore {
		t.Errorf("after /clear: leaf unchanged = %q, want advanced", leafAfter)
	}
}

// TestAppModel_ClearResetsScroll verifies the ErrContextReset dispatch
// snaps the viewport back to the top so the scroll offset does not
// refer to entries that no longer exist.
func TestAppModel_ClearResetsScroll(t *testing.T) {
	m := newTestApp(t, 80, 24)
	m.viewport.AppendText("user", strings.Repeat("line\n", 50))
	// ScrollUp unconditionally disables autoScroll (ScrollDown would
	// re-enable it if we happened to land at the bottom).
	m.viewport.ScrollUp()
	if m.viewport.AutoScroll() {
		t.Fatalf("precondition: AutoScroll should be false after ScrollUp")
	}

	m.handleSlashCommand("/clear")
	if !m.viewport.AutoScroll() {
		t.Errorf("after /clear: AutoScroll = false, want true (ResetScroll re-enables)")
	}
}

// TestAppModel_ClsSlashCommand verifies /cls clears the viewport
// WITHOUT mutating session state: LeafID is unchanged before/after.
func TestAppModel_ClsSlashCommand(t *testing.T) {
	m := newTestApp(t, 80, 24)
	m.viewport.AppendText("user", "hello")
	m.viewport.AppendText("assistant", "world")
	if m.viewport.ElementCount() != 2 {
		t.Fatalf("seed: ElementCount = %d, want 2", m.viewport.ElementCount())
	}
	leafBefore := m.session.Runtime().State.LeafID()

	m.input.SetValue("ignored")
	m.handleSlashCommand("/cls")
	if m.input.Value() != "" {
		t.Errorf("after /cls: Value = %q, want empty", m.input.Value())
	}
	if m.viewport.ElementCount() != 0 {
		t.Errorf("after /cls: ElementCount = %d, want 0 (no success message for /cls)", m.viewport.ElementCount())
	}
	// /cls performs NO state mutation.
	if leafAfter := m.session.Runtime().State.LeafID(); leafAfter != leafBefore {
		t.Errorf("after /cls: leaf changed: %q → %q (state must be untouched)", leafBefore, leafAfter)
	}
}

func TestAppModel_TreeSlashCommand(t *testing.T) {
	m := newTestApp(t, 80, 24)
	if m.treeView.Visible() {
		t.Fatal("treeView should start hidden")
	}
	m.handleSlashCommand("/tree")
	if !m.treeView.Visible() {
		t.Error("/tree should show the tree view")
	}
}

func TestAppModel_ModelSlashCommand_Prints(t *testing.T) {
	m := newTestApp(t, 80, 24)
	m.handleSlashCommand("/model")
	if m.viewport.ElementCount() == 0 {
		t.Error("/model did not append output to viewport")
	}
}

func TestAppModel_UnknownSlashCommand(t *testing.T) {
	m := newTestApp(t, 80, 24)
	m.handleSlashCommand("/nosuch")
	if m.viewport.ElementCount() == 0 {
		t.Error("unknown command should append diagnostic to viewport")
	}
}

func TestAppModel_SubmitEmptyReturnsNil(t *testing.T) {
	m := newTestApp(t, 80, 24)
	cmd := m.submitInput()
	if cmd != nil {
		t.Errorf("submitInput with empty text: cmd = %v, want nil", cmd)
	}
}

func TestAppModel_HandleTextDelta(t *testing.T) {
	m := newTestApp(t, 80, 24)
	m.handleAgentEvent(agent.MessageUpdateEvent{
		When:  time.Now(),
		Delta: llm.TextDelta{ContentIndex: 0, Text: "Hello "},
	})
	m.handleAgentEvent(agent.MessageUpdateEvent{
		When:  time.Now(),
		Delta: llm.TextDelta{ContentIndex: 0, Text: "world"},
	})
	if got := m.assistantBuf.String(); got != "Hello world" {
		t.Errorf("assistantBuf = %q, want 'Hello world'", got)
	}
	// flushAssistantText pushes the buffer to the viewport.
	m.flushAssistantText()
	if m.viewport.ElementCount() == 0 {
		t.Error("no elements appended after flush")
	}
}

func TestAppModel_HandleToolCallAndResult(t *testing.T) {
	m := newTestApp(t, 80, 24)
	args, _ := json.Marshal(map[string]string{"path": "/tmp/x"})
	m.handleAgentEvent(agent.ToolCallEvent{
		Call: tools.ToolCall{ID: "t1", Name: "read", Args: args},
	})
	if m.viewport.ElementCount() != 1 {
		t.Fatalf("viewport elements = %d, want 1", m.viewport.ElementCount())
	}
	el, ok := m.pendingTools["t1"]
	if !ok {
		t.Fatal("pendingTools[t1] not set")
	}
	if el.Running != true {
		t.Error("tool element should be running")
	}
	if el.Name != "read" {
		t.Errorf("Name = %q, want 'read'", el.Name)
	}

	m.handleAgentEvent(agent.ToolResultEvent{
		ToolID: "t1",
		Result: tools.ToolResult{
			Content: []llm.ContentBlock{llm.TextContent{Text: "file contents"}},
		},
	})
	if el.Running {
		t.Error("tool element should not be running after result")
	}
	if el.Result != "file contents" {
		t.Errorf("Result = %q, want 'file contents'", el.Result)
	}
}

func TestAppModel_HandleToolResultError(t *testing.T) {
	m := newTestApp(t, 80, 24)
	m.handleAgentEvent(agent.ToolCallEvent{
		Call: tools.ToolCall{ID: "e1", Name: "bash"},
	})
	m.handleAgentEvent(agent.ToolResultEvent{
		ToolID: "e1",
		Result: tools.ToolResult{
			IsError: true,
			Content: []llm.ContentBlock{llm.TextContent{Text: "boom"}},
		},
	})
	el := m.pendingTools["e1"]
	if !el.IsError {
		t.Error("IsError should be true")
	}
	if el.Result != "boom" {
		t.Errorf("Result = %q, want 'boom'", el.Result)
	}
}

func TestAppModel_RequestQuitClosesChannel(t *testing.T) {
	quit := make(chan struct{})
	m := newTestApp(t, 80, 24)
	m.quitCh = quit
	m.requestQuit()
	select {
	case <-quit:
	default:
		t.Error("quit channel not closed after requestQuit")
	}
}

func TestExtractTextFromResult(t *testing.T) {
	blocks := []llm.ContentBlock{
		llm.TextContent{Text: "part1 "},
		llm.TextContent{Text: "part2"},
	}
	if got := ExtractTextFromResult(blocks); got != "part1 part2" {
		t.Errorf("ExtractTextFromResult = %q, want 'part1 part2'", got)
	}
}

func TestAppModel_DiagnosticsFromKeybindConflicts(t *testing.T) {
	// Build settings with a deliberate conflict: same key bound to two
	// actions. ResolveWithConflicts should produce a diagnostic.
	settings := config.DefaultSettings()
	settings.Keybindings = map[string]config.Keybinding{
		ActionSubmit: {"ctrl+j"},
		ActionAbort:  {"ctrl+j"}, // same as submit
	}

	client := fauxprovider.NewWithResponse("r")
	opts := agent.SessionOptions{
		Model:     "faux",
		Settings:  settings,
		LLMClient: client,
		Tools:     []tools.HeadlessTool{tools.NewReadTool(tools.OSReadOperations{})},
	}
	rt, err := agent.CreateAgentSessionRuntime(context.Background(), t.TempDir(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSessionRuntime: %v", err)
	}
	sess := agent.NewAgentSession(rt)
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	m := NewAppModel(AppOptions{
		Session:  sess,
		Settings: settings,
		Theme:    DarkTheme,
		Width:    80,
		Height:   24,
	})
	if len(m.diagnostics) == 0 {
		t.Error("expected keybind-conflict diagnostics; got none")
	}
}

// --- 10.11 smoke tests: layout at multiple dimensions + resize mid-stream ---

func TestAppModel_RendersAt80x24(t *testing.T) {
	m := newTestApp(t, 80, 24)
	if m.footer == nil {
		t.Error("80-wide should still construct a footer")
	}
	out := m.View()
	if out == "" {
		t.Error("View empty at 80x24")
	}
}

func TestAppModel_RendersAt120x40(t *testing.T) {
	m := newTestApp(t, 120, 40)
	if m.footer == nil {
		t.Error("120-wide should still construct a footer")
	}
	if m.height != 40 {
		t.Errorf("height = %d, want 40", m.height)
	}
	out := m.View()
	if out == "" {
		t.Error("View empty at 120x40")
	}
}

func TestAppModel_RendersAt200x50(t *testing.T) {
	m := newTestApp(t, 200, 50)
	out := m.View()
	if out == "" {
		t.Error("View empty at 200x50")
	}
}

// TestAppModel_RendersAtNarrowWidth replaces the old "below sidebar
// threshold" test. With the sidebar gone there is no threshold, but
// the layout must still render something sensible on a narrow terminal.
func TestAppModel_RendersAtNarrowWidth(t *testing.T) {
	m := newTestApp(t, 40, 24)
	if m.footer == nil {
		t.Error("footer should still be constructed on narrow terminals")
	}
	out := m.View()
	if out == "" {
		t.Error("View empty at 40x24")
	}
}

func TestAppModel_ResizeMidStream(t *testing.T) {
	// Start at 80x24, simulate in-flight assistant text, then resize
	// to 120x40 and verify the layout adapts without losing buffered
	// content.
	m := newTestApp(t, 80, 24)
	m.handleAgentEvent(agent.MessageUpdateEvent{
		When:  time.Now(),
		Delta: llm.TextDelta{ContentIndex: 0, Text: "partial response"},
	})
	if got := m.assistantBuf.String(); got != "partial response" {
		t.Fatalf("buffer before resize = %q", got)
	}

	// Resize via WindowSizeMsg — the same path bubbletea uses.
	model, _ := m.Update(tea.WindowSizeMsg{Width: 120, Height: 40})
	m = model.(*AppModel)
	if m.width != 120 || m.height != 40 {
		t.Errorf("after resize: %dx%d, want 120x40", m.width, m.height)
	}
	// Buffered assistant text survives the resize.
	if got := m.assistantBuf.String(); got != "partial response" {
		t.Errorf("buffer after resize = %q, want 'partial response'", got)
	}
}

func TestAppModel_QuitKeybindingExitsCleanly(t *testing.T) {
	// Submit /quit; requestQuit closes the quit channel and returns
	// tea.Quit, which bubbletea uses to exit the program.
	m := newTestApp(t, 80, 24)
	quit := make(chan struct{})
	m.quitCh = quit

	cmd := m.handleSlashCommand("/quit")
	if cmd == nil {
		t.Fatal("/quit returned nil cmd; want tea.Quit")
	}
	// cmd is tea.Quit; executing it returns tea.QuitMsg. We just verify
	// the channel closed.
	select {
	case <-quit:
	default:
		t.Error("quit channel not closed after /quit")
	}
}

// TestAppModel_FooterReflectsRuntime verifies the footer is populated
// from the runtime: cwd, model id, and the context window all appear
// in the rendered View().
func TestAppModel_FooterReflectsRuntime(t *testing.T) {
	m := newTestApp(t, 80, 24)
	out := m.View()

	// Model id from rt.Options.Model.
	if !strings.Contains(out, "faux") {
		t.Errorf("footer missing model id 'faux': %q", out)
	}
	// Context window from rt.Options.ContextWindow (200000 → "200k").
	if !strings.Contains(out, "200k") {
		t.Errorf("footer missing context window '200k': %q", out)
	}
	// Cwd from rt.Cwd. The footer substitutes a $HOME prefix with "~",
	// so accept either the raw cwd or the "~"-prefixed tail.
	cwd := m.session.Runtime().Cwd
	if cwd != "" {
		tail := cwd
		if home, err := os.UserHomeDir(); err == nil && home != "" && strings.HasPrefix(cwd, home) {
			tail = "~" + cwd[len(home):]
		}
		if !strings.Contains(out, cwd) && !strings.Contains(out, tail) {
			t.Errorf("footer missing cwd (%q or %q): %q", cwd, tail, out)
		}
	}
}
