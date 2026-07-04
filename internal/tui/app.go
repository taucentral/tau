// Package tui — app.go — top-level bubbletea Model.
//
// AppModel ties the viewport, footer, input, and tree-view components
// into a single vertical-stack layout. It subscribes to the agent's
// event bus via a goroutine that forwards each event as a tea.Msg so
// the Elm-architecture Update loop can react.
package tui

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/coevin/tau/internal/agent"
	"github.com/coevin/tau/internal/config"
	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/slash"
	"github.com/coevin/tau/internal/tui/components"
)

// AppOptions configures AppModel construction.
type AppOptions struct {
	Session  *agent.AgentSession
	Settings config.Settings
	Theme    Theme
	Width    int
	Height   int
	AgentDir string
	QuitCh   chan struct{} // closed when the user requests quit
}

// AppModel is the root bubbletea Model for the interactive TUI.
type AppModel struct {
	session  *agent.AgentSession
	settings config.Settings
	theme    Theme
	keys     Keybindings
	viewport *components.Viewport
	footer   *components.Footer
	input    *components.InputModel
	treeView *components.TreeView
	width    int
	height   int

	// Layout constants.
	inputHeight  int
	footerHeight int

	// State.
	running    bool // a turn is in progress
	turnCancel context.CancelFunc
	quitCh     chan struct{}
	mu         sync.Mutex // guards running and turnCancel

	// Accumulated assistant text for the current streaming response.
	assistantBuf strings.Builder

	// Pending tool calls keyed by tool-use ID.
	pendingTools map[string]*components.ToolElement

	// Registered slash commands for autocomplete.
	slashCommands []string
	slashRegistry *slash.Registry

	// Diagnostic messages to show (e.g., keybind conflicts).
	diagnostics []string
}

// eventMsg wraps an agent.Event for delivery through bubbletea's
// Update loop.
type eventMsg struct{ evt agent.Event }

// turnEndMsg signals that a turn has completed.
type turnEndMsg struct{ err error }

// AppModel implements tea.Model.
var _ tea.Model = (*AppModel)(nil)

// NewAppModel returns a fully-wired AppModel. Call Init() to get the
// initial bubbletea command, then drive it with the program loop.
func NewAppModel(opts AppOptions) *AppModel {
	theme := opts.Theme
	if theme.Name == "" {
		theme = DarkTheme
	}
	width := opts.Width
	if width <= 0 {
		width = 80
	}
	height := opts.Height
	if height <= 0 {
		height = 24
	}

	inputHeight := 3
	footerHeight := components.FooterHeightLines
	mainHeight := height - inputHeight - footerHeight - 1 // -1 for the separator
	mainWidth := width                                    // viewport spans the full width

	keys, conflicts := ResolveWithConflicts(opts.Settings.Keybindings)
	diags := ConflictsAsDiagnostics(conflicts)

	registry := slash.DefaultRegistry()

	m := &AppModel{
		session:       opts.Session,
		settings:      opts.Settings,
		theme:         theme,
		keys:          keys,
		viewport:      components.NewViewport(theme, mainWidth, mainHeight),
		footer:        components.NewFooter(theme, width),
		input:         components.NewInput(theme, width, inputHeight),
		treeView:      components.NewTreeView(theme, width, height),
		width:         width,
		height:        height,
		inputHeight:   inputHeight,
		footerHeight:  footerHeight,
		quitCh:        opts.QuitCh,
		pendingTools:  make(map[string]*components.ToolElement),
		slashRegistry: registry,
		slashCommands: registry.Names(),
		diagnostics:   diags,
	}
	m.input.SetCommands(m.slashCommands)

	// Populate footer from session metadata. Cwd, model, provider tag,
	// and thinking level are static across the session; tokens update
	// dynamically via UsageDelta events in handleAgentEvent.
	rt := opts.Session.Runtime()
	m.footer.SetCwd(rt.Cwd)
	m.footer.SetModel(rt.Options.Model)
	m.footer.SetProviderAPI(string(rt.Options.ProviderAPI))
	m.footer.SetThinkingLevel(string(rt.Options.ThinkingLevel))
	m.footer.SetTokens(0, rt.Options.ContextWindow)

	return m
}

// Init implements tea.Model. It focuses the input textarea.
func (m *AppModel) Init() tea.Cmd {
	return m.input.Focus()
}

// Update implements tea.Model.
func (m *AppModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.handleResize(msg.Width, msg.Height)
		return m, nil

	case tea.KeyMsg:
		return m.handleKey(msg)

	case eventMsg:
		m.handleAgentEvent(msg.evt)
		return m, nil

	case turnEndMsg:
		m.finishTurn(msg.err)
		return m, nil
	}

	// Forward to sub-components.
	model, cmd := m.input.Update(msg)
	m.input = model.(*components.InputModel)
	return m, cmd
}

// View implements tea.Model.
func (m *AppModel) View() string {
	// Tree view overlay takes priority when visible.
	if m.treeView.Visible() {
		return m.treeView.View()
	}

	// Full-width conversation viewport. The right-hand sidebar that
	// used to live here is gone — metadata moved to the footer.
	content := m.viewport.View()

	// Separator between content and input.
	separator := lipgloss.NewStyle().Foreground(m.theme.Muted).Render(strings.Repeat("─", m.width))

	// Input area.
	inputView := m.input.View()

	// Diagnostics overlay (e.g., keybind-conflict warnings from startup).
	var overlay string
	if len(m.diagnostics) > 0 {
		overlay = m.theme.WarningStyle().Render(strings.Join(m.diagnostics, "\n"))
	}

	// Footer (cwd + tokens + model) anchors the bottom of the stack.
	footerView := m.footer.View()

	if overlay != "" {
		return lipgloss.JoinVertical(lipgloss.Left,
			content,
			separator,
			overlay,
			inputView,
			footerView,
		)
	}

	return lipgloss.JoinVertical(lipgloss.Left,
		content,
		separator,
		inputView,
		footerView,
	)
}

// handleResize updates component dimensions on terminal resize.
func (m *AppModel) handleResize(width, height int) {
	m.width = width
	m.height = height

	mainHeight := height - m.inputHeight - m.footerHeight - 1 // -1 for the separator
	mainWidth := width                                        // viewport spans the full width

	m.viewport.SetDimensions(mainWidth, mainHeight)
	m.footer.SetWidth(width)
	m.input.SetDimensions(width, m.inputHeight)
	m.treeView.SetDimensions(width, height)
}

// handleKey dispatches key events to the appropriate action based on
// the configured keybindings.
func (m *AppModel) handleKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	keyStr := msg.String()

	// Tree view has priority when visible.
	if m.treeView.Visible() {
		switch {
		case m.keys.Matches(ActionTreeUp, keyStr):
			m.treeView.CursorUp()
			return m, nil
		case m.keys.Matches(ActionTreeDown, keyStr):
			m.treeView.CursorDown()
			return m, nil
		case m.keys.Matches(ActionTreeOpen, keyStr):
			sel := m.treeView.Selected()
			if sel != nil {
				// Checkout of the selected entry is wired by the slash
				// command layer when it is registered. For now, hide
				// the tree and let the user continue.
				m.treeView.Hide()
			}
			return m, nil
		case m.keys.Matches(ActionQuit, keyStr) || keyStr == "esc":
			m.treeView.Hide()
			return m, nil
		}
		return m, nil
	}

	// Global keybindings.
	switch {
	case m.keys.Matches(ActionQuit, keyStr):
		m.requestQuit()
		return m, tea.Quit

	case m.keys.Matches(ActionAbort, keyStr):
		m.cancelTurn()
		return m, nil

	case m.keys.Matches(ActionSubmit, keyStr):
		return m, m.submitInput()

	case m.keys.Matches(ActionScrollUp, keyStr):
		m.viewport.ScrollUp()
		return m, nil

	case m.keys.Matches(ActionScrollDown, keyStr):
		m.viewport.ScrollDown()
		return m, nil
	}

	// Slash command autocomplete.
	if strings.HasPrefix(m.input.Value(), "/") {
		switch keyStr {
		case "tab":
			m.input.AutocompleteAccept()
			return m, nil
		case "up", "down":
			if m.input.AutocompleteActive() {
				m.input.AutocompleteNext()
				return m, nil
			}
		}
	}

	// Default: forward to the input textarea.
	model, cmd := m.input.Update(msg)
	m.input = model.(*components.InputModel)
	// Trigger autocomplete update after each keystroke.
	if m.input.Value() != "" && strings.HasPrefix(m.input.Value(), "/") {
		m.input.AutocompleteStart()
	}
	return m, cmd
}

// submitInput reads the textarea value and launches a turn.
func (m *AppModel) submitInput() tea.Cmd {
	text := strings.TrimSpace(m.input.Value())
	if text == "" {
		return nil
	}

	// Slash commands are dispatched via the registry once registered;
	// until then /quit, /clear, and /help are handled inline.
	if strings.HasPrefix(text, "/") {
		return m.handleSlashCommand(text)
	}

	// Add user message to the viewport.
	m.viewport.AppendText("user", text)

	// Save to history and reset input.
	m.input.AppendHistory(text)
	m.input.Reset()

	// Start the turn.
	return m.startTurn(text)
}

// handleSlashCommand dispatches a slash command via the registry.
// Sentinel returns drive UI side effects (clear viewport, reset context,
// show tree, request quit); ordinary output is rendered as a system message.
func (m *AppModel) handleSlashCommand(input string) tea.Cmd {
	out, err := m.slashRegistry.Execute(context.Background(), input, m.session.AsCommandSession())
	m.input.Reset()
	switch {
	case errors.Is(err, slash.ErrQuitRequested):
		m.requestQuit()
		return tea.Quit
	case errors.Is(err, slash.ErrContextReset):
		// /clear returns BOTH a success message and ErrContextReset.
		// Clear first, THEN render the success message so the recovery
		// hint (/checkout <oldLeafID>) survives as the sole viewport
		// element. Appending before the clear would wipe the message.
		m.viewport.Clear()
		m.viewport.ResetScroll()
		if strings.TrimSpace(out) != "" {
			m.viewport.AppendText("system", out)
		}
		return nil
	case errors.Is(err, slash.ErrClearScreen):
		// /cls: viewport-only clear. No state mutation, no scroll reset.
		// (ErrClearViewport alias routes here too — same error value.)
		m.viewport.Clear()
		return nil
	case errors.Is(err, slash.ErrShowTree):
		m.treeView.Show()
		return nil
	case err != nil:
		m.viewport.AppendText("system", fmt.Sprintf("error: %v", err))
		return nil
	default:
		if strings.TrimSpace(out) != "" {
			m.viewport.AppendText("system", out)
		}
		return nil
	}
}

// startTurn launches an agent turn. Events published by the agent loop
// arrive via SubscribeEvents (which forwards them as eventMsg using
// p.Send), so this command's only job is to run the turn to completion
// and return a turnEndMsg. Keeping event handling on the bubbletea main
// loop avoids racing with concurrent Update calls.
func (m *AppModel) startTurn(prompt string) tea.Cmd {
	m.mu.Lock()
	if m.running {
		m.mu.Unlock()
		m.viewport.AppendText("system", "a turn is already running; press Esc to abort")
		return nil
	}
	m.running = true
	m.mu.Unlock()

	m.assistantBuf.Reset()
	m.pendingTools = make(map[string]*components.ToolElement)

	ctx, cancel := context.WithCancel(context.Background())
	m.mu.Lock()
	m.turnCancel = cancel
	m.mu.Unlock()

	return func() tea.Msg {
		err := m.session.Run(ctx, prompt)
		return turnEndMsg{err: err}
	}
}

// handleAgentEvent translates an agent.Event into viewport updates.
// Called on the bubbletea main loop; never racey with itself.
func (m *AppModel) handleAgentEvent(evt agent.Event) {
	switch e := evt.(type) {
	case agent.MessageUpdateEvent:
		switch d := e.Delta.(type) {
		case llm.TextDelta:
			m.assistantBuf.WriteString(d.Text)
		case llm.ToolCallDelta:
			if d.Name != "" && d.ID != "" {
				m.flushAssistantText()
				el := &components.ToolElement{
					Name:    d.Name,
					Running: true,
				}
				m.pendingTools[d.ID] = el
				m.viewport.AppendElement(el)
			}
		case llm.UsageDelta:
			m.footer.SetTokens(d.InputTokens+d.OutputTokens, m.session.Runtime().Options.ContextWindow)
		}
	case agent.ToolCallEvent:
		m.flushAssistantText()
		el := &components.ToolElement{
			Name:    e.Call.Name,
			Args:    string(e.Call.Args),
			Running: true,
		}
		m.pendingTools[e.Call.ID] = el
		m.viewport.AppendElement(el)
	case agent.ToolResultEvent:
		if el, ok := m.pendingTools[e.ToolID]; ok {
			el.Running = false
			el.IsError = e.Result.IsError
			el.Result = ExtractTextFromResult(e.Result.Content)
		}
	case agent.TurnEndEvent:
		m.flushAssistantText()
	}
}

// flushAssistantText pushes accumulated text deltas as a single
// assistant message element.
func (m *AppModel) flushAssistantText() {
	if m.assistantBuf.Len() > 0 {
		m.viewport.AppendText("assistant", m.assistantBuf.String())
		m.assistantBuf.Reset()
	}
}

// finishTurn clears the running state.
func (m *AppModel) finishTurn(err error) {
	m.mu.Lock()
	m.running = false
	m.turnCancel = nil
	m.mu.Unlock()

	if err != nil {
		m.viewport.AppendText("system", fmt.Sprintf("turn error: %v", err))
	}
}

// cancelTurn cancels the in-flight turn.
func (m *AppModel) cancelTurn() {
	m.mu.Lock()
	cancel := m.turnCancel
	m.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

// requestQuit signals the quit channel and cancels any turn.
func (m *AppModel) requestQuit() {
	m.cancelTurn()
	if m.quitCh != nil {
		select {
		case <-m.quitCh:
		default:
			close(m.quitCh)
		}
	}
}

// SubscribeEvents starts a goroutine that forwards agent events to
// the bubbletea program. Call this once after the program is created
// so events from the agent loop arrive as eventMsg on the bubbletea
// main loop. The goroutine exits when the session shuts down (bus
// closes subscriber channels) or when done is closed.
func SubscribeEvents(p *tea.Program, session *agent.AgentSession, done <-chan struct{}) {
	bus := session.Runtime().EventBus
	ch := bus.Subscribe()
	go func() {
		for {
			select {
			case evt, ok := <-ch:
				if !ok {
					return
				}
				p.Send(eventMsg{evt: evt})
			case <-done:
				return
			}
		}
	}()
}

// ExtractTextFromResult concatenates text content blocks.
// This is a convenience exported for the app layer.
func ExtractTextFromResult(blocks []llm.ContentBlock) string {
	var sb strings.Builder
	for _, b := range blocks {
		if t, ok := b.(llm.TextContent); ok {
			sb.WriteString(t.Text)
		}
	}
	return sb.String()
}
