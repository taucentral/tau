// Package slash implements tau's slash-command registry.
//
// Slash commands are user-invoked directives typed into the input
// textarea: /fork, /checkout, /tree, /compact, /model, /label, /clear,
// /help, /quit. Each command implements the Command interface; the
// Registry dispatches by name and aggregates the list for /help and
// for the TUI's autocomplete overlay.
//
// The package depends only on internal/agent and standard library
// types so it can be reused by the TUI, the RPC server, and tests
// without cycles.
package slash

import (
	"context"
	"errors"
	"sort"
	"strings"

	"github.com/taucentral/tau/internal/agent"
)

// Command is the interface every slash command implements. Execute
// receives the raw argument string (everything after the command name,
// trimmed of leading whitespace) and the wired session.
//
// Execute returns the text to display to the user (rendered as a
// system-role viewport element) and/or a typed sentinel error from
// this package. Non-sentinel errors are surfaced as red diagnostic
// text in the viewport.
type Command interface {
	// Name is the canonical invocation including the leading slash
	// (e.g. "/fork", "/checkout"). Lowercase, no whitespace.
	Name() string

	// ShortHelp is a one-line description for /help. Should be brief
	// enough to fit on a single terminal line; no trailing period.
	ShortHelp() string

	// Execute runs the command. ctx is the agent context; args is the
	// trimmed argument text after the command name (may be empty);
	// session is the CommandSession view over the wired agent session
	// the command operates on. Internal callers obtain the view via
	// (*agent.AgentSession).AsCommandSession(); external plugin
	// commands see only the agent.CommandSession interface (re-exported
	// by pkg/tau/sdk.go as tau.CommandSession) and never the concrete
	// *agent.AgentSession, which they cannot name.
	Execute(ctx context.Context, args string, session agent.CommandSession) (string, error)
}

// Sentinel errors used to signal side effects outside the slash package
// (clearing the viewport, resetting context, quitting the program, opening
// the tree view). Callers MUST type-check via errors.Is.
var (
	// ErrQuitRequested signals that the user invoked /quit and the
	// caller should close the program's quit channel.
	ErrQuitRequested = errors.New("slash: quit requested")

	// ErrClearScreen signals that the caller should clear the
	// conversational viewport WITHOUT mutating session state. Returned
	// by /cls. The state tree on disk is untouched.
	ErrClearScreen = errors.New("slash: clear screen")

	// ErrContextReset signals that the caller should clear the
	// conversational viewport, reset scroll position, and reset the
	// input buffer. Returned by /clear, which appends a ClearMarker
	// entry before signalling. The caller clears the viewport FIRST,
	// then renders the success message (returned alongside the error)
	// so the recovery hint (/checkout <oldLeafID>) survives as the
	// sole viewport element.
	ErrContextReset = errors.New("slash: context reset")

	// ErrShowTree signals that the caller should open the tree-view
	// overlay. Returned by /tree.
	ErrShowTree = errors.New("slash: show tree")
)

// ErrClearViewport is preserved as a deprecated alias for ErrClearScreen so
// errors.Is(err, ErrClearViewport) continues to match. The alias is the
// SAME error value as ErrClearScreen (var assignment), so errors.Is
// against either name matches both. Use ErrClearScreen in new code.
//
// Deprecated: use ErrClearScreen.
var ErrClearViewport = ErrClearScreen

// Registry holds the set of registered commands keyed by Name().
type Registry struct {
	commands map[string]Command
}

// NewRegistry returns an empty registry.
func NewRegistry() *Registry {
	return &Registry{commands: make(map[string]Command)}
}

// Register adds c to the registry. Registering a duplicate name
// overwrites the previous registration; the caller is responsible for
// not registering the same name twice.
func (r *Registry) Register(c Command) {
	r.commands[c.Name()] = c
}

// Lookup returns the command with the given name (including the leading
// slash) and reports whether it exists.
func (r *Registry) Lookup(name string) (Command, bool) {
	c, ok := r.commands[name]
	return c, ok
}

// Names returns all registered command names (including leading slash),
// sorted alphabetically. Used by /help and the autocomplete overlay.
func (r *Registry) Names() []string {
	out := make([]string, 0, len(r.commands))
	for name := range r.commands {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// All returns all registered commands sorted alphabetically by name.
func (r *Registry) All() []Command {
	names := r.Names()
	out := make([]Command, 0, len(names))
	for _, n := range names {
		out = append(out, r.commands[n])
	}
	return out
}

// Parse splits an input line into a slash-command name and its argument
// string. Returns ok=false when input is not a slash command (does not
// start with "/") or when the command name is empty (e.g. just "/").
//
// The returned name always starts with "/". The args string is the
// remainder of the input with leading whitespace removed; trailing
// whitespace is preserved so commands can distinguish "/label foo "
// from "/label foo".
func Parse(input string) (name, args string, ok bool) {
	trimmed := strings.TrimLeft(input, " \t")
	if !strings.HasPrefix(trimmed, "/") {
		return "", "", false
	}
	rest := trimmed[1:]
	if rest == "" {
		return "", "", false
	}
	// Split on the first whitespace run.
	spaceIdx := strings.IndexAny(rest, " \t")
	if spaceIdx < 0 {
		return "/" + rest, "", true
	}
	name = "/" + rest[:spaceIdx]
	args = strings.TrimLeft(rest[spaceIdx:], " \t")
	return name, args, true
}

// Execute parses input, looks up the command, and runs it. Returns
// ErrUnknownCommand when input looks like a slash command but the name
// is not registered. Returns ErrNotASlashCommand when input does not
// start with "/" (the caller should treat it as ordinary input).
func (r *Registry) Execute(ctx context.Context, input string, session agent.CommandSession) (string, error) {
	name, args, ok := Parse(input)
	if !ok {
		return "", ErrNotASlashCommand
	}
	cmd, found := r.commands[name]
	if !found {
		return "", ErrUnknownCommand
	}
	return cmd.Execute(ctx, args, session)
}

// ErrUnknownCommand is returned by Registry.Execute when the parsed
// command name is not registered.
var ErrUnknownCommand = errors.New("slash: unknown command")

// ErrNotASlashCommand is returned by Registry.Execute when the input
// does not parse as a slash command. The caller should treat the input
// as ordinary user text.
var ErrNotASlashCommand = errors.New("slash: not a slash command")

// DefaultRegistry returns a Registry populated with every built-in
// command. This is the registry the TUI and RPC server use by default.
func DefaultRegistry() *Registry {
	r := NewRegistry()
	help := &helpCommand{}
	for _, c := range []Command{
		newForkCommand(),
		newCheckoutCommand(),
		newTreeCommand(),
		newCompactCommand(),
		newModelCommand(),
		newLabelCommand(),
		newClearCommand(),
		newClsCommand(),
		help,
		newQuitCommand(),
	} {
		r.Register(c)
	}
	// Late-bind the registry reference so /help can list its siblings.
	help.registry = r
	return r
}
