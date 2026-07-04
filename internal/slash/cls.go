// cls.go — /cls command (viewport-only clear).
//
// /cls empties the conversational viewport in the TUI. It does NOT mutate
// the session state tree — every entry remains on disk and in the
// in-memory manager. /cls is purely a UI affordance for users who want
// visual de-cluttering without losing context. Use /clear for state reset.
//
// The actual clearing is delegated to the caller via ErrClearScreen; this
// command does not have a reference to the viewport.
package slash

import (
	"context"

	"github.com/coevin/tau/internal/agent"
)

type clsCommand struct{}

func newClsCommand() Command { return clsCommand{} }

func (clsCommand) Name() string { return "/cls" }

func (clsCommand) ShortHelp() string {
	return "Clear the conversational viewport (history is preserved on disk)"
}

func (clsCommand) Execute(_ context.Context, _ string, _ agent.CommandSession) (string, error) {
	// Return ErrClearScreen; the caller clears the viewport. No session
	// access required — /cls works even with a nil session (e.g. when
	// the agent loop has not yet wired one up).
	return "", ErrClearScreen
}
