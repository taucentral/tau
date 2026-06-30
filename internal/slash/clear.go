// clear.go — /clear command.
//
// /clear empties the conversational viewport in the TUI. It does not
// mutate the session state tree — every entry remains on disk and in
// the in-memory manager. /clear is purely a UI affordance for users
// who want to reduce visual clutter without losing history.
//
// The actual clearing is delegated to the caller via ErrClearViewport;
// this command does not have a reference to the viewport.
package slash

import (
	"context"
	"errors"

	"github.com/coevin/tau/internal/agent"
)

type clearCommand struct{}

func newClearCommand() Command { return clearCommand{} }

func (clearCommand) Name() string { return "/clear" }
func (clearCommand) ShortHelp() string {
	return "Clear the conversational viewport (history is preserved on disk)"
}

func (clearCommand) Execute(_ context.Context, _ string, session agent.CommandSession) (string, error) {
	if session == nil {
		return "", errors.New("/clear: session is nil")
	}
	// Return ErrClearViewport; the caller clears the viewport.
	return "", ErrClearViewport
}
