// quit.go — /quit command.
//
// /quit signals the caller to gracefully shut down the program. The
// actual shutdown sequencing (closing channels, calling session.Run
// a final time if needed) is owned by the caller; this command just
// returns ErrQuitRequested.
package slash

import (
	"context"

	"github.com/taucentral/tau/internal/agent"
)

type quitCommand struct{}

func newQuitCommand() Command { return quitCommand{} }

func (quitCommand) Name() string      { return "/quit" }
func (quitCommand) ShortHelp() string { return "Exit tau" }

func (quitCommand) Execute(_ context.Context, _ string, _ agent.CommandSession) (string, error) {
	return "", ErrQuitRequested
}
