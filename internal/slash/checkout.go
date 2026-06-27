// checkout.go — /checkout <entryID> command.
//
// /checkout moves the active leaf to the named entry. The entry must
// exist in the current session tree; otherwise the command emits a
// diagnostic and leaves the leaf unchanged.
package slash

import (
	"context"
	"errors"
	"fmt"

	"github.com/coevin/tau/internal/agent"
)

type checkoutCommand struct{}

func newCheckoutCommand() Command { return checkoutCommand{} }

func (checkoutCommand) Name() string      { return "/checkout" }
func (checkoutCommand) ShortHelp() string { return "Move the active leaf to <entryID>" }

func (checkoutCommand) Execute(_ context.Context, args string, session *agent.AgentSession) (string, error) {
	if session == nil {
		return "", errors.New("/checkout: session is nil")
	}
	if args == "" {
		return "", errors.New("/checkout: usage: /checkout <entryID>")
	}
	rt := session.Runtime()
	if err := rt.State.SetLeaf(args); err != nil {
		return "", fmt.Errorf("/checkout: %w", err)
	}
	return fmt.Sprintf("checked out %s; leaf is now %s", args, rt.State.LeafID()), nil
}
