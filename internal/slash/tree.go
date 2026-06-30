// tree.go — /tree command.
//
// /tree opens the TUI's tree-view overlay. The actual rendering is
// done by the TUI layer; this command just signals via ErrShowTree so
// the caller (the TUI's Update loop) knows to call treeView.Show().
package slash

import (
	"context"
	"errors"

	"github.com/coevin/tau/internal/agent"
)

type treeCommand struct{}

func newTreeCommand() Command { return treeCommand{} }

func (treeCommand) Name() string      { return "/tree" }
func (treeCommand) ShortHelp() string { return "Open the branch tree viewer" }

func (treeCommand) Execute(_ context.Context, _ string, session agent.CommandSession) (string, error) {
	if session == nil {
		return "", errors.New("/tree: session is nil")
	}
	// Return ErrShowTree; the caller opens the overlay.
	return "", ErrShowTree
}
