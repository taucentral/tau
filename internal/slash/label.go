// label.go — /label <text> command.
//
// /label attaches a short text label to the current leaf entry. The
// label is stored as a LabelPayload entry so the tree viewer can render
// it next to the labelled node.
package slash

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/coevin/tau/internal/agent"
	"github.com/coevin/tau/internal/state"
)

type labelCommand struct{}

func newLabelCommand() Command { return labelCommand{} }

func (labelCommand) Name() string      { return "/label" }
func (labelCommand) ShortHelp() string { return "Attach a text label to the current leaf" }

func (labelCommand) Execute(_ context.Context, args string, session *agent.AgentSession) (string, error) {
	if session == nil {
		return "", errors.New("/label: session is nil")
	}
	text := strings.TrimSpace(args)
	if text == "" {
		return "", errors.New("/label: usage: /label <text>")
	}
	rt := session.Runtime()
	if _, err := rt.State.Append(state.Entry{
		Kind:    state.KindLabel,
		Payload: state.LabelPayload{Label: text},
	}); err != nil {
		return "", fmt.Errorf("/label: %w", err)
	}
	return fmt.Sprintf("labelled leaf %s: %s", rt.State.LeafID(), text), nil
}
