// clear.go — /clear command (state-resetting).
//
// /clear appends a ClearMarker entry as a child of the current leaf. The
// state manager advances its leaf pointer to the new ClearMarker entry.
// BuildContext treats KindClearMarker as a hard context barrier: when
// walking leaf → root, the first ClearMarker encountered truncates the
// walk; entries older than it are excluded from the model's context.
//
// Subsequent appends descend from the ClearMarker, so post-clear
// conversation lives entirely in the new branch. The old context remains
// reachable via /checkout <oldLeafID>: walking from the pre-clear leaf
// never visits the ClearMarker (it's a descendant, not an ancestor), so
// the original context is fully restored.
//
// The actual viewport clear / scroll reset / input reset is delegated to
// the caller via ErrContextReset; this command does not have references
// to the viewport, scroll, or input components.
package slash

import (
	"context"
	"errors"
	"fmt"

	"github.com/coevin/tau/internal/agent"
	"github.com/coevin/tau/internal/state"
)

type clearCommand struct{}

func newClearCommand() Command { return clearCommand{} }

func (clearCommand) Name() string { return "/clear" }

func (clearCommand) ShortHelp() string {
	return "Reset the conversation: archive current context and start fresh. Use /checkout to return"
}

func (clearCommand) Execute(_ context.Context, _ string, session agent.CommandSession) (string, error) {
	if session == nil {
		return "", errors.New("/clear: session is nil")
	}
	rt := session.Runtime()
	if rt == nil || rt.State() == nil {
		return "", errors.New("/clear: runtime or state is nil")
	}
	oldLeaf := rt.State().LeafID()
	newID, err := rt.State().Append(state.Entry{
		Payload: state.ClearMarkerPayload{},
	})
	if err != nil {
		return "", fmt.Errorf("/clear: append clear marker: %w", err)
	}
	return fmt.Sprintf(
		"Cleared. Previous context archived under leaf %s; use /checkout %s to return.",
		newID, oldLeaf,
	), ErrContextReset
}
