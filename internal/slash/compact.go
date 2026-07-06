// compact.go — /compact command.
//
// /compact manually triggers the compaction pipeline. The compactor
// runs MaybeCompact with the current model and context window; if the
// session is already under the threshold, /compact reports that no
// compaction was needed.
package slash

import (
	"context"
	"errors"
	"fmt"

	"github.com/taucentral/tau/internal/agent"
)

type compactCommand struct{}

func newCompactCommand() Command { return compactCommand{} }

func (compactCommand) Name() string      { return "/compact" }
func (compactCommand) ShortHelp() string { return "Manually trigger compaction of the current session" }

func (compactCommand) Execute(ctx context.Context, _ string, session agent.CommandSession) (string, error) {
	if session == nil {
		return "", errors.New("/compact: session is nil")
	}
	rt := session.Runtime()
	if rt.Compactor() == nil {
		return "", errors.New("/compact: compactor not configured")
	}
	model := rt.Options().Model()
	window := rt.Options().ContextWindow()
	result, err := rt.Compactor().MaybeCompact(ctx, rt.State(), model, window)
	if err != nil {
		return "", fmt.Errorf("/compact: %w", err)
	}
	if !result.Compacted {
		return fmt.Sprintf("no compaction needed: %s", result.Reason), nil
	}
	return fmt.Sprintf("compaction complete: archived %d entries, %d→%d tokens",
		result.ArchivedCount, result.PreCompactionTokens, result.PostCompactionTokens), nil
}
