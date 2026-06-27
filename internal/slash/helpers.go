// helpers.go — shared utilities for slash commands.
package slash

import (
	"errors"

	"github.com/coevin/tau/internal/agent"
	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/state"
)

// summarizerClient returns the LLM client the slash command should pass
// to state.Manager.BranchWithSummary. Returns nil when no client is
// wired — BranchWithSummary then returns ErrBranchWithSummaryUnsupported
// and the caller can fall back to a plain Branch.
func summarizerClient(session *agent.AgentSession) llm.LLMClient {
	if session == nil {
		return nil
	}
	return session.Runtime().Options.LLMClient
}

// isUnsupported reports whether err is the ErrBranchWithSummaryUnsupported
// sentinel from the state package. Wrapped in a helper so slash command
// files do not need to import state directly.
func isUnsupported(err error) bool {
	return errors.Is(err, state.ErrBranchWithSummaryUnsupported)
}
