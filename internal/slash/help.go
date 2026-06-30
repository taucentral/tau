// help.go — /help command.
//
// /help lists every registered command with a one-line description.
// Output is formatted as a fixed-width table for readability.
package slash

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"text/tabwriter"

	"github.com/coevin/tau/internal/agent"
)

// helpCommand needs a reference to its Registry so it can list the
// other commands. The reference is set by DefaultRegistry after the
// registry is constructed.
type helpCommand struct {
	registry *Registry
}

func (h *helpCommand) Name() string      { return "/help" }
func (h *helpCommand) ShortHelp() string { return "Show this list of commands" }

func (h *helpCommand) Execute(_ context.Context, _ string, _ agent.CommandSession) (string, error) {
	if h.registry == nil {
		return "", errors.New("/help: registry not wired")
	}
	var b strings.Builder
	w := tabwriter.NewWriter(&b, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "Command\tDescription")
	for _, c := range h.registry.All() {
		fmt.Fprintf(w, "%s\t%s\n", c.Name(), c.ShortHelp())
	}
	w.Flush()
	return b.String(), nil
}
