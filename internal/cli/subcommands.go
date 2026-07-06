// subcommands.go — handlers for tau's metadata subcommands.
//
// These subcommands short-circuit before any agent session is built. They
// touch only the config layer (paths, models file) and emit output to
// stdout/stderr before returning. Dispatch routes to them via
// dispatchSubcommand (config, update) and directly for --list-models.
//
// The handlers are kept in this file so dispatch.go remains a thin router.

package cli

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"text/tabwriter"

	"github.com/taucentral/tau/internal/config"
)

// builtinModels is the static list of well-known provider models merged
// with the user's models.json for --list-models. These entries exist so
// the command produces useful output on a fresh install; users shadow or
// extend them via models.json. Pricing is per-million tokens in USD and
// is intentionally conservative — accuracy is the user's responsibility
// once they override via models.json.
var builtinModels = []config.ModelDefinition{
	{
		ID:            "claude-opus-4-5",
		Name:          "Claude Opus 4.5",
		API:           config.APIAnthropic,
		ContextWindow: 200000,
		MaxTokens:     32000,
		Cost: &config.ModelCost{
			Input: 15, Output: 75, CacheRead: 1.5, CacheWrite: 18.75,
		},
	},
	{
		ID:            "claude-sonnet-4-5",
		Name:          "Claude Sonnet 4.5",
		API:           config.APIAnthropic,
		ContextWindow: 200000,
		MaxTokens:     16000,
		Cost: &config.ModelCost{
			Input: 3, Output: 15, CacheRead: 0.3, CacheWrite: 3.75,
		},
	},
	{
		ID:            "claude-haiku-4-5",
		Name:          "Claude Haiku 4.5",
		API:           config.APIAnthropic,
		ContextWindow: 200000,
		MaxTokens:     8192,
		Cost: &config.ModelCost{
			Input: 1, Output: 5,
		},
	},
	{
		ID:            "gpt-4o",
		Name:          "GPT-4o",
		API:           config.APIOpenAI,
		ContextWindow: 128000,
		MaxTokens:     16384,
		Cost: &config.ModelCost{
			Input: 2.5, Output: 10,
		},
	},
	{
		ID:            "gpt-4o-mini",
		Name:          "GPT-4o mini",
		API:           config.APIOpenAI,
		ContextWindow: 128000,
		MaxTokens:     16384,
		Cost: &config.ModelCost{
			Input: 0.15, Output: 0.6,
		},
	},
}

// runConfigSubcommand implements `tau config`. With --path it prints the
// resolved agent directory and exits. Without flags, it prints the agent
// directory plus the path to settings.json and a hint for editing — v1
// does not launch $EDITOR itself because doing so safely requires a TTY
// check and process-management that belongs in the interactive mode.
func runConfigSubcommand(_ context.Context, args Args) error {
	if hasSubcommandArg(args, "--path") {
		agentDir, err := config.AgentDir()
		if err != nil {
			return fmt.Errorf("resolve agent dir: %w", err)
		}
		fmt.Println(agentDir)
		return nil
	}
	agentDir, err := config.AgentDir()
	if err != nil {
		return fmt.Errorf("resolve agent dir: %w", err)
	}
	settingsPath := filepath.Join(agentDir, "settings.json")
	fmt.Printf("Agent directory: %s\n", agentDir)
	fmt.Printf("Settings file:   %s\n", settingsPath)
	fmt.Println("Edit settings.json directly, or run `tau` in interactive mode to use the wizard.")
	return nil
}

// runUpdateSubcommand implements `tau update`. Per task 9.3, v1 is a
// stub: tau ships as a single static binary built via `make build` /
// `go install`; there is no in-process self-update path. The handler
// prints where to look for updates and returns nil so the exit code is 0
// (this is informational, not an error condition).
func runUpdateSubcommand(_ context.Context, _ Args) error {
	fmt.Fprintln(os.Stderr, "tau: self-update is not bundled with this binary.")
	fmt.Fprintln(os.Stderr, "Use your package manager (e.g., `go install ./cmd/tau@latest`,")
	fmt.Fprintln(os.Stderr, "`brew upgrade`, or your distribution's updater) to upgrade.")
	return nil
}

// listModels implements `tau --list-models`. Prints one model per line
// with provider, ID, context window, and pricing columns. Sources:
// built-in defaults plus the user's models.json (the file's entries
// shadow built-ins with the same ID).
func listModels(_ context.Context, _ Args) error {
	agentDir, err := config.AgentDir()
	if err != nil {
		return fmt.Errorf("resolve agent dir: %w", err)
	}
	modelsPath := filepath.Join(agentDir, "models.json")
	mf, err := config.LoadModelsFile(modelsPath)
	if err != nil {
		return fmt.Errorf("load %s: %w", modelsPath, err)
	}
	merged := mergeModelLists(builtinModels, mf)
	printModelsTable(os.Stdout, merged)
	return nil
}

// hasSubcommandArg reports whether needle appears in args.SubcommandArgs.
func hasSubcommandArg(args Args, needle string) bool {
	for _, a := range args.SubcommandArgs {
		if a == needle {
			return true
		}
	}
	return false
}

// mergeModelLists returns the effective model list for --list-models:
// built-ins plus everything in mf, with file entries shadowing built-ins
// by ID. Provider-attached models in mf inherit their provider name as
// the "provider" column when the model itself doesn't carry one.
func mergeModelLists(builtins []config.ModelDefinition, mf *config.ModelsFile) []config.ModelDefinition {
	seen := make(map[string]int, len(builtins))
	out := make([]config.ModelDefinition, 0, len(builtins)+len(mf.Models))
	for _, m := range builtins {
		seen[m.ID] = len(out)
		out = append(out, annotateProvider(m, string(m.API)))
	}
	// Top-level models.
	for _, m := range mf.Models {
		prov := string(m.API)
		if prov == "" {
			prov = "user"
		}
		entry := annotateProvider(m, prov)
		if idx, ok := seen[m.ID]; ok {
			out[idx] = entry
		} else {
			seen[m.ID] = len(out)
			out = append(out, entry)
		}
	}
	// Provider-attached models.
	for name, p := range mf.Providers {
		for _, m := range p.Models {
			entry := annotateProvider(m, name)
			if idx, ok := seen[m.ID]; ok {
				out[idx] = entry
			} else {
				seen[m.ID] = len(out)
				out = append(out, entry)
			}
		}
	}
	return out
}

// annotateProvider returns a copy of m with provider set if empty. Used
// by mergeModelLists so the caller can label provider-attached models
// with their provider name without mutating the input slice.
func annotateProvider(m config.ModelDefinition, provider string) config.ModelDefinition {
	out := m
	if out.Name == "" {
		out.Name = out.ID
	}
	// Stash the provider label in the Headers map only if there's room;
	// otherwise leave the model as-is. The list-models printer reads
	// api as the provider column, so we set API to the provider label
	// when the model didn't carry one.
	if out.API == "" {
		out.API = config.ModelAPI(provider)
	}
	return out
}

// printModelsTable writes one row per model with tab-aligned columns.
// The header row labels the fields requested by the cli spec:
// provider, model ID, context window, and pricing.
func printModelsTable(w io.Writer, models []config.ModelDefinition) {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "PROVIDER\tMODEL ID\tCONTEXT\tCOST (IN/OUT $/M)")
	for _, m := range models {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n",
			string(m.API),
			m.ID,
			formatContextWindow(m.ContextWindow),
			formatCost(m.Cost),
		)
	}
	_ = tw.Flush()
}

// formatContextWindow renders the context-window token count as a
// human-friendly string (e.g. "200000" → "200k"). "0" / unknown renders
// as "-" so the column stays readable.
func formatContextWindow(n int) string {
	if n <= 0 {
		return "-"
	}
	if n >= 1000 {
		// Round to nearest thousand for compactness.
		k := (n + 500) / 1000
		return fmt.Sprintf("%dk", k)
	}
	return fmt.Sprintf("%d", n)
}

// formatCost renders a ModelCost as "in/out" per-million-token USD.
// Missing fields render as "0"; a nil Cost renders as "-" so users can
// spot unpriced entries at a glance.
func formatCost(c *config.ModelCost) string {
	if c == nil {
		return "-"
	}
	return fmt.Sprintf("%g/%g", c.Input, c.Output)
}
