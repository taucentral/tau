// model.go — /model [<id>] command.
//
// Without an argument, /model prints the active model id alongside the
// protocol family of the wired client, followed by a list of every
// model declared in models.json (the active entry marked with "←").
//
// With an argument, /model validates <id> against models.json when one
// is configured, refuses cross-API switches honestly (the runtime
// cannot rebuild the LLMClient mid-session), and only then records a
// ModelChange entry and updates the runtime options.
//
// When no models.json is configured, /model cannot validate the id; it
// applies the change but emits a warning that the switch is unverified
// and that a cross-provider change requires restarting tau. This is the
// honest middle ground: don't refuse the user's choice on a configur-
// ation gap, but don't claim "model: a → b" is verified either.
//
// Reference: third-party/pi/packages/coding-agent/src/core/agent-session.ts:1453
// (pi's setModel throws when the model has no configured auth; tau's
// analogue is to refuse unknown ids when a models.json is configured).
package slash

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/coevin/tau/internal/agent"
	"github.com/coevin/tau/internal/config"
	"github.com/coevin/tau/internal/state"
)

type modelCommand struct{}

func newModelCommand() Command { return modelCommand{} }

func (modelCommand) Name() string      { return "/model" }
func (modelCommand) ShortHelp() string { return "Show the active model + list, or switch to <id>" }

func (modelCommand) Execute(_ context.Context, args string, session *agent.AgentSession) (string, error) {
	if session == nil {
		return "", errors.New("/model: session is nil")
	}
	rt := session.Runtime()
	current := rt.Options.Model
	currentAPI := rt.Options.ProviderAPI
	known := rt.Options.KnownModels

	args = strings.TrimSpace(args)
	if args == "" {
		return reportActiveModel(current, currentAPI, known), nil
	}

	// Short-circuit when the user re-selects the active model. Avoids
	// appending a meaningless ModelChange entry.
	if equalFoldID(current, args) {
		return fmt.Sprintf("model: %s (already active)", current), nil
	}

	// Without a models.json, /model cannot validate the requested id.
	// Apply the change but be honest in the response — don't claim the
	// switch was verified the way the previous implementation did.
	if len(known) == 0 {
		if _, err := rt.State.Append(state.Entry{
			Kind:    state.KindModelChange,
			Payload: state.ModelChangePayload{Model: args},
		}); err != nil {
			return "", fmt.Errorf("/model: %w", err)
		}
		rt.Options.Model = args
		return fmt.Sprintf("model: %s → %s\n  warning: no models.json configured; id not validated. Restart tau with --model %q if the provider changed.", current, args, args), nil
	}

	// KnownModels populated: validate the requested id.
	match := findKnownModel(known, args, currentAPI)
	if match == nil {
		return "", fmt.Errorf("/model: %q is not in models.json\n%s", args, formatModelList(current, known))
	}

	// Cross-API switch: refuse honestly. The runtime cannot rebuild the
	// LLMClient mid-session, so we don't pretend the switch succeeded.
	// Empty API on either side relaxes the check (configuration gap, not
	// a real cross-provider switch).
	if currentAPI != "" && match.Model.API != "" && match.Model.API != currentAPI {
		providerHint := match.Provider
		if providerHint == "" {
			providerHint = "(top-level)"
		}
		return "", fmt.Errorf("/model: %q uses API %q but the wired client speaks %q\n  Restart tau with --model %q (or --provider %s) to switch providers",
			args, match.Model.API, currentAPI, args, providerHint)
	}

	if _, err := rt.State.Append(state.Entry{
		Kind:    state.KindModelChange,
		Payload: state.ModelChangePayload{Model: match.Model.ID},
	}); err != nil {
		return "", fmt.Errorf("/model: %w", err)
	}
	// Canonicalise against the registry's casing so downstream wire
	// payloads carry exactly what models.json declares (model ids are
	// case-sensitive on the provider wire; letting users accumulate
	// differently-cased duplicates would leak as request errors later).
	rt.Options.Model = match.Model.ID
	return fmt.Sprintf("model: %s → %s", current, match.Model.ID), nil
}

// reportActiveModel formats the no-args response: the active id (and
// its API when known), followed by a listing of declared models with
// the active entry marked "←". When no models.json is configured, the
// listing is replaced by a hint pointing the user at configuration.
func reportActiveModel(current string, currentAPI config.ModelAPI, known []config.KnownModel) string {
	var b strings.Builder
	switch {
	case current == "":
		b.WriteString("active model: (unset)")
	case currentAPI != "":
		fmt.Fprintf(&b, "active model: %s [%s]", current, currentAPI)
	default:
		fmt.Fprintf(&b, "active model: %s", current)
	}
	if len(known) == 0 {
		b.WriteString("\n  (no models.json configured; configure one to enable /model <id> validation)")
		return b.String()
	}
	b.WriteString("\n")
	b.WriteString(formatModelList(current, known))
	return b.String()
}

// formatModelList renders the declared models as a flat, alphabetised
// list with the active entry marked "←". Used both by the no-args
// listing and by the error path when a user-supplied id is unknown, so
// the user sees the available choices in either case.
func formatModelList(current string, known []config.KnownModel) string {
	sorted := make([]config.KnownModel, len(known))
	copy(sorted, known)
	sort.SliceStable(sorted, func(i, j int) bool {
		ki, kj := sorted[i], sorted[j]
		if ki.Provider != kj.Provider {
			return ki.Provider < kj.Provider
		}
		return ki.Model.ID < kj.Model.ID
	})
	var b strings.Builder
	b.WriteString("available models:")
	for _, km := range sorted {
		marker := " "
		if equalFoldID(km.Model.ID, current) {
			marker = "←"
		}
		provider := km.Provider
		if provider == "" {
			provider = "(top-level)"
		}
		api := string(km.Model.API)
		if api == "" {
			api = "?"
		}
		fmt.Fprintf(&b, "\n  %s %s [%s] %s", marker, km.Model.ID, api, provider)
	}
	return b.String()
}

// findKnownModel looks up id in known, case-insensitively. When the id
// is ambiguous (declared under multiple providers), prefers the entry
// whose API matches preferAPI; otherwise returns the first match in
// slice order. Returns nil when no entry matches.
func findKnownModel(known []config.KnownModel, id string, preferAPI config.ModelAPI) *config.KnownModel {
	var first *config.KnownModel
	for i := range known {
		if !equalFoldID(known[i].Model.ID, id) {
			continue
		}
		if first == nil {
			first = &known[i]
		}
		if preferAPI != "" && known[i].Model.API == preferAPI {
			return &known[i]
		}
	}
	return first
}

// equalFoldID compares two model ids case-insensitively after trimming
// surrounding whitespace. Model ids are conventionally lowercase but
// users may type them with stray capitalisation; matching leniently
// here matches user expectation without weakening on-disk validation.
func equalFoldID(a, b string) bool {
	return strings.EqualFold(strings.TrimSpace(a), strings.TrimSpace(b))
}
