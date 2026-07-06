package compaction

import (
	"context"
	"fmt"
	"strings"

	"github.com/taucentral/tau/internal/llm"
	"github.com/taucentral/tau/internal/state"
)

// summarizationMaxTokens is the output cap passed to the LLM when generating
// a structured summary. 2048 leaves headroom for the model to produce all six
// sections without truncating mid-list.
const summarizationMaxTokens = 2048

// SummarizationPrompt is the prompt template used the first time compaction
// runs in a session. The template has two substitution slots:
//
//	{{.Entries}}       — rendered conversation being archived
//	{{.FileContext}}   — rendered file-tracking section (may be empty)
const SummarizationPrompt = `You are summarizing an agent conversation so future turns can continue with reduced context. Read the conversation below and produce a structured summary with EXACTLY these sections, in this order, using Markdown headings:

## Goal
What the user is trying to accomplish.

## Constraints & Preferences
Any rules, preferences, or constraints the user has stated or implied.

## Progress
Three bullet sub-sections: Done, In Progress, Blocked. Omit a sub-section if empty.

## Key Decisions
Important decisions made and their rationale.

## Next Steps
The immediate next actions the agent should take.

## Critical Context
Anything else the agent needs to know to continue. Include the file-operation list verbatim if present.

Be concise. Each section should be at most a few sentences or bullets. Do not include the original conversation verbatim; extract and condense.

Conversation to summarize (oldest first, most-recent last):

` + "%s" + `

` + "%s" + `

Now write the summary.`

// UpdateSummarizationPrompt is the prompt template used when compaction runs
// again in a session that already has a Compaction entry. Slots:
//
//	{{.PreviousSummary}} — the prior structured summary
//	{{.NewEntries}}      — rendered new conversation being archived
//	{{.FileContext}}     — rendered file-tracking section (may be empty)
const UpdateSummarizationPrompt = `You are updating an existing summary with newer conversation. The previous summary appears first; the new conversation to merge in appears after. Produce an updated summary with the SAME structure (Goal, Constraints & Preferences, Progress, Key Decisions, Next Steps, Critical Context). Preserve facts from the previous summary that are still relevant. Drop facts that have been superseded. Add new facts from the new conversation.

Previous summary:

` + "%s" + `

New conversation to merge in (oldest first):

` + "%s" + `

` + "%s" + `

Now write the updated summary.`

// Summarizer wraps an LLM client and the active model id. The zero value is
// not usable; construct via NewSummarizer.
type Summarizer struct {
	Client llm.LLMClient
	Model  string
}

// NewSummarizer returns a Summarizer that calls client under model.
func NewSummarizer(client llm.LLMClient, model string) *Summarizer {
	return &Summarizer{Client: client, Model: model}
}

// Summarize produces a structured summary of the entries being archived.
// previousSummary is "" for a first-time summarization (uses
// SummarizationPrompt); non-empty values use UpdateSummarizationPrompt.
//
// archived must be in root → leaf (chronological) order. fileTracking carries
// the file operations extracted from the entire session (not just archived);
// it's rendered as the "Critical Context" section.
func (s *Summarizer) Summarize(
	ctx context.Context,
	previousSummary string,
	archived []state.Entry,
	fileTracking FileTracking,
) (string, error) {
	if s.Client == nil {
		return "", fmt.Errorf("compaction: Summarizer.Client is nil")
	}
	entriesText := renderEntriesForPrompt(archived)
	fileContext := fileTracking.CriticalContextSection()

	var prompt string
	if strings.TrimSpace(previousSummary) == "" {
		fileClause := ""
		if fileContext != "" {
			fileClause = fileContext + "\n"
		}
		prompt = fmt.Sprintf(SummarizationPrompt, entriesText, fileClause)
	} else {
		fileClause := ""
		if fileContext != "" {
			fileClause = fileContext + "\n"
		}
		prompt = fmt.Sprintf(UpdateSummarizationPrompt, previousSummary, entriesText, fileClause)
	}

	return s.callLLM(ctx, prompt)
}

// callLLM issues a single-turn Stream request and concatenates TextDelta
// fragments into the output. Returns the Final.Err if the provider sets one.
func (s *Summarizer) callLLM(ctx context.Context, prompt string) (string, error) {
	maxTokens := summarizationMaxTokens
	req := llm.Request{
		Model:     s.Model,
		MaxTokens: &maxTokens,
		Messages: []llm.Message{{
			Role: llm.RoleUser,
			Content: []llm.ContentBlock{
				llm.TextContent{Text: prompt},
			},
		}},
	}
	ch, err := s.Client.Stream(ctx, req)
	if err != nil {
		return "", fmt.Errorf("compaction: summarizer stream start: %w", err)
	}
	var sb strings.Builder
	finalSeen := false
	for delta := range ch {
		switch d := delta.(type) {
		case llm.TextDelta:
			sb.WriteString(d.Text)
		case llm.Final:
			finalSeen = true
			if d.Err != nil {
				return "", fmt.Errorf("compaction: summarizer final: %w", d.Err)
			}
		}
	}
	if !finalSeen {
		return "", llm.ErrFinalMissing
	}
	out := strings.TrimSpace(sb.String())
	if out == "" {
		return "", fmt.Errorf("compaction: summarizer returned empty summary")
	}
	return out, nil
}

// renderEntriesForPrompt formats entries as "[role] content" lines in order.
// Non-conversational entries are skipped. ToolUse/ToolResult blocks are
// rendered compactly so the model can see what tools did without the full
// result payload bloating the prompt.
func renderEntriesForPrompt(entries []state.Entry) string {
	var sb strings.Builder
	for _, e := range entries {
		switch e.Kind {
		case state.KindMessage:
			mp, ok := e.Payload.(state.MessagePayload)
			if !ok {
				continue
			}
			fmt.Fprintf(&sb, "[%s] ", mp.Role)
			for _, b := range mp.Content {
				renderBlockForPrompt(&sb, b)
			}
			sb.WriteString("\n")
		case state.KindBranchSummary:
			bp, _ := e.Payload.(state.BranchSummaryPayload)
			fmt.Fprintf(&sb, "[branch-summary] %s\n", bp.Summary)
		case state.KindCompaction:
			cp, _ := e.Payload.(state.CompactionPayload)
			fmt.Fprintf(&sb, "[prior-compaction] %s\n", cp.Summary)
		case state.KindSessionHeader, state.KindThinkingLevelChange,
			state.KindModelChange, state.KindLabel, state.KindSessionInfo,
			state.KindCustom, state.KindCustomMessage:
			// Non-conversational kinds intentionally omitted from the
			// summarization prompt (no user/assistant content to carry over).
		}
	}
	if sb.Len() == 0 {
		return "(no conversational entries)"
	}
	return sb.String()
}

// renderBlockForPrompt writes one content block's compact textual form.
func renderBlockForPrompt(sb *strings.Builder, b llm.ContentBlock) {
	switch v := b.(type) {
	case llm.TextContent:
		sb.WriteString(v.Text)
		sb.WriteString(" ")
	case llm.ToolUse:
		fmt.Fprintf(sb, "(tool %s input=%s) ", v.Name, string(v.Input))
	case llm.ToolResult:
		fmt.Fprintf(sb, "(tool-result for %s ", v.ToolUseID)
		for _, rb := range v.Content {
			renderBlockForPrompt(sb, rb)
		}
		if v.IsError {
			sb.WriteString("[error]")
		}
		sb.WriteString(") ")
	case llm.ThinkingContent:
		// Skip thinking content in the prompt; it's rarely useful for
		// summary and bloats the input.
	case llm.ImageContent:
		sb.WriteString("(image) ")
	}
}
