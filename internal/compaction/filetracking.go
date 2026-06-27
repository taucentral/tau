// Package compaction implements the multi-stage compaction pipeline per the
// compaction spec.
//
// Compaction runs at the start of an agentic turn when the active context
// exceeds (contextWindow - reserveTokens). The pipeline is:
//
//  1. Trigger check    — MaybeCompact compares the current context token
//     count against the per-model budget; below budget is a no-op.
//  2. Protection stage — mark entries that MUST survive compaction
//     (SessionInfo, Label, most-recent file reads).
//  3. Cut-point stage   — walk backward from the leaf, never splitting
//     ToolUse/ToolResult pairs, until the retained region fits the budget.
//  4. Summarization     — invoke the LLM with a structured-summary prompt
//     over the archived region; store the result as a Compaction entry.
//  5. Sliding-window    — record FirstKeptEntryID on the new Compaction
//     entry so BuildContext can drop archived entries from future contexts.
//
// File-operation tracking (filetracking.go) records paths read/modified by
// tools so the summarizer can include them in the "Critical Context" section.
package compaction

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/state"
)

// DefaultFileReadTools names the tools treated as file reads for protection
// and tracking. The list is lowercase-matched against ToolUse.Name.
var DefaultFileReadTools = []string{"read"}

// DefaultFileWriteTools names the tools treated as file modifications.
var DefaultFileWriteTools = []string{"edit", "write", "patch"}

// FileOperation records a single file read or modification by a tool.
type FileOperation struct {
	// Path is the file path extracted from the tool input.
	Path string `json:"path"`
	// Operation is "read" or "modify".
	Operation string `json:"operation"`
	// EntryID is the ID of the entry that holds the ToolUse block.
	EntryID string `json:"entryId"`
	// Timestamp is the entry's timestamp.
	Timestamp time.Time `json:"timestamp"`
}

// FileTracking aggregates file operations for inclusion in the summarizer's
// "Critical Context" section. The slices are in chronological order
// (oldest first).
type FileTracking struct {
	Reads         []FileOperation `json:"reads"`
	Modifications []FileOperation `json:"modifications"`
}

// ExtractFileTracking walks entries (in any order) and collects file
// operations from ToolUse blocks whose Name matches one of the configured
// read/write tool names. The returned slices are sorted chronologically
// (oldest first); duplicate (path, operation, entryID) tuples are merged.
//
// Path extraction looks for the first non-empty of these JSON fields in
// the ToolUse input: "path", "file_path", "filePath". If none is present,
// the operation is recorded with Path="" and skipped by DistinctPaths.
func ExtractFileTracking(entries []state.Entry, readTools, writeTools []string) FileTracking {
	readSet := toLowerSet(readTools)
	writeSet := toLowerSet(writeTools)

	var out FileTracking
	for _, e := range entries {
		if e.Kind != state.KindMessage {
			continue
		}
		mp, ok := e.Payload.(state.MessagePayload)
		if !ok {
			continue
		}
		for _, b := range mp.Content {
			tu, ok := b.(llm.ToolUse)
			if !ok {
				continue
			}
			name := strings.ToLower(tu.Name)
			path := extractPath(tu.Input)
			if path == "" {
				continue
			}
			op := FileOperation{
				Path:      path,
				EntryID:   e.ID,
				Timestamp: e.Timestamp,
			}
			if _, isRead := readSet[name]; isRead {
				op.Operation = "read"
				out.Reads = append(out.Reads, op)
			} else if _, isWrite := writeSet[name]; isWrite {
				op.Operation = "modify"
				out.Modifications = append(out.Modifications, op)
			}
		}
	}
	sortChronologically(out.Reads)
	sortChronologically(out.Modifications)
	return out
}

// DistinctPaths returns the unique file paths across both operations,
// preserving the most-recent operation's kind (read vs modify) per path.
// Used by ProtectionList to identify the most-recent read per file.
func (ft FileTracking) DistinctPaths() map[string]FileOperation {
	out := make(map[string]FileOperation, len(ft.Reads)+len(ft.Modifications))
	// Walk reads and modifications; for each path, keep the latest by timestamp.
	// On a tie, modifications win (they reflect the current on-disk state).
	for _, op := range ft.Reads {
		existing, ok := out[op.Path]
		if !ok || op.Timestamp.After(existing.Timestamp) {
			out[op.Path] = op
		}
	}
	for _, op := range ft.Modifications {
		existing, ok := out[op.Path]
		if !ok || !op.Timestamp.Before(existing.Timestamp) {
			out[op.Path] = op
		}
	}
	return out
}

// CriticalContextSection renders the file tracking as a markdown section
// suitable for inclusion in the summarization prompt. Returns "" when both
// lists are empty so callers can omit the section entirely.
func (ft FileTracking) CriticalContextSection() string {
	if len(ft.Reads) == 0 && len(ft.Modifications) == 0 {
		return ""
	}
	var sb strings.Builder
	sb.WriteString("## Critical Context — Files\n")
	if reads := ft.uniquePaths(ft.Reads); len(reads) > 0 {
		sb.WriteString("\nFiles read:\n")
		for _, p := range reads {
			fmt.Fprintf(&sb, "- %s\n", p)
		}
	}
	if mods := ft.uniquePaths(ft.Modifications); len(mods) > 0 {
		sb.WriteString("\nFiles modified:\n")
		for _, p := range mods {
			fmt.Fprintf(&sb, "- %s\n", p)
		}
	}
	return sb.String()
}

// uniquePaths returns the sorted-unique paths from a FileOperation slice.
func (FileTracking) uniquePaths(ops []FileOperation) []string {
	seen := make(map[string]struct{}, len(ops))
	for _, op := range ops {
		if op.Path == "" {
			continue
		}
		seen[op.Path] = struct{}{}
	}
	out := make([]string, 0, len(seen))
	for p := range seen {
		out = append(out, p)
	}
	sort.Strings(out)
	return out
}

// toLowerSet builds a lowercase lookup set from a slice of names.
func toLowerSet(names []string) map[string]struct{} {
	out := make(map[string]struct{}, len(names))
	for _, n := range names {
		out[strings.ToLower(n)] = struct{}{}
	}
	return out
}

// extractPath pulls the first non-empty path-like field from a ToolUse input
// JSON payload. Returns "" when no path field is present or the input is not
// a JSON object.
func extractPath(input json.RawMessage) string {
	if len(input) == 0 {
		return ""
	}
	var obj map[string]any
	if err := json.Unmarshal(input, &obj); err != nil {
		return ""
	}
	for _, key := range []string{"path", "file_path", "filePath"} {
		if v, ok := obj[key]; ok {
			if s, ok := v.(string); ok && s != "" {
				return s
			}
		}
	}
	return ""
}

// sortChronologically sorts operations oldest-first by timestamp. Ties break
// on EntryID for determinism.
func sortChronologically(ops []FileOperation) {
	sort.SliceStable(ops, func(i, j int) bool {
		if ops[i].Timestamp.Equal(ops[j].Timestamp) {
			return ops[i].EntryID < ops[j].EntryID
		}
		return ops[i].Timestamp.Before(ops[j].Timestamp)
	})
}
