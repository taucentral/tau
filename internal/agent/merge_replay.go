// merge_replay.go — helpers for MergePolicyReplay conflict detection.
//
// MergeState(MergePolicyReplay) replays the child's write-tool-calls
// against the parent's current write-set. When a file the child writes
// was also written by the parent, the replay stops and returns
// ErrOrchestrationConflict wrapping a *ConflictReportShell populated with:
//
//   - Phase: copied from MergeSpecShell.Phase (populated by the
//     orchestrator with the running phase's Name).
//   - File: the absolute path of the conflicting file.
//   - LineRange: [first, last] line range of the child's intended edit,
//     computed from the tool call's input. For edit calls, the range is
//     derived from old_string's first occurrence in the parent's current
//     on-disk file. For write calls, the range covers the whole content
//     ([1, newline_count(content)+1]). For other write tools (patch,
//     etc.) the range is [0,0] (whole-file, unscoped).
//
// Reads (read tool) do NOT conflict — only file-mutating tools (edit,
// write, patch) are classified as writes. This avoids false positives
// where a child's read of a file the parent wrote triggers a spurious
// conflict.
//
// File classification uses compaction.DefaultFileReadTools and
// compaction.DefaultFileWriteTools so the replay conflict logic and the
// compaction pipeline share one source of truth for "which tools mutate
// files."

package agent

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"github.com/coevin/tau/internal/compaction"
	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/state"
)

// writeToolSet is the lowercase lookup set of tool names that mutate
// files. Mirrors compaction.DefaultFileWriteTools so the replay conflict
// logic and the compaction pipeline agree on what counts as a write.
var writeToolSet = func() map[string]struct{} {
	s := make(map[string]struct{}, len(compaction.DefaultFileWriteTools))
	for _, t := range compaction.DefaultFileWriteTools {
		s[strings.ToLower(t)] = struct{}{}
	}
	return s
}()

// isWriteTool reports whether name is a file-mutating tool.
func isWriteTool(name string) bool {
	_, ok := writeToolSet[strings.ToLower(name)]
	return ok
}

// writeOp records a single file-mutating tool call extracted from a
// child entry's content blocks.
type writeOp struct {
	toolName string
	input    json.RawMessage
}

// parentWriteFiles returns the set of file paths mutated by the parent's
// state-tree branch (root → leaf). Uses compaction.ExtractFileTracking so
// read tools are excluded; only DefaultFileWriteTools contribute.
func parentWriteFiles(mgr state.Manager) map[string]bool {
	out := map[string]bool{}
	entries, err := parentPathEntries(mgr)
	if err != nil {
		return out
	}
	ft := compaction.ExtractFileTracking(entries, compaction.DefaultFileReadTools, compaction.DefaultFileWriteTools)
	for _, mod := range ft.Modifications {
		out[mod.Path] = true
	}
	return out
}

// parentPathEntries returns the entries on the parent's root → leaf
// path, excluding the SessionHeader root. Used for file-tracking
// extraction (write-set computation).
func parentPathEntries(mgr state.Manager) ([]state.Entry, error) {
	tree, err := mgr.Tree()
	if err != nil {
		return nil, nil
	}
	leaf := mgr.LeafID()
	if leaf == "" {
		return nil, nil
	}
	walk, err := tree.Path(leaf)
	if err != nil {
		return nil, nil
	}
	if len(walk) > 0 {
		walk = walk[1:] // drop SessionHeader root
	}
	return walk, nil
}

// childWriteOps walks entries (a child's shadow or root→leaf path) and
// returns the file-mutating tool calls in chronological order. Each op
// carries the tool name and raw input JSON for conflict detection.
func childWriteOps(entries []state.Entry) []writeOp {
	var ops []writeOp
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
			if !isWriteTool(tu.Name) {
				continue
			}
			ops = append(ops, writeOp{toolName: tu.Name, input: tu.Input})
		}
	}
	return ops
}

// detectConflict checks whether op's file is in parentFiles and, if so,
// builds a *ConflictReportShell with the phase, file, and line range.
// Returns nil when there is no conflict (file not in parentFiles).
//
// Path resolution: the child's file_path is resolved against cwd so the
// LineRange lookup reads the parent's current on-disk file.
func detectConflict(op writeOp, parentFiles map[string]bool, phase, cwd string) *ConflictReportShell {
	path := compaction.PathFromInput(op.input)
	if path == "" {
		return nil
	}
	if !parentFiles[path] {
		return nil
	}
	return &ConflictReportShell{
		Phase:     phase,
		File:      path,
		LineRange: lineRangeForOp(op, filepath.Join(cwd, path)),
	}
}

// lineRangeForOp computes the [first, last] line range (1-indexed,
// inclusive) for a write tool call. Returns [0,0] when the range cannot
// be determined.
//
//   - edit ({file_path, old_string, new_string}): reads absPath, finds
//     old_string's first occurrence, returns its line span. [0,0] when
//     old_string is absent from the file (the parent already mutated
//     those lines).
//   - write ({file_path, content}): returns [1, newline_count(content)+1]
//     — the whole-file write range.
//   - patch / other write tools: [0,0].
func lineRangeForOp(op writeOp, absPath string) [2]int {
	switch strings.ToLower(op.toolName) {
	case "edit":
		old := stringField(op.input, "old_string")
		return lineRangeOfEdit(absPath, old)
	case "write":
		content := stringField(op.input, "content")
		return lineRangeOfWrite(content)
	default:
		return [2]int{0, 0}
	}
}

// lineRangeOfEdit returns the [first, last] line span of the first
// occurrence of oldString in the file at absPath. Line numbers are
// 1-indexed and inclusive. Returns [0,0] when oldString is empty, the
// file cannot be read, or oldString is not found.
func lineRangeOfEdit(absPath, oldString string) [2]int {
	if oldString == "" {
		return [2]int{0, 0}
	}
	data, err := os.ReadFile(absPath)
	if err != nil {
		return [2]int{0, 0}
	}
	idx := bytes.Index(data, []byte(oldString))
	if idx < 0 {
		return [2]int{0, 0}
	}
	startLine := bytes.Count(data[:idx], []byte("\n")) + 1
	endLine := startLine + bytes.Count(data[idx:idx+len(oldString)], []byte("\n"))
	return [2]int{startLine, endLine}
}

// lineRangeOfWrite returns [1, N+1] where N is the number of newlines in
// content. Represents the whole-file write range.
func lineRangeOfWrite(content string) [2]int {
	return [2]int{1, strings.Count(content, "\n") + 1}
}

// stringField extracts a string-valued field from a tool call's input
// JSON. Returns "" when the field is absent or the input is not a JSON
// object.
func stringField(input json.RawMessage, key string) string {
	if len(input) == 0 {
		return ""
	}
	var m map[string]any
	if err := json.Unmarshal(input, &m); err != nil {
		return ""
	}
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}
