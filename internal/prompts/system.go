// system.go — assemble the system prompt for an agent turn.
//
// Order of assembly (later sections append, not replace):
//
//  1. Built-in defaults (hardcoded baseline identity).
//  2. Global SYSTEM.md at <ConfigDir>/SYSTEM.md.
//  3. Global CLAUDE.md at <ConfigDir>/CLAUDE.md (user-level context,
//     matching Claude Code's ~/.claude/CLAUDE.md convention via symlink).
//  4. Ancestor walk for AGENTS.md from ProjectDir up to a VCS marker
//     (`.git`/`.hg`/`.svn`/`.jj`), outer-to-inner.
//  5. Ancestor walk for CLAUDE.md, same walk.
//
// Each section is optional. Missing files are silently skipped (most
// projects have none on first run). Empty results are allowed — the
// caller can supply an empty system list to the LLM if no content was
// found.
//
// Each non-empty source becomes one llm.TextContent block prefixed
// with a small provenance header so the model can attribute guidance.
//
// Reference: pi has no ancestor-walk equivalent; this is a tau
// extension that brings parity with Claude Code and Codex CLI, which
// both walk parent directories concatenating CLAUDE.md/AGENTS.md.

package prompts

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/coevin/tau/internal/llm"
)

// BuiltInSystem is the hardcoded baseline system prompt. It establishes
// the agent's identity and core behaviors without which the model would
// not know it is acting as a coding agent. Project and global files
// augment this; they do not replace it.
const BuiltInSystem = `You are tau, an interactive command-line coding agent. You help the user with software engineering tasks: reading and writing code, running commands, debugging, and explaining. You prefer focused changes over sprawling rewrites. You say when you are unsure.`

// vcsMarkers are the directory entries that indicate a repository root.
// The walk stops at the first ancestor containing any of these. The
// list mirrors modern VCS systems: git, Mercurial, Subversion, Jujutsu.
// `.git` may be a directory (normal repos) or a file (worktrees /
// submodules); we accept either via os.Stat.
var vcsMarkers = []string{".git", ".hg", ".svn", ".jj"}

// Assembler composes the system prompt from built-in defaults plus any
// matching files in the global config dir, the project dir, and the
// project's ancestor directories up to a VCS-marker boundary.
//
// Walk behavior is governed by three fields. Defaults preserve today's
// single-file behavior for projects whose ProjectDir is the repo root:
//
//   - StopDir: when non-empty, the walk stops at this directory
//     (inclusive) regardless of VCS markers. Overrides both the marker
//     scan and WalkToRoot.
//   - WalkToRoot: when true, ignore VCS markers and walk all the way to
//     the filesystem root. StopDir still wins if set.
//   - MaxAncestorDepth: caps the number of ancestor directories
//     visited. 0 means unlimited. 1 reads only ProjectDir and its
//     parent.
type Assembler struct {
	// ConfigDir is the global config directory (typically
	// ~/.config/tau). Empty disables global lookups.
	ConfigDir string
	// ProjectDir is the cwd of the agent session. Empty disables
	// project lookups.
	ProjectDir string
	// BuiltIn, if non-empty, overrides BuiltInSystem. Tests use this
	// to make assertions deterministic.
	BuiltIn string
	// StopDir, when non-empty, overrides the VCS-marker scan and caps
	// the ancestor walk at this directory (inclusive).
	StopDir string
	// WalkToRoot, when true, ignores VCS markers and walks to the
	// filesystem root. StopDir still wins if both are set.
	WalkToRoot bool
	// MaxAncestorDepth caps the ancestor walk count. 0 means
	// unlimited. 1 reads only ProjectDir and its parent.
	MaxAncestorDepth int
}

// WalkOpts is the walk configuration accepted by NewAssemblerWithWalk.
// The zero value preserves the pre-walk behavior for projects whose
// ProjectDir is the repo root.
type WalkOpts struct {
	// StopDir, when non-empty, overrides the VCS-marker scan and caps
	// the ancestor walk at this directory (inclusive).
	StopDir string
	// WalkToRoot, when true, ignores VCS markers and walks to the
	// filesystem root. StopDir still wins if both are set.
	WalkToRoot bool
	// MaxAncestorDepth caps the ancestor walk count. 0 means
	// unlimited. 1 reads only ProjectDir and its parent.
	MaxAncestorDepth int
}

// NewAssembler returns an Assembler using the standard locations with
// default walk behavior (VCS-marker scan, unlimited depth). Equivalent
// to NewAssemblerWithWalk with zero-value WalkOpts.
func NewAssembler(configDir, projectDir string) *Assembler {
	return &Assembler{
		ConfigDir:  configDir,
		ProjectDir: projectDir,
		BuiltIn:    BuiltInSystem,
	}
}

// NewAssemblerWithWalk returns an Assembler with the supplied walk
// options. This is the canonical constructor; the runtime factory uses
// it to honor Settings.Prompts.* knobs.
func NewAssemblerWithWalk(configDir, projectDir string, opts WalkOpts) *Assembler {
	return &Assembler{
		ConfigDir:        configDir,
		ProjectDir:       projectDir,
		BuiltIn:          BuiltInSystem,
		StopDir:          opts.StopDir,
		WalkToRoot:       opts.WalkToRoot,
		MaxAncestorDepth: opts.MaxAncestorDepth,
	}
}

// Assemble composes the system prompt. Returns a slice of text blocks;
// the order matches the assembly rule (built-in → global → project,
// with project files in outer-to-inner walk order). File-read errors
// are returned as-is (the caller decides whether to proceed with
// partial results or fail).
//
// Assemble is safe to call from goroutines; the Assembler is immutable
// after construction. Each call walks the filesystem fresh — live
// edits to context files take effect on the next round-trip.
func (a *Assembler) Assemble(ctx context.Context) ([]llm.ContentBlock, error) {
	var blocks []llm.ContentBlock
	if a == nil {
		return blocks, nil
	}
	// Built-in baseline always comes first; it establishes identity.
	if builtIn := strings.TrimSpace(a.builtInText()); builtIn != "" {
		blocks = append(blocks, textBlock("tau: identity", builtIn))
	}
	// Global SYSTEM.md (rare in practice; mostly enterprise rollouts).
	globalPath := filepath.Join(a.ConfigDir, "SYSTEM.md")
	if blk, exists, err := loadSection("global system", globalPath); err != nil {
		return nil, fmt.Errorf("prompts: read global SYSTEM.md: %w", err)
	} else if exists {
		blocks = append(blocks, blk)
	}
	// Global user-level CLAUDE.md — the fifth global lookup. Matches
	// Claude Code's ~/.claude/CLAUDE.md convention; users wanting
	// cross-tool compat symlink <ConfigDir>/CLAUDE.md → ~/.claude/CLAUDE.md.
	userClaudePath := filepath.Join(a.ConfigDir, "CLAUDE.md")
	if blk, exists, err := loadSection("user claude", userClaudePath); err != nil {
		return nil, fmt.Errorf("prompts: read user CLAUDE.md: %w", err)
	} else if exists {
		blocks = append(blocks, blk)
	}
	// Ancestor walk for project AGENTS.md and CLAUDE.md. Both walks
	// share a stop directory so they cover the same ancestor chain.
	stopDir := a.resolveStopDir()
	agentsBlocks, err := loadSectionsFromAncestors("project agents", a.ProjectDir, stopDir, "AGENTS.md", a.MaxAncestorDepth)
	if err != nil {
		return nil, fmt.Errorf("prompts: walk AGENTS.md: %w", err)
	}
	blocks = append(blocks, agentsBlocks...)
	claudeBlocks, err := loadSectionsFromAncestors("project claude", a.ProjectDir, stopDir, "CLAUDE.md", a.MaxAncestorDepth)
	if err != nil {
		return nil, fmt.Errorf("prompts: walk CLAUDE.md: %w", err)
	}
	blocks = append(blocks, claudeBlocks...)
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
		return blocks, nil
	}
}

// resolveStopDir returns the ancestor directory at which the walk
// should stop (inclusive). Returns "" when the walk should continue
// to the filesystem root.
//
// Resolution order:
//
//  1. a.StopDir, if set — explicit user override (returned as-is,
//     even if empty for "walk to root").
//  2. a.WalkToRoot, if true — "" (scan to filesystem root).
//  3. Otherwise, the first ancestor of ProjectDir containing a VCS
//     marker; "" if none found.
func (a *Assembler) resolveStopDir() string {
	if a.StopDir != "" || a.WalkToRoot {
		return a.StopDir
	}
	if a.ProjectDir == "" {
		return ""
	}
	dir := a.ProjectDir
	for {
		for _, marker := range vcsMarkers {
			if _, err := os.Stat(filepath.Join(dir, marker)); err == nil {
				return dir
			}
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "" // filesystem root, no marker found
		}
		dir = parent
	}
}

// builtInText returns the built-in baseline. Falls back to the
// hardcoded constant if Assembler.BuiltIn is empty.
func (a *Assembler) builtInText() string {
	if a.BuiltIn != "" {
		return a.BuiltIn
	}
	return BuiltInSystem
}

// loadSection reads path and wraps the content in a text block with a
// provenance header. Returns (block, exists, err). exists=false when
// the file is missing; err is non-nil only for genuine read failures
// (permission, too large, etc.).
func loadSection(label, path string) (llm.ContentBlock, bool, error) {
	if path == "" || path == string(filepath.Separator) {
		// Empty dir configuration; nothing to read.
		return llm.TextContent{}, false, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return llm.TextContent{}, false, nil
		}
		return llm.TextContent{}, false, err
	}
	body := strings.TrimSpace(string(data))
	if body == "" {
		return llm.TextContent{}, false, nil
	}
	return textBlock(label, body), true, nil
}

// loadSectionsFromAncestors walks startDir upward, reading `filename`
// in each ancestor up to stopDir (inclusive), the filesystem root, or
// maxDepth ancestors (whichever comes first). Returns one block per
// non-empty file, in outer-to-inner order (parent first, child last).
//
// The label is augmented with a depth suffix so the model can attribute
// guidance across hierarchy levels:
//
//   - startDir → "(cwd)"
//   - stopDir (when non-empty and reached) → "(repo-root)"
//   - intermediate → "(N up)" where N is depth from startDir
//
// When stopDir is empty (WalkToRoot or no VCS marker), the outermost
// visited directory is labeled "(N up)" with its depth rather than
// "(repo-root)". Missing files are silently skipped. No caching — each
// call walks the filesystem fresh.
func loadSectionsFromAncestors(label, startDir, stopDir, filename string, maxDepth int) ([]llm.ContentBlock, error) {
	if startDir == "" || startDir == string(filepath.Separator) {
		return nil, nil
	}
	// Collect ancestor directories in walk order (startDir first,
	// outermost last). Bound by stopDir, filesystem root, and
	// maxDepth (maxDepth=0 means unlimited).
	var dirs []string
	dir := startDir
	depth := 0
	for {
		dirs = append(dirs, dir)
		if stopDir != "" && dir == stopDir {
			break
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break // filesystem root
		}
		depth++
		if maxDepth > 0 && depth > maxDepth {
			break // next iteration would exceed the cap
		}
		dir = parent
	}
	// Iterate outer-to-inner so the model sees parent policies before
	// child overrides. depthFromCwd is 0 for startDir, 1 for its
	// parent, etc.
	var blocks []llm.ContentBlock
	for i := len(dirs) - 1; i >= 0; i-- {
		d := dirs[i]
		depthFromCwd := i
		var suffix string
		switch {
		case depthFromCwd == 0:
			suffix = "(cwd)"
		case stopDir != "" && d == stopDir:
			suffix = "(repo-root)"
		default:
			suffix = fmt.Sprintf("(%d up)", depthFromCwd)
		}
		blk, exists, err := loadSection(label+" "+suffix, filepath.Join(d, filename))
		if err != nil {
			return nil, err
		}
		if exists {
			blocks = append(blocks, blk)
		}
	}
	return blocks, nil
}

// textBlock formats a labelled system-prompt section. The header
// attribute is a hint to the model — a real provider might strip it
// or convert it to a markdown heading; we keep it inline for clarity.
func textBlock(label, body string) llm.ContentBlock {
	return llm.TextContent{
		Text: fmt.Sprintf("[source: %s]\n%s", label, body),
	}
}
