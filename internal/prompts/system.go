// system.go — assemble the system prompt for an agent turn.
//
// Order of assembly (later sections append, not replace):
//
//  1. Built-in defaults (hardcoded baseline identity).
//  2. Global SYSTEM.md at <ConfigDir>/SYSTEM.md.
//  3. Project AGENTS.md at <ProjectDir>/AGENTS.md.
//  4. Project CLAUDE.md at <ProjectDir>/CLAUDE.md.
//
// Each section is optional. Missing files are silently skipped (most
// projects have none on first run). Empty results are allowed — the
// caller can supply an empty system list to the LLM if no content was
// found.
//
// Each non-empty source becomes one llm.TextContent block prefixed
// with a small provenance header so the model can attribute guidance.

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

// Assembler composes the system prompt from built-in defaults plus any
// matching files in the global config dir or the project dir.
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
}

// NewAssembler returns an Assembler using the standard locations.
func NewAssembler(configDir, projectDir string) *Assembler {
	return &Assembler{
		ConfigDir:  configDir,
		ProjectDir: projectDir,
		BuiltIn:    BuiltInSystem,
	}
}

// Assemble composes the system prompt. Returns a slice of text blocks;
// the order matches the assembly rule (built-in → global → project).
// File-read errors are returned as-is (the caller decides whether to
// proceed with partial results or fail).
//
// Assemble is safe to call from goroutines; the Assembler is immutable
// after NewAssembler.
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
	// Project AGENTS.md — the cross-vendor convention.
	agentsPath := filepath.Join(a.ProjectDir, "AGENTS.md")
	if blk, exists, err := loadSection("project agents", agentsPath); err != nil {
		return nil, fmt.Errorf("prompts: read AGENTS.md: %w", err)
	} else if exists {
		blocks = append(blocks, blk)
	}
	// Project CLAUDE.md — legacy tau-specific convention.
	claudePath := filepath.Join(a.ProjectDir, "CLAUDE.md")
	if blk, exists, err := loadSection("project claude", claudePath); err != nil {
		return nil, fmt.Errorf("prompts: read CLAUDE.md: %w", err)
	} else if exists {
		blocks = append(blocks, blk)
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
		return blocks, nil
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

// textBlock formats a labelled system-prompt section. The header
// attribute is a hint to the model — a real provider might strip it
// or convert it to a markdown heading; we keep it inline for clarity.
func textBlock(label, body string) llm.ContentBlock {
	return llm.TextContent{
		Text: fmt.Sprintf("[source: %s]\n%s", label, body),
	}
}
