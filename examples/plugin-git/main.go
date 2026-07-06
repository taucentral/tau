// Command tau-plugin-git is a reference plugin for the tau agent. It
// exposes two tools to the model:
//
//   - git.status: returns working-tree status as JSON
//   - git.diff:   returns unstaged (or staged) diff as text
//
// Build (from the module root):
//
//	go build -o tau-plugin-git ./examples/plugin-git
//
// Install:
//
//	cp tau-plugin-git ~/.config/tau/plugins/
//
// Once installed, tau discovers the plugin on next start and exposes
// git.status / git.diff to the model. See examples/plugin-git/README.md
// for the full manual test checklist.
//
// The plugin is a thin shell around the local `git` binary. It assumes
// git is on PATH; missing git or non-repo cwds surface as ToolResult
// errors that the model can see and recover from.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/taucentral/tau/internal/plugins"
	tauproto "github.com/taucentral/tau/internal/proto"
)

// ExitNoCookie is the exit code when the magic cookie is missing. The
// host's go-plugin runner interprets a non-zero exit as a startup
// failure.
const ExitNoCookie = 2

// gitServer implements tauproto.PluginServer for the reference plugin.
type gitServer struct {
	tauproto.UnimplementedPluginServer
}

func main() {
	if _, ok := os.LookupEnv(plugins.MagicCookieKey); !ok {
		// Refuse to start without the magic cookie. This prevents the
		// binary from being mistaken for a normal CLI tool.
		fmt.Fprintln(os.Stderr, "tau-plugin-git: missing magic cookie; refusing to start")
		os.Exit(ExitNoCookie)
	}
	(&plugins.PluginAdapter{Server: gitServer{}}).Serve()
}

func (gitServer) Handshake(_ context.Context, _ *tauproto.Empty) (*tauproto.ProtocolVersion, error) {
	return &tauproto.ProtocolVersion{Version: int32(plugins.ProtocolVersion)}, nil
}

// ListTools advertises fully-qualified tool names. The host stores the
// advertised name verbatim; the wire RPC back to us uses just the local
// portion ("status", "diff"), which is what Execute's switch matches.
func (gitServer) ListTools(_ *tauproto.Empty, stream tauproto.Plugin_ListToolsServer) error {
	tools := []*tauproto.ToolSchema{
		{
			Name:        "git.status",
			Description: "Show the working-tree status of a git repository as a JSON list of file entries. Each entry has x (staged status char), y (working-tree status char), and path. An empty list means a clean tree.",
			JsonSchema: `{
				"type":"object",
				"properties":{
					"path":{"type":"string","description":"Optional pathspec to limit the report."}
				},
				"additionalProperties":false
			}`,
		},
		{
			Name:        "git.diff",
			Description: "Show unstaged changes (or staged when cached=true) as a unified diff. Use stat=true for a one-line summary per file. Returns raw diff text.",
			JsonSchema: `{
				"type":"object",
				"properties":{
					"cached":{"type":"boolean","description":"Inspect staged changes instead of the working tree."},
					"stat":{"type":"boolean","description":"Return a one-line diffstat per file instead of the full diff."},
					"path":{"type":"string","description":"Optional pathspec to limit the diff."}
				},
				"additionalProperties":false
			}`,
		},
	}
	for _, t := range tools {
		if err := stream.Send(t); err != nil {
			return err
		}
	}
	return nil
}

func (s gitServer) Execute(_ context.Context, call *tauproto.ToolCall) (*tauproto.ToolResult, error) {
	cwd := call.GetCwd()
	if cwd == "" {
		var err error
		cwd, err = os.Getwd()
		if err != nil {
			return errorResult(call, fmt.Sprintf("git: cannot determine cwd: %v", err)), nil
		}
	}
	switch call.GetName() {
	case "status":
		return runStatus(call, cwd)
	case "diff":
		return runDiff(call, cwd)
	default:
		return nil, fmt.Errorf("tau-plugin-git: unknown tool %q", call.GetName())
	}
}

func (gitServer) Shutdown(_ context.Context, _ *tauproto.Empty) (*tauproto.Empty, error) {
	return &tauproto.Empty{}, nil
}

// fileStatus is one entry in the JSON status report. X is the staged
// status character, Y is the working-tree status character, Path is the
// file path (post-rename if applicable). See `git status --porcelain`
// docs for the meaning of each character.
type fileStatus struct {
	X    string `json:"x"`
	Y    string `json:"y"`
	Path string `json:"path"`
}

// statusArgs is the typed argument list for git.status.
type statusArgs struct {
	Path string `json:"path"`
}

// diffArgs is the typed argument list for git.diff.
type diffArgs struct {
	Cached bool   `json:"cached"`
	Stat   bool   `json:"stat"`
	Path   string `json:"path"`
}

// runStatus invokes `git status --porcelain=v1 -z` in cwd, optionally
// limited to args.Path, and returns the parsed entries as a JSON block.
// An empty working tree returns "[]" (not null) so the model can tell
// "clean tree" from "missing data".
func runStatus(call *tauproto.ToolCall, cwd string) (*tauproto.ToolResult, error) {
	var args statusArgs
	if len(call.GetArgs()) > 0 {
		if err := json.Unmarshal(call.GetArgs(), &args); err != nil {
			return errorResult(call, fmt.Sprintf("git.status: invalid args: %v", err)), nil
		}
	}
	cmdArgs := []string{"status", "--porcelain=v1", "-z"}
	if args.Path != "" {
		cmdArgs = append(cmdArgs, "--", args.Path)
	}
	out, err := runGit(cwd, cmdArgs...)
	if err != nil {
		return errorResult(call, fmt.Sprintf("git.status: %v", err)), nil
	}
	body, err := json.Marshal(parsePorcelainV1Z(out))
	if err != nil {
		return errorResult(call, fmt.Sprintf("git.status: marshal: %v", err)), nil
	}
	return &tauproto.ToolResult{
		CallId: call.GetId(),
		Content: []*tauproto.ContentBlock{
			{Variant: &tauproto.ContentBlock_Json{Json: &tauproto.JsonBlock{Json: string(body)}}},
		},
	}, nil
}

// parsePorcelainV1Z parses `git status --porcelain=v1 -z` output.
// Entries are NUL-terminated. Each entry is "XY PATH" where X is the
// staged status, Y is the working-tree status, and PATH is the rest
// (after a single space). Renames are encoded by git as "ORIG -> NEW"
// inside a single entry; we surface the post-rename path only.
func parsePorcelainV1Z(output []byte) []fileStatus {
	if len(output) == 0 {
		return []fileStatus{}
	}
	// -z separates entries with NUL; trim a trailing NUL so split
	// doesn't produce a spurious empty entry at the end.
	entries := bytes.Split(bytes.TrimRight(output, "\x00"), []byte("\x00"))
	out := make([]fileStatus, 0, len(entries))
	for _, e := range entries {
		if len(e) < 3 {
			// Need at least XY + space; shorter entries can't be valid.
			continue
		}
		x := string(e[0:1])
		y := string(e[1:2])
		path := string(e[3:])
		// For renames git writes "ORIG -> NEW"; keep the post-rename path.
		if idx := strings.Index(path, " -> "); idx >= 0 {
			path = path[idx+len(" -> "):]
		}
		out = append(out, fileStatus{X: x, Y: y, Path: path})
	}
	return out
}

// runDiff invokes `git diff` in cwd with optional flags. Returns the
// raw diff output as a text block. With Stat=true, returns the diffstat
// summary instead. With Cached=true, inspects staged changes.
func runDiff(call *tauproto.ToolCall, cwd string) (*tauproto.ToolResult, error) {
	var args diffArgs
	if len(call.GetArgs()) > 0 {
		if err := json.Unmarshal(call.GetArgs(), &args); err != nil {
			return errorResult(call, fmt.Sprintf("git.diff: invalid args: %v", err)), nil
		}
	}
	cmdArgs := []string{"diff"}
	if args.Cached {
		cmdArgs = append(cmdArgs, "--cached")
	}
	if args.Stat {
		cmdArgs = append(cmdArgs, "--stat")
	}
	if args.Path != "" {
		cmdArgs = append(cmdArgs, "--", args.Path)
	}
	out, err := runGit(cwd, cmdArgs...)
	if err != nil {
		return errorResult(call, fmt.Sprintf("git.diff: %v", err)), nil
	}
	return &tauproto.ToolResult{
		CallId: call.GetId(),
		Content: []*tauproto.ContentBlock{
			{Variant: &tauproto.ContentBlock_Text{Text: &tauproto.TextBlock{Text: string(out)}}},
		},
	}, nil
}

// runGit executes git with the given arguments in cwd and returns its
// stdout. A non-zero exit or missing git binary produces a descriptive
// error; stderr is captured and included in the error text so the model
// sees git's own diagnostic.
func runGit(cwd string, args ...string) ([]byte, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = cwd
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		if errors.Is(err, exec.ErrNotFound) {
			return nil, errors.New("git binary not found in PATH")
		}
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			stderrText := strings.TrimSpace(stderr.String())
			if stderrText != "" {
				return nil, fmt.Errorf("exit %d: %s", exitErr.ExitCode(), stderrText)
			}
			return nil, fmt.Errorf("exit %d", exitErr.ExitCode())
		}
		return nil, err
	}
	return out, nil
}

// errorResult constructs a ToolResult with IsError=true and a single
// text block containing the message. The host surfaces this verbatim to
// the model; the model can decide to retry or proceed.
func errorResult(call *tauproto.ToolCall, msg string) *tauproto.ToolResult {
	return &tauproto.ToolResult{
		CallId:  call.GetId(),
		IsError: true,
		Content: []*tauproto.ContentBlock{
			{Variant: &tauproto.ContentBlock_Text{Text: &tauproto.TextBlock{Text: msg}}},
		},
	}
}
