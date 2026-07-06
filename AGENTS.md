# AGENTS.md

This file is a short pointer for AI agents working inside the Go module
(`tau/`). The canonical guide lives one directory up at
[`../CLAUDE.md`](../CLAUDE.md). Read it before making changes.

## What's here

`tau/` is the Go module root (`module github.com/taucentral/tau`, Go 1.25+).
The interesting subdirectories:

| Path | What |
|---|---|
| `cmd/tau/` | main entry point |
| `internal/cli/` | argv parsing, TTY detection, dispatch to run modes |
| `internal/config/` | Settings/Paths/Auth/Trust/Models/Resolve primitives |
| `internal/llm/` | provider-agnostic client layer (anthropic, openai, faux) |
| `internal/tools/` | Tool interface, Registry, built-ins |
| `internal/state/` | bbolt-backed append-only state tree |
| `internal/compaction/` | summarization pipeline over the state tree |
| `internal/prompts/` | system prompt assembler + template loader |
| `internal/plugins/` | go-plugin Manager (subprocess lifecycle) |
| `internal/agent/` | AgentSession, AgentSessionRuntime, EventBus |
| `internal/slash/` | Registry of slash commands |
| `internal/tui/` | bubbletea AppModel + components (split-pane layout) |
| `internal/modes/` | interactive / print / rpc run-mode handlers |
| `pkg/tau/` | public SDK (CreateAgentSession + re-exported types) |
| `examples/` | sdk-embed, plugin-git reference programs |
| `test/e2e/` | end-to-end test harness |

## Go-specific notes

- **No CGO.** Pure Go. `ldd bin/tau` must report "not a dynamic executable".
- **Go 1.25 toolchain** declared in go.mod. Don't downgrade.
- **Direct deps only** for things the binary actually needs. The Go module
  graph is otherwise pruned. See `go.mod` for the canonical list.
- **Tests are part of done.** Use table-driven form with `t.Run` subtests.
  Every helper that allocates a resource registers `t.Cleanup`.
- **Race detector is mandatory.** The verification gate runs
  `go test -race -timeout 120s ./...`.

## How to verify a change

```sh
cd /home/bigpod/dev/tau/tau
go build ./...
go vet ./...
go test -race -timeout 120s ./...
```

All three must pass. See the root `CLAUDE.md` for the full checklist and
the conventions that apply across the whole repository.
