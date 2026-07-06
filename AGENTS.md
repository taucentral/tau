# AGENTS.md

This file is a short pointer for AI agents working inside the Go module
(`tau/`). The canonical guide lives one directory up at
[`../CLAUDE.md`](../CLAUDE.md). Read it before making changes.

## What's here

`tau/` is the Go module root (`module github.com/taucentral/tau`, Go 1.25+).
It is **library-only** — the canonical `tau` binary lives in the separate
module [`github.com/taucentral/tau-cli`](https://github.com/taucentral/tau-cli).
The interesting subdirectories:

| Path | What |
|---|---|
| `internal/config/` | Settings/Paths/Auth/Trust/Models/Resolve primitives |
| `internal/llm/` | provider-agnostic client layer (anthropic, openai, faux) |
| `internal/tools/` | Tool interface, Registry, built-ins |
| `internal/state/` | bbolt-backed append-only state tree |
| `internal/compaction/` | summarization pipeline over the state tree |
| `internal/prompts/` | system prompt assembler + template loader |
| `internal/plugins/` | go-plugin Manager (subprocess lifecycle) |
| `internal/agent/` | AgentSession, AgentSessionRuntime, EventBus |
| `internal/slash/` | Registry of slash commands |
| `pkg/tau/` | public SDK (CreateAgentSession + re-exported types) |
| `pkg/tau/modes/` | print / rpc run-mode handlers (interactive mode lives in tau-cli) |
| `examples/` | sdk-embed, plugin-git reference programs |
| `test/e2e/` | end-to-end test harness (agent loop; no CLI wiring) |

## Go-specific notes

- **No CGO.** Pure Go.
- **Go 1.25 toolchain** declared in go.mod. Don't downgrade.
- **No charmbracelet/* direct deps.** The TUI moved to tau-cli; this module
  is embedder-friendly. The only `mattn/go-*` indirects that remain are
  pulled in transitively via `hashicorp/go-plugin` → `hashicorp/go-hclog`.
- **Direct deps only** for things the library actually needs. See `go.mod`
  for the canonical list.
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
