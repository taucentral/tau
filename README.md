# tau

`tau` is a native Go coding agent — a single-binary, process-isolated, tree-state
port of the [`pi`](https://github.com/earendil-works/pi) TypeScript coding agent.

It implements four subsystems:

- **Plugin layer** (`hashicorp/go-plugin` over gRPC) — process-isolated tools.
- **State tree** (`bbolt`) — conversation history as a navigable DAG.
- **Unified LLM client** — Anthropic Messages + OpenAI Chat Completions providers.
- **TUI** (`charmbracelet/bubbletea`) — split-pane interactive interface.

## Build & run

```sh
cd tau
make build         # produces ./bin/tau (static binary, CGO disabled)
make run           # build + run
./bin/tau --help
```

## Test

```sh
make test          # unit + integration tests
make e2e           # end-to-end tests (may need TAU_RUN_E2E=1)
```

## Develop

```sh
make lint          # golangci-lint
make fmt           # gofmt + goimports
make proto         # regenerate gRPC code from internal/proto/plugin.proto
make tidy          # go mod tidy
```

## Project layout

```
tau/
├── cmd/tau/        binary entrypoint
├── internal/       agent, llm, state, compaction, tools, plugins, tui, modes,
│                   cli, slash, config, prompts, util, proto
├── pkg/tau/        public SDK
├── examples/       reference plugin
└── test/           integration + e2e tests
```

See the project root `CLAUDE.md` for conventions and
`openspec/changes/initial/` for the design artifacts.
