# tau

`tau` is a native Go coding-agent SDK — a library-only port of the
[`pi`](https://github.com/earendil-works/pi) TypeScript coding agent.

The canonical `tau` binary lives in a separate module,
[`github.com/taucentral/tau-cli`](https://github.com/taucentral/tau-cli). This
repo exists for embedders who want the agent loop, state tree, plugin layer,
and LLM client surface without paying for the TUI's transitive
`charmbracelet/*` dependency footprint.

It implements four subsystems:

- **Plugin layer** (`hashicorp/go-plugin` over gRPC) — process-isolated tools.
- **State tree** (`bbolt`) — conversation history as a navigable DAG.
- **Unified LLM client** — Anthropic Messages + OpenAI Chat Completions providers.
- **Public SDK** (`pkg/tau`) — type aliases + constructors that external
  modules can import without reaching into `internal/`.

## Test

```sh
cd tau
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
├── internal/       agent, compaction, config, fauxprovider, llm, orchestrator,
│                   plugins, prompts, proto, slash, state, storage, tools
├── pkg/tau/        public SDK (aliases + constructors)
│   └── modes/      print / rpc run-mode handlers (interactive mode lives in tau-cli)
├── examples/       plugin-git, sdk-abac, sdk-custom-provider, sdk-embed
└── test/e2e/       agent-loop integration tests
```
