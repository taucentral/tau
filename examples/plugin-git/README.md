# tau-plugin-git

A reference plugin for the [tau](../../) agent. Exposes two tools to the
model:

| Tool | Description |
| --- | --- |
| `git.status` | Working-tree status as a JSON list of `{x, y, path}` entries. |
| `git.diff` | Unstaged (or staged when `cached=true`) diff as raw text. |

The plugin is a thin shell around the local `git` binary. It assumes git
is on `PATH`; a missing git or a non-repository cwd surfaces to the model
as a `ToolResult` error it can see and recover from.

## Build

From the tau module root:

```sh
go build -o tau-plugin-git ./examples/plugin-git
```

The output is a single static binary named `tau-plugin-git` (append
`.exe` on Windows).

## Install

Drop the binary into tau's plugins directory. The file name **must**
start with `tau-plugin-` so the host's discovery recognizes it.

### POSIX (Linux / macOS)

```sh
mkdir -p ~/.config/tau/plugins
cp tau-plugin-git ~/.config/tau/plugins/
chmod +x ~/.config/tau/plugins/tau-plugin-git
```

### Windows

```cmd
mkdir "%APPDATA%\tau\plugins"
copy tau-plugin-git.exe "%APPDATA%\tau\plugins\"
```

### Project-local override

To shadow the global copy for one project, place the binary at
`<project>/plugins/tau-plugin-git`. tau loads the project-local copy
and emits a diagnostic naming the shadowed global path.

## Tool schemas

### `git.status`

```json
{
  "type": "object",
  "properties": {
    "path": {"type": "string", "description": "Optional pathspec to limit the report."}
  },
  "additionalProperties": false
}
```

Returns a JSON array. Each element is `{"x": "<staged>", "y": "<worktree>", "path": "<file>"}`. An empty array means a clean tree. The status characters match `git status --porcelain=v1` (e.g. `M` modified, `A` added, `D` deleted, `?` untracked, `R` renamed).

### `git.diff`

```json
{
  "type": "object",
  "properties": {
    "cached": {"type": "boolean", "description": "Inspect staged changes instead of the working tree."},
    "stat":   {"type": "boolean", "description": "Return a one-line diffstat per file instead of the full diff."},
    "path":   {"type": "string",  "description": "Optional pathspec to limit the diff."}
  },
  "additionalProperties": false
}
```

Returns the raw unified diff (or diffstat when `stat=true`) as text.

## Manual E2E test checklist

Run this sequence after building + installing the plugin:

1. **Discovery**: Start tau in any directory. Verify the startup banner
   (or `--debug` log) reports `plugin-git` discovered and spawned.
2. **Schema presence**: Run `tau --print-tools` (or the equivalent in
   the runtime you have) and confirm `git.status` and `git.diff` appear
   in the merged tool list.
3. **`git.status` happy path**: Inside a git repo with at least one
   uncommitted file, run:
   ```sh
   tau --print "what's my git status?"
   ```
   - **Expected**: the model invokes `git.status`, the response contains
     a JSON array, and the array reflects the actual modified files.
4. **`git.status` clean tree**: In a clean repo, repeat step 3. Expected
   response: `[]`.
5. **`git.status` path filter**: With `subdir/file.txt` modified, run:
   ```sh
   tau --print "show git status for subdir only"
   ```
   - **Expected**: only `subdir/file.txt` (or its children) appear.
6. **`git.diff` unstaged**: With an unstaged change, run:
   ```sh
   tau --print "show my unstaged diff"
   ```
   - **Expected**: response contains a unified diff with `---` / `+++`
     headers matching the modified file.
7. **`git.diff` staged**: Stage a change, then run:
   ```sh
   tau --print "show my staged diff"
   ```
   - **Expected**: model invokes `git.diff` with `{"cached": true}` and
     the response reflects the staged change.
8. **`git.diff` stat**: Run:
   ```sh
   tau --print "show me a one-line diffstat"
   ```
   - **Expected**: response contains lines like `file.go | 12 +++++-----`.
9. **Non-repo cwd**: From a directory that is not a git repo, run:
   ```sh
   tau --print "what's my git status?"
   ```
   - **Expected**: the model invokes `git.status`, the host receives a
     `ToolResult` with `IsError=true`, and the model recovers (apologizes
     and asks the user to navigate to a repo, or similar).
10. **Missing git** (optional): Rename `git` on `PATH` to a temporary
    name, start tau, ask for status. Expected: `ToolResult` error
    "git binary not found in PATH". Restore git afterward.
11. **Crash recovery**: While tau is running, `kill -9` the
    `tau-plugin-git` subprocess. Verify tau emits a "plugin crashed"
    diagnostic. On the next `git.status` or `git.diff` call, the plugin
    should re-spawn and the call should succeed (default policy
    `on-next-call`).

## Architecture

The plugin uses the
[`hashicorp/go-plugin`](https://github.com/hashicorp/go-plugin) gRPC
integration. The host spawns this binary as a subprocess; the binary
refuses to start without the magic cookie set by the host, then serves
the `Plugin` gRPC service (`Handshake`, `ListTools`, `Execute`,
`Shutdown`).

Code layout:

- `main.go` — entrypoint, gRPC server, tool dispatch, git helpers.
  - `gitServer` implements `tauproto.PluginServer`.
  - `runStatus` / `runDiff` shell out to git and shape the result.
  - `parsePorcelainV1Z` parses `git status --porcelain=v1 -z` output.
  - `runGit` is the shared exec wrapper with stderr capture.

Use this file as a template for new plugins: replace the `ListTools`
schemas and `Execute` switch, keep the cookie check and
`PluginAdapter.Serve()` call.
