# Plugins cookbook

Plugins are prompt bundles discovered from `~/.cubi/plugins/`. Each direct child directory is a namespace, and each Markdown file under `commands/` becomes a slash command.

```text
~/.cubi/plugins/mytools/
  plugin.json
  commands/review.md
```

`plugin.json`, `manifest.json`, or `package.json` may contain `{ "version": "1.0.0" }`; the version is shown by `cubi plugins list`. Command files use the filename as the command name, so the example above creates `/mytools:review`. The first non-empty Markdown line becomes the help summary; the whole file is injected as the prompt when invoked. Extra user text is appended as `User argument: ...`.

Use `cubi plugins list` to inspect discovered bundles. Use `cubi plugins reload` after editing plugin files; it reloads skills and plugins and reports added or removed bundles.

## Managing plugins from the CLI

| Command | What it does |
| --- | --- |
| `cubi plugins list [--json]` | Enumerate every `~/.cubi/plugins/<name>/manifest.json`. `--json` prints an array of `{name, version, path, commands}`. |
| `cubi plugins show <name> [--json]` | Print the manifest (pretty-printed), handler entry path, permissions block, and discovered slash commands. |
| `cubi plugins remove <name> [--force] [--yes]` | Refuses to delete if the resolved path is not a child of the plugins root, or if the directory contains anything beyond the scaffolder file set (`manifest.json`, `handler.sh`/`handler.cmd`, `README.md`, `commands/`). `--force` overrides the extra-files check; `--yes` skips the confirmation prompt. |
| `cubi plugins run <name> [args...] [--json]` | Executes the manifest `entry` script with the trailing arguments passed verbatim (`Command::arg`, so Windows quoting stays correct). Prompts for confirmation unless `permissions.shell` is `true`. |
| `cubi plugins new <name>` | Scaffolds a fresh bundle with a deny-by-default permissions block. |

### Per-plugin permissions

`manifest.json` accepts an optional `permissions` object. Missing keys
default to `false`, so a missing block is equivalent to deny-by-default.

```json
{
  "name": "mytools",
  "version": "0.1.0",
  "entry": "handler.sh",
  "permissions": {
    "network": false,
    "fs_write": false,
    "shell": false
  }
}
```

`shell: true` lets `cubi plugins run` skip the per-invocation
confirmation when stdin is a TTY. The other two flags are advisory for
now and surfaced verbatim by `cubi plugins show`; future tool-permission
machinery will consult them at dispatch time.

## Smoke-testing MCP servers

`cubi mcp test <server> [--tool <name>] [--json]` connects to a single
configured MCP server, lists its tools, and calls each tool (or just the
one named via `--tool`) with stub arguments synthesized from the JSON
schema. The output is a request / response / elapsed-ms triple per
tool, or one JSON envelope per tool with `--json`:

```sh
cubi mcp test fs --tool list_directory --json | jq .
```

Synthesized arguments fill the schema's required keys with type-zero
values (`""`, `0`, `false`, `[]`, `{}`); optional keys are omitted to
keep the payload minimal. Errors flow through the same `UserError`
classifier as the rest of cubi.

