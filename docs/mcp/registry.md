# MCP registry

Cubi ships a small, curated catalog of Model Context Protocol servers at
[`docs/mcp/registry.json`](./registry.json). The catalog is embedded in the
binary at build time (`include_str!`), so `cubi mcp search` and
`cubi mcp install` work without any network access and without contacting
a marketplace.

The registry is **only a set of install templates**. The user's actual
configuration still lives in `~/.cubi/mcp.json`. Installing an entry copies
its `command`/`args`/`env` (or `httpUrl`/`headers`) into that file with the
user-supplied env-var values; from then on Cubi treats the server like any
other entry the user wrote by hand.

## Subcommands

```
cubi mcp search [<query>] [--json]
cubi mcp install <name> [--force] [--env K=V]... [--json]
cubi mcp uninstall <name> [--json]
cubi mcp test <server> [--tool <name>] [--json]
```

In the REPL the same operations are exposed as `/mcp-search`,
`/mcp-install`, `/mcp-uninstall`, and (existing) `/mcp-tools`. After an
install or uninstall the REPL automatically calls `/mcp-reload` so new tools
become available in the running session.

`install` prompts (via stdin) for every env var marked `required: true`.
Pass `--env KEY=value` one or more times to skip the prompts (required for
non-interactive use); the install will refuse to run if a required var is
still missing and stdin is not a TTY.

`install` validates the freshly-written entry by issuing a single
`tools/list` round-trip against the server (the same handshake
`cubi mcp test` uses). On failure the entry is **left in place** so the user
can edit env vars or args by hand and re-run `cubi mcp test <name>`.

## Schema

Each entry in `registry.json` is one JSON object with these fields:

| field | required | notes |
|---|---|---|
| `name` | yes | Short, kebab-cased identifier used as the `mcp.json` key. Must be unique within the registry. |
| `description` | yes | One-line summary shown by `cubi mcp search`. |
| `transport` | yes | `"stdio"` or `"http"`. |
| `command`, `args` | stdio only | Process to spawn. `args` defaults to `[]`. |
| `http_url`, `headers`, `oauth_provider` | http only | URL to POST to; optional static headers; optional OAuth provider key resolved from `~/.cubi/oauth.json`. |
| `env` | optional | Map of env-var name → `{ "required": bool, "description": string }`. Required vars are prompted at install time. |
| `homepage` | yes | Link to the upstream MCP server repo or docs. Must point at the source you copied the template from. |
| `license` | yes | SPDX identifier (e.g. `MIT`). |
| `tags` | optional | Free-form tags surfaced by search; conventionally include `official` for entries maintained by upstream MCP authors. |

### Worked example

```json
{
  "name": "github",
  "description": "Read/write GitHub issues, PRs, and repository contents via the GitHub API.",
  "transport": "stdio",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-github"],
  "env": {
    "GITHUB_PERSONAL_ACCESS_TOKEN": {
      "required": true,
      "description": "Personal access token; create at https://github.com/settings/tokens"
    }
  },
  "homepage": "https://github.com/modelcontextprotocol/servers/tree/main/src/github",
  "license": "MIT",
  "tags": ["git", "github", "official"]
}
```

When the user runs `cubi mcp install github`, Cubi prompts for
`GITHUB_PERSONAL_ACCESS_TOKEN`, writes:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..." }
    }
  }
}
```

into `~/.cubi/mcp.json`, then runs `tools/list` to confirm the server starts.

## Adding a new server

1. Open a PR that appends an entry to `docs/mcp/registry.json`. Keep the
   array sorted however you like — `search` always sorts results by name.
2. Run `cargo test --quiet`. The `tests/mcp_registry.rs` suite validates the
   schema (non-empty required fields, unique names, valid transport).
3. Set `homepage` to the canonical source repo of the server you're
   templating, and pick the smallest set of `env` vars that lets the server
   start. If a var is optional, mark `required: false` so it doesn't block
   non-interactive installs.

## Custom servers

The registry is for one-command setup of common servers. **Anything not in
the registry can still be added directly to `~/.cubi/mcp.json`** with the
same schema documented in [the MCP section of the README](../../README.md).
The registry never deletes or rewrites entries it didn't install.
