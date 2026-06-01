<p align="center">
  <img src="docs/assets/cubi-banner.png" alt="Cubi — a pocket-sized AI for your shell" width="100%">
</p>

# Cubi

A pocket-sized AI for your shell. Cubi is a Rust-based command-line AI chat
application with local model inference through Ollama (or any OpenAI-compatible
local server), a streaming native-tool-calling agent loop, and MCP support.

<div align="center">

![Rust](https://img.shields.io/badge/rust-1.92.0-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)

</div>

## ✨ Features

- 🤖 **Local AI** — runs fully offline against Ollama, llama-server, LM Studio,
  or any OpenAI-compatible local backend
- ⚡ **Streaming agent loop** — tokens stream live; the model calls built-in or
  MCP tools, sees the results, and keeps going (up to 12 round-trips per turn)
- 🧰 **Built-in tools** — shell, filesystem, git, web fetch/search, long-lived
  bash REPL, Jupyter notebooks, LSP code-intel, OS notifications, and a
  meta-`agent_run` tool for spawning focused subagents
- 🔌 **MCP support** — load external Model Context Protocol servers from
  `~/.cubi/mcp.json` and call their tools alongside built-ins
- 🧰 **MCP registry** — `cubi mcp search/install/uninstall` for one-command
  setup of common Model Context Protocol servers (filesystem, github,
  gitlab, slack, sqlite, postgres, fetch, time, …); see
  [`docs/mcp/registry.md`](docs/mcp/registry.md)
- 🛡️ **Trust + plan mode + admin policy** — every write/exec path is gated by
  per-directory trust; `/plan` switches to a read-only mode; admins can ship a
  policy file with a tool deny-list
- 🧠 **Project memory + sessions** — auto-injected `CUBI.md` per project,
  cross-session persistent notes, auto-checkpointed sessions with
  `/resume`/`/fork`/`/rewind`/`/compact`
- 🌿 **Git workflow** — `/diff`, `/commit`, `/commit-push-pr`, `/review`,
  `/worktree`, `/branch`, `/tag`, `/autofix-pr`, `/pr_comments` shell out to
  your installed `git`/`gh`
- 🧩 **Plugins + skills + hooks** — reusable Markdown skill packs, namespaced
  slash-command bundles, and `PreToolUse`/`PostToolUse`/`UserPromptSubmit`/etc.
  lifecycle hooks
- 🗺️ **Repo-map** — tree-sitter-based outline of the project's symbols,
  available as the `repo_map` tool and `/repomap` slash command
- 🌐 **Headless-browser tool** (feature `browser`) — `browser_open` /
  `browser_eval` / `browser_screenshot` / `browser_text` / `browser_close`
  backed by chromiumoxide for web debugging tasks. Off by default to keep the
  lean binary; enable with `cargo install --features browser`.
- 🔐 **Tamper-evident receipts** (`--receipts <path>`) — hash-chained JSONL
  audit log of every tool call and lifecycle event; optional Ed25519 signing
  via `cubi keys init`. Verify with `cubi verify-receipts`.

## 🚀 Quick Start

```bash
# 1. Install Ollama and pull the default model
brew install ollama && ollama serve &
ollama pull qwen3:8b

# 2. Build and run cubi
git clone https://github.com/peterchoi1014/cubi.git
cd cubi && cargo install --path .
cubi
```

You should see:

```
   ┌───────┐
   │ ▣   ▣ │   Cubi
   │   ◡   │   a pocket-sized AI for your shell
   └───────┘
    ░░░░░░░

Type /help to list all available slash commands.
Start chatting! (Ctrl+C to interrupt, /quit to exit)

You:
```

For full installation, model selection, and non-Ollama backend setup, see
**[INSTALL.md](INSTALL.md)**.

## 📖 Usage

Inside the REPL, type `/help` to list every slash command, or `/help <cmd>` for
per-command detail. The full command surface is the single source of truth in
[`src/commands.rs`](src/commands.rs).

A few common ones to get started:

| Command | Description |
| --- | --- |
| `/model [name]` | Show or switch the active model |
| `/save <f.json>` / `/load <f.json>` | Persist or restore the conversation |
| `/sessions` / `/resume [id]` | List or resume auto-saved sessions |
| `/plan` | Toggle plan mode (read-only) |
| `/diff` / `/commit <msg>` / `/review` | Git workflow shortcuts |
| `/doctor` | Run environment health checks |
| `/quit` | Exit |

Command-line use (no REPL):

```bash
cubi -p "summarize this repo"           # one-shot prompt
echo "what is rust?" | cubi             # piped stdin
cubi --json -p "list files"             # line-delimited JSON events
cubi --resume                           # resume the latest session
cubi --list-sessions                    # list all saved sessions
cubi completions bash                   # shell completions
```

See the [headless cookbook](docs/headless.md) for scripts/pipelines, exit
codes, and the JSON event schema.

## 📚 Documentation

- **[INSTALL.md](INSTALL.md)** — prerequisites, install steps, model selection,
  non-Ollama backends
- **[DEVELOPMENT.md](DEVELOPMENT.md)** — build, test, lint, project structure,
  contributing
- [`docs/headless.md`](docs/headless.md) — scripts, JSON output, exit codes
- [`docs/sessions.md`](docs/sessions.md) — saved sessions: resume, delete, prune
- [`docs/plugins.md`](docs/plugins.md) — plugin discovery & command authoring
- [`docs/troubleshooting.md`](docs/troubleshooting.md) — common startup/MCP/UX
  issues
- `docs/cubi.1` — roff man page (`man cubi` after install)

## 🐛 Troubleshooting

The most common ones:

- **`could not connect to localhost:11434`** — Ollama isn't running. Start it
  with `ollama serve`.
- **`Model 'X' not found`** — `ollama pull X`, then re-run `cubi`.
- **Need a debug trace** — `CUBI_LOG=cubi=debug cubi -p "..."` or pass
  `--debug` for full cause chains.

For everything else (auth, rate limits, MCP server failures, slow responses,
etc.), see [`docs/troubleshooting.md`](docs/troubleshooting.md) or run
`cubi doctor`.

## 🗺️ Roadmap

Highlights still to come:

- [ ] RAG (Retrieval Augmented Generation) support
- [ ] Multi-modal support (images, audio)
- [ ] Web interface
- [ ] Distributed inference across remote workers
- [ ] Conversation search and tagging
- [ ] Export to additional formats (PDF)
- [ ] Cross-platform shell tool (`pwsh` on Windows)

## 🤝 Contributing

Contributions welcome — see [DEVELOPMENT.md](DEVELOPMENT.md) for the build/test
loop and code conventions. Open a PR against `main`.

## 📝 License

MIT — see [LICENSE](LICENSE).

## 🙏 Acknowledgments

- **[Ollama](https://ollama.ai/)** — Local AI model runtime
- **Rust community** — For amazing tools and libraries

---

<div align="center">

**Built with ❤️ using Rust and Ollama**

[Report Bug](https://github.com/peterchoi1014/cubi/issues) · [Request Feature](https://github.com/peterchoi1014/cubi/issues)

</div>
