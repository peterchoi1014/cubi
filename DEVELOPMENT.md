# Developing Cubi

## Build

```bash
# Debug build (fast compile, slow runtime)
cargo build

# Release build (slow compile, fast runtime; LTO + single codegen unit)
cargo build --release
```

The release profile in [`Cargo.toml`](Cargo.toml) enables `lto = "thin"`,
`codegen-units = 1`, and `strip = "symbols"`, yielding a ~7.5 MB binary.

## Test, lint, format

```bash
cargo test --quiet                          # unit + integration suite
cargo clippy --all-targets -- -D warnings   # lints; CI gates on this
cargo fmt --all                             # also gated as --check in CI
```

A typical pre-push loop:

```bash
cargo fmt --all && cargo clippy --all-targets -- -D warnings && cargo test --quiet
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   CLI Interface                      │
│              (Colored Terminal UI)                   │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│                  AI Executor                         │
│         (Task Management & Coordination)             │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│            LLM Backend (Ollama or OpenAI)            │
│            (HTTP API Communication)                  │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│         Local Model Runtime (Ollama, llama.cpp,      │
│              LM Studio, etc.)                        │
└─────────────────────────────────────────────────────┘
```

## Project structure

```
cubi/
├── src/
│   ├── main.rs            # Application entry point
│   ├── cli/               # Terminal UI, command dispatch, agent-loop driver
│   ├── commands.rs        # Slash-command registry (single source of truth)
│   ├── agent_loop.rs      # Streaming tool-calling loop + `agent_run` meta-tool
│   ├── executor.rs        # AI executor & model switching
│   ├── llm.rs             # Provider abstraction (Ollama / OpenAI / Fake)
│   ├── ollama.rs          # Ollama HTTP client (streaming + tool calls)
│   ├── thinking_filter.rs # Strip <think>...</think> blocks from streams
│   ├── builtin_tools.rs   # bash, fs, web, repl, notebook, worktree, lsp, ...
│   ├── lsp_client.rs      # JSON-RPC client for the `lsp` built-in tool
│   ├── mcp_client.rs      # MCP transport (stdio + HTTP)
│   ├── mcp_config.rs      # ~/.cubi/mcp.json loader
│   ├── mcp_manager.rs     # MCP server lifecycle & tool routing
│   ├── permissions.rs     # Project trust store + plan-mode gates
│   ├── project_memory.rs  # CUBI.md discovery & loading
│   ├── memdir.rs          # Cross-session persistent memory store
│   ├── sessions.rs        # Per-project auto-saved session checkpoints
│   ├── todos.rs           # Per-project todo list
│   ├── hooks.rs           # Lifecycle hook registry (PreToolUse, etc.)
│   ├── file_mentions.rs   # `@file` mentions + user-defined Markdown commands
│   ├── git_cmds.rs        # Shell-out helpers for git slash commands
│   └── onboarding.rs      # First-run setup
├── tests/                 # Integration tests (assert_cmd)
├── docs/                  # User-facing docs and man page
├── Cargo.toml             # Dependencies + release profile
├── INSTALL.md             # Installation guide
├── DEVELOPMENT.md         # This file
└── README.md              # Overview + quick start
```

## Key components

- **CLI** (`cli/`) — terminal UI, slash-command dispatch, drives the streaming
  agent loop and persists sessions.
- **Slash-command registry** (`commands.rs`) — every command's trigger, usage,
  help line, and `Cmd` tag in one table so `/help`, the welcome banner, and
  dispatch can't drift apart.
- **Agent loop** (`agent_loop.rs`) — runs the tool-calling loop (≤12
  round-trips per turn) and exposes the `agent_run` meta-tool for subagents.
- **Built-in tools** (`builtin_tools.rs`) — shell, filesystem, web, REPL,
  notebook, git worktree, and LSP code-intel implementations.
- **Headless browser** (`browser_tool.rs`, feature `browser`) —
  chromiumoxide-backed session manager wired into `builtin_tools.rs` as
  the `browser_*` tool family. Gated behind the `browser` cargo feature;
  default off.
- **MCP layer** (`mcp_client.rs`, `mcp_config.rs`, `mcp_manager.rs`) — loads
  external MCP servers and exposes their tools to the model.
- **Permissions** (`permissions.rs`) — trust store consulted by every
  write/exec tool; also enforces `/plan` mode.
- **Memory & sessions** (`project_memory.rs`, `memdir.rs`, `sessions.rs`,
  `todos.rs`) — `CUBI.md` auto-injection, cross-session memdir, per-project
  session checkpoints, todos.
- **Executor & LLM clients** (`executor.rs`, `llm.rs`, `ollama.rs`) — model
  switching and streaming chat-completion calls with tool support.
- **Thinking filter** (`thinking_filter.rs`) — strips `<think>...</think>`
  blocks from Qwen3-family responses; `CUBI_KEEP_THINKING=1` disables.
- **Repo-map** (`repomap.rs`) — walks the project honoring `.gitignore`,
  parses Rust/Python/JS/TS with tree-sitter (or a regex fallback when the
  `tree-sitter` feature is off), and renders a compact file + symbol outline.
  Backs the `repo_map` built-in tool and the `/repomap` slash command;
  results are mtime-cached under `<cache_dir>/cubi/repomap/`.

## Main dependencies

See [`Cargo.toml`](Cargo.toml) for the complete list.

- `tokio` — async runtime
- `reqwest` — HTTP client (with `stream` feature for Ollama and MCP)
- `futures-util` — stream combinators for token-by-token streaming
- `serde` / `serde_json` — JSON serialization
- `colored` — terminal colors
- `rustyline` — readline-style input with history
- `anyhow` — error handling
- `dirs` — cross-platform home/config directory resolution

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/<short-name>`
3. Make focused commits — describe **what** changed and **why**; no
   "Co-authored-by" trailers for AI assistants
4. Run the full pre-push loop above
5. Open a PR against `main`

### Commit message conventions

Commit messages should describe the change and its rationale. Avoid:

- Process noise ("as requested", "per user instruction", agent logs/URLs)
- Co-authored-by trailers for LLMs or AI assistants
- References to internal tooling or instruction prompts

Use imperative mood in the subject line ("Add X", "Fix Y", "Refactor Z").

### Code review focus

Reviewers prioritize:

1. **Functional bugs** — logic errors, crashes, concurrency issues, security
2. **Regressions** — behavior changes in existing features, removed test
   coverage, breaking CLI/config changes
3. **Idiomatic Rust** — `Result` error handling, avoiding `unwrap`/`expect`,
   ownership clarity, `cargo fmt` / `cargo clippy -D warnings` clean
4. **Tests** — added or updated for new/changed behavior

Style-only nits are not surfaced unless they materially affect readability.
