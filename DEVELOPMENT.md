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

### Regression bench suite

`cubi bench` runs a small curated set of self-contained tasks against
any local model and writes per-task results + a `summary.json`. The
quick suite (easy tasks only) runs nightly in `.github/workflows/bench.yml`
against `qwen3:8b`. See [`bench/README.md`](bench/README.md) for the
task format and how to add a new task. Regular CI does **not** run
`cubi bench` because it has no local model; the harness itself is
covered by unit tests in `src/bench.rs` and `tests/bench.rs`.

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
│   ├── consensus.rs       # Multi-model `consensus_run` meta-tool + arbitration
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
- **Consensus** (`consensus.rs`) — backs the `consensus_run` meta-tool and
  `/consensus` slash command: spawns N subagents in parallel against
  caller-supplied models, then arbitrates via `vote`, `best-of-n`, or
  `judge`. Subagents are LLM-only (no tools) in the MVP so the parallel
  dispatch needs no shared-mutable `McpManager` state. Anti-recursion
  is enforced via `agent_loop::without_meta_tools`, which strips both
  meta-tools in lockstep so neither can be invoked from a subagent.
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

## Receipts format

`--receipts <path>` writes a tamper-evident, append-only JSONL audit
log. The intent is that an auditor (or a CI tool, or an external
verifier in Python / Go) can re-walk the chain offline and prove that
no entry has been mutated since the session ran.

### File layout

```
<path>                          one JSON object per line, append-only
<path>.payloads/<sha256>.json   the full payload referenced by each entry
```

The receipts file stays small and grep-able; large blobs (tool args,
tool results) live in the sidecar directory keyed by the SHA-256 hash
the receipts entry commits to.

### Entry shape

```json
{
  "seq": 42,
  "ts": "2026-05-31T12:35:31Z",
  "event": "tool_call",
  "name": "bash",
  "payload_sha256": "abc...",
  "prev_hash": "def...",
  "this_hash": "ghi...",
  "sig": "base64-encoded ed25519 signature of this_hash (optional)"
}
```

`event` is one of `session_start`, `user_message`, `tool_call`,
`tool_result`, `assistant_message`, `session_end`. `name` appears on
`tool_call` / `tool_result`; `tool_result` also carries `ok: bool`;
`session_start` carries `model` and `cwd`. `prev_hash` is `""` for the
first entry of a fresh file (cubi recovers the chain tip if the file
already exists, so resuming appends rather than forking).

### Hash chain

```
this_hash = sha256( prev_hash_utf8_bytes || canonical(record_without_this_hash_or_sig) )
```

Both inputs are concatenated byte-wise (no separator, no length
prefix). The output is hex-encoded lowercase, 64 chars.

### Canonical serialization

The canonical form is byte-stable sorted-key JSON with no whitespace.

- **Objects**: keys sorted in lexicographic (`BTreeMap`) order; each
  key serialized with `serde_json`'s standard string escaping;
  `key`, `:`, `value`, `,` joined directly.
- **Arrays**: items in input order, joined by `,`, surrounded by
  `[ ]`.
- **Strings / numbers / bools / null**: serialized exactly as
  `serde_json` would (no extra whitespace).

Equivalent in pseudo-Python:

```python
def canonical(v):
    if isinstance(v, dict):
        return b"{" + b",".join(
            json.dumps(k, ensure_ascii=False).encode() + b":" + canonical(v[k])
            for k in sorted(v)
        ) + b"}"
    if isinstance(v, list):
        return b"[" + b",".join(canonical(x) for x in v) + b"]"
    return json.dumps(v, ensure_ascii=False).encode()
```

`payload_sha256` is `sha256(canonical(payload_blob))` over the full
payload object (the one written to `<path>.payloads/<hash>.json`). A
verifier re-reads the sidecar, re-applies `canonical`, and re-hashes
it to confirm the claim.

### Signatures

When `cubi keys init` has been run, every entry's `this_hash` is signed
with the Ed25519 private key at `~/.cubi/keys/ed25519.priv`. The
signature is over the ASCII bytes of the lowercase-hex `this_hash`
string (not the raw 32-byte digest) so a verifier can sign / verify
without re-hashing. The base64-encoded 64-byte signature is stored in
the entry's `sig` field.

The matching public key is written to `~/.cubi/keys/ed25519.pub` in the
standard `ssh-ed25519 <base64-wire-format> cubi-receipts` single-line
shape, so `cubi verify-receipts --pub-key …` (and any other tool that
already speaks SSH wire format) can consume it directly.

### Failure mode

The receipts side-channel must NEVER block tool execution. If a write
fails mid-session (full disk, deleted directory, etc.), cubi logs one
`tracing::warn!` to stderr and the session continues without further
audit entries. The on-disk chain is still verifiable up to the last
successful write.
