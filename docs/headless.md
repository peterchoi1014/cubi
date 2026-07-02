# Headless cubi

Use headless mode when cubi is part of a script or pipeline.

- Inline prompt: `cubi -p "summarize this repo"` or `cubi --prompt "..."`.
- Piped prompt: `git diff | cubi -p "summarize"` keeps the diff on stdin for shell composition; without `-p`, cubi reads piped stdin as the prompt: `printf 'hello' | cubi`.
- System prompt: `cubi --system ./system.txt -p "review"` prepends file contents as instructions.
- JSON events: `cubi --json --no-stream -p "run tests"` emits line-delimited `token`, `tool_call`, `tool_result`, `tool_timeout`, `compacted`, `budget_error`, `consensus_start`, `consensus_subagent_result`, `consensus_decision`, and `done` events.
- Streaming: one-shot mode buffers by default for predictable scripts; add `--stream` for live tokens.

## `cubi exec` shorthand

`cubi exec <prompt words>` is the script-friendly entry point. It joins
the remaining argv with single spaces and runs the same code path as
`-p "<joined>" --json --no-stream --no-banner`, so each invocation emits
one JSON event per line and exits when the model is done.

```sh
cubi exec list the riskiest files in HEAD | jq -c .
cubi --system ./tone.txt exec rewrite this paragraph more concisely
```

Flags meant to influence the run (e.g. `--system`, `--model`) must
appear *before* `exec`; everything after `exec` is treated as prompt
text.

Exit codes: `0` ok, `2` usage/config, `10` model/API, `11` tool failure, `12` context-window budget exceeded, `13` network (connect refused / DNS / TLS), `130` cancelled.

## Error events

The `error` JSON event has been extended (backward-compatibly) with
optional `kind`, `exit_code`, and `hint` fields. The legacy `message`
field is preserved unchanged.

```json
{"type":"error","message":"could not connect to localhost:11434","kind":"connect_refused","exit_code":13,"hint":"is `ollama serve` running on localhost:11434?"}
```

Known `kind` values: `config`, `auth`, `quota`, `rate_limited`,
`connect_refused`, `dns`, `tls`, `timeout`, `server_error`,
`bad_request`, `cancelled`, `tool`, `budget`, `other`. The set may
grow; consumers should treat unknown values as `other`.

Set `--debug` (or `CUBI_DEBUG=1`, or any non-empty `RUST_BACKTRACE`)
to also print the full anyhow cause chain to stderr in non-JSON mode.

## Quiet output

Headless / JSON mode automatically suppresses every decorative element:
the startup banner, the tool spinner, slash-command edits via `/edit`,
and color codes (when piped or with `NO_COLOR`). Three orthogonal env
knobs let interactive users disable the same affordances individually:

- `CUBI_NO_BANNER=1` — skip the one-line startup banner.
- `CUBI_NO_SPINNER=1` — disable the elapsed-time spinner around tool
  calls. Also honored by `NO_COLOR` and `CUBI_NO_COLOR`.
- `CUBI_EDITOR=…` — pin the editor `/edit` opens (otherwise falls back
  through `$VISUAL`, `$EDITOR`, and the platform default).
- `CUBI_NUM_CTX=…` — raise the Ollama runtime context cap (`num_ctx`).
  cubi derives `num_ctx` from the model's known window, clamped to a safe
  default; a positive override raises that ceiling (still bounded by the
  model's real window).
- `CUBI_READ_FILE_MAX_LINES=…` / `CUBI_READ_FILE_MAX_BYTES=…` — tune the
  cap applied to unbounded `read_file` calls (defaults 400 lines / 50 KiB).
  Raise for large-context models; lower for small-context ones. Ranged
  reads (`start_line`/`end_line`) are never capped.

Examples:

```sh
git diff | cubi -p "summarize the risky changes"
cubi --json --no-stream -p "list the failing checks" | jq -c .
cat release-notes.md | cubi --system tone.txt
```

## Structured event tap (`--events <path>`)

`--events <path>` (or `CUBI_EVENTS=…`) opens the path in append+create
mode and writes one JSON line per internal event. The shape is a strict
superset of `--trace-tools` — it captures full turn lifecycles plus tool
rationales and MCP transitions in addition to tool start/complete. If
the path can't be opened, cubi prints one warning (suppressed in JSON
mode) and continues without the tap rather than aborting the run.

Event types currently emitted:

- `turn_start` — `{type, ts, turn, model}` at the top of each agent turn.
- `tool_call_start` — `{type, ts, tool, args}` with secrets redacted
  through the same `redact_secrets` helper as `--trace-tools` and
  `--print-config`.
- `tool_rationale` — `{type, ts, tool, rationale}` when
  `--explain-tools` (or `CUBI_EXPLAIN_TOOLS=1`) is set; rationale is the
  assistant message that accompanied the tool call, falling back to the
  MCP manifest description, then to `(no description)`.
- `tool_call_complete` — `{type, ts, tool, ok, result_chars}`.
- `mcp_status_change` — `{type, ts, before, after}` where each side is
  `{ok, failed, not_loaded}`. Emitted when the agent loop detects an
  MCP server transition during a turn.
- `turn_end` — `{type, ts, usage, model}`. `usage` carries
  `{prompt_tokens, completion_tokens, elapsed_ms}`.
- `consensus_start` — `{type, ts, goal, models, strategy}`. Emitted
  before any subagent dispatches when the `consensus_run` meta-tool
  (or the `/consensus` slash command) starts a run. `strategy` is one
  of `vote`, `best-of-n`, `judge`.
- `consensus_subagent_result` — `{type, ts, model, ok, steps_used,
  elapsed_ms, prompt_tokens, completion_tokens, error?}`. One event per
  subagent. `error` is present and non-null only when `ok` is false.
- `consensus_decision` — `{type, ts, winner_model, decision_reason}`
  after arbitration completes. `decision_reason` is free-form text
  (e.g. "majority vote 2/3", "judge `qwen3:8b` picked `devstral`: …",
  "best-of-n: `devstral` scored 9 (judge `…`)").

`--trace-tools <path>` still produces its original `tool_start` /
`tool_complete` record shape for back-compat, but new integrations
should prefer `--events`; the same redaction rules apply.

## Tamper-evident receipts (`--receipts <path>`)

`--receipts <path>` (or `CUBI_RECEIPTS=…`, or `receipts` in
`~/.cubi/config.json`; precedence is flag > env > config) opens a
hash-chained JSONL audit log. Every session, user message, tool call,
tool result, assistant message, and session end produces one append-only
entry whose `this_hash` covers `prev_hash` plus the canonical
serialization of the rest of the record. Full args/results live in
`<path>.payloads/<sha256>.json` so the receipts file itself stays small
and grep-able.

If `cubi keys init` has been run, every subsequent entry is signed with
the Ed25519 key under `~/.cubi/keys/`. Verify a log (and optionally its
signatures) with:

```sh
cubi verify-receipts /path/to/r.jsonl                         # chain + payloads
cubi verify-receipts /path/to/r.jsonl --pub-key ~/.cubi/keys/ed25519.pub
cubi verify-receipts /path/to/r.jsonl --no-verify-payloads --json
```

Exit codes: `0` ok, `2` tamper detected (chain break, payload mismatch,
or signature mismatch — the offending `seq` is reported on stderr), `13`
I/O error. Receipts are a side-channel: when the file can't be written
mid-session, cubi degrades to a single `tracing::warn!`, disables the
writer for the rest of the process, and the session continues. See
`DEVELOPMENT.md` § "Receipts format" for the on-disk shape and
canonical-serialization algorithm.

> ⚠️ **Security note.** Unlike `--events`/`--trace-tools`, which run
> tool arguments and results through `redact_secrets` before
> emitting, the payload sidecars under `<path>.payloads/` store the
> **raw** tool args and results — including any API keys, tokens, file
> contents, or other secrets the session touches. Treat the receipts
> file and its `*.payloads/` directory as **sensitive**:
>
> - Keep them on a private filesystem with restrictive permissions
>   (`chmod 600` / a user-only directory).
> - Add the receipts path and `*.payloads/` to `.gitignore` (and to
>   any container image / CI artifact exclusion lists) so they are
>   never committed or uploaded.
> - Rotate or scrub them before sharing a session log with anyone
>   else.

