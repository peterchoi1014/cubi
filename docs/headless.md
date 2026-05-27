# Headless cubi

Use headless mode when cubi is part of a script or pipeline.

- Inline prompt: `cubi -p "summarize this repo"` or `cubi --prompt "..."`.
- Piped prompt: `git diff | cubi -p "summarize"` keeps the diff on stdin for shell composition; without `-p`, cubi reads piped stdin as the prompt: `printf 'hello' | cubi`.
- System prompt: `cubi --system ./system.txt -p "review"` prepends file contents as instructions.
- JSON events: `cubi --json --no-stream -p "run tests"` emits line-delimited `token`, `tool_call`, `tool_result`, `tool_timeout`, `compacted`, `budget_error`, and `done` events.
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

Exit codes: `0` ok, `2` usage/config, `10` model/API, `11` tool failure, `12` context-window budget exceeded, `130` cancelled.

Examples:

```sh
git diff | cubi -p "summarize the risky changes"
cubi --json --no-stream -p "list the failing checks" | jq -c .
cat release-notes.md | cubi --system tone.txt
```
