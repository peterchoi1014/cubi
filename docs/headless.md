# Headless cubi

Use headless mode when cubi is part of a script or pipeline.

- Inline prompt: `cubi -p "summarize this repo"` or `cubi --prompt "..."`.
- Piped prompt: `git diff | cubi -p "summarize"` keeps the diff on stdin for shell composition; without `-p`, cubi reads piped stdin as the prompt: `printf 'hello' | cubi`.
- System prompt: `cubi --system ./system.txt -p "review"` prepends file contents as instructions.
- JSON events: `cubi --json --no-stream -p "run tests"` emits line-delimited `token`, `tool_call`, `tool_result`, `tool_timeout`, and `done` events.
- Streaming: one-shot mode buffers by default for predictable scripts; add `--stream` for live tokens.

Exit codes: `0` ok, `2` usage/config, `10` model/API, `11` tool failure, `130` cancelled.

Examples:

```sh
git diff | cubi -p "summarize the risky changes"
cubi --json --no-stream -p "list the failing checks" | jq -c .
cat release-notes.md | cubi --system tone.txt
```
