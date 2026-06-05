# Troubleshooting cubi

- Ollama not running: start it with `ollama serve`, then retry. Headless model/API failures exit with code `10`.
- Model struggles with tools: switch to a tool-capable model such as `qwen3:4b` using `/model qwen3:4b` or `CUBI_MODEL=qwen3:4b`.
- MCP dots in the banner: `MCP: ●1 ●1(name) ●0(disabled)` means one server is healthy and one failed. Check `~/.cubi/mcp.json`, auth headers/OAuth, and whether the server process or URL is reachable. For common servers, `cubi mcp install <name>` writes a known-good template and validates it via a `tools/list` round-trip; see [`mcp/registry.md`](mcp/registry.md).
- Tool timed out: increase `tool_timeout_secs` in `~/.cubi/config.json` or pass `_timeout_secs` in a tool call when appropriate.
- No color or too much color: set `NO_COLOR=1` or `CUBI_COLOR=off`; set `CUBI_COLOR=on` to force color.
- Completions missing: run `cubi completions bash`, `cubi completions zsh`, or `cubi completions fish` and install the printed script using your shell's completion directory.
- Internal diagnostic logs: set `CUBI_LOG` to enable structured tracing on
  stderr. Examples: `CUBI_LOG=cubi=debug cubi -p "hi"` for everything,
  `CUBI_LOG=cubi::llm=debug,cubi::mcp=warn cubi` for targeted modules.
  Output goes to stderr only — JSON output on stdout stays clean. With
  `CUBI_LOG` unset, no subscriber is installed and the binary is quiet.
