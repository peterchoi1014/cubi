# ai-chat-cli — Roadmap

This roadmap captures the implementation plan derived from a prior architectural
review and a feature inventory of the recovered
[`peterchoi1014/claude-code`](https://github.com/peterchoi1014/claude-code)
source tree.

> ⚠️ **Important caveat.** The `peterchoi1014/claude-code` repository contains
> code **recovered from a leaked source map of Anthropic's proprietary product**
> (see its own README's "Legal / ethical note"). It is used here **only as
> architectural inspiration** — we copy ideas, names, command surfaces, and file
> layout, but **do not copy code verbatim** into `ai-chat-cli`. All
> implementations must be original work written against public Anthropic
> documentation, the MCP spec, and our own design.

Items already shipped in this PR are marked `[x]`. Everything else is open work
for future PRs.

---

## A. Built-in tools to add (`src/builtin_tools.rs` extensions)

- [ ] **Subagent / Task tool** — main model spawns isolated worker agents
      (`AgentTool`, `TaskCreate/Get/List/Update/Stop/OutputTool`)
- [x] **`TodoWrite` tool** + `/todos` UI — persistent per-session task checklist
      *(now persisted across restarts at `~/.ai-chat-cli/todos/<cwd-key>.json`)*
- [x] **Plan mode** — read-only "plan first, then apply" toggle
      *(`/plan` toggle now gates `/mcp-call`; full tool gating expands as more
      write tools land)*
- [x] **`ask_user` tool** — model pauses and asks a clarifying question
      *(user-driven `/ask <question>` for now; model-triggered version comes
      with native tool-calling)*
- [ ] **Git worktree tool** + `/worktree` — isolated worktree for risky changes
- [ ] **`web_fetch`, `web_search`** — network tools (permission-gated)
- [ ] **LSP-backed code intel tool** — hover / definition / references
- [ ] **Jupyter notebook tool** — cell-level edits to `.ipynb`
- [ ] **Persistent REPL tool** — long-lived shell / Python / Node session
- [ ] **Cross-platform shell tool** — `bash` vs `pwsh` based on host OS
- [ ] **Time tools** — `Sleep`, `ScheduleCron`
- [ ] **Skills system** — reusable Markdown skill packs in
      `~/.ai-chat-cli/skills/*` + `SkillTool`
- [ ] **Tool-search tool** — model searches the registry instead of receiving
      the full tool list in every prompt
- [ ] **Structured output helpers** — `Brief`, `SyntheticOutput`
- [ ] **Inter-agent messaging** — `SendMessage`, `RemoteTrigger`
- [ ] **MCP resources & prompts** — `resources/list`, `resources/read`,
      per-server OAuth, interactive approval

→ Foundation: central tool registry (analogous to leaked `src/tools.ts` /
`src/Tool.ts`) + `enabled_tools` config so users can disable any tool.

---

## B. Slash commands to add (`src/cli.rs` / `src/commands/`)

Currently shipped (pre-PR): `/help`, `/clear`, `/history`, `/model`, `/save`,
`/load`, `/batch`, `/mcp-tools`, `/mcp-call`, `/mcp-reload`, `/quit`.

Shipped in this PR:

- [x] `/plan`, `/todos`, `/todo-add`, `/todo-done`, `/todo-rm`, `/todo-clear`
- [x] `/init`, `/memory`, `/memory-reload`
- [x] `/status`, `/version`, `/export` (with overwrite protection)
- [x] `/ask` (user-driven clarifying-question stand-in, single-turn)
- [x] `/sessions`, `/resume` (auto-saved per-project checkpoints)
- [x] `/diff`, `/commit`, `/review` (git workflow; `/commit` is plan-mode-aware)
- [x] `/trust` (project-trust gate for write/exec tools)

Still to add (grouped by area):

- **Project / workspace:** `/add-dir`, `/files`, `/init-verifiers`
- **Git workflow:** `/commit-push-pr`, `/branch`, `/tag`, `/pr_comments`,
  `/security-review`, `/autofix-pr`, `/issue`, `/undo`
- **Agent control:** `/agents`, `/tasks`, `/teleport`, `/rewind`, `/passes`,
  `/effort`, `/compact`
- **Output / theming:** `/theme`, `/color`, `/output-style`, `/statusline`,
  `/keybindings`, `/vim`
- **Auth / accounts:** `/login`, `/logout`, `/oauth-refresh`,
  `/privacy-settings`
- **MCP / plugins / skills:** `/mcp`, `/plugin`, `/reload-plugins`, `/skills`,
  `/hooks`
- **Diagnostics / perf:** `/stats`, `/usage`, `/cost`, `/perf-issue`,
  `/heapdump`, `/debug-tool-call`, `/env`, `/doctor`, `/bug`, `/permissions`,
  `/config`
- **Lifecycle:** `/upgrade`, `/install`, `/install-github-app`,
  `/install-slack-app`, `/sandbox-toggle`, `/reset-limits`
- **Social / sharing:** `/share`, `/copy`, `/feedback`, `/release-notes`,
  `/stickers`

Foundation work: the flat `match` in `cli.rs::handle_command` is now a
`SlashCommand` registry (`src/commands.rs`) — adding a command requires a row
in `COMMANDS` and an exhaustive arm on `Cmd`. Still to do: user-defined
Markdown commands as first-class plugins (cf. leaked
`createMovedToPluginCommand.ts`).

---

## C. New subsystems / modules

1. **Onboarding** (`bootstrap/`, `setup.ts`, `projectOnboardingState.ts`) —
   first-run flow: pick model, scan project, write `AICHAT.md`, set trust
   level. ✅ Shipped: `src/onboarding.rs` runs once, lets the user pick a
   model from `ollama list`, offers `/trust`, offers `AICHAT.md`. Persisted
   to `~/.ai-chat-cli/config.json`.
2. **Permissions system** (`utils/permissions/`) — project trust, per-tool
   allow/deny, "trust this folder" prompts, enterprise-managed policy. ✅
   Foundation shipped: `src/permissions.rs` enforces a per-project trust
   store with a path sandbox; `bash`, `edit_file`, `write_file` are gated.
   Per-tool allow/deny lists and enterprise policy are still open.
3. **Memory & compaction** (`services/compact/`, `SessionMemory/`,
   `extractMemories/`, `memdir/`) — automatic in-session compaction plus
   cross-session persistent memory at `~/.ai-chat-cli/memdir/`.
4. **Proactive completions** (`PromptSuggestion/`, `fileSuggestions.ts`) —
   suggest next prompts and `@file` references while the user types.
5. **Multi-agent layer** (`utils/swarm/`, `coordinator/`, `assistant/`,
   `tasks/`) — teammates over tmux / iTerm / in-process backends.
6. **API hardening** (`services/api/`, `claudeAiLimits*`, `tokenEstimation.ts`)
   — token estimator, rate limiter, retry-with-backoff in `ollama.rs`.
7. **LSP bridge** (`services/lsp/`) — diagnostics after `edit_file`,
   jump-to-definition.
8. **Notifications + sleep prevention** (`notifier.ts`, `preventSleep.ts`) —
   OS notifications on long-run completion, `caffeinate`-style awake.
9. **Interactive MCP approval** (`mcpServerApproval.tsx`).
10. **Telemetry / debug log** (`analytics/`, `diagnosticTracking.ts`,
    `internalLogging.ts`) — opt-in; aligns with future `tracing` proposal.
11. **Voice input** (`voice.ts`, `voiceStreamSTT.ts`, `voiceKeyterms.ts`).
12. **Vim mode + remappable keybindings** (`vim/`, `keybindings/`).
13. **Themable output styles** (`outputStyles/`) — concise / explanatory /
    markdown / etc., per-session.
14. **Hooks** — `PreToolUse`, `PostToolUse`, `UserPromptSubmit`, `Stop`,
    `SessionStart`, `Notification`.
15. **Plugin system** (`plugins/`, `services/plugins/`, `reload-plugins`) —
    `~/.ai-chat-cli/plugins/*` discoverable bundles.
16. **Headless / remote mode** (`server/`, `remote/`, `upstreamproxy/`,
    `bridge/`) — daemon + remote client; pairs naturally with the unused
    `distributed.rs` (Repartir).
17. **Auto-saved sessions + checkpointing** (`sessionStorage.js`,
    `sessionStart.js`, `history.ts`) — `/resume`, `/rewind` with file-mutation
    rollback. ✅ Foundation shipped: `src/sessions.rs` auto-checkpoints to
    `~/.ai-chat-cli/sessions/<cwd-key>/<id>.json` after every turn; `/sessions`
    + `/resume [id]` are wired up. `/rewind` and file-mutation rollback are
    still open.
18. **TUI rewrite** (`screens/`, `components/`, Ink → `ratatui`) — panes for
    chat / tool output / todos / status line.
19. **Deep-link / browser integration** (`claudeInChrome/`, `deepLink/`,
    `chrome` command) — `aichat://` URLs, Chrome native-messaging bridge.
20. **Tip-of-the-day + release notes** (`buddy/`, `tips/`, `release-notes`).
21. **Centralized schemas** (`schemas/`, `types/`) — stricter validation than
    `serde_json::Value` everywhere.
22. **Migrations** — versioned config / session migration framework.
23. **Enterprise policy** (`policyLimits/`, `remoteManagedSettings/`) —
    admin-pushed config, tool denylists.
24. **Settings sync** (`settingsSync/`) — cross-machine sync via Git repo.

---

## D. Implementation priorities

1. Agent loop + native tool-calling + streaming.
2. Permissions system + path sandboxing + project-trust prompt. ✅ Shipped.
3. **Plan mode + `TodoWrite` + `AskUserQuestion`.** ✅ Plan mode now gates all
   built-in write/exec tools; `/todos` and `/ask` already in place.
4. `/init` + `AICHAT.md` + memdir + onboarding flow. ✅ Onboarding wizard
   shipped; memdir (cross-session persistent memory) is still open.
5. Auto-saved sessions + `/resume` + `/rewind` checkpoints + compaction.
   ✅ Sessions + `/resume` shipped; `/rewind` and compaction are still open.
6. Slash-command registry + custom Markdown commands + `@file` mentions +
   prompt suggestions. ✅ Registry shipped (`src/commands.rs`); user-defined
   Markdown commands and `@file` mentions still open.
7. Subagents (`AgentTool`) + task management tools.
8. Git tools: `/commit`, `/commit-push-pr`, `/diff`, `/review`, worktree tools.
   ✅ `/diff`, `/commit`, `/review` shipped (`src/git_cmds.rs`).
9. Multi-provider LLM abstraction + token estimator + rate-limit / retry.
10. Web tools (`web_fetch`, `web_search`) + LSP service & tool + REPL tool +
    notebook tool.
11. Hooks, plugins, skills, MCP resources / prompts / OAuth, MCP approval UI.
12. TUI (ratatui) rewrite with panes, vim keybindings, output styles, themes,
    statusline.
13. Headless / remote / server mode, deep-link integration, voice,
    notifications.
14. Telemetry, migrations, enterprise policy, settings sync.
15. Tests, CI, tracing, `clap` flags + config file (cross-cutting; do alongside
    everything).
