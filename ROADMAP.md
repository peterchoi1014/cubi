# cubi — Roadmap

This roadmap captures the implementation plan derived from a prior architectural
review and a feature inventory of the recovered
[`peterchoi1014/claude-code`](https://github.com/peterchoi1014/claude-code)
source tree.

> ⚠️ **Important caveat.** The `peterchoi1014/claude-code` repository contains
> code **recovered from a leaked source map of Anthropic's proprietary product**
> (see its own README's "Legal / ethical note"). It is used here **only as
> architectural inspiration** — we copy ideas, names, command surfaces, and file
> layout, but **do not copy code verbatim** into `cubi`. All
> implementations must be original work written against public Anthropic
> documentation, the MCP spec, and our own design.

Items already shipped are marked `[x]`. Everything else is open work
for future PRs. Current crate version: **0.3.0**.

---

## A. Built-in tools to add (`src/builtin_tools.rs` extensions)

- [x] **Subagent / Task tool** — main model spawns isolated worker agents
      *(`agent_run` meta-tool: fresh context, same tools minus `agent_run`, step cap)*
- [x] **`TodoWrite` tool** + `/todos` UI — persistent per-session task checklist
      *(now persisted across restarts at `~/.cubi/todos/<cwd-key>.json`)*
- [x] **Plan mode** — read-only "plan first, then apply" toggle
      *(`/plan` toggle now gates `/mcp-call`; full tool gating expands as more
      write tools land)*
- [x] **`ask_user` tool** — model pauses and asks a clarifying question
      *(user-driven `/ask <question>` for now; model-triggered version comes
      with native tool-calling)*
- [x] **Git worktree tool** — `worktree` builtin (list/add/remove), auto-trusts
      new paths; `/worktree` slash command also shipped
- [x] **`web_fetch`, `web_search`** — network tools (permission-gated)
      *(HTTP(S) GET with 64 KB cap; DuckDuckGo lite-mode scrape — no API key)*
- [x] **LSP-backed code intel tool** — hover / definition / references
      *(`lsp` builtin: caller specifies server command + 1-based line/col)*
- [x] **Jupyter notebook tool** — cell-level edits to `.ipynb`
      *(`notebook` builtin: list/read/insert/replace/delete; pure JSON, no Jupyter dep)*
- [x] **Persistent REPL tool** — long-lived shell / Python / Node session
      *(bash-only for now; `repl_start` / `repl_eval` / `repl_close`)*
- [x] **Cross-platform shell tool** — `bash` vs `pwsh` based on host OS
      *(`shell` builtin: POSIX `sh -c` on Unix, `pwsh -Command` on Windows;
      same plan-mode + trust gates as `bash`)*
- [x] **Time tools** — `Sleep`, `ScheduleCron`
      *(`sleep` blocks up to 60s; `schedule` manages a list of cron-like
      entries in `~/.cubi/schedule.json` for an external runner)*
- [x] **Skills system** — reusable Markdown skill packs in
      `~/.cubi/skills/*` + `SkillTool`
- [x] **Tool-search tool** — model searches the registry instead of receiving
      the full tool list in every prompt
- [x] **Structured output helpers** — `Brief`, `SyntheticOutput`
      *(`brief` distills text into title/bullets/summary; `synthetic_output`
      fills a JSON Schema's properties with type-appropriate defaults — both
      deterministic, no extra model call)*
- [x] **Inter-agent messaging** — `SendMessage`, `RemoteTrigger`
      *(`send_message` / `recv_messages` use `~/.cubi/messages/`;
      `remote_trigger` drops payload+ts files in `~/.cubi/triggers/`)*
- [x] **MCP resources & prompts** — `resources/list`, `resources/read`,
      `prompts/list`, `prompts/get`; per-server OAuth and interactive approval
      ✅ resources + interactive approval + prompts all shipped; OAuth Phase 1
      backend shipped (persisted tokens + MCP HTTP auth injection), with full
      provider-device/code flows still open
- [x] **Sleep prevention** — `prevent_sleep` builtin tool wraps `caffeinate`
      (macOS), `systemd-inhibit` (Linux), and `SetThreadExecutionState` via
      PowerShell (Windows). Hard-capped at 4 hours.

→ Foundation: central tool registry (analogous to leaked `src/tools.ts` /
`src/Tool.ts`) + `enabled_tools` config so users can disable any tool.

---

## B. Slash commands to add (`src/cli.rs` / `src/commands/`)

Currently shipped (pre-PR): `/help`, `/clear`, `/history`, `/model`, `/save`,
`/load`, `/batch`, `/mcp-tools`, `/mcp-call`, `/mcp-reload`, `/quit`.

Shipped previously:

- [x] `/plan`, `/todos`, `/todo-add`, `/todo-done`, `/todo-rm`, `/todo-clear`
- [x] `/init`, `/memory`, `/memory-reload`
- [x] `/status`, `/version`, `/export` (with overwrite protection)
- [x] `/ask` (user-driven clarifying-question stand-in, single-turn)
- [x] `/sessions`, `/resume` (auto-saved per-project checkpoints)
- [x] `/diff`, `/commit`, `/review` (git workflow; `/commit` is plan-mode-aware)
- [x] `/trust` (project-trust gate for write/exec tools)
- [x] `/permissions` (lists trusted roots and gated built-in tools; also
      surfaces the active admin policy overlay if one is present)
- [x] `/memdir`, `/memdir-add`, `/memdir-rm`, `/memdir-clear`
      (cross-session persistent memory at `~/.cubi/memdir/`)
- [x] `/rewind` (history surgery; **now also rolls back file mutations
      recorded by `edit_file` / `write_file` during the rewound turns**),
      `/compact` (automatic summarization)
- [x] `/worktree` (list/add/remove; auto-trusts new path, plan-mode-aware)
- [x] `/branch` (list/create/switch; mutating actions plan-mode-aware)
- [x] `/tag` (list/create; create plan-mode-aware)
- [x] `/files` (lists tracked files via `git ls-files`)
- [x] `/add-dir` (trust an additional directory for write/exec tools)
- [x] `/doctor`, `/env`, `/config`, `/bug` (diagnostics & runtime transparency)
- [x] `/init-verifiers` ✅
- [x] `/commit-push-pr` ✅, `/issue` ✅, `/undo` ✅, `/pr_comments` ✅,
      `/security-review` ✅, `/autofix-pr` ✅
- [x] `/agents` ✅, `/tasks` ✅, `/teleport` ✅, `/passes` ✅, `/effort` ✅
- [x] `/theme` ✅ (now persists to `AppConfig.theme`), `/color` ✅ (now
      persists to `AppConfig.color`), `/output-style` ✅ (now persists to
      `AppConfig.output_style` and is injected as a system steering prompt),
      `/statusline` ✅, `/keybindings` ✅, `/vim` ✅ (config flag now
      persists to `AppConfig.vim_mode`; full vim-mode TUI still needs a
      ratatui port — see Section C #12)
- [x] `/login` ✅, `/logout` ✅, `/oauth-refresh` ✅ *(OAuth Phase 1 backend:
      persisted token store + in-process token reload)*, `/privacy-settings` ✅
- [x] `/mcp` ✅, `/plugin` ✅ (now backed by `plugins.rs`, lists discovered
      `~/.cubi/plugins/<name>/commands/*.md`), `/reload-plugins` ✅
      (now reloads both skills and plugin bundles), `/skills` ✅, `/hooks` ✅
- [x] `/stats` ✅, `/usage` ✅, `/cost` ✅, `/perf-issue` ✅, `/heapdump` ✅,
      `/debug-tool-call` ✅, `/doctor` ✅, `/env` ✅, `/bug` ✅,
      `/permissions` ✅, `/config` ✅
- [x] `/upgrade` ✅, `/install` ✅, `/install-github-app` ✅,
      `/install-slack-app` ✅, `/sandbox-toggle` ✅ *(alias for `/plan`)*,
      `/reset-limits` ✅
- [x] `/share` ✅, `/copy` ✅, `/feedback` ✅, `/release-notes` ✅,
      `/stickers` ✅

**New in this release (0.3.0):**

- [x] `/settings-sync init|push|pull|status` — git-backed cross-machine sync
      of `~/.cubi/` (config, memdir, skills, plugins)
- [x] `/policy` — show the read-only admin-managed policy overlay, including
      the source path and any tool deny-list
- [x] `/tip` — print a usage tip (also shown on startup in TTY mode)
- [x] `/mcp-prompts [server[:name]]` — list MCP prompts exposed by configured
      servers, or render a specific one (`prompts/list`, `prompts/get`)

Foundation work:

- [x] The flat `match` in `cli.rs::handle_command` is a `SlashCommand`
      registry (`src/commands.rs`) — adding a command requires a row in
      `COMMANDS` and an exhaustive arm on `Cmd`.
- [x] User-defined Markdown commands as first-class plugins
      (`src/plugins.rs`; namespaced `/<plugin>:<command>` triggers loaded
      from `~/.cubi/plugins/<name>/commands/*.md`). The pre-existing
      flat `~/.cubi/commands/*.md` loader in `file_mentions.rs`
      continues to work for un-namespaced commands.

---

## C. New subsystems / modules

1. **Onboarding** (`bootstrap/`, `setup.ts`, `projectOnboardingState.ts`) —
   first-run flow: pick model, scan project, write `CUBI.md`, set trust
   level. ✅ Shipped: `src/onboarding.rs` runs once, lets the user pick a
   model from `ollama list`, offers `/trust`, offers `CUBI.md`. Persisted
   to `~/.cubi/config.json`.
2. **Permissions system** (`utils/permissions/`) — project trust, per-tool
   allow/deny, "trust this folder" prompts, enterprise-managed policy. ✅
   Shipped: `src/permissions.rs` enforces a per-project trust store with a
   path sandbox; `bash`, `edit_file`, `write_file` are gated. Per-tool
   allow/deny lists shipped. Enterprise-managed policy ✅ shipped via
   `src/policy.rs` (read-only `policy.json` overlay).
3. **Memory & compaction** (`services/compact/`, `SessionMemory/`,
   `extractMemories/`, `memdir/`) — automatic in-session compaction plus
   cross-session persistent memory at `~/.cubi/memdir/`. ✅ Shipped:
   `src/memdir.rs` + `/memdir*` slash commands; `/compact` summarizes old
   turns; automatic model-driven memory extraction shipped.
4. **Proactive completions** (`PromptSuggestion/`, `fileSuggestions.ts`) —
   suggest next prompts and `@file` references while the user types.
   ⏳ **Deferred** — requires TUI integration (see #18); will be revisited
   together with the ratatui port. Today's `@file` mention expansion in
   `src/file_mentions.rs` covers the post-submit half.
5. **Multi-agent layer** (`utils/swarm/`, `coordinator/`, `assistant/`,
   `tasks/`) — teammates over tmux / iTerm / in-process backends.
   ⏳ **Deferred** — significant cross-cutting work; the in-process half is
   covered by `agent_run` (subagents with fresh context). External tmux /
   iTerm backends are future work.
6. **API hardening** (`services/api/`, `claudeAiLimits*`, `tokenEstimation.ts`)
   — token estimator, rate limiter, retry-with-backoff in `ollama.rs`.
   ✅ Retry-with-backoff shipped.
7. **LSP bridge** (`services/lsp/`) — diagnostics after `edit_file`,
   jump-to-definition. ✅ `lsp` builtin tool shipped (`hover` / `definition`
   / `references`). Automatic post-edit diagnostics would need an editor
   loop integration and is deferred until the ratatui port.
8. **Notifications + sleep prevention** (`notifier.ts`, `preventSleep.ts`) —
   OS notifications on long-run completion, `caffeinate`-style awake. ✅
   `notify` builtin tool shipped (osascript/notify-send/PowerShell);
   ✅ `prevent_sleep` builtin tool shipped (caffeinate / systemd-inhibit /
   SetThreadExecutionState).
9. **Interactive MCP approval** (`mcpServerApproval.tsx`). ✅ Shipped.
10. **Telemetry / debug log** (`analytics/`, `diagnosticTracking.ts`,
    `internalLogging.ts`) — opt-in. ✅ Shipped: `src/telemetry.rs` appends
    JSON-line events to `~/.cubi/telemetry.log` when
    `AppConfig.telemetry` (or `CUBI_TELEMETRY=1`) is set. Each tool call
    records its duration and ok-flag.
11. **Voice input** (`voice.ts`, `voiceStreamSTT.ts`, `voiceKeyterms.ts`).
    ⏳ **Deferred** — needs an audio pipeline (cpal/whisper.cpp) that is too
    large to land in a single automated pass.
12. **Vim mode + remappable keybindings** (`vim/`, `keybindings/`).
    🟡 Config flag persisted (`AppConfig.vim_mode`, surfaced by `/vim`);
    full vim-mode TUI still pending the ratatui port (see #18).
13. **Themable output styles** (`outputStyles/`) — concise / explanatory /
    markdown / etc., per-session. ✅ Shipped: `src/output_styles.rs` +
    `src/themes.rs`. Output-style preset is injected as a system steering
    prompt on every session; theme persists in config and is consumed by
    the bundled palettes (used as a stable target for the future ratatui
    port's printer layer).
14. **Hooks** — `PreToolUse`, `PostToolUse`, `UserPromptSubmit`, `Stop`,
    `SessionStart`, `Notification`. ✅ Shipped in `src/hooks.rs` with
    `/hooks list|add|rm`.
15. **Plugin system** (`plugins/`, `services/plugins/`, `reload-plugins`) —
    `~/.cubi/plugins/*` discoverable bundles. ✅ Shipped: `src/plugins.rs`
    loads `~/.cubi/plugins/<name>/commands/*.md` as namespaced
    `/<plugin>:<command>` triggers; `/plugin` lists them, `/reload-plugins`
    refreshes.
16. **Headless / remote mode** (`server/`, `remote/`, `upstreamproxy/`,
    `bridge/`) — daemon + remote client; pairs naturally with the unused
    `distributed.rs`. ⏳ **Deferred** — requires a server crate
    plus auth + on-wire protocol design.
17. **Auto-saved sessions + checkpointing** (`sessionStorage.js`,
    `sessionStart.js`, `history.ts`) — `/resume`, `/rewind` with
    file-mutation rollback. ✅ Sessions + `/resume` + `/rewind` shipped.
    ✅ File-mutation rollback shipped: `src/file_rollback.rs` journals
    pre-images recorded by `edit_file` / `write_file` per turn; `/rewind n`
    restores them and deletes files created in the rewound turns.
18. **TUI rewrite** (`screens/`, `components/`, Ink → `ratatui`) — panes for
    chat / tool output / todos / status line. ⏳ **Deferred** — a full
    ratatui port is multi-week work and would touch every printer in the
    repo. The plumbing for it (themes, output-style presets, palettes) is
    in place so the port can begin without further refactors.
19. **Deep-link / browser integration** (`claudeInChrome/`, `deepLink/`,
    `chrome` command) — `aichat://` URLs, Chrome native-messaging bridge.
    ⏳ **Deferred** — needs a separate native-messaging host binary plus a
    Chrome extension; out of scope for the CLI crate.
20. **Tip-of-the-day + release notes** (`buddy/`, `tips/`, `release-notes`).
    ✅ Shipped: `src/tips.rs` prints a daily tip on startup (TTY only) and
    on demand via `/tip`; tips are sourced from a built-in pool plus
    `~/.cubi/tips/*.txt`. `/release-notes` was already in place.
21. **Centralized schemas** (`schemas/`, `types/`) — stricter validation than
    `serde_json::Value` everywhere. ✅ Shipped: `src/schemas.rs` exposes a
    uniform `require_str` / `require_number` / `require_bool` / `optional_str`
    vocabulary. Existing tools will migrate over time; new tools should
    prefer the helpers.
22. **Migrations** — versioned config / session migration framework.
    ✅ Shipped: `src/migrations.rs`; `AppConfig.config_version` stamped at
    load time, forward-only migrations idempotent and never downgrade a
    future-version file.
23. **Enterprise policy** (`policyLimits/`, `remoteManagedSettings/`) —
    admin-pushed config, tool denylists. ✅ Shipped: `src/policy.rs` loads
    a read-only `policy.json` from `$CUBI_POLICY_FILE`, then
    `/etc/cubi/policy.json`, then `~/.cubi/policy.json`. The
    deny-list is checked *before* the user's allow-list in the agent
    loop, so a user config can't undo an admin denial.
24. **Settings sync** (`settingsSync/`) — cross-machine sync via Git repo.
    ✅ Shipped: `src/settings_sync.rs` + `/settings-sync init|push|pull|status`.

---

## D. Implementation priorities

1. Agent loop + native tool-calling + streaming. ✅ Shipped (NDJSON streaming
   in `ollama.rs`; `agent_loop.rs` drives multi-step tool round-trips via
   Ollama's `tools` parameter, with a 12-step safety cap).
2. Permissions system + path sandboxing + project-trust prompt. ✅ Shipped.
3. **Plan mode + `TodoWrite` + `AskUserQuestion`.** ✅ Plan mode gates all
   built-in write/exec tools; `/todos` and `/ask` already in place.
4. `/init` + `CUBI.md` + memdir + onboarding flow. ✅ All shipped.
5. Auto-saved sessions + `/resume` + `/rewind` checkpoints + compaction.
   ✅ Shipped. File-mutation rollback on rewind ✅ shipped (this release).
6. Slash-command registry + custom Markdown commands + `@file` mentions +
   prompt suggestions. ✅ Registry shipped; `@file` mentions shipped;
   user-defined Markdown commands shipped both as flat
   `~/.cubi/commands/*.md` and namespaced plugin bundles. Proactive
   prompt suggestions remain deferred (see Section C #4).
7. Subagents (`AgentTool`) + task management tools. ✅ `agent_run` shipped.
8. Git tools: `/commit`, `/commit-push-pr`, `/diff`, `/review`, worktree tools.
   ✅ All shipped.
9. Multi-provider LLM abstraction + token estimator + rate-limit / retry.
   ✅ Multi-provider + retry shipped.
10. Web tools (`web_fetch`, `web_search`) + LSP service & tool + REPL tool +
    notebook tool. ✅ All four shipped.
11. Hooks, plugins, skills, MCP resources / prompts / OAuth, MCP approval UI.
    ✅ Hooks UI, skills, plugins, MCP resources, MCP prompts, and approval
    UI all shipped. OAuth Phase 1 backend shipped; full hosted-provider OAuth
    flows remain open.
12. TUI (ratatui) rewrite with panes, vim keybindings, output styles, themes,
    statusline. ⏳ **Deferred** (themes/output-style preset plumbing is in
    place so this can begin cleanly).
13. Headless / remote / server mode, deep-link integration, voice,
    notifications. 🟡 OS notifications + sleep prevention shipped; daemon
    + deep-link + voice remain deferred.
14. Telemetry, migrations, enterprise policy, settings sync. ✅ All four
    shipped in this release.
15. Tests, CI, tracing, `clap` flags + config file (cross-cutting; do alongside
    everything).

---

## Explicitly deferred to a future release

These items are tracked above but cannot realistically be completed in a
single automated pass. They are listed here so the next contributor has a
clear starting point:

- **OAuth backend (Phase 2+)** (Section A MCP OAuth; touches /login flows).
  Phase 1 shipped (persisted token backend + MCP HTTP auth injection); full
  provider/device authorization flows and refresh endpoints remain open.
- **Headless / remote / server / daemon mode** (Section C #16).
- **Full TUI rewrite to ratatui** (Section C #18 / Section D #12). Plumbing
  for themes, output-style presets, and palettes is in place; the port
  itself remains future work.
- **Proactive prompt suggestions** (Section C #4). Needs TUI integration.
- **Vim-mode TUI** (Section C #12). Config flag persists; full keybinding
  layer waits on the ratatui port.
- **Voice input** (Section C #11). Needs an audio pipeline.
- **Multi-agent layer over tmux / iTerm** (Section C #5). In-process
  subagents are shipped via `agent_run`.
- **Deep-link / Chrome native-messaging bridge** (Section C #19). Needs a
  separate native-messaging host binary.
