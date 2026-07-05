//! Slash-command registry.
//!
//! Before this module existed, `cli.rs` carried two parallel structures:
//! a flat `match` in `handle_command` that did the actual dispatch, and a
//! separate `command_help()` array used by `/help` and the welcome banner.
//! Adding a command meant editing both places, and a drift-guard test had to
//! exist to catch the inevitable mismatch.
//!
//! This module collapses them into a single source of truth: [`COMMANDS`].
//! Each entry carries its trigger string, the usage line shown in help, the
//! short description, and the [`Cmd`] tag that `handle_command` matches on.
//! The parser ([`parse`]) maps an input line onto `(Cmd, args)` purely from
//! that table, so help and dispatch can no longer disagree by construction.
//!
//! When adding a new slash command:
//!
//!   1. Add a variant to [`Cmd`].
//!   2. Add a row to [`COMMANDS`].
//!   3. Add an arm to the `match cmd` in `ChatCLI::handle_command`.
//!
//! The compiler will refuse to build until step 3 is done because `Cmd`
//! exhaustiveness is checked.

/// Tag for each registered slash command. Variants intentionally have no
/// payload — argument parsing lives in the individual command handlers so
/// the registry stays a flat lookup table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cmd {
    Help,
    Status,
    Plan,
    Init,
    Memory,
    MemoryReload,
    Todos,
    TodoAdd,
    TodoDone,
    TodoRm,
    TodoClear,
    Ask,
    Clear,
    History,
    Export,
    Save,
    Load,
    Batch,
    McpTools,
    McpCall,
    McpReload,
    Model,
    Version,
    Sessions,
    Resume,
    Trust,
    Diff,
    Commit,
    CommitPushPr,
    Undo,
    Review,
    Worktree,
    Branch,
    Tag,
    Files,
    AddDir,
    Memdir,
    MemdirAdd,
    MemdirRm,
    MemdirClear,
    Rewind,
    Compact,
    Doctor,
    Env,
    Config,
    Permissions,
    ToolAllow,
    ToolDeny,
    Hooks,
    Skills,
    Stats,
    Usage,
    McpResources,
    McpRead,
    McpSearch,
    McpInstall,
    McpUninstall,
    Bug,
    Issue,
    // -- New in 0.2.0 --
    InitVerifiers,
    PrComments,
    SecurityReview,
    AutofixPr,
    Agents,
    Tasks,
    Teleport,
    Passes,
    Effort,
    Theme,
    Color,
    OutputStyle,
    Statusline,
    Keybindings,
    Vim,
    Login,
    Logout,
    OauthRefresh,
    PrivacySettings,
    Mcp,
    Plugin,
    ReloadPlugins,
    Cost,
    PerfIssue,
    Heapdump,
    DebugToolCall,
    Upgrade,
    Install,
    InstallGithubApp,
    InstallSlackApp,
    SandboxToggle,
    ResetLimits,
    Share,
    Copy,
    Feedback,
    ReleaseNotes,
    Stickers,
    // -- Added by the roadmap completion pass --
    SettingsSync,
    Policy,
    Tip,
    McpPrompts,
    // -- Phase 1: streaming/markdown/stats UX --
    Stream,
    Markdown,
    StatsFooter,
    // -- Phase 9: context management --
    Pin,
    Pins,
    Unpin,
    // -- Phase 9C: UX polish --
    Edit,
    Quit,
    // -- Phase 12: session branching --
    Fork,
    Repomap,
    // -- Phase 5 differentiator: multi-model consensus --
    Consensus,
}

/// One row in the slash-command registry.
pub struct SlashCommandSpec {
    /// The literal trigger, e.g. `"/help"`. Must start with `/` and contain
    /// no spaces.
    pub name: &'static str,
    /// Usage line shown to the user, including any argument placeholders,
    /// e.g. `"/save [-f] <filename>"`. The trigger and the usage may differ
    /// (e.g. `name = "/save"`, `usage = "/save [-f] <filename>"`).
    pub usage: &'static str,
    /// One-line description shown by `/help` and the welcome banner.
    pub help: &'static str,
    /// Tag matched by `ChatCLI::handle_command`.
    pub cmd: Cmd,
    /// The command's subcommand vocabulary, in display order. The FIRST
    /// entry is the default subcommand (used when the user types the bare
    /// command). Empty for commands that take no subcommands. This is the
    /// single source of truth consumed by subcommand autocompletion.
    pub subcommands: &'static [&'static str],
}

/// Single source of truth for the slash-command surface. Order here is the
/// order shown in `/help` and the welcome banner, so keep related commands
/// grouped.
pub const COMMANDS: &[SlashCommandSpec] = &[
    SlashCommandSpec {
        name: "/help",
        usage: "/help",
        help: "Show this help message",
        cmd: Cmd::Help,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/status",
        usage: "/status",
        help: "Show session status",
        cmd: Cmd::Status,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/stats",
        usage: "/stats",
        help: "Show session statistics",
        cmd: Cmd::Stats,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/usage",
        usage: "/usage [footer on|off]",
        help: "Per-turn token usage + cost table",
        cmd: Cmd::Usage,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/plan",
        usage: "/plan",
        help: "Toggle plan mode (read-only)",
        cmd: Cmd::Plan,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/init",
        usage: "/init",
        help: "Create starter CUBI.md",
        cmd: Cmd::Init,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/memory",
        usage: "/memory",
        help: "Show project memory (CUBI.md)",
        cmd: Cmd::Memory,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/memory-reload",
        usage: "/memory-reload",
        help: "Re-read CUBI.md from disk",
        cmd: Cmd::MemoryReload,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/todos",
        usage: "/todos",
        help: "List todos",
        cmd: Cmd::Todos,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/todo-add",
        usage: "/todo-add <text>",
        help: "Add a todo",
        cmd: Cmd::TodoAdd,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/todo-done",
        usage: "/todo-done <n>",
        help: "Mark todo n as done",
        cmd: Cmd::TodoDone,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/todo-rm",
        usage: "/todo-rm <n>",
        help: "Remove todo n",
        cmd: Cmd::TodoRm,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/todo-clear",
        usage: "/todo-clear",
        help: "Clear all todos",
        cmd: Cmd::TodoClear,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/ask",
        usage: "/ask <q>",
        help: "Record a clarifying question (single-turn)",
        cmd: Cmd::Ask,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/clear",
        usage: "/clear",
        help: "Clear conversation history",
        cmd: Cmd::Clear,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/history",
        usage: "/history [next|prev|<N>]",
        help: "Page conversation history; /history N trims",
        cmd: Cmd::History,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/export",
        usage: "/export [-f] <f.md>",
        help: "Export conversation as Markdown",
        cmd: Cmd::Export,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/save",
        usage: "/save [-f] <f.json>",
        help: "Save conversation (-f overwrites)",
        cmd: Cmd::Save,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/load",
        usage: "/load <f.json>",
        help: "Load conversation",
        cmd: Cmd::Load,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/batch",
        usage: "/batch <f>",
        help: "Process batch file",
        cmd: Cmd::Batch,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/sessions",
        usage: "/sessions",
        help: "List auto-saved sessions for this project",
        cmd: Cmd::Sessions,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/resume",
        usage: "/resume [id]",
        help: "Resume the latest (or named) auto-saved session",
        cmd: Cmd::Resume,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/fork",
        usage: "/fork",
        help: "Branch the current session at the last completed turn and continue in the fork",
        cmd: Cmd::Fork,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/repomap",
        usage: "/repomap [scope]",
        help: "Print a compact outline of the project's files and top-level symbols",
        cmd: Cmd::Repomap,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/trust",
        usage: "/trust [revoke]",
        help: "Trust this project (or pass `revoke` to undo)",
        cmd: Cmd::Trust,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/diff",
        usage: "/diff [path]",
        help: "Show `git diff` for the working tree",
        cmd: Cmd::Diff,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/commit",
        usage: "/commit [-a] <msg>",
        help: "Run git commit (-a stages tracked files first)",
        cmd: Cmd::Commit,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/commit-push-pr",
        usage: "/commit-push-pr [-a] <msg>",
        help: "Commit, push, and print a GitHub PR URL",
        cmd: Cmd::CommitPushPr,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/undo",
        usage: "/undo [hard]",
        help: "Undo the latest commit (or hard reset HEAD~1)",
        cmd: Cmd::Undo,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/review",
        usage: "/review",
        help: "Ask the model to review the current `git diff`",
        cmd: Cmd::Review,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/worktree",
        usage: "/worktree [list|add <path> [branch]|remove <path>]",
        help: "Manage git worktrees (add auto-trusts the new path)",
        cmd: Cmd::Worktree,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/branch",
        usage: "/branch [list|create <name>|switch <name>]",
        help: "List, create, or switch git branches",
        cmd: Cmd::Branch,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/tag",
        usage: "/tag [list|<name>|create <name> [-m <msg>]]",
        help: "List or create git tags",
        cmd: Cmd::Tag,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/files",
        usage: "/files",
        help: "List files tracked by git in this project",
        cmd: Cmd::Files,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/add-dir",
        usage: "/add-dir <path>",
        help: "Trust an additional directory for write/exec tools",
        cmd: Cmd::AddDir,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/memdir",
        usage: "/memdir",
        help: "List cross-session persistent memories",
        cmd: Cmd::Memdir,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/memdir-add",
        usage: "/memdir-add <text>",
        help: "Add a persistent memory",
        cmd: Cmd::MemdirAdd,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/memdir-rm",
        usage: "/memdir-rm <n>",
        help: "Remove memory by index",
        cmd: Cmd::MemdirRm,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/memdir-clear",
        usage: "/memdir-clear",
        help: "Clear all persistent memories",
        cmd: Cmd::MemdirClear,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/rewind",
        usage: "/rewind [n]",
        help: "Remove the last n exchanges (default 1)",
        cmd: Cmd::Rewind,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/compact",
        usage: "/compact [preview]",
        help: "Summarize old turns (preview = dry-run, no mutation)",
        cmd: Cmd::Compact,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/pin",
        usage: "/pin <text>",
        help: "Pin text as a persistent system note that survives /compact",
        cmd: Cmd::Pin,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/pins",
        usage: "/pins",
        help: "List pinned items with 1-based indices",
        cmd: Cmd::Pins,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/unpin",
        usage: "/unpin <idx>",
        help: "Remove the pinned item at the given 1-based index",
        cmd: Cmd::Unpin,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/skills",
        usage: "/skills [list|run <name>|enable <name>|disable <name>]",
        help: "List or run reusable Markdown skills",
        cmd: Cmd::Skills,
        subcommands: &["list", "run", "enable", "disable"],
    },
    SlashCommandSpec {
        name: "/hooks",
        usage: "/hooks [list | add <event> <cmd> | rm <n>]",
        help: "List, add, or remove lifecycle hooks",
        cmd: Cmd::Hooks,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/mcp-tools",
        usage: "/mcp-tools",
        help: "List available MCP tools",
        cmd: Cmd::McpTools,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/mcp-call",
        usage: "/mcp-call <t> <a>",
        help: "Call MCP tool",
        cmd: Cmd::McpCall,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/mcp-reload",
        usage: "/mcp-reload",
        help: "Reload MCP configuration",
        cmd: Cmd::McpReload,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/mcp-resources",
        usage: "/mcp-resources [server]",
        help: "List MCP resources",
        cmd: Cmd::McpResources,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/mcp-read",
        usage: "/mcp-read <uri>",
        help: "Read an MCP resource by URI",
        cmd: Cmd::McpRead,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/mcp-search",
        usage: "/mcp-search [<query>]",
        help: "Search the embedded MCP registry",
        cmd: Cmd::McpSearch,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/mcp-install",
        usage: "/mcp-install <name> [--force] [--env K=V]...",
        help: "Install an MCP server from the registry",
        cmd: Cmd::McpInstall,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/mcp-uninstall",
        usage: "/mcp-uninstall <name>",
        help: "Remove an MCP server from ~/.cubi/mcp.json",
        cmd: Cmd::McpUninstall,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/model",
        usage: "/model [name]",
        help: "Show or switch the active model",
        cmd: Cmd::Model,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/version",
        usage: "/version",
        help: "Show version",
        cmd: Cmd::Version,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/doctor",
        usage: "/doctor",
        help: "Sanity-check the runtime (Ollama, model, config dir, git)",
        cmd: Cmd::Doctor,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/env",
        usage: "/env",
        help: "Show resolved runtime environment",
        cmd: Cmd::Env,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/config",
        usage: "/config",
        help: "Show contents of ~/.cubi/config.json",
        cmd: Cmd::Config,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/permissions",
        usage: "/permissions",
        help: "List trusted directories and gated built-in tools",
        cmd: Cmd::Permissions,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/tool-allow",
        usage: "/tool-allow <name>",
        help: "Allow a specific tool in this trust store",
        cmd: Cmd::ToolAllow,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/tool-deny",
        usage: "/tool-deny <name>",
        help: "Deny a specific tool in this trust store",
        cmd: Cmd::ToolDeny,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/bug",
        usage: "/bug [summary]",
        help: "Print a pre-filled GitHub bug URL with runtime info",
        cmd: Cmd::Bug,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/issue",
        usage: "/issue [title]",
        help: "Print a pre-filled GitHub feature request URL",
        cmd: Cmd::Issue,
        subcommands: &[],
    },
    // -- New in 0.2.0 --
    SlashCommandSpec {
        name: "/init-verifiers",
        usage: "/init-verifiers",
        help: "Detect project verifier commands (build/test/lint) and print/save them",
        cmd: Cmd::InitVerifiers,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/pr_comments",
        usage: "/pr_comments [pr#]",
        help: "Show PR review comments via `gh pr view --comments`",
        cmd: Cmd::PrComments,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/security-review",
        usage: "/security-review",
        help: "Ask the model to security-review the current `git diff`",
        cmd: Cmd::SecurityReview,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/autofix-pr",
        usage: "/autofix-pr [pr#]",
        help: "Fetch PR review comments and ask the model to propose fixes",
        cmd: Cmd::AutofixPr,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/agents",
        usage: "/agents [list|enable <name>|disable <name>]",
        help: "List custom agents and enable/disable them (~/.cubi/agents)",
        cmd: Cmd::Agents,
        subcommands: &["list", "enable", "disable"],
    },
    SlashCommandSpec {
        name: "/tasks",
        usage: "/tasks",
        help: "Alias for /todos (per-project task list)",
        cmd: Cmd::Tasks,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/teleport",
        usage: "/teleport <path>",
        help: "Change cwd to a trusted directory (use /trust to approve first)",
        cmd: Cmd::Teleport,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/passes",
        usage: "/passes [n]",
        help: "Show or set the agent-loop max passes (1..=12)",
        cmd: Cmd::Passes,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/effort",
        usage: "/effort [low|medium|high]",
        help: "Show or set agent effort (maps to agent-loop pass budget)",
        cmd: Cmd::Effort,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/theme",
        usage: "/theme [auto|light|dark]",
        help: "Show or set the colored-output theme",
        cmd: Cmd::Theme,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/color",
        usage: "/color [on|off]",
        help: "Toggle colored output for this session",
        cmd: Cmd::Color,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/output-style",
        usage: "/output-style [concise|markdown|explanatory]",
        help: "Show or set the assistant output style",
        cmd: Cmd::OutputStyle,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/statusline",
        usage: "/statusline",
        help: "Show the contents of the status line",
        cmd: Cmd::Statusline,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/keybindings",
        usage: "/keybindings",
        help: "Show the active rustyline keybindings",
        cmd: Cmd::Keybindings,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/vim",
        usage: "/vim [on|off]",
        help: "Toggle vim-style readline editing",
        cmd: Cmd::Vim,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/login",
        usage: "/login <provider> <access-token> [--refresh-token <token>] [--expires-in <seconds>]",
        help: "Store an OAuth token for a provider (persisted in ~/.cubi/oauth.json)",
        cmd: Cmd::Login,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/logout",
        usage: "/logout [provider]",
        help: "Forget a provider API key for this process and remove its persisted OAuth token",
        cmd: Cmd::Logout,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/oauth-refresh",
        usage: "/oauth-refresh [provider]",
        help: "Load stored OAuth tokens into this process and show token status",
        cmd: Cmd::OauthRefresh,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/privacy-settings",
        usage: "/privacy-settings [telemetry on|off]",
        help: "Show or set local privacy preferences",
        cmd: Cmd::PrivacySettings,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/mcp",
        usage: "/mcp [list|enable <name>|disable <name>|add <name> <command|url>|remove <name>|reload]",
        help: "Show overall MCP status (servers, tools, resources)",
        cmd: Cmd::Mcp,
        subcommands: &["list", "enable", "disable", "add", "remove", "reload"],
    },
    SlashCommandSpec {
        name: "/plugin",
        usage: "/plugin [list]",
        help: "List plugins discovered in ~/.cubi/plugins/",
        cmd: Cmd::Plugin,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/reload-plugins",
        usage: "/reload-plugins",
        help: "Rescan the plugins directory",
        cmd: Cmd::ReloadPlugins,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/cost",
        usage: "/cost",
        help: "Show estimated session cost (always $0 for local Ollama)",
        cmd: Cmd::Cost,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/perf-issue",
        usage: "/perf-issue [summary]",
        help: "Print a pre-filled GitHub perf-issue URL with runtime info",
        cmd: Cmd::PerfIssue,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/heapdump",
        usage: "/heapdump",
        help: "Print process resident-set / heap info if available",
        cmd: Cmd::Heapdump,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/debug-tool-call",
        usage: "/debug-tool-call [on|off]",
        help: "Toggle verbose tool-call debug logging",
        cmd: Cmd::DebugToolCall,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/upgrade",
        usage: "/upgrade",
        help: "Print upgrade instructions for cubi",
        cmd: Cmd::Upgrade,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/install",
        usage: "/install",
        help: "Print install instructions for cubi + Ollama",
        cmd: Cmd::Install,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/install-github-app",
        usage: "/install-github-app",
        help: "Show the GitHub-app install URL (placeholder)",
        cmd: Cmd::InstallGithubApp,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/install-slack-app",
        usage: "/install-slack-app",
        help: "Show the Slack-app install URL (placeholder)",
        cmd: Cmd::InstallSlackApp,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/sandbox-toggle",
        usage: "/sandbox-toggle",
        help: "Toggle strict-sandbox mode (alias for /plan)",
        cmd: Cmd::SandboxToggle,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/reset-limits",
        usage: "/reset-limits",
        help: "Clear the in-process rate-limit / retry backoff counters",
        cmd: Cmd::ResetLimits,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/share",
        usage: "/share <file.md>",
        help: "Export this conversation as a shareable Markdown file",
        cmd: Cmd::Share,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/copy",
        usage: "/copy",
        help: "Copy the last assistant message to the system clipboard",
        cmd: Cmd::Copy,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/feedback",
        usage: "/feedback [text]",
        help: "Print the feedback URL (pre-fills text if provided)",
        cmd: Cmd::Feedback,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/release-notes",
        usage: "/release-notes",
        help: "Print release notes for the current version",
        cmd: Cmd::ReleaseNotes,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/stickers",
        usage: "/stickers",
        help: "Print a friendly ASCII sticker sheet",
        cmd: Cmd::Stickers,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/settings-sync",
        usage: "/settings-sync [init <remote>|push [msg]|pull|status]",
        help: "Sync ~/.cubi/ via git (cross-machine config + memdir + skills)",
        cmd: Cmd::SettingsSync,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/policy",
        usage: "/policy",
        help: "Show the active admin-managed policy overlay (read-only)",
        cmd: Cmd::Policy,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/tip",
        usage: "/tip",
        help: "Show a quick tip about using cubi",
        cmd: Cmd::Tip,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/mcp-prompts",
        usage: "/mcp-prompts [server[:prompt]]",
        help: "List MCP prompts (or fetch a specific one)",
        cmd: Cmd::McpPrompts,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/stream",
        usage: "/stream [on|off]",
        help: "Toggle live token streaming (default: on)",
        cmd: Cmd::Stream,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/markdown",
        usage: "/markdown [on|off]",
        help: "Toggle markdown rendering (applies when streaming is off)",
        cmd: Cmd::Markdown,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/stats-footer",
        usage: "/stats-footer [on|off]",
        help: "Toggle per-turn usage footer (default: off)",
        cmd: Cmd::StatsFooter,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/edit",
        usage: "/edit [seed text]",
        help: "Open $EDITOR to compose the next prompt",
        cmd: Cmd::Edit,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/quit",
        usage: "/quit",
        help: "Exit the chat",
        cmd: Cmd::Quit,
        subcommands: &[],
    },
    SlashCommandSpec {
        name: "/consensus",
        usage: "/consensus <strategy> <model1,model2,...> [tools|--tools] [isolate|--isolate] [concurrency:<n>|c:<n>] [--max-steps <n>] [--isolated-time-cap-secs <seconds>] [judge:<model>] <goal>",
        help: "Run N-model consensus; --isolate uses tool worktrees (trusted clean cwd, /plan off), c:<n> capped at 2",
        cmd: Cmd::Consensus,
        subcommands: &[],
    },
];

/// Returns every registered command name in display order.
pub fn command_names() -> impl Iterator<Item = &'static str> {
    COMMANDS.iter().map(|s| s.name)
}

/// Returns every registered command name that starts with `typed`.
///
/// `typed` may include the leading slash or not — both `"/re"` and
/// `"re"` are accepted. Used by tab-completion in the REPL and as the
/// first pass for "did you mean?" suggestions.
pub fn prefix_matches(typed: &str) -> Vec<&'static str> {
    let stripped = normalize_typed_command(typed);
    if stripped.is_empty() {
        return Vec::new();
    }
    command_names()
        .filter(|name| {
            name.strip_prefix('/')
                .is_some_and(|n| n.starts_with(&stripped))
        })
        .collect()
}

/// Suggest likely slash commands for an unknown command head.
///
/// Prefix matches win because they preserve muscle-memory shortcuts (`/re`
/// should list all `/re…` commands). If there are no prefix matches, fall
/// back to a small Levenshtein search so typos such as `/relase-notes` still
/// get a useful hint without flooding unrelated commands.
pub fn suggestions(typed: &str) -> Vec<&'static str> {
    let typed = normalize_typed_command(typed);
    if typed.is_empty() {
        return Vec::new();
    }

    let prefix = prefix_matches(&typed);
    if !prefix.is_empty() {
        return prefix;
    }

    let threshold = levenshtein_threshold(typed.len());
    let mut ranked: Vec<(usize, &'static str)> = command_names()
        .filter_map(|name| {
            let bare = name.strip_prefix('/').unwrap_or(name);
            let distance = levenshtein(&typed, bare);
            (distance <= threshold).then_some((distance, name))
        })
        .collect();
    ranked.sort_by(|(a_dist, a_name), (b_dist, b_name)| {
        a_dist.cmp(b_dist).then_with(|| a_name.cmp(b_name))
    });
    ranked.into_iter().map(|(_, name)| name).collect()
}

fn normalize_typed_command(typed: &str) -> String {
    let head = typed.split_whitespace().next().unwrap_or(typed);
    head.strip_prefix('/').unwrap_or(head).to_ascii_lowercase()
}

fn levenshtein_threshold(len: usize) -> usize {
    match len {
        0..=4 => 1,
        5..=10 => 2,
        _ => 3,
    }
}

fn levenshtein(a: &str, b: &str) -> usize {
    let mut previous: Vec<usize> = (0..=b.chars().count()).collect();
    let mut current = vec![0; previous.len()];

    for (i, ca) in a.chars().enumerate() {
        current[0] = i + 1;
        for (j, cb) in b.chars().enumerate() {
            let substitution = previous[j] + usize::from(ca != cb);
            let insertion = current[j] + 1;
            let deletion = previous[j + 1] + 1;
            current[j + 1] = substitution.min(insertion).min(deletion);
        }
        std::mem::swap(&mut previous, &mut current);
    }

    previous[b.chars().count()]
}

/// Look up a single slash command spec by exact name. Returns `None` for
/// unknown commands; prefix matching is intentionally not applied here so
/// `/help <name>` is unambiguous.
pub fn find_command(name: &str) -> Option<&'static SlashCommandSpec> {
    COMMANDS.iter().find(|s| s.name == name)
}

/// Returns the subcommand vocabulary for `cmd` (in display order, first entry
/// is the default). Empty for commands that take no subcommands or that are
/// not present in the registry.
pub fn subcommands(cmd: Cmd) -> &'static [&'static str] {
    COMMANDS
        .iter()
        .find(|s| s.cmd == cmd)
        .map(|s| s.subcommands)
        .unwrap_or(&[])
}

/// Returns the default (first) subcommand for `cmd`, or `None` when the
/// command has no subcommand vocabulary.
#[cfg_attr(not(test), allow(dead_code))]
pub fn default_subcommand(cmd: Cmd) -> Option<&'static str> {
    subcommands(cmd).first().copied()
}

/// The argument hint for a subcommand (e.g. `"<name>"` for `/agents enable`),
/// parsed from the command's `usage` string, or `None` when the subcommand
/// takes no args (e.g. `list`).
///
/// The hint is sourced from the `[...]` alternatives group of the `usage`
/// string. Alternatives are split on `|` at bracket depth 0, where depth
/// increments on `<` and decrements on `>`, so a nested pipe such as the one in
/// `<command|url>` is not treated as a separator. The returned slice is a
/// subslice of the `'static` `usage`, so the `&'static str` return is sound.
pub fn subcommand_arg_hint(cmd: Cmd, sub: &str) -> Option<&'static str> {
    let usage: &'static str = COMMANDS.iter().find(|s| s.cmd == cmd)?.usage;
    // Content between the first `[` and its matching `]` (a subslice of the
    // `'static` usage, so everything derived from it stays `'static`).
    let open = usage.find('[')?;
    let close = open + usage[open..].find(']')?;
    let group: &'static str = &usage[open + 1..close];

    // Split the alternatives on `|` at bracket depth 0 only.
    let mut depth = 0i32;
    let mut start = 0usize;
    let mut alts: Vec<&'static str> = Vec::new();
    for (i, c) in group.char_indices() {
        match c {
            '<' => depth += 1,
            '>' => depth -= 1,
            '|' if depth == 0 => {
                alts.push(&group[start..i]);
                start = i + c.len_utf8();
            }
            _ => {}
        }
    }
    alts.push(&group[start..]);

    for alt in alts {
        let alt = alt.trim();
        // Split off the first whitespace-delimited word (the subcommand name).
        let (name, rest) = match alt.find(char::is_whitespace) {
            Some(i) => (&alt[..i], alt[i..].trim()),
            None => (alt, ""),
        };
        if name == sub {
            return (!rest.is_empty()).then_some(rest);
        }
    }
    None
}

/// Parses an input line that starts with `/` into a `(Cmd, args)` pair.
///
/// * `args` is everything after the first whitespace, with leading/trailing
///   whitespace trimmed. For a bare `/cmd` with no arguments, `args` is `""`.
/// * Returns `None` for unknown commands so the caller can surface a "try
///   `/help`" hint.
/// * Both `/exit` and `/quit` are accepted as an exit signal (alias kept for
///   backward compatibility, intentionally undocumented in `/help`).
/// * If `head` doesn't match a command exactly, the parser falls back to
///   unique-prefix matching (e.g. `/q` -> `/quit`, `/he` -> `/help`).
///   Ambiguous prefixes return `None` so the caller can show the standard
///   "Unknown command" hint rather than silently picking one.
pub fn parse(input: &str) -> Option<(Cmd, &str)> {
    let input = input.trim();
    let (head, rest) = match input.find(char::is_whitespace) {
        Some(i) => (&input[..i], input[i..].trim()),
        None => (input, ""),
    };

    // Undocumented alias kept so old muscle memory still works.
    if head == "/exit" {
        return Some((Cmd::Quit, rest));
    }

    if let Some(spec) = COMMANDS.iter().find(|s| s.name == head) {
        return Some((spec.cmd, rest));
    }

    // Prefix fallback: accept any unambiguous `/xy...` prefix. Skip the
    // leading `/` so a typed `/q` matches `/quit` even though it isn't a
    // prefix of the literal string `/quit` minus its slash. Require at
    // least one non-slash char so a bare `/` doesn't match everything.
    let typed = head.strip_prefix('/')?;
    if typed.is_empty() {
        return None;
    }
    let mut matches = COMMANDS.iter().filter(|s| {
        s.name
            .strip_prefix('/')
            .is_some_and(|n| n.starts_with(typed))
    });
    let first = matches.next()?;
    if matches.next().is_some() {
        // More than one candidate — ambiguous, refuse to guess.
        return None;
    }
    Some((first.cmd, rest))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn every_spec_has_a_unique_name_starting_with_slash() {
        let mut seen = HashSet::new();
        for spec in COMMANDS {
            assert!(
                spec.name.starts_with('/'),
                "command name `{}` does not start with /",
                spec.name
            );
            assert!(
                !spec.name.contains(char::is_whitespace),
                "command name `{}` contains whitespace",
                spec.name
            );
            assert!(
                seen.insert(spec.name),
                "duplicate command name in registry: {}",
                spec.name
            );
        }
    }

    #[test]
    fn every_spec_has_a_unique_cmd_tag() {
        let mut seen = HashSet::new();
        for spec in COMMANDS {
            assert!(
                seen.insert(spec.cmd),
                "two registry entries share Cmd tag {:?}",
                spec.cmd
            );
        }
    }

    #[test]
    fn parse_bare_command_returns_empty_args() {
        let (cmd, args) = parse("/help").expect("registered");
        assert_eq!(cmd, Cmd::Help);
        assert_eq!(args, "");
    }

    #[test]
    fn parse_with_args_trims_whitespace() {
        let (cmd, args) = parse("/save  foo.json  ").expect("registered");
        assert_eq!(cmd, Cmd::Save);
        assert_eq!(args, "foo.json");
    }

    #[test]
    fn parse_unknown_returns_none() {
        assert!(parse("/no-such-command").is_none());
        assert!(parse("/no-such arg").is_none());
    }

    #[test]
    fn parse_exit_aliases_quit() {
        let (cmd, _) = parse("/exit").expect("alias works");
        assert_eq!(cmd, Cmd::Quit);
    }

    #[test]
    fn parse_unique_prefix_resolves() {
        // `/q` is unambiguous — only `/quit` matches.
        let (cmd, args) = parse("/q").expect("unique prefix matches");
        assert_eq!(cmd, Cmd::Quit);
        assert_eq!(args, "");
        // Prefix with args still routes correctly.
        let (cmd, args) = parse("/qu  foo").expect("prefix with args matches");
        assert_eq!(cmd, Cmd::Quit);
        assert_eq!(args, "foo");
    }

    #[test]
    fn parse_ambiguous_prefix_returns_none() {
        // `/re` matches /release-notes, /reload-plugins, /reset-limits,
        // /resume, /review, /rewind — refuse to guess.
        assert!(parse("/re").is_none());
    }

    #[test]
    fn command_inventory_exposes_registered_names() {
        let names: Vec<_> = command_names().collect();
        assert!(names.contains(&"/release-notes"));
        assert!(names.contains(&"/reload-plugins"));
        assert!(names.contains(&"/reset-limits"));
        assert!(names.contains(&"/resume"));
        assert!(names.contains(&"/review"));
        assert!(names.contains(&"/rewind"));
    }

    #[test]
    fn prefix_matches_returns_all_matching_commands() {
        let matches = prefix_matches("/re");
        assert!(matches.contains(&"/release-notes"));
        assert!(matches.contains(&"/reload-plugins"));
        assert!(matches.contains(&"/reset-limits"));
        assert!(matches.contains(&"/resume"));
        assert!(matches.contains(&"/review"));
        assert!(matches.contains(&"/rewind"));
    }

    #[test]
    fn suggestions_prefer_prefix_matches() {
        let suggestions = suggestions("/re");
        assert!(suggestions.contains(&"/release-notes"));
        assert!(suggestions.contains(&"/reload-plugins"));
        assert!(suggestions.contains(&"/reset-limits"));
        assert!(suggestions.contains(&"/resume"));
        assert!(suggestions.contains(&"/review"));
        assert!(suggestions.contains(&"/rewind"));
    }

    #[test]
    fn suggestions_fall_back_to_levenshtein_for_typos() {
        assert_eq!(suggestions("/relase-notes"), vec!["/release-notes"]);
        assert!(suggestions("/rewiew").contains(&"/review"));
    }

    #[test]
    fn parse_bare_slash_returns_none() {
        assert!(parse("/").is_none());
    }

    #[test]
    fn registry_covers_core_commands() {
        let names: HashSet<&str> = COMMANDS.iter().map(|s| s.name).collect();
        for must in [
            "/help",
            "/quit",
            "/save",
            "/load",
            "/batch",
            "/export",
            "/memory",
            "/memory-reload",
            "/todo-add",
            "/todo-rm",
        ] {
            assert!(
                names.contains(must),
                "missing core command in registry: {must}"
            );
        }
    }

    #[test]
    fn consensus_help_mentions_isolated_tool_worktrees() {
        let spec = find_command("/consensus").expect("consensus command is registered");
        assert!(spec.usage.contains("[isolate|--isolate]"));
        assert!(spec.usage.contains("concurrency:<n>"));
        assert!(spec.usage.contains("--max-steps"));
        assert!(spec.usage.contains("--isolated-time-cap-secs"));
        assert!(spec.help.contains("tool worktrees"));
        assert!(spec.help.contains("c:<n> capped at 2"));
    }

    #[test]
    fn subcommands_expose_managed_vocab() {
        assert_eq!(
            subcommands(Cmd::Skills),
            &["list", "run", "enable", "disable"]
        );
        assert_eq!(subcommands(Cmd::Agents), &["list", "enable", "disable"]);
        assert_eq!(
            subcommands(Cmd::Mcp),
            &["list", "enable", "disable", "add", "remove", "reload"]
        );
    }

    #[test]
    fn default_subcommand_is_first_entry() {
        assert_eq!(default_subcommand(Cmd::Skills), Some("list"));
        assert_eq!(default_subcommand(Cmd::Agents), Some("list"));
        assert_eq!(default_subcommand(Cmd::Mcp), Some("list"));
    }

    #[test]
    fn commands_without_subcommands_are_empty() {
        assert!(subcommands(Cmd::Help).is_empty());
        assert_eq!(default_subcommand(Cmd::Help), None);
        // A representative sampling of unmanaged commands.
        assert!(subcommands(Cmd::Quit).is_empty());
        assert!(subcommands(Cmd::Status).is_empty());
    }

    #[test]
    fn only_the_three_managed_commands_carry_a_vocab() {
        let with_vocab: Vec<&str> = COMMANDS
            .iter()
            .filter(|s| !s.subcommands.is_empty())
            .map(|s| s.name)
            .collect();
        assert_eq!(with_vocab, vec!["/skills", "/agents", "/mcp"]);
    }

    #[test]
    fn managed_usage_strings_reflect_new_vocab() {
        let skills = find_command("/skills").expect("registered");
        assert!(skills.usage.contains("enable <name>"));
        assert!(skills.usage.contains("disable <name>"));
        let agents = find_command("/agents").expect("registered");
        assert!(agents.usage.contains("enable <name>"));
        assert!(agents.usage.contains("disable <name>"));
        let mcp = find_command("/mcp").expect("registered");
        assert!(mcp.usage.contains("reload"));
        assert!(mcp.usage.contains("enable <name>"));
        assert!(mcp.usage.contains("add <name> <command|url>"));
    }

    #[test]
    fn subcommand_arg_hint_parses_usage_alternatives() {
        // A subcommand that takes an arg.
        assert_eq!(subcommand_arg_hint(Cmd::Agents, "enable"), Some("<name>"));
        // A subcommand that takes no args.
        assert_eq!(subcommand_arg_hint(Cmd::Agents, "list"), None);
        // Nested `|` inside `<...>` must NOT split the alternative.
        assert_eq!(
            subcommand_arg_hint(Cmd::Mcp, "add"),
            Some("<name> <command|url>")
        );
        assert_eq!(subcommand_arg_hint(Cmd::Mcp, "reload"), None);
        // Unknown subcommand -> None.
        assert_eq!(subcommand_arg_hint(Cmd::Mcp, "bogus"), None);
        // A command without a `[...]` group -> None.
        assert_eq!(subcommand_arg_hint(Cmd::Status, "list"), None);
    }
}
