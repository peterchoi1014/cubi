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
    Quit,
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
    },
    SlashCommandSpec {
        name: "/status",
        usage: "/status",
        help: "Show session status",
        cmd: Cmd::Status,
    },
    SlashCommandSpec {
        name: "/stats",
        usage: "/stats",
        help: "Show session statistics",
        cmd: Cmd::Stats,
    },
    SlashCommandSpec {
        name: "/usage",
        usage: "/usage",
        help: "Show session usage statistics",
        cmd: Cmd::Usage,
    },
    SlashCommandSpec {
        name: "/plan",
        usage: "/plan",
        help: "Toggle plan mode (read-only)",
        cmd: Cmd::Plan,
    },
    SlashCommandSpec {
        name: "/init",
        usage: "/init",
        help: "Create starter AICHAT.md",
        cmd: Cmd::Init,
    },
    SlashCommandSpec {
        name: "/memory",
        usage: "/memory",
        help: "Show project memory (AICHAT.md)",
        cmd: Cmd::Memory,
    },
    SlashCommandSpec {
        name: "/memory-reload",
        usage: "/memory-reload",
        help: "Re-read AICHAT.md from disk",
        cmd: Cmd::MemoryReload,
    },
    SlashCommandSpec {
        name: "/todos",
        usage: "/todos",
        help: "List todos",
        cmd: Cmd::Todos,
    },
    SlashCommandSpec {
        name: "/todo-add",
        usage: "/todo-add <text>",
        help: "Add a todo",
        cmd: Cmd::TodoAdd,
    },
    SlashCommandSpec {
        name: "/todo-done",
        usage: "/todo-done <n>",
        help: "Mark todo n as done",
        cmd: Cmd::TodoDone,
    },
    SlashCommandSpec {
        name: "/todo-rm",
        usage: "/todo-rm <n>",
        help: "Remove todo n",
        cmd: Cmd::TodoRm,
    },
    SlashCommandSpec {
        name: "/todo-clear",
        usage: "/todo-clear",
        help: "Clear all todos",
        cmd: Cmd::TodoClear,
    },
    SlashCommandSpec {
        name: "/ask",
        usage: "/ask <q>",
        help: "Record a clarifying question (single-turn)",
        cmd: Cmd::Ask,
    },
    SlashCommandSpec {
        name: "/clear",
        usage: "/clear",
        help: "Clear conversation history",
        cmd: Cmd::Clear,
    },
    SlashCommandSpec {
        name: "/history",
        usage: "/history",
        help: "Show conversation history",
        cmd: Cmd::History,
    },
    SlashCommandSpec {
        name: "/export",
        usage: "/export [-f] <f.md>",
        help: "Export conversation as Markdown",
        cmd: Cmd::Export,
    },
    SlashCommandSpec {
        name: "/save",
        usage: "/save [-f] <f.json>",
        help: "Save conversation (-f overwrites)",
        cmd: Cmd::Save,
    },
    SlashCommandSpec {
        name: "/load",
        usage: "/load <f.json>",
        help: "Load conversation",
        cmd: Cmd::Load,
    },
    SlashCommandSpec {
        name: "/batch",
        usage: "/batch <f>",
        help: "Process batch file",
        cmd: Cmd::Batch,
    },
    SlashCommandSpec {
        name: "/sessions",
        usage: "/sessions",
        help: "List auto-saved sessions for this project",
        cmd: Cmd::Sessions,
    },
    SlashCommandSpec {
        name: "/resume",
        usage: "/resume [id]",
        help: "Resume the latest (or named) auto-saved session",
        cmd: Cmd::Resume,
    },
    SlashCommandSpec {
        name: "/trust",
        usage: "/trust [revoke]",
        help: "Trust this project (or pass `revoke` to undo)",
        cmd: Cmd::Trust,
    },
    SlashCommandSpec {
        name: "/diff",
        usage: "/diff [path]",
        help: "Show `git diff` for the working tree",
        cmd: Cmd::Diff,
    },
    SlashCommandSpec {
        name: "/commit",
        usage: "/commit [-a] <msg>",
        help: "Run git commit (-a stages tracked files first)",
        cmd: Cmd::Commit,
    },
    SlashCommandSpec {
        name: "/commit-push-pr",
        usage: "/commit-push-pr [-a] <msg>",
        help: "Commit, push, and print a GitHub PR URL",
        cmd: Cmd::CommitPushPr,
    },
    SlashCommandSpec {
        name: "/undo",
        usage: "/undo [hard]",
        help: "Undo the latest commit (or hard reset HEAD~1)",
        cmd: Cmd::Undo,
    },
    SlashCommandSpec {
        name: "/review",
        usage: "/review",
        help: "Ask the model to review the current `git diff`",
        cmd: Cmd::Review,
    },
    SlashCommandSpec {
        name: "/worktree",
        usage: "/worktree [list|add <path> [branch]|remove <path>]",
        help: "Manage git worktrees (add auto-trusts the new path)",
        cmd: Cmd::Worktree,
    },
    SlashCommandSpec {
        name: "/branch",
        usage: "/branch [list|create <name>|switch <name>]",
        help: "List, create, or switch git branches",
        cmd: Cmd::Branch,
    },
    SlashCommandSpec {
        name: "/tag",
        usage: "/tag [list|<name>|create <name> [-m <msg>]]",
        help: "List or create git tags",
        cmd: Cmd::Tag,
    },
    SlashCommandSpec {
        name: "/files",
        usage: "/files",
        help: "List files tracked by git in this project",
        cmd: Cmd::Files,
    },
    SlashCommandSpec {
        name: "/add-dir",
        usage: "/add-dir <path>",
        help: "Trust an additional directory for write/exec tools",
        cmd: Cmd::AddDir,
    },
    SlashCommandSpec {
        name: "/memdir",
        usage: "/memdir",
        help: "List cross-session persistent memories",
        cmd: Cmd::Memdir,
    },
    SlashCommandSpec {
        name: "/memdir-add",
        usage: "/memdir-add <text>",
        help: "Add a persistent memory",
        cmd: Cmd::MemdirAdd,
    },
    SlashCommandSpec {
        name: "/memdir-rm",
        usage: "/memdir-rm <n>",
        help: "Remove memory by index",
        cmd: Cmd::MemdirRm,
    },
    SlashCommandSpec {
        name: "/memdir-clear",
        usage: "/memdir-clear",
        help: "Clear all persistent memories",
        cmd: Cmd::MemdirClear,
    },
    SlashCommandSpec {
        name: "/rewind",
        usage: "/rewind [n]",
        help: "Remove the last n exchanges (default 1)",
        cmd: Cmd::Rewind,
    },
    SlashCommandSpec {
        name: "/compact",
        usage: "/compact",
        help: "Summarize old turns to reduce context length",
        cmd: Cmd::Compact,
    },
    SlashCommandSpec {
        name: "/skills",
        usage: "/skills [list|run <name>]",
        help: "List or run reusable Markdown skills",
        cmd: Cmd::Skills,
    },
    SlashCommandSpec {
        name: "/hooks",
        usage: "/hooks [list | add <event> <cmd> | rm <n>]",
        help: "List, add, or remove lifecycle hooks",
        cmd: Cmd::Hooks,
    },
    SlashCommandSpec {
        name: "/mcp-tools",
        usage: "/mcp-tools",
        help: "List available MCP tools",
        cmd: Cmd::McpTools,
    },
    SlashCommandSpec {
        name: "/mcp-call",
        usage: "/mcp-call <t> <a>",
        help: "Call MCP tool",
        cmd: Cmd::McpCall,
    },
    SlashCommandSpec {
        name: "/mcp-reload",
        usage: "/mcp-reload",
        help: "Reload MCP configuration",
        cmd: Cmd::McpReload,
    },
    SlashCommandSpec {
        name: "/mcp-resources",
        usage: "/mcp-resources [server]",
        help: "List MCP resources",
        cmd: Cmd::McpResources,
    },
    SlashCommandSpec {
        name: "/mcp-read",
        usage: "/mcp-read <uri>",
        help: "Read an MCP resource by URI",
        cmd: Cmd::McpRead,
    },
    SlashCommandSpec {
        name: "/model",
        usage: "/model [name]",
        help: "Show or switch the active model",
        cmd: Cmd::Model,
    },
    SlashCommandSpec {
        name: "/version",
        usage: "/version",
        help: "Show version",
        cmd: Cmd::Version,
    },
    SlashCommandSpec {
        name: "/doctor",
        usage: "/doctor",
        help: "Sanity-check the runtime (Ollama, model, config dir, git)",
        cmd: Cmd::Doctor,
    },
    SlashCommandSpec {
        name: "/env",
        usage: "/env",
        help: "Show resolved runtime environment",
        cmd: Cmd::Env,
    },
    SlashCommandSpec {
        name: "/config",
        usage: "/config",
        help: "Show contents of ~/.ai-chat-cli/config.json",
        cmd: Cmd::Config,
    },
    SlashCommandSpec {
        name: "/permissions",
        usage: "/permissions",
        help: "List trusted directories and gated built-in tools",
        cmd: Cmd::Permissions,
    },
    SlashCommandSpec {
        name: "/tool-allow",
        usage: "/tool-allow <name>",
        help: "Allow a specific tool in this trust store",
        cmd: Cmd::ToolAllow,
    },
    SlashCommandSpec {
        name: "/tool-deny",
        usage: "/tool-deny <name>",
        help: "Deny a specific tool in this trust store",
        cmd: Cmd::ToolDeny,
    },
    SlashCommandSpec {
        name: "/bug",
        usage: "/bug [summary]",
        help: "Print a pre-filled GitHub bug URL with runtime info",
        cmd: Cmd::Bug,
    },
    SlashCommandSpec {
        name: "/issue",
        usage: "/issue [title]",
        help: "Print a pre-filled GitHub feature request URL",
        cmd: Cmd::Issue,
    },
    // -- New in 0.2.0 --
    SlashCommandSpec {
        name: "/init-verifiers",
        usage: "/init-verifiers",
        help: "Detect project verifier commands (build/test/lint) and print/save them",
        cmd: Cmd::InitVerifiers,
    },
    SlashCommandSpec {
        name: "/pr_comments",
        usage: "/pr_comments [pr#]",
        help: "Show PR review comments via `gh pr view --comments`",
        cmd: Cmd::PrComments,
    },
    SlashCommandSpec {
        name: "/security-review",
        usage: "/security-review",
        help: "Ask the model to security-review the current `git diff`",
        cmd: Cmd::SecurityReview,
    },
    SlashCommandSpec {
        name: "/autofix-pr",
        usage: "/autofix-pr [pr#]",
        help: "Fetch PR review comments and ask the model to propose fixes",
        cmd: Cmd::AutofixPr,
    },
    SlashCommandSpec {
        name: "/agents",
        usage: "/agents",
        help: "List background/sub-agent sessions",
        cmd: Cmd::Agents,
    },
    SlashCommandSpec {
        name: "/tasks",
        usage: "/tasks",
        help: "Alias for /todos (per-project task list)",
        cmd: Cmd::Tasks,
    },
    SlashCommandSpec {
        name: "/teleport",
        usage: "/teleport <path>",
        help: "Change cwd to a trusted directory (use /trust to approve first)",
        cmd: Cmd::Teleport,
    },
    SlashCommandSpec {
        name: "/passes",
        usage: "/passes [n]",
        help: "Show or set the agent-loop max passes (1..=12)",
        cmd: Cmd::Passes,
    },
    SlashCommandSpec {
        name: "/effort",
        usage: "/effort [low|medium|high]",
        help: "Show or set agent effort (maps to agent-loop pass budget)",
        cmd: Cmd::Effort,
    },
    SlashCommandSpec {
        name: "/theme",
        usage: "/theme [auto|light|dark]",
        help: "Show or set the colored-output theme",
        cmd: Cmd::Theme,
    },
    SlashCommandSpec {
        name: "/color",
        usage: "/color [on|off]",
        help: "Toggle colored output for this session",
        cmd: Cmd::Color,
    },
    SlashCommandSpec {
        name: "/output-style",
        usage: "/output-style [concise|markdown|explanatory]",
        help: "Show or set the assistant output style",
        cmd: Cmd::OutputStyle,
    },
    SlashCommandSpec {
        name: "/statusline",
        usage: "/statusline",
        help: "Show the contents of the status line",
        cmd: Cmd::Statusline,
    },
    SlashCommandSpec {
        name: "/keybindings",
        usage: "/keybindings",
        help: "Show the active rustyline keybindings",
        cmd: Cmd::Keybindings,
    },
    SlashCommandSpec {
        name: "/vim",
        usage: "/vim [on|off]",
        help: "Toggle vim-style readline editing",
        cmd: Cmd::Vim,
    },
    SlashCommandSpec {
        name: "/login",
        usage: "/login <provider> <access-token> [--refresh-token <token>] [--expires-in <seconds>]",
        help: "Store an OAuth token for a provider (persisted in ~/.ai-chat-cli/oauth.json)",
        cmd: Cmd::Login,
    },
    SlashCommandSpec {
        name: "/logout",
        usage: "/logout [provider]",
        help: "Forget the stored API key for a provider",
        cmd: Cmd::Logout,
    },
    SlashCommandSpec {
        name: "/oauth-refresh",
        usage: "/oauth-refresh [provider]",
        help: "Load stored OAuth tokens into this process and show token status",
        cmd: Cmd::OauthRefresh,
    },
    SlashCommandSpec {
        name: "/privacy-settings",
        usage: "/privacy-settings [telemetry on|off]",
        help: "Show or set local privacy preferences",
        cmd: Cmd::PrivacySettings,
    },
    SlashCommandSpec {
        name: "/mcp",
        usage: "/mcp",
        help: "Show overall MCP status (servers, tools, resources)",
        cmd: Cmd::Mcp,
    },
    SlashCommandSpec {
        name: "/plugin",
        usage: "/plugin [list]",
        help: "List plugins discovered in ~/.ai-chat-cli/plugins/",
        cmd: Cmd::Plugin,
    },
    SlashCommandSpec {
        name: "/reload-plugins",
        usage: "/reload-plugins",
        help: "Rescan the plugins directory",
        cmd: Cmd::ReloadPlugins,
    },
    SlashCommandSpec {
        name: "/cost",
        usage: "/cost",
        help: "Show estimated session cost (always $0 for local Ollama)",
        cmd: Cmd::Cost,
    },
    SlashCommandSpec {
        name: "/perf-issue",
        usage: "/perf-issue [summary]",
        help: "Print a pre-filled GitHub perf-issue URL with runtime info",
        cmd: Cmd::PerfIssue,
    },
    SlashCommandSpec {
        name: "/heapdump",
        usage: "/heapdump",
        help: "Print process resident-set / heap info if available",
        cmd: Cmd::Heapdump,
    },
    SlashCommandSpec {
        name: "/debug-tool-call",
        usage: "/debug-tool-call [on|off]",
        help: "Toggle verbose tool-call debug logging",
        cmd: Cmd::DebugToolCall,
    },
    SlashCommandSpec {
        name: "/upgrade",
        usage: "/upgrade",
        help: "Print upgrade instructions for ai-chat-cli",
        cmd: Cmd::Upgrade,
    },
    SlashCommandSpec {
        name: "/install",
        usage: "/install",
        help: "Print install instructions for ai-chat-cli + Ollama",
        cmd: Cmd::Install,
    },
    SlashCommandSpec {
        name: "/install-github-app",
        usage: "/install-github-app",
        help: "Show the GitHub-app install URL (placeholder)",
        cmd: Cmd::InstallGithubApp,
    },
    SlashCommandSpec {
        name: "/install-slack-app",
        usage: "/install-slack-app",
        help: "Show the Slack-app install URL (placeholder)",
        cmd: Cmd::InstallSlackApp,
    },
    SlashCommandSpec {
        name: "/sandbox-toggle",
        usage: "/sandbox-toggle",
        help: "Toggle strict-sandbox mode (alias for /plan)",
        cmd: Cmd::SandboxToggle,
    },
    SlashCommandSpec {
        name: "/reset-limits",
        usage: "/reset-limits",
        help: "Clear the in-process rate-limit / retry backoff counters",
        cmd: Cmd::ResetLimits,
    },
    SlashCommandSpec {
        name: "/share",
        usage: "/share <file.md>",
        help: "Export this conversation as a shareable Markdown file",
        cmd: Cmd::Share,
    },
    SlashCommandSpec {
        name: "/copy",
        usage: "/copy",
        help: "Copy the last assistant message to the system clipboard",
        cmd: Cmd::Copy,
    },
    SlashCommandSpec {
        name: "/feedback",
        usage: "/feedback [text]",
        help: "Print the feedback URL (pre-fills text if provided)",
        cmd: Cmd::Feedback,
    },
    SlashCommandSpec {
        name: "/release-notes",
        usage: "/release-notes",
        help: "Print release notes for the current version",
        cmd: Cmd::ReleaseNotes,
    },
    SlashCommandSpec {
        name: "/stickers",
        usage: "/stickers",
        help: "Print a friendly ASCII sticker sheet",
        cmd: Cmd::Stickers,
    },
    SlashCommandSpec {
        name: "/settings-sync",
        usage: "/settings-sync [init <remote>|push [msg]|pull|status]",
        help: "Sync ~/.ai-chat-cli/ via git (cross-machine config + memdir + skills)",
        cmd: Cmd::SettingsSync,
    },
    SlashCommandSpec {
        name: "/policy",
        usage: "/policy",
        help: "Show the active admin-managed policy overlay (read-only)",
        cmd: Cmd::Policy,
    },
    SlashCommandSpec {
        name: "/tip",
        usage: "/tip",
        help: "Show a quick tip about using ai-chat-cli",
        cmd: Cmd::Tip,
    },
    SlashCommandSpec {
        name: "/mcp-prompts",
        usage: "/mcp-prompts [server[:prompt]]",
        help: "List MCP prompts (or fetch a specific one)",
        cmd: Cmd::McpPrompts,
    },
    SlashCommandSpec {
        name: "/quit",
        usage: "/quit",
        help: "Exit the chat",
        cmd: Cmd::Quit,
    },
];

/// Parses an input line that starts with `/` into a `(Cmd, args)` pair.
///
/// * `args` is everything after the first whitespace, with leading/trailing
///   whitespace trimmed. For a bare `/cmd` with no arguments, `args` is `""`.
/// * Returns `None` for unknown commands so the caller can surface a "try
///   `/help`" hint.
/// * Both `/exit` and `/quit` are accepted as an exit signal (alias kept for
///   backward compatibility, intentionally undocumented in `/help`).
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

    let spec = COMMANDS.iter().find(|s| s.name == head)?;
    Some((spec.cmd, rest))
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
}
