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
    Review,
    Memdir,
    MemdirAdd,
    MemdirRm,
    MemdirClear,
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
        name: "/review",
        usage: "/review",
        help: "Ask the model to review the current `git diff`",
        cmd: Cmd::Review,
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
