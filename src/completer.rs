//! Slash-command tab completion for the REPL.
//!
//! Hooked into rustyline via `Editor::set_helper`. Activates only when
//! the current line starts with `/`.
//!
//! Two completion modes:
//! * **Head completion** — typing the slash command name itself
//!   (e.g. `/q<Tab>` → `/quit`). Lists matching commands from the
//!   registry.
//! * **Argument completion** — typing args after a known command
//!   (e.g. `/resume <Tab>` → list of session ids). Source data is
//!   listed lazily from disk on the first tab press and cached for
//!   subsequent presses within the same REPL session. Refreshable via
//!   [`SlashHelper::invalidate_caches`] after a `/save` / `/load` /
//!   `/sessions` / `/reload-plugins` / `/mcp-reload` so the
//!   suggestions don't go stale.

use crate::commands;
use rustyline::Helper;
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use std::sync::RwLock;

/// Cap on per-command arg suggestions so a giant sessions directory
/// can't make rustyline render an unwieldy menu on Tab.
const MAX_ARG_SUGGESTIONS: usize = 64;

#[derive(Default)]
struct Caches {
    sessions: Option<Vec<String>>,
    plugins: Option<Vec<String>>,
    mcp_servers: Option<Vec<String>>,
}

pub struct SlashHelper {
    caches: RwLock<Caches>,
}

impl SlashHelper {
    pub fn new() -> Self {
        Self {
            caches: RwLock::new(Caches::default()),
        }
    }

    /// Forget all cached lookup data so the next Tab press re-reads from
    /// disk. Call this after a slash command that mutates the underlying
    /// filesystem (sessions / plugins / MCP config).
    #[allow(dead_code)]
    pub fn invalidate_caches(&self) {
        if let Ok(mut c) = self.caches.write() {
            *c = Caches::default();
        }
    }

    fn sessions(&self) -> Vec<String> {
        if let Ok(c) = self.caches.read() {
            if let Some(v) = c.sessions.as_ref() {
                return v.clone();
            }
        }
        let v = list_sessions_silent();
        if let Ok(mut c) = self.caches.write() {
            c.sessions = Some(v.clone());
        }
        v
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn plugins(&self) -> Vec<String> {
        if let Ok(c) = self.caches.read() {
            if let Some(v) = c.plugins.as_ref() {
                return v.clone();
            }
        }
        let v = list_plugins_silent();
        if let Ok(mut c) = self.caches.write() {
            c.plugins = Some(v.clone());
        }
        v
    }

    fn mcp_servers(&self) -> Vec<String> {
        if let Ok(c) = self.caches.read() {
            if let Some(v) = c.mcp_servers.as_ref() {
                return v.clone();
            }
        }
        let v = list_mcp_servers_silent();
        if let Ok(mut c) = self.caches.write() {
            c.mcp_servers = Some(v.clone());
        }
        v
    }
}

impl Default for SlashHelper {
    fn default() -> Self {
        Self::new()
    }
}

impl Helper for SlashHelper {}
impl Highlighter for SlashHelper {}
impl Validator for SlashHelper {}

impl Hinter for SlashHelper {
    type Hint = String;
}

impl Completer for SlashHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> Result<(usize, Vec<Pair>), ReadlineError> {
        let head_end = line.find(char::is_whitespace).unwrap_or(line.len());
        let head = &line[..head_end];
        if !head.starts_with('/') {
            return Ok((pos, Vec::new()));
        }

        if pos <= head_end {
            // Completing the slash-command name itself.
            let candidates: Vec<Pair> = slash_command_candidates(head)
                .into_iter()
                .map(|name| Pair {
                    display: name.to_string(),
                    replacement: format!("{} ", name),
                })
                .collect();
            return Ok((0, candidates));
        }

        // Argument position. Find the current word the cursor is on.
        let (arg_start, word) = current_arg_word(line, pos);
        let suggestions = arg_suggestions(self, line, head, word, &arg_start.position);
        let pairs: Vec<Pair> = suggestions
            .into_iter()
            .take(MAX_ARG_SUGGESTIONS)
            .map(|name| Pair {
                display: name.clone(),
                replacement: name,
            })
            .collect();
        Ok((arg_start.byte, pairs))
    }
}

fn slash_command_candidates(head: &str) -> Vec<&'static str> {
    if head == "/" {
        commands::command_names().collect()
    } else {
        commands::prefix_matches(head)
    }
}

/// Where in `line` the current arg word starts, and whether the cursor
/// is on the first arg (immediately after the command) or a later one.
struct ArgWordStart {
    /// Byte offset where the word starts (rustyline expects byte offset
    /// for the replacement anchor).
    byte: usize,
    /// 0-based position of the arg word among the args (0 = first arg).
    position: usize,
}

/// Returns the start of the word that the cursor `pos` is in, plus the
/// word's current text.
fn current_arg_word(line: &str, pos: usize) -> (ArgWordStart, &str) {
    // Slice up to cursor and find the last whitespace before it.
    let prefix = &line[..pos];
    let word_start = prefix
        .rfind(char::is_whitespace)
        .map(|i| i + 1)
        .unwrap_or(0);
    let word = &line[word_start..pos];
    // Count whitespace-separated args before this word. The first token
    // (the slash command) is index -1 conceptually; arg position 0 means
    // "the very first argument".
    let position = prefix[..word_start]
        .split_whitespace()
        .count()
        .saturating_sub(1);
    (
        ArgWordStart {
            byte: word_start,
            position,
        },
        word,
    )
}

fn arg_suggestions(
    helper: &SlashHelper,
    _line: &str,
    head: &str,
    word: &str,
    arg_pos: &usize,
) -> Vec<String> {
    // Resolve the head to a canonical command name so prefix-typed
    // commands (`/sav<Tab>`) still get arg completion.
    let canonical = canonical_command(head);
    // Managed commands (those carrying a subcommand vocabulary in the
    // registry) suggest their subcommands at the first argument position.
    if *arg_pos == 0 {
        if let Some((cmd, _)) = commands::parse(head) {
            let subs = commands::subcommands(cmd);
            if !subs.is_empty() {
                let pool = subs.iter().map(|s| s.to_string());
                return if word.is_empty() {
                    pool.collect()
                } else {
                    pool.filter(|s| s.starts_with(word)).collect()
                };
            }
        }
    }
    let pool: Vec<String> = match canonical.as_deref() {
        Some("/resume") | Some("/load") | Some("/save") if *arg_pos == 0 => helper.sessions(),
        Some("/plugin") if *arg_pos == 0 => vec!["list".to_string()],
        Some("/mcp-call")
        | Some("/mcp-tools")
        | Some("/mcp-resources")
        | Some("/mcp-read")
        | Some("/mcp-prompts")
            if *arg_pos == 0 =>
        {
            helper.mcp_servers()
        }
        _ => return Vec::new(),
    };
    if word.is_empty() {
        pool
    } else {
        pool.into_iter().filter(|s| s.starts_with(word)).collect()
    }
}

/// Maps a typed head (`/sav`, `/save`) to its canonical command name by
/// asking the parser. Returns `None` when the head doesn't resolve to a
/// known command.
fn canonical_command(head: &str) -> Option<String> {
    let (cmd, _) = commands::parse(head)?;
    commands::COMMANDS
        .iter()
        .find(|s| s.cmd == cmd)
        .map(|s| s.name.to_string())
}

// ---- silent lookup helpers ----------------------------------------------

/// Lists session ids found on disk. Returns an empty vector on any IO
/// failure so tab completion stays responsive even if the sessions dir
/// is missing or unreadable.
fn list_sessions_silent() -> Vec<String> {
    crate::sessions::SessionStore::list_all()
        .map(|metas| metas.into_iter().map(|m| m.id).collect())
        .unwrap_or_default()
}

fn list_plugins_silent() -> Vec<String> {
    crate::plugins::load_plugins()
        .into_iter()
        .map(|p| p.name)
        .collect()
}

fn list_mcp_servers_silent() -> Vec<String> {
    crate::mcp_config::McpConfig::load()
        .map(|cfg| cfg.mcp_servers.into_keys().collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustyline::history::DefaultHistory;

    fn complete(line: &str, pos: usize) -> Vec<String> {
        let helper = SlashHelper::new();
        let history = DefaultHistory::new();
        let ctx = rustyline::Context::new(&history);
        let (_, pairs) = helper.complete(line, pos, &ctx).expect("complete ok");
        pairs.into_iter().map(|p| p.display).collect()
    }

    #[test]
    fn completes_bare_slash_with_all_commands() {
        let out = complete("/", 1);
        assert!(out.contains(&"/help".to_string()));
        assert!(out.contains(&"/quit".to_string()));
        assert_eq!(out.len(), commands::COMMANDS.len());
    }

    #[test]
    fn completes_unique_prefix() {
        let out = complete("/q", 2);
        assert_eq!(out, vec!["/quit"]);
    }

    #[test]
    fn completes_ambiguous_prefix_to_all_candidates() {
        let out = complete("/he", 3);
        assert!(out.contains(&"/help".to_string()));
        assert!(out.contains(&"/heapdump".to_string()));
    }

    #[test]
    fn skips_non_slash_input() {
        let out = complete("hello", 5);
        assert!(out.is_empty());
    }

    #[test]
    fn plugin_first_arg_suggests_list() {
        let out = complete("/plugin ", 8);
        assert_eq!(out, vec!["list".to_string()]);
    }

    #[test]
    fn plugin_filters_by_word_prefix() {
        let out = complete("/plugin l", 9);
        assert_eq!(out, vec!["list".to_string()]);
        let out = complete("/plugin z", 9);
        assert!(out.is_empty());
    }

    #[test]
    fn plugin_second_position_returns_no_suggestions() {
        // arg_pos == 1 → no canned suggestions for `/plugin`.
        let out = complete("/plugin list ", 13);
        assert!(out.is_empty());
    }

    #[test]
    fn mcp_first_arg_suggests_subcommands() {
        let out = complete("/mcp ", 5);
        assert_eq!(
            out,
            vec![
                "list".to_string(),
                "enable".to_string(),
                "disable".to_string(),
                "add".to_string(),
                "remove".to_string(),
                "reload".to_string(),
            ]
        );
    }

    #[test]
    fn mcp_first_arg_filters_by_word_prefix() {
        let out = complete("/mcp e", 6);
        assert_eq!(out, vec!["enable".to_string()]);
        let out = complete("/mcp r", 6);
        assert_eq!(out, vec!["remove".to_string(), "reload".to_string()]);
    }

    #[test]
    fn skills_first_arg_suggests_subcommands() {
        let out = complete("/skills ", 8);
        assert_eq!(
            out,
            vec![
                "list".to_string(),
                "run".to_string(),
                "enable".to_string(),
                "disable".to_string(),
            ]
        );
    }

    #[test]
    fn agents_first_arg_suggests_subcommands() {
        let out = complete("/agents d", 9);
        assert_eq!(out, vec!["disable".to_string()]);
    }

    #[test]
    fn managed_command_second_position_returns_no_suggestions() {
        // arg_pos == 1 → subcommand suggestions do not fire.
        let out = complete("/mcp enable ", 12);
        assert!(out.is_empty());
    }

    #[test]
    fn unmanaged_command_has_no_subcommand_suggestions() {
        // `/status` carries no subcommand vocabulary.
        let out = complete("/status ", 8);
        assert!(out.is_empty());
    }

    #[test]
    fn canonical_command_resolves_prefix() {
        assert_eq!(canonical_command("/save"), Some("/save".to_string()));
        assert_eq!(canonical_command("/sav"), Some("/save".to_string()));
        assert_eq!(canonical_command("/nope"), None);
    }

    #[test]
    fn current_arg_word_counts_position_correctly() {
        let (start, word) = current_arg_word("/plugin ", 8);
        assert_eq!(start.position, 0);
        assert_eq!(word, "");
        let (start, word) = current_arg_word("/mcp-call srv too", 17);
        assert_eq!(start.position, 1);
        assert_eq!(word, "too");
    }

    #[test]
    fn invalidate_resets_cache() {
        let helper = SlashHelper::new();
        // Prime the cache.
        let _ = helper.plugins();
        assert!(helper.caches.read().unwrap().plugins.is_some());
        helper.invalidate_caches();
        assert!(helper.caches.read().unwrap().plugins.is_none());
    }
}
