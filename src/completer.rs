//! Slash-command tab completion for the REPL.
//!
//! Hooked into rustyline via `Editor::set_helper`. Activates only when
//! the current line starts with `/` so it can't get in the way of
//! normal prose input.

use crate::commands::COMMANDS;
use rustyline::Helper;
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;

pub struct SlashHelper;

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
        // Only complete the first token, and only if it starts with `/`.
        // Once the user has typed a space they're into command args, which
        // we don't try to complete here.
        let head_end = line.find(char::is_whitespace).unwrap_or(line.len());
        if pos > head_end {
            return Ok((pos, Vec::new()));
        }
        let head = &line[..head_end];
        let Some(typed) = head.strip_prefix('/') else {
            return Ok((pos, Vec::new()));
        };

        let candidates: Vec<Pair> = COMMANDS
            .iter()
            .filter_map(|spec| {
                let name = spec.name.strip_prefix('/')?;
                if name.starts_with(typed) {
                    Some(Pair {
                        // The replacement keeps the leading `/` and adds
                        // a trailing space so the user can immediately
                        // start typing args after Tab-completing.
                        display: spec.name.to_string(),
                        replacement: format!("{} ", spec.name),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Replace from the very start of the line — we own the entire
        // `/cmd` head.
        Ok((0, candidates))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustyline::history::DefaultHistory;

    fn complete(line: &str, pos: usize) -> Vec<String> {
        let helper = SlashHelper;
        let history = DefaultHistory::new();
        let ctx = rustyline::Context::new(&history);
        let (_, pairs) = helper.complete(line, pos, &ctx).expect("complete ok");
        pairs.into_iter().map(|p| p.display).collect()
    }

    #[test]
    fn completes_unique_prefix() {
        let out = complete("/q", 2);
        assert_eq!(out, vec!["/quit"]);
    }

    #[test]
    fn completes_ambiguous_prefix_to_all_candidates() {
        let out = complete("/he", 3);
        // Should include /help and /heapdump at minimum.
        assert!(out.contains(&"/help".to_string()));
        assert!(out.contains(&"/heapdump".to_string()));
    }

    #[test]
    fn skips_non_slash_input() {
        let out = complete("hello", 5);
        assert!(out.is_empty());
    }

    #[test]
    fn skips_completion_in_arg_position() {
        // Cursor is after the space — we're in arg territory, not the
        // command name. Don't suggest anything.
        let out = complete("/save f", 7);
        assert!(out.is_empty());
    }
}
