//! Helpers for folding multi-line REPL input into a single user turn.
//!
//! Two complementary triggers are supported:
//!
//! * **Triple-quote fence** — a line starting with `"""` opens a block. If
//!   only `"""` is on the line, every subsequent line until a line equal to
//!   `"""` is part of the body. If the opener has trailing content
//!   (`"""text`), that trailing content becomes the first body line and the
//!   closer is still a bare `"""` on its own line. A single-line
//!   `"""text"""` is also accepted as a complete block with `text` as the
//!   body.
//! * **Backslash continuation** — a line ending in an unescaped `\` is
//!   joined to the next line with the trailing backslash *and* newline
//!   dropped. Stackable across many lines.
//!
//! The folding logic is factored as a pure helper so it can be unit-tested
//! without driving rustyline. The REPL collects the relevant lines (it
//! already knows when a block is "open" via [`opener_kind`]) and then calls
//! [`fold_multiline`] once to produce the submitted body.
//!
//! The detection helpers (`opener_kind`, `is_continuation`) live here too so
//! the REPL never duplicates the rules.

/// Classifies the first line a user types after the normal prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpenerKind {
    /// Plain single-line input — submit as-is.
    None,
    /// `"""...` style: keep reading body lines until a `"""` closer.
    Fence,
    /// `"""text"""` on one line — already complete, no continuation.
    FenceClosedInline,
    /// Trailing unescaped `\` — keep reading until a line without one.
    Backslash,
}

/// Inspects a single freshly-read line and classifies it.
pub(crate) fn opener_kind(line: &str) -> OpenerKind {
    if let Some(rest) = line.strip_prefix("\"\"\"") {
        if rest.len() >= 3 && rest.ends_with("\"\"\"") {
            return OpenerKind::FenceClosedInline;
        }
        return OpenerKind::Fence;
    }
    if is_continuation(line) {
        return OpenerKind::Backslash;
    }
    OpenerKind::None
}

/// Returns true if `line` ends in an unescaped backslash — i.e. an odd
/// number of trailing backslashes. `foo\\` is *not* a continuation (the
/// trailing pair is an escaped backslash); `foo\` is.
pub(crate) fn is_continuation(line: &str) -> bool {
    let trailing = line.chars().rev().take_while(|c| *c == '\\').count();
    trailing % 2 == 1
}

/// Folds a slice of raw input lines into a single submitted body and
/// reports how many input lines were consumed.
///
/// `lines[0]` must be the trigger line as returned by the readline. The
/// caller is responsible for ensuring the slice is "complete" — i.e. ends
/// with the fence closer or the first non-continuation line. If the slice
/// is open-ended (no closer found), every line is still consumed and the
/// best-effort body is returned.
pub(crate) fn fold_multiline(lines: &[String]) -> (String, usize) {
    if lines.is_empty() {
        return (String::new(), 0);
    }
    let first = &lines[0];

    if let Some(rest) = first.strip_prefix("\"\"\"") {
        // Inline `"""text"""` — closed on a single line.
        if rest.len() >= 3 && rest.ends_with("\"\"\"") {
            let inner = &rest[..rest.len() - 3];
            return (inner.to_string(), 1);
        }
        let mut body: Vec<String> = Vec::new();
        if !rest.is_empty() {
            body.push(rest.to_string());
        }
        let mut consumed = 1;
        for line in &lines[1..] {
            consumed += 1;
            if line == "\"\"\"" {
                return (body.join("\n"), consumed);
            }
            body.push(line.clone());
        }
        // No closer found — return what we have. The interactive REPL keeps
        // reading until it sees a closer, so this branch is mainly a
        // defensive landing for tests with truncated input.
        (body.join("\n"), consumed)
    } else if is_continuation(first) {
        let mut body = String::new();
        let mut consumed = 0;
        for line in lines {
            consumed += 1;
            if is_continuation(line) {
                // Drop the trailing `\` and concatenate without a newline.
                body.push_str(&line[..line.len() - 1]);
            } else {
                body.push_str(line);
                return (body, consumed);
            }
        }
        (body, consumed)
    } else {
        (first.clone(), 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(input: &[&str]) -> Vec<String> {
        input.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn opener_classifies_correctly() {
        assert_eq!(opener_kind("hi"), OpenerKind::None);
        assert_eq!(opener_kind("\"\"\""), OpenerKind::Fence);
        assert_eq!(opener_kind("\"\"\"foo"), OpenerKind::Fence);
        assert_eq!(
            opener_kind("\"\"\"foo\"\"\""),
            OpenerKind::FenceClosedInline
        );
        assert_eq!(opener_kind("foo\\"), OpenerKind::Backslash);
        assert_eq!(opener_kind("foo\\\\"), OpenerKind::None);
        assert_eq!(opener_kind("foo\\\\\\"), OpenerKind::Backslash);
    }

    #[test]
    fn fold_bare_fence_body() {
        let (body, n) = fold_multiline(&lines(&["\"\"\"", "foo", "bar", "\"\"\""]));
        assert_eq!(body, "foo\nbar");
        assert_eq!(n, 4);
    }

    #[test]
    fn fold_inline_open_fence() {
        let (body, n) = fold_multiline(&lines(&["\"\"\"foo", "bar", "\"\"\""]));
        assert_eq!(body, "foo\nbar");
        assert_eq!(n, 3);
    }

    #[test]
    fn fold_inline_closed_fence() {
        let (body, n) = fold_multiline(&lines(&["\"\"\"hello world\"\"\""]));
        assert_eq!(body, "hello world");
        assert_eq!(n, 1);
    }

    #[test]
    fn fold_empty_fence() {
        let (body, n) = fold_multiline(&lines(&["\"\"\"", "\"\"\""]));
        assert_eq!(body, "");
        assert_eq!(n, 2);
    }

    #[test]
    fn fold_backslash_continuation_two_lines() {
        let (body, n) = fold_multiline(&lines(&["foo\\", "bar"]));
        assert_eq!(body, "foobar");
        assert_eq!(n, 2);
    }

    #[test]
    fn fold_backslash_continuation_stacked() {
        let (body, n) = fold_multiline(&lines(&["a\\", "b\\", "c\\", "d"]));
        assert_eq!(body, "abcd");
        assert_eq!(n, 4);
    }

    #[test]
    fn fold_passthrough_plain_line() {
        let (body, n) = fold_multiline(&lines(&["hello there"]));
        assert_eq!(body, "hello there");
        assert_eq!(n, 1);
    }

    #[test]
    fn fold_fence_without_closer_consumes_remaining() {
        // Truncated input (no closing `"""`): caller still gets a body.
        let (body, n) = fold_multiline(&lines(&["\"\"\"", "foo", "bar"]));
        assert_eq!(body, "foo\nbar");
        assert_eq!(n, 3);
    }

    #[test]
    fn fold_mixed_fence_contains_backslashes_verbatim() {
        let (body, n) = fold_multiline(&lines(&["\"\"\"", "foo\\", "bar", "\"\"\""]));
        // Inside a fence we don't honor backslash continuation — that
        // would make pasting code blocks lossy.
        assert_eq!(body, "foo\\\nbar");
        assert_eq!(n, 4);
    }
}
