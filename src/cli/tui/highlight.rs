//! Dependency-free syntax highlighting for the TUI transcript's fenced code
//! blocks.
//!
//! This is a **pure**, deterministic, hand-rolled tokenizer: it performs no I/O
//! and has no third-party dependencies. It returns one owned [`Line<'static>`]
//! per input line, coloring keywords, string/char literals, comments, and
//! numbers while leaving everything else at the default foreground. The caller
//! ([`super::markdown`]) is responsible for prepending the dim `│ ` left border
//! to each returned row — this module emits **no** border.
//!
//! Cross-line constructs (multi-line `/* … */` block comments and Python
//! triple-quoted strings `"""…"""` / `'''…'''`) are colored across rows via a
//! small [`Cross`] state threaded between lines.
//!
//! Supported languages (with common aliases):
//!   * `rust` / `rs` — `//` + `/* */` comments, `"…"` strings, `'x'` char
//!     literals (lifetime `'a` left alone), numbers, keywords.
//!   * `python` / `py` — `#` comments, `"…"` / `'…'` strings incl. triple
//!     quotes, numbers, keywords.
//!   * `javascript` / `js` / `typescript` / `ts` / `tsx` / `jsx` — `//` +
//!     `/* */` comments, `"…"` / `'…'` strings, numbers, keywords.
//!   * `json` — no comments, `"…"` strings, numbers, `true`/`false`/`null`.
//!   * `bash` / `sh` / `shell` / `zsh` — `#` comments, `"…"` / `'…'` strings,
//!     numbers, keywords.
//!   * `go` / `golang` — `//` + `/* */` comments, `"…"` strings, `'x'` runes,
//!     numbers, keywords.
//!   * anything else (including empty) — a **generic** highlighter that colors
//!     `"…"` / `'…'` strings, numbers, and `#` / `//` line comments only.
//!
//! Robustness: the tokenizer operates on `char`s (never byte-index slicing that
//! could split a multibyte character), never panics on empty/odd input, and an
//! unterminated string or comment simply colors to the end of the line (or, for
//! block comments / triple strings, until a later closing delimiter).

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};

/// Cross-line tokenizer state carried between input rows.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Cross {
    /// Ordinary scanning.
    Normal,
    /// Inside a `/* … */` block comment awaiting its `*/` close.
    BlockComment,
    /// Inside a Python triple-quoted string; the char is the quote (`"`/`'`).
    TripleString(char),
}

/// Per-language tokenizer configuration.
struct Lang {
    /// Line-comment prefixes (e.g. `//`, `#`); scanning to end of line.
    line_comments: &'static [&'static str],
    /// Optional `(open, close)` block-comment delimiters (e.g. `/*`, `*/`).
    block_comment: Option<(&'static str, &'static str)>,
    /// Whether Python-style triple-quoted strings (`"""` / `'''`) apply.
    triple_strings: bool,
    /// Quote chars that begin a full (possibly escaped) string literal.
    string_quotes: &'static [char],
    /// Whether `'` uses the char/rune-literal heuristic (rust/go) rather than
    /// being a full string quote.
    char_quote: bool,
    /// Keyword set colored as keywords.
    keywords: &'static [&'static str],
}

/// Highlight a fenced code block. Returns one styled row per input line (NO
/// left border — the caller adds it). `lang` is the fence info string (e.g.
/// `"rust"`, `"python"`, `"ts"`); unknown/empty maps to a generic highlighter.
pub(crate) fn highlight(code: &str, lang: &str) -> Vec<Line<'static>> {
    let cfg = lang_config(lang);
    let mut out: Vec<Line<'static>> = Vec::new();
    let mut cross = Cross::Normal;
    for line in code.lines() {
        let (row, next) = highlight_line(line, &cfg, cross);
        out.push(row);
        cross = next;
    }
    out
}

// ---- styles (hardcoded for now; a later slice will theme these) ------------

fn keyword_style() -> Style {
    Style::default().fg(Color::Magenta)
}
fn string_style() -> Style {
    Style::default().fg(Color::Green)
}
fn comment_style() -> Style {
    // "dim" comment color == ANSI bright black.
    Style::default().fg(Color::DarkGray)
}
fn number_style() -> Style {
    Style::default().fg(Color::Yellow)
}
fn base_style() -> Style {
    Style::default()
}

// ---- language table --------------------------------------------------------

fn lang_config(lang: &str) -> Lang {
    // Take the first whitespace-delimited token and lowercase it so info
    // strings like "rust,ignore" or "ts {.line-numbers}" still resolve.
    let key = lang
        .split(|c: char| c.is_whitespace() || c == ',')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();

    match key.as_str() {
        "rust" | "rs" => Lang {
            line_comments: &["//"],
            block_comment: Some(("/*", "*/")),
            triple_strings: false,
            string_quotes: &['"'],
            char_quote: true,
            keywords: RUST_KEYWORDS,
        },
        "python" | "py" => Lang {
            line_comments: &["#"],
            block_comment: None,
            triple_strings: true,
            string_quotes: &['"', '\''],
            char_quote: false,
            keywords: PYTHON_KEYWORDS,
        },
        "javascript" | "js" | "typescript" | "ts" | "tsx" | "jsx" => Lang {
            line_comments: &["//"],
            block_comment: Some(("/*", "*/")),
            triple_strings: false,
            string_quotes: &['"', '\'', '`'],
            char_quote: false,
            keywords: JS_KEYWORDS,
        },
        "json" => Lang {
            line_comments: &[],
            block_comment: None,
            triple_strings: false,
            string_quotes: &['"'],
            char_quote: false,
            keywords: JSON_KEYWORDS,
        },
        "bash" | "sh" | "shell" | "zsh" => Lang {
            line_comments: &["#"],
            block_comment: None,
            triple_strings: false,
            string_quotes: &['"', '\''],
            char_quote: false,
            keywords: BASH_KEYWORDS,
        },
        "go" | "golang" => Lang {
            line_comments: &["//"],
            block_comment: Some(("/*", "*/")),
            triple_strings: false,
            string_quotes: &['"', '`'],
            char_quote: true,
            keywords: GO_KEYWORDS,
        },
        // Generic fallback: strings, numbers, and `#` / `//` line comments.
        _ => Lang {
            line_comments: &["#", "//"],
            block_comment: None,
            triple_strings: false,
            string_quotes: &['"', '\''],
            char_quote: false,
            keywords: &[],
        },
    }
}

// ---- per-line tokenizer ----------------------------------------------------

/// Tokenize a single line given the entering [`Cross`] state. Returns the
/// styled row and the state to carry into the next line.
fn highlight_line(line: &str, lang: &Lang, mut cross: Cross) -> (Line<'static>, Cross) {
    let chars: Vec<char> = line.chars().collect();
    let n = chars.len();
    let mut b = RowBuilder::new();
    let mut i = 0;

    while i < n {
        // Continuation: inside a multi-line block comment.
        if cross == Cross::BlockComment {
            let close = lang.block_comment.map(|(_, c)| c).unwrap_or("*/");
            match scan_for_seq(&chars, i, close) {
                Some(end) => {
                    push_range(&mut b, &chars, i, end, comment_style());
                    i = end;
                    cross = Cross::Normal;
                }
                None => {
                    push_range(&mut b, &chars, i, n, comment_style());
                    i = n;
                }
            }
            continue;
        }

        // Continuation: inside a triple-quoted string.
        if let Cross::TripleString(q) = cross {
            match scan_for_triple(&chars, i, q) {
                Some(end) => {
                    push_range(&mut b, &chars, i, end, string_style());
                    i = end;
                    cross = Cross::Normal;
                }
                None => {
                    push_range(&mut b, &chars, i, n, string_style());
                    i = n;
                }
            }
            continue;
        }

        let c = chars[i];

        // Line comment (scans to end of line).
        if let Some(prefix) = lang.line_comments.iter().find(|p| matches_at(&chars, i, p)) {
            let _ = prefix;
            push_range(&mut b, &chars, i, n, comment_style());
            i = n;
            continue;
        }

        // Block comment open.
        if let Some((open, close)) = lang.block_comment
            && matches_at(&chars, i, open)
        {
            match scan_for_seq(&chars, i + open.chars().count(), close) {
                Some(end) => {
                    push_range(&mut b, &chars, i, end, comment_style());
                    i = end;
                }
                None => {
                    push_range(&mut b, &chars, i, n, comment_style());
                    i = n;
                    cross = Cross::BlockComment;
                }
            }
            continue;
        }

        // Triple-quoted string open (Python).
        if lang.triple_strings && lang.string_quotes.contains(&c) && matches_triple(&chars, i, c) {
            match scan_for_triple(&chars, i + 3, c) {
                Some(end) => {
                    push_range(&mut b, &chars, i, end, string_style());
                    i = end;
                }
                None => {
                    push_range(&mut b, &chars, i, n, string_style());
                    i = n;
                    cross = Cross::TripleString(c);
                }
            }
            continue;
        }

        // Char / rune literal heuristic (rust/go): only a short, well-formed
        // `'x'` / `'\n'` is a literal; a bare `'a` (lifetime) stays default.
        if lang.char_quote && c == '\'' {
            if let Some(end) = scan_char_literal(&chars, i) {
                push_range(&mut b, &chars, i, end, string_style());
                i = end;
            } else {
                b.push(c, base_style());
                i += 1;
            }
            continue;
        }

        // Full string literal (double/single/backtick per language).
        if lang.string_quotes.contains(&c) {
            let end = scan_string(&chars, i, c);
            push_range(&mut b, &chars, i, end, string_style());
            i = end;
            continue;
        }

        // Number literal: only when not glued to the tail of an identifier.
        if c.is_ascii_digit() && (i == 0 || !is_ident_char(chars[i - 1])) {
            let mut j = i + 1;
            while j < n && (chars[j].is_ascii_alphanumeric() || chars[j] == '.' || chars[j] == '_')
            {
                j += 1;
            }
            push_range(&mut b, &chars, i, j, number_style());
            i = j;
            continue;
        }

        // Identifier / keyword.
        if is_ident_start(c) {
            let mut j = i + 1;
            while j < n && is_ident_char(chars[j]) {
                j += 1;
            }
            let word: String = chars[i..j].iter().collect();
            let style = if lang.keywords.contains(&word.as_str()) {
                keyword_style()
            } else {
                base_style()
            };
            push_range(&mut b, &chars, i, j, style);
            i = j;
            continue;
        }

        // Anything else: default foreground.
        b.push(c, base_style());
        i += 1;
    }

    (b.finish(), cross)
}

// ---- scanning helpers ------------------------------------------------------

/// True when the ASCII string `s` matches `chars` starting at index `i`.
fn matches_at(chars: &[char], i: usize, s: &str) -> bool {
    let mut idx = i;
    for sc in s.chars() {
        if idx >= chars.len() || chars[idx] != sc {
            return false;
        }
        idx += 1;
    }
    true
}

/// True when three consecutive `quote` chars begin at index `i`.
fn matches_triple(chars: &[char], i: usize, quote: char) -> bool {
    i + 2 < chars.len() && chars[i] == quote && chars[i + 1] == quote && chars[i + 2] == quote
}

/// Scan from `start` for the ASCII sequence `seq`. Returns the exclusive end
/// index just past `seq`, or `None` if it does not occur on this line.
fn scan_for_seq(chars: &[char], start: usize, seq: &str) -> Option<usize> {
    let seq_len = seq.chars().count();
    let mut j = start;
    while j < chars.len() {
        if matches_at(chars, j, seq) {
            return Some(j + seq_len);
        }
        j += 1;
    }
    None
}

/// Scan from `start` for a closing triple quote. Returns the exclusive end
/// index just past the closing `"""` / `'''`, or `None` if unterminated on
/// this line. Honors backslash escapes.
fn scan_for_triple(chars: &[char], start: usize, quote: char) -> Option<usize> {
    let n = chars.len();
    let mut j = start;
    while j < n {
        if chars[j] == '\\' {
            j += 2;
            continue;
        }
        if matches_triple(chars, j, quote) {
            return Some(j + 3);
        }
        j += 1;
    }
    None
}

/// Scan a single-line string starting at the opening `quote` (index `i`).
/// Returns the exclusive end index (just past the closing quote, or the end of
/// line if unterminated). Honors backslash escapes.
fn scan_string(chars: &[char], i: usize, quote: char) -> usize {
    let n = chars.len();
    let mut j = i + 1;
    while j < n {
        if chars[j] == '\\' {
            j += 2;
            continue;
        }
        if chars[j] == quote {
            return j + 1;
        }
        j += 1;
    }
    n
}

/// Recognize a short char/rune literal at index `i` (the opening `'`). Returns
/// the exclusive end index past the closing `'`, or `None` (e.g. a lifetime
/// `'static`, which must stay default-styled).
fn scan_char_literal(chars: &[char], i: usize) -> Option<usize> {
    let n = chars.len();
    let mut j = i + 1;
    if j >= n {
        return None;
    }
    if chars[j] == '\\' {
        // Escaped char: `'\n'`, `'\''`, `'\\'`, …
        j += 2;
    } else if chars[j] == '\'' {
        // Empty `''` is not a valid char literal.
        return None;
    } else {
        j += 1;
    }
    if j < n && chars[j] == '\'' {
        Some(j + 1)
    } else {
        None
    }
}

fn is_ident_start(c: char) -> bool {
    c == '_' || c.is_alphabetic()
}

fn is_ident_char(c: char) -> bool {
    c == '_' || c.is_alphanumeric()
}

/// Push `chars[from..to]` into the builder with a single `style`.
fn push_range(b: &mut RowBuilder, chars: &[char], from: usize, to: usize, style: Style) {
    for &c in &chars[from..to] {
        b.push(c, style);
    }
}

// ---- span coalescing -------------------------------------------------------

/// Accumulates chars into contiguous same-style [`Span`]s, minimizing span
/// count. Emits owned `String`s so the returned [`Line`] is `'static`.
struct RowBuilder {
    spans: Vec<Span<'static>>,
    buf: String,
    style: Style,
}

impl RowBuilder {
    fn new() -> Self {
        Self {
            spans: Vec::new(),
            buf: String::new(),
            style: Style::default(),
        }
    }

    fn push(&mut self, c: char, style: Style) {
        if !self.buf.is_empty() && style != self.style {
            self.flush();
        }
        self.style = style;
        self.buf.push(c);
    }

    fn flush(&mut self) {
        if !self.buf.is_empty() {
            self.spans
                .push(Span::styled(std::mem::take(&mut self.buf), self.style));
        }
    }

    fn finish(mut self) -> Line<'static> {
        self.flush();
        // A Line must carry at least one span so empty rows still render.
        if self.spans.is_empty() {
            self.spans.push(Span::styled(String::new(), base_style()));
        }
        Line::from(self.spans)
    }
}

// ---- keyword tables --------------------------------------------------------

const RUST_KEYWORDS: &[&str] = &[
    "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum", "extern",
    "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub",
    "ref", "return", "self", "Self", "static", "struct", "super", "trait", "true", "type", "union",
    "unsafe", "use", "where", "while", "Some", "None", "Ok", "Err",
];

const PYTHON_KEYWORDS: &[&str] = &[
    "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif",
    "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda",
    "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield", "None",
    "True", "False", "self",
];

const JS_KEYWORDS: &[&str] = &[
    "abstract",
    "as",
    "async",
    "await",
    "break",
    "case",
    "catch",
    "class",
    "const",
    "continue",
    "debugger",
    "default",
    "delete",
    "do",
    "else",
    "enum",
    "export",
    "extends",
    "false",
    "finally",
    "for",
    "from",
    "function",
    "get",
    "if",
    "implements",
    "import",
    "in",
    "instanceof",
    "interface",
    "let",
    "new",
    "null",
    "of",
    "private",
    "protected",
    "public",
    "readonly",
    "return",
    "set",
    "static",
    "super",
    "switch",
    "this",
    "throw",
    "true",
    "try",
    "type",
    "typeof",
    "undefined",
    "var",
    "void",
    "while",
    "yield",
];

const JSON_KEYWORDS: &[&str] = &["true", "false", "null"];

const BASH_KEYWORDS: &[&str] = &[
    "if", "then", "else", "elif", "fi", "for", "while", "until", "do", "done", "case", "esac",
    "in", "function", "select", "time", "return", "export", "local", "readonly", "declare",
    "unset", "source",
];

const GO_KEYWORDS: &[&str] = &[
    "break",
    "case",
    "chan",
    "const",
    "continue",
    "default",
    "defer",
    "else",
    "fallthrough",
    "for",
    "func",
    "go",
    "goto",
    "if",
    "import",
    "interface",
    "map",
    "package",
    "range",
    "return",
    "select",
    "struct",
    "switch",
    "type",
    "var",
    "nil",
    "true",
    "false",
    "iota",
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Concatenated plain text of every span in a line.
    fn line_text(line: &Line<'_>) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    /// The style attached to the first span whose text equals `needle`.
    fn style_of<'a>(line: &'a Line<'a>, needle: &str) -> Option<Style> {
        line.spans
            .iter()
            .find(|s| s.content.as_ref() == needle)
            .map(|s| s.style)
    }

    /// Whether any span in the whole block carries a non-default foreground.
    fn any_colored(rows: &[Line<'_>]) -> bool {
        rows.iter()
            .flat_map(|r| r.spans.iter())
            .any(|s| s.style.fg.is_some())
    }

    #[test]
    fn rust_colors_keyword_string_and_comment() {
        let rows = highlight("fn main() { let s = \"hi\"; } // done", "rust");
        assert_eq!(rows.len(), 1);
        let line = &rows[0];
        assert_eq!(
            style_of(line, "fn").and_then(|s| s.fg),
            Some(Color::Magenta),
            "`fn` keyword is magenta"
        );
        assert_eq!(
            style_of(line, "let").and_then(|s| s.fg),
            Some(Color::Magenta),
            "`let` keyword is magenta"
        );
        assert_eq!(
            style_of(line, "\"hi\"").and_then(|s| s.fg),
            Some(Color::Green),
            "string literal is green"
        );
        assert_eq!(
            style_of(line, "// done").and_then(|s| s.fg),
            Some(Color::DarkGray),
            "line comment is dim"
        );
    }

    #[test]
    fn rust_colors_number() {
        let rows = highlight("let x = 42;", "rust");
        assert_eq!(
            style_of(&rows[0], "42").and_then(|s| s.fg),
            Some(Color::Yellow),
            "number literal is yellow"
        );
    }

    #[test]
    fn rust_lifetime_is_not_a_string() {
        // `'a` is a lifetime, not a char literal — it must stay default.
        let rows = highlight("fn f<'a>(x: &'a str) {}", "rust");
        assert!(
            !rows[0]
                .spans
                .iter()
                .any(|s| s.content.as_ref().contains('\'') && s.style.fg == Some(Color::Green)),
            "lifetime must not be colored as a string"
        );
    }

    #[test]
    fn rust_char_literal_is_string() {
        let rows = highlight("let c = 'x';", "rust");
        assert_eq!(
            style_of(&rows[0], "'x'").and_then(|s| s.fg),
            Some(Color::Green),
            "char literal is green"
        );
    }

    #[test]
    fn python_colors_def_and_comment() {
        let rows = highlight("def f():  # hello", "python");
        assert_eq!(
            style_of(&rows[0], "def").and_then(|s| s.fg),
            Some(Color::Magenta),
            "`def` keyword is magenta"
        );
        assert_eq!(
            style_of(&rows[0], "# hello").and_then(|s| s.fg),
            Some(Color::DarkGray),
            "`#` comment is dim"
        );
    }

    #[test]
    fn block_comment_spans_multiple_lines() {
        let src = "let a = 1; /* start\nstill comment\nend */ let b = 2;";
        let rows = highlight(src, "rust");
        assert_eq!(rows.len(), 3);
        // Middle line is entirely comment-colored.
        assert_eq!(
            rows[1].spans.first().and_then(|s| s.style.fg),
            Some(Color::DarkGray),
            "middle of a block comment is dim"
        );
        assert_eq!(line_text(&rows[1]), "still comment");
        // The `let b` after the close is highlighted normally again.
        assert_eq!(
            style_of(&rows[2], "let").and_then(|s| s.fg),
            Some(Color::Magenta),
            "code resumes after `*/`"
        );
    }

    #[test]
    fn python_triple_string_spans_multiple_lines() {
        let src = "x = \"\"\"line one\nline two\"\"\"\ny = 1";
        let rows = highlight(src, "python");
        assert_eq!(rows.len(), 3);
        assert_eq!(
            rows[1].spans.first().and_then(|s| s.style.fg),
            Some(Color::Green),
            "inside a triple-quoted string stays green"
        );
        assert_eq!(
            style_of(&rows[2], "1").and_then(|s| s.fg),
            Some(Color::Yellow),
            "code resumes after the closing triple quote"
        );
    }

    #[test]
    fn unknown_language_uses_generic_path() {
        // Generic: strings, numbers, and `#` / `//` comments only; no keywords.
        let rows = highlight("value = \"str\" 12 # note", "cobol");
        assert_eq!(
            style_of(&rows[0], "\"str\"").and_then(|s| s.fg),
            Some(Color::Green),
            "generic colors strings"
        );
        assert_eq!(
            style_of(&rows[0], "12").and_then(|s| s.fg),
            Some(Color::Yellow),
            "generic colors numbers"
        );
        assert_eq!(
            style_of(&rows[0], "# note").and_then(|s| s.fg),
            Some(Color::DarkGray),
            "generic colors `#` comments"
        );
        assert!(
            style_of(&rows[0], "value").and_then(|s| s.fg).is_none(),
            "generic path highlights no keywords"
        );
    }

    #[test]
    fn multibyte_content_does_not_panic() {
        let src = "let s = \"héllo → 世界\"; // café ☕\n# 日本語";
        let rows = highlight(src, "rust");
        assert_eq!(rows.len(), 2);
        // Round-trips content without splitting multibyte chars.
        assert!(line_text(&rows[0]).contains("héllo → 世界"));
        assert!(line_text(&rows[1]).contains("日本語"));
    }

    #[test]
    fn empty_and_odd_input_do_not_panic() {
        assert!(highlight("", "rust").is_empty());
        // Unterminated string / block comment just colors to end.
        let rows = highlight("\"unterminated\nlet x = 1;", "rust");
        assert_eq!(rows.len(), 2);
        let _ = highlight("'", "rust");
        let _ = highlight("/*", "rust");
        let _ = highlight("\"\"\"", "python");
    }

    #[test]
    fn json_has_no_comments_but_colors_keywords() {
        let rows = highlight("{\"ok\": true, \"n\": 3} // not-a-comment", "json");
        // In JSON, `//` is not a comment, so `not-a-comment` stays default.
        assert!(
            style_of(&rows[0], "// not-a-comment")
                .and_then(|s| s.fg)
                .is_none(),
            "json has no line comments"
        );
        assert_eq!(
            style_of(&rows[0], "true").and_then(|s| s.fg),
            Some(Color::Magenta),
            "json `true` keyword is colored"
        );
        assert!(any_colored(&rows), "json still colors strings/numbers");
    }
}
