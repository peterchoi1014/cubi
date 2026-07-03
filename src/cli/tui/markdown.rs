//! Convert a markdown string into styled ratatui rows for the TUI transcript.
//!
//! This is a **pure**, deterministic renderer: it performs no I/O and returns
//! owned [`Line<'static>`] rows so callers can push them straight into the
//! transcript model. It intentionally reuses the *ideas* of
//! [`crate::cli::render::polish_markdown`] (fenced-block detection + language
//! labels) but emits ratatui [`Span`]s instead of ANSI escape strings.
//!
//! Supported constructs (Milestone A):
//!   * ATX headings (`#`..`######`) — bold, leading `#`s stripped.
//!   * Inline **bold** (`**x**` / `__x__`) and *italic* (`*x*` / `_x_`).
//!   * Inline code (`` `x` ``) rendered in a distinct style.
//!   * Bullet (`- ` / `* ` / `+ `) and numbered (`1. ` / `1) `) lists.
//!   * Fenced code blocks — a dim `lang ▏` label row followed by each code
//!     line prefixed with a dim `│ ` left border (no syntax highlighting yet).
//!   * Plain paragraphs pass through as default-foreground rows.
//!
//! Wrapping is left to ratatui's `Paragraph::wrap` downstream; this renderer
//! never hard-truncates. `width` is accepted for future wrapping decisions.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use super::diff;

/// Render `text` (which may contain markdown) into wrapped-ready rows for a
/// transcript `width` columns wide. Pure: no I/O, deterministic.
pub(crate) fn render(text: &str, width: u16) -> Vec<Line<'static>> {
    // `width` is reserved for future wrapping heuristics; ratatui's
    // `Paragraph::wrap` still performs the final soft-wrap on our rows.
    let _ = width;

    let base = Style::default();
    let code_style = Style::default().fg(Color::Cyan);
    let border_style = Style::default().add_modifier(Modifier::DIM);

    let mut out: Vec<Line<'static>> = Vec::new();
    let mut lines = text.lines().peekable();

    while let Some(line) = lines.next() {
        if let Some(info) = fence_open(line) {
            // Dim language-label row before the block body.
            let label = if info.is_empty() {
                "code".to_string()
            } else {
                info
            };
            out.push(Line::from(Span::styled(format!("{label} ▏"), border_style)));

            // Diff/patch fences color their body instead of the plain framing:
            // collect the block body, then hand it to `diff::render_diff`.
            let is_diff = {
                let lang = label.split_whitespace().next().unwrap_or("");
                lang.eq_ignore_ascii_case("diff") || lang.eq_ignore_ascii_case("patch")
            };
            if is_diff {
                let mut body = String::new();
                while let Some(next) = lines.peek() {
                    if fence_close(next) {
                        let _ = lines.next();
                        break;
                    }
                    let code_line = lines.next().unwrap();
                    if !body.is_empty() {
                        body.push('\n');
                    }
                    body.push_str(code_line);
                }
                out.extend(diff::render_diff(&body, width));
                continue;
            }

            // Body: each line gets a dim `│ ` border + plain code text, up to
            // the matching closing fence (which is consumed but not emitted).
            while let Some(next) = lines.peek() {
                if fence_close(next) {
                    let _ = lines.next();
                    break;
                }
                let code_line = lines.next().unwrap();
                out.push(Line::from(vec![
                    Span::styled("│ ".to_string(), border_style),
                    Span::styled(code_line.to_string(), base),
                ]));
            }
            continue;
        }

        out.push(render_line(line, base, code_style));
    }

    out
}

/// Render a single non-fence line into a styled [`Line`], detecting headings
/// and list items before falling back to inline-parsed prose.
fn render_line(line: &str, base: Style, code_style: Style) -> Line<'static> {
    let trimmed = line.trim_start();

    // ATX headings: 1..=6 leading '#', then a space-separated title.
    if trimmed.starts_with('#') {
        let hashes = trimmed.chars().take_while(|&c| c == '#').count();
        if (1..=6).contains(&hashes) {
            let rest = trimmed[hashes..].trim_start();
            let heading_style = base.add_modifier(Modifier::BOLD);
            return Line::from(parse_inline(rest, heading_style, code_style));
        }
    }

    // Ordered / unordered list items.
    if let Some((prefix, content, indent)) = list_item(line) {
        let mut spans: Vec<Span<'static>> = Vec::new();
        if !indent.is_empty() {
            spans.push(Span::styled(indent, base));
        }
        spans.push(Span::styled(prefix, base));
        spans.extend(parse_inline(&content, base, code_style));
        return Line::from(spans);
    }

    // Plain paragraph line.
    Line::from(parse_inline(line, base, code_style))
}

/// Detect a list item. Returns `(marker, content, indent)` where `marker` is
/// the rendered prefix (`"• "` for bullets, `"N. "` for ordered items).
fn list_item(line: &str) -> Option<(String, String, String)> {
    let indent_len = line.len() - line.trim_start().len();
    let indent = line[..indent_len].to_string();
    let rest = &line[indent_len..];

    if let Some(stripped) = rest
        .strip_prefix("- ")
        .or_else(|| rest.strip_prefix("* "))
        .or_else(|| rest.strip_prefix("+ "))
    {
        return Some(("• ".to_string(), stripped.to_string(), indent));
    }

    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if !digits.is_empty() {
        let after = &rest[digits.len()..];
        if let Some(stripped) = after
            .strip_prefix(". ")
            .or_else(|| after.strip_prefix(") "))
        {
            return Some((format!("{digits}. "), stripped.to_string(), indent));
        }
    }

    None
}

/// Parse inline markdown (code, bold, italic) within `text`, applying `base`
/// as the starting style. Emphasis nests recursively so `**a `b` c**` keeps
/// both the bold and the inline-code styling.
fn parse_inline(text: &str, base: Style, code_style: Style) -> Vec<Span<'static>> {
    let chars: Vec<char> = text.chars().collect();
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut buf = String::new();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let c = chars[i];

        // Inline code: `code` (highest precedence, no nested parsing).
        if c == '`'
            && let Some(close) = find_single(&chars, i + 1, '`')
            && close > i + 1
        {
            if !buf.is_empty() {
                spans.push(Span::styled(std::mem::take(&mut buf), base));
            }
            let code: String = chars[i + 1..close].iter().collect();
            spans.push(Span::styled(code, code_style));
            i = close + 1;
            continue;
        }

        // Bold: **x** or __x__.
        if (c == '*' || c == '_')
            && i + 1 < len
            && chars[i + 1] == c
            && let Some(close) = find_double(&chars, i + 2, c)
        {
            if !buf.is_empty() {
                spans.push(Span::styled(std::mem::take(&mut buf), base));
            }
            let inner: String = chars[i + 2..close].iter().collect();
            let inner_style = base.add_modifier(Modifier::BOLD);
            spans.extend(parse_inline(&inner, inner_style, code_style));
            i = close + 2;
            continue;
        }

        // Italic: *x* or _x_.
        if (c == '*' || c == '_')
            && let Some(close) = find_single(&chars, i + 1, c)
            && close > i + 1
        {
            if !buf.is_empty() {
                spans.push(Span::styled(std::mem::take(&mut buf), base));
            }
            let inner: String = chars[i + 1..close].iter().collect();
            let inner_style = base.add_modifier(Modifier::ITALIC);
            spans.extend(parse_inline(&inner, inner_style, code_style));
            i = close + 1;
            continue;
        }

        buf.push(c);
        i += 1;
    }

    if !buf.is_empty() {
        spans.push(Span::styled(buf, base));
    }

    // A Line must have at least one span so empty lines still render.
    if spans.is_empty() {
        spans.push(Span::styled(String::new(), base));
    }

    spans
}

/// Index of the next `target` char at or after `start`, if any.
fn find_single(chars: &[char], start: usize, target: char) -> Option<usize> {
    (start..chars.len()).find(|&j| chars[j] == target)
}

/// Index of the first of two consecutive `target` chars at or after `start`.
fn find_double(chars: &[char], start: usize, target: char) -> Option<usize> {
    let mut j = start;
    while j + 1 < chars.len() {
        if chars[j] == target && chars[j + 1] == target {
            return Some(j);
        }
        j += 1;
    }
    None
}

/// Returns the fenced-block info string when `line` opens a fence. Mirrors
/// [`crate::cli::render`]'s fence detection.
fn fence_open(line: &str) -> Option<String> {
    let t = line.trim_start();
    if !t.starts_with("```") {
        return None;
    }
    Some(t.trim_start_matches('`').trim().to_string())
}

/// Whether `line` is a closing code fence (only backticks after trimming).
fn fence_close(line: &str) -> bool {
    let t = line.trim();
    t == "```" || (t.starts_with("```") && t.chars().all(|c| c == '`'))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Concatenated plain text of every span in a line.
    fn line_text(line: &Line<'_>) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    /// Whether any span in the line carries the given modifier.
    fn line_has_modifier(line: &Line<'_>, m: Modifier) -> bool {
        line.spans.iter().any(|s| s.style.add_modifier.contains(m))
    }

    #[test]
    fn heading_is_bold_without_hashes() {
        let rows = render("## Hello World", 80);
        assert_eq!(rows.len(), 1);
        assert_eq!(line_text(&rows[0]), "Hello World");
        assert!(
            !line_text(&rows[0]).contains('#'),
            "hashes must be stripped: {:?}",
            line_text(&rows[0])
        );
        assert!(
            line_has_modifier(&rows[0], Modifier::BOLD),
            "heading must be bold"
        );
    }

    #[test]
    fn inline_code_gets_code_style() {
        let rows = render("use the `foo()` call", 80);
        assert_eq!(rows.len(), 1);
        let code_span = rows[0]
            .spans
            .iter()
            .find(|s| s.content.as_ref() == "foo()")
            .expect("inline code span present");
        assert_eq!(
            code_span.style.fg,
            Some(Color::Cyan),
            "inline code carries the distinct code style"
        );
    }

    #[test]
    fn bold_span_present() {
        let rows = render("this is **strong** text", 80);
        let strong = rows[0]
            .spans
            .iter()
            .find(|s| s.content.as_ref() == "strong")
            .expect("bold span present");
        assert!(strong.style.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn italic_span_present() {
        let rows = render("this is _soft_ text", 80);
        let soft = rows[0]
            .spans
            .iter()
            .find(|s| s.content.as_ref() == "soft")
            .expect("italic span present");
        assert!(soft.style.add_modifier.contains(Modifier::ITALIC));
    }

    #[test]
    fn fenced_block_has_label_and_bordered_rows() {
        let input = "```rust\nfn x() {}\nlet y = 1;\n```";
        let rows = render(input, 80);
        // Row 0: dim language label ending in the label marker.
        assert!(
            line_text(&rows[0]).starts_with("rust"),
            "label row starts with language: {:?}",
            line_text(&rows[0])
        );
        assert!(
            line_text(&rows[0]).contains('▏'),
            "label row has the label marker"
        );
        assert!(
            line_has_modifier(&rows[0], Modifier::DIM),
            "label row is dim"
        );
        // Following rows are bordered code lines.
        assert_eq!(rows.len(), 3, "label + two code rows");
        assert!(
            line_text(&rows[1]).starts_with("│ "),
            "code row has left border: {:?}",
            line_text(&rows[1])
        );
        assert!(line_text(&rows[1]).contains("fn x() {}"));
        assert!(line_text(&rows[2]).starts_with("│ "));
        assert!(line_text(&rows[2]).contains("let y = 1;"));
    }

    #[test]
    fn bullet_list_renders_bullet() {
        let rows = render("- first item", 80);
        assert_eq!(rows.len(), 1);
        assert!(
            line_text(&rows[0]).starts_with("• "),
            "bullet prefix present: {:?}",
            line_text(&rows[0])
        );
        assert!(line_text(&rows[0]).contains("first item"));
    }

    #[test]
    fn numbered_list_preserves_number() {
        let rows = render("3. third", 80);
        assert_eq!(line_text(&rows[0]), "3. third");
    }

    #[test]
    fn diff_fence_colors_add_and_remove_rows() {
        let input = "```diff\n@@ -1 +1 @@\n-old line\n+new line\n```";
        let rows = render(input, 80);
        // Row 0 is still the dim `diff ▏` label.
        assert!(
            line_text(&rows[0]).starts_with("diff"),
            "label row starts with language: {:?}",
            line_text(&rows[0])
        );
        assert!(line_text(&rows[0]).contains('▏'));
        // Body rows are diff-colored, NOT plain `│ `-bordered.
        let add = rows
            .iter()
            .find(|r| line_text(r) == "+new line")
            .expect("addition row present and unbordered");
        assert_eq!(
            add.spans.first().and_then(|s| s.style.fg),
            Some(Color::Green),
            "addition row is green"
        );
        let del = rows
            .iter()
            .find(|r| line_text(r) == "-old line")
            .expect("deletion row present and unbordered");
        assert_eq!(
            del.spans.first().and_then(|s| s.style.fg),
            Some(Color::Red),
            "deletion row is red"
        );
        // No plain `│ ` framing was applied to the diff body.
        assert!(
            !rows.iter().any(|r| line_text(r).starts_with("│ ")),
            "diff body must not use the plain code border"
        );
    }

    #[test]
    fn plain_paragraph_passthrough() {
        let rows = render("just some plain text", 80);
        assert_eq!(rows.len(), 1);
        assert_eq!(line_text(&rows[0]), "just some plain text");
    }
}
