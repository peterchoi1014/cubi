//! Colored unified-diff rendering for the TUI transcript.
//!
//! This is a **pure**, deterministic renderer: it performs no I/O and returns
//! owned [`Line<'static>`] rows. When a fenced markdown block declares a
//! `diff`/`patch` language (or a tool emits patch-shaped output), the body is a
//! unified diff and is colored here instead of being wrapped in the plain
//! `│ `-bordered framing that [`super::markdown`] uses for ordinary code.
//!
//! Coloring rules (mirrors common diff palettes):
//!   * `+` additions → green, `-` deletions → red.
//!   * `@@ … @@` hunk headers → cyan.
//!   * `+++ ` / `--- ` file headers → dim bold (NOT colored as add/remove).
//!   * everything else → default foreground.
//!
//! Wrapping is left to ratatui's `Paragraph::wrap` downstream; this renderer
//! never hard-truncates. `width` is accepted for future wrapping decisions.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// True when `text` looks like a unified diff: it either contains an
/// `@@ … @@` hunk header, or has several body lines beginning with `+`/`-`
/// that are not the `+++`/`---` file-header markers.
pub(crate) fn looks_like_diff(text: &str) -> bool {
    let mut change_lines = 0usize;
    for line in text.lines() {
        if is_hunk_header(line) {
            return true;
        }
        if is_addition(line) || is_deletion(line) {
            change_lines += 1;
        }
    }
    change_lines >= 2
}

/// Render a unified diff into colored rows. Pure/deterministic; owned
/// [`Line<'static>`]. Does not hard-truncate — soft-wrapping is downstream.
pub(crate) fn render_diff(patch: &str, width: u16) -> Vec<Line<'static>> {
    // `width` is reserved for future wrapping heuristics; ratatui's
    // `Paragraph::wrap` still performs the final soft-wrap on our rows.
    let _ = width;

    patch.lines().map(render_diff_line).collect()
}

/// Style a single diff line by its leading marker.
fn render_diff_line(line: &str) -> Line<'static> {
    let style = if is_file_header(line) {
        // `+++ ` / `--- ` file headers: dim bold, never add/remove colors.
        Style::default()
            .add_modifier(Modifier::DIM)
            .add_modifier(Modifier::BOLD)
    } else if is_hunk_header(line) {
        Style::default().fg(Color::Cyan)
    } else if is_addition(line) {
        Style::default().fg(Color::Green)
    } else if is_deletion(line) {
        Style::default().fg(Color::Red)
    } else {
        Style::default()
    };
    Line::from(Span::styled(line.to_string(), style))
}

/// A `@@ … @@` hunk header.
fn is_hunk_header(line: &str) -> bool {
    line.starts_with("@@")
}

/// A `+++ ` / `--- ` file-header marker.
fn is_file_header(line: &str) -> bool {
    line.starts_with("+++") || line.starts_with("---")
}

/// An addition line (`+` but not the `+++` file header).
fn is_addition(line: &str) -> bool {
    line.starts_with('+') && !line.starts_with("+++")
}

/// A deletion line (`-` but not the `---` file header).
fn is_deletion(line: &str) -> bool {
    line.starts_with('-') && !line.starts_with("---")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Concatenated plain text of every span in a line.
    fn line_text(line: &Line<'_>) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    /// Foreground color of a line's first span, if any.
    fn line_fg(line: &Line<'_>) -> Option<Color> {
        line.spans.first().and_then(|s| s.style.fg)
    }

    const SAMPLE: &str =
        "--- a/file.rs\n+++ b/file.rs\n@@ -1,3 +1,3 @@\n unchanged\n-removed line\n+added line";

    #[test]
    fn additions_are_green() {
        let rows = render_diff(SAMPLE, 80);
        let add = rows
            .iter()
            .find(|r| line_text(r) == "+added line")
            .expect("addition row present");
        assert_eq!(line_fg(add), Some(Color::Green));
    }

    #[test]
    fn deletions_are_red() {
        let rows = render_diff(SAMPLE, 80);
        let del = rows
            .iter()
            .find(|r| line_text(r) == "-removed line")
            .expect("deletion row present");
        assert_eq!(line_fg(del), Some(Color::Red));
    }

    #[test]
    fn hunk_header_is_cyan() {
        let rows = render_diff(SAMPLE, 80);
        let hunk = rows
            .iter()
            .find(|r| line_text(r).starts_with("@@"))
            .expect("hunk header present");
        assert_eq!(line_fg(hunk), Some(Color::Cyan));
    }

    #[test]
    fn file_headers_not_miscolored_as_add_or_remove() {
        let rows = render_diff(SAMPLE, 80);
        let plus_hdr = rows
            .iter()
            .find(|r| line_text(r) == "+++ b/file.rs")
            .expect("+++ header present");
        let minus_hdr = rows
            .iter()
            .find(|r| line_text(r) == "--- a/file.rs")
            .expect("--- header present");
        // Neither is green (add) nor red (remove); both are dim bold.
        assert_ne!(line_fg(plus_hdr), Some(Color::Green));
        assert_ne!(line_fg(minus_hdr), Some(Color::Red));
        assert!(
            plus_hdr.spans[0].style.add_modifier.contains(Modifier::DIM),
            "+++ header is dim"
        );
        assert!(
            plus_hdr.spans[0]
                .style
                .add_modifier
                .contains(Modifier::BOLD),
            "+++ header is bold"
        );
        assert!(
            minus_hdr.spans[0]
                .style
                .add_modifier
                .contains(Modifier::DIM),
            "--- header is dim"
        );
    }

    #[test]
    fn context_line_is_default_fg() {
        let rows = render_diff(SAMPLE, 80);
        let ctx = rows
            .iter()
            .find(|r| line_text(r) == " unchanged")
            .expect("context row present");
        assert_eq!(line_fg(ctx), None);
    }

    #[test]
    fn looks_like_diff_true_on_real_hunk() {
        assert!(looks_like_diff(SAMPLE));
        assert!(looks_like_diff("@@ -1 +1 @@\n-a\n+b"));
    }

    #[test]
    fn looks_like_diff_false_on_prose() {
        assert!(!looks_like_diff(
            "This is a paragraph of prose.\nIt has multiple lines but no diff markers."
        ));
        // A lone leading '-' bullet is not enough to look like a diff.
        assert!(!looks_like_diff("- a single bullet point"));
    }
}
