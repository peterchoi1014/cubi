//! ANSI SGR → ratatui spans parser for framed tool output.
//!
//! Captured command output (`/skills`, `/agents`, `/mcp`, and `!`-shell) is
//! built with the `colored` crate and carries ANSI escape sequences. ratatui
//! does not interpret ANSI, so the transcript renderer calls [`ansi_spans`] to
//! turn a single row of raw text into a sequence of styled [`Span`]s carrying
//! **visible text only** — the escape bytes never leak into the rendered
//! output.
//!
//! Only the SGR (`\x1b[…m`) subset the `colored` crate emits is interpreted:
//! reset, bold/dim (and their clears), italic, underline, the default-fg
//! reset, and the 16 basic/bright foreground colors. Background SGR params are
//! accepted but ignored. Anything else — other CSI sequences (cursor moves),
//! OSC sequences, and 256-color / truecolor SGR params — is consumed and
//! dropped so a raw `\x1b`/`[…m` can never appear on screen (arbitrary
//! sequences can arrive via `!`-shell output).

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;

/// Parse `text` (a single logical row, no embedded `\n`) into styled spans of
/// visible text, starting from `base`. A plain string with no escapes yields a
/// single `base`-styled span. Unknown/unsupported escape sequences are
/// consumed without leaking any bytes into the returned spans.
pub(super) fn ansi_spans(text: &str, base: Style) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut cur = base;
    let mut buf = String::new();
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if c != '\x1b' {
            buf.push(c);
            continue;
        }
        // An escape introducer. Dispatch on the following byte.
        match chars.peek().copied() {
            Some('[') => {
                chars.next(); // consume '['
                // Collect parameter/intermediate bytes up to and including the
                // final byte in '@'..='~'.
                let mut params = String::new();
                let mut final_byte = None;
                for f in chars.by_ref() {
                    if ('@'..='~').contains(&f) {
                        final_byte = Some(f);
                        break;
                    }
                    params.push(f);
                }
                // Only 'm' (SGR) changes style; every other CSI final byte
                // (cursor moves, erases, …) is consumed and dropped.
                if final_byte == Some('m') {
                    if !buf.is_empty() {
                        spans.push(Span::styled(std::mem::take(&mut buf), cur));
                    }
                    cur = apply_sgr(&params, base, cur);
                }
            }
            Some(']') => {
                chars.next(); // consume ']'
                // OSC: consume up to a BEL (`\x07`) or ST (`\x1b\\`).
                loop {
                    match chars.next() {
                        None => break,
                        Some('\x07') => break,
                        Some('\x1b') => {
                            if chars.peek() == Some(&'\\') {
                                chars.next();
                            }
                            break;
                        }
                        Some(_) => {}
                    }
                }
            }
            // A lone ESC or any other escape: drop the ESC only.
            _ => {}
        }
    }
    if !buf.is_empty() {
        spans.push(Span::styled(buf, cur));
    }
    // Guarantee at least one span so a plain (or empty) string maps to one
    // base-styled span, matching the previous single-style row behavior.
    if spans.is_empty() {
        spans.push(Span::styled(String::new(), base));
    }
    spans
}

/// Apply one SGR sequence's `params` (the `;`-separated numbers between `\x1b[`
/// and `m`) to `cur`, returning the updated style. `base` is the style a reset
/// (`0`) restores. Multiple params apply left to right.
fn apply_sgr(params: &str, base: Style, cur: Style) -> Style {
    // An empty parameter list (`\x1b[m`) means reset.
    if params.is_empty() {
        return base;
    }
    let codes: Vec<&str> = params.split(';').collect();
    let mut style = cur;
    let mut i = 0;
    while i < codes.len() {
        let code: u16 = codes[i].parse().unwrap_or(0);
        match code {
            0 => style = base,
            1 => style = style.add_modifier(Modifier::BOLD),
            2 => style = style.add_modifier(Modifier::DIM),
            22 => {
                style = style
                    .remove_modifier(Modifier::BOLD)
                    .remove_modifier(Modifier::DIM);
            }
            3 => style = style.add_modifier(Modifier::ITALIC),
            23 => style = style.remove_modifier(Modifier::ITALIC),
            4 => style = style.add_modifier(Modifier::UNDERLINED),
            24 => style = style.remove_modifier(Modifier::UNDERLINED),
            39 => style = style.fg(base.fg.unwrap_or(Color::Reset)),
            30..=37 => style = style.fg(basic_color(code - 30)),
            90..=97 => style = style.fg(bright_color(code - 90)),
            // Extended fg/bg: `38;5;n`, `38;2;r;g;b` (and `48;…`). Consume the
            // trailing params so they can't be misread as separate SGR codes;
            // the color itself is ignored (approximated as no change).
            38 | 48 => match codes.get(i + 1).copied() {
                Some("5") => i += 2,
                Some("2") => i += 4,
                _ => {}
            },
            // Background colors (40-47, 100-107) and `49` default-bg are
            // accepted but ignored.
            _ => {}
        }
        i += 1;
    }
    style
}

/// Map an SGR 30-37 index to a ratatui basic color.
fn basic_color(n: u16) -> Color {
    match n {
        0 => Color::Black,
        1 => Color::Red,
        2 => Color::Green,
        3 => Color::Yellow,
        4 => Color::Blue,
        5 => Color::Magenta,
        6 => Color::Cyan,
        7 => Color::White,
        _ => Color::Reset,
    }
}

/// Map an SGR 90-97 index to a ratatui bright color (`colored` emits these for
/// its `bright_*` helpers).
fn bright_color(n: u16) -> Color {
    match n {
        0 => Color::DarkGray,
        1 => Color::LightRed,
        2 => Color::LightGreen,
        3 => Color::LightYellow,
        4 => Color::LightBlue,
        5 => Color::LightMagenta,
        6 => Color::LightCyan,
        7 => Color::White,
        _ => Color::Reset,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Join the visible text of all spans into one string.
    fn joined(spans: &[Span<'static>]) -> String {
        spans.iter().map(|s| s.content.as_ref()).collect()
    }

    #[test]
    fn plain_string_yields_one_base_styled_span() {
        let base = Style::default()
            .fg(Color::Reset)
            .add_modifier(Modifier::DIM);
        let spans = ansi_spans("just plain text", base);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].content.as_ref(), "just plain text");
        assert_eq!(spans[0].style, base);
    }

    #[test]
    fn sgr_color_and_modifiers_produce_expected_style_and_visible_text() {
        let base = Style::default();
        // `colored`'s bright_green() → 92; bold → 1; reset → 0.
        let spans = ansi_spans("\x1b[92mgreen\x1b[0m plain", base);
        // Visible text must contain no escape bytes.
        let text = joined(&spans);
        assert_eq!(text, "green plain");
        assert!(!text.contains('\x1b'));
        // The "green" span must carry LightGreen (bright green).
        let green_span = spans
            .iter()
            .find(|s| s.content.as_ref() == "green")
            .expect("a span with the text 'green'");
        assert_eq!(green_span.style.fg, Some(Color::LightGreen));

        // Bold + basic red, then a `22` clears the bold/dim.
        let spans = ansi_spans("\x1b[1;31mred\x1b[22mafter", base);
        let red = spans
            .iter()
            .find(|s| s.content.as_ref() == "red")
            .expect("'red' span");
        assert_eq!(red.style.fg, Some(Color::Red));
        assert!(red.style.add_modifier.contains(Modifier::BOLD));
        let after = spans
            .iter()
            .find(|s| s.content.as_ref() == "after")
            .expect("'after' span");
        assert!(!after.style.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn reset_restores_base_style() {
        let base = Style::default()
            .fg(Color::Reset)
            .add_modifier(Modifier::DIM);
        let spans = ansi_spans("\x1b[92mx\x1b[0my", base);
        let y = spans
            .iter()
            .find(|s| s.content.as_ref() == "y")
            .expect("'y' span");
        assert_eq!(y.style, base);
    }

    #[test]
    fn unknown_and_extended_sgr_are_consumed_without_leaking_bytes() {
        let base = Style::default();
        // 256-color and truecolor fg params, plus an unknown SGR code.
        let spans = ansi_spans("\x1b[38;5;201mA\x1b[38;2;10;20;30mB\x1b[99mC", base);
        let text = joined(&spans);
        assert_eq!(text, "ABC");
        assert!(!text.contains('\x1b'));
        assert!(
            !text.contains('m'),
            "no stray SGR final byte leaked: {text:?}"
        );
    }

    #[test]
    fn cursor_move_csi_and_osc_are_consumed_without_leaking_bytes() {
        let base = Style::default();
        // A cursor-position CSI (`\x1b[2J` erase / `\x1b[10;5H` move) and an OSC
        // title sequence terminated by BEL, then by ST.
        let spans = ansi_spans(
            "\x1b[2J\x1b[10;5Hvisible\x1b]0;my title\x07tail\x1b]2;x\x1b\\end",
            base,
        );
        let text = joined(&spans);
        assert_eq!(text, "visibletailend");
        assert!(!text.contains('\x1b'));
        assert!(!text.contains(']'), "no stray OSC bytes leaked: {text:?}");
    }
}
