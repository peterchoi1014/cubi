//! Pure rendering for the Phase 2 terminal UI.
//!
//! [`draw`] lays out three stacked regions and renders the current
//! [`AppState`] into a ratatui [`Frame`]. It performs **no** terminal I/O of
//! its own — the caller (the Slice B render task) owns the `Terminal` and
//! invokes `terminal.draw(|f| draw(f, &state))`. This keeps rendering a pure
//! function of `(frame, state)`, testable with a `TestBackend`.

use super::app::{AppState, LineKind};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph, Wrap};

/// Maximum rows the composer *text* region may occupy before it stops growing
/// (borders are added on top of this).
const MAX_COMPOSER_ROWS: u16 = 6;

/// Render `state` into `frame`. Regions, top to bottom:
///   1. scrollable, word-wrapped transcript (`Constraint::Min(1)`),
///   2. a fully-bordered composer pinned at the bottom
///      (`Constraint::Length(n)`), whose border carries status info at its
///      four corners.
pub(super) fn draw(frame: &mut Frame, state: &AppState) {
    let area = frame.area();

    let composer_rows = composer_height(state.composer());
    let chunks =
        Layout::vertical([Constraint::Min(1), Constraint::Length(composer_rows)]).split(area);
    let transcript_area = chunks[0];
    let composer_area = chunks[1];

    // --- Transcript -------------------------------------------------------
    let mut lines: Vec<Line> = Vec::new();
    for entry in state.transcript() {
        match entry.kind {
            // Bold role headers with their own accent colors.
            LineKind::UserHeader => lines.push(Line::styled(
                entry.text.clone(),
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            )),
            LineKind::AssistantHeader => lines.push(Line::styled(
                entry.text.clone(),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            // A blank separator row between turns.
            LineKind::Blank => lines.push(Line::from(String::new())),
            // Prose (user + assistant bodies) flows through the render_prose
            // seam the leader later rewires to the markdown renderer.
            LineKind::User | LineKind::Assistant => {
                lines.extend(render_prose(&entry.text, transcript_area.width));
            }
            // Status / footer keep their own styling.
            LineKind::Status => {
                for row in entry.text.split('\n') {
                    lines.push(Line::styled(
                        row.to_string(),
                        Style::default().fg(Color::Cyan),
                    ));
                }
            }
            LineKind::Footer => {
                for row in entry.text.split('\n') {
                    lines.push(Line::styled(
                        row.to_string(),
                        Style::default().add_modifier(Modifier::DIM),
                    ));
                }
            }
            // Framed tool-call block: bold-blue header, dim `│ `-indented
            // output rows, and a green ✓ / red ✗ status row.
            LineKind::ToolHeader => lines.push(Line::styled(
                entry.text.clone(),
                Style::default()
                    .fg(Color::Blue)
                    .add_modifier(Modifier::BOLD),
            )),
            LineKind::ToolOutput => {
                for row in entry.text.split('\n') {
                    lines.push(Line::styled(
                        format!("│ {row}"),
                        Style::default().add_modifier(Modifier::DIM),
                    ));
                }
            }
            LineKind::ToolStatus => {
                let color = if entry.text.starts_with('✗') {
                    Color::Red
                } else {
                    Color::Green
                };
                lines.push(Line::styled(entry.text.clone(), Style::default().fg(color)));
            }
        }
    }
    // The in-progress streamed reply, shown below the finalized scrollback and
    // under the `Cubi` header pushed when the turn began. Routed through the
    // same prose seam so live streaming matches finalized replies.
    if !state.active_reply().is_empty() {
        lines.extend(render_prose(state.active_reply(), transcript_area.width));
    }
    // Word-wrap long lines to the transcript width instead of truncating them.
    // `trim: false` preserves leading indentation (e.g. code blocks).
    let transcript = Paragraph::new(lines).wrap(Wrap { trim: false });
    // Scroll is measured from the bottom so the view auto-follows the newest
    // output. Clamp against the wrapped line count (which respects the same
    // `Wrap` setting) so PageUp/PageDown/mouse-wheel can't scroll into blank
    // space, and `scroll_from_bottom == 0` always pins to the last line.
    let total_rows = transcript.line_count(transcript_area.width) as u16;
    let max_offset = total_rows.saturating_sub(transcript_area.height);
    let from_bottom = state.scroll_from_bottom().min(max_offset);
    let top = max_offset.saturating_sub(from_bottom);
    let transcript = transcript.scroll((top, 0));
    frame.render_widget(transcript, transcript_area);

    // --- Composer (bordered, info at the four corners) --------------------
    let status = state.status();
    let top_left = status.path_display();
    let top_right = status.session_details.clone();
    let bottom_left = status.model_ctx_display();
    let progress = if state.thinking() && !state.thinking_label().is_empty() {
        state.thinking_label().to_string()
    } else {
        "ready".to_string()
    };
    let bottom_right = format!("{progress} · {}", status.usage_display());

    let dim = Style::default().add_modifier(Modifier::DIM);
    let composer_block = Block::bordered()
        .title_top(Line::from(Span::styled(top_left, dim)).left_aligned())
        .title_top(Line::from(Span::styled(top_right, dim)).right_aligned())
        .title_bottom(Line::from(Span::styled(bottom_left, dim)).left_aligned())
        .title_bottom(Line::from(Span::styled(bottom_right, dim)).right_aligned());
    let inner = composer_block.inner(composer_area);
    let composer = Paragraph::new(compose_lines(state.composer())).block(composer_block);
    frame.render_widget(composer, composer_area);

    // Place the terminal cursor at the composer caret position, inside the
    // block's inner area (which now accounts for all four borders).
    let (cx, cy) = caret_position(state.composer(), state.cursor());
    let cursor_x = inner.x.saturating_add(cx);
    let cursor_y = inner.y.saturating_add(cy);
    // Clamp inside the composer's inner area to avoid drawing off-region.
    let cursor_x = cursor_x.min(inner.x.saturating_add(inner.width.saturating_sub(1)));
    let cursor_y = cursor_y.min(inner.y.saturating_add(inner.height.saturating_sub(1)));
    frame.set_cursor_position((cursor_x, cursor_y));
}

/// The composer's height in rows: one row per logical line, at least 1, capped
/// at [`MAX_COMPOSER_ROWS`], plus two rows for the top and bottom borders.
fn composer_height(composer: &str) -> u16 {
    let logical = composer.split('\n').count().max(1) as u16;
    logical.min(MAX_COMPOSER_ROWS).saturating_add(2)
}

/// Split the composer text into styled lines for the paragraph.
fn compose_lines(composer: &str) -> Vec<Line<'static>> {
    composer
        .split('\n')
        .map(|row| Line::from(Span::raw(row.to_string())))
        .collect()
}

/// The single seam every piece of transcript *prose* (user messages, finalized
/// assistant replies, and the streaming `active_reply`) is routed through.
///
/// Render prose (assistant/user text) into styled rows via the TUI markdown
/// renderer, so headings, bold/italic, inline code, lists, and fenced code
/// blocks are legible in the transcript. `Paragraph::wrap` still wraps the
/// returned rows to the viewport width.
fn render_prose(text: &str, width: u16) -> Vec<Line<'static>> {
    super::markdown::render(text, width)
}

/// Compute the (column, row) caret position within the composer text for a
/// byte-offset `cursor`. Column counts characters (not bytes) on the caret's
/// line; row counts newlines before the caret.
fn caret_position(composer: &str, cursor: usize) -> (u16, u16) {
    let cursor = cursor.min(composer.len());
    let before = &composer[..cursor];
    let row = before.matches('\n').count() as u16;
    let col = match before.rfind('\n') {
        Some(nl) => before[nl + 1..].chars().count() as u16,
        None => before.chars().count() as u16,
    };
    (col, row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::ui_sink::RenderEvent;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    /// Read a full buffer row into a `String` (symbols concatenated).
    fn row_to_string(buf: &ratatui::buffer::Buffer, y: u16) -> String {
        let width = buf.area.width;
        let mut s = String::new();
        for x in 0..width {
            s.push_str(buf[(x, y)].symbol());
        }
        s
    }

    fn buffer_contains(buf: &ratatui::buffer::Buffer, needle: &str) -> bool {
        (0..buf.area.height).any(|y| row_to_string(buf, y).contains(needle))
    }

    #[test]
    fn caret_position_tracks_lines_and_columns() {
        assert_eq!(caret_position("", 0), (0, 0));
        assert_eq!(caret_position("abc", 3), (3, 0));
        assert_eq!(caret_position("foo\nba", 6), (2, 1));
        // UTF-8: 'é' is 2 bytes but 1 column.
        assert_eq!(caret_position("é", 2), (1, 0));
    }

    #[test]
    fn draw_places_composer_text_on_an_inner_row() {
        let mut state = AppState::new();
        for c in "hello world".chars() {
            state.edit(super::super::app::EditAction::InsertChar(c));
        }
        let mut terminal = Terminal::new(TestBackend::new(40, 10)).unwrap();
        terminal.draw(|f| draw(f, &state)).unwrap();
        let buf = terminal.backend().buffer();
        // Composer text sits on the inner row (row 8): the last row (9) is now
        // the bottom border, the top border is above it.
        let inner = row_to_string(buf, 8);
        assert!(
            inner.contains("hello world"),
            "composer text missing from inner row: {inner:?}"
        );
    }

    #[test]
    fn draw_renders_corner_status_on_the_border() {
        let mut state = AppState::new();
        state.apply(RenderEvent::StatusSnapshot(
            crate::cli::status::StatusState {
                model: "qwen3:8b".to_string(),
                context_used: Some(1000),
                context_window: Some(8000),
                cwd: std::path::PathBuf::from("/tmp/project"),
                prompt_tokens: 10,
                completion_tokens: 20,
                cost: "$0.00 (local)".to_string(),
                session_details: "ollama · 0 msgs · sessions ok".to_string(),
            },
        ));
        let mut terminal = Terminal::new(TestBackend::new(80, 10)).unwrap();
        terminal.draw(|f| draw(f, &state)).unwrap();
        let buf = terminal.backend().buffer();
        // The model id now rides the composer's bottom border (bottom-left
        // corner), and the path rides the top border (top-left corner).
        assert!(
            buffer_contains(buf, "qwen3:8b"),
            "status model id missing from buffer"
        );
        // With an empty composer, composer_rows = 3 → rows 7 (top border),
        // 8 (text), 9 (bottom border). The path is on the top border row.
        let top_border = row_to_string(buf, 7);
        assert!(
            top_border.contains("/tmp/project"),
            "path missing from top border row: {top_border:?}"
        );
    }

    #[test]
    fn draw_wraps_long_transcript_lines() {
        let mut state = AppState::new();
        // A line far wider than the 30-col transcript must wrap, not truncate:
        // its tail token should still be present somewhere in the buffer.
        let long = "start ".to_string() + &"word ".repeat(30) + "END_TAIL";
        state.apply(RenderEvent::Status(long));
        let mut terminal = Terminal::new(TestBackend::new(30, 20)).unwrap();
        terminal.draw(|f| draw(f, &state)).unwrap();
        let buf = terminal.backend().buffer();
        assert!(
            buffer_contains(buf, "END_TAIL"),
            "long line was truncated instead of wrapped"
        );
    }

    #[test]
    fn draw_renders_transcript_and_active_reply() {
        let mut state = AppState::new();
        state.apply(RenderEvent::Status("session started".into()));
        state.apply(RenderEvent::AssistantToken("streaming".into()));
        let mut terminal = Terminal::new(TestBackend::new(60, 12)).unwrap();
        terminal.draw(|f| draw(f, &state)).unwrap();
        let buf = terminal.backend().buffer();
        assert!(buffer_contains(buf, "session started"));
        assert!(buffer_contains(buf, "streaming"));
        // The first streamed token surfaces a `Cubi` role header above it.
        assert!(buffer_contains(buf, "Cubi"));
    }

    #[test]
    fn draw_renders_a_framed_tool_block() {
        let mut state = AppState::new();
        state.apply(RenderEvent::ToolStarted { name: "fs".into() });
        state.apply(RenderEvent::ToolFinished {
            name: "fs".into(),
            ok: true,
            output: "hello output".into(),
            elapsed_ms: 1234,
        });
        let mut terminal = Terminal::new(TestBackend::new(60, 12)).unwrap();
        terminal.draw(|f| draw(f, &state)).unwrap();
        let buf = terminal.backend().buffer();
        // Header, indented output row, and the ✓ status with elapsed time.
        assert!(buffer_contains(buf, "⚙ fs"), "missing tool header");
        assert!(
            buffer_contains(buf, "│ hello output"),
            "missing indented output"
        );
        assert!(buffer_contains(buf, "✓ fs (1.2s)"), "missing ok status row");
    }

    #[test]
    fn draw_renders_error_tool_status() {
        let mut state = AppState::new();
        state.apply(RenderEvent::ToolStarted { name: "run".into() });
        state.apply(RenderEvent::ToolFinished {
            name: "run".into(),
            ok: false,
            output: String::new(),
            elapsed_ms: 42,
        });
        let mut terminal = Terminal::new(TestBackend::new(60, 12)).unwrap();
        terminal.draw(|f| draw(f, &state)).unwrap();
        let buf = terminal.backend().buffer();
        assert!(
            buffer_contains(buf, "✗ run (42ms)"),
            "missing error status row"
        );
    }

    #[test]
    fn render_prose_splits_lines_plainly() {
        let rows = render_prose("first\nsecond", 40);
        assert_eq!(rows.len(), 2);
        // Plain default-fg content (no styling applied in the Milestone-A body).
        let joined: String = rows
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.clone()))
            .collect();
        assert_eq!(joined, "firstsecond");
    }

    #[test]
    fn draw_renders_role_headers_and_prose_blocks() {
        let mut state = AppState::new();
        // A full user → assistant exchange.
        state.push_user_turn("what is rust?");
        state.apply(RenderEvent::AssistantFinal("a systems language".into()));
        let mut terminal = Terminal::new(TestBackend::new(60, 14)).unwrap();
        terminal.draw(|f| draw(f, &state)).unwrap();
        let buf = terminal.backend().buffer();
        // Both role headers and both prose bodies must be visible.
        assert!(buffer_contains(buf, "You"), "missing You header");
        assert!(buffer_contains(buf, "Cubi"), "missing Cubi header");
        assert!(buffer_contains(buf, "what is rust?"), "missing user prose");
        assert!(
            buffer_contains(buf, "a systems language"),
            "missing assistant prose"
        );

        // A blank separator row exists between the user body and the assistant
        // header (spacing between turns is preserved).
        let rows: Vec<String> = (0..buf.area.height)
            .map(|y| row_to_string(buf, y))
            .collect();
        let user_row = rows
            .iter()
            .position(|r| r.contains("what is rust?"))
            .expect("user prose row");
        let cubi_row = rows
            .iter()
            .position(|r| r.contains("Cubi"))
            .expect("Cubi header row");
        assert!(cubi_row > user_row, "Cubi header should follow user prose");
        assert!(
            rows[user_row + 1..cubi_row]
                .iter()
                .any(|r| r.trim().is_empty()),
            "expected a blank separator row between turns"
        );
    }

    #[test]
    fn transcript_auto_follows_bottom_and_scroll_up_reveals_older() {
        // More transcript lines than the small viewport can show at once.
        let mut state = AppState::new();
        for i in 0..40 {
            state.apply(RenderEvent::Status(format!("line-{i:02}")));
        }
        // Tiny terminal so only a few transcript rows are visible.
        let mut terminal = Terminal::new(TestBackend::new(20, 8)).unwrap();

        // Default (scroll_from_bottom == 0): the newest line must be visible
        // and the oldest must be scrolled off.
        terminal.draw(|f| draw(f, &state)).unwrap();
        {
            let buf = terminal.backend().buffer();
            assert!(buffer_contains(buf, "line-39"), "bottom not auto-followed");
            assert!(
                !buffer_contains(buf, "line-00"),
                "oldest line should be scrolled off at the bottom"
            );
        }

        // Scrolling up far enough reveals the oldest line (clamped to the top).
        state.scroll_up(100);
        terminal.draw(|f| draw(f, &state)).unwrap();
        {
            let buf = terminal.backend().buffer();
            assert!(
                buffer_contains(buf, "line-00"),
                "scroll_up did not reveal the oldest line"
            );
        }

        // Scrolling back down returns to the bottom (auto-follow).
        state.scroll_down(100);
        terminal.draw(|f| draw(f, &state)).unwrap();
        {
            let buf = terminal.backend().buffer();
            assert!(
                buffer_contains(buf, "line-39"),
                "scroll_down did not return to the bottom"
            );
        }
    }
}
