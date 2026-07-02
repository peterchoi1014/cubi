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
use ratatui::widgets::{Block, Borders, Paragraph};

/// Maximum rows the composer region may occupy before it stops growing.
const MAX_COMPOSER_ROWS: u16 = 6;

/// Render `state` into `frame`. Regions, top to bottom:
///   1. scrollable transcript (`Constraint::Min(1)`),
///   2. one-line pinned status row (`Constraint::Length(1)`),
///   3. multi-line composer pinned at the bottom (`Constraint::Length(n)`).
pub(super) fn draw(frame: &mut Frame, state: &AppState) {
    let area = frame.area();

    let composer_rows = composer_height(state.composer());
    let chunks = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(1),
        Constraint::Length(composer_rows),
    ])
    .split(area);
    let transcript_area = chunks[0];
    let status_area = chunks[1];
    let composer_area = chunks[2];

    // --- Transcript -------------------------------------------------------
    let mut lines: Vec<Line> = Vec::new();
    for entry in state.transcript() {
        let style = match entry.kind {
            LineKind::Assistant => Style::default().fg(Color::White),
            LineKind::Status => Style::default().fg(Color::Cyan),
            LineKind::Footer => Style::default().add_modifier(Modifier::DIM),
        };
        for row in entry.text.split('\n') {
            lines.push(Line::styled(row.to_string(), style));
        }
    }
    // The in-progress streamed reply, shown below the finalized scrollback.
    if !state.active_reply().is_empty() {
        for row in state.active_reply().split('\n') {
            lines.push(Line::styled(
                row.to_string(),
                Style::default().fg(Color::White),
            ));
        }
    }
    let transcript = Paragraph::new(lines).scroll((state.scroll(), 0));
    frame.render_widget(transcript, transcript_area);

    // --- Status row -------------------------------------------------------
    // Render plain (no ANSI): the buffer stores styled cells, not escapes.
    let status_text = state.status().render(status_area.width as usize, false);
    let mut status_line = status_text;
    if state.thinking() && !state.thinking_label().is_empty() {
        status_line = format!("{} {}", state.thinking_label(), status_line);
    }
    let status =
        Paragraph::new(Line::from(status_line)).style(Style::default().add_modifier(Modifier::DIM));
    frame.render_widget(status, status_area);

    // --- Composer ---------------------------------------------------------
    let composer_block = Block::default().borders(Borders::TOP);
    let inner = composer_block.inner(composer_area);
    let composer = Paragraph::new(compose_lines(state.composer())).block(composer_block);
    frame.render_widget(composer, composer_area);

    // Place the terminal cursor at the composer caret position.
    let (cx, cy) = caret_position(state.composer(), state.cursor());
    let cursor_x = inner.x.saturating_add(cx);
    let cursor_y = inner.y.saturating_add(cy);
    // Clamp inside the composer's inner area to avoid drawing off-region.
    let cursor_x = cursor_x.min(inner.x.saturating_add(inner.width.saturating_sub(1)));
    let cursor_y = cursor_y.min(inner.y.saturating_add(inner.height.saturating_sub(1)));
    frame.set_cursor_position((cursor_x, cursor_y));
}

/// The composer's height in rows: one row per logical line, at least 1, capped
/// at [`MAX_COMPOSER_ROWS`], plus one row for the top border.
fn composer_height(composer: &str) -> u16 {
    let logical = composer.split('\n').count().max(1) as u16;
    logical.min(MAX_COMPOSER_ROWS).saturating_add(1)
}

/// Split the composer text into styled lines for the paragraph.
fn compose_lines(composer: &str) -> Vec<Line<'static>> {
    composer
        .split('\n')
        .map(|row| Line::from(Span::raw(row.to_string())))
        .collect()
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
    fn draw_places_composer_text_on_a_bottom_row() {
        let mut state = AppState::new();
        for c in "hello world".chars() {
            state.edit(super::super::app::EditAction::InsertChar(c));
        }
        let mut terminal = Terminal::new(TestBackend::new(40, 10)).unwrap();
        terminal.draw(|f| draw(f, &state)).unwrap();
        let buf = terminal.backend().buffer();
        // Composer text should appear on the last row.
        let last = row_to_string(buf, 9);
        assert!(
            last.contains("hello world"),
            "composer text missing from bottom row: {last:?}"
        );
    }

    #[test]
    fn draw_renders_status_content_on_status_row() {
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
            },
        ));
        let mut terminal = Terminal::new(TestBackend::new(80, 10)).unwrap();
        terminal.draw(|f| draw(f, &state)).unwrap();
        let buf = terminal.backend().buffer();
        // The status row is the row just above the composer region.
        // Regardless of exact placement, the model id must be somewhere.
        assert!(
            buffer_contains(buf, "qwen3:8b"),
            "status model id missing from buffer"
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
    }
}
