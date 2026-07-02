//! Pure application state for the Phase 2 terminal UI.
//!
//! [`AppState`] is owned exclusively by the (future) render task; it performs
//! **no I/O**. It is mutated only through [`AppState::apply`] (folding a
//! [`RenderEvent`] from the agent loop into the view model) and
//! [`AppState::edit`] (applying a local key/edit action to the composer). All
//! methods are synchronous and side-effect free, which keeps them trivially
//! unit-testable.

use crate::cli::status::StatusState;
use crate::cli::ui_sink::RenderEvent;
use std::path::PathBuf;

/// A single finalized line/block in the scrollback transcript.
///
/// Kept as a thin newtype over `String` (rather than a bare `Vec<String>`
/// element) so Slice B can later attach per-block styling metadata — e.g.
/// distinguishing assistant replies from status/footer lines — without
/// changing this module's public shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptLine {
    pub text: String,
    pub kind: LineKind,
}

/// The provenance of a transcript line, used later by the widgets layer to
/// pick a style. Rendering treats unknown kinds as plain text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineKind {
    /// A finalized assistant reply.
    Assistant,
    /// A status / info line.
    Status,
    /// The post-turn usage footer.
    Footer,
}

/// A local editing action against the composer, translated from a key event
/// by the (future) render task. Deliberately independent of `crossterm` so
/// this module stays pure and testable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditAction {
    /// Insert a character at the cursor.
    InsertChar(char),
    /// Delete the character immediately before the cursor.
    Backspace,
    /// Move the cursor one character to the left.
    MoveLeft,
    /// Move the cursor one character to the right.
    MoveRight,
    /// Insert a newline at the cursor (multiline composer).
    Newline,
}

/// The result of interpreting user input at a higher level than [`EditAction`]
/// — what the render task hands back to the agent loop in Slice B.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UserAction {
    /// The user submitted this (possibly multi-line) message.
    SubmitLine(String),
    /// The user asked to quit the TUI.
    Quit,
}

/// The entire view model for the TUI, owned by the render task.
#[derive(Debug, Clone)]
pub struct AppState {
    /// Finalized scrollback (assistant replies, status lines, footers).
    transcript: Vec<TranscriptLine>,
    /// The in-progress assistant reply being streamed token-by-token. Empty
    /// when no reply is streaming.
    active_reply: String,
    /// The multi-line composer buffer.
    composer: String,
    /// Cursor position as a **byte** offset into `composer` (always on a UTF-8
    /// char boundary).
    cursor: usize,
    /// Vertical scroll offset (rows) into the transcript region.
    scroll: u16,
    /// The current pinned status snapshot rendered on the status row.
    status: StatusState,
    /// Whether the model is currently "thinking" (spinner analogue).
    thinking: bool,
    /// The label shown while thinking (e.g. "thinking…").
    thinking_label: String,
}

impl AppState {
    /// Create an empty state with a placeholder status snapshot. Slice B
    /// refreshes the status via [`RenderEvent::StatusSnapshot`].
    pub fn new() -> Self {
        Self {
            transcript: Vec::new(),
            active_reply: String::new(),
            composer: String::new(),
            cursor: 0,
            scroll: 0,
            status: placeholder_status(),
            thinking: false,
            thinking_label: String::new(),
        }
    }

    // --- Read accessors used by the widgets layer / render task -----------

    pub fn transcript(&self) -> &[TranscriptLine] {
        &self.transcript
    }

    pub fn active_reply(&self) -> &str {
        &self.active_reply
    }

    pub fn composer(&self) -> &str {
        &self.composer
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn scroll(&self) -> u16 {
        self.scroll
    }

    pub fn status(&self) -> &StatusState {
        &self.status
    }

    pub fn thinking(&self) -> bool {
        self.thinking
    }

    pub fn thinking_label(&self) -> &str {
        &self.thinking_label
    }

    // --- Mutators ---------------------------------------------------------

    /// Fold one [`RenderEvent`] from the agent loop into the view model.
    /// Pure: mutates `self` only, never performs I/O.
    pub fn apply(&mut self, ev: RenderEvent) {
        match ev {
            RenderEvent::AssistantToken(tok) => {
                self.active_reply.push_str(&tok);
            }
            RenderEvent::AssistantFinal(content) => {
                // A buffered (non-streaming) reply supersedes any partial
                // streamed text for this turn.
                self.active_reply.clear();
                self.push_line(content, LineKind::Assistant);
            }
            RenderEvent::Status(msg) => {
                self.push_line(msg, LineKind::Status);
            }
            RenderEvent::UsageFooter { stats, window } => {
                self.push_line(format_footer(&stats, window), LineKind::Footer);
            }
            RenderEvent::BeginThinking { label, .. } => {
                self.thinking = true;
                self.thinking_label = label;
                // Start a fresh active reply for the upcoming step.
                self.active_reply.clear();
            }
            RenderEvent::EndThinking => {
                self.thinking = false;
                self.finalize_active_reply();
            }
            RenderEvent::StatusSnapshot(status) => {
                self.status = status;
            }
            // Forward-looking seams — no view-model effect yet.
            RenderEvent::ToolStarted { .. } | RenderEvent::ToolFinished { .. } => {}
        }
    }

    /// Move the streamed `active_reply` into the transcript as a finalized
    /// assistant block, if non-empty.
    fn finalize_active_reply(&mut self) {
        if !self.active_reply.is_empty() {
            let content = std::mem::take(&mut self.active_reply);
            self.push_line(content, LineKind::Assistant);
        }
    }

    fn push_line(&mut self, text: String, kind: LineKind) {
        self.transcript.push(TranscriptLine { text, kind });
    }

    /// Apply a local editing action to the composer. Cursor arithmetic is
    /// UTF-8 aware: the cursor is always kept on a char boundary.
    pub fn edit(&mut self, action: EditAction) {
        match action {
            EditAction::InsertChar(c) => {
                self.composer.insert(self.cursor, c);
                self.cursor += c.len_utf8();
            }
            EditAction::Newline => {
                self.composer.insert(self.cursor, '\n');
                self.cursor += 1;
            }
            EditAction::Backspace => {
                if self.cursor > 0 {
                    let prev = prev_boundary(&self.composer, self.cursor);
                    self.composer.replace_range(prev..self.cursor, "");
                    self.cursor = prev;
                }
            }
            EditAction::MoveLeft => {
                if self.cursor > 0 {
                    self.cursor = prev_boundary(&self.composer, self.cursor);
                }
            }
            EditAction::MoveRight => {
                if self.cursor < self.composer.len() {
                    self.cursor = next_boundary(&self.composer, self.cursor);
                }
            }
        }
    }

    /// Return the composer contents, clearing the buffer and resetting the
    /// cursor. Used when the user submits a message.
    pub fn take_composer(&mut self) -> String {
        self.cursor = 0;
        std::mem::take(&mut self.composer)
    }

    /// Scroll the transcript up by `n` rows (toward older content).
    pub fn scroll_up(&mut self, n: u16) {
        self.scroll = self.scroll.saturating_sub(n);
    }

    /// Scroll the transcript down by `n` rows (toward newer content).
    pub fn scroll_down(&mut self, n: u16) {
        self.scroll = self.scroll.saturating_add(n);
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

/// Byte offset of the char boundary immediately before `idx` (which must be a
/// boundary and `> 0`).
fn prev_boundary(s: &str, idx: usize) -> usize {
    let mut i = idx - 1;
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Byte offset of the char boundary immediately after `idx` (which must be a
/// boundary and `< s.len()`).
fn next_boundary(s: &str, idx: usize) -> usize {
    let mut i = idx + 1;
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// Format a one-line usage footer for the transcript. Deliberately plain (no
/// ANSI) — the widgets layer owns styling.
fn format_footer(stats: &crate::ollama::ChatStats, window: Option<usize>) -> String {
    let mut s = format!(
        "{} in · {} out · {} ms",
        stats.prompt_tokens, stats.completion_tokens, stats.elapsed_ms
    );
    if let Some(w) = window {
        s.push_str(&format!(" · ctx {w}"));
    }
    s
}

/// A neutral placeholder status snapshot for a freshly-created [`AppState`].
fn placeholder_status() -> StatusState {
    StatusState {
        model: String::new(),
        context_used: None,
        context_window: None,
        cwd: PathBuf::new(),
        prompt_tokens: 0,
        completion_tokens: 0,
        cost: "—".to_string(),
        session_details: String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ollama::ChatStats;

    fn sample_status() -> StatusState {
        StatusState {
            model: "qwen3:8b".to_string(),
            context_used: Some(1000),
            context_window: Some(8000),
            cwd: PathBuf::from("/tmp/project"),
            prompt_tokens: 10,
            completion_tokens: 20,
            cost: "$0.00 (local)".to_string(),
            session_details: "ollama · 0 msgs · sessions ok".to_string(),
        }
    }

    #[test]
    fn assistant_tokens_append_to_active_reply() {
        let mut s = AppState::new();
        s.apply(RenderEvent::AssistantToken("Hel".into()));
        s.apply(RenderEvent::AssistantToken("lo".into()));
        assert_eq!(s.active_reply(), "Hello");
        assert!(s.transcript().is_empty());
    }

    #[test]
    fn thinking_lifecycle_finalizes_active_reply_into_transcript() {
        let mut s = AppState::new();
        s.apply(RenderEvent::BeginThinking {
            label: "thinking…".into(),
            continuation: false,
        });
        assert!(s.thinking());
        assert_eq!(s.thinking_label(), "thinking…");
        s.apply(RenderEvent::AssistantToken("hi".into()));
        s.apply(RenderEvent::AssistantToken(" there".into()));
        s.apply(RenderEvent::EndThinking);
        assert!(!s.thinking());
        assert!(s.active_reply().is_empty());
        assert_eq!(s.transcript().len(), 1);
        assert_eq!(s.transcript()[0].text, "hi there");
        assert_eq!(s.transcript()[0].kind, LineKind::Assistant);
    }

    #[test]
    fn begin_thinking_starts_a_fresh_reply() {
        let mut s = AppState::new();
        s.apply(RenderEvent::AssistantToken("stale".into()));
        s.apply(RenderEvent::BeginThinking {
            label: "processing…".into(),
            continuation: true,
        });
        assert!(s.active_reply().is_empty());
    }

    #[test]
    fn assistant_final_finalizes_and_clears_partial() {
        let mut s = AppState::new();
        s.apply(RenderEvent::AssistantToken("partial".into()));
        s.apply(RenderEvent::AssistantFinal("buffered reply".into()));
        assert!(s.active_reply().is_empty());
        assert_eq!(s.transcript().len(), 1);
        assert_eq!(s.transcript()[0].text, "buffered reply");
    }

    #[test]
    fn status_and_footer_push_transcript_lines() {
        let mut s = AppState::new();
        s.apply(RenderEvent::Status("connected".into()));
        s.apply(RenderEvent::UsageFooter {
            stats: ChatStats {
                prompt_tokens: 12,
                completion_tokens: 34,
                elapsed_ms: 56,
            },
            window: Some(8000),
        });
        assert_eq!(s.transcript().len(), 2);
        assert_eq!(s.transcript()[0].kind, LineKind::Status);
        assert_eq!(s.transcript()[0].text, "connected");
        assert_eq!(s.transcript()[1].kind, LineKind::Footer);
        assert!(s.transcript()[1].text.contains("12 in"));
        assert!(s.transcript()[1].text.contains("34 out"));
        assert!(s.transcript()[1].text.contains("ctx 8000"));
    }

    #[test]
    fn status_snapshot_sets_status() {
        let mut s = AppState::new();
        s.apply(RenderEvent::StatusSnapshot(sample_status()));
        assert_eq!(s.status().model, "qwen3:8b");
        assert_eq!(s.status().context_used, Some(1000));
    }

    #[test]
    fn tool_events_are_noops() {
        let mut s = AppState::new();
        s.apply(RenderEvent::ToolStarted { name: "fs".into() });
        s.apply(RenderEvent::ToolFinished { name: "fs".into() });
        assert!(s.transcript().is_empty());
        assert!(s.active_reply().is_empty());
    }

    #[test]
    fn edit_insert_and_backspace_move_cursor() {
        let mut s = AppState::new();
        for c in "abc".chars() {
            s.edit(EditAction::InsertChar(c));
        }
        assert_eq!(s.composer(), "abc");
        assert_eq!(s.cursor(), 3);
        s.edit(EditAction::Backspace);
        assert_eq!(s.composer(), "ab");
        assert_eq!(s.cursor(), 2);
    }

    #[test]
    fn edit_cursor_movement_and_mid_insert() {
        let mut s = AppState::new();
        for c in "ac".chars() {
            s.edit(EditAction::InsertChar(c));
        }
        // Move left once (cursor between 'a' and 'c'), insert 'b'.
        s.edit(EditAction::MoveLeft);
        assert_eq!(s.cursor(), 1);
        s.edit(EditAction::InsertChar('b'));
        assert_eq!(s.composer(), "abc");
        assert_eq!(s.cursor(), 2);
        // Move right past 'c'; further right is a no-op at end.
        s.edit(EditAction::MoveRight);
        assert_eq!(s.cursor(), 3);
        s.edit(EditAction::MoveRight);
        assert_eq!(s.cursor(), 3);
    }

    #[test]
    fn edit_is_utf8_aware() {
        let mut s = AppState::new();
        s.edit(EditAction::InsertChar('é')); // 2 bytes
        s.edit(EditAction::InsertChar('x'));
        assert_eq!(s.composer(), "éx");
        assert_eq!(s.cursor(), 3);
        s.edit(EditAction::MoveLeft); // over 'x'
        assert_eq!(s.cursor(), 2);
        s.edit(EditAction::MoveLeft); // over 'é' (2 bytes)
        assert_eq!(s.cursor(), 0);
        s.edit(EditAction::MoveRight); // back over 'é'
        assert_eq!(s.cursor(), 2);
        s.edit(EditAction::Backspace); // delete 'é'
        assert_eq!(s.composer(), "x");
        assert_eq!(s.cursor(), 0);
    }

    #[test]
    fn edit_multiline_newline() {
        let mut s = AppState::new();
        for c in "foo".chars() {
            s.edit(EditAction::InsertChar(c));
        }
        s.edit(EditAction::Newline);
        for c in "bar".chars() {
            s.edit(EditAction::InsertChar(c));
        }
        assert_eq!(s.composer(), "foo\nbar");
        assert_eq!(s.cursor(), 7);
    }

    #[test]
    fn take_composer_returns_and_clears() {
        let mut s = AppState::new();
        for c in "hi".chars() {
            s.edit(EditAction::InsertChar(c));
        }
        let taken = s.take_composer();
        assert_eq!(taken, "hi");
        assert_eq!(s.composer(), "");
        assert_eq!(s.cursor(), 0);
    }

    #[test]
    fn backspace_at_start_is_noop() {
        let mut s = AppState::new();
        s.edit(EditAction::Backspace);
        assert_eq!(s.composer(), "");
        assert_eq!(s.cursor(), 0);
    }
}
