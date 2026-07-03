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

use super::theme::Theme;

/// Maximum tool-output rows shown inside a framed tool block before the rest
/// are collapsed into a dim `… N more lines` note.
const MAX_TOOL_OUTPUT_ROWS: usize = 12;

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

/// The provenance of a transcript line, used by the widgets layer to pick a
/// style. Rendering treats unknown kinds as plain text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineKind {
    /// A bold role header introducing a user turn (rendered e.g. green `You`).
    UserHeader,
    /// A bold role header introducing an assistant turn (rendered e.g. cyan
    /// `Cubi`).
    AssistantHeader,
    /// A finalized user message body (prose).
    User,
    /// A finalized assistant reply (prose).
    Assistant,
    /// A blank spacer row used to separate consecutive turns.
    Blank,
    /// A status / info line.
    Status,
    /// The post-turn usage footer.
    Footer,
    /// The header row opening a framed tool-call block (`⚙ <name>`).
    ToolHeader,
    /// A single indented output row inside a tool-call block (dim, `│ ` prefix
    /// added by the widgets layer).
    ToolOutput,
    /// A tool-output row belonging to a block whose output is a unified diff.
    /// The widgets layer colors these via [`super::diff`] instead of the plain
    /// `│ ` framing used for [`LineKind::ToolOutput`].
    ToolDiff,
    /// The trailing status row of a tool-call block. Its leading glyph is `✓`
    /// (success) or `✗` (error), which the widgets layer inspects to pick a
    /// green/red style.
    ToolStatus,
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
    /// How many rows the view is scrolled up from the bottom of the
    /// transcript. `0` means "pinned to the bottom" so the newest output is
    /// always visible (auto-follow); larger values reveal older content. The
    /// widgets layer clamps this to the scrollable range at render time using
    /// the wrapped line count, so it never needs to know the viewport size.
    scroll_from_bottom: u16,
    /// The current pinned status snapshot rendered on the status row.
    status: StatusState,
    /// Whether the model is currently "thinking" (spinner analogue).
    thinking: bool,
    /// The label shown while thinking (e.g. "thinking…").
    thinking_label: String,
    /// Spinner frame index for the thinking indicator, advanced by the render
    /// task on each redraw tick. Purely presentational.
    spinner_frame: usize,
    /// Milliseconds elapsed since thinking began, set by the render task (which
    /// owns the clock) before each draw. `0` when not thinking.
    thinking_elapsed_ms: u64,
    /// Whether a `Cubi` role header has already been pushed for the current
    /// assistant turn. Set when the header is emitted and reset when a new user
    /// turn begins, so streaming, buffered, and multi-step replies all share a
    /// single header per turn.
    assistant_header_shown: bool,
    /// The active color palette. Sourced from the persisted `/theme` choice in
    /// `run_tui`; defaults to `auto` (today's appearance) so a state built in
    /// tests renders exactly as before.
    theme: Theme,
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
            scroll_from_bottom: 0,
            status: placeholder_status(),
            thinking: false,
            thinking_label: String::new(),
            spinner_frame: 0,
            thinking_elapsed_ms: 0,
            assistant_header_shown: false,
            theme: Theme::default(),
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

    /// Rows the view is scrolled up from the bottom (0 = pinned to newest).
    pub fn scroll_from_bottom(&self) -> u16 {
        self.scroll_from_bottom
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

    /// Current spinner frame index for the thinking indicator.
    pub fn spinner_frame(&self) -> usize {
        self.spinner_frame
    }

    /// Milliseconds elapsed since thinking began (`0` when not thinking).
    pub fn thinking_elapsed_ms(&self) -> u64 {
        self.thinking_elapsed_ms
    }

    /// Set the presentational thinking-animation state. Called by the render
    /// task (which owns the clock and redraw tick) before each draw so the
    /// widgets layer stays a pure function of `AppState`.
    pub fn set_thinking_anim(&mut self, frame: usize, elapsed_ms: u64) {
        self.spinner_frame = frame;
        self.thinking_elapsed_ms = elapsed_ms;
    }

    /// The active color palette used by the widgets layer.
    pub fn theme(&self) -> Theme {
        self.theme
    }

    /// Set the active color palette. Called once by `run_tui` after resolving
    /// the persisted `/theme` choice.
    pub fn set_theme(&mut self, theme: Theme) {
        self.theme = theme;
    }

    // --- Mutators ---------------------------------------------------------

    /// Fold one [`RenderEvent`] from the agent loop into the view model.
    /// Pure: mutates `self` only, never performs I/O.
    pub fn apply(&mut self, ev: RenderEvent) {
        match ev {
            RenderEvent::AssistantToken(tok) => {
                self.ensure_assistant_header();
                self.active_reply.push_str(&tok);
            }
            RenderEvent::AssistantFinal(content) => {
                // A buffered (non-streaming) reply supersedes any partial
                // streamed text for this turn.
                self.ensure_assistant_header();
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
                self.ensure_assistant_header();
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
            RenderEvent::ToolStarted { name } => {
                // Open a framed tool block below any prior content. Does NOT
                // touch the assistant-header latch, so the tool block appears
                // between assistant steps without triggering a second `Cubi`
                // header for the turn.
                self.ensure_blank_separator();
                self.push_line(format!("⚙ {name}"), LineKind::ToolHeader);
            }
            RenderEvent::ToolFinished {
                name,
                ok,
                output,
                elapsed_ms,
            } => {
                self.push_tool_output(&output);
                let glyph = if ok { '✓' } else { '✗' };
                let status = format!("{glyph} {name} ({})", format_elapsed(elapsed_ms));
                self.push_line(status, LineKind::ToolStatus);
            }
        }
    }

    /// Push the (already capped) tool output as indented rows, truncating to
    /// [`MAX_TOOL_OUTPUT_ROWS`] lines with a dim `… N more lines` note. Empty
    /// output produces no rows.
    fn push_tool_output(&mut self, output: &str) {
        let trimmed = output.trim_end_matches('\n');
        if trimmed.is_empty() {
            return;
        }
        let rows: Vec<&str> = trimmed.split('\n').collect();
        let shown = rows.len().min(MAX_TOOL_OUTPUT_ROWS);
        // Patch-shaped output is colored as a diff; everything else keeps the
        // plain `│ `-bordered framing (unchanged Slice-1 behavior).
        let kind = if super::diff::looks_like_diff(trimmed) {
            LineKind::ToolDiff
        } else {
            LineKind::ToolOutput
        };
        for row in &rows[..shown] {
            self.push_line((*row).to_string(), kind);
        }
        if rows.len() > shown {
            let more = rows.len() - shown;
            self.push_line(format!("… {more} more lines"), LineKind::ToolOutput);
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

    /// Push a blank spacer row unless the transcript is empty or already ends
    /// in one — this is how consecutive turns get a visual separator.
    fn ensure_blank_separator(&mut self) {
        match self.transcript.last() {
            Some(last) if last.kind != LineKind::Blank => {
                self.push_line(String::new(), LineKind::Blank);
            }
            None => {} // nothing to separate from yet
            _ => {}    // already blank
        }
    }

    /// Emit a single `Cubi` role header for the current assistant turn (with a
    /// leading blank separator), if one has not been emitted already. Idempotent
    /// across the tokens/thinking/final events of one turn and across multi-step
    /// turns, so the header never duplicates.
    fn ensure_assistant_header(&mut self) {
        if !self.assistant_header_shown {
            self.ensure_blank_separator();
            self.push_line("Cubi".to_string(), LineKind::AssistantHeader);
            self.assistant_header_shown = true;
        }
    }

    /// Begin a user turn: a `You` role header followed by the message body as
    /// prose (with a leading blank separator between turns). Resets the
    /// assistant-header latch so the upcoming reply gets its own `Cubi` header.
    /// Used by both the live submit path and resumed-history seeding so the two
    /// look identical.
    pub fn push_user_turn(&mut self, text: &str) {
        self.ensure_blank_separator();
        self.push_line("You".to_string(), LineKind::UserHeader);
        self.push_line(text.to_string(), LineKind::User);
        self.assistant_header_shown = false;
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

    /// Scroll the transcript up by `n` rows (toward older content). Moving away
    /// from the bottom disables auto-follow until the view returns to the
    /// bottom.
    pub fn scroll_up(&mut self, n: u16) {
        self.scroll_from_bottom = self.scroll_from_bottom.saturating_add(n);
    }

    /// Scroll the transcript down by `n` rows (toward newer content). Reaching
    /// the bottom (0) re-enables auto-follow.
    pub fn scroll_down(&mut self, n: u16) {
        self.scroll_from_bottom = self.scroll_from_bottom.saturating_sub(n);
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

/// Format a tool's elapsed time for its status row: seconds with one decimal
/// at or above one second (e.g. `1.2s`), otherwise milliseconds (e.g. `840ms`).
fn format_elapsed(elapsed_ms: u64) -> String {
    if elapsed_ms >= 1000 {
        format!("{:.1}s", elapsed_ms as f64 / 1000.0)
    } else {
        format!("{elapsed_ms}ms")
    }
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
        // The first token emits a single `Cubi` header (no leading blank when
        // the transcript is empty), and no more.
        assert_eq!(s.transcript().len(), 1);
        assert_eq!(s.transcript()[0].kind, LineKind::AssistantHeader);
        assert_eq!(s.transcript()[0].text, "Cubi");
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
        // A `Cubi` header (pushed on BeginThinking) followed by the finalized
        // reply body.
        assert_eq!(s.transcript().len(), 2);
        assert_eq!(s.transcript()[0].kind, LineKind::AssistantHeader);
        assert_eq!(s.transcript()[1].text, "hi there");
        assert_eq!(s.transcript()[1].kind, LineKind::Assistant);
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
        // Header (from the first token) + the buffered body; the header is not
        // duplicated by the final event.
        assert_eq!(s.transcript().len(), 2);
        assert_eq!(s.transcript()[0].kind, LineKind::AssistantHeader);
        assert_eq!(s.transcript()[1].kind, LineKind::Assistant);
        assert_eq!(s.transcript()[1].text, "buffered reply");
    }

    #[test]
    fn push_user_turn_emits_header_prose_and_resets_latch() {
        let mut s = AppState::new();
        s.push_user_turn("what is rust?");
        // First turn on an empty transcript: no leading blank separator.
        assert_eq!(s.transcript().len(), 2);
        assert_eq!(s.transcript()[0].kind, LineKind::UserHeader);
        assert_eq!(s.transcript()[0].text, "You");
        assert_eq!(s.transcript()[1].kind, LineKind::User);
        assert_eq!(s.transcript()[1].text, "what is rust?");
    }

    #[test]
    fn full_turn_has_single_headers_and_blank_separators() {
        let mut s = AppState::new();
        s.push_user_turn("hello");
        // Multi-step assistant turn: two thinking steps must share ONE header.
        s.apply(RenderEvent::BeginThinking {
            label: "thinking…".into(),
            continuation: false,
        });
        s.apply(RenderEvent::AssistantToken("step one".into()));
        s.apply(RenderEvent::EndThinking);
        s.apply(RenderEvent::BeginThinking {
            label: "more…".into(),
            continuation: true,
        });
        s.apply(RenderEvent::AssistantToken("step two".into()));
        s.apply(RenderEvent::EndThinking);

        let kinds: Vec<LineKind> = s.transcript().iter().map(|l| l.kind).collect();
        assert_eq!(
            kinds,
            vec![
                LineKind::UserHeader,
                LineKind::User,
                LineKind::Blank,
                LineKind::AssistantHeader,
                LineKind::Assistant,
                LineKind::Assistant,
            ]
        );
        // Exactly one assistant header despite two steps.
        assert_eq!(
            s.transcript()
                .iter()
                .filter(|l| l.kind == LineKind::AssistantHeader)
                .count(),
            1
        );
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
    fn tool_started_pushes_a_framed_header() {
        let mut s = AppState::new();
        s.apply(RenderEvent::ToolStarted { name: "fs".into() });
        // A blank separator (none, transcript empty) then the header row.
        assert_eq!(s.transcript().len(), 1);
        assert_eq!(s.transcript()[0].kind, LineKind::ToolHeader);
        assert_eq!(s.transcript()[0].text, "⚙ fs");
    }

    #[test]
    fn tool_finished_pushes_output_and_ok_status() {
        let mut s = AppState::new();
        s.apply(RenderEvent::ToolStarted { name: "fs".into() });
        s.apply(RenderEvent::ToolFinished {
            name: "fs".into(),
            ok: true,
            output: "line one\nline two".into(),
            elapsed_ms: 1234,
        });
        let kinds: Vec<LineKind> = s.transcript().iter().map(|l| l.kind).collect();
        assert_eq!(
            kinds,
            vec![
                LineKind::ToolHeader,
                LineKind::ToolOutput,
                LineKind::ToolOutput,
                LineKind::ToolStatus,
            ]
        );
        let status = s.transcript().last().unwrap();
        assert_eq!(status.kind, LineKind::ToolStatus);
        assert_eq!(status.text, "✓ fs (1.2s)");
    }

    #[test]
    fn tool_finished_error_uses_cross_glyph_and_ms() {
        let mut s = AppState::new();
        s.apply(RenderEvent::ToolStarted { name: "run".into() });
        s.apply(RenderEvent::ToolFinished {
            name: "run".into(),
            ok: false,
            output: String::new(),
            elapsed_ms: 840,
        });
        // Empty output → header + status only.
        assert_eq!(s.transcript().len(), 2);
        let status = s.transcript().last().unwrap();
        assert_eq!(status.kind, LineKind::ToolStatus);
        assert_eq!(status.text, "✗ run (840ms)");
    }

    #[test]
    fn tool_output_truncates_to_twelve_rows_with_more_note() {
        let mut s = AppState::new();
        let output = (0..20)
            .map(|i| format!("row-{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        s.apply(RenderEvent::ToolStarted { name: "big".into() });
        s.apply(RenderEvent::ToolFinished {
            name: "big".into(),
            ok: true,
            output,
            elapsed_ms: 10,
        });
        let output_rows: Vec<&TranscriptLine> = s
            .transcript()
            .iter()
            .filter(|l| l.kind == LineKind::ToolOutput)
            .collect();
        // 12 shown rows + 1 "… N more lines" note.
        assert_eq!(output_rows.len(), 13);
        assert_eq!(output_rows.last().unwrap().text, "… 8 more lines");
    }

    #[test]
    fn diff_shaped_tool_output_is_marked_as_tool_diff() {
        let mut s = AppState::new();
        s.apply(RenderEvent::ToolStarted {
            name: "apply".into(),
        });
        s.apply(RenderEvent::ToolFinished {
            name: "apply".into(),
            ok: true,
            output: "@@ -1 +1 @@\n-old\n+new".into(),
            elapsed_ms: 10,
        });
        let kinds: Vec<LineKind> = s.transcript().iter().map(|l| l.kind).collect();
        assert_eq!(
            kinds,
            vec![
                LineKind::ToolHeader,
                LineKind::ToolDiff,
                LineKind::ToolDiff,
                LineKind::ToolDiff,
                LineKind::ToolStatus,
            ],
            "unified-diff tool output must be tagged ToolDiff, not ToolOutput"
        );
    }

    #[test]
    fn non_diff_tool_output_stays_plain_tool_output() {
        let mut s = AppState::new();
        s.apply(RenderEvent::ToolStarted { name: "fs".into() });
        s.apply(RenderEvent::ToolFinished {
            name: "fs".into(),
            ok: true,
            output: "plain line one\nplain line two".into(),
            elapsed_ms: 10,
        });
        assert!(
            s.transcript().iter().all(|l| l.kind != LineKind::ToolDiff),
            "prose tool output must not be colored as a diff"
        );
    }

    #[test]
    fn tool_block_does_not_add_a_second_assistant_header() {
        // A tool block appears between assistant steps; the `Cubi` header
        // latch must remain intact so only one header is emitted per turn.
        let mut s = AppState::new();
        s.push_user_turn("do it");
        s.apply(RenderEvent::BeginThinking {
            label: "thinking…".into(),
            continuation: false,
        });
        s.apply(RenderEvent::AssistantToken("calling a tool".into()));
        s.apply(RenderEvent::EndThinking);
        s.apply(RenderEvent::ToolStarted { name: "fs".into() });
        s.apply(RenderEvent::ToolFinished {
            name: "fs".into(),
            ok: true,
            output: "ok".into(),
            elapsed_ms: 5,
        });
        s.apply(RenderEvent::BeginThinking {
            label: "more…".into(),
            continuation: true,
        });
        s.apply(RenderEvent::AssistantToken("all done".into()));
        s.apply(RenderEvent::EndThinking);
        assert_eq!(
            s.transcript()
                .iter()
                .filter(|l| l.kind == LineKind::AssistantHeader)
                .count(),
            1,
            "exactly one Cubi header despite a tool block between steps"
        );
    }

    #[test]
    fn format_elapsed_formats_seconds_and_millis() {
        assert_eq!(format_elapsed(999), "999ms");
        assert_eq!(format_elapsed(1000), "1.0s");
        assert_eq!(format_elapsed(1234), "1.2s");
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
