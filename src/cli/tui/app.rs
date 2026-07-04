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
    /// Past submitted composer lines, oldest first, used for Up/Down recall.
    /// Seeded from the persisted REPL history at TUI start so recall spans
    /// sessions. Empties and consecutive duplicates are never pushed.
    input_history: Vec<String>,
    /// Current history-navigation cursor. `None` means "not navigating" (the
    /// composer holds a live draft); `Some(i)` means the composer currently
    /// mirrors `input_history[i]`.
    history_index: Option<usize>,
    /// The live draft stashed when history navigation begins, restored when the
    /// user steps back past the newest entry (exiting navigation).
    history_draft: String,
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
            input_history: Vec::new(),
            history_index: None,
            history_draft: String::new(),
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
        // Captured command output (`/skills`, `/mcp`, `/agents`, `!`-shell) is
        // built with the `colored` crate and carries ANSI escape sequences.
        // ratatui does not interpret ANSI, so it would render the raw escape
        // bytes as literal garbage. Strip them FIRST, then run diff-detection
        // and row-splitting on the visible text so both are based on what the
        // user actually sees. The transcript renders its own styled spans.
        let cleaned = crate::out::strip_ansi(output);
        let trimmed = cleaned.trim_end_matches('\n');
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
    /// UTF-8 aware: the cursor is always kept on a char boundary. Any edit
    /// exits input-history navigation: the current buffer becomes a fresh
    /// draft.
    pub fn edit(&mut self, action: EditAction) {
        self.exit_history_nav();
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

    // --- Tab completion ---------------------------------------------------

    /// Complete the token at the cursor in place. Completes a leading slash
    /// command name when the cursor is within the first word, or an `@file`
    /// mention as a filesystem path relative to the current directory;
    /// otherwise a no-op. Panic-free and UTF-8 aware; IO errors yield no
    /// completion.
    pub fn complete(&mut self) {
        // Slash-command name: composer starts with '/' and the cursor sits
        // within the first (whitespace-delimited) word.
        if self.composer.starts_with('/') {
            let word_end = self
                .composer
                .find(char::is_whitespace)
                .unwrap_or(self.composer.len());
            if self.cursor <= word_end {
                self.complete_slash(word_end);
                return;
            }
            // Past the command name: for the managed commands (those with a
            // subcommand vocabulary) complete the first argument word against
            // that vocabulary.
            if self.complete_subcommand(word_end) {
                return;
            }
        }
        // `@file` mention: the token containing the cursor starts with '@'.
        let (start, end) = self.token_bounds();
        if self.composer[start..end].starts_with('@') {
            self.complete_file(start, end);
        }
    }

    /// Byte bounds of the whitespace-delimited token containing the cursor.
    fn token_bounds(&self) -> (usize, usize) {
        let s = &self.composer;
        let mut start = self.cursor;
        while start > 0 {
            let prev = prev_boundary(s, start);
            if s[prev..start]
                .chars()
                .next()
                .is_some_and(char::is_whitespace)
            {
                break;
            }
            start = prev;
        }
        let mut end = self.cursor;
        while end < s.len() {
            let next = next_boundary(s, end);
            if s[end..next].chars().next().is_some_and(char::is_whitespace) {
                break;
            }
            end = next;
        }
        (start, end)
    }

    /// Complete the leading slash command spanning `0..word_end`.
    fn complete_slash(&mut self, word_end: usize) {
        let word = self.composer[..word_end].to_string();
        let matches = crate::commands::prefix_matches(&word);
        match matches.as_slice() {
            [] => {}
            [only] => {
                // Append a trailing space only when the command is not already
                // followed by whitespace, so completing mid-line (e.g. the
                // cursor sitting before the space in "/hel foo") does not insert
                // a second space.
                let followed_by_space = self.composer[word_end..]
                    .chars()
                    .next()
                    .is_some_and(char::is_whitespace);
                let replacement = if followed_by_space {
                    only.to_string()
                } else {
                    format!("{only} ")
                };
                self.replace_range(0, word_end, &replacement);
            }
            many => {
                let common = common_prefix(many);
                if common.len() > word_end {
                    self.replace_range(0, word_end, &common);
                }
            }
        }
    }

    /// Complete the first argument word (the *subcommand*) for a managed
    /// command whose head spans `0..word_end`. Returns `true` when it handled
    /// the completion (so the caller stops), `false` to fall through to `@file`
    /// completion.
    ///
    /// Behaviour: an empty partial (e.g. `/skills `) inserts the DEFAULT
    /// subcommand; a unique prefix match is replaced and given a trailing
    /// space; multiple matches extend to their longest common prefix.
    ///
    /// Only fires when the cursor sits within the first argument word; a third
    /// token or later is left untouched. UTF-8 / byte safe throughout.
    fn complete_subcommand(&mut self, word_end: usize) -> bool {
        // Resolve the head to a command with a subcommand vocabulary. Prefix
        // heads (`/mc`) resolve via the parser too.
        let head = &self.composer[..word_end];
        let Some((cmd, _)) = crate::commands::parse(head) else {
            return false;
        };
        let subs = crate::commands::subcommands(cmd);
        if subs.is_empty() {
            return false;
        }

        // Byte bounds of the first argument word (skipping the whitespace that
        // separates it from the command name).
        let rest = &self.composer[word_end..];
        let ws_len = rest.len() - rest.trim_start().len();
        let sub_start = word_end + ws_len;
        let after = &self.composer[sub_start..];
        let sub_end = after
            .find(char::is_whitespace)
            .map(|i| sub_start + i)
            .unwrap_or(self.composer.len());

        // Only complete when the cursor is within the first argument word.
        if self.cursor < sub_start || self.cursor > sub_end {
            return false;
        }

        let word = self.composer[sub_start..sub_end].to_string();
        if word.is_empty() {
            // Bare command + space: insert the default (first) subcommand.
            let default = subs[0];
            self.replace_range(sub_start, sub_end, &format!("{default} "));
            return true;
        }

        let matches: Vec<&&str> = subs.iter().filter(|s| s.starts_with(&word)).collect();
        match matches.as_slice() {
            [] => {}
            [only] => {
                let followed_by_space = self.composer[sub_end..]
                    .chars()
                    .next()
                    .is_some_and(char::is_whitespace);
                let replacement = if followed_by_space {
                    (**only).to_string()
                } else {
                    format!("{only} ")
                };
                self.replace_range(sub_start, sub_end, &replacement);
            }
            many => {
                let names: Vec<&str> = many.iter().map(|s| **s).collect();
                let common = common_prefix(&names);
                if common.len() > word.len() {
                    self.replace_range(sub_start, sub_end, &common);
                }
            }
        }
        true
    }

    /// Complete the `@file` token spanning `start..end` against the filesystem.
    fn complete_file(&mut self, start: usize, end: usize) {
        let token = self.composer[start..end].to_string();
        let partial = &token[1..]; // strip the leading '@'
        // Split into an already-typed directory portion (kept verbatim, with
        // its trailing '/') and the partial file name being completed.
        let (dir, file_prefix) = match partial.rfind('/') {
            Some(pos) => (&partial[..=pos], &partial[pos + 1..]),
            None => ("", partial),
        };
        let read_dir_path = if dir.is_empty() {
            std::path::PathBuf::from(".")
        } else {
            std::path::PathBuf::from(dir)
        };
        let Ok(entries) = std::fs::read_dir(&read_dir_path) else {
            return;
        };
        let mut candidates: Vec<(String, bool)> = Vec::new();
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with(file_prefix) {
                let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
                candidates.push((name, is_dir));
            }
        }
        match candidates.as_slice() {
            [] => {}
            [(name, is_dir)] => {
                let mut replacement = format!("@{dir}{name}");
                if *is_dir {
                    replacement.push('/');
                }
                self.replace_range(start, end, &replacement);
            }
            many => {
                let names: Vec<&str> = many.iter().map(|(n, _)| n.as_str()).collect();
                let common = common_prefix(&names);
                if common.len() > file_prefix.len() {
                    let replacement = format!("@{dir}{common}");
                    self.replace_range(start, end, &replacement);
                }
            }
        }
    }

    /// Replace the composer bytes in `start..end` (both char boundaries) with
    /// `text`, leaving the cursor at the end of the inserted text.
    fn replace_range(&mut self, start: usize, end: usize, text: &str) {
        self.composer.replace_range(start..end, text);
        self.cursor = start + text.len();
    }

    // --- Input-history recall ---------------------------------------------

    /// Push a submitted line into input history and end any navigation. Empty
    /// lines and consecutive duplicates are skipped. Called on submit and when
    /// seeding history from the persisted REPL file.
    pub fn push_history(&mut self, line: &str) {
        if !line.trim().is_empty() && self.input_history.last().map(String::as_str) != Some(line) {
            self.input_history.push(line.to_string());
        }
        self.history_index = None;
        self.history_draft.clear();
    }

    /// Seed input history from an iterator of prior lines (e.g. the persisted
    /// REPL history), applying the same skip rules as [`Self::push_history`].
    pub fn seed_input_history<I>(&mut self, lines: I)
    where
        I: IntoIterator<Item = String>,
    {
        for line in lines {
            self.push_history(&line);
        }
    }

    /// Recall the previous (older) input-history entry into the composer.
    ///
    /// Returns `true` if it consumed the key by recalling an entry, or `false`
    /// if the render loop should instead scroll the transcript up — namely when
    /// there is no history, the composer is multi-line and not yet navigating,
    /// or navigation is already at the oldest entry.
    pub fn history_prev(&mut self) -> bool {
        if self.input_history.is_empty() {
            return false;
        }
        match self.history_index {
            None => {
                // Only enter navigation from an empty or single-line draft.
                if self.composer.contains('\n') {
                    return false;
                }
                self.history_draft = std::mem::take(&mut self.composer);
                let idx = self.input_history.len() - 1;
                self.history_index = Some(idx);
                self.set_composer(self.input_history[idx].clone());
                true
            }
            Some(0) => false,
            Some(i) => {
                let idx = i - 1;
                self.history_index = Some(idx);
                self.set_composer(self.input_history[idx].clone());
                true
            }
        }
    }

    /// Step to the next (newer) input-history entry, or restore the stashed
    /// draft and exit navigation when stepping past the newest entry.
    ///
    /// Returns `true` if it consumed the key, or `false` (not navigating) if the
    /// render loop should instead scroll the transcript down.
    pub fn history_next(&mut self) -> bool {
        match self.history_index {
            None => false,
            Some(i) if i + 1 < self.input_history.len() => {
                let idx = i + 1;
                self.history_index = Some(idx);
                self.set_composer(self.input_history[idx].clone());
                true
            }
            Some(_) => {
                self.history_index = None;
                let draft = std::mem::take(&mut self.history_draft);
                self.set_composer(draft);
                true
            }
        }
    }

    /// Replace the whole composer with `text` and move the cursor to its end.
    fn set_composer(&mut self, text: String) {
        self.composer = text;
        self.cursor = self.composer.len();
    }

    /// End input-history navigation without changing the composer: the current
    /// buffer becomes a fresh draft. Called on any edit.
    fn exit_history_nav(&mut self) {
        self.history_index = None;
        self.history_draft.clear();
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

/// Longest common string prefix of `items` (assumed non-empty). Trimming with
/// `pop` keeps the result on a char boundary, so it is safe for multibyte text.
fn common_prefix<S: AsRef<str>>(items: &[S]) -> String {
    let mut prefix = items[0].as_ref().to_string();
    for item in &items[1..] {
        while !item.as_ref().starts_with(&prefix) {
            prefix.pop();
        }
    }
    prefix
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
    fn tool_output_strips_ansi_escape_sequences() {
        // Captured `/skills` output carries `colored`-crate ANSI escapes;
        // ratatui can't interpret them, so they must be stripped before the
        // rows land in the transcript.
        let mut s = AppState::new();
        s.apply(RenderEvent::ToolStarted {
            name: "skills".into(),
        });
        s.apply(RenderEvent::ToolFinished {
            name: "skills".into(),
            ok: true,
            output: "\x1b[1m\x1b[93mSkills:\x1b[0m".into(),
            elapsed_ms: 10,
        });
        let output_line = s
            .transcript()
            .iter()
            .find(|l| l.kind == LineKind::ToolOutput)
            .expect("expected a ToolOutput row");
        assert_eq!(output_line.text, "Skills:");
        assert!(
            !output_line.text.contains('\x1b'),
            "no ANSI escape bytes must survive into the transcript"
        );
    }

    #[test]
    fn diff_with_ansi_still_detected_as_tool_diff_after_stripping() {
        // A diff wrapped in ANSI color codes must still be recognized as a diff
        // once the escapes are stripped, and each cleaned row tagged ToolDiff.
        let mut s = AppState::new();
        s.apply(RenderEvent::ToolStarted {
            name: "apply".into(),
        });
        s.apply(RenderEvent::ToolFinished {
            name: "apply".into(),
            ok: true,
            output: "\x1b[36m@@ -1 +1 @@\x1b[0m\n\x1b[31m-old\x1b[0m\n\x1b[32m+new\x1b[0m".into(),
            elapsed_ms: 10,
        });
        let diff_rows: Vec<&TranscriptLine> = s
            .transcript()
            .iter()
            .filter(|l| l.kind == LineKind::ToolDiff)
            .collect();
        assert_eq!(
            diff_rows.len(),
            3,
            "all three cleaned diff rows are ToolDiff"
        );
        assert_eq!(diff_rows[0].text, "@@ -1 +1 @@");
        assert!(
            diff_rows.iter().all(|l| !l.text.contains('\x1b')),
            "diff rows must be ANSI-free"
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

    // --- Tab completion ---------------------------------------------------

    /// Type each char of `text` into the composer, leaving the cursor at end.
    fn typed(text: &str) -> AppState {
        let mut s = AppState::new();
        for c in text.chars() {
            s.edit(EditAction::InsertChar(c));
        }
        s
    }

    #[test]
    fn complete_unique_slash_prefix_appends_space() {
        let mut s = typed("/hel");
        s.complete();
        assert_eq!(s.composer(), "/help ");
        assert_eq!(s.cursor(), s.composer().len());
    }

    #[test]
    fn complete_extends_common_prefix_for_ambiguous_slash() {
        // `/re` matches several commands; completion extends to their longest
        // common prefix without picking one, and does not append a space.
        let matches = crate::commands::prefix_matches("/re");
        assert!(
            matches.len() > 1,
            "test premise: /re must be ambiguous, got {matches:?}"
        );
        let common = common_prefix(&matches);
        let mut s = typed("/re");
        s.complete();
        assert_eq!(s.composer(), common);
        assert!(common.starts_with("/re"));
        assert!(!s.composer().ends_with(' '));
    }

    #[test]
    fn complete_no_match_is_noop() {
        let mut s = typed("/zzzznotacommand");
        s.complete();
        assert_eq!(s.composer(), "/zzzznotacommand");
    }

    #[test]
    fn complete_mid_line_slash_does_not_double_space() {
        // Cursor sits right after "/hel" (before the existing space); completing
        // must not insert a second space ahead of the trailing argument.
        let mut s = typed("/hel foo");
        for _ in 0..4 {
            s.edit(EditAction::MoveLeft);
        }
        s.complete();
        assert_eq!(s.composer(), "/help foo");
    }

    #[test]
    fn complete_file_mention_against_temp_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("hello.txt"), b"hi").unwrap();
        // Use an absolute path in the token so the test does not depend on the
        // process-global current directory.
        let base = dir.path().to_string_lossy().into_owned();
        let mut s = typed(&format!("@{base}/hel"));
        s.complete();
        assert_eq!(s.composer(), format!("@{base}/hello.txt"));
        assert_eq!(s.cursor(), s.composer().len());
    }

    #[test]
    fn complete_file_mention_appends_slash_for_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();
        let base = dir.path().to_string_lossy().into_owned();
        let mut s = typed(&format!("@{base}/sub"));
        s.complete();
        assert_eq!(s.composer(), format!("@{base}/subdir/"));
    }

    #[test]
    fn complete_subcommand_unique_prefix_appends_space() {
        // `/mcp e` uniquely matches `enable`.
        let mut s = typed("/mcp e");
        s.complete();
        assert_eq!(s.composer(), "/mcp enable ");
        assert_eq!(s.cursor(), s.composer().len());
    }

    #[test]
    fn complete_subcommand_empty_partial_inserts_default() {
        // `/skills ` (trailing space, empty subcommand) inserts the default.
        let mut s = typed("/skills ");
        s.complete();
        assert_eq!(s.composer(), "/skills list ");
        assert_eq!(s.cursor(), s.composer().len());
    }

    #[test]
    fn complete_subcommand_ambiguous_extends_common_prefix() {
        // `/mcp r` matches both `remove` and `reload` → extend to `re`, no
        // trailing space and no arbitrary pick.
        let mut s = typed("/mcp r");
        s.complete();
        assert_eq!(s.composer(), "/mcp re");
        assert!(!s.composer().ends_with(' '));
    }

    #[test]
    fn complete_subcommand_unmanaged_command_is_unaffected() {
        // `/help` carries no subcommand vocabulary; a trailing-space Tab must
        // not synthesize a subcommand (and the `@file` path is a no-op here).
        let mut s = typed("/help ");
        s.complete();
        assert_eq!(s.composer(), "/help ");
    }

    #[test]
    fn complete_subcommand_no_match_is_noop() {
        // `/mcp zz` matches no subcommand → composer unchanged.
        let mut s = typed("/mcp zz");
        s.complete();
        assert_eq!(s.composer(), "/mcp zz");
    }

    #[test]
    fn complete_subcommand_does_not_touch_second_argument() {
        // Cursor in the SECOND arg word (`foo`) → subcommand completion does
        // not fire, leaving the line untouched.
        let mut s = typed("/mcp enable foo");
        s.complete();
        assert_eq!(s.composer(), "/mcp enable foo");
    }

    // --- Input-history recall ---------------------------------------------

    #[test]
    fn history_prev_recalls_newest_first_then_walks_back_to_empty() {
        let mut s = AppState::new();
        s.push_history("first");
        s.push_history("second");
        // Not navigating yet: Up recalls the newest entry.
        assert!(s.history_prev());
        assert_eq!(s.composer(), "second");
        assert_eq!(s.cursor(), s.composer().len());
        // Up again steps to the older entry.
        assert!(s.history_prev());
        assert_eq!(s.composer(), "first");
        // Already at the oldest: Up returns false (loop should scroll).
        assert!(!s.history_prev());
        assert_eq!(s.composer(), "first");
        // Down steps back to the newer entry.
        assert!(s.history_next());
        assert_eq!(s.composer(), "second");
        // Down past the newest restores the (empty) draft and exits nav.
        assert!(s.history_next());
        assert_eq!(s.composer(), "");
        // No longer navigating: Down returns false.
        assert!(!s.history_next());
    }

    #[test]
    fn history_prev_stashes_and_restores_a_live_draft() {
        let mut s = AppState::new();
        s.push_history("recalled");
        for c in "draft".chars() {
            s.edit(EditAction::InsertChar(c));
        }
        assert!(s.history_prev());
        assert_eq!(s.composer(), "recalled");
        // Stepping past the newest restores the stashed draft.
        assert!(s.history_next());
        assert_eq!(s.composer(), "draft");
    }

    #[test]
    fn editing_resets_history_navigation() {
        let mut s = AppState::new();
        s.push_history("alpha");
        s.push_history("beta");
        assert!(s.history_prev());
        assert_eq!(s.composer(), "beta");
        // An edit exits navigation: the buffer is now a fresh draft.
        s.edit(EditAction::InsertChar('!'));
        assert_eq!(s.composer(), "beta!");
        // Down no longer navigates history (returns false → scroll).
        assert!(!s.history_next());
        // Up starts a new navigation from the newest entry, stashing "beta!".
        assert!(s.history_prev());
        assert_eq!(s.composer(), "beta");
        assert!(s.history_next());
        assert_eq!(s.composer(), "beta!");
    }

    #[test]
    fn history_prev_false_when_history_empty() {
        let mut s = AppState::new();
        assert!(!s.history_prev());
        assert!(!s.history_next());
    }

    #[test]
    fn push_history_skips_empties_and_consecutive_duplicates() {
        let mut s = AppState::new();
        s.push_history("x");
        s.push_history("x"); // consecutive dup skipped
        s.push_history("   "); // empty (whitespace) skipped
        s.push_history("y");
        // Newest-first recall should see exactly y, x.
        assert!(s.history_prev());
        assert_eq!(s.composer(), "y");
        assert!(s.history_prev());
        assert_eq!(s.composer(), "x");
        assert!(!s.history_prev());
    }

    #[test]
    fn history_prev_does_not_recall_into_a_multiline_draft() {
        let mut s = AppState::new();
        s.push_history("cmd");
        for c in "line".chars() {
            s.edit(EditAction::InsertChar(c));
        }
        s.edit(EditAction::Newline);
        // Multi-line draft: Up must not recall (returns false → scroll).
        assert!(!s.history_prev());
        assert_eq!(s.composer(), "line\n");
    }

    #[test]
    fn seed_input_history_applies_skip_rules() {
        let mut s = AppState::new();
        s.seed_input_history(["a", "a", "", "b"].into_iter().map(str::to_string));
        assert!(s.history_prev());
        assert_eq!(s.composer(), "b");
        assert!(s.history_prev());
        assert_eq!(s.composer(), "a");
        assert!(!s.history_prev());
    }
}
