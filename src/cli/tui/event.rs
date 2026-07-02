//! Key-event mapping for the Phase 2 TUI render task.
//!
//! [`map_key`] is a pure function from a `crossterm` [`KeyEvent`] to a
//! high-level [`Action`], so it is trivially unit-testable without a terminal.
//! The render task performs the side effects; this module only classifies.

use super::app::EditAction;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// A high-level interpretation of one key press, consumed by the render task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Action {
    /// A local composer edit (insert/backspace/cursor move/newline).
    Edit(EditAction),
    /// Submit the current composer contents as a user message.
    Submit,
    /// Cooperative cancel of the in-flight turn (Ctrl-C). Under raw mode this
    /// replaces the SIGINT the terminal would otherwise deliver.
    Cancel,
    /// Quit the TUI session (Ctrl-D).
    Quit,
    /// Scroll the transcript toward older content.
    ScrollUp,
    /// Scroll the transcript toward newer content.
    ScrollDown,
    /// No actionable interpretation (unhandled key).
    None,
}

/// Classify a key event. Ctrl-C / Ctrl-D take precedence over the character
/// they carry. `Alt+Enter` inserts a literal newline (multi-line composer);
/// a bare `Enter` submits.
pub(super) fn map_key(ev: KeyEvent) -> Action {
    let ctrl = ev.modifiers.contains(KeyModifiers::CONTROL);
    let alt = ev.modifiers.contains(KeyModifiers::ALT);
    match ev.code {
        KeyCode::Char('c') if ctrl => Action::Cancel,
        KeyCode::Char('d') if ctrl => Action::Quit,
        // Ignore other control-chords so they don't land as literal text.
        KeyCode::Char(_) if ctrl => Action::None,
        KeyCode::Char(c) => Action::Edit(EditAction::InsertChar(c)),
        KeyCode::Enter if alt => Action::Edit(EditAction::Newline),
        KeyCode::Enter => Action::Submit,
        KeyCode::Backspace => Action::Edit(EditAction::Backspace),
        KeyCode::Left => Action::Edit(EditAction::MoveLeft),
        KeyCode::Right => Action::Edit(EditAction::MoveRight),
        // Up/Down scroll the transcript. Under alternate-scroll mode the mouse
        // wheel is delivered as Up/Down key presses, so this also drives
        // wheel scrolling without capturing the mouse.
        KeyCode::Up => Action::ScrollUp,
        KeyCode::Down => Action::ScrollDown,
        KeyCode::PageUp => Action::ScrollUp,
        KeyCode::PageDown => Action::ScrollDown,
        _ => Action::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn key_mod(code: KeyCode, mods: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, mods)
    }

    #[test]
    fn plain_char_inserts() {
        assert_eq!(
            map_key(key(KeyCode::Char('a'))),
            Action::Edit(EditAction::InsertChar('a'))
        );
    }

    #[test]
    fn enter_submits_and_alt_enter_newlines() {
        assert_eq!(map_key(key(KeyCode::Enter)), Action::Submit);
        assert_eq!(
            map_key(key_mod(KeyCode::Enter, KeyModifiers::ALT)),
            Action::Edit(EditAction::Newline)
        );
    }

    #[test]
    fn backspace_and_arrows_map_to_edits() {
        assert_eq!(
            map_key(key(KeyCode::Backspace)),
            Action::Edit(EditAction::Backspace)
        );
        assert_eq!(
            map_key(key(KeyCode::Left)),
            Action::Edit(EditAction::MoveLeft)
        );
        assert_eq!(
            map_key(key(KeyCode::Right)),
            Action::Edit(EditAction::MoveRight)
        );
    }

    #[test]
    fn ctrl_c_cancels_and_ctrl_d_quits() {
        assert_eq!(
            map_key(key_mod(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            Action::Cancel
        );
        assert_eq!(
            map_key(key_mod(KeyCode::Char('d'), KeyModifiers::CONTROL)),
            Action::Quit
        );
    }

    #[test]
    fn other_control_chords_and_unknown_keys_are_none() {
        assert_eq!(
            map_key(key_mod(KeyCode::Char('a'), KeyModifiers::CONTROL)),
            Action::None
        );
        assert_eq!(map_key(key(KeyCode::Esc)), Action::None);
    }

    #[test]
    fn arrow_and_page_keys_scroll() {
        assert_eq!(map_key(key(KeyCode::Up)), Action::ScrollUp);
        assert_eq!(map_key(key(KeyCode::Down)), Action::ScrollDown);
        assert_eq!(map_key(key(KeyCode::PageUp)), Action::ScrollUp);
        assert_eq!(map_key(key(KeyCode::PageDown)), Action::ScrollDown);
    }
}
