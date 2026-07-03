//! The single ratatui-facing color palette for the TUI.
//!
//! Every color the transcript renderers ([`super::widgets`],
//! [`super::markdown`], [`super::diff`], [`super::highlight`]) paint is sourced
//! from one [`Theme`] value instead of a scattering of hardcoded
//! [`ratatui::style::Color`] constants. This lets `--tui` honor the persisted
//! `/theme` choice (`auto` / `light` / `dark`, see [`crate::themes`]).
//!
//! Semantic fields map one-to-one to the colors that used to be hardcoded:
//!   * role headers, status/footer, tool-call blocks,
//!   * unified-diff add/remove/hunk/file-header rows,
//!   * fenced-code borders/labels, inline code, headings,
//!   * and the syntax-highlighter's keyword/string/comment/number tokens.
//!
//! Elements that were only ever *dim* or *bold* (no explicit foreground —
//! footers, tool output, code borders/labels, diff file headers, headings, the
//! thinking spinner) keep their [`Modifier`](ratatui::style::Modifier) at the
//! call site. Their palette color defaults to [`Color::Reset`] so applying it
//! renders identically to the previous "no explicit fg" look while still
//! flowing through the one palette (a future theme could recolor them).
//!
//! The `auto` preset reproduces today's dark-ish appearance exactly, so users
//! who never set `/theme` see no change.

use ratatui::style::Color;

/// A complete ratatui color palette for the TUI. Every renderer reads its
/// colors from here. Cheap to copy (`Copy`) so it can live on `AppState` and
/// be threaded by value into the pure sub-renderers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Theme {
    // --- role headers -----------------------------------------------------
    /// The `You` user role header.
    pub(crate) role_user: Color,
    /// The `Cubi` assistant role header.
    pub(crate) role_assistant: Color,

    // --- status / footer --------------------------------------------------
    /// Status lines in the transcript.
    pub(crate) status: Color,
    /// Dim footer / composer-corner metadata (usage, path, model, spinner).
    pub(crate) usage_dim: Color,
    /// The thinking-indicator spinner text in the composer corner.
    pub(crate) spinner: Color,

    // --- tool-call block --------------------------------------------------
    /// The `⚙ tool` header row.
    pub(crate) tool_header: Color,
    /// Dim `│ `-indented tool output rows.
    pub(crate) tool_output_dim: Color,
    /// The `✓ tool` success status row.
    pub(crate) tool_ok: Color,
    /// The `✗ tool` error status row.
    pub(crate) tool_err: Color,

    // --- unified diff -----------------------------------------------------
    /// `+` addition rows.
    pub(crate) diff_add: Color,
    /// `-` deletion rows.
    pub(crate) diff_del: Color,
    /// `@@ … @@` hunk-header rows.
    pub(crate) diff_hunk: Color,
    /// `+++ ` / `--- ` file-header rows (dim bold, not add/remove colors).
    pub(crate) diff_file_header: Color,

    // --- markdown ---------------------------------------------------------
    /// The dim `│ ` left border on fenced code rows.
    pub(crate) code_border: Color,
    /// The dim `lang ▏` fenced-code language label row.
    pub(crate) code_label: Color,
    /// Inline `` `code` `` spans.
    pub(crate) inline_code: Color,
    /// ATX heading text (bold; color reset by default).
    pub(crate) heading: Color,

    // --- syntax highlighter ----------------------------------------------
    /// Language keywords.
    pub(crate) syntax_keyword: Color,
    /// String / char literals.
    pub(crate) syntax_string: Color,
    /// Comments.
    pub(crate) syntax_comment: Color,
    /// Numeric literals.
    pub(crate) syntax_number: Color,
}

impl Theme {
    /// The `auto` preset: reproduces today's hardcoded dark-ish appearance so
    /// nothing changes for users who never set `/theme`.
    pub(crate) const AUTO: Theme = Theme {
        role_user: Color::Green,
        role_assistant: Color::Cyan,
        status: Color::Cyan,
        usage_dim: Color::Reset,
        spinner: Color::Reset,
        tool_header: Color::Blue,
        tool_output_dim: Color::Reset,
        tool_ok: Color::Green,
        tool_err: Color::Red,
        diff_add: Color::Green,
        diff_del: Color::Red,
        diff_hunk: Color::Cyan,
        diff_file_header: Color::Reset,
        code_border: Color::Reset,
        code_label: Color::Reset,
        inline_code: Color::Cyan,
        heading: Color::Reset,
        syntax_keyword: Color::Magenta,
        syntax_string: Color::Green,
        syntax_comment: Color::DarkGray,
        syntax_number: Color::Yellow,
    };

    /// The `dark` preset: tuned for dark terminals. Mirrors the `auto`
    /// (dark-ish) look, with a colored spinner.
    pub(crate) const DARK: Theme = Theme {
        role_user: Color::LightGreen,
        role_assistant: Color::LightCyan,
        status: Color::Cyan,
        usage_dim: Color::Reset,
        spinner: Color::Cyan,
        tool_header: Color::LightBlue,
        tool_output_dim: Color::Reset,
        tool_ok: Color::LightGreen,
        tool_err: Color::LightRed,
        diff_add: Color::LightGreen,
        diff_del: Color::LightRed,
        diff_hunk: Color::LightCyan,
        diff_file_header: Color::Reset,
        code_border: Color::Reset,
        code_label: Color::Reset,
        inline_code: Color::LightCyan,
        heading: Color::Reset,
        syntax_keyword: Color::LightMagenta,
        syntax_string: Color::LightGreen,
        syntax_comment: Color::DarkGray,
        syntax_number: Color::LightYellow,
    };

    /// The `light` preset: darker, higher-contrast colors that stay legible on
    /// a light terminal background (avoids the washed-out bright variants).
    pub(crate) const LIGHT: Theme = Theme {
        role_user: Color::Green,
        role_assistant: Color::Blue,
        status: Color::Blue,
        usage_dim: Color::Reset,
        spinner: Color::Blue,
        tool_header: Color::Blue,
        tool_output_dim: Color::Reset,
        tool_ok: Color::Green,
        tool_err: Color::Red,
        diff_add: Color::Green,
        diff_del: Color::Red,
        diff_hunk: Color::Blue,
        diff_file_header: Color::Reset,
        code_border: Color::Reset,
        code_label: Color::Reset,
        inline_code: Color::Blue,
        heading: Color::Reset,
        syntax_keyword: Color::Blue,
        syntax_string: Color::Green,
        syntax_comment: Color::DarkGray,
        syntax_number: Color::Magenta,
    };

    /// Resolve a user-facing theme name to a preset. Case-insensitive;
    /// `"light"` / `"dark"` select their preset and anything else (including
    /// `"auto"` and unknown/corrupt values) falls back to `auto`. Mirrors
    /// [`crate::themes::palette_for`] so the TUI and the `colored`-based CLI
    /// agree on the name mapping.
    pub(crate) fn from_name(name: &str) -> Theme {
        match name.trim().to_ascii_lowercase().as_str() {
            "light" => Theme::LIGHT,
            "dark" => Theme::DARK,
            _ => Theme::AUTO,
        }
    }
}

impl Default for Theme {
    /// The default theme is `auto`, preserving today's appearance.
    fn default() -> Self {
        Theme::AUTO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_name_is_case_insensitive_and_maps_presets() {
        assert_eq!(Theme::from_name("light"), Theme::LIGHT);
        assert_eq!(Theme::from_name("LIGHT"), Theme::LIGHT);
        assert_eq!(Theme::from_name("  Dark  "), Theme::DARK);
        assert_eq!(Theme::from_name("dark"), Theme::DARK);
    }

    #[test]
    fn unknown_and_auto_fall_back_to_auto() {
        assert_eq!(Theme::from_name("auto"), Theme::AUTO);
        assert_eq!(Theme::from_name("nonsense"), Theme::AUTO);
        assert_eq!(Theme::from_name(""), Theme::AUTO);
        assert_eq!(Theme::default(), Theme::AUTO);
    }

    #[test]
    fn light_and_dark_differ_in_at_least_one_field() {
        // The whole point of theming: at least one semantic color must change
        // between presets so `/theme` is observable.
        assert_ne!(Theme::LIGHT.role_assistant, Theme::DARK.role_assistant);
        assert_ne!(Theme::LIGHT.syntax_keyword, Theme::DARK.syntax_keyword);
    }

    #[test]
    fn auto_preserves_todays_hardcoded_colors() {
        // Guardrail: the `auto` preset must equal the colors the TUI used
        // before theming, so an unset `/theme` looks identical.
        let t = Theme::AUTO;
        assert_eq!(t.role_user, Color::Green);
        assert_eq!(t.role_assistant, Color::Cyan);
        assert_eq!(t.status, Color::Cyan);
        assert_eq!(t.tool_header, Color::Blue);
        assert_eq!(t.tool_ok, Color::Green);
        assert_eq!(t.tool_err, Color::Red);
        assert_eq!(t.diff_add, Color::Green);
        assert_eq!(t.diff_del, Color::Red);
        assert_eq!(t.diff_hunk, Color::Cyan);
        assert_eq!(t.inline_code, Color::Cyan);
        assert_eq!(t.syntax_keyword, Color::Magenta);
        assert_eq!(t.syntax_string, Color::Green);
        assert_eq!(t.syntax_comment, Color::DarkGray);
        assert_eq!(t.syntax_number, Color::Yellow);
    }
}
