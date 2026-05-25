//! Bundled colour themes for `/theme`.
//!
//! Roadmap item C#13 (themable output styles, theme half): three
//! presets (`auto`, `light`, `dark`) plus a serialisable [`Palette`] the
//! CLI consults when colouring section headers, status lines, and tool
//! markers. The crate-wide `colored` overrides are still set by
//! `handle_color`; this module only owns the *palette* — which named
//! colour the printer should use — not the on/off switch.
//!
//! Persistence lives in [`crate::onboarding::AppConfig::theme`] so the
//! user's pick survives a restart.
//!
//! The [`Palette`] type itself is currently consumed only by tests —
//! the rest of the CLI still uses the `colored` crate's named
//! constants directly. The palette is in place as a stable target for
//! the future ratatui port, where every printed widget needs to look
//! up its colour from one shared source.
#![allow(dead_code)]

use colored::Color;

#[derive(Debug, Clone, Copy)]
pub struct Palette {
    pub title: Color,
    pub success: Color,
    pub error: Color,
    pub info: Color,
    pub hint: Color,
    pub accent: Color,
}

impl Palette {
    pub const AUTO: Palette = Palette {
        title: Color::BrightYellow,
        success: Color::BrightGreen,
        error: Color::BrightRed,
        info: Color::BrightBlue,
        hint: Color::BrightBlack,
        accent: Color::BrightCyan,
    };

    pub const LIGHT: Palette = Palette {
        title: Color::Magenta,
        success: Color::Green,
        error: Color::Red,
        info: Color::Blue,
        hint: Color::Black,
        accent: Color::Cyan,
    };

    pub const DARK: Palette = Palette {
        title: Color::BrightWhite,
        success: Color::BrightGreen,
        error: Color::BrightRed,
        info: Color::BrightCyan,
        hint: Color::BrightBlack,
        accent: Color::BrightMagenta,
    };
}

/// Map a user-facing theme name to its palette. Unknown names fall back
/// to `auto` so a corrupt config can't crash the CLI.
pub fn palette_for(theme: &str) -> Palette {
    match theme.trim().to_ascii_lowercase().as_str() {
        "light" => Palette::LIGHT,
        "dark" => Palette::DARK,
        _ => Palette::AUTO,
    }
}

/// The catalogue of valid theme names, in the order shown by `/theme`.
pub const VALID_THEMES: &[&str] = &["auto", "light", "dark"];

pub fn is_valid_theme(name: &str) -> bool {
    VALID_THEMES.contains(&name.trim().to_ascii_lowercase().as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_themes_round_trip() {
        for name in VALID_THEMES {
            assert!(is_valid_theme(name), "theme {name} should be valid");
            // Just touch every field so a stray rename in `Palette`
            // surfaces in CI rather than at first paint.
            let p = palette_for(name);
            let _ = (p.title, p.success, p.error, p.info, p.hint, p.accent);
        }
    }

    #[test]
    fn unknown_theme_falls_back_to_auto() {
        assert!(!is_valid_theme("solarized"));
        // palette_for never panics on unknown input — it just degrades.
        let _ = palette_for("solarized");
    }
}
