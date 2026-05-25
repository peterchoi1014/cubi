//! Per-session output-style presets (`/output-style`).
//!
//! Roadmap item C#13 (output-style half): the model is steered toward a
//! concise / explanatory / markdown reply shape by prepending a small
//! system prompt to every request. This module owns the
//! preset → prompt-fragment mapping so `cli.rs` doesn't have to.
//!
//! Persistence lives in [`crate::onboarding::AppConfig::output_style`].

/// Recognised preset names, in the order shown by `/output-style`.
pub const VALID_STYLES: &[&str] = &["concise", "markdown", "explanatory"];

/// Default preset used when the user has not set one.
pub const DEFAULT_STYLE: &str = "markdown";

pub fn is_valid_style(name: &str) -> bool {
    VALID_STYLES.contains(&name.trim().to_ascii_lowercase().as_str())
}

/// Returns the system-prompt fragment for the given preset. Unknown
/// names fall back to [`DEFAULT_STYLE`] so a corrupt config never
/// silently drops the steering.
pub fn system_prompt_for(style: &str) -> &'static str {
    match style.trim().to_ascii_lowercase().as_str() {
        "concise" => {
            "Output style: concise. Reply in <=3 short sentences whenever possible. \
             Skip preamble and avoid lists unless the user explicitly asks for one."
        }
        "explanatory" => {
            "Output style: explanatory. Walk the user through your reasoning step by \
             step, including the why behind any tool call, and call out trade-offs you \
             considered."
        }
        _ => {
            "Output style: markdown. Use headings, bullet lists, and fenced code blocks \
             where they aid clarity. Keep prose tight."
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_styles_have_prompts() {
        for s in VALID_STYLES {
            assert!(is_valid_style(s));
            assert!(!system_prompt_for(s).is_empty());
        }
    }

    #[test]
    fn unknown_style_falls_back_to_default_prompt() {
        assert!(!is_valid_style("rude"));
        // Unknown input must not panic and must produce a real prompt.
        let p = system_prompt_for("rude");
        assert!(p.contains("markdown"));
    }
}
