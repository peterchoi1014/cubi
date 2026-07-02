//! Typed, provider-agnostic status snapshot rendered as a single status line.
use crate::style::CubiStyle;
use std::path::{Path, PathBuf};

/// Immutable snapshot of everything the status line can show. Built by the
/// caller (member B) from live cubi state; rendered by [`StatusState::render`].
#[derive(Debug, Clone)]
pub struct StatusState {
    /// Current model id, e.g. "qwen3:8b" (from `AIExecutor::get_model()`).
    pub model: String,
    /// Estimated tokens currently in the conversation window
    /// (`llm::estimate_conversation_tokens`). `None` if unknown.
    pub context_used: Option<usize>,
    /// Logical context window for the model (`llm::context_window_for_model`).
    /// `None` when the model family is unknown.
    pub context_window: Option<usize>,
    /// Current working directory (`std::env::current_dir()`).
    pub cwd: PathBuf,
    /// Cumulative provider-reported prompt tokens for this run (session_stats).
    pub prompt_tokens: u64,
    /// Cumulative provider-reported completion tokens for this run.
    pub completion_tokens: u64,
    /// Pre-formatted cost string from `pricing::format_cost(...)`:
    /// "$0.0123", "$0.00 (local)", or "—" (unknown). Member B computes it so
    /// this module has no dependency on pricing internals.
    pub cost: String,
}

/// Column separator used between status-line segments.
const SEP: &str = " · ";

impl StatusState {
    /// Render one status line, collapsing fields to fit `width` columns.
    /// When `color` is false, emit no ANSI escapes at all (plain text).
    /// Never emits a trailing newline; never wraps to a second line.
    ///
    /// The `~`-abbreviation of the home directory is the only environment
    /// touch: it reads `$HOME` once purely to shorten the path for display.
    /// All formatting logic lives in the pure [`StatusState::render_with_home`]
    /// helper, which takes the home directory as an explicit argument.
    pub fn render(&self, width: usize, color: bool) -> String {
        let home = std::env::var_os("HOME").map(PathBuf::from);
        self.render_with_home(width, color, home.as_deref())
    }

    /// Pure formatter: identical to [`StatusState::render`] but with the home
    /// directory injected explicitly (no environment or filesystem reads), so
    /// the whole pipeline is deterministic and unit-testable.
    fn render_with_home(&self, width: usize, color: bool, home: Option<&Path>) -> String {
        // --- Painters (no-ops when `color` is false → zero ANSI escapes) ---
        let paint_model = |s: &str| -> String {
            if color {
                s.bright_cyan().to_string()
            } else {
                s.to_string()
            }
        };
        let paint_dim = |s: &str| -> String {
            if color {
                s.bright_black().to_string()
            } else {
                s.to_string()
            }
        };
        let paint_pct = |p: usize| -> String {
            let t = format!("{p}%");
            if !color {
                t
            } else if p >= 95 {
                t.bright_red().to_string()
            } else if p >= 80 {
                t.bright_yellow().to_string()
            } else {
                // Normal (< 80%): no color.
                t
            }
        };
        let sep = paint_dim(SEP);

        // --- Precomputed, provider-agnostic building blocks ---
        let pct = match (self.context_used, self.context_window) {
            (Some(u), Some(w)) if w > 0 => Some((100.0 * u as f64 / w as f64).round() as usize),
            _ => None,
        };

        // Full ctx segment: `ctx {used}/{window} {pct}%`, degrading to
        // `ctx {used}` when only the window is unknown, and dropping entirely
        // when the used count is unknown. Returns (plain, painted).
        let ctx_full: Option<(String, String)> = match (self.context_used, self.context_window, pct)
        {
            (Some(u), Some(w), Some(p)) => Some((
                format!("ctx {}/{} {p}%", abbrev(u as u64), abbrev(w as u64)),
                format!(
                    "ctx {}/{} {}",
                    abbrev(u as u64),
                    abbrev(w as u64),
                    paint_pct(p)
                ),
            )),
            (Some(u), None, _) => {
                let s = format!("ctx {}", abbrev(u as u64));
                Some((s.clone(), s))
            }
            _ => None,
        };

        // Compact ctx segment: `ctx {pct}%` (or `ctx {used}` when the window
        // is unknown).
        let ctx_compact: Option<(String, String)> =
            match (self.context_used, self.context_window, pct) {
                (Some(_), Some(_), Some(p)) => {
                    Some((format!("ctx {p}%"), format!("ctx {}", paint_pct(p))))
                }
                (Some(u), None, _) => {
                    let s = format!("ctx {}", abbrev(u as u64));
                    Some((s.clone(), s))
                }
                _ => None,
            };

        // Bare pct segment: `{pct}%` (or bare `{used}` when window unknown).
        let pct_bare: Option<(String, String)> = match (self.context_used, self.context_window, pct)
        {
            (Some(_), Some(_), Some(p)) => Some((format!("{p}%"), paint_pct(p))),
            (Some(u), None, _) => {
                let s = abbrev(u as u64);
                Some((s.clone(), s))
            }
            _ => None,
        };

        let path_full = abbrev_home(&self.cwd, home);
        let path_base = self
            .cwd
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path_full.clone());

        let tokens_full = format!(
            "tok {} in/{} out",
            abbrev(self.prompt_tokens),
            abbrev(self.completion_tokens)
        );
        let tokens_compact = format!(
            "tok {}/{}",
            abbrev_compact(self.prompt_tokens),
            abbrev_compact(self.completion_tokens)
        );

        let cost_full = self.cost.clone();
        let cost_compact = compact_cost(&self.cost);

        // --- Width-collapse tiers (always keep model + ctx% + path basename) ---
        let parts: Vec<String> = if width >= 100 {
            // Full row.
            let mut parts = vec![paint_model(&self.model)];
            if let Some((_, painted)) = &ctx_full {
                parts.push(painted.clone());
            }
            parts.push(path_full.clone());
            parts.push(tokens_full.clone());
            parts.push(paint_dim(&cost_full));
            parts
        } else if width >= 80 {
            // Keep all, but middle-truncate the path to fit and compact tokens.
            // Compute the width budget from the *plain* (escape-free) lengths so
            // ANSI escapes never count toward columns.
            let mut plain_others: Vec<usize> = vec![self.model.chars().count()];
            if let Some((plain, _)) = &ctx_full {
                plain_others.push(plain.chars().count());
            }
            plain_others.push(tokens_compact.chars().count());
            plain_others.push(cost_full.chars().count());
            let n = plain_others.len() + 1; // + path segment
            let others_len: usize = plain_others.iter().sum();
            let sep_len = SEP.chars().count() * n.saturating_sub(1);
            let remaining = width.saturating_sub(others_len + sep_len);
            let path_disp = middle_truncate(&path_full, remaining.max(3));

            let mut parts = vec![paint_model(&self.model)];
            if let Some((_, painted)) = &ctx_full {
                parts.push(painted.clone());
            }
            parts.push(path_disp);
            parts.push(tokens_compact.clone());
            parts.push(paint_dim(&cost_full));
            parts
        } else if width >= 60 {
            // model · ctx {pct}% · path basename · compact cost.
            let mut parts = vec![paint_model(&self.model)];
            if let Some((_, painted)) = &ctx_compact {
                parts.push(painted.clone());
            }
            parts.push(path_base.clone());
            parts.push(paint_dim(&cost_compact));
            parts
        } else if width >= 40 {
            // model · ctx {pct}% · path basename · cost.
            let mut parts = vec![paint_model(&self.model)];
            if let Some((_, painted)) = &ctx_compact {
                parts.push(painted.clone());
            }
            parts.push(path_base.clone());
            parts.push(paint_dim(&cost_full));
            parts
        } else if width >= 20 {
            // model · {pct}% · path basename.
            let mut parts = vec![paint_model(&self.model)];
            if let Some((_, painted)) = &pct_bare {
                parts.push(painted.clone());
            }
            parts.push(path_base.clone());
            parts
        } else {
            // model only.
            vec![paint_model(&self.model)]
        };

        parts.join(&sep)
    }
}

/// One-decimal "k" abbreviation for numbers >= 1000, else the plain number.
/// e.g. `999` → `"999"`, `1000` → `"1.0k"`, `41300` → `"41.3k"`.
fn abbrev(n: u64) -> String {
    if n >= 1000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}

/// Whole-number "k" abbreviation used in the compact token layout.
/// e.g. `41300` → `"41k"`, `7800` → `"8k"`, `999` → `"999"`.
fn abbrev_compact(n: u64) -> String {
    if n >= 1000 {
        format!("{}k", (n as f64 / 1000.0).round() as u64)
    } else {
        n.to_string()
    }
}

/// Strip a trailing parenthetical (e.g. the `" (local)"` marker) so the cost
/// fits tighter columns: `"$0.00 (local)"` → `"$0.00"`; other strings pass
/// through unchanged.
fn compact_cost(s: &str) -> String {
    match s.split_once(" (") {
        Some((head, _)) => head.to_string(),
        None => s.to_string(),
    }
}

/// Replace a `$HOME` prefix in `path` with `~`. Pure: `home` is supplied by
/// the caller rather than read from the environment.
fn abbrev_home(path: &Path, home: Option<&Path>) -> String {
    if let Some(h) = home {
        if path == h {
            return "~".to_string();
        }
        if let Ok(rest) = path.strip_prefix(h) {
            return format!("~/{}", rest.display());
        }
    }
    path.display().to_string()
}

/// Middle-truncate `s` to at most `max` characters, inserting `…` in the gap.
/// Truncation happens on character boundaries so multibyte paths stay valid.
fn middle_truncate(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        return s.to_string();
    }
    if max == 0 {
        return String::new();
    }
    if max == 1 {
        return "…".to_string();
    }
    let keep = max - 1;
    let head = keep / 2;
    let tail = keep - head;
    let head_s: String = chars[..head].iter().collect();
    let tail_s: String = chars[chars.len() - tail..].iter().collect();
    format!("{head_s}…{tail_s}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sample home directory used to exercise `~` abbreviation deterministically.
    const HOME: &str = "/home/dev";

    /// Builds the canonical snapshot behind the contract's example row:
    /// `qwen3:8b · ctx 8.2k/32.8k 25% · ~/repos/cubi · tok 41.3k in/7.8k out · $0.00 (local)`.
    fn sample() -> StatusState {
        StatusState {
            model: "qwen3:8b".to_string(),
            context_used: Some(8200),
            context_window: Some(32800),
            cwd: PathBuf::from(format!("{HOME}/repos/cubi")),
            prompt_tokens: 41300,
            completion_tokens: 7800,
            cost: "$0.00 (local)".to_string(),
        }
    }

    fn home() -> Option<&'static Path> {
        Some(Path::new(HOME))
    }

    #[test]
    fn wide_row_exact_string() {
        let got = sample().render_with_home(120, false, home());
        assert_eq!(
            got,
            "qwen3:8b · ctx 8.2k/32.8k 25% · ~/repos/cubi · tok 41.3k in/7.8k out · $0.00 (local)"
        );
    }

    #[test]
    fn tier_80_99_keeps_ctx_and_compacts_tokens() {
        // Short path (no truncation needed) so the layout is deterministic.
        let got = sample().render_with_home(90, false, home());
        assert_eq!(
            got,
            "qwen3:8b · ctx 8.2k/32.8k 25% · ~/repos/cubi · tok 41k/8k · $0.00 (local)"
        );
    }

    #[test]
    fn tier_80_99_middle_truncates_long_path() {
        let mut s = sample();
        s.cwd = PathBuf::from("/home/dev/a/very/deeply/nested/workspace/project/cubi");
        let got = s.render_with_home(80, false, home());
        // Model, ctx%, compact tokens, cost survive; path is middle-truncated.
        assert!(got.starts_with("qwen3:8b · ctx 8.2k/32.8k 25% · "));
        assert!(got.contains('…'), "expected ellipsis, got: {got}");
        assert!(got.ends_with(" · tok 41k/8k · $0.00 (local)"));
        assert!(got.chars().count() <= 80, "line exceeds width: {got}");
    }

    #[test]
    fn tier_60_79_exact_string_uses_compact_cost() {
        let got = sample().render_with_home(70, false, home());
        // Compact cost drops the " (local)" marker at this tier.
        assert_eq!(got, "qwen3:8b · ctx 25% · cubi · $0.00");
    }

    #[test]
    fn tier_40_59_exact_string_uses_full_cost() {
        let got = sample().render_with_home(50, false, home());
        assert_eq!(got, "qwen3:8b · ctx 25% · cubi · $0.00 (local)");
    }

    #[test]
    fn tier_20_39_exact_string_is_model_pct_basename() {
        let got = sample().render_with_home(30, false, home());
        assert_eq!(got, "qwen3:8b · 25% · cubi");
    }

    #[test]
    fn tier_below_20_is_model_only() {
        let got = sample().render_with_home(10, false, home());
        assert_eq!(got, "qwen3:8b");
    }

    #[test]
    fn color_false_emits_no_ansi_escapes() {
        for width in [120usize, 90, 70, 50, 30, 10] {
            let got = sample().render_with_home(width, false, home());
            assert!(
                !got.contains('\x1b'),
                "width {width} leaked an ANSI escape: {got:?}"
            );
        }
    }

    #[test]
    fn render_never_emits_trailing_newline() {
        for width in [120usize, 90, 70, 50, 30, 10] {
            let got = sample().render_with_home(width, false, home());
            assert!(!got.ends_with('\n'), "width {width} added a newline");
            assert!(!got.contains('\n'), "width {width} wrapped to two lines");
        }
    }

    #[test]
    fn unknown_context_drops_ctx_segment() {
        let mut s = sample();
        s.context_used = None; // used unknown → whole ctx segment dropped
        let got = s.render_with_home(120, false, home());
        assert_eq!(
            got,
            "qwen3:8b · ~/repos/cubi · tok 41.3k in/7.8k out · $0.00 (local)"
        );
        assert!(!got.contains("ctx"), "ctx should be dropped: {got}");
    }

    #[test]
    fn unknown_window_degrades_to_ctx_used_only() {
        let mut s = sample();
        s.context_window = None; // window unknown → show just `ctx {used}`
        let got = s.render_with_home(120, false, home());
        assert_eq!(
            got,
            "qwen3:8b · ctx 8.2k · ~/repos/cubi · tok 41.3k in/7.8k out · $0.00 (local)"
        );
        // And no percentage can be shown without a window.
        assert!(!got.contains('%'), "no pct without a window: {got}");
    }

    #[test]
    fn cost_variants_render_verbatim() {
        // Priced cost.
        let mut priced = sample();
        priced.cost = "$0.0123".to_string();
        assert!(
            priced
                .render_with_home(120, false, home())
                .ends_with("· $0.0123")
        );

        // Local cost.
        assert!(
            sample()
                .render_with_home(120, false, home())
                .ends_with("· $0.00 (local)")
        );

        // Unknown cost.
        let mut unknown = sample();
        unknown.cost = "—".to_string();
        assert!(
            unknown
                .render_with_home(120, false, home())
                .ends_with("· —")
        );
    }

    #[test]
    fn home_abbreviation() {
        // Exact home directory collapses to "~".
        assert_eq!(abbrev_home(Path::new(HOME), home()), "~");
        // Sub-path collapses the home prefix.
        assert_eq!(
            abbrev_home(Path::new("/home/dev/repos/cubi"), home()),
            "~/repos/cubi"
        );
        // A path outside home is left absolute.
        assert_eq!(abbrev_home(Path::new("/tmp/work"), home()), "/tmp/work");
        // No known home → left absolute.
        assert_eq!(
            abbrev_home(Path::new("/home/dev/repos/cubi"), None),
            "/home/dev/repos/cubi"
        );
    }

    #[test]
    fn k_abbreviation_boundaries() {
        assert_eq!(abbrev(999), "999");
        assert_eq!(abbrev(1000), "1.0k");
        assert_eq!(abbrev(8200), "8.2k");
        assert_eq!(abbrev(41300), "41.3k");
        // Compact variant rounds to whole k.
        assert_eq!(abbrev_compact(999), "999");
        assert_eq!(abbrev_compact(1000), "1k");
        assert_eq!(abbrev_compact(41300), "41k");
        assert_eq!(abbrev_compact(7800), "8k");
    }

    #[test]
    fn pct_rounds_to_nearest_integer() {
        let mut s = sample();
        s.context_used = Some(1);
        s.context_window = Some(3); // 33.33% → 33
        assert!(
            s.render_with_home(120, false, home())
                .contains("ctx 1/3 33%")
        );
    }

    #[test]
    fn zero_window_is_treated_as_unknown_pct() {
        let mut s = sample();
        s.context_window = Some(0); // avoid divide-by-zero → no percentage
        let got = s.render_with_home(120, false, home());
        // A zero window yields no usable percentage, so the ctx segment is
        // dropped and the rest of the line still renders without panicking.
        assert!(!got.contains('%'), "no pct for a zero window: {got}");
        assert_eq!(
            got,
            "qwen3:8b · ~/repos/cubi · tok 41.3k in/7.8k out · $0.00 (local)"
        );
    }

    #[test]
    fn compact_cost_strips_parenthetical() {
        assert_eq!(compact_cost("$0.00 (local)"), "$0.00");
        assert_eq!(compact_cost("$0.0123"), "$0.0123");
        assert_eq!(compact_cost("—"), "—");
    }

    #[test]
    fn middle_truncate_preserves_width_and_boundaries() {
        assert_eq!(middle_truncate("short", 10), "short");
        let t = middle_truncate("abcdefghij", 5);
        assert_eq!(t.chars().count(), 5);
        assert!(t.contains('…'));
        // Multibyte input stays valid UTF-8 (would panic on a byte-slice cut).
        let _ = middle_truncate("~/répös/çubî/déeply/nested", 8);
    }

    #[test]
    fn public_render_matches_helper_for_paths_outside_home() {
        // A path guaranteed not under the test machine's $HOME stays absolute,
        // so the public `render` (which reads $HOME) is deterministic here.
        let mut s = sample();
        s.cwd = PathBuf::from("/tmp/cubi-status-test/proj");
        let got = s.render(120, false);
        assert_eq!(got, s.render_with_home(120, false, None));
        assert!(!got.contains('\x1b'));
    }
}
