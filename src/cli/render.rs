use super::*;

/// Computes context-window utilization as a percentage (0..=100, but
/// may exceed 100 when the provider's reported prompt tokens overflow
/// the heuristic window — we still surface it rather than clamp so
/// users see something is wrong).
pub(crate) fn utilization_pct(prompt_tokens: u64, window: usize) -> u32 {
    if window == 0 {
        return 0;
    }
    let w = window as u64;
    let pct = prompt_tokens.saturating_mul(100) / w;
    u32::try_from(pct).unwrap_or(u32::MAX)
}

/// Threshold above which a fenced code block gets gutter line numbers.
pub(super) const CODE_BLOCK_LINENO_THRESHOLD: usize = 12;

/// Polishes assistant markdown output for terminal rendering:
///
/// * Emits a dim language label above a fenced code block when the fence
///   info-string is non-empty (`─ rust ─` style on a TTY, just `rust` in
///   `NO_COLOR` mode).
/// * Long fenced blocks (more than [`CODE_BLOCK_LINENO_THRESHOLD`] lines)
///   get a right-aligned dim line-number gutter; shorter blocks render
///   exactly as today.
/// * Inline links `[text](url)` render as underline + dim (`text`)
///   instead of the previous bold-cyan; with no color the link text
///   appears verbatim with parenthesised URL.
///
/// All other markdown is passed through unchanged. This is intentional:
/// the function focuses on the polish branches called out by the spec
/// and leaves headings / bullets / emphasis to the caller's
/// existing rendering pipeline.
pub(super) fn polish_markdown(content: &str, color: bool) -> String {
    let mut out = String::new();
    let mut lines = content.lines().peekable();
    while let Some(line) = lines.next() {
        if let Some(info) = fence_open(line) {
            // Collect the fenced block body up to the matching closer.
            let mut body: Vec<&str> = Vec::new();
            let mut closed = false;
            while let Some(next) = lines.peek() {
                if fence_close(next) {
                    let _ = lines.next();
                    closed = true;
                    break;
                }
                body.push(lines.next().unwrap());
            }
            render_code_block(&info, &body, color, &mut out);
            if !closed {
                // Best-effort: model emitted an unclosed fence. We've
                // already consumed the body; add nothing else.
            }
            continue;
        }
        out.push_str(&render_inline_links(line, color));
        out.push('\n');
    }
    out
}

/// Returns the info string when `line` opens a fenced code block (the
/// info string is everything after the backticks, trimmed).
fn fence_open(line: &str) -> Option<String> {
    let t = line.trim_start();
    if !t.starts_with("```") {
        return None;
    }
    Some(t.trim_start_matches('`').trim().to_string())
}

fn fence_close(line: &str) -> bool {
    let t = line.trim();
    t == "```" || (t.starts_with("```") && t.chars().all(|c| c == '`'))
}

fn render_code_block(info: &str, body: &[&str], color: bool, out: &mut String) {
    if !info.is_empty() {
        if color {
            out.push_str(&format!("{}\n", format!("─ {} ─", info).bright_black()));
        } else {
            out.push_str(info);
            out.push('\n');
        }
    }
    let n = body.len();
    if n > CODE_BLOCK_LINENO_THRESHOLD {
        let width = n.to_string().len();
        for (i, line) in body.iter().enumerate() {
            let num = format!("{:>width$}", i + 1, width = width);
            if color {
                out.push_str(&format!("{} {}\n", num.bright_black(), line));
            } else {
                out.push_str(&format!("{} {}\n", num, line));
            }
        }
    } else {
        for line in body {
            out.push_str(line);
            out.push('\n');
        }
    }
}

/// Rewrites inline `[text](url)` to a toned-down style: underline + dim
/// when color is on, `text (url)` when not.
fn render_inline_links(line: &str, color: bool) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    loop {
        let Some(lb) = rest.find('[') else {
            out.push_str(rest);
            break;
        };
        // Find the matching ']' that's immediately followed by '('.
        let after_lb = &rest[lb + 1..];
        let Some(rb_rel) = after_lb.find(']') else {
            out.push_str(&rest[..=lb]);
            rest = &rest[lb + 1..];
            continue;
        };
        let rb = lb + 1 + rb_rel;
        if rest.get(rb + 1..rb + 2) != Some("(") {
            out.push_str(&rest[..=rb]);
            rest = &rest[rb + 1..];
            continue;
        }
        let after_lp = &rest[rb + 2..];
        let Some(rp_rel) = after_lp.find(')') else {
            out.push_str(&rest[..=rb]);
            rest = &rest[rb + 1..];
            continue;
        };
        let rp = rb + 2 + rp_rel;
        let text = &rest[lb + 1..rb];
        let url = &rest[rb + 2..rp];
        out.push_str(&rest[..lb]);
        if color {
            // Underline + dim ("\x1b[2;4m...\x1b[0m"); colored crate
            // doesn't expose dim+underline together so we emit raw.
            out.push_str(&format!("\x1b[2;4m{}\x1b[0m", text));
        } else {
            out.push_str(text);
            out.push_str(" (");
            out.push_str(url);
            out.push(')');
        }
        rest = &rest[rp + 1..];
    }
    out
}

/// Compact pixel-art chibi mascot for Cubi. Four rows of half-block /
/// quarter-block glyphs that read as a small cube-shaped character
/// with a face. Inspired by Claude Code's single-glyph welcome, but
/// expanded into a tiny sprite so a fresh REPL has more personality.
pub(super) fn mascot_rows(color: bool) -> Vec<String> {
    let art = ["▄▀▀▄", "█◕◕█", "█ ◡█", "▀▄▄▀"];
    art.iter()
        .map(|line| {
            if color {
                line.bright_cyan().to_string()
            } else {
                (*line).to_string()
            }
        })
        .collect()
}

pub(super) fn welcome_banner_rows(color: bool) -> Vec<String> {
    let stylize = |name: &'static str| {
        if color {
            name.bright_cyan().to_string()
        } else {
            name.to_string()
        }
    };

    let tagline = if color {
        "a pocket-sized AI".bright_white().to_string()
    } else {
        "a pocket-sized AI".to_string()
    };

    let mut rows = vec![String::new()];
    rows.extend(mascot_rows(color));
    rows.extend([
        String::new(),
        format!("hi, i'm Cubi — {}", tagline),
        format!(
            "{} · {} to exit · Tab completes slash commands · Ctrl-R searches history",
            stylize("/help"),
            stylize("/quit")
        ),
        "Commands:".to_string(),
    ]);

    for chunk in commands::command_names().collect::<Vec<_>>().chunks(5) {
        rows.push(
            chunk
                .iter()
                .map(|name| stylize(name))
                .collect::<Vec<_>>()
                .join("  "),
        );
    }

    rows
}

/// Pure formatter for the concise startup banner. Kept free of any
/// process state (env, config, IO) so it can be unit-tested directly.
///
/// Shape: `cubi v{ver} • {model} ({provider}) • mcp: ok=N failed=M not_loaded=K • sessions {label}`.
/// Zero-valued segments of the MCP triple are omitted so a clean
/// "everything green" run shows just `mcp: ok=N`.
///
/// `color=false` returns a plain-text line suitable for `NO_COLOR` /
/// piped runs; `color=true` paints the prefix and separators dim.
#[allow(clippy::too_many_arguments)]
pub(super) fn format_banner(
    version: &str,
    model: &str,
    provider: &str,
    mcp_ok: usize,
    mcp_failed: usize,
    mcp_not_loaded: usize,
    sessions: crate::sessions::SessionStoreStatus,
    color: bool,
) -> String {
    let sep = if color {
        "•".bright_black().to_string()
    } else {
        "•".to_string()
    };
    let head = if color {
        format!("cubi v{}", version).bright_cyan().to_string()
    } else {
        format!("cubi v{}", version)
    };
    let mcp_segment = format_mcp_health_segment(mcp_ok, mcp_failed, mcp_not_loaded);
    format!(
        "{head} {sep} {model} ({provider}) {sep} {mcp} {sep} sessions {label}",
        model = model,
        provider = provider,
        mcp = mcp_segment,
        label = sessions.label(),
    )
}

/// Returns the `mcp: ok=N failed=M not_loaded=K` segment of the banner.
/// Zero-valued components are omitted so a healthy run reads as
/// `mcp: ok=N`; the all-zero state collapses to `mcp: none`.
pub(crate) fn format_mcp_health_segment(ok: usize, failed: usize, not_loaded: usize) -> String {
    if ok == 0 && failed == 0 && not_loaded == 0 {
        return "mcp: none".to_string();
    }
    let mut parts: Vec<String> = Vec::new();
    if ok > 0 {
        parts.push(format!("ok={ok}"));
    }
    if failed > 0 {
        parts.push(format!("failed={failed}"));
    }
    if not_loaded > 0 {
        parts.push(format!("not_loaded={not_loaded}"));
    }
    format!("mcp: {}", parts.join(" "))
}

impl ChatCLI {
    pub(super) fn print_welcome(&self) {
        // Show the mascot + command grid first so a fresh REPL is
        // immediately self-describing, then drop the concise one-line
        // status line that summarizes model / MCP / sessions state.
        let color = crate::style::should_color();
        for row in welcome_banner_rows(color) {
            println!("{}", row);
        }
        let model = self.executor.get_model();
        let provider = self.executor.provider_name();
        let sessions = self
            .session_store
            .as_ref()
            .map(|s| s.status())
            .unwrap_or(crate::sessions::SessionStoreStatus::Missing);
        let (mcp_ok, mcp_failed, mcp_not_loaded) = self.mcp_counts;
        let line = format_banner(
            env!("CARGO_PKG_VERSION"),
            model,
            provider,
            mcp_ok,
            mcp_failed,
            mcp_not_loaded,
            sessions,
            color,
        );
        println!("{}", line);
    }

    /// Renders the model's final reply when streaming is off. When
    /// markdown rendering is enabled and stdout is a TTY, runs the reply
    /// through [`polish_markdown`] (fenced code blocks get a dim
    /// language label, long blocks get line numbers, inline links are
    /// underline+dim); otherwise prints the plain colored text the
    /// streaming path uses.
    pub(super) fn render_final_reply(&self, content: &str) {
        if self.headless_mode {
            println!("{content}");
            return;
        }
        print!("{} ", "AI:".bright_blue().bold());
        if self.markdown_enabled && std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            println!();
            let color = std::env::var("NO_COLOR").is_err();
            let rendered = polish_markdown(content, color);
            print!("{}", rendered);
        } else {
            println!("{}", content.bright_white());
        }
    }
}

/// Prints a one-line dim footer summarizing token usage and wall time for
/// the just-completed turn. Only the fields the provider actually returned
/// are shown; missing fields are skipped to avoid printing "0 in / 0 out".
///
/// When `window` is known and `prompt_tokens > 0`, appends a
/// `(N% of M-token window)` suffix so users can eyeball how close
/// they are to /compact territory.
pub(super) fn print_stats_footer(stats: &ChatStats, window: Option<usize>) {
    let mut parts: Vec<String> = Vec::new();
    if stats.prompt_tokens > 0 || stats.completion_tokens > 0 {
        parts.push(format!(
            "{} in / {} out",
            stats.prompt_tokens, stats.completion_tokens
        ));
    }
    if stats.elapsed_ms > 0 {
        parts.push(format!("{} ms", stats.elapsed_ms));
        if stats.completion_tokens > 0 {
            let tps = (stats.completion_tokens as f64) * 1000.0 / (stats.elapsed_ms as f64);
            parts.push(format!("{:.1} tok/s", tps));
        }
    }
    if parts.is_empty() {
        return;
    }
    let mut line = parts.join(" · ");
    if let Some(w) = window {
        if stats.prompt_tokens > 0 && w > 0 {
            let pct = utilization_pct(stats.prompt_tokens, w);
            line.push_str(&format!(" ({}% of {}-token window)", pct, w));
        }
    }
    println!("{} {}", "↳".bright_black(), line.bright_black());
}

#[cfg(test)]
mod tests {
    use super::utilization_pct;

    #[test]
    fn utilization_pct_basic_ratios() {
        assert_eq!(utilization_pct(0, 1000), 0);
        assert_eq!(utilization_pct(250, 1000), 25);
        assert_eq!(utilization_pct(1000, 1000), 100);
        assert_eq!(utilization_pct(1500, 1000), 150);
    }

    #[test]
    fn utilization_pct_zero_window_is_safe() {
        assert_eq!(utilization_pct(42, 0), 0);
    }

    use super::format_banner;
    use super::format_mcp_health_segment;
    use crate::sessions::SessionStoreStatus;

    #[test]
    fn banner_plain_contains_all_segments() {
        let line = format_banner(
            "9.9.9",
            "llama3",
            "ollama",
            2,
            1,
            0,
            SessionStoreStatus::Ok,
            false,
        );
        assert!(line.contains("cubi v9.9.9"), "version: {}", line);
        assert!(line.contains("llama3 (ollama)"), "model/provider: {}", line);
        assert!(line.contains("mcp: ok=2 failed=1"), "mcp: {}", line);
        assert!(line.contains("sessions ok"), "sessions: {}", line);
        // Plain (no color) must contain no ANSI escape sequences.
        assert!(
            !line.contains('\u{1b}'),
            "should not contain ANSI: {:?}",
            line
        );
    }

    #[test]
    fn banner_session_status_labels() {
        for (status, label) in [
            (SessionStoreStatus::Ok, "ok"),
            (SessionStoreStatus::ReadOnly, "ro"),
            (SessionStoreStatus::Missing, "missing"),
        ] {
            let line = format_banner("1.0.0", "m", "p", 0, 0, 0, status, false);
            assert!(line.ends_with(&format!("sessions {}", label)), "{}", line);
        }
    }

    #[test]
    fn banner_all_zero_mcp_renders_none() {
        let line = format_banner(
            "0.1.0",
            "m",
            "p",
            0,
            0,
            0,
            SessionStoreStatus::Missing,
            false,
        );
        assert!(line.contains("mcp: none"), "{}", line);
    }

    #[test]
    fn mcp_health_segment_omits_zero_components() {
        assert_eq!(format_mcp_health_segment(2, 0, 0), "mcp: ok=2");
        assert_eq!(format_mcp_health_segment(0, 1, 0), "mcp: failed=1");
        assert_eq!(format_mcp_health_segment(0, 0, 3), "mcp: not_loaded=3");
        assert_eq!(
            format_mcp_health_segment(2, 1, 3),
            "mcp: ok=2 failed=1 not_loaded=3"
        );
        // All-zero collapses to a single "none" token.
        assert_eq!(format_mcp_health_segment(0, 0, 0), "mcp: none");
    }

    use super::polish_markdown;

    #[test]
    fn polish_passthrough_short_block_no_info() {
        let input = "before\n```\nfoo\nbar\n```\nafter\n";
        let out = polish_markdown(input, false);
        // Short block with no info string: no language label, no line numbers.
        assert!(
            out.contains("foo\nbar\n"),
            "code body should be unchanged: {:?}",
            out
        );
        assert!(!out.contains(" 1 foo"), "no line numbers: {:?}", out);
        // Fence markers are stripped (we render code without the ``` lines).
        assert!(!out.contains("```"), "fence markers gone: {:?}", out);
    }

    #[test]
    fn polish_short_block_emits_language_label() {
        let input = "```rust\nfn x(){}\n```\n";
        let out = polish_markdown(input, false);
        assert!(out.starts_with("rust\n"), "label first line: {:?}", out);
        assert!(out.contains("fn x(){}\n"), "body kept: {:?}", out);
    }

    #[test]
    fn polish_long_block_gets_line_numbers() {
        let mut body = String::new();
        for i in 0..15 {
            body.push_str(&format!("L{}\n", i));
        }
        let input = format!("```py\n{}```\n", body);
        let out = polish_markdown(&input, false);
        assert!(out.starts_with("py\n"), "label first: {:?}", out);
        // 15 lines → width=2; line 1 right-padded as ` 1`, line 15 as `15`.
        assert!(out.contains(" 1 L0\n"), "1st line numbered: {:?}", out);
        assert!(out.contains("15 L14\n"), "15th line numbered: {:?}", out);
    }

    #[test]
    fn polish_links_no_color_inlines_url() {
        let out = polish_markdown("see [docs](https://example.com) here\n", false);
        assert_eq!(out, "see docs (https://example.com) here\n");
    }

    #[test]
    fn polish_links_color_uses_underline_dim() {
        let out = polish_markdown("[x](u)\n", true);
        assert!(
            out.contains("\x1b[2;4mx\x1b[0m"),
            "expected underline+dim ANSI: {:?}",
            out
        );
        assert!(!out.contains("(u)"), "URL hidden in color mode: {:?}", out);
    }

    #[test]
    fn polish_passes_plain_text_through() {
        let out = polish_markdown("hello world\n", false);
        assert_eq!(out, "hello world\n");
    }

    #[test]
    fn polish_link_no_match_left_intact() {
        let out = polish_markdown("[no closing paren\n", false);
        assert_eq!(out, "[no closing paren\n");
    }
}
