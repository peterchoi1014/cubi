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

pub(super) fn welcome_banner_rows(color: bool) -> Vec<String> {
    let stylize = |name: &'static str| {
        if color {
            name.bright_cyan().to_string()
        } else {
            name.to_string()
        }
    };

    let mut rows = vec![
        String::new(),
        format!(
            "hi, i'm Cubi — {}",
            if color {
                "a pocket-sized AI".bright_white().to_string()
            } else {
                "a pocket-sized AI".to_string()
            }
        ),
        format!(
            "{} · {} to exit · Tab completes slash commands · Ctrl-R searches history",
            stylize("/help"),
            stylize("/quit")
        ),
        "Commands:".to_string(),
    ];

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

impl ChatCLI {
    pub(super) fn print_welcome(&self) {
        // "Cubi" mascot — a tiny isometric cube with pixel-block eyes
        // and a smile. Box-drawing characters keep the silhouette crisp
        // on any monospace font; the small offset shadow sells the 3D.
        let mascot = [
            r#"  ┌───────┐  "#,
            r#"  │ ▣   ▣ │  "#,
            r#"  │   ◡   │  "#,
            r#"  └───────┘  "#,
            r#"   ░░░░░░░   "#,
        ];
        let mut rows = welcome_banner_rows(true);
        if let Some(line) = &self.mcp_health_line {
            rows.insert(3, line.clone());
        }
        println!();
        for (i, row) in rows.iter().enumerate() {
            let m = mascot.get(i).copied().unwrap_or("             ");
            println!("{}  {}", m.bright_cyan(), row);
        }
        println!();
    }

    /// Renders the model's final reply when streaming is off. Uses termimad
    /// for markdown when enabled and the terminal supports it; falls back to
    /// the same colored plain-text the streaming path produces.
    pub(super) fn render_final_reply(&self, content: &str) {
        if self.headless_mode {
            println!("{content}");
            return;
        }
        print!("{} ", "AI:".bright_blue().bold());
        if self.markdown_enabled && std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            // termimad prints with its own trailing newline; we leave the
            // outer println!("\n") in agent_turn to add the post-reply gap.
            println!();
            let skin = termimad::MadSkin::default();
            skin.print_text(content);
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
}
