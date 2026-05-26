use super::*;

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
        let rows = welcome_banner_rows(true);
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
pub(super) fn print_stats_footer(stats: &ChatStats) {
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
    println!(
        "{} {}",
        "↳".bright_black(),
        parts.join(" · ").bright_black()
    );
}
