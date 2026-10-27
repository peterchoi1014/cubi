use super::{ChatCLI, Cmd, agent_loop};
use crate::onboarding::AppConfig;
use crate::style::CubiStyle;
use crate::{oauth, output_styles, themes};
use anyhow::Result;

impl ChatCLI {
    /// `/passes` and `/effort` show or update the session pass setting.
    pub(super) fn handle_effort(&self, cmd: Cmd, args: &str) {
        let arg = args.trim();
        if arg.is_empty() {
            let current = self
                .config
                .read(|config| config.passes.clone())
                .unwrap_or_else(|| format!("{}", agent_loop::MAX_AGENT_STEPS));
            println!(
                "{} max passes: {} (default {})",
                "ℹ".bright_blue(),
                current.bright_cyan(),
                agent_loop::MAX_AGENT_STEPS
            );
            return;
        }
        let n = if matches!(cmd, Cmd::Effort) {
            match arg.to_ascii_lowercase().as_str() {
                "low" => 4,
                "medium" | "med" => 8,
                "high" => 12,
                other => {
                    if let Ok(n) = other.parse::<u32>() {
                        n
                    } else {
                        eprintln!(
                            "{} Use low|medium|high or a number 1..=12",
                            "Error:".bright_red()
                        );
                        return;
                    }
                }
            }
        } else {
            match arg.parse::<u32>() {
                Ok(n) => n,
                Err(_) => {
                    eprintln!("{} /passes expects a number 1..=12", "Error:".bright_red());
                    return;
                }
            }
        };
        if !(1..=12).contains(&n) {
            eprintln!("{} must be 1..=12", "Error:".bright_red());
            return;
        }
        self.config
            .update(|config| config.passes = Some(n.to_string()));
        println!("{} max passes set to {}", "✓".bright_green(), n);
    }

    /// Loads the on-disk AppConfig, applies `mutate`, persists. Errors
    /// are surfaced but never fatal — handlers fall back to "in-memory
    /// only" (still updates this session's typed configuration) so a
    /// read-only $HOME doesn't break the slash commands.
    pub(super) fn update_config(&self, mutate: impl FnOnce(&mut AppConfig)) -> Result<()> {
        let mut cfg = AppConfig::load();
        mutate(&mut cfg);
        self.config.update(|config| config.app = cfg.clone());
        cfg.save()
    }

    pub(super) fn handle_theme(&self, args: &str) {
        let arg = args.trim().to_ascii_lowercase();
        if arg.is_empty() {
            println!(
                "{} theme: {} (set with /theme [{}])",
                "ℹ".bright_blue(),
                self.config
                    .read(|config| config.ui.theme.clone())
                    .bright_cyan(),
                themes::VALID_THEMES.join("|")
            );
            return;
        }
        if !themes::is_valid_theme(&arg) {
            eprintln!(
                "{} Unknown theme '{}'. Expected {}.",
                "Error:".bright_red(),
                arg,
                themes::VALID_THEMES.join("|")
            );
            return;
        }
        self.config.update(|config| config.ui.theme = arg.clone());
        if let Err(e) = self.update_config(|c| c.theme = Some(arg.clone())) {
            eprintln!(
                "{} theme set in this session but not persisted: {}",
                "Warn:".bright_yellow(),
                e
            );
        }
        println!("{} theme set to {}", "✓".bright_green(), arg);
    }

    pub(super) fn handle_color(&self, args: &str) {
        let arg = args.trim().to_ascii_lowercase();
        match arg.as_str() {
            "" => println!(
                "{} colored output is {} (toggle with /color on|off)",
                "ℹ".bright_blue(),
                if colored::control::SHOULD_COLORIZE.should_colorize() {
                    "on"
                } else {
                    "off"
                }
            ),
            "on" => {
                self.config
                    .update(|config| config.ui.color = Some("on".to_string()));
                crate::style::set_color_override(true);
                if let Err(e) = self.update_config(|c| c.color = Some("on".to_string())) {
                    eprintln!("{} not persisted: {}", "Warn:".bright_yellow(), e);
                }
                println!("{} colored output ON", "✓".bright_green());
            }
            "off" => {
                self.config
                    .update(|config| config.ui.color = Some("off".to_string()));
                crate::style::set_color_override(false);
                if let Err(e) = self.update_config(|c| c.color = Some("off".to_string())) {
                    eprintln!("{} not persisted: {}", "Warn:".bright_yellow(), e);
                }
                println!("colored output OFF");
            }
            other => eprintln!(
                "{} Unknown value '{}'. Expected on|off.",
                "Error:".bright_red(),
                other
            ),
        }
    }

    pub(super) fn handle_output_style(&self, args: &str) {
        let arg = args.trim().to_ascii_lowercase();
        if arg.is_empty() {
            println!(
                "{} output style: {} (set with /output-style [{}])",
                "ℹ".bright_blue(),
                self.config
                    .read(|config| config.ui.output_style.clone())
                    .bright_cyan(),
                output_styles::VALID_STYLES.join("|")
            );
            return;
        }
        if !output_styles::is_valid_style(&arg) {
            eprintln!(
                "{} Unknown style '{}'. Expected {}.",
                "Error:".bright_red(),
                arg,
                output_styles::VALID_STYLES.join("|")
            );
            return;
        }
        self.config
            .update(|config| config.ui.output_style = arg.clone());
        if let Err(e) = self.update_config(|c| c.output_style = Some(arg.clone())) {
            eprintln!(
                "{} style set in this session but not persisted: {}",
                "Warn:".bright_yellow(),
                e
            );
        }
        println!("{} output style set to {}", "✓".bright_green(), arg);
    }

    pub(super) fn handle_vim(&self, args: &str) {
        let arg = args.trim().to_ascii_lowercase();
        match arg.as_str() {
            "" => println!(
                "{} vim mode: {} (toggle with /vim on|off; takes effect next session)",
                "ℹ".bright_blue(),
                self.config
                    .read(|config| config.ui.vim_mode.clone())
                    .bright_cyan()
            ),
            "on" | "off" => {
                self.config
                    .update(|config| config.ui.vim_mode = arg.clone());
                if let Err(e) = self.update_config(|c| c.vim_mode = Some(arg.clone())) {
                    eprintln!("{} not persisted: {}", "Warn:".bright_yellow(), e);
                }
                println!(
                    "{} vim mode {} (restart the CLI to apply)",
                    "✓".bright_green(),
                    arg
                );
            }
            other => eprintln!(
                "{} Unknown value '{}'. Expected on|off.",
                "Error:".bright_red(),
                other
            ),
        }
    }

    pub(super) fn handle_login(&self, args: &str) {
        let parsed = match oauth::parse_login_args(args) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{} {}", "Error:".bright_red(), e);
                println!(
                    "{} Usage: /login <provider> <access-token> [--refresh-token <token>] [--expires-in <seconds>]",
                    "Info:".bright_yellow()
                );
                return;
            }
        };
        let mut store = oauth::OAuthStore::load();
        store.upsert_login(&parsed);
        if let Err(e) = store.save() {
            eprintln!(
                "{} failed to persist OAuth token: {}",
                "Error:".bright_red(),
                e
            );
            return;
        }
        let env_var = oauth::provider_env_var(&parsed.provider);
        self.config.update(|config| {
            config
                .provider
                .keys
                .insert(env_var.clone(), parsed.access_token.clone().into());
        });
        println!(
            "{} OAuth token saved for '{}' and loaded into {}",
            "✓".bright_green(),
            parsed.provider.bright_cyan(),
            env_var.bright_cyan()
        );
    }

    pub(super) fn handle_logout(&self, args: &str) {
        let provider = if args.trim().is_empty() {
            "ollama"
        } else {
            args.trim()
        };
        let mut store = oauth::OAuthStore::load();
        let removed = store.remove_provider(provider);
        if let Err(e) = store.save() {
            eprintln!(
                "{} failed to persist OAuth store: {}",
                "Warn:".bright_yellow(),
                e
            );
        }
        let var = oauth::provider_env_var(provider);
        self.config.update(|config| {
            config.provider.keys.remove(&var);
        });
        println!(
            "{} cleared {} from this process{}",
            "✓".bright_green(),
            var.bright_cyan(),
            if removed {
                " and removed persisted OAuth token"
            } else {
                ""
            }
        );
    }

    pub(super) fn show_oauth_refresh(&self, args: &str) {
        let provider_filter = args.trim().to_ascii_lowercase();
        let store = oauth::OAuthStore::load();
        if store.providers.is_empty() {
            println!(
                "{} No OAuth tokens are stored yet. Use /login <provider> <access-token> first.",
                "ℹ".bright_blue()
            );
            return;
        }

        let mut refreshed = 0usize;
        for (provider, token) in &store.providers {
            if !provider_filter.is_empty() && provider != &provider_filter {
                continue;
            }
            let env_var = oauth::provider_env_var(provider);
            if token.is_expired() {
                println!(
                    "{} {} token is expired; not loading {}",
                    "⚠".bright_yellow(),
                    provider.bright_cyan(),
                    env_var.bright_cyan()
                );
                continue;
            }
            self.config.update(|config| {
                config
                    .provider
                    .keys
                    .insert(env_var.clone(), token.access_token.clone().into());
            });
            refreshed += 1;
            match token.expires_at_unix {
                Some(ts) => println!(
                    "{} refreshed {} into {} (expires at unix={})",
                    "✓".bright_green(),
                    provider.bright_cyan(),
                    env_var.bright_cyan(),
                    ts.to_string().bright_black()
                ),
                None => println!(
                    "{} refreshed {} into {} (no expiry set)",
                    "✓".bright_green(),
                    provider.bright_cyan(),
                    env_var.bright_cyan()
                ),
            }
        }
        if refreshed == 0 {
            if !provider_filter.is_empty() {
                println!(
                    "{} No non-expired token found for provider '{}'.",
                    "ℹ".bright_blue(),
                    provider_filter.bright_cyan()
                );
            } else {
                println!(
                    "{}",
                    "ℹ No non-expired OAuth tokens to refresh.".bright_blue()
                );
            }
        }
    }

    pub(super) fn handle_privacy_settings(&self, args: &str) {
        let arg = args.trim();
        if arg.is_empty() {
            let telemetry = self
                .config
                .read(|config| config.telemetry.clone())
                .unwrap_or_else(|| "off".to_string());
            println!(
                "\n{}\n  telemetry: {} (no remote analytics implemented yet)\n  \
                 local data: ~/.cubi/ (sessions, memdir, schedule, messages, triggers)\n  \
                 set with: /privacy-settings telemetry on|off\n",
                "Privacy:".bright_yellow().bold(),
                telemetry.bright_cyan()
            );
            return;
        }
        let parts: Vec<&str> = arg.split_whitespace().collect();
        match parts.as_slice() {
            ["telemetry", v] if matches!(*v, "on" | "off") => {
                self.config
                    .update(|config| config.telemetry = Some((*v).to_string()));
                println!("{} telemetry set to {}", "✓".bright_green(), v);
            }
            _ => eprintln!(
                "{} Usage: /privacy-settings telemetry on|off",
                "Error:".bright_red()
            ),
        }
    }

    pub(super) fn handle_debug_tool_call(&self, args: &str) {
        let arg = args.trim().to_ascii_lowercase();
        match arg.as_str() {
            "" => println!(
                "{} debug-tool-call: {} (toggle with /debug-tool-call on|off)",
                "ℹ".bright_blue(),
                self.config
                    .read(|config| config.debug_tool_call.clone())
                    .unwrap_or_else(|| "off".to_string())
                    .bright_cyan()
            ),
            "on" => {
                self.config
                    .update(|config| config.debug_tool_call = Some("on".to_string()));
                println!("{} debug-tool-call ON", "✓".bright_green());
            }
            "off" => {
                self.config.update(|config| config.debug_tool_call = None);
                println!("debug-tool-call OFF");
            }
            other => eprintln!(
                "{} Unknown value '{}'. Expected on|off.",
                "Error:".bright_red(),
                other
            ),
        }
    }

    pub(super) fn reset_limits(&self) {
        // The Ollama client tracks retries per-call (no persistent state), so
        // "resetting" is informational. Clear the typed override if set so
        // backoff returns to defaults.
        self.config
            .update(|config| config.rate_limit_backoff_ms = None);
        println!(
            "{} Rate-limit / retry state cleared (Ollama retries are per-call).",
            "✓".bright_green()
        );
    }
}
