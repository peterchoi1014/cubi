//! First-run onboarding wizard + persistent user config.
//!
//! Before this module existed, the only way to override the hard-coded
//! default model was the `CUBI_MODEL` environment variable. New
//! users got dropped onto whichever model the binary shipped with and
//! had no opportunity to (a) trust their project for write/exec tools or
//! (b) opt in to a starter `CUBI.md`.
//!
//! This wizard runs once, gated on `config.onboarded == false`. It:
//!
//!   1. Lists the installed Ollama models and lets the user pick one.
//!   2. Offers to trust the current working directory (writes into the
//!      same `Permissions` store as `/trust`).
//!   3. Offers to create the starter `CUBI.md` for the project.
//!
//! The selected model is persisted to `~/.cubi/config.json`. On
//! subsequent runs, model resolution is:
//! `CUBI_MODEL` ▸ `config.default_model` ▸ baked-in fallback.
//!
//! Resolution order is deliberate: the env var still wins so CI / shell
//! aliases that pre-date this module keep working.

use crate::style::CubiStyle;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::ollama::OllamaClient;
use crate::permissions::Permissions;
use crate::project_memory;

/// Persistent user-level configuration. Lives next to the trust store at
/// `~/.cubi/config.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Model selected during onboarding (or via a future `/config` command).
    /// `None` means "no preference, fall through to the baked-in default".
    #[serde(default)]
    pub default_model: Option<String>,
    /// Set to `true` once the wizard has run to completion. The wizard
    /// only runs while this is `false`, so users never get pestered twice.
    #[serde(default)]
    pub onboarded: bool,
    /// UI theme: "auto" | "light" | "dark". Drives the bundled colour
    /// palette in `themes.rs`.
    #[serde(default)]
    pub theme: Option<String>,
    /// Output-style preset: "concise" | "markdown" | "explanatory".
    /// Surfaces as a system prompt prefix injected by `cli.rs`.
    #[serde(default)]
    pub output_style: Option<String>,
    /// Coloured output toggle: "on" | "off". `None` means follow the
    /// `colored` crate's default (TTY detection).
    #[serde(default)]
    pub color: Option<String>,
    /// Vim-mode toggle for the readline editor: "on" | "off".
    #[serde(default)]
    pub vim_mode: Option<String>,
    /// Opt-in telemetry / debug logging to
    /// `~/.cubi/telemetry.log`. Off by default.
    #[serde(default)]
    pub telemetry: bool,
    /// Default wall-clock timeout for model-requested tool execution.
    ///
    /// * `Some(n)` where `n > 0`: wrap each tool call in `n` seconds.
    /// * `Some(0)` or `None`: no wall-clock timeout (tools run until they
    ///   complete or the user cancels). `None` is the implicit default
    ///   for configs written before this field existed; `Some(0)` is the
    ///   explicit opt-out.
    #[serde(default = "default_tool_timeout_secs")]
    pub tool_timeout_secs: Option<u64>,
    /// Max number of LLM HTTP retry attempts on transient failures
    /// (connect errors, 408/429/5xx). Respects `Retry-After` when
    /// present. `0` disables retries entirely. Default is 2.
    #[serde(default = "default_llm_max_retries")]
    pub llm_max_retries: u32,
    /// Auto-fire `/compact` when the prompt token estimate crosses
    /// `compact_threshold_pct` of the active model's context window.
    /// Disable by setting to `false` if you want strict control over when
    /// summarization runs.
    #[serde(default = "default_auto_compact")]
    pub auto_compact: bool,
    /// Threshold (percent of the model's context window) at which an
    /// auto-compaction fires before the next user turn. Clamped to
    /// 50..=95 on load — values outside that band are clipped with a
    /// `tracing::warn!` so the user notices, but the binary still
    /// starts. Default: 80.
    #[serde(default = "default_compact_threshold_pct")]
    pub compact_threshold_pct: u8,
    /// Default `--receipts <path>`. When set, every cubi session
    /// produces a hash-chained JSONL audit log at this path (unless
    /// overridden by the CLI flag or `CUBI_RECEIPTS` env). `None` (the
    /// default) keeps the side-channel opt-in.
    #[serde(default)]
    pub receipts: Option<String>,
    /// Schema version for the on-disk config. Bumped by `migrations.rs`
    /// when a breaking change to this struct is introduced; older configs
    /// are migrated forward on load.
    #[serde(default)]
    pub config_version: u32,
}

fn default_tool_timeout_secs() -> Option<u64> {
    Some(60)
}

fn default_llm_max_retries() -> u32 {
    2
}

fn default_auto_compact() -> bool {
    true
}

fn default_compact_threshold_pct() -> u8 {
    80
}

/// Allowed inclusive range for `compact_threshold_pct`. Values outside
/// the band are clamped on load.
pub const COMPACT_THRESHOLD_MIN: u8 = 50;
pub const COMPACT_THRESHOLD_MAX: u8 = 95;

/// Clamps `compact_threshold_pct` into [COMPACT_THRESHOLD_MIN,
/// COMPACT_THRESHOLD_MAX]. Logs a warning when a value is changed so
/// users who hand-edit config.json see why their setting was ignored.
pub fn clamp_compact_threshold(pct: u8) -> u8 {
    if pct < COMPACT_THRESHOLD_MIN {
        tracing::warn!(
            target: "cubi::onboarding",
            value = pct,
            min = COMPACT_THRESHOLD_MIN,
            "compact_threshold_pct below allowed range; clamping",
        );
        return COMPACT_THRESHOLD_MIN;
    }
    if pct > COMPACT_THRESHOLD_MAX {
        tracing::warn!(
            target: "cubi::onboarding",
            value = pct,
            max = COMPACT_THRESHOLD_MAX,
            "compact_threshold_pct above allowed range; clamping",
        );
        return COMPACT_THRESHOLD_MAX;
    }
    pct
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            default_model: None,
            onboarded: false,
            theme: None,
            output_style: None,
            color: None,
            vim_mode: None,
            telemetry: false,
            tool_timeout_secs: default_tool_timeout_secs(),
            llm_max_retries: default_llm_max_retries(),
            auto_compact: default_auto_compact(),
            compact_threshold_pct: default_compact_threshold_pct(),
            receipts: None,
            config_version: 0,
        }
    }
}

impl AppConfig {
    pub fn storage_path() -> Option<PathBuf> {
        Some(
            crate::sessions::home_dir()?
                .join(".cubi")
                .join("config.json"),
        )
    }

    /// Loads the on-disk config. Missing or unreadable files yield a
    /// default (un-onboarded) config rather than erroring — a missing
    /// file just means "first run on this machine".
    pub fn load() -> Self {
        let Some(path) = Self::storage_path() else {
            return Self::default();
        };
        let mut cfg: Self = match fs::read_to_string(&path) {
            Ok(raw) => serde_json::from_str(&raw).unwrap_or_default(),
            Err(_) => Self::default(),
        };
        cfg.compact_threshold_pct = clamp_compact_threshold(cfg.compact_threshold_pct);
        cfg
    }

    pub fn save(&self) -> Result<()> {
        let path = Self::storage_path().context("Could not resolve home directory")?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create {}", parent.display()))?;
        }
        let json = serde_json::to_string_pretty(self)?;
        fs::write(&path, json).with_context(|| format!("Failed to write {}", path.display()))?;
        Ok(())
    }
}

/// Resolves the model to use at startup. Precedence:
///
///   1. `CUBI_MODEL` environment variable.
///   2. `config.default_model` from `~/.cubi/config.json`.
///   3. The baked-in `fallback` constant from `main`.
pub fn resolve_model(config: &AppConfig, fallback: &str) -> String {
    if let Ok(env) = std::env::var("CUBI_MODEL") {
        if !env.is_empty() {
            return env;
        }
    }
    if let Some(model) = &config.default_model {
        if !model.is_empty() {
            return model.clone();
        }
    }
    fallback.to_string()
}

/// Heuristic check: returns `true` when `model` matches a known family
/// that does NOT reliably emit native `tool_calls` against Ollama's
/// `tools:` field. Used to print a startup warning when the active model
/// is being asked to drive the agent loop in `agent_loop.rs`.
///
/// Conservative on purpose — only the families with documented poor
/// tool-calling behavior at small sizes are flagged. Anything else falls
/// through as "probably fine" so we don't nag users running tool-capable
/// models we just haven't seen yet.
pub fn is_known_non_tool_capable(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    // Strip any `:tag` suffix so we match on the family name.
    let family = m.split(':').next().unwrap_or(&m);
    // Tiny llama3.2 variants (1b, 3b-without-tools-tuning) are the main
    // offender that triggered this check.
    if family == "llama3.2" {
        if let Some(tag) = m.split(':').nth(1) {
            // The 1b tag is known-bad; 3b is borderline but works for
            // simple cases — only flag 1b to avoid false positives.
            return tag.starts_with("1b");
        }
    }
    matches!(
        family,
        "tinyllama" | "smollm" | "smollm2" | "gemma" | "gemma2" | "gemma3" | "phi" | "phi3"
    ) || family.starts_with("orca-mini")
}

/// Runs the first-run wizard if appropriate, mutating `config` and the
/// shared `permissions` store as the user makes choices. Idempotent
/// across runs because the wizard sets `config.onboarded = true` on its
/// way out.
///
/// The wizard is suppressed when stdin is not a TTY (CI / piped input),
/// when `CUBI_NO_ONBOARD=1` is set (escape hatch for scripted
/// installs), or when `config.onboarded` is already true.
pub async fn run_if_needed(
    config: &mut AppConfig,
    ollama: &OllamaClient,
    permissions: &Arc<Mutex<Permissions>>,
) -> Result<()> {
    if config.onboarded {
        return Ok(());
    }
    if std::env::var("CUBI_NO_ONBOARD")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false)
    {
        // Still mark as onboarded so we don't repeatedly check the env var.
        config.onboarded = true;
        let _ = config.save();
        return Ok(());
    }
    if !io::stdin().is_terminal() {
        // Non-interactive shell: don't block on a prompt that nobody can
        // see. Leave `onboarded = false` so the next interactive run
        // gets the wizard.
        return Ok(());
    }

    println!();
    println!("{}", "Welcome to Cubi!".bright_cyan().bold());
    println!(
        "{}",
        "Let's set a few things up. You can change any of these later.".bright_white()
    );
    println!();

    // 1) Model picker.
    match ollama.list_models().await {
        Ok(models) if !models.is_empty() => {
            println!("{}", "Available Ollama models:".bright_yellow().bold());
            for (i, m) in models.iter().enumerate() {
                let label = if is_known_non_tool_capable(m) {
                    format!("{}  {}", m, "(chat only — no tools)".bright_black())
                } else {
                    m.bright_cyan().to_string()
                };
                println!("  {}. {}", i + 1, label);
            }
            println!(
                "{} Recommended for tool-calling: {}. Alternatives: {} (best for code) or {} (smallest).",
                "ℹ".bright_blue(),
                "qwen3:8b".bright_cyan(),
                "devstral".bright_cyan(),
                "qwen3:4b".bright_cyan(),
            );
            let pick = prompt(&format!(
                "Pick a default model [1-{}] (press Enter to skip): ",
                models.len()
            ))?;
            let trimmed = pick.trim();
            if !trimmed.is_empty() {
                if let Ok(idx) = trimmed.parse::<usize>() {
                    if (1..=models.len()).contains(&idx) {
                        config.default_model = Some(models[idx - 1].clone());
                        println!(
                            "{} Default model: {}",
                            "✓".bright_green(),
                            models[idx - 1].bright_cyan()
                        );
                    } else {
                        println!(
                            "{} Index out of range; leaving the default unchanged.",
                            "ℹ".bright_blue()
                        );
                    }
                } else {
                    // Allow typing a model name directly.
                    config.default_model = Some(trimmed.to_string());
                    println!(
                        "{} Default model: {}",
                        "✓".bright_green(),
                        trimmed.bright_cyan()
                    );
                }
            }
        }
        Ok(_) => {
            println!(
                "{} No Ollama models installed yet. We'll use the baked-in default for now; \
                 install one with `ollama pull <model>` and re-run.",
                "ℹ".bright_blue()
            );
        }
        Err(e) => {
            println!(
                "{} Couldn't list Ollama models ({}); skipping model picker.",
                "Warn:".bright_yellow(),
                e
            );
        }
    }
    println!();

    // 2) Project trust.
    if let Ok(cwd) = std::env::current_dir() {
        let already = permissions.lock().unwrap().contains(&cwd);
        if already {
            println!(
                "{} {} is already trusted.",
                "ℹ".bright_blue(),
                cwd.display().to_string().bright_cyan()
            );
        } else {
            let yn = prompt(&format!(
                "Trust this project ({}) for write/exec tools? [y/N]: ",
                cwd.display()
            ))?;
            if is_yes(&yn) {
                let mut perms = permissions.lock().unwrap();
                match perms.trust_dir(&cwd) {
                    Ok(_) => {
                        if let Err(e) = perms.save() {
                            eprintln!(
                                "{} Failed to persist trust store: {}",
                                "Warn:".bright_yellow(),
                                e
                            );
                        } else {
                            println!(
                                "{} Trusted {}",
                                "✓".bright_green(),
                                cwd.display().to_string().bright_cyan()
                            );
                        }
                    }
                    Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
                }
            }
        }
    }
    println!();

    // 3) CUBI.md.
    if project_memory::find_memory_path().is_none() {
        let yn = prompt("Create a starter CUBI.md in this project? [y/N]: ")?;
        if is_yes(&yn) {
            match project_memory::write_starter_if_absent() {
                Ok(true) => println!(
                    "{} Wrote starter {}",
                    "✓".bright_green(),
                    project_memory::MEMORY_FILENAME.bright_cyan()
                ),
                Ok(false) => {} // someone else created it between checks
                Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
            }
        }
    }

    config.onboarded = true;
    if let Err(e) = config.save() {
        eprintln!(
            "{} Failed to persist config: {} (wizard will run again next time)",
            "Warn:".bright_yellow(),
            e
        );
    }
    println!("{}\n", "Setup complete.".bright_green().bold());
    Ok(())
}

fn prompt(message: &str) -> Result<String> {
    print!("{}", message);
    io::stdout().flush().ok();
    let mut buf = String::new();
    io::stdin()
        .read_line(&mut buf)
        .context("Failed to read from stdin")?;
    Ok(buf)
}

fn is_yes(s: &str) -> bool {
    matches!(s.trim().to_ascii_lowercase().as_str(), "y" | "yes")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // `resolve_model` reads from process-global env state. Serialize the
    // tests that touch `CUBI_MODEL` so parallel test threads don't
    // race each other.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn is_yes_accepts_common_affirmations() {
        for s in ["y", "Y", "yes", "YES", "  yes  "] {
            assert!(is_yes(s), "expected yes for {:?}", s);
        }
        for s in ["", "n", "no", "maybe", "yeah"] {
            assert!(!is_yes(s), "expected not-yes for {:?}", s);
        }
    }

    #[test]
    fn resolve_model_env_beats_config() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("CUBI_MODEL").ok();
        unsafe {
            std::env::set_var("CUBI_MODEL", "env-model");
        }
        let cfg = AppConfig {
            default_model: Some("config-model".into()),
            onboarded: true,
            ..AppConfig::default()
        };
        assert_eq!(resolve_model(&cfg, "fallback"), "env-model");
        unsafe {
            std::env::remove_var("CUBI_MODEL");
        }
        if let Some(v) = prev {
            unsafe { std::env::set_var("CUBI_MODEL", v) }
        }
    }

    #[test]
    fn resolve_model_falls_through_to_default() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("CUBI_MODEL").ok();
        unsafe {
            std::env::remove_var("CUBI_MODEL");
        }
        let cfg = AppConfig::default();
        assert_eq!(resolve_model(&cfg, "fallback"), "fallback");
        let cfg2 = AppConfig {
            default_model: Some("foo".into()),
            onboarded: true,
            ..AppConfig::default()
        };
        assert_eq!(resolve_model(&cfg2, "fallback"), "foo");
        if let Some(v) = prev {
            unsafe { std::env::set_var("CUBI_MODEL", v) };
        }
    }

    #[test]
    fn known_non_tool_capable_flags_expected_families() {
        for m in [
            "llama3.2:1b",
            "llama3.2:1b-instruct-q4_0",
            "tinyllama",
            "tinyllama:1.1b",
            "smollm2:1.7b",
            "gemma3:1b",
            "phi3:mini",
        ] {
            assert!(is_known_non_tool_capable(m), "expected flag for {m}");
        }
    }

    #[test]
    fn known_non_tool_capable_passes_tool_capable_models() {
        for m in [
            "qwen3:4b",
            "qwen2.5:3b",
            "qwen2.5:7b",
            "llama3.1:8b",
            "llama3.2:3b",
            "mistral:7b-instruct-v0.3",
            "phi4-mini",
            "gemma4",
            "gemma4:31b",
            "glm-5.2",
            "zai-org/GLM-5.2",
        ] {
            assert!(!is_known_non_tool_capable(m), "unexpected flag for {m}");
        }
    }

    #[test]
    fn clamp_compact_threshold_inside_range_is_identity() {
        assert_eq!(clamp_compact_threshold(50), 50);
        assert_eq!(clamp_compact_threshold(80), 80);
        assert_eq!(clamp_compact_threshold(95), 95);
    }

    #[test]
    fn clamp_compact_threshold_below_min_is_clipped() {
        assert_eq!(clamp_compact_threshold(0), COMPACT_THRESHOLD_MIN);
        assert_eq!(clamp_compact_threshold(49), COMPACT_THRESHOLD_MIN);
    }

    #[test]
    fn clamp_compact_threshold_above_max_is_clipped() {
        assert_eq!(clamp_compact_threshold(96), COMPACT_THRESHOLD_MAX);
        assert_eq!(clamp_compact_threshold(255), COMPACT_THRESHOLD_MAX);
    }
}
