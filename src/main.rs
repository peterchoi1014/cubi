mod agent_loop;
mod builtin_tools;
mod cli;
mod commands;
mod executor;
mod file_mentions;
mod file_rollback;
mod git_cmds;
mod hooks;
#[allow(dead_code)]
mod llm;
mod lsp_client;
mod mcp_client;
mod mcp_config;
mod mcp_manager;
mod memdir;
mod migrations;
mod ollama;
mod onboarding;
mod output_styles;
mod permissions;
pub mod plugins;
mod policy;
mod project_memory;
mod schemas;
mod sessions;
mod settings_sync;
pub mod skills;
mod telemetry;
mod themes;
mod tips;
mod todos;

use anyhow::{Context, Result};
use cli::ChatCLI;
use colored::*;
use executor::AIExecutor;
use mcp_manager::McpManager;
use onboarding::AppConfig;
use permissions::Permissions;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

/// Default model used when the user has not configured one. Can be overridden
/// at runtime by setting the `AI_CHAT_CLI_MODEL` environment variable.
const DEFAULT_MODEL: &str = "llama3.2:1b";

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize permissions early — the onboarding wizard may want to
    // mutate it when the user trusts the cwd.
    let permissions = Arc::new(Mutex::new(Permissions::load()));

    // Persistent user config (model preference, onboarding flag, ...).
    let mut config = AppConfig::load();

    // Apply forward-only config migrations and persist if anything
    // changed (e.g. first time this binary saw the file).
    if migrations::migrate_config(&mut config)
        && let Err(e) = config.save()
    {
        eprintln!(
            "{} could not persist migrated config: {}",
            "Warn:".bright_yellow(),
            e
        );
    }

    // Initialise telemetry early so onboarding events can be recorded.
    telemetry::init(config.telemetry);

    // Apply persisted UI prefs from config (theme/output-style/color/vim)
    // into the env-var slots that the rest of the CLI already reads.
    if let Some(t) = &config.theme {
        // SAFETY: single-threaded during startup.
        unsafe { std::env::set_var("AICHAT_THEME", t) };
    }
    if let Some(s) = &config.output_style {
        unsafe { std::env::set_var("AICHAT_OUTPUT_STYLE", s) };
    }
    if let Some(c) = &config.color {
        unsafe { std::env::set_var("AICHAT_COLOR", c) };
        match c.as_str() {
            "off" => colored::control::set_override(false),
            "on" => colored::control::set_override(true),
            _ => {}
        }
    }
    if let Some(v) = &config.vim_mode {
        unsafe { std::env::set_var("AICHAT_VIM_MODE", v) };
    }

    // First-run wizard. No-ops if already onboarded, in non-interactive
    // shells, or when `AI_CHAT_CLI_NO_ONBOARD=1` is set.
    let ollama_client = ollama::OllamaClient::new();
    if let Err(e) = onboarding::run_if_needed(&mut config, &ollama_client, &permissions).await {
        eprintln!(
            "{} onboarding wizard failed: {} (continuing with defaults)",
            "Warn:".bright_yellow(),
            e
        );
    }

    // Resolve the model from env > config > baked-in fallback. This
    // removes the previous hard-coded lock-in.
    let model_owned = onboarding::resolve_model(&config, DEFAULT_MODEL);
    let model: &str = &model_owned;
    let cpu_workers = 6;

    println!("{}", "Initializing AI Chat CLI...".bright_cyan());

    // Shared plan-mode flag, observed by built-in write/exec tools.
    let plan_mode = Arc::new(AtomicBool::new(false));

    // Create executor before provider-specific startup checks.
    let executor = AIExecutor::new(model.to_string(), cpu_workers)
        .await
        .context("Failed to create AI executor")?;

    if executor.provider_name() == "openai" {
        let base_url = std::env::var("OPENAI_BASE_URL")
            .ok()
            .or_else(|| std::env::var("AI_CHAT_CLI_BASE_URL").ok())
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        println!(
            "{} Using OpenAI-compatible provider at {}",
            "✓".bright_green(),
            base_url.bright_cyan()
        );
        println!(
            "{} Using model: {}",
            "✓".bright_green(),
            model.bright_cyan()
        );
    } else {
        match ollama_client.list_models().await {
            Ok(models) => {
                println!(
                    "{} {}",
                    "✓".bright_green(),
                    "Connected to Ollama".bright_white()
                );

                if !models.iter().any(|m| m.starts_with(model)) {
                    eprintln!(
                        "{} Model '{}' not found. Available models: {:?}",
                        "Warning:".bright_yellow(),
                        model,
                        models
                    );
                    eprintln!(
                        "\nInstall the model with: {}",
                        format!("ollama pull {}", model).bright_cyan()
                    );
                    std::process::exit(1);
                }

                println!(
                    "{} Using model: {}",
                    "✓".bright_green(),
                    model.bright_cyan()
                );
            }
            Err(e) => {
                eprintln!("{} {}", "Error:".bright_red().bold(), e);
                eprintln!("\n{}", "Make sure Ollama is running:".bright_yellow());
                eprintln!("  {}", "ollama serve".bright_cyan());
                std::process::exit(1);
            }
        }
    }

    // Initialize MCP. We hand it a shared FileJournal so the CLI's
    // `/rewind` can roll back any `edit_file`/`write_file` mutations
    // recorded by the built-in tool registry.
    let journal = file_rollback::FileJournal::default();
    let mcp_manager = match McpManager::new_with_journal(
        Arc::clone(&permissions),
        Arc::clone(&plan_mode),
        journal.clone(),
    )
    .await
    {
        Ok(manager) => {
            if manager.has_tools() {
                let tool_count = manager.list_tools().len();
                println!("{} Loaded {} MCP tool(s)", "✓".bright_green(), tool_count);
                Some(manager)
            } else {
                println!(
                    "{} No MCP tools configured (create ~/.ai-chat-cli/mcp.json)",
                    "ℹ".bright_blue()
                );
                None
            }
        }
        Err(e) => {
            eprintln!(
                "{} Failed to initialize MCP: {}",
                "Warning:".bright_yellow(),
                e
            );
            None
        }
    };

    println!("{} AI executor ready", "✓".bright_green());

    // Tip-of-the-day banner. Suppressed in non-TTY contexts so logs
    // stay quiet under CI.
    if std::io::IsTerminal::is_terminal(&std::io::stdout())
        && let Some(tip) = tips::tip_of_the_day()
    {
        println!("{} {}", "💡 tip:".bright_yellow(), tip);
    }

    // Create and run CLI
    let mut cli = ChatCLI::new(executor, mcp_manager, permissions, plan_mode, journal);
    let run_result = cli.run().await;

    // Shut down MCP cleanly while we still have an async context. The Drop
    // impl is only a best-effort fallback and intentionally does not spin up
    // a nested runtime.
    cli.shutdown().await;

    run_result?;

    Ok(())
}
