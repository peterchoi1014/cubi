mod agent_loop;
mod builtin_tools;
mod cli;
mod commands;
mod executor;
mod git_cmds;
mod mcp_client;
mod mcp_config;
mod mcp_manager;
mod ollama;
mod onboarding;
mod permissions;
mod project_memory;
mod sessions;
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

    // Check if Ollama is running
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

    // Shared plan-mode flag, observed by built-in write/exec tools.
    let plan_mode = Arc::new(AtomicBool::new(false));

    // Initialize MCP
    let mcp_manager = match McpManager::new(Arc::clone(&permissions), Arc::clone(&plan_mode)).await
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

    // Create executor
    let executor = AIExecutor::new(model.to_string(), cpu_workers)
        .await
        .context("Failed to create AI executor")?;

    println!("{} AI executor ready", "✓".bright_green());

    // Create and run CLI
    let mut cli = ChatCLI::new(executor, mcp_manager, permissions, plan_mode);
    let run_result = cli.run().await;

    // Shut down MCP cleanly while we still have an async context. The Drop
    // impl is only a best-effort fallback and intentionally does not spin up
    // a nested runtime.
    cli.shutdown().await;

    run_result?;

    Ok(())
}
