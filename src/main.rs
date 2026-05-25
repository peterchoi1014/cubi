mod builtin_tools;
mod cli;
mod commands;
mod executor;
mod mcp_client;
mod mcp_config;
mod mcp_manager;
mod ollama;
mod project_memory;
mod todos;

use anyhow::{Context, Result};
use cli::ChatCLI;
use colored::*;
use executor::AIExecutor;
use mcp_manager::McpManager;

/// Default model used when the user has not configured one. Can be overridden
/// at runtime by setting the `AI_CHAT_CLI_MODEL` environment variable.
const DEFAULT_MODEL: &str = "llama3.2:1b";

#[tokio::main]
async fn main() -> Result<()> {
    // Resolve the model from $AI_CHAT_CLI_MODEL, falling back to DEFAULT_MODEL.
    // This removes the previous hard-coded lock-in and is the first small
    // step toward the configurable onboarding flow tracked in ROADMAP.md.
    let model_owned =
        std::env::var("AI_CHAT_CLI_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let model: &str = &model_owned;
    let cpu_workers = 6;

    println!("{}", "Initializing AI Chat CLI...".bright_cyan());

    // Check if Ollama is running
    let client = ollama::OllamaClient::new();
    match client.list_models().await {
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

    // Initialize MCP
    let mcp_manager = match McpManager::new().await {
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
    let mut cli = ChatCLI::new(executor, mcp_manager);
    let run_result = cli.run().await;

    // Shut down MCP cleanly while we still have an async context. The Drop
    // impl is only a best-effort fallback and intentionally does not spin up
    // a nested runtime.
    cli.shutdown().await;

    run_result?;

    Ok(())
}
