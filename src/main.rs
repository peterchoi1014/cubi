mod agent_loop;
mod builtin_tools;
mod cli;
mod commands;
mod compat;
mod completer;
mod completions;
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
mod oauth;
mod ollama;
mod onboarding;
mod out;
mod output_styles;
mod permissions;
pub mod plugins;
mod policy;
mod project_memory;
mod schemas;
mod sessions;
mod settings_sync;
pub mod skills;
mod spinner;
mod style;
mod telemetry;
mod themes;
mod tips;
mod todos;

use crate::style::CubiStyle;
use anyhow::{Context, Result};
use cli::ChatCLI;
use executor::AIExecutor;
use mcp_manager::McpManager;
use onboarding::AppConfig;
use permissions::Permissions;
use sessions::{DeleteSessionResult, SessionStore};
use std::io::{IsTerminal, Read};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

/// Default model used when the user has not configured one. Can be overridden
/// at runtime by setting the `CUBI_MODEL` environment variable.
///
/// Picked because Qwen3 4B currently has the best native tool-calling
/// reliability of any small (<5B) model on Ollama — important because the
/// agent loop in `agent_loop.rs` advertises ~27 built-in tools plus any MCP
/// tools via Ollama's `tools:` field. Tiny non-tool-trained models (the
/// previous `llama3.2:1b` default) routinely garbled their replies into
/// pseudo-JSON instead of either calling a tool or answering normally.
const DEFAULT_MODEL: &str = "qwen3:4b";

#[tokio::main]
async fn main() -> Result<()> {
    // Lightweight argv handling. We don't pull in clap because the chat
    // loop has no flags of its own; this just makes `cubi --version`,
    // `cubi --help`, and `cubi --resume [id]` Do What People Expect
    // instead of dropping them straight into the REPL. Use `args_os()`
    // so non-UTF-8 argv can't panic the binary.
    let argv: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    let mut primary = PrimaryCommand::Interactive;
    let mut one_shot_prompt: Option<String> = None;
    let mut cli_flags = cli::CliFlags::default();
    let mut stream_explicit = false;

    let mut i = 0;
    while i < argv.len() {
        let Some(arg) = argv[i].to_str() else {
            eprintln!("cubi: arguments must be valid UTF-8. Run `cubi --help` for usage.");
            std::process::exit(2);
        };
        match arg {
            "--no-stream" => {
                cli_flags.stream = false;
                stream_explicit = true;
            }
            "--stream" => {
                cli_flags.stream = true;
                stream_explicit = true;
            }
            "--no-markdown" => cli_flags.markdown = false,
            "--markdown" => cli_flags.markdown = true,
            "--show-stats-footer" => cli_flags.stats_footer = true,
            "--version" | "-V" | "-v" | "version" => {
                println!("cubi {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "--help" | "-h" | "help" => {
                print_help();
                return Ok(());
            }
            "--prompt" | "-p" => {
                i += 1;
                let Some(value) = argv.get(i).and_then(|a| a.to_str()) else {
                    eprintln!(
                        "cubi: {arg} requires inline prompt text. Use stdin without -p for piped prompts."
                    );
                    std::process::exit(2);
                };
                set_prompt(&mut one_shot_prompt, value.to_string());
            }
            _ if arg.starts_with("--prompt=") => {
                set_prompt(
                    &mut one_shot_prompt,
                    arg.trim_start_matches("--prompt=").to_string(),
                );
            }
            "completions" => {
                let Some(shell) = argv.get(i + 1).and_then(|a| a.to_str()) else {
                    eprintln!(
                        "cubi: completions requires one of: bash, zsh, fish. Run `cubi --help` for usage."
                    );
                    std::process::exit(2);
                };
                if argv.get(i + 2).is_some() {
                    eprintln!(
                        "cubi: completions requires exactly one shell argument (bash, zsh, fish)."
                    );
                    std::process::exit(2);
                }
                if let Some(script) = completions::script(shell) {
                    print!("{script}");
                    return Ok(());
                }
                eprintln!(
                    "cubi: completions requires one of: bash, zsh, fish. Run `cubi --help` for usage."
                );
                std::process::exit(2);
            }
            "--resume" | "-r" | "resume" => {
                set_primary(&mut primary, PrimaryCommand::Resume(String::new()));
                if let Some(next) = argv.get(i + 1).and_then(|a| a.to_str())
                    && (!next.starts_with('-') || next.is_empty())
                {
                    primary = PrimaryCommand::Resume(next.to_string());
                    i += 1;
                }
            }
            "--list-sessions" => set_primary(&mut primary, PrimaryCommand::ListSessions),
            "--delete-session" => {
                i += 1;
                let Some(id) = argv.get(i).and_then(|a| a.to_str()) else {
                    eprintln!("cubi: --delete-session requires a session id or unique prefix.");
                    std::process::exit(2);
                };
                set_primary(&mut primary, PrimaryCommand::DeleteSession(id.to_string()));
            }
            _ => {
                eprintln!("cubi: unrecognized argument {arg:?}. Run `cubi --help` for usage.");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    if one_shot_prompt.is_some() && !matches!(primary, PrimaryCommand::Interactive) {
        eprintln!(
            "cubi: --prompt cannot be combined with --resume, --list-sessions, or --delete-session."
        );
        std::process::exit(2);
    }

    if matches!(primary, PrimaryCommand::Interactive)
        && one_shot_prompt.is_none()
        && !std::io::stdin().is_terminal()
    {
        let mut input = String::new();
        std::io::stdin()
            .read_to_string(&mut input)
            .context("Failed to read prompt from stdin")?;
        if input.trim().is_empty() {
            eprintln!("cubi: stdin prompt was empty.");
            std::process::exit(2);
        }
        one_shot_prompt = Some(input);
    }

    match &primary {
        PrimaryCommand::ListSessions => {
            print_sessions_table()?;
            return Ok(());
        }
        PrimaryCommand::DeleteSession(id) => {
            delete_session(id)?;
            return Ok(());
        }
        PrimaryCommand::Interactive | PrimaryCommand::Resume(_) => {}
    }
    let headless = one_shot_prompt.is_some();
    if headless && !stream_explicit {
        cli_flags.stream = false;
    }

    // Rebrand back-compat: promote legacy AI_CHAT_CLI_*/AICHAT_* env vars
    // to their new CUBI_* names and rename ~/.ai-chat-cli/ → ~/.cubi/
    // exactly once. Both no-op if there's nothing to migrate.
    compat::promote_legacy_env();
    compat::migrate_config_dir();

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
        unsafe { std::env::set_var("CUBI_THEME", t) };
    }
    if let Some(s) = &config.output_style {
        unsafe { std::env::set_var("CUBI_OUTPUT_STYLE", s) };
    }
    if let Some(c) = &config.color {
        unsafe { std::env::set_var("CUBI_COLOR", c) };
        match c.as_str() {
            "off" => colored::control::set_override(false),
            "on" => colored::control::set_override(true),
            _ => {}
        }
    }
    if let Some(v) = &config.vim_mode {
        unsafe { std::env::set_var("CUBI_VIM_MODE", v) };
    }
    style::init_color_control();

    // First-run wizard. No-ops if already onboarded, in non-interactive
    // shells, or when `CUBI_NO_ONBOARD=1` is set.
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

    status_line(
        headless,
        format!("{}", "Initializing Cubi...".bright_cyan()),
    );

    // Shared plan-mode flag, observed by built-in write/exec tools.
    let plan_mode = Arc::new(AtomicBool::new(false));

    // Create executor before provider-specific startup checks.
    let executor = AIExecutor::new(model.to_string(), cpu_workers)
        .await
        .context("Failed to create AI executor")?;

    if executor.provider_name() == "openai" {
        let base_url = std::env::var("OPENAI_BASE_URL")
            .ok()
            .or_else(|| std::env::var("CUBI_BASE_URL").ok())
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        status_line(
            headless,
            format!(
                "{} Using OpenAI-compatible provider at {}",
                "✓".bright_green(),
                base_url.bright_cyan()
            ),
        );
        status_line(
            headless,
            format!(
                "{} Using model: {}",
                "✓".bright_green(),
                model.bright_cyan()
            ),
        );
    } else {
        match ollama_client.list_models().await {
            Ok(models) => {
                status_line(
                    headless,
                    format!(
                        "{} {}",
                        "✓".bright_green(),
                        "Connected to Ollama".bright_white()
                    ),
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

                status_line(
                    headless,
                    format!(
                        "{} Using model: {}",
                        "✓".bright_green(),
                        model.bright_cyan()
                    ),
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

    // Warn when the active model is known to not reliably support native
    // tool calling. The agent loop in `agent_loop.rs` advertises ~27
    // built-in tools plus any MCP tools, and small chat-only models tend
    // to echo schemas back as content instead of emitting `tool_calls`.
    if onboarding::is_known_non_tool_capable(model) {
        eprintln!(
            "{} Model '{}' is not known to reliably support tool calling. \
             Responses may be malformed when tools are attached. \
             Consider switching to {} (best), {}, or {}.",
            "Warning:".bright_yellow(),
            model.bright_cyan(),
            "qwen3:4b".bright_cyan(),
            "qwen2.5:3b".bright_cyan(),
            "phi4-mini".bright_cyan(),
        );
    }

    // Initialize MCP. We hand it a shared FileJournal so the CLI's
    // `/rewind` can roll back any `edit_file`/`write_file` mutations
    // recorded by the built-in tool registry.
    let journal = file_rollback::FileJournal::default();
    let mcp_manager = match McpManager::new_with_journal_quiet(
        Arc::clone(&permissions),
        Arc::clone(&plan_mode),
        journal.clone(),
        headless,
    )
    .await
    {
        Ok(manager) => {
            if manager.has_tools() {
                let tool_count = manager.list_tools().len();
                status_line(
                    headless,
                    format!("{} Loaded {} MCP tool(s)", "✓".bright_green(), tool_count),
                );
                Some(manager)
            } else {
                status_line(
                    headless,
                    format!(
                        "{} No MCP tools configured (create ~/.cubi/mcp.json)",
                        "ℹ".bright_blue()
                    ),
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

    status_line(
        headless,
        format!("{} AI executor ready", "✓".bright_green()),
    );

    // Tip-of-the-day banner. Suppressed in non-TTY contexts so logs
    // stay quiet under CI.
    if !headless
        && std::io::IsTerminal::is_terminal(&std::io::stdout())
        && let Some(tip) = tips::tip_of_the_day()
    {
        println!("{} {}", "💡 tip:".bright_yellow(), tip);
    }

    // Create and run CLI
    let mut cli = ChatCLI::new_with_flags(
        executor,
        mcp_manager,
        permissions,
        plan_mode,
        journal,
        cli_flags,
    );
    if let PrimaryCommand::Resume(target) = &primary {
        cli.resume_session(target);
    }
    let run_result = if let Some(prompt) = one_shot_prompt {
        cli.run_one_shot(&prompt).await
    } else {
        cli.run().await
    };

    // Shut down MCP cleanly while we still have an async context. The Drop
    // impl is only a best-effort fallback and intentionally does not spin up
    // a nested runtime.
    cli.shutdown().await;

    run_result?;

    Ok(())
}

#[derive(Debug)]
enum PrimaryCommand {
    Interactive,
    Resume(String),
    ListSessions,
    DeleteSession(String),
}

fn set_prompt(slot: &mut Option<String>, value: String) {
    if value.trim().is_empty() {
        eprintln!("cubi: --prompt/-p requires non-empty inline prompt text.");
        std::process::exit(2);
    }
    if slot.replace(value).is_some() {
        eprintln!("cubi: --prompt/-p may only be provided once.");
        std::process::exit(2);
    }
}

fn set_primary(slot: &mut PrimaryCommand, value: PrimaryCommand) {
    if !matches!(slot, PrimaryCommand::Interactive) {
        eprintln!("cubi: only one command may be provided. Run `cubi --help` for usage.");
        std::process::exit(2);
    }
    *slot = value;
}

fn status_line(headless: bool, msg: impl std::fmt::Display) {
    out::status_line(headless, msg);
}

fn print_help() {
    println!(
        "cubi {} — a pocket-sized AI for your shell\n\n\
         USAGE:\n  cubi                         Start the interactive chat REPL\n  \
         cubi -p <prompt>             Run one prompt, print the reply, and exit\n  \
         cubi --prompt <prompt>       Same as -p\n  \
         echo <prompt> | cubi         Read a one-shot prompt from stdin\n  \
         cubi --resume [<id>]         Resume a prior chat (most recent in this\n  \
                                      directory if no id is given; falls back to\n  \
                                      global latest)\n  \
         cubi --list-sessions         List saved sessions newest-first\n  \
         cubi --delete-session <id>   Delete by full id or unique prefix\n  \
         cubi completions <shell>     Print a completion script (bash, zsh, fish)\n  \
         cubi --version               Print version and exit\n  \
         cubi --help                  Print this help and exit\n\n\
         OUTPUT FLAGS (can be combined with chat commands):\n  \
         --stream / --no-stream         Stream tokens live (default) or wait\n  \
                                         for the full reply\n  \
         --markdown / --no-markdown     Enable / disable markdown rendering\n  \
                                         (markdown only applies in --no-stream\n  \
                                         mode; auto-disabled for non-TTY stdout)\n  \
         --show-stats-footer            Print a token/timing footer after\n  \
                                         each reply\n\n\
         Notes:\n  -p/--prompt requires inline text and does not read stdin. Without -p,\n  \
         piped stdin becomes the one-shot prompt. One-shot mode buffers by default;\n  \
         pass --stream to stream tokens.\n\n\
         Once inside the REPL, type /help to list slash commands.",
        env!("CARGO_PKG_VERSION")
    );
}

fn print_sessions_table() -> Result<()> {
    let sessions = SessionStore::list_all()?;
    println!("{:<24} {:<12} {:>5} CWD", "ID", "MTIME", "MSGS");
    if sessions.is_empty() {
        println!("(no sessions saved yet)");
        return Ok(());
    }
    let width = terminal_width();
    for meta in sessions {
        let fixed = 24 + 1 + 12 + 1 + 5 + 1;
        let cwd_width = width.saturating_sub(fixed).max(20);
        println!(
            "{:<24} {:<12} {:>5} {}",
            meta.id,
            format_mtime(meta.modified_at),
            meta.message_count,
            truncate_display(&meta.cwd, cwd_width)
        );
    }
    Ok(())
}

fn delete_session(id: &str) -> Result<()> {
    match SessionStore::delete_by_prefix(id)? {
        DeleteSessionResult::Deleted(meta) => {
            println!("Deleted session {}", meta.id);
            Ok(())
        }
        DeleteSessionResult::NotFound => {
            eprintln!("cubi: no session matches '{id}'.");
            std::process::exit(2);
        }
        DeleteSessionResult::Ambiguous(candidates) => {
            eprintln!("cubi: session prefix '{id}' is ambiguous. Candidates:");
            for meta in candidates {
                eprintln!("  {}  {}", meta.id, meta.cwd);
            }
            std::process::exit(2);
        }
    }
}

fn terminal_width() -> usize {
    std::env::var("COLUMNS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|w| *w >= 40)
        .unwrap_or(100)
}

fn truncate_display(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let keep = max_chars.saturating_sub(1);
    let mut out: String = s.chars().take(keep).collect();
    out.push('…');
    out
}

fn format_mtime(secs: u64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let age = now.saturating_sub(secs);
    if age < 60 {
        return "now".to_string();
    }
    if age < 3_600 {
        return format!("{}m ago", age / 60);
    }
    if age < 86_400 {
        return format!("{}h ago", age / 3_600);
    }
    if age < 7 * 86_400 {
        return format!("{}d ago", age / 86_400);
    }
    format_session_time(secs)
}

fn format_session_time(secs: u64) -> String {
    let (y, m, d, hour, minute, _) = crate::sessions::civil_from_unix(secs);
    format!("{:04}-{:02}-{:02} {:02}:{:02}", y, m, d, hour, minute)
}
