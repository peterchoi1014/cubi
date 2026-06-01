mod agent_loop;
#[cfg(feature = "browser")]
mod browser_tool;
mod builtin_tools;
mod cli;
mod commands;
mod compat;
mod completer;
mod completions;
mod doctor;
mod event_sink;
mod executor;
mod exit_code;
mod file_mentions;
mod file_rollback;
mod git_cmds;
mod hooks;
mod json_events;
#[allow(dead_code)]
mod llm;
mod lsp_client;
mod mcp_cli;
mod mcp_client;
mod mcp_config;
mod mcp_manager;
mod mcp_registry;
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
mod pricing;
mod project_memory;
mod repomap;
mod schemas;
mod script;
mod sessions;
mod sessions_diff;
mod settings_sync;
pub mod skills;
mod spinner;
mod style;
mod telemetry;
mod themes;
mod thinking_filter;
mod tips;
mod todos;
mod trace_tools;
#[allow(dead_code)]
mod user_error;

use crate::style::CubiStyle;
use anyhow::{Context, Result};
use cli::ChatCLI;
use executor::AIExecutor;
use exit_code::ExitCode;
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
// Default model when neither $CUBI_MODEL nor config.default_model is set.
// Bumped from qwen3:4b to qwen3:8b: the 8B variant is materially more
// reliable at multi-turn native tool calling (which the agent loop
// depends on) while still fitting in <6GB RAM. The 4B fallback is still
// advertised below for users on smaller machines.
const DEFAULT_MODEL: &str = "qwen3:8b";

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

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
    let mut prune_older_than: Option<u64> = None;
    let mut dry_run = false;
    // When `cubi run <file.md>` is used, the parsed transcript is
    // stashed here and applied (prefill history, tools toggle) after
    // ChatCLI is constructed but before the final prompt is sent.
    let mut run_script: Option<script::RunScript> = None;
    // `--trace-tools <path>` (or CUBI_TRACE_TOOLS env var) JSONL audit
    // log target. Stored as a String so we can hand it to ToolTracer
    // once everything else is parsed.
    let mut trace_tools_path: Option<String> = None;
    let mut events_path: Option<String> = None;
    let mut doctor_fix = false;

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
            "--json" => cli_flags.json = true,
            "--system" => {
                i += 1;
                let Some(path) = argv.get(i).and_then(|a| a.to_str()) else {
                    eprintln!("cubi: --system requires a file path.");
                    std::process::exit(2);
                };
                match std::fs::read_to_string(path) {
                    Ok(prompt) => cli_flags.system_prompt = Some(prompt),
                    Err(err) => {
                        eprintln!("cubi: failed to read --system file '{}': {}", path, err);
                        std::process::exit(2);
                    }
                }
            }
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
            "plugins" => {
                let Some(subcommand) = argv.get(i + 1).and_then(|a| a.to_str()) else {
                    eprintln!(
                        "cubi: plugins requires one of: list, reload, new, show, remove, run."
                    );
                    std::process::exit(2);
                };
                match subcommand {
                    "list" => {
                        // Optional `--json`.
                        let mut json = false;
                        let mut consumed = 1usize;
                        if let Some(next) = argv.get(i + 2).and_then(|a| a.to_str()) {
                            if next == "--json" {
                                json = true;
                                consumed = 2;
                            } else {
                                eprintln!(
                                    "cubi: plugins list takes no positional arguments (only --json)."
                                );
                                std::process::exit(2);
                            }
                        }
                        set_primary(
                            &mut primary,
                            if json {
                                PrimaryCommand::PluginsListJson
                            } else {
                                PrimaryCommand::PluginsList
                            },
                        );
                        i += consumed;
                    }
                    "reload" => {
                        if argv.get(i + 2).is_some() {
                            eprintln!("cubi: plugins reload does not accept extra arguments.");
                            std::process::exit(2);
                        }
                        set_primary(&mut primary, PrimaryCommand::PluginsReload);
                        i += 1;
                    }
                    "new" => {
                        let Some(name) = argv.get(i + 2).and_then(|a| a.to_str()) else {
                            eprintln!("cubi: plugins new requires a plugin name.");
                            std::process::exit(2);
                        };
                        if argv.get(i + 3).is_some() {
                            eprintln!(
                                "cubi: plugins new takes exactly one argument (the plugin name)."
                            );
                            std::process::exit(2);
                        }
                        set_primary(&mut primary, PrimaryCommand::PluginsNew(name.to_string()));
                        i += 2;
                    }
                    "show" => {
                        let Some(name) = argv.get(i + 2).and_then(|a| a.to_str()) else {
                            eprintln!("cubi: plugins show requires a plugin name.");
                            std::process::exit(2);
                        };
                        let mut json = false;
                        let mut consumed = 2usize;
                        if let Some(extra) = argv.get(i + 3).and_then(|a| a.to_str()) {
                            if extra == "--json" {
                                json = true;
                                consumed = 3;
                            } else {
                                eprintln!(
                                    "cubi: plugins show takes a plugin name (and optional --json)."
                                );
                                std::process::exit(2);
                            }
                        }
                        set_primary(
                            &mut primary,
                            PrimaryCommand::PluginsShow(name.to_string(), json),
                        );
                        i += consumed;
                    }
                    "remove" | "rm" => {
                        // Accept flags in any order between the
                        // subcommand and the plugin name. We accept at
                        // most one name.
                        let mut name: Option<String> = None;
                        let mut force = false;
                        let mut yes = false;
                        let mut j = i + 2;
                        while let Some(arg) = argv.get(j).and_then(|a| a.to_str()) {
                            match arg {
                                "--force" => force = true,
                                "--yes" => yes = true,
                                _ if arg.starts_with('-') => {
                                    eprintln!("cubi: plugins remove: unknown flag {arg:?}");
                                    std::process::exit(2);
                                }
                                _ => {
                                    if name.is_some() {
                                        eprintln!(
                                            "cubi: plugins remove takes exactly one plugin name."
                                        );
                                        std::process::exit(2);
                                    }
                                    name = Some(arg.to_string());
                                }
                            }
                            j += 1;
                        }
                        let Some(name) = name else {
                            eprintln!("cubi: plugins remove requires a plugin name.");
                            std::process::exit(2);
                        };
                        set_primary(
                            &mut primary,
                            PrimaryCommand::PluginsRemove { name, force, yes },
                        );
                        i = j - 1;
                    }
                    "run" => {
                        let Some(name) = argv.get(i + 2).and_then(|a| a.to_str()) else {
                            eprintln!("cubi: plugins run requires a plugin name.");
                            std::process::exit(2);
                        };
                        let mut run_args: Vec<String> = Vec::new();
                        let mut json = false;
                        let mut j = i + 3;
                        while let Some(arg) = argv.get(j) {
                            let s = arg.to_string_lossy().into_owned();
                            if s == "--json" {
                                json = true;
                            } else {
                                run_args.push(s);
                            }
                            j += 1;
                        }
                        set_primary(
                            &mut primary,
                            PrimaryCommand::PluginsRun {
                                name: name.to_string(),
                                args: run_args,
                                json,
                            },
                        );
                        i = j - 1;
                    }
                    _ => {
                        eprintln!(
                            "cubi: plugins requires one of: list, reload, new, show, remove, run."
                        );
                        std::process::exit(2);
                    }
                }
            }
            "mcp" => {
                let Some(subcommand) = argv.get(i + 1).and_then(|a| a.to_str()) else {
                    eprintln!("cubi: mcp requires one of: test, search, install, uninstall.");
                    std::process::exit(2);
                };
                match subcommand {
                    "test" => {
                        let Some(server) = argv.get(i + 2).and_then(|a| a.to_str()) else {
                            eprintln!("cubi: mcp test requires a server name.");
                            std::process::exit(2);
                        };
                        let mut tool: Option<String> = None;
                        let mut json = false;
                        let mut j = i + 3;
                        while let Some(arg) = argv.get(j).and_then(|a| a.to_str()) {
                            match arg {
                                "--json" => json = true,
                                "--tool" => {
                                    j += 1;
                                    let Some(t) = argv.get(j).and_then(|a| a.to_str()) else {
                                        eprintln!("cubi: mcp test --tool requires a tool name.");
                                        std::process::exit(2);
                                    };
                                    tool = Some(t.to_string());
                                }
                                _ if arg.starts_with("--tool=") => {
                                    tool = Some(arg.trim_start_matches("--tool=").to_string());
                                }
                                _ => {
                                    eprintln!("cubi: mcp test: unexpected argument {arg:?}");
                                    std::process::exit(2);
                                }
                            }
                            j += 1;
                        }
                        set_primary(
                            &mut primary,
                            PrimaryCommand::McpTest {
                                server: server.to_string(),
                                tool,
                                json,
                            },
                        );
                        i = j - 1;
                    }
                    "search" => {
                        let mut query = String::new();
                        let mut json = false;
                        let mut j = i + 2;
                        while let Some(arg) = argv.get(j).and_then(|a| a.to_str()) {
                            match arg {
                                "--json" => json = true,
                                _ if arg.starts_with("--") => {
                                    eprintln!("cubi: mcp search: unexpected argument {arg:?}");
                                    std::process::exit(2);
                                }
                                _ => {
                                    if !query.is_empty() {
                                        query.push(' ');
                                    }
                                    query.push_str(arg);
                                }
                            }
                            j += 1;
                        }
                        set_primary(&mut primary, PrimaryCommand::McpSearch { query, json });
                        i = j - 1;
                    }
                    "install" => {
                        let Some(name) = argv.get(i + 2).and_then(|a| a.to_str()) else {
                            eprintln!("cubi: mcp install requires a server name.");
                            std::process::exit(2);
                        };
                        let mut force = false;
                        let mut json = false;
                        let mut envs: Vec<(String, String)> = Vec::new();
                        let mut j = i + 3;
                        while let Some(arg) = argv.get(j).and_then(|a| a.to_str()) {
                            match arg {
                                "--force" => force = true,
                                "--json" => json = true,
                                "--env" => {
                                    j += 1;
                                    let Some(kv) = argv.get(j).and_then(|a| a.to_str()) else {
                                        eprintln!("cubi: mcp install --env requires KEY=VALUE.");
                                        std::process::exit(2);
                                    };
                                    let Some((k, v)) = kv.split_once('=') else {
                                        eprintln!(
                                            "cubi: mcp install --env expects KEY=VALUE, got {kv:?}."
                                        );
                                        std::process::exit(2);
                                    };
                                    envs.push((k.to_string(), v.to_string()));
                                }
                                _ if arg.starts_with("--env=") => {
                                    let kv = arg.trim_start_matches("--env=");
                                    let Some((k, v)) = kv.split_once('=') else {
                                        eprintln!(
                                            "cubi: mcp install --env expects KEY=VALUE, got {kv:?}."
                                        );
                                        std::process::exit(2);
                                    };
                                    envs.push((k.to_string(), v.to_string()));
                                }
                                _ => {
                                    eprintln!("cubi: mcp install: unexpected argument {arg:?}");
                                    std::process::exit(2);
                                }
                            }
                            j += 1;
                        }
                        set_primary(
                            &mut primary,
                            PrimaryCommand::McpInstall {
                                name: name.to_string(),
                                force,
                                json,
                                envs,
                            },
                        );
                        i = j - 1;
                    }
                    "uninstall" => {
                        let Some(name) = argv.get(i + 2).and_then(|a| a.to_str()) else {
                            eprintln!("cubi: mcp uninstall requires a server name.");
                            std::process::exit(2);
                        };
                        let mut json = false;
                        let mut j = i + 3;
                        while let Some(arg) = argv.get(j).and_then(|a| a.to_str()) {
                            match arg {
                                "--json" => json = true,
                                _ => {
                                    eprintln!("cubi: mcp uninstall: unexpected argument {arg:?}");
                                    std::process::exit(2);
                                }
                            }
                            j += 1;
                        }
                        set_primary(
                            &mut primary,
                            PrimaryCommand::McpUninstall {
                                name: name.to_string(),
                                json,
                            },
                        );
                        i = j - 1;
                    }
                    _ => {
                        eprintln!("cubi: mcp requires one of: test, search, install, uninstall.");
                        std::process::exit(2);
                    }
                }
            }
            "--no-banner" => cli_flags.no_banner = true,
            "--debug" => {
                user_error::set_debug_flag(true);
            }
            "--trace-tools" => {
                i += 1;
                let Some(path) = argv.get(i).and_then(|a| a.to_str()) else {
                    eprintln!("cubi: --trace-tools requires a file path.");
                    std::process::exit(2);
                };
                trace_tools_path = Some(path.to_string());
            }
            _ if arg.starts_with("--trace-tools=") => {
                trace_tools_path = Some(arg.trim_start_matches("--trace-tools=").to_string());
            }
            "--events" => {
                i += 1;
                let Some(path) = argv.get(i).and_then(|a| a.to_str()) else {
                    eprintln!("cubi: --events requires a file path.");
                    std::process::exit(2);
                };
                events_path = Some(path.to_string());
            }
            _ if arg.starts_with("--events=") => {
                events_path = Some(arg.trim_start_matches("--events=").to_string());
            }
            "--explain-tools" => {
                cli_flags.explain_tools = true;
            }
            "--usage-footer" => {
                cli_flags.usage_footer = true;
            }
            "--quiet" => {
                cli_flags.quiet = true;
                cli_flags.no_banner = true;
            }
            "--print-config" => set_primary(&mut primary, PrimaryCommand::PrintConfig),
            "run" => {
                i += 1;
                let Some(path) = argv.get(i).and_then(|a| a.to_str()) else {
                    eprintln!(
                        "cubi: run requires a markdown script path. Usage: cubi run <file.md>"
                    );
                    std::process::exit(2);
                };
                let script = match script::load(path) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("cubi: failed to load run script '{}': {:#}", path, e);
                        std::process::exit(2);
                    }
                };
                // Apply frontmatter overrides: model via env so
                // resolve_model picks it up; system via cli_flags.
                if let Some(model) = &script.model {
                    // SAFETY: argv parsing is single-threaded.
                    unsafe { std::env::set_var("CUBI_MODEL", model) };
                }
                if let Some(sys) = &script.system {
                    cli_flags.system_prompt = Some(sys.clone());
                }
                set_prompt(&mut one_shot_prompt, script.prompt.clone());
                cli_flags.json = true;
                cli_flags.stream = false;
                stream_explicit = true;
                cli_flags.no_banner = true;
                run_script = Some(script);
            }
            "exec" => {
                // `cubi exec <prompt words>` — script-friendly one-shot
                // shorthand for `cubi -p "<joined>" --json --no-stream
                // --no-banner`. Remaining argv (everything after `exec`)
                // is joined with single spaces to form the prompt.
                let rest: Vec<String> = argv[i + 1..]
                    .iter()
                    .map(|a| a.to_string_lossy().into_owned())
                    .collect();
                if rest.is_empty() {
                    eprintln!("cubi: exec requires a prompt. Usage: cubi exec <prompt>");
                    std::process::exit(2);
                }
                let joined = rest.join(" ");
                set_prompt(&mut one_shot_prompt, joined);
                cli_flags.json = true;
                cli_flags.stream = false;
                stream_explicit = true;
                cli_flags.no_banner = true;
                // Everything after `exec` was consumed as prompt text.
                break;
            }
            "doctor" => {
                set_primary(&mut primary, PrimaryCommand::Doctor);
            }
            "--fix" => {
                doctor_fix = true;
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
                if let Some(next) = argv.get(i + 1).and_then(|a| a.to_str()) {
                    if !next.starts_with('-') || next.is_empty() {
                        primary = PrimaryCommand::Resume(next.to_string());
                        i += 1;
                    }
                }
            }
            "--list-sessions" => set_primary(&mut primary, PrimaryCommand::ListSessions),
            "--prune-sessions" => set_primary(&mut primary, PrimaryCommand::PruneSessions),
            "--older-than" => {
                i += 1;
                let Some(value) = argv.get(i).and_then(|a| a.to_str()) else {
                    eprintln!("cubi: --older-than requires a duration like 30d, 2w, 6m, or 1y.");
                    std::process::exit(2);
                };
                match parse_duration_secs(value) {
                    Some(secs) => prune_older_than = Some(secs),
                    None => {
                        eprintln!(
                            "cubi: invalid --older-than duration '{value}' (use 30d, 2w, 6m, or 1y)."
                        );
                        std::process::exit(2);
                    }
                }
            }
            "--dry-run" => dry_run = true,
            "--delete-session" => {
                i += 1;
                let Some(id) = argv.get(i).and_then(|a| a.to_str()) else {
                    eprintln!("cubi: --delete-session requires a session id or unique prefix.");
                    std::process::exit(2);
                };
                set_primary(&mut primary, PrimaryCommand::DeleteSession(id.to_string()));
            }
            "--diff-sessions" => {
                let Some(a) = argv.get(i + 1).and_then(|a| a.to_str()) else {
                    eprintln!(
                        "cubi: --diff-sessions requires two session ids or unique prefixes: --diff-sessions <a> <b>."
                    );
                    std::process::exit(2);
                };
                let Some(b) = argv.get(i + 2).and_then(|a| a.to_str()) else {
                    eprintln!(
                        "cubi: --diff-sessions requires two session ids or unique prefixes: --diff-sessions <a> <b>."
                    );
                    std::process::exit(2);
                };
                if a.starts_with('-') || b.starts_with('-') {
                    eprintln!(
                        "cubi: --diff-sessions arguments must be session ids or prefixes, not flags."
                    );
                    std::process::exit(2);
                }
                set_primary(
                    &mut primary,
                    PrimaryCommand::DiffSessions {
                        a: a.to_string(),
                        b: b.to_string(),
                    },
                );
                i += 2;
            }
            _ => {
                eprintln!("cubi: unrecognized argument {arg:?}. Run `cubi --help` for usage.");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    if std::env::var("CUBI_NO_BANNER")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false)
    {
        cli_flags.no_banner = true;
    }
    if !cli_flags.quiet
        && std::env::var("CUBI_QUIET")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
    {
        cli_flags.quiet = true;
        cli_flags.no_banner = true;
    }
    if cli_flags.quiet {
        // Suppress the global "thinking…" / tool-call spinner via the
        // existing env knob so we don't have to thread a flag through
        // every spinner call site.
        // SAFETY: this runs during early argv processing in `main`,
        // before any task is spawned that reads CUBI_NO_SPINNER or
        // otherwise concurrently touches the process environment.
        // Tokio's runtime worker threads exist by this point (since
        // we're inside `#[tokio::main]`), but they're idle until we
        // hand them work below, so there are no concurrent env
        // readers racing with this set_var.
        unsafe { std::env::set_var("CUBI_NO_SPINNER", "1") };
    }
    if std::env::var("CUBI_EXPLAIN_TOOLS")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false)
    {
        cli_flags.explain_tools = true;
    }

    if doctor_fix && !matches!(primary, PrimaryCommand::Doctor) {
        eprintln!("cubi: --fix is only valid with the `doctor` subcommand.");
        std::process::exit(2);
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
        PrimaryCommand::Doctor => {
            let ok = doctor::run(cli_flags.json, doctor_fix).await;
            if !ok {
                std::process::exit(2);
            }
            return Ok(());
        }
        PrimaryCommand::PrintConfig => {
            print_config()?;
            return Ok(());
        }
        PrimaryCommand::ListSessions => {
            print_sessions(cli_flags.json)?;
            return Ok(());
        }
        PrimaryCommand::DeleteSession(id) => {
            delete_session(id)?;
            return Ok(());
        }
        PrimaryCommand::DiffSessions { a, b } => {
            run_diff_sessions(a.as_str(), b.as_str(), cli_flags.json)?;
            return Ok(());
        }
        PrimaryCommand::PluginsList => {
            let plugins = plugins::load_plugins();
            plugins::print_plugin_list(&plugins);
            return Ok(());
        }
        PrimaryCommand::PluginsListJson => {
            let plugins = plugins::load_plugins();
            println!("{}", plugins::plugin_list_json(&plugins));
            return Ok(());
        }
        PrimaryCommand::PluginsShow(name, json) => {
            let plugins = plugins::load_plugins();
            if !plugins::show_plugin(&plugins, name, *json) {
                eprintln!("cubi: no plugin named '{}'.", name);
                std::process::exit(2);
            }
            return Ok(());
        }
        PrimaryCommand::PluginsRemove { name, force, yes } => {
            let parent = plugins::plugins_dir()
                .ok_or_else(|| anyhow::anyhow!("could not resolve plugins directory"))?;
            match plugins::resolve_remove_target(&parent, name, *force) {
                Ok(root) => {
                    if !*yes && !confirm_yn(&format!("Remove plugin {}?", root.display())) {
                        eprintln!("cubi: aborted.");
                        return Ok(());
                    }
                    if let Err(e) = std::fs::remove_dir_all(&root) {
                        eprintln!("cubi: failed to remove {}: {}", root.display(), e);
                        std::process::exit(2);
                    }
                    println!("+ removed {}", root.display());
                    return Ok(());
                }
                Err(plugins::RemoveError::NotFound) => {
                    eprintln!("cubi: no plugin named '{}'.", name);
                    std::process::exit(2);
                }
                Err(plugins::RemoveError::PathEscape) => {
                    eprintln!(
                        "cubi: refusing to remove '{}': path is not a child of {}.",
                        name,
                        parent.display()
                    );
                    std::process::exit(2);
                }
                Err(plugins::RemoveError::HasExtraFiles(items)) => {
                    eprintln!(
                        "cubi: refusing to remove '{}': contains unexpected files: {}. \
                         Re-run with --force.",
                        name,
                        items.join(", ")
                    );
                    std::process::exit(2);
                }
            }
        }
        PrimaryCommand::PluginsRun { name, args, json } => {
            let exit_code = run_plugin(name, args, *json).await;
            std::process::exit(exit_code);
        }
        PrimaryCommand::McpTest { server, tool, json } => {
            let code = run_mcp_test(server, tool.as_deref(), *json).await;
            std::process::exit(code);
        }
        PrimaryCommand::McpSearch { query, json } => {
            let code = mcp_cli::run_mcp_search(query, *json);
            std::process::exit(code);
        }
        PrimaryCommand::McpInstall {
            name,
            force,
            json,
            envs,
        } => {
            let code = mcp_cli::run_mcp_install(name, *force, *json, envs).await;
            std::process::exit(code);
        }
        PrimaryCommand::McpUninstall { name, json } => {
            let code = mcp_cli::run_mcp_uninstall(name, *json);
            std::process::exit(code);
        }
        PrimaryCommand::PluginsReload => {
            let before = plugins::load_plugins();
            let skills = skills::load_skills();
            let after = plugins::load_plugins();
            plugins::print_reload_summary(&before, &after, skills.len());
            return Ok(());
        }
        PrimaryCommand::PluginsNew(name) => match plugins::scaffold_new(name) {
            Ok(root) => {
                println!("✓ Scaffolded plugin '{}' at {}", name, root.display());
                println!("  Next steps:");
                println!("    1. Edit handler script in {}", root.display());
                println!("    2. Run `cubi plugins reload` to pick it up");
                println!("    3. Invoke `/{}:<command>` from the REPL", name);
                return Ok(());
            }
            Err(e) => {
                eprintln!("cubi: failed to scaffold plugin '{}': {:#}", name, e);
                std::process::exit(2);
            }
        },
        PrimaryCommand::PruneSessions => {
            let Some(age_secs) = prune_older_than else {
                eprintln!("cubi: --prune-sessions requires --older-than <duration>.");
                std::process::exit(2);
            };
            prune_sessions(age_secs, dry_run)?;
            return Ok(());
        }
        PrimaryCommand::Interactive | PrimaryCommand::Resume(_) => {}
    }
    let headless = one_shot_prompt.is_some();
    if headless && !stream_explicit {
        cli_flags.stream = cli_flags.json;
    }
    if cli_flags.json {
        cli_flags.markdown = false;
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
    if migrations::migrate_config(&mut config) {
        if let Err(e) = config.save() {
            eprintln!(
                "{} could not persist migrated config: {}",
                "Warn:".bright_yellow(),
                e
            );
        }
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

    if executor.provider_name() == "fake" {
        status_line(
            headless,
            format!("{} Using fake test provider", "✓".bright_green()),
        );
    } else if executor.provider_name() == "openai" {
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
                    exit_code::exit(if headless {
                        ExitCode::Model
                    } else {
                        ExitCode::Usage
                    });
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
                exit_code::exit(if headless {
                    ExitCode::Model
                } else {
                    ExitCode::Usage
                });
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
             Consider switching to {} (best), {} (best for code), or {} (smallest).",
            "Warning:".bright_yellow(),
            model.bright_cyan(),
            "qwen3:8b".bright_cyan(),
            "devstral".bright_cyan(),
            "qwen3:4b".bright_cyan(),
        );
    }

    // Initialize MCP. We hand it a shared FileJournal so the CLI's
    // `/rewind` can roll back any `edit_file`/`write_file` mutations
    // recorded by the built-in tool registry.
    let journal = file_rollback::FileJournal::default();
    if !headless {
        match McpManager::health_check_configured().await {
            Ok(health) => {
                let ok = health
                    .iter()
                    .filter(|h| matches!(h.state, mcp_manager::McpHealthState::Ready))
                    .count();
                let failed = health
                    .iter()
                    .filter(|h| matches!(h.state, mcp_manager::McpHealthState::Failed(_)))
                    .count();
                // `health_check_configured` returns one entry per
                // configured server, so anything not Ok or Failed
                // (currently nothing in the state machine, but future-
                // proofing) is counted as "not loaded".
                let total = health.len();
                let not_loaded = total.saturating_sub(ok + failed);
                cli_flags.mcp_counts = (ok, failed, not_loaded);
            }
            Err(e) => eprintln!(
                "{} Failed to check MCP server health: {}",
                "Warning:".bright_yellow(),
                e
            ),
        }
    }
    let mcp_manager = match McpManager::new_with_journal_quiet(
        Arc::clone(&permissions),
        Arc::clone(&plan_mode),
        journal.clone(),
        headless,
    )
    .await
    {
        Ok(mut manager) => {
            manager.set_tool_timeout_secs(config.tool_timeout_secs);
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

    // `cubi run <file.md>` honors `tools: false` by detaching the MCP
    // manager before the CLI is constructed.
    let mcp_manager = if matches!(run_script.as_ref().and_then(|s| s.tools), Some(false)) {
        None
    } else {
        mcp_manager
    };

    status_line(
        headless,
        format!("{} AI executor ready", "✓".bright_green()),
    );

    // Tip-of-the-day banner. Suppressed in non-TTY contexts so logs
    // stay quiet under CI; also suppressed by --quiet.
    if !headless
        && !cli_flags.no_banner
        && !cli_flags.quiet
        && std::io::IsTerminal::is_terminal(&std::io::stdout())
    {
        if let Some(tip) = tips::tip_of_the_day() {
            println!("{} {}", "💡 tip:".bright_yellow(), tip);
        }
    }

    let json_output = cli_flags.json;

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
    if let Some(script) = run_script.take() {
        cli.preload_history(script.prefill);
    }
    if let Some(tracer) = trace_tools::ToolTracer::from_args(trace_tools_path.as_deref()) {
        cli.set_tool_tracer(Some(Arc::new(tracer)));
    }
    if let Some(sink) = event_sink::EventSink::from_args(events_path.as_deref()) {
        match sink.probe() {
            Ok(()) => cli.set_event_sink(Some(Arc::new(sink))),
            Err(err) => {
                let summary = format!(
                    "cubi: failed to open --events path {:?}: {}",
                    sink.path(),
                    err
                );
                user_error::print_user_warning(
                    &summary,
                    Some("structured event tap disabled for this session"),
                    json_output && headless,
                );
            }
        }
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

    if let Err(err) = run_result {
        let debug = user_error::debug_mode();
        // Already a classified UserError? Use it directly.
        if err.downcast_ref::<user_error::UserError>().is_some() {
            match err.downcast::<user_error::UserError>() {
                Ok(ue) => {
                    let code = ue.exit_code;
                    user_error::report_user_error(&ue, json_output && headless, debug);
                    exit_code::exit(code);
                }
                Err(_) => unreachable!(),
            }
        }
        if let Some(exit) = err.downcast_ref::<exit_code::AppExit>() {
            // Preserve the legacy exit code & message; promote to a
            // typed UserError so JSON/human paths share one shape.
            let msg = exit.message.clone();
            let code = exit.code;
            let mut ue = user_error::UserError::new(
                kind_for_legacy_exit(code),
                if msg.is_empty() {
                    format!("exit code {}", code.as_i32())
                } else {
                    msg
                },
            );
            ue.exit_code = code;
            ue.cause = Some(err);
            user_error::report_user_error(&ue, json_output && headless, debug);
            exit_code::exit(code);
        }
        let ue = user_error::UserError::from_anyhow(err);
        let fallback_code = if headless {
            exit_code::ExitCode::Model
        } else {
            exit_code::ExitCode::Usage
        };
        let mut ue = ue;
        ue.exit_code = fallback_code;
        user_error::report_user_error(&ue, json_output && headless, debug);
        exit_code::exit(fallback_code);
    }

    Ok(())
}

#[derive(Debug)]
enum PrimaryCommand {
    Interactive,
    Resume(String),
    ListSessions,
    DeleteSession(String),
    DiffSessions {
        a: String,
        b: String,
    },
    PluginsList,
    PluginsListJson,
    PluginsReload,
    PluginsNew(String),
    PluginsShow(String, bool),
    PluginsRemove {
        name: String,
        force: bool,
        yes: bool,
    },
    PluginsRun {
        name: String,
        args: Vec<String>,
        json: bool,
    },
    McpTest {
        server: String,
        tool: Option<String>,
        json: bool,
    },
    McpSearch {
        query: String,
        json: bool,
    },
    McpInstall {
        name: String,
        force: bool,
        json: bool,
        envs: Vec<(String, String)>,
    },
    McpUninstall {
        name: String,
        json: bool,
    },
    PruneSessions,
    Doctor,
    PrintConfig,
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

/// Maps a legacy [`ExitCode`] back into the closest [`user_error::ErrorKind`]
/// so existing `AppExit`-bearing errors can flow through the
/// classified-error path uniformly.
fn kind_for_legacy_exit(code: ExitCode) -> user_error::ErrorKind {
    match code {
        ExitCode::Ok => user_error::ErrorKind::Other,
        ExitCode::Usage => user_error::ErrorKind::Config,
        ExitCode::Model => user_error::ErrorKind::Other,
        ExitCode::Tool => user_error::ErrorKind::Tool,
        ExitCode::Budget => user_error::ErrorKind::Budget,
        ExitCode::Network => user_error::ErrorKind::ConnectRefused,
        ExitCode::Cancelled => user_error::ErrorKind::Cancelled,
    }
}

fn status_line(headless: bool, msg: impl std::fmt::Display) {
    out::status_line(headless, msg);
}

/// Read a single y/N answer from stdin; defaults to `false` on any
/// non-`y` reply or read failure. Skips the prompt entirely (auto-
/// confirms with `false`) when stdin is not a terminal so headless
/// invocations don't deadlock.
fn confirm_yn(question: &str) -> bool {
    use std::io::{self, IsTerminal, Write};
    if !io::stdin().is_terminal() {
        return false;
    }
    print!("{} [y/N] ", question);
    let _ = io::stdout().flush();
    let mut buf = String::new();
    if io::stdin().read_line(&mut buf).is_err() {
        return false;
    }
    matches!(buf.trim().to_ascii_lowercase().as_str(), "y" | "yes")
}

/// Generates stub arguments for an MCP tool call from a JSON schema.
/// Used by `cubi mcp test` so users can ping every tool without
/// constructing request bodies by hand.
fn synthetic_args_from_schema(schema: &serde_json::Value) -> serde_json::Value {
    use serde_json::{Map, Value};
    let kind = schema
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("object");
    match kind {
        "object" => {
            let mut obj = Map::new();
            if let Some(props) = schema.get("properties").and_then(|v| v.as_object()) {
                let required: std::collections::HashSet<&str> = schema
                    .get("required")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|i| i.as_str()).collect())
                    .unwrap_or_default();
                for (key, prop_schema) in props {
                    // Populate required keys eagerly; skip optionals to keep the
                    // synthetic payload minimal.
                    if required.contains(key.as_str()) || required.is_empty() {
                        obj.insert(key.clone(), synthetic_args_from_schema(prop_schema));
                    }
                }
            }
            Value::Object(obj)
        }
        "string" => Value::String(String::new()),
        "number" | "integer" => Value::from(0),
        "boolean" => Value::Bool(false),
        "array" => Value::Array(Vec::new()),
        "null" => Value::Null,
        _ => Value::Object(Map::new()),
    }
}

/// Implements `cubi plugins run <name> [args...]`. Returns the process
/// exit code so the caller can pass it to `std::process::exit`.
async fn run_plugin(name: &str, args: &[String], json: bool) -> i32 {
    let plugins = plugins::load_plugins();
    let Some(plugin) = plugins.iter().find(|p| p.name == name) else {
        eprintln!("cubi: no plugin named '{}'.", name);
        return 2;
    };
    let manifest = plugins::PluginManifest::load(&plugin.root);
    let entry = manifest
        .as_ref()
        .and_then(|m| m.entry.clone())
        .unwrap_or_else(|| {
            if cfg!(windows) {
                "handler.cmd".to_string()
            } else {
                "handler.sh".to_string()
            }
        });
    let handler = plugin.root.join(&entry);
    if !handler.exists() {
        eprintln!(
            "cubi: plugin '{}' has no handler at {}.",
            name,
            handler.display()
        );
        return 2;
    }
    // Confirm shell execution unless the manifest explicitly opts in.
    let perms = manifest.as_ref().map(|m| m.permissions).unwrap_or_default();
    if !perms.shell {
        if json {
            // Headless/JSON callers cannot answer an interactive prompt; refuse
            // rather than silently bypassing the shell-permission check.
            eprintln!(
                "cubi: plugin '{}' requires permissions.shell=true to run in --json mode.",
                name
            );
            return 2;
        }
        if !confirm_yn(&format!(
            "Plugin '{}' wants to execute {}. Allow?",
            name,
            handler.display()
        )) {
            eprintln!("cubi: aborted.");
            return 2;
        }
    }
    let mut cmd = std::process::Command::new(&handler);
    for a in args {
        cmd.arg(a);
    }
    match cmd.status() {
        Ok(status) => status.code().unwrap_or(0),
        Err(e) => {
            eprintln!("cubi: failed to exec {}: {}", handler.display(), e);
            2
        }
    }
}

/// Implements `cubi mcp test <server> [--tool NAME] [--json]`.
async fn run_mcp_test(server: &str, only_tool: Option<&str>, json: bool) -> i32 {
    use crate::mcp_config::McpConfig;
    let config = match McpConfig::load() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("cubi: could not load MCP config: {e}");
            return 2;
        }
    };
    let Some(server_config) = config.mcp_servers.get(server) else {
        eprintln!("cubi: no MCP server named '{}'.", server);
        return 2;
    };
    let mut client = match McpManager::connect_for_test(server_config).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("cubi: failed to connect to '{}': {:#}", server, e);
            return 2;
        }
    };
    let tools = match client.list_tools().await {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cubi: failed to list tools on '{}': {:#}", server, e);
            let _ = client.shutdown().await;
            return 2;
        }
    };
    let candidates: Vec<_> = if let Some(filter) = only_tool {
        tools.into_iter().filter(|t| t.name == filter).collect()
    } else {
        tools
    };
    if candidates.is_empty() {
        eprintln!("cubi: no tools matched on '{}'.", server);
        let _ = client.shutdown().await;
        return 2;
    }
    let mut exit = 0i32;
    for tool in &candidates {
        let request = synthetic_args_from_schema(&tool.input_schema);
        let started = std::time::Instant::now();
        let result = client.call_tool(&tool.name, request.clone()).await;
        let elapsed = started.elapsed().as_millis();
        match result {
            Ok(r) => {
                let resp = mcp_client::tool_result_to_json(&r);
                if json {
                    let env = serde_json::json!({
                        "tool": tool.name.as_str(),
                        "request": request,
                        "response": resp,
                        "elapsed_ms": elapsed,
                    });
                    println!("{}", env);
                } else {
                    println!("─ tool: {}", tool.name);
                    println!("  request:  {}", request);
                    println!("  response: {}", resp);
                    println!("  elapsed:  {} ms", elapsed);
                }
            }
            Err(e) => {
                exit = 11;
                let ue = user_error::UserError::new(
                    user_error::ErrorKind::Tool,
                    format!("tool '{}' failed: {:#}", tool.name, e),
                );
                if json {
                    let env = serde_json::json!({
                        "tool": tool.name.as_str(),
                        "request": request,
                        "error": ue.summary,
                        "elapsed_ms": elapsed,
                    });
                    println!("{}", env);
                } else {
                    eprintln!("─ tool: {}", tool.name);
                    eprintln!("  error:    {}", ue.summary);
                    eprintln!("  elapsed:  {} ms", elapsed);
                }
            }
        }
    }
    let _ = client.shutdown().await;
    exit
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
         cubi --list-sessions --json  List saved sessions as a JSON array\n  \
         cubi --delete-session <id>   Delete by full id or unique prefix\n  \
         cubi --diff-sessions <a> <b> [--json]\n  \
                                     Structured diff between two saved sessions.\n  \
                                     Reports model drift, common-prefix length,\n  \
                                     first divergence, and pinned-item delta.\n  \
         cubi --prune-sessions --older-than <duration> [--dry-run]\n  \
                                     Delete old session files (30d, 2w, 6m, 1y)\n  \
         cubi exec <prompt words>     One-shot, JSON output, no banner, no stream\n  \
                                      (shorthand for -p \"<words>\" --json --no-stream --no-banner)\n  \
         cubi run <file.md>           Run a Markdown script (optional YAML\n  \
                                      frontmatter: model, system, tools); uses\n  \
                                      headless --json output\n  \
         cubi plugins list [--json]   List discovered plugin bundles\n  \
         cubi plugins show <name> [--json]\n  \
                                      Show a plugin's manifest, handler, and permissions\n  \
         cubi plugins remove <name> [--force] [--yes]\n  \
                                      Remove a scaffolded plugin (refuses extras unless --force)\n  \
         cubi plugins run <name> [args...] [--json]\n  \
                                      Execute a plugin handler with extra arguments\n  \
         cubi plugins reload          Rediscover skills and plugin bundles\n  \
         cubi plugins new <name>      Scaffold ~/.cubi/plugins/<name>/ with a\n  \
                                      manifest, handler stub, and README\n  \
         cubi mcp test <server> [--tool <name>] [--json]\n  \
                                      Connect, list tools, and call each with stub args\n  \
         cubi mcp search [<query>] [--json]\n  \
                                      Search the embedded MCP registry\n  \
         cubi mcp install <name> [--force] [--env K=V]... [--json]\n  \
                                      Install a registry entry into ~/.cubi/mcp.json\n  \
         cubi mcp uninstall <name> [--json]\n  \
                                      Remove a server from ~/.cubi/mcp.json\n  \
         cubi doctor                  Run preflight checks and exit (0 ok, 2 fail)\n  \
         cubi doctor --fix            Run checks and apply safe automated fixes\n  \
                                      (create missing sessions dir, write a\n  \
                                      stub config, install shell completions)\n  \
         cubi doctor --json           Same, machine-readable JSON output\n  \
         cubi --print-config          Print the resolved config as JSON and exit\n  \
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
                                        each reply\n  \
         --system <file>                 Prepend file contents as a system\n  \
                                        message before chat starts\n  \
         --no-banner                    Suppress the welcome banner and tip\n  \
                                         of the day (also honors CUBI_NO_BANNER)\n  \
         --debug                         Show the full error cause chain on\n  \
                                         failure (also honors CUBI_DEBUG=1 and\n  \
                                         RUST_BACKTRACE)\n  \
         --json                          Emit machine-readable output where\n  \
                                        supported (session arrays or headless\n  \
                                        line-delimited events)\n  \
         --trace-tools <path>           Append a JSONL audit line per tool\n  \
                                        start/complete (also honors\n  \
                                        CUBI_TRACE_TOOLS env). Superset is\n  \
                                        now `--events`.\n  \
         --events <path>                Append every internal event (turn\n  \
                                        boundaries, tool calls, rationales,\n  \
                                        MCP transitions) as JSONL. Also\n  \
                                        honors CUBI_EVENTS env.\n  \
         --explain-tools                Surface a one-line rationale before\n  \
                                        each tool call (or `tool_rationale`\n  \
                                        event in headless JSON mode). Also\n  \
                                        honors CUBI_EXPLAIN_TOOLS=1.\n  \
         --usage-footer                 Append a one-line usage footer\n  \
                                        after each REPL turn (also\n  \
                                        togglable via /usage footer).\n  \
         --quiet                        Suppress banner, tip-of-the-day,\n  \
                                        spinner, stats/usage footers in one\n  \
                                        switch. Implies --no-banner. Does not\n  \
                                        affect assistant output, slash command\n  \
                                        output, errors, --events or JSON\n  \
                                        events. Also honors CUBI_QUIET=1.\n\n\
         Headless exit codes:\n  0 ok · 2 usage/config · 10 model/API error · 11 tool error · 12 context budget · 13 network · 130 cancelled\n\n\
         Notes:\n  -p/--prompt requires inline text and does not read stdin. Without -p,\n  \
         piped stdin becomes the one-shot prompt. One-shot mode buffers by default;\n  \
         pass --stream to stream tokens.\n\n\
         Once inside the REPL, type /help to list slash commands.",
        env!("CARGO_PKG_VERSION")
    );
}

fn print_config() -> Result<()> {
    let config = AppConfig::load();
    let mut value = serde_json::to_value(&config)?;
    redact_secrets(&mut value);
    if let Some(obj) = value.as_object_mut() {
        let path = AppConfig::storage_path()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        obj.insert("_config_path".to_string(), serde_json::Value::String(path));
        obj.insert(
            "_resolved_model".to_string(),
            serde_json::Value::String(onboarding::resolve_model(&config, DEFAULT_MODEL)),
        );
    }
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

fn redact_secrets(value: &mut serde_json::Value) {
    // Single source of truth lives in `trace_tools` so `--print-config`
    // and `--trace-tools` apply identical redaction rules.
    crate::trace_tools::redact_secrets(value);
}

fn print_sessions(json: bool) -> Result<()> {
    let sessions = SessionStore::list_all()?;
    if json {
        println!("{}", serde_json::to_string_pretty(&sessions)?);
        return Ok(());
    }
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

fn run_diff_sessions(a: &str, b: &str, json: bool) -> Result<()> {
    let session_a = resolve_session_for_diff(a)?;
    let session_b = resolve_session_for_diff(b)?;
    let diff = sessions_diff::diff(&session_a, &session_b);
    if json {
        println!("{}", serde_json::to_string_pretty(&diff)?);
        return Ok(());
    }
    print_session_diff_pretty(&diff);
    Ok(())
}

fn resolve_session_for_diff(prefix: &str) -> Result<sessions::SessionFile> {
    match sessions::load_by_prefix(prefix)? {
        sessions::LoadSessionResult::Found(session, _meta) => Ok(*session),
        sessions::LoadSessionResult::NotFound => {
            eprintln!("cubi: no session matches '{prefix}'.");
            std::process::exit(2);
        }
        sessions::LoadSessionResult::Ambiguous(candidates) => {
            eprintln!("cubi: session prefix '{prefix}' is ambiguous. Candidates:");
            for meta in candidates {
                eprintln!("  {}  {}", meta.id, meta.cwd);
            }
            std::process::exit(2);
        }
    }
}

fn print_session_diff_pretty(d: &sessions_diff::SessionDiff) {
    println!("Session A: {}  (model: {})", d.id_a, d.model_a);
    println!("Session B: {}  (model: {})", d.id_b, d.model_b);
    if d.identical_id {
        println!("(both prefixes resolve to the same session)");
        return;
    }
    if d.model_drift {
        println!("Model drift: {} → {}", d.model_a, d.model_b);
    }
    println!(
        "Messages:  {} → {} (Δ {:+})",
        d.count_a,
        d.count_b,
        (d.count_b as i64) - (d.count_a as i64)
    );
    println!("Common prefix: {} message(s)", d.common_prefix_len);
    match &d.divergence {
        Some(div) => {
            println!("First divergence at index {}:", div.index);
            println!("  A [{}] {}", div.a_role, div.a_preview);
            println!("  B [{}] {}", div.b_role, div.b_preview);
        }
        None => println!("Histories are identical."),
    }
    if !d.pinned_added.is_empty() {
        println!("Pinned items added in B:");
        for s in &d.pinned_added {
            println!("  + {s}");
        }
    }
    if !d.pinned_removed.is_empty() {
        println!("Pinned items missing from B:");
        for s in &d.pinned_removed {
            println!("  - {s}");
        }
    }
}

fn prune_sessions(age_secs: u64, dry_run: bool) -> Result<()> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let cutoff = now.saturating_sub(age_secs);
    let report = SessionStore::prune_older_than(cutoff, dry_run)?;
    if dry_run {
        for item in &report.items {
            println!(
                "would prune {}  {} bytes  {}",
                item.id,
                item.bytes,
                item.path.display()
            );
        }
        println!(
            "Would prune {} session(s), freeing {} bytes.",
            report.items.len(),
            report.bytes
        );
    } else {
        println!(
            "Pruned {} session(s), freeing {} bytes.",
            report.items.len(),
            report.bytes
        );
    }
    Ok(())
}

/// Installs a tracing subscriber driven by the `CUBI_LOG` env var
/// (e.g. `CUBI_LOG=cubi=debug`). When unset, no subscriber is installed
/// so the binary stays quiet by default. Output is always stderr —
/// never stdout — to avoid polluting machine-readable JSON output.
fn init_tracing() {
    let Ok(filter) = std::env::var("CUBI_LOG") else {
        return;
    };
    if filter.is_empty() {
        return;
    }
    let env_filter = match tracing_subscriber::EnvFilter::try_new(&filter) {
        Ok(f) => f,
        Err(err) => {
            eprintln!("cubi: ignoring invalid CUBI_LOG={filter:?}: {err}");
            return;
        }
    };
    let _ = tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_writer(std::io::stderr)
        .with_target(true)
        .try_init();
}

fn parse_duration_secs(input: &str) -> Option<u64> {
    let (number, unit) = input.split_at(input.len().saturating_sub(1));
    let value = number.parse::<u64>().ok()?;
    if value == 0 {
        return None;
    }
    let days = match unit {
        "d" => value,
        "w" => value.checked_mul(7)?,
        "m" => value.checked_mul(30)?,
        "y" => value.checked_mul(365)?,
        _ => return None,
    };
    days.checked_mul(86_400)
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

#[cfg(test)]
mod synthetic_args_tests {
    use super::synthetic_args_from_schema;
    use serde_json::json;

    #[test]
    fn object_with_required_keys_populates_them() {
        let schema = json!({
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer"},
                "force": {"type": "boolean"}
            },
            "required": ["command"]
        });
        let v = synthetic_args_from_schema(&schema);
        assert_eq!(v["command"], "");
        // Optional keys are intentionally skipped when `required` is non-empty.
        assert!(v.get("timeout").is_none());
        assert!(v.get("force").is_none());
    }

    #[test]
    fn object_without_required_includes_all_properties() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "number"}
            }
        });
        let v = synthetic_args_from_schema(&schema);
        assert_eq!(v["a"], "");
        assert_eq!(v["b"], 0);
    }

    #[test]
    fn scalar_branches_have_distinct_zero_values() {
        assert_eq!(
            synthetic_args_from_schema(&json!({"type": "string"})),
            json!("")
        );
        assert_eq!(
            synthetic_args_from_schema(&json!({"type": "number"})),
            json!(0)
        );
        assert_eq!(
            synthetic_args_from_schema(&json!({"type": "integer"})),
            json!(0)
        );
        assert_eq!(
            synthetic_args_from_schema(&json!({"type": "boolean"})),
            json!(false)
        );
        assert_eq!(
            synthetic_args_from_schema(&json!({"type": "array"})),
            json!([])
        );
        assert_eq!(
            synthetic_args_from_schema(&json!({"type": "null"})),
            json!(null)
        );
    }

    #[test]
    fn missing_type_defaults_to_object() {
        let v = synthetic_args_from_schema(&json!({}));
        assert!(v.is_object());
        assert_eq!(v.as_object().unwrap().len(), 0);
    }
}

#[cfg(test)]
mod redact_tests {
    use super::redact_secrets;
    use serde_json::json;

    #[test]
    fn redact_replaces_key_token_secret_fields() {
        let mut v = json!({
            "default_model": "qwen3:4b",
            "api_key": "abc",
            "auth_token": "xyz",
            "my_secret": "shh",
            "user_password": "hunter2",
            "nested": { "inner_key": "val", "ok": "fine" }
        });
        redact_secrets(&mut v);
        assert_eq!(v["default_model"], "qwen3:4b");
        assert_eq!(v["api_key"], "<redacted>");
        assert_eq!(v["auth_token"], "<redacted>");
        assert_eq!(v["my_secret"], "<redacted>");
        assert_eq!(v["user_password"], "<redacted>");
        assert_eq!(v["nested"]["inner_key"], "<redacted>");
        assert_eq!(v["nested"]["ok"], "fine");
    }
}
