use crate::agent_loop::{self, AGENT_TOOL_NAME, MAX_AGENT_STEPS, SUBAGENT_DEFAULT_STEPS};
use crate::commands::{self, COMMANDS, Cmd};
use crate::completer::SlashHelper;
use crate::executor::AIExecutor;
use crate::exit_code::{self, ExitCode};
use crate::file_mentions::{self, UserCommand};
use crate::file_rollback::FileJournal;
use crate::git_cmds;
use crate::hooks::{HookDef, HookEvent, HookRegistry, HooksConfig};
use crate::mcp_manager::McpManager;
use crate::memdir::Memdir;
use crate::oauth;
use crate::ollama::{ChatStats, Message, ToolCall};
use crate::onboarding::AppConfig;
use crate::output_styles;
use crate::permissions::Permissions;
use crate::plugins::{self, Plugin};
use crate::policy::Policy;
use crate::project_memory;
use crate::sessions::{DeleteSessionResult, FindSessionResult, SessionFile, SessionStore};
use crate::settings_sync;
use crate::skills::{self, Skill};
use crate::style::CubiStyle;
use crate::themes;
use crate::todos::TodoList;
use anyhow::{Context, Result};
use rustyline::Editor;
use rustyline::error::ReadlineError;
use rustyline::history::DefaultHistory;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Sentinel prefix used to tag system messages that should only influence the
/// next assistant turn (e.g. `/ask`). After the model responds once, any
/// system message starting with this prefix is stripped from `history` so the
/// hint does not poison every subsequent turn.
const SINGLE_TURN_SYSTEM_TAG: &str = "SYSTEM[single-turn]:";

/// Prefix used to tag the auto-injected project-memory system message so it
/// can be located and replaced on `/memory-reload`.
const PROJECT_MEMORY_PREFIX: &str = "SYSTEM: Project memory loaded from";

/// Prefix used to tag system messages that came from `/pin`. Recognized
/// by `compact` so pinned context is preserved across summarization.
const PINNED_SYSTEM_TAG: &str = "SYSTEM[pinned]:";

mod agent;
mod edit_cmd;
mod multiline;
mod render;
mod repl;
mod spinner;

#[cfg(test)]
#[cfg(test)]
use render::welcome_banner_rows;
#[cfg(test)]
use repl::repl_history_path;

pub struct ChatCLI {
    executor: AIExecutor,
    history: Vec<Message>,
    mcp_manager: Option<McpManager>,
    todos: TodoList,
    /// Cross-session persistent memory store (`~/.cubi/memdir/`).
    memdir: Memdir,
    /// User-defined Markdown commands loaded from disk.
    user_commands: Vec<UserCommand>,
    /// Hook registry for lifecycle events.
    hooks: HookRegistry,
    /// Loaded Markdown skills.
    skills: Vec<Skill>,
    /// Per-session approved tools for untrusted directories.
    approved_tools: HashSet<String>,
    /// Shared with `BuiltinToolRegistry` so write/exec tools (`bash`,
    /// `edit_file`, `write_file`) observe `/plan` toggles instantly.
    plan_mode: Arc<AtomicBool>,
    permissions: Arc<Mutex<Permissions>>,
    /// Per-project on-disk session checkpoint store, or `None` when no
    /// home directory could be resolved (sessions degrade silently).
    session_store: Option<SessionStore>,
    /// The session currently being appended to. Lazily initialized so
    /// the file isn't created until the user actually sends a message.
    current_session: Option<SessionFile>,
    /// Shared with `BuiltinToolRegistry` — receives one pre-image
    /// snapshot per `edit_file`/`write_file` so `/rewind` can restore.
    journal: FileJournal,
    /// Discovered plugin bundles (see `plugins.rs`). Refreshed by
    /// `/reload-plugins`.
    plugins: Vec<Plugin>,
    /// Read-only admin policy overlay (see `policy.rs`). Loaded once at
    /// startup; surfaced by `/permissions`.
    policy: Policy,
    /// Whether to stream tokens live (the default). When false, the model's
    /// response is buffered until complete and then printed in one shot —
    /// which is also the only mode that triggers markdown rendering.
    stream_enabled: bool,
    /// Whether to render the assistant's final reply as markdown via
    /// termimad. Only takes effect when `stream_enabled == false`. Auto-
    /// disabled when stdout is not a TTY.
    markdown_enabled: bool,
    /// Whether to print a one-line usage footer (tokens, ms, tok/s) after
    /// each assistant turn. Off by default to avoid noise; togglable via
    /// `/stats-footer`.
    stats_footer_enabled: bool,
    /// Cumulative provider-reported usage for *this run* (does not persist
    /// across `/resume`). Surfaced by `/stats`.
    session_stats: ChatStats,
    /// Suppresses REPL-only decoration so one-shot output remains pipeable.
    headless_mode: bool,
    /// Emits line-delimited JSON events in headless mode.
    json_enabled: bool,
    /// (loaded, configured) MCP server counts shown in the one-line
    /// startup banner. Both are 0 when MCP is fully disabled.
    mcp_counts: (usize, usize),
    /// Suppresses the welcome banner and tip-of-the-day output.
    no_banner: bool,
    /// Persistent user config. Read at startup so the agent loop can
    /// consult e.g. `auto_compact` / `compact_threshold_pct` without
    /// re-parsing the JSON every turn.
    app_config: AppConfig,
    /// User-curated pinned items. Each entry is also rendered as a
    /// `PINNED_SYSTEM_TAG`-prefixed system message at the front of
    /// `history` so the LLM sees it on every turn; `/compact` is aware
    /// of the tag and preserves these messages verbatim.
    pinned: Vec<String>,
    /// Optional `--trace-tools` JSONL audit log. When `Some`, every
    /// tool dispatch in `agent_turn` writes a tool_start + tool_complete
    /// pair to the configured path.
    tool_tracer: Option<Arc<crate::trace_tools::ToolTracer>>,
}

/// Initial UX flags resolved from CLI argv in main.rs. Kept as a tiny POD
/// struct so the cli/main boundary stays explicit rather than threaded
/// through positional bools.
#[derive(Debug, Clone)]
pub struct CliFlags {
    pub stream: bool,
    pub markdown: bool,
    pub stats_footer: bool,
    pub system_prompt: Option<String>,
    pub json: bool,
    pub mcp_counts: (usize, usize),
    pub no_banner: bool,
}

impl Default for CliFlags {
    fn default() -> Self {
        // Auto-detect TTY for markdown so piped/redirected output stays
        // plain. Other flags default to their "good UX" values.
        Self {
            stream: true,
            markdown: std::io::IsTerminal::is_terminal(&std::io::stdout()),
            stats_footer: false,
            system_prompt: None,
            json: false,
            mcp_counts: (0, 0),
            no_banner: false,
        }
    }
}

impl ChatCLI {
    fn emit_status(&self, msg: impl std::fmt::Display) {
        if self.json_enabled && self.headless_mode {
            return;
        }
        crate::out::status_line(self.headless_mode, msg);
    }

    fn emit_json_event(value: serde_json::Value) {
        crate::json_events::emit(true, &value);
    }

    fn emit_json_event_if(enabled: bool, value: serde_json::Value) {
        crate::json_events::emit(enabled, &value);
    }

    #[allow(dead_code)]
    pub fn new(
        executor: AIExecutor,
        mcp_manager: Option<McpManager>,
        permissions: Arc<Mutex<Permissions>>,
        plan_mode: Arc<AtomicBool>,
        journal: FileJournal,
    ) -> Self {
        Self::new_with_flags(
            executor,
            mcp_manager,
            permissions,
            plan_mode,
            journal,
            CliFlags::default(),
        )
    }

    pub fn new_with_flags(
        executor: AIExecutor,
        mcp_manager: Option<McpManager>,
        permissions: Arc<Mutex<Permissions>>,
        plan_mode: Arc<AtomicBool>,
        journal: FileJournal,
        flags: CliFlags,
    ) -> Self {
        let mut cli = Self {
            executor,
            history: Vec::new(),
            mcp_manager,
            todos: TodoList::load_for_current_dir(),
            memdir: Memdir::load(),
            user_commands: file_mentions::load_user_commands(),
            hooks: HookRegistry::load(),
            skills: skills::load_skills(),
            approved_tools: HashSet::new(),
            plan_mode,
            permissions,
            session_store: SessionStore::for_current_dir(),
            current_session: None,
            journal,
            plugins: plugins::load_plugins(),
            policy: Policy::load(),
            stream_enabled: flags.stream,
            markdown_enabled: flags.markdown,
            stats_footer_enabled: flags.stats_footer,
            session_stats: ChatStats::default(),
            headless_mode: false,
            json_enabled: flags.json,
            mcp_counts: flags.mcp_counts,
            no_banner: flags.no_banner,
            app_config: AppConfig::load(),
            pinned: Vec::new(),
            tool_tracer: None,
        };

        if let Some(system_prompt) = flags.system_prompt {
            cli.history.push(Message::text("system", system_prompt));
        }

        // Auto-inject project memory (CUBI.md) into context, if present.
        cli.inject_project_memory();

        // Auto-inject cross-session memdir into context, if non-empty.
        cli.inject_memdir();

        // Steer reply formatting via the configured output style. We push
        // a system message rather than mutating per-prompt so the
        // preset rides along with every assistant turn for the session.
        let style = std::env::var("CUBI_OUTPUT_STYLE")
            .unwrap_or_else(|_| output_styles::DEFAULT_STYLE.to_string());
        cli.history.push(Message::text(
            "system",
            format!("SYSTEM: {}", output_styles::system_prompt_for(&style)),
        ));

        // Tools are advertised to the model via Ollama's native `tools:`
        // field in `agent_loop::build_tool_specs`. We deliberately do NOT
        // also inject them as a system message: small models tend to echo
        // the schema back as content when they see it twice. Keep this
        // path empty unless we discover a model that benefits from the
        // duplication.

        cli
    }

    pub fn save_conversation(&self, filename: &str, force: bool) -> Result<()> {
        check_overwrite_allowed(filename, force, "/save")?;
        let json = serde_json::to_string_pretty(&self.history)?;
        fs::write(filename, json)?;
        Ok(())
    }

    /// Loads a conversation from `filename`, leaving the existing `history`
    /// untouched if the file is missing or fails to parse. This avoids the
    /// previous footgun where a typo'd path or a corrupt JSON file silently
    /// wiped the current conversation.
    pub fn load_conversation(&mut self, filename: &str) -> Result<()> {
        let json = fs::read_to_string(filename)
            .with_context(|| format!("Failed to read '{}'", filename))?;
        let parsed: Vec<Message> = serde_json::from_str(&json).with_context(|| {
            format!(
                "Failed to parse '{}' as a saved conversation (expected JSON array of messages)",
                filename
            )
        })?;
        self.history = parsed;
        Ok(())
    }

    /// Appends pre-rendered messages to the end of history. Used by
    /// `cubi run <file.md>` to seed a replayed transcript before the
    /// final user turn is sent through `run_one_shot`. No side effects
    /// (no `agent_turn`, no checkpoint) — pure history mutation.
    pub fn preload_history(&mut self, msgs: Vec<Message>) {
        self.history.extend(msgs);
    }

    /// Wires a tool tracer into the agent loop. `None` disables
    /// tracing; passing a fresh `ToolTracer` enables the JSONL audit
    /// log for subsequent tool dispatches.
    pub fn set_tool_tracer(&mut self, tracer: Option<Arc<crate::trace_tools::ToolTracer>>) {
        self.tool_tracer = tracer;
    }

    /// Runs a single prompt without the welcome banner or rustyline REPL.
    /// Human-facing progress stays on stderr; only model reply tokens/content
    /// are written to stdout so callers can pipe the result.
    pub async fn run_one_shot(&mut self, prompt: &str) -> Result<()> {
        self.headless_mode = true;
        if self.json_enabled {
            self.markdown_enabled = false;
        }
        self.hooks.fire_session_start(self.executor.get_model());
        let expanded = file_mentions::expand_file_mentions(prompt);
        let turn_start = self.history.len();
        self.history.push(Message::text("user", expanded));
        self.journal.start_turn();
        let result = self.agent_turn(turn_start).await;
        self.hooks.fire_stop();
        result
    }

    /// Tries to match a `/command` against user-defined and plugin Markdown
    /// commands. Returns `true` if a command was matched and handled.
    fn try_user_command(&mut self, input: &str) -> bool {
        let input = input.trim();
        let (head, args) = match input.find(char::is_whitespace) {
            Some(i) => (&input[..i], input[i..].trim()),
            None => (input, ""),
        };
        // Strip the leading `/` to get the command name.
        let cmd_name = head.strip_prefix('/').unwrap_or(head).to_lowercase();

        let matched_user = self
            .user_commands
            .iter()
            .find(|c| c.name == cmd_name)
            .cloned();

        if let Some(user_cmd) = matched_user {
            // Inject the Markdown body as a single-turn system message.
            let body = if args.is_empty() {
                user_cmd.body.clone()
            } else {
                format!("{}\n\nUser argument: {}", user_cmd.body, args)
            };

            self.history.push(Message::text(
                "system",
                format!(
                    "{} User command /{} (from {}):\n\n{}",
                    SINGLE_TURN_SYSTEM_TAG,
                    user_cmd.name,
                    user_cmd.path.display(),
                    body
                ),
            ));
            println!(
                "{} Applied user command /{}",
                "✓".bright_green(),
                user_cmd.name.bright_cyan()
            );
            return true;
        }

        let Some(plugin_cmd) = plugins::resolve(&self.plugins, head).cloned() else {
            return false;
        };
        let plugin_name = head
            .strip_prefix('/')
            .unwrap_or(head)
            .split_once(':')
            .map(|(ns, _)| ns)
            .unwrap_or_default()
            .to_string();

        let body = if args.is_empty() {
            plugin_cmd.body.clone()
        } else {
            format!("{}\n\nUser argument: {}", plugin_cmd.body, args)
        };

        self.history.push(Message::text(
            "system",
            format!(
                "{} Plugin command {} (from {}):\n\n{}",
                SINGLE_TURN_SYSTEM_TAG,
                plugin_cmd.trigger(&plugin_name),
                plugin_cmd.path.display(),
                body
            ),
        ));
        println!(
            "{} Applied plugin command {}",
            "✓".bright_green(),
            plugin_cmd.trigger(&plugin_name).bright_cyan()
        );
        true
    }

    async fn handle_command(&mut self, input: &str) -> Result<bool> {
        let Some((cmd, args)) = commands::parse(input) else {
            let head = input.split_whitespace().next().unwrap_or(input);
            let candidates = commands::suggestions(head);
            println!("{} {}", "Unknown command:".bright_red(), head);
            if candidates.is_empty() {
                println!("Type {} for available commands", "/help".bright_cyan());
            } else {
                println!("Did you mean?");
                for name in &candidates {
                    println!("  {}", name.bright_cyan());
                }
            }
            return Ok(true);
        };

        match cmd {
            Cmd::Quit => {
                println!("{}", "Goodbye!".bright_cyan());
                return Ok(false);
            }
            Cmd::Clear => {
                self.history.clear();
                // Drop the in-memory session pointer too: leaving it set
                // would make the next assistant turn overwrite the saved
                // checkpoint with an (almost) empty history, effectively
                // discarding the prior conversation from disk. A fresh
                // `current_session` is allocated lazily on the next
                // `checkpoint_session` call.
                self.current_session = None;
                // Also drop the file-rollback journal — `/clear` is a
                // hard reset, so further `/rewind`s should not reach
                // back into the now-discarded conversation.
                self.journal.reset();
                println!("{}", "Conversation history cleared.".yellow());
            }
            Cmd::History => {
                self.show_history();
            }
            Cmd::Help => {
                self.show_help();
            }
            Cmd::Model => {
                if args.is_empty() {
                    println!("Current model: {}", self.executor.get_model().bright_cyan());
                } else {
                    match self.executor.switch_model(args.to_string()).await {
                        Ok(_) => {
                            println!(
                                "{} Switched to model: {}",
                                "✓".bright_green(),
                                args.bright_cyan()
                            );
                            self.history.clear();
                            // Same reasoning as `/clear`: don't keep
                            // overwriting the prior session's checkpoint
                            // with a fresh history after a model switch.
                            self.current_session = None;
                        }
                        Err(e) => {
                            eprintln!("{} {}", "Error:".bright_red(), e);
                        }
                    }
                }
            }
            Cmd::McpTools => {
                self.show_mcp_tools();
            }
            Cmd::McpCall => {
                if self.plan_mode.load(Ordering::SeqCst) {
                    println!(
                        "{} Plan mode is on — refusing /mcp-call. Toggle off with /plan first.",
                        "✗".bright_red()
                    );
                    return Ok(true);
                }
                let parts: Vec<&str> = args.splitn(2, ' ').collect();
                if args.is_empty() || parts.len() < 2 {
                    println!(
                        "{} Usage: /mcp-call <tool_name> <json_args>",
                        "Info:".bright_yellow()
                    );
                    println!("Example: /mcp-call add {{\"a\": 5, \"b\": 3}}");
                } else {
                    let tool_name = parts[0];
                    let args_str = parts[1];

                    match serde_json::from_str(args_str) {
                        Ok(json_args) => {
                            if let Err(e) = self.call_mcp_tool(tool_name, json_args).await {
                                eprintln!("{} {}", "Error:".bright_red(), e);
                            }
                        }
                        Err(e) => {
                            eprintln!("{} Invalid JSON: {}", "Error:".bright_red(), e);
                        }
                    }
                }
            }
            Cmd::McpReload => {
                if let Err(e) = self.reload_mcp().await {
                    eprintln!("{} Failed to reload MCP: {}", "Error:".bright_red(), e);
                } else {
                    println!("{} MCP configuration reloaded", "✓".bright_green());
                }
            }
            Cmd::McpResources => {
                self.show_mcp_resources(args).await;
            }
            Cmd::McpRead => {
                self.read_mcp_resource(args).await;
            }
            Cmd::Save => {
                if args.is_empty() {
                    println!("{} Usage: /save [-f] <filename>", "Info:".bright_yellow());
                    println!("       Pass -f to overwrite an existing file.");
                    println!("Example: /save my_chat.json");
                } else {
                    match parse_force_and_filename(args) {
                        Some((force, filename)) => match self.save_conversation(filename, force) {
                            Ok(()) => println!(
                                "{} Conversation saved to {}",
                                "✓".bright_green(),
                                filename.bright_cyan()
                            ),
                            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
                        },
                        None => {
                            println!("{} Usage: /save [-f] <filename>", "Info:".bright_yellow());
                            println!("       Pass -f to overwrite an existing file.");
                        }
                    }
                }
            }
            Cmd::Load => {
                if args.is_empty() {
                    println!("{} Usage: /load <filename>", "Info:".bright_yellow());
                    println!("Example: /load my_chat.json");
                } else if let Err(e) = self.load_conversation(args) {
                    // History intentionally preserved; surface the error
                    // chain so the user can tell read vs. parse failures
                    // apart.
                    eprintln!(
                        "{} Failed to load (existing conversation kept): {:#}",
                        "Error:".bright_red(),
                        e
                    );
                } else {
                    println!(
                        "{} Conversation loaded from {}",
                        "✓".bright_green(),
                        args.bright_cyan()
                    );
                }
            }
            Cmd::Batch => {
                if args.is_empty() {
                    println!("{} Usage: /batch <filename>", "Info:".bright_yellow());
                    println!("Example: /batch prompts.txt");
                    println!(
                        "\nBatch file format (one prompt per line, blank lines and #-comments are skipped):"
                    );
                    println!("  # warm-up");
                    println!("  What is Rust?");
                    println!("  Write hello world in Python");
                    println!("  Explain recursion");
                } else {
                    match self.process_batch_file(args).await {
                        Ok(BatchSummary { ok, failed }) => {
                            if failed == 0 {
                                println!(
                                    "{} Batch complete — {}/{} prompts succeeded",
                                    "✓".bright_green(),
                                    ok,
                                    ok + failed
                                );
                            } else {
                                println!(
                                    "{} Batch finished with errors — {} ok, {} failed",
                                    "!".bright_yellow(),
                                    ok,
                                    failed
                                );
                            }
                        }
                        Err(e) => {
                            eprintln!("{} Batch processing failed: {}", "Error:".bright_red(), e);
                        }
                    }
                }
            }
            Cmd::Version => {
                println!("{} {}", "cubi".bright_cyan(), env!("CARGO_PKG_VERSION"));
            }
            Cmd::Doctor => {
                self.run_doctor().await;
            }
            Cmd::Env => {
                self.show_env();
            }
            Cmd::Config => {
                self.show_config();
            }
            Cmd::Permissions => {
                self.show_permissions();
            }
            Cmd::ToolAllow => {
                self.handle_tool_allow(args);
            }
            Cmd::ToolDeny => {
                self.handle_tool_deny(args);
            }
            Cmd::Bug => {
                self.show_bug_url(args);
            }
            Cmd::Issue => {
                self.show_issue_url(args);
            }
            Cmd::Undo => {
                self.run_undo(args);
            }
            Cmd::Status => {
                self.show_status();
            }
            Cmd::Stats | Cmd::Usage => {
                self.show_stats();
            }
            Cmd::Stream => {
                let Some(next) = parse_toggle(args, self.stream_enabled) else {
                    eprintln!(
                        "{} Usage: /stream [on|off] (got {:?})",
                        "Error:".bright_red(),
                        args
                    );
                    return Ok(false);
                };
                self.stream_enabled = next;
                println!(
                    "{} Streaming is now {}",
                    "✓".bright_green(),
                    if self.stream_enabled {
                        "on".bright_green()
                    } else {
                        "off".bright_yellow()
                    }
                );
                if self.stream_enabled && self.markdown_enabled {
                    println!(
                        "  {} Markdown rendering only applies with `/stream off`.",
                        "ℹ".bright_blue()
                    );
                }
            }
            Cmd::Markdown => {
                let Some(next) = parse_toggle(args, self.markdown_enabled) else {
                    eprintln!(
                        "{} Usage: /markdown [on|off] (got {:?})",
                        "Error:".bright_red(),
                        args
                    );
                    return Ok(false);
                };
                self.markdown_enabled = next;
                println!(
                    "{} Markdown rendering is now {}",
                    "✓".bright_green(),
                    if self.markdown_enabled {
                        "on".bright_green()
                    } else {
                        "off".bright_yellow()
                    }
                );
                if self.markdown_enabled && self.stream_enabled {
                    println!(
                        "  {} Takes effect when streaming is off (`/stream off`).",
                        "ℹ".bright_blue()
                    );
                }
            }
            Cmd::StatsFooter => {
                let Some(next) = parse_toggle(args, self.stats_footer_enabled) else {
                    eprintln!(
                        "{} Usage: /stats-footer [on|off] (got {:?})",
                        "Error:".bright_red(),
                        args
                    );
                    return Ok(false);
                };
                self.stats_footer_enabled = next;
                println!(
                    "{} Per-turn stats footer is now {}",
                    "✓".bright_green(),
                    if self.stats_footer_enabled {
                        "on".bright_green()
                    } else {
                        "off".bright_yellow()
                    }
                );
            }
            Cmd::Plan => {
                self.toggle_plan_mode();
            }
            Cmd::Init => {
                self.run_init();
            }
            Cmd::Memory => {
                self.show_memory();
            }
            Cmd::MemoryReload => {
                self.inject_project_memory();
                match project_memory::read_memory_with_path() {
                    Ok(Some((p, _))) => println!(
                        "{} Reloaded project memory from {}",
                        "✓".bright_green(),
                        p.display().to_string().bright_cyan()
                    ),
                    Ok(None) => println!(
                        "{} No {} found in cwd or any ancestor",
                        "ℹ".bright_blue(),
                        project_memory::MEMORY_FILENAME.bright_cyan()
                    ),
                    Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
                }
            }
            Cmd::Todos => {
                self.todos.render();
            }
            Cmd::TodoAdd => {
                if args.is_empty() {
                    println!("{} Usage: /todo-add <text>", "Info:".bright_yellow());
                } else {
                    self.todos.add(args);
                    self.persist_todos();
                    println!("{} Added todo", "✓".bright_green());
                }
            }
            Cmd::TodoDone => {
                if args.is_empty() {
                    println!("{} Usage: /todo-done <index>", "Info:".bright_yellow());
                } else {
                    match args.parse::<usize>() {
                        Ok(n) => {
                            if self.todos.mark_done(n) {
                                self.persist_todos();
                                println!("{} Marked todo {} as done", "✓".bright_green(), n);
                            } else {
                                eprintln!("{} No todo with index {}", "Error:".bright_red(), n);
                            }
                        }
                        Err(_) => {
                            eprintln!("{} Usage: /todo-done <index>", "Error:".bright_red());
                        }
                    }
                }
            }
            Cmd::TodoRm => {
                if args.is_empty() {
                    println!("{} Usage: /todo-rm <index>", "Info:".bright_yellow());
                } else {
                    match args.parse::<usize>() {
                        Ok(n) => {
                            if self.todos.remove(n) {
                                self.persist_todos();
                                println!("{} Removed todo {}", "✓".bright_green(), n);
                            } else {
                                eprintln!("{} No todo with index {}", "Error:".bright_red(), n);
                            }
                        }
                        Err(_) => {
                            eprintln!("{} Usage: /todo-rm <index>", "Error:".bright_red());
                        }
                    }
                }
            }
            Cmd::TodoClear => {
                self.todos.clear();
                self.persist_todos();
                println!("{} Cleared todos", "✓".bright_green());
            }
            Cmd::Memdir => {
                self.memdir.render();
            }
            Cmd::MemdirAdd => {
                if args.is_empty() {
                    println!("{} Usage: /memdir-add <text>", "Info:".bright_yellow());
                } else {
                    let source = std::env::current_dir()
                        .ok()
                        .map(|p| p.display().to_string());
                    self.memdir.add(args, source.as_deref());
                    self.persist_memdir();
                    self.inject_memdir();
                    println!("{} Memory added", "✓".bright_green());
                }
            }
            Cmd::MemdirRm => {
                if args.is_empty() {
                    println!("{} Usage: /memdir-rm <index>", "Info:".bright_yellow());
                } else {
                    match args.parse::<usize>() {
                        Ok(n) => {
                            if self.memdir.remove(n) {
                                self.persist_memdir();
                                self.inject_memdir();
                                println!("{} Removed memory {}", "✓".bright_green(), n);
                            } else {
                                eprintln!("{} No memory with index {}", "Error:".bright_red(), n);
                            }
                        }
                        Err(_) => {
                            eprintln!("{} Usage: /memdir-rm <index>", "Error:".bright_red());
                        }
                    }
                }
            }
            Cmd::MemdirClear => {
                self.memdir.clear();
                self.persist_memdir();
                self.inject_memdir();
                println!("{} Cleared all memories", "✓".bright_green());
            }
            Cmd::Rewind => {
                self.rewind(args);
            }
            Cmd::Compact => {
                if let Err(e) = self.compact().await {
                    eprintln!("{} Compaction failed: {}", "Error:".bright_red(), e);
                }
            }
            Cmd::Pin => {
                if args.is_empty() {
                    println!("{} Usage: /pin <text>", "Info:".bright_yellow());
                } else {
                    let idx = self.pin(args);
                    println!(
                        "{} Pinned item #{}",
                        "✓".bright_green(),
                        idx.to_string().bright_cyan()
                    );
                }
            }
            Cmd::Pins => {
                self.show_pins();
            }
            Cmd::Unpin => {
                if args.is_empty() {
                    println!("{} Usage: /unpin <idx>", "Info:".bright_yellow());
                } else {
                    match args.parse::<usize>() {
                        Ok(n) if n >= 1 => match self.unpin(n) {
                            Some(text) => println!(
                                "{} Unpinned #{}: {}",
                                "✓".bright_green(),
                                n,
                                text.chars().take(60).collect::<String>().bright_cyan()
                            ),
                            None => eprintln!(
                                "{} No pinned item with index {}",
                                "Error:".bright_red(),
                                n
                            ),
                        },
                        _ => eprintln!(
                            "{} /unpin requires a 1-based index, got {:?}",
                            "Error:".bright_red(),
                            args
                        ),
                    }
                }
            }
            Cmd::Edit => {
                let seed = if args.is_empty() {
                    // Fall back to the last assistant message so the
                    // user can refine a prior answer. If there is none,
                    // open an empty buffer.
                    self.history
                        .iter()
                        .rev()
                        .find(|m| m.role == "assistant")
                        .map(|m| m.content.clone())
                        .unwrap_or_default()
                } else {
                    args.to_string()
                };
                let editor = edit_cmd::resolve_editor();
                let outcome = edit_cmd::run_editor_session(&seed, |path| {
                    edit_cmd::spawn_editor_blocking(&editor, path)
                });
                match outcome {
                    Ok(edit_cmd::EditOutcome::Submit(body)) => {
                        let expanded = file_mentions::expand_file_mentions(&body);
                        let turn_start = self.history.len();
                        self.history.push(Message::text("user", &expanded));
                        self.journal.start_turn();
                        if let Err(e) = self.agent_turn(turn_start).await {
                            eprintln!("{} {}\n", "Error:".bright_red().bold(), e);
                        }
                    }
                    Ok(edit_cmd::EditOutcome::Empty) => {
                        println!(
                            "{} editor returned empty buffer — nothing submitted",
                            "ℹ".bright_blue()
                        );
                    }
                    Ok(edit_cmd::EditOutcome::Unchanged) => {
                        println!(
                            "{} editor buffer unchanged — nothing submitted",
                            "ℹ".bright_blue()
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            "{} could not run editor ({}): {}",
                            "Error:".bright_red(),
                            editor,
                            e
                        );
                    }
                }
            }
            Cmd::Skills => {
                self.handle_skills(args).await;
            }
            Cmd::Hooks => {
                self.handle_hooks(args);
            }
            Cmd::Ask => {
                if args.is_empty() {
                    println!("{} Usage: /ask <question>", "Info:".bright_yellow());
                    println!("Records a clarifying question to be answered on the next turn.");
                } else {
                    self.ask_user(args);
                }
            }
            Cmd::Sessions => {
                self.handle_sessions(args);
            }
            Cmd::Resume => {
                self.resume_session(args);
            }
            Cmd::Trust => {
                self.handle_trust(args);
            }
            Cmd::Diff => {
                self.run_diff(args);
            }
            Cmd::Commit => {
                if self.plan_mode.load(Ordering::SeqCst) {
                    println!(
                        "{} Plan mode is on — refusing /commit. Toggle off with /plan first.",
                        "✗".bright_red()
                    );
                    return Ok(true);
                }
                self.run_commit(args);
            }
            Cmd::CommitPushPr => {
                self.run_commit_push_pr(args);
            }
            Cmd::Review => {
                self.run_review().await;
            }
            Cmd::Worktree => {
                self.run_worktree(args);
            }
            Cmd::Branch => {
                self.run_branch(args);
            }
            Cmd::Tag => {
                self.run_tag(args);
            }
            Cmd::Files => {
                self.run_files();
            }
            Cmd::AddDir => {
                self.handle_add_dir(args);
            }
            Cmd::Export => {
                if args.is_empty() {
                    println!(
                        "{} Usage: /export [-f] <filename.md>",
                        "Info:".bright_yellow()
                    );
                    println!("       Pass -f to overwrite an existing file.");
                } else {
                    match parse_force_and_filename(args) {
                        Some((force, filename)) => match self.export_markdown(filename, force) {
                            Ok(()) => println!(
                                "{} Conversation exported to {}",
                                "✓".bright_green(),
                                filename.bright_cyan()
                            ),
                            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
                        },
                        None => {
                            println!(
                                "{} Usage: /export [-f] <filename.md>",
                                "Info:".bright_yellow()
                            );
                            println!("       Pass -f to overwrite an existing file.");
                        }
                    }
                }
            }
            Cmd::InitVerifiers => self.run_init_verifiers(),
            Cmd::PrComments => self.run_pr_comments(args),
            Cmd::SecurityReview => self.run_security_review().await,
            Cmd::AutofixPr => self.run_autofix_pr(args).await,
            Cmd::Agents => self.show_agents(),
            Cmd::Tasks => self.todos.render(),
            Cmd::Teleport => self.run_teleport(args),
            Cmd::Passes | Cmd::Effort => self.handle_effort(cmd, args),
            Cmd::Theme => self.handle_theme(args),
            Cmd::Color => self.handle_color(args),
            Cmd::OutputStyle => self.handle_output_style(args),
            Cmd::Statusline => self.show_statusline(),
            Cmd::Keybindings => self.show_keybindings(),
            Cmd::Vim => self.handle_vim(args),
            Cmd::Login => self.handle_login(args),
            Cmd::Logout => self.handle_logout(args),
            Cmd::OauthRefresh => self.show_oauth_refresh(args),
            Cmd::PrivacySettings => self.handle_privacy_settings(args),
            Cmd::Mcp => self.show_mcp_status(),
            Cmd::Plugin => self.show_plugins(),
            Cmd::ReloadPlugins => self.reload_plugins(),
            Cmd::Cost => self.show_cost(),
            Cmd::PerfIssue => self.show_perf_issue_url(args),
            Cmd::Heapdump => self.show_heap_info(),
            Cmd::DebugToolCall => self.handle_debug_tool_call(args),
            Cmd::Upgrade => self.show_upgrade(),
            Cmd::Install => self.show_install(),
            Cmd::InstallGithubApp => self.show_install_github_app(),
            Cmd::InstallSlackApp => self.show_install_slack_app(),
            Cmd::SandboxToggle => self.toggle_plan_mode(),
            Cmd::ResetLimits => self.reset_limits(),
            Cmd::Share => self.run_share(args),
            Cmd::Copy => self.run_copy(),
            Cmd::Feedback => self.show_feedback_url(args),
            Cmd::ReleaseNotes => self.show_release_notes(),
            Cmd::Stickers => self.show_stickers(),
            Cmd::SettingsSync => self.handle_settings_sync(args),
            Cmd::Policy => self.show_policy(),
            Cmd::Tip => self.show_tip(),
            Cmd::McpPrompts => self.show_mcp_prompts(args).await,
        }
        Ok(true)
    }

    async fn process_batch_file(&self, filename: &str) -> Result<BatchSummary> {
        let content = fs::read_to_string(filename)
            .with_context(|| format!("Failed to read batch file '{}'", filename))?;

        // Strip blank lines and `#`-prefixed comment lines so users can
        // annotate their batch files without those getting sent to the model.
        let prompts: Vec<String> = content
            .lines()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty() && !s.starts_with('#'))
            .map(|s| s.to_string())
            .collect();

        if prompts.is_empty() {
            println!(
                "{} Batch file contained no prompts (after stripping blanks/comments).",
                "ℹ".bright_blue()
            );
            return Ok(BatchSummary { ok: 0, failed: 0 });
        }

        println!("Processing {} prompts...", prompts.len());

        let mut summary = BatchSummary { ok: 0, failed: 0 };
        for (i, prompt) in prompts.iter().enumerate() {
            println!("\n[{}/{}] {}", i + 1, prompts.len(), prompt);
            match self
                .executor
                .chat(vec![Message::text("user", prompt.clone())])
                .await
            {
                Ok(response) => {
                    println!("Response: {}", response);
                    summary.ok += 1;
                }
                Err(e) => {
                    // Don't abort the rest of the batch — surface the error
                    // and keep going so a single bad prompt doesn't sink the
                    // whole run.
                    eprintln!("{} prompt {} failed: {}", "Error:".bright_red(), i + 1, e);
                    summary.failed += 1;
                }
            }
        }

        Ok(summary)
    }

    fn show_mcp_tools(&self) {
        if let Some(mcp) = &self.mcp_manager {
            let tools = mcp.list_tools();
            if tools.is_empty() {
                println!("{}", "No MCP tools available.".yellow());
                return;
            }

            println!("\n{}", "Available MCP Tools:".bright_yellow().bold());
            println!("{}", "=".repeat(60).bright_black());

            // Group by built-in vs external
            let mut builtin = Vec::new();
            let mut external = Vec::new();

            for (server_name, tool) in mcp.get_tools_with_server().values() {
                if server_name == "builtin" {
                    builtin.push(tool);
                } else {
                    external.push((server_name, tool));
                }
            }

            if !builtin.is_empty() {
                println!("\n{}", "Built-in Tools:".bright_blue().bold());
                for tool in builtin {
                    println!("\n  {} {}", "●".bright_green(), tool.name.bright_cyan());
                    println!("    {}", tool.description);
                }
            }

            if !external.is_empty() {
                println!("\n{}", "External MCP Servers:".bright_blue().bold());
                for (server, tool) in external {
                    println!(
                        "\n  {} {} (from {})",
                        "●".bright_green(),
                        tool.name.bright_cyan(),
                        server.bright_magenta()
                    );
                    println!("    {}", tool.description);
                }
            }

            println!("\n{}\n", "=".repeat(60).bright_black());
            println!("Use {} <tool> <args> to execute", "/mcp-call".bright_cyan());
        }
    }

    async fn call_mcp_tool(&mut self, tool_name: &str, arguments: serde_json::Value) -> Result<()> {
        if let Some(mcp) = &mut self.mcp_manager {
            println!("{} Calling tool '{}'...", "⚙".bright_blue(), tool_name);

            let result = mcp.call_tool(tool_name, arguments).await?;

            for content in &result.content {
                if content.content_type == "text" {
                    println!("{} {}", "✓".bright_green(), content.text);
                }
            }
        } else {
            anyhow::bail!("MCP not initialized");
        }

        Ok(())
    }

    async fn show_mcp_resources(&mut self, args: &str) {
        let Some(mcp) = &mut self.mcp_manager else {
            eprintln!("{} MCP not initialized", "Error:".bright_red());
            return;
        };
        let server_filter = args.trim();
        match mcp.list_resources().await {
            Ok(resources) => {
                let filtered: Vec<_> = resources
                    .into_iter()
                    .filter(|(server, _)| server_filter.is_empty() || server == server_filter)
                    .collect();
                if filtered.is_empty() {
                    println!("{} No MCP resources found.", "ℹ".bright_blue());
                    return;
                }
                println!("\n{}", "MCP resources:".bright_yellow().bold());
                for (server, resource) in filtered {
                    let description = resource.description.unwrap_or_default();
                    println!(
                        "  [{}] {} - {}",
                        server.bright_cyan(),
                        resource.uri.bright_white(),
                        description.bright_black()
                    );
                }
                println!();
            }
            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
        }
    }

    async fn read_mcp_resource(&mut self, args: &str) {
        let uri = args.trim();
        if uri.is_empty() {
            println!("{} Usage: /mcp-read <uri>", "Info:".bright_yellow());
            return;
        }
        let Some(mcp) = &mut self.mcp_manager else {
            eprintln!("{} MCP not initialized", "Error:".bright_red());
            return;
        };
        let matches: Vec<_> = match mcp.list_resources().await {
            Ok(resources) => resources
                .into_iter()
                .filter(|(_, r)| r.uri == uri)
                .collect(),
            Err(e) => {
                eprintln!("{} {}", "Error:".bright_red(), e);
                return;
            }
        };
        if matches.is_empty() {
            eprintln!("{} No MCP resource with URI {}", "Error:".bright_red(), uri);
            return;
        }
        if matches.len() > 1 {
            eprintln!(
                "{} Resource URI is ambiguous across servers. Use /mcp-resources <server> first.",
                "Error:".bright_red()
            );
            return;
        }
        let server = &matches[0].0;
        match mcp.read_resource(server, uri).await {
            Ok(contents) => {
                println!(
                    "\n{} {} ({})",
                    "Resource:".bright_yellow().bold(),
                    uri.bright_cyan(),
                    server.bright_cyan()
                );
                for content in contents {
                    if let Some(text) = content.text {
                        println!("{}", text);
                    } else {
                        println!("{} {}", "ℹ".bright_blue(), content.uri);
                    }
                }
                println!();
            }
            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
        }
    }

    async fn reload_mcp(&mut self) -> Result<()> {
        // Shutdown existing MCP connections
        if let Some(mcp) = &mut self.mcp_manager {
            mcp.shutdown().await;
        }

        // Reload configuration and reconnect
        self.mcp_manager =
            match McpManager::new(Arc::clone(&self.permissions), Arc::clone(&self.plan_mode)).await
            {
                Ok(manager) => Some(manager),
                Err(e) => {
                    eprintln!("{} {}", "Warning:".bright_yellow(), e);
                    None
                }
            };

        Ok(())
    }

    fn show_help(&self) {
        println!("\n{}", "Available Commands:".bright_yellow().bold());
        for spec in COMMANDS {
            println!("  {} - {}", spec.usage.bright_cyan(), spec.help);
        }
        println!();
    }

    fn show_history(&self) {
        if self.history.is_empty() {
            println!("{}", "No conversation history yet.".yellow());
            return;
        }

        println!("\n{}", "Conversation History:".bright_yellow().bold());
        println!("{}", "-".repeat(60).bright_black());

        for (i, msg) in self.history.iter().enumerate() {
            let role = if msg.role == "user" {
                "You".bright_green().bold()
            } else {
                "AI".bright_blue().bold()
            };

            println!("{} [{}]: {}", role, i + 1, msg.content);
        }
        println!("{}\n", "-".repeat(60).bright_black());
    }

    fn show_status(&self) {
        let mcp_tool_count = self
            .mcp_manager
            .as_ref()
            .map(|m| m.list_tools().len())
            .unwrap_or(0);
        let cwd = std::env::current_dir().ok();
        let perms = self.permissions.lock().unwrap();
        let trusted_here = cwd.as_deref().map(|p| perms.contains(p)).unwrap_or(false);
        let trusted_count = perms.trusted_count();
        drop(perms);

        println!("\n{}", "Status:".bright_yellow().bold());
        println!("  {}: {}", "model".bright_cyan(), self.executor.get_model());
        println!("  {}: {}", "messages".bright_cyan(), self.history.len());
        println!(
            "  {}: {}",
            "plan mode".bright_cyan(),
            if self.plan_mode.load(Ordering::SeqCst) {
                "on".bright_green()
            } else {
                "off".bright_black()
            }
        );
        println!(
            "  {}: {} ({} pending)",
            "todos".bright_cyan(),
            self.todos.len(),
            self.todos.pending()
        );
        println!("  {}: {}", "mcp tools".bright_cyan(), mcp_tool_count);
        println!(
            "  {}: {} ({} trusted root{} total)",
            "cwd trust".bright_cyan(),
            if trusted_here {
                "trusted".bright_green()
            } else {
                "not trusted".bright_red()
            },
            trusted_count,
            if trusted_count == 1 { "" } else { "s" }
        );
        println!();
    }

    fn show_stats(&self) {
        let total = self.history.len();
        let user = self.history.iter().filter(|m| m.role == "user").count();
        let assistant = self
            .history
            .iter()
            .filter(|m| m.role == "assistant")
            .count();
        let system = self.history.iter().filter(|m| m.role == "system").count();
        let token_estimate = self
            .history
            .iter()
            .map(|m| m.content.len())
            .sum::<usize>()
            .div_ceil(4);

        println!("\n{}", "Session statistics:".bright_yellow().bold());
        println!("  {}: {}", "messages".bright_cyan(), total);
        println!(
            "  {}: user={}, assistant={}, system={}",
            "breakdown".bright_cyan(),
            user,
            assistant,
            system
        );
        println!(
            "  {}: {}",
            "session id".bright_cyan(),
            self.current_session
                .as_ref()
                .map(|s| s.id.as_str())
                .unwrap_or("<inactive>")
        );
        println!("  {}: {}", "model".bright_cyan(), self.executor.get_model());
        println!("  {}: ~{}", "tokens".bright_cyan(), token_estimate);
        if !self.session_stats.is_empty() {
            println!(
                "  {}: {} in / {} out (this run)",
                "provider tokens".bright_cyan(),
                self.session_stats.prompt_tokens,
                self.session_stats.completion_tokens
            );
            if self.session_stats.elapsed_ms > 0 {
                println!(
                    "  {}: {} ms (this run)",
                    "model time".bright_cyan(),
                    self.session_stats.elapsed_ms
                );
            }
        }
        println!(
            "  {}: {}/{} pending",
            "todos".bright_cyan(),
            self.todos.pending(),
            self.todos.len()
        );
        println!(
            "  {}: {}",
            "memdir entries".bright_cyan(),
            self.memdir.len()
        );
        println!();
    }

    /// `/doctor` — runs a sanity check on the runtime environment.
    /// Each probe prints a ✓ / ✗ line; the function never returns an error
    /// so users always see the full report.
    async fn run_doctor(&self) {
        println!("\n{}", "Doctor:".bright_yellow().bold());

        // 1. Ollama reachability + model listing.
        let ollama = crate::ollama::OllamaClient::new();
        let model = self.executor.get_model();
        match ollama.list_models().await {
            Ok(models) => {
                println!(
                    "  {} Ollama reachable at {} ({} model{} installed)",
                    "✓".bright_green(),
                    "http://localhost:11434".bright_cyan(),
                    models.len(),
                    if models.len() == 1 { "" } else { "s" }
                );
                // Mirror the startup check in main.rs (prefix match handles
                // `name` vs `name:latest`).
                if models.iter().any(|m| m.starts_with(model)) {
                    println!(
                        "  {} Current model '{}' is installed",
                        "✓".bright_green(),
                        model.bright_cyan()
                    );
                } else {
                    println!(
                        "  {} Current model '{}' is not installed (try `ollama pull {}`)",
                        "✗".bright_red(),
                        model.bright_cyan(),
                        model
                    );
                }
            }
            Err(e) => {
                println!(
                    "  {} Ollama unreachable at {}: {}",
                    "✗".bright_red(),
                    "http://localhost:11434".bright_cyan(),
                    e
                );
                println!(
                    "  {} Skipping model check (Ollama not reachable)",
                    "ℹ".bright_blue()
                );
            }
        }

        // 2. Config directory writable.
        match dirs::home_dir().map(|h| h.join(".cubi")) {
            Some(dir) => match fs::create_dir_all(&dir) {
                Ok(()) => {
                    // Use a unique probe filename and `create_new` so we never
                    // truncate a pre-existing file the user may have placed here.
                    let nanos = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos())
                        .unwrap_or(0);
                    let probe = dir.join(format!(
                        ".doctor-write-probe-{}-{}",
                        std::process::id(),
                        nanos
                    ));
                    match std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&probe)
                    {
                        Ok(mut f) => {
                            use std::io::Write;
                            let write_res = f.write_all(b"ok");
                            drop(f);
                            let _ = fs::remove_file(&probe);
                            match write_res {
                                Ok(()) => println!(
                                    "  {} Config dir writable: {}",
                                    "✓".bright_green(),
                                    dir.display().to_string().bright_cyan()
                                ),
                                Err(e) => println!(
                                    "  {} Config dir not writable ({}): {}",
                                    "✗".bright_red(),
                                    dir.display(),
                                    e
                                ),
                            }
                        }
                        Err(e) => println!(
                            "  {} Config dir not writable ({}): {}",
                            "✗".bright_red(),
                            dir.display(),
                            e
                        ),
                    }
                }
                Err(e) => println!(
                    "  {} Could not create config dir {}: {}",
                    "✗".bright_red(),
                    dir.display(),
                    e
                ),
            },
            None => println!("  {} Could not resolve home directory", "✗".bright_red()),
        }

        // 3. `git` on PATH.
        match std::process::Command::new("git").arg("--version").output() {
            Ok(out) if out.status.success() => {
                let v = String::from_utf8_lossy(&out.stdout);
                println!(
                    "  {} {} on PATH ({})",
                    "✓".bright_green(),
                    "git".bright_cyan(),
                    v.trim()
                );
            }
            Ok(out) => println!(
                "  {} `git --version` exited with status {}",
                "✗".bright_red(),
                out.status
            ),
            Err(e) => println!(
                "  {} {} not found on PATH: {}",
                "✗".bright_red(),
                "git".bright_cyan(),
                e
            ),
        }

        println!();
    }

    /// `/env` — prints the resolved runtime: model, project dir, trust
    /// status, plan mode, MCP server count, CUBI.md presence, memdir
    /// entries, session checkpoint count, todo count.
    fn show_env(&self) {
        let cwd = std::env::current_dir().ok();
        let perms = self.permissions.lock().unwrap();
        let trusted_here = cwd.as_deref().map(|p| perms.contains(p)).unwrap_or(false);
        let trusted_count = perms.trusted_count();
        drop(perms);

        let mcp_tool_count = self
            .mcp_manager
            .as_ref()
            .map(|m| m.list_tools().len())
            .unwrap_or(0);

        let aichat_path = project_memory::read_memory_with_path()
            .ok()
            .flatten()
            .map(|(p, _)| p);

        let session_count = self
            .session_store
            .as_ref()
            .and_then(|s| s.list().ok())
            .map(|v| v.len())
            .unwrap_or(0);

        println!("\n{}", "Environment:".bright_yellow().bold());
        println!(
            "  {}: {} {}",
            "binary".bright_cyan(),
            "cubi".bright_white(),
            format!("v{}", env!("CARGO_PKG_VERSION")).bright_black()
        );
        println!("  {}: {}", "model".bright_cyan(), self.executor.get_model());
        println!(
            "  {}: {}",
            "cwd".bright_cyan(),
            cwd.as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "<unknown>".to_string())
        );
        println!(
            "  {}: {} ({} trusted root{} total)",
            "cwd trust".bright_cyan(),
            if trusted_here {
                "trusted".bright_green()
            } else {
                "not trusted".bright_red()
            },
            trusted_count,
            if trusted_count == 1 { "" } else { "s" }
        );
        println!(
            "  {}: {}",
            "plan mode".bright_cyan(),
            if self.plan_mode.load(Ordering::SeqCst) {
                "on".bright_green()
            } else {
                "off".bright_black()
            }
        );
        println!(
            "  {}: {}",
            "history messages".bright_cyan(),
            self.history.len()
        );
        println!("  {}: {}", "mcp tools".bright_cyan(), mcp_tool_count);
        println!(
            "  {}: {} ({} pending)",
            "todos".bright_cyan(),
            self.todos.len(),
            self.todos.pending()
        );
        println!(
            "  {}: {} entries",
            "memdir".bright_cyan(),
            self.memdir.len()
        );
        println!(
            "  {}: {}",
            "session checkpoints".bright_cyan(),
            session_count
        );
        match aichat_path {
            Some(p) => println!(
                "  {}: {}",
                "project memory".bright_cyan(),
                p.display().to_string().bright_cyan()
            ),
            None => println!(
                "  {}: {}",
                "project memory".bright_cyan(),
                "(none — run /init to create CUBI.md)".bright_black()
            ),
        }
        println!();
    }

    /// `/config` — print the contents of `~/.cubi/config.json`.
    fn show_config(&self) {
        let Some(path) = crate::onboarding::AppConfig::storage_path() else {
            eprintln!(
                "{} Could not resolve home directory.",
                "Error:".bright_red()
            );
            return;
        };
        println!(
            "\n{} ({}):",
            "Config".bright_yellow().bold(),
            path.display().to_string().bright_cyan()
        );
        println!("{}", "-".repeat(60).bright_black());
        match fs::read_to_string(&path) {
            Ok(raw) => println!("{}", raw.trim_end()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => println!(
                "{} No config file yet — it is written on first onboarding.",
                "ℹ".bright_blue()
            ),
            Err(e) => eprintln!("{} Failed to read config: {}", "Error:".bright_red(), e),
        }
        println!("{}\n", "-".repeat(60).bright_black());
    }

    /// `/permissions` — list trusted directories and the built-in tools
    /// gated by the trust store.
    fn show_permissions(&self) {
        let perms = self.permissions.lock().unwrap();
        let roots: Vec<_> = perms.trusted_roots().cloned().collect();
        let allowed: Vec<_> = perms.allowed_tools().cloned().collect();
        let denied: Vec<_> = perms.denied_tools().cloned().collect();
        drop(perms);

        let store_path = dirs::home_dir().map(|h| h.join(".cubi").join("trusted_dirs.json"));

        println!("\n{}", "Permissions:".bright_yellow().bold());
        if let Some(p) = &store_path {
            println!(
                "  {}: {}",
                "trust store".bright_cyan(),
                p.display().to_string().bright_cyan()
            );
        }
        if roots.is_empty() {
            println!(
                "  {} No directories are currently trusted. Run /trust in a project to approve it.",
                "ℹ".bright_blue()
            );
        } else {
            println!(
                "  {} {} trusted root{}:",
                "✓".bright_green(),
                roots.len(),
                if roots.len() == 1 { "" } else { "s" }
            );
            for (i, r) in roots.iter().enumerate() {
                println!("    {}. {}", i + 1, r.display().to_string().bright_cyan());
            }
        }
        println!(
            "  {}: bash, write_file, edit_file (write/exec only inside a trusted root)",
            "gated tools".bright_cyan()
        );
        println!(
            "  {}: {}",
            "allowed tools".bright_cyan(),
            if allowed.is_empty() {
                "all tools allowed unless denied".to_string()
            } else {
                allowed.join(", ")
            }
        );
        println!(
            "  {}: {}",
            "denied tools".bright_cyan(),
            if denied.is_empty() {
                "none".to_string()
            } else {
                denied.join(", ")
            }
        );
        if !self.policy.denied_tools.is_empty() || self.policy.note.is_some() {
            let policy_path = Policy::active_path()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "<none>".to_string());
            println!(
                "  {}: {} (from {})",
                "admin policy denies".bright_red(),
                if self.policy.denied_tools.is_empty() {
                    "(none)".to_string()
                } else {
                    self.policy
                        .denied_tools
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                },
                policy_path.bright_black()
            );
            if let Some(note) = &self.policy.note {
                println!("    {} {}", "note:".bright_black(), note);
            }
        }
        println!(
            "  {} Plan mode also blocks write/exec tools regardless of trust.",
            "ℹ".bright_blue()
        );
        println!();
    }

    fn handle_tool_allow(&self, args: &str) {
        let tool = args.trim();
        if tool.is_empty() {
            println!("{} Usage: /tool-allow <name>", "Info:".bright_yellow());
            return;
        }
        let mut perms = self.permissions.lock().unwrap();
        perms.allow_tool(tool);
        match perms.save() {
            Ok(()) => println!("{} Allowed tool {}", "✓".bright_green(), tool.bright_cyan()),
            Err(e) => eprintln!(
                "{} Failed to persist permissions: {}",
                "Error:".bright_red(),
                e
            ),
        }
    }

    fn handle_tool_deny(&self, args: &str) {
        let tool = args.trim();
        if tool.is_empty() {
            println!("{} Usage: /tool-deny <name>", "Info:".bright_yellow());
            return;
        }
        let mut perms = self.permissions.lock().unwrap();
        perms.deny_tool(tool);
        match perms.save() {
            Ok(()) => println!("{} Denied tool {}", "✓".bright_green(), tool.bright_cyan()),
            Err(e) => eprintln!(
                "{} Failed to persist permissions: {}",
                "Error:".bright_red(),
                e
            ),
        }
    }

    /// `/bug` — print a pre-filled GitHub Issues URL for this repo with
    /// the runtime info from `/env` URL-encoded into the body. Optional
    /// `args` become the issue title.
    fn show_bug_url(&self, args: &str) {
        let title = if args.trim().is_empty() {
            "Bug report".to_string()
        } else {
            args.trim().to_string()
        };

        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "<unknown>".to_string());
        let mcp_tool_count = self
            .mcp_manager
            .as_ref()
            .map(|m| m.list_tools().len())
            .unwrap_or(0);
        let plan_mode = if self.plan_mode.load(Ordering::SeqCst) {
            "on"
        } else {
            "off"
        };

        let body = format!(
            "## Describe the bug\n\
             <!-- A clear and concise description. -->\n\n\
             ## To reproduce\n\
             1. ...\n\n\
             ## Expected behavior\n\
             ...\n\n\
             ## Environment\n\
             - cubi: v{version}\n\
             - model: {model}\n\
             - cwd: {cwd}\n\
             - plan mode: {plan_mode}\n\
             - mcp tools: {mcp}\n\
             - os: {os} ({arch})\n",
            version = env!("CARGO_PKG_VERSION"),
            model = self.executor.get_model(),
            cwd = cwd,
            plan_mode = plan_mode,
            mcp = mcp_tool_count,
            os = std::env::consts::OS,
            arch = std::env::consts::ARCH,
        );

        let url = format!(
            "https://github.com/peterchoi1014/cubi/issues/new?title={}&body={}",
            url_encode(&title),
            url_encode(&body),
        );

        println!("\n{}", "Bug report:".bright_yellow().bold());
        println!(
            "  {} Open this URL to file an issue with runtime info pre-filled:\n",
            "ℹ".bright_blue()
        );
        println!("  {}\n", url.bright_cyan());
    }

    /// Handles `/trust` and `/trust revoke`. Both forms operate on the
    /// current working directory.
    fn show_issue_url(&self, args: &str) {
        let title = if args.trim().is_empty() {
            "Feature request".to_string()
        } else {
            args.trim().to_string()
        };
        let body = "## Problem\n<!-- What limitation are you hitting? -->\n\n## Proposed solution\n<!-- Describe the feature you want. -->\n\n## Alternatives considered\n<!-- Optional -->\n";
        let url = format!(
            "https://github.com/peterchoi1014/cubi/issues/new?labels=enhancement&title={}&body={}",
            url_encode(&title),
            url_encode(body),
        );

        println!("\n{}", "Feature request:".bright_yellow().bold());
        println!(
            "  {} Open this URL to file a feature request:\n",
            "ℹ".bright_blue()
        );
        println!("  {}\n", url.bright_cyan());
    }

    fn load_global_hooks(&self) -> Vec<HookDef> {
        let Some(path) = crate::hooks::global_hooks_path() else {
            return Vec::new();
        };
        fs::read_to_string(path)
            .ok()
            .and_then(|raw| serde_json::from_str::<HooksConfig>(&raw).ok())
            .map(|cfg| cfg.hooks)
            .unwrap_or_default()
    }

    fn handle_hooks(&mut self, args: &str) {
        let trimmed = args.trim();
        let mut hooks = self.load_global_hooks();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("list") {
            println!("\n{}", "Hooks:".bright_yellow().bold());
            if hooks.is_empty() {
                println!("  {} No hooks configured.", "ℹ".bright_blue());
            } else {
                for (idx, hook) in hooks.iter().enumerate() {
                    println!(
                        "  {}. {} {} {}",
                        idx + 1,
                        hook.event.as_str().bright_cyan(),
                        hook.match_tool
                            .as_deref()
                            .map(|m| format!("[match_tool={}]", m))
                            .unwrap_or_default()
                            .bright_black(),
                        hook.command.bright_white()
                    );
                }
            }
            println!();
            return;
        }

        if let Some(rest) = trimmed.strip_prefix("add ") {
            let mut parts = rest.trim().splitn(2, char::is_whitespace);
            let event_name = parts.next().unwrap_or("");
            let command = parts.next().unwrap_or("").trim();
            let Some(event) = HookEvent::parse(event_name) else {
                eprintln!(
                    "{} Unknown hook event '{}'.",
                    "Error:".bright_red(),
                    event_name
                );
                return;
            };
            if command.is_empty() {
                println!(
                    "{} Usage: /hooks add <event> <command>",
                    "Info:".bright_yellow()
                );
                return;
            }
            hooks.push(HookDef {
                event,
                match_tool: None,
                command: command.to_string(),
            });
            match crate::hooks::save_global(&hooks) {
                Ok(()) => {
                    self.hooks = HookRegistry::load();
                    println!("{} Hook added.", "✓".bright_green());
                }
                Err(e) => eprintln!("{} Failed to save hooks: {}", "Error:".bright_red(), e),
            }
            return;
        }

        if let Some(rest) = trimmed.strip_prefix("rm ") {
            let Ok(index) = rest.trim().parse::<usize>() else {
                eprintln!("{} Usage: /hooks rm <n>", "Error:".bright_red());
                return;
            };
            if index == 0 || index > hooks.len() {
                eprintln!("{} No hook with index {}", "Error:".bright_red(), index);
                return;
            }
            hooks.remove(index - 1);
            match crate::hooks::save_global(&hooks) {
                Ok(()) => {
                    self.hooks = HookRegistry::load();
                    println!("{} Hook removed.", "✓".bright_green());
                }
                Err(e) => eprintln!("{} Failed to save hooks: {}", "Error:".bright_red(), e),
            }
            return;
        }

        println!(
            "{} Usage: /hooks [list | add <event> <cmd> | rm <n>]",
            "Info:".bright_yellow()
        );
    }

    async fn handle_skills(&mut self, args: &str) {
        let trimmed = args.trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("list") {
            println!("\n{}", "Skills:".bright_yellow().bold());
            if self.skills.is_empty() {
                println!("  {} No skills found in ~/.cubi/skills", "ℹ".bright_blue());
            } else {
                for skill in &self.skills {
                    println!(
                        "  {} - {}",
                        skill.name.bright_cyan(),
                        skill.description.bright_white()
                    );
                }
            }
            println!();
            return;
        }

        let Some(name) = trimmed.strip_prefix("run ") else {
            println!(
                "{} Usage: /skills [list|run <name>]",
                "Info:".bright_yellow()
            );
            return;
        };
        let lookup = name.trim().to_ascii_lowercase();
        let Some(skill) = self.skills.iter().find(|s| s.name == lookup).cloned() else {
            eprintln!("{} Unknown skill '{}'", "Error:".bright_red(), name.trim());
            return;
        };

        self.history.push(Message::text(
            "system",
            format!(
                "{} Skill /{} (from {}):\n\n{}",
                SINGLE_TURN_SYSTEM_TAG,
                skill.name,
                skill.path.display(),
                skill.body
            ),
        ));
        println!(
            "{} Applied skill /{}",
            "✓".bright_green(),
            skill.name.bright_cyan()
        );
        self.journal.start_turn();
        // For /skills, the user "message" is the system-injected skill body;
        // there's no plain user turn to roll back, so snapshot whatever the
        // history length happens to be now.
        let turn_start = self.history.len();
        if let Err(e) = self.agent_turn(turn_start).await {
            eprintln!("{} {}", "Error:".bright_red(), e);
        }
    }

    fn handle_trust(&self, args: &str) {
        let cwd = match std::env::current_dir() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("{} Could not read cwd: {}", "Error:".bright_red(), e);
                return;
            }
        };
        let mut perms = self.permissions.lock().unwrap();
        let result = match args.trim() {
            "" | "add" => perms.trust_dir(&cwd).map(|added| (added, true)),
            "revoke" | "remove" | "rm" => perms.revoke_dir(&cwd).map(|removed| (removed, false)),
            other => {
                eprintln!(
                    "{} Unknown argument '{}'. Usage: /trust [revoke]",
                    "Error:".bright_red(),
                    other
                );
                return;
            }
        };
        match result {
            Ok((changed, added)) => {
                if let Err(e) = perms.save() {
                    eprintln!(
                        "{} Failed to persist trust store: {}",
                        "Warn:".bright_yellow(),
                        e
                    );
                }
                let verb = if added { "trusted" } else { "revoked" };
                if changed {
                    println!(
                        "{} {} {}",
                        "✓".bright_green(),
                        verb.bright_green(),
                        cwd.display().to_string().bright_cyan()
                    );
                } else if added {
                    println!(
                        "{} {} was already trusted",
                        "ℹ".bright_blue(),
                        cwd.display().to_string().bright_cyan()
                    );
                } else {
                    println!(
                        "{} {} was not in the trust list",
                        "ℹ".bright_blue(),
                        cwd.display().to_string().bright_cyan()
                    );
                }
            }
            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
        }
    }

    fn handle_sessions(&mut self, args: &str) {
        let trimmed = args.trim();
        if let Some(rest) = trimmed.strip_prefix("delete") {
            let id = rest.trim();
            if id.is_empty() {
                println!(
                    "{} Usage: /sessions delete <id-or-prefix>",
                    "Info:".bright_yellow()
                );
                return;
            }
            self.delete_session_by_prefix(id);
            return;
        }
        if !trimmed.is_empty() {
            println!(
                "{} Usage: /sessions [delete <id-or-prefix>]",
                "Info:".bright_yellow()
            );
            return;
        }
        self.show_sessions();
    }

    fn delete_session_by_prefix(&mut self, id: &str) {
        match SessionStore::find_by_prefix(id) {
            Ok(FindSessionResult::Found(meta)) => {
                if !self.confirm_session_delete(&meta.id) {
                    println!("{} Delete cancelled.", "ℹ".bright_blue());
                    return;
                }
                match SessionStore::delete_by_prefix(&meta.id) {
                    Ok(DeleteSessionResult::Deleted(meta)) => {
                        if self
                            .current_session
                            .as_ref()
                            .map(|s| s.id == meta.id)
                            .unwrap_or(false)
                        {
                            self.current_session = None;
                        }
                        println!("Deleted session {}", meta.id.bright_cyan());
                    }
                    Ok(DeleteSessionResult::NotFound) => {
                        eprintln!("cubi: no session matches '{}'.", id);
                    }
                    Ok(DeleteSessionResult::Ambiguous(_)) => {
                        eprintln!("cubi: session disappeared while deleting '{}'.", id);
                    }
                    Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
                }
            }
            Ok(FindSessionResult::NotFound) => eprintln!("cubi: no session matches '{}'.", id),
            Ok(FindSessionResult::Ambiguous(candidates)) => {
                eprintln!("cubi: session prefix '{}' is ambiguous. Candidates:", id);
                for meta in candidates {
                    eprintln!("  {}  {}", meta.id, meta.cwd);
                }
            }
            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
        }
    }

    fn confirm_session_delete(&self, id: &str) -> bool {
        use std::io::{self, Write};
        print!("Delete session {}? [y/N] ", id.bright_cyan());
        let _ = io::stdout().flush();
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .map(|_| matches!(input.trim().to_ascii_lowercase().as_str(), "y" | "yes"))
            .unwrap_or(false)
    }

    /// Lists checkpointed sessions for the current project, newest first.
    fn show_sessions(&self) {
        let Some(store) = &self.session_store else {
            println!(
                "{} Sessions disabled: could not resolve home directory.",
                "ℹ".bright_blue()
            );
            return;
        };
        match store.list() {
            Ok(list) if list.is_empty() => {
                println!(
                    "{} No sessions saved yet for this project.",
                    "ℹ".bright_blue()
                );
            }
            Ok(list) => {
                println!("\n{}", "Sessions (newest first):".bright_yellow().bold());
                for (i, meta) in list.iter().enumerate() {
                    let active = self
                        .current_session
                        .as_ref()
                        .map(|s| s.id == meta.id)
                        .unwrap_or(false);
                    let marker = if active { "▶" } else { " " };
                    println!(
                        "  {} {}. {} ({} msgs, model={}) {}",
                        marker.bright_green(),
                        i + 1,
                        meta.id.bright_cyan(),
                        meta.message_count,
                        meta.model.bright_magenta(),
                        if meta.preview.is_empty() {
                            String::new()
                        } else {
                            format!("— {}", meta.preview.bright_white())
                        }
                    );
                }
                println!(
                    "\nUse {} to resume the most recent, or {} <id> to resume a specific one.\n",
                    "/resume".bright_cyan(),
                    "/resume".bright_cyan()
                );
            }
            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
        }
    }

    /// Resumes the latest session, or a named one if `args` is non-empty.
    pub fn resume_session(&mut self, args: &str) {
        let Some(store) = &self.session_store else {
            println!(
                "{} Sessions disabled: could not resolve home directory.",
                "ℹ".bright_blue()
            );
            return;
        };
        let target = args.trim();
        let loaded = if target.is_empty() {
            store.latest_for_current_dir_preferred()
        } else {
            store.load(target)
        };
        match loaded {
            Ok(Some(session)) => {
                println!(
                    "{} Resumed session {} ({} messages)",
                    "✓".bright_green(),
                    session.id.bright_cyan(),
                    session.history.len()
                );
                self.history = session.history.clone();
                self.pinned = session.pinned.clone();
                self.current_session = Some(session);
                // Re-inject project memory so resumed sessions see the
                // current `CUBI.md`, not a snapshot from when the
                // session was first created.
                self.inject_project_memory();
            }
            Ok(None) => {
                if target.is_empty() {
                    println!(
                        "{} No sessions to resume. Start chatting and one will be created.",
                        "ℹ".bright_blue()
                    );
                } else {
                    eprintln!(
                        "{} No session with id '{}'. Use /sessions to list.",
                        "Error:".bright_red(),
                        target
                    );
                }
            }
            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
        }
    }

    /// Writes the current history to the per-project session store.
    /// Lazily allocates a new session on first call. Failures are
    /// logged as warnings but never abort the chat — the user always
    /// has the in-memory copy and can `/save` manually.
    fn checkpoint_session(&mut self) {
        let Some(store) = &self.session_store else {
            return;
        };
        if self.current_session.is_none() {
            self.current_session = Some(store.new_session(self.executor.get_model().to_string()));
        }
        if let Some(session) = self.current_session.as_mut() {
            // Refresh the model field so a `/model <name>` mid-session
            // is reflected in the snapshot.
            session.model = self.executor.get_model().to_string();
            session.history = self.history.clone();
            session.pinned = self.pinned.clone();
            if let Err(e) = store.save(session) {
                eprintln!(
                    "{} Failed to checkpoint session: {}",
                    "Warn:".bright_yellow(),
                    e
                );
            }
        }
    }

    /// `/diff [path]` — print the current `git diff`. Empty diffs get a
    /// short hint instead of a blank line so the user knows the command
    /// actually ran.
    fn run_diff(&self, args: &str) {
        match git_cmds::diff(args) {
            Ok(out) => {
                if !out.exit_ok {
                    eprintln!(
                        "{} git diff failed: {}",
                        "Error:".bright_red(),
                        out.stderr.trim()
                    );
                    return;
                }
                if out.stdout.trim().is_empty() {
                    println!("{} No changes in the working tree.", "ℹ".bright_blue());
                } else {
                    print!("{}", out.stdout);
                    if !out.stderr.trim().is_empty() {
                        eprintln!("{}", out.stderr.trim().bright_black());
                    }
                }
            }
            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
        }
    }

    /// `/commit [-a] <msg>` — wrap `git commit`. Always echoes git's own
    /// stdout/stderr so the user sees the resulting commit summary.
    fn run_commit(&self, args: &str) {
        let Some((stage_all, msg)) = git_cmds::parse_commit_args(args) else {
            println!("{} Usage: /commit [-a] <message>", "Info:".bright_yellow());
            println!("       -a stages tracked files before committing.");
            return;
        };
        match git_cmds::commit(stage_all, msg) {
            Ok(out) => {
                if out.exit_ok {
                    if !out.stdout.trim().is_empty() {
                        print!("{}", out.stdout);
                    }
                    println!("{} Commit created.", "✓".bright_green());
                } else {
                    eprintln!(
                        "{} git commit failed (exit non-zero).",
                        "Error:".bright_red()
                    );
                    if !out.stdout.trim().is_empty() {
                        eprintln!("{}", out.stdout.trim());
                    }
                    if !out.stderr.trim().is_empty() {
                        eprintln!("{}", out.stderr.trim());
                    }
                }
            }
            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
        }
    }

    fn run_undo(&self, args: &str) {
        if self.plan_mode.load(Ordering::SeqCst) {
            println!(
                "{} Plan mode is on — refusing /undo. Toggle off with /plan first.",
                "✗".bright_red()
            );
            return;
        }
        let cwd = match std::env::current_dir() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("{} Could not read cwd: {}", "Error:".bright_red(), e);
                return;
            }
        };
        if let Err(e) = self.permissions.lock().unwrap().check_exec(&cwd) {
            eprintln!("{} {}", "Error:".bright_red(), e);
            return;
        }

        let hard = matches!(args.trim(), "hard");
        if !args.trim().is_empty() && !hard {
            println!("{} Usage: /undo [hard]", "Info:".bright_yellow());
            return;
        }

        let result = if hard {
            git_cmds::git_reset_hard_head1()
        } else {
            git_cmds::git_revert_head()
        };

        match result {
            Ok(out) if out.exit_ok => {
                if !out.stdout.trim().is_empty() {
                    print!("{}", out.stdout);
                }
                if !out.stderr.trim().is_empty() {
                    eprintln!("{}", out.stderr.trim());
                }
                println!(
                    "{} {}",
                    "✓".bright_green(),
                    if hard {
                        "Reset HEAD to HEAD~1."
                    } else {
                        "Reverted HEAD."
                    }
                );
            }
            Ok(out) => {
                eprintln!("{} git undo failed.", "Error:".bright_red());
                if !out.stdout.trim().is_empty() {
                    eprintln!("{}", out.stdout.trim());
                }
                if !out.stderr.trim().is_empty() {
                    eprintln!("{}", out.stderr.trim());
                }
            }
            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
        }
    }

    fn run_commit_push_pr(&self, args: &str) {
        if self.plan_mode.load(Ordering::SeqCst) {
            println!(
                "{} Plan mode is on — refusing /commit-push-pr. Toggle off with /plan first.",
                "✗".bright_red()
            );
            return;
        }
        let cwd = match std::env::current_dir() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("{} Could not read cwd: {}", "Error:".bright_red(), e);
                return;
            }
        };
        if let Err(e) = self.permissions.lock().unwrap().check_exec(&cwd) {
            eprintln!("{} {}", "Error:".bright_red(), e);
            return;
        }

        let Some((stage_all, msg)) = git_cmds::parse_commit_args(args) else {
            println!(
                "{} Usage: /commit-push-pr [-a] <message>",
                "Info:".bright_yellow()
            );
            return;
        };

        let commit = match git_cmds::commit(stage_all, msg) {
            Ok(out) if out.exit_ok => out,
            Ok(out) => {
                eprintln!("{} git commit failed.", "Error:".bright_red());
                if !out.stdout.trim().is_empty() {
                    eprintln!("{}", out.stdout.trim());
                }
                if !out.stderr.trim().is_empty() {
                    eprintln!("{}", out.stderr.trim());
                }
                return;
            }
            Err(e) => {
                eprintln!("{} {}", "Error:".bright_red(), e);
                return;
            }
        };
        if !commit.stdout.trim().is_empty() {
            print!("{}", commit.stdout);
        }
        if !commit.stderr.trim().is_empty() {
            eprintln!("{}", commit.stderr.trim());
        }

        let push = match git_cmds::git_push() {
            Ok(out) if out.exit_ok => out,
            Ok(out) => {
                eprintln!("{} git push failed.", "Error:".bright_red());
                if !out.stdout.trim().is_empty() {
                    eprintln!("{}", out.stdout.trim());
                }
                if !out.stderr.trim().is_empty() {
                    eprintln!("{}", out.stderr.trim());
                }
                return;
            }
            Err(e) => {
                eprintln!("{} {}", "Error:".bright_red(), e);
                return;
            }
        };
        if !push.stdout.trim().is_empty() {
            print!("{}", push.stdout);
        }
        if !push.stderr.trim().is_empty() {
            eprintln!("{}", push.stderr.trim());
        }

        let branch = match git_cmds::current_branch() {
            Ok(out) if out.exit_ok => out.stdout.trim().to_string(),
            Ok(out) => {
                eprintln!(
                    "{} Failed to read current branch: {}",
                    "Error:".bright_red(),
                    out.stderr.trim()
                );
                return;
            }
            Err(e) => {
                eprintln!("{} {}", "Error:".bright_red(), e);
                return;
            }
        };
        let remote = match git_cmds::remote_get_url("origin") {
            Ok(out) if out.exit_ok => out.stdout.trim().to_string(),
            Ok(out) => {
                eprintln!(
                    "{} Failed to read origin URL: {}",
                    "Error:".bright_red(),
                    out.stderr.trim()
                );
                return;
            }
            Err(e) => {
                eprintln!("{} {}", "Error:".bright_red(), e);
                return;
            }
        };
        let Some((owner, repo)) = github_repo_from_remote(&remote) else {
            eprintln!(
                "{} Could not parse GitHub origin URL: {}",
                "Error:".bright_red(),
                remote
            );
            return;
        };
        let url = format!(
            "https://github.com/{owner}/{repo}/compare/{}?expand=1",
            url_encode(&branch)
        );
        println!("{} Commit pushed. Open this PR URL:", "✓".bright_green());
        println!("  {}", url.bright_cyan());
    }

    /// `/review` — ask the model to review the current `git diff HEAD`.
    /// The exchange is transient: it's printed to stdout but not added
    /// to the persistent conversation history, so successive `/review`
    /// calls don't accumulate stale diffs in context.
    async fn run_review(&self) {
        let out = match git_cmds::diff_for_review() {
            Ok(out) => out,
            Err(e) => {
                eprintln!("{} {}", "Error:".bright_red(), e);
                return;
            }
        };
        if !out.exit_ok {
            eprintln!(
                "{} git diff failed: {}",
                "Error:".bright_red(),
                out.stderr.trim()
            );
            return;
        }
        if out.stdout.trim().is_empty() {
            println!(
                "{} No changes to review (working tree clean against HEAD).",
                "ℹ".bright_blue()
            );
            return;
        }

        // Cap the diff so we don't blow the model's context on a huge
        // change. The truncation marker tells both the model and the
        // user that the bottom of the diff was omitted.
        const MAX_DIFF_CHARS: usize = 20_000;
        let (diff_for_model, truncated) = if out.stdout.len() > MAX_DIFF_CHARS {
            let truncated: String = out.stdout.chars().take(MAX_DIFF_CHARS).collect();
            (truncated, true)
        } else {
            (out.stdout.clone(), false)
        };

        let truncation_note = if truncated {
            "\n\n[diff was truncated for length; review only what's shown above]"
        } else {
            ""
        };
        let review_messages = vec![
            Message::text(
                "system",
                "You are a focused code reviewer. Read the diff and respond with: \
                 (1) a one-sentence summary, (2) bugs or correctness issues, \
                 (3) style/clarity nits, (4) any tests that look missing. \
                 Be terse — use bullet points, no preamble.",
            ),
            Message::text(
                "user",
                format!(
                    "Please review this `git diff`:\n\n```diff\n{}\n```{}",
                    diff_for_model, truncation_note
                ),
            ),
        ];

        println!(
            "{} Asking the model to review the diff...",
            "⚙".bright_blue()
        );
        match self.executor.chat(review_messages).await {
            Ok(response) => {
                println!("\n{}\n", "Review:".bright_yellow().bold());
                println!("{}\n", response.bright_white());
                if truncated {
                    println!(
                        "{} Diff was truncated to {} characters; re-run after splitting \
                         the change for a complete review.",
                        "ℹ".bright_blue(),
                        MAX_DIFF_CHARS
                    );
                }
            }
            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
        }
    }

    /// `/worktree [list|add <path> [branch]|remove <path>]` — thin wrapper
    /// over `git worktree`. Mutating subcommands (`add`/`remove`) are
    /// refused in plan mode and require the cwd to be trusted, mirroring
    /// the `worktree` builtin tool. `add` also auto-trusts the new path
    /// so subsequent write/exec tool calls there don't fail.
    fn run_worktree(&self, args: &str) {
        let Some(action) = git_cmds::parse_worktree_args(args) else {
            println!(
                "{} Usage: /worktree [list | add <path> [branch] | remove <path>]",
                "Info:".bright_yellow()
            );
            return;
        };

        let mutating = !matches!(action, git_cmds::WorktreeAction::List);
        if mutating && self.plan_mode.load(Ordering::SeqCst) {
            println!(
                "{} Plan mode is on — refusing /worktree {}. Toggle off with /plan first.",
                "✗".bright_red(),
                match action {
                    git_cmds::WorktreeAction::Add { .. } => "add",
                    git_cmds::WorktreeAction::Remove { .. } => "remove",
                    git_cmds::WorktreeAction::List => unreachable!(),
                }
            );
            return;
        }
        if mutating {
            let cwd = match std::env::current_dir() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("{} Could not read cwd: {}", "Error:".bright_red(), e);
                    return;
                }
            };
            if let Err(e) = self.permissions.lock().unwrap().check_exec(&cwd) {
                eprintln!("{} {}", "Error:".bright_red(), e);
                return;
            }
        }

        let result = match action {
            git_cmds::WorktreeAction::List => git_cmds::worktree_list(),
            git_cmds::WorktreeAction::Add { path, branch } => git_cmds::worktree_add(path, branch),
            git_cmds::WorktreeAction::Remove { path } => git_cmds::worktree_remove(path),
        };

        let out = match result {
            Ok(out) => out,
            Err(e) => {
                eprintln!("{} {}", "Error:".bright_red(), e);
                return;
            }
        };

        if !out.exit_ok {
            eprintln!(
                "{} git worktree failed: {}",
                "Error:".bright_red(),
                out.stderr.trim()
            );
            if !out.stdout.trim().is_empty() {
                eprintln!("{}", out.stdout.trim());
            }
            return;
        }

        if let git_cmds::WorktreeAction::Add { path, .. } = action {
            // Auto-trust the new worktree path, matching the `worktree`
            // builtin tool's behavior.
            let trust_msg = {
                let mut perms = self.permissions.lock().unwrap();
                match perms.trust_dir(Path::new(path)) {
                    Ok(true) => match perms.save() {
                        Ok(()) => " (auto-trusted)".to_string(),
                        Err(e) => format!(" (auto-trusted in-memory but failed to persist: {e})"),
                    },
                    Ok(false) => " (already trusted)".to_string(),
                    Err(e) => format!(" (could not auto-trust: {e})"),
                }
            };
            println!(
                "{} Worktree created at {}{}",
                "✓".bright_green(),
                path.bright_cyan(),
                trust_msg
            );
        } else if let git_cmds::WorktreeAction::Remove { path } = action {
            println!(
                "{} Worktree removed: {}",
                "✓".bright_green(),
                path.bright_cyan()
            );
        }

        if !out.stdout.trim().is_empty() {
            print!("{}", out.stdout);
        }
        if !out.stderr.trim().is_empty() {
            eprintln!("{}", out.stderr.trim().bright_black());
        }
    }

    /// `/branch [list|create <name>|switch <name>]` — thin wrapper over
    /// `git branch` / `git switch`. Mutating subcommands are plan-mode
    /// gated and require trust, same as `/commit`.
    fn run_branch(&self, args: &str) {
        let Some(action) = git_cmds::parse_branch_args(args) else {
            println!(
                "{} Usage: /branch [list | create <name> | switch <name>]",
                "Info:".bright_yellow()
            );
            return;
        };

        let mutating = !matches!(action, git_cmds::BranchAction::List);
        if mutating && self.plan_mode.load(Ordering::SeqCst) {
            println!(
                "{} Plan mode is on — refusing /branch {}. Toggle off with /plan first.",
                "✗".bright_red(),
                match action {
                    git_cmds::BranchAction::Create { .. } => "create",
                    git_cmds::BranchAction::Switch { .. } => "switch",
                    git_cmds::BranchAction::List => unreachable!(),
                }
            );
            return;
        }
        if mutating {
            let cwd = match std::env::current_dir() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("{} Could not read cwd: {}", "Error:".bright_red(), e);
                    return;
                }
            };
            if let Err(e) = self.permissions.lock().unwrap().check_exec(&cwd) {
                eprintln!("{} {}", "Error:".bright_red(), e);
                return;
            }
        }

        let result = match action {
            git_cmds::BranchAction::List => git_cmds::branch_list(),
            git_cmds::BranchAction::Create { name } => git_cmds::branch_create(name),
            git_cmds::BranchAction::Switch { name } => git_cmds::branch_switch(name),
        };

        match result {
            Ok(out) => {
                if !out.exit_ok {
                    eprintln!(
                        "{} git branch failed: {}",
                        "Error:".bright_red(),
                        out.stderr.trim()
                    );
                    return;
                }
                if !out.stdout.trim().is_empty() {
                    print!("{}", out.stdout);
                }
                if !out.stderr.trim().is_empty() {
                    eprintln!("{}", out.stderr.trim().bright_black());
                }
                match action {
                    git_cmds::BranchAction::Create { name } => println!(
                        "{} Branch {} created.",
                        "✓".bright_green(),
                        name.bright_cyan()
                    ),
                    git_cmds::BranchAction::Switch { name } => println!(
                        "{} Switched to branch {}.",
                        "✓".bright_green(),
                        name.bright_cyan()
                    ),
                    git_cmds::BranchAction::List => {}
                }
            }
            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
        }
    }

    /// `/tag [list|<name>|create <name> [-m <msg>]]` — thin wrapper over
    /// `git tag`. Creation is plan-mode gated and requires trust.
    fn run_tag(&self, args: &str) {
        let Some(action) = git_cmds::parse_tag_args(args) else {
            println!(
                "{} Usage: /tag [list | <name> | create <name> [-m <msg>]]",
                "Info:".bright_yellow()
            );
            return;
        };

        let mutating = !matches!(action, git_cmds::TagAction::List);
        if mutating && self.plan_mode.load(Ordering::SeqCst) {
            println!(
                "{} Plan mode is on — refusing /tag create. Toggle off with /plan first.",
                "✗".bright_red()
            );
            return;
        }
        if mutating {
            let cwd = match std::env::current_dir() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("{} Could not read cwd: {}", "Error:".bright_red(), e);
                    return;
                }
            };
            if let Err(e) = self.permissions.lock().unwrap().check_exec(&cwd) {
                eprintln!("{} {}", "Error:".bright_red(), e);
                return;
            }
        }

        let result = match action {
            git_cmds::TagAction::List => git_cmds::tag_list(),
            git_cmds::TagAction::Create { name, message } => git_cmds::tag_create(name, message),
        };

        match result {
            Ok(out) => {
                if !out.exit_ok {
                    eprintln!(
                        "{} git tag failed: {}",
                        "Error:".bright_red(),
                        out.stderr.trim()
                    );
                    return;
                }
                if !out.stdout.trim().is_empty() {
                    print!("{}", out.stdout);
                }
                if let git_cmds::TagAction::Create { name, .. } = action {
                    println!("{} Tag {} created.", "✓".bright_green(), name.bright_cyan());
                }
            }
            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
        }
    }

    /// `/files` — list files tracked by git in this project, via
    /// `git ls-files`. Lets the user (and the model, if they ask) get a
    /// quick project inventory without leaving the chat.
    fn run_files(&self) {
        match git_cmds::ls_files() {
            Ok(out) => {
                if !out.exit_ok {
                    eprintln!(
                        "{} git ls-files failed: {}",
                        "Error:".bright_red(),
                        out.stderr.trim()
                    );
                    return;
                }
                let trimmed = out.stdout.trim();
                if trimmed.is_empty() {
                    println!("{} No tracked files in this repo.", "ℹ".bright_blue());
                    return;
                }
                let count = trimmed.lines().count();
                print!("{}", out.stdout);
                println!(
                    "{} {} tracked files.",
                    "ℹ".bright_blue(),
                    count.to_string().bright_cyan()
                );
            }
            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
        }
    }

    /// `/add-dir <path>` — trust an additional directory for write/exec
    /// tools. Companion to `/trust`, which only operates on the cwd.
    fn handle_add_dir(&self, args: &str) {
        let path_str = args.trim();
        if path_str.is_empty() {
            println!("{} Usage: /add-dir <path>", "Info:".bright_yellow());
            return;
        }
        let path = Path::new(path_str);
        let canonical = match path.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                eprintln!(
                    "{} Could not resolve {}: {}",
                    "Error:".bright_red(),
                    path_str,
                    e
                );
                return;
            }
        };
        if !canonical.is_dir() {
            eprintln!(
                "{} {} is not a directory",
                "Error:".bright_red(),
                canonical.display()
            );
            return;
        }
        let mut perms = self.permissions.lock().unwrap();
        match perms.trust_dir(&canonical) {
            Ok(added) => {
                if let Err(e) = perms.save() {
                    eprintln!(
                        "{} Failed to persist trust store: {}",
                        "Warn:".bright_yellow(),
                        e
                    );
                }
                if added {
                    println!(
                        "{} trusted {}",
                        "✓".bright_green(),
                        canonical.display().to_string().bright_cyan()
                    );
                } else {
                    println!(
                        "{} {} was already trusted",
                        "ℹ".bright_blue(),
                        canonical.display().to_string().bright_cyan()
                    );
                }
            }
            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
        }
    }

    fn toggle_plan_mode(&mut self) {
        // `fetch_xor(true)` toggles the flag and returns the *previous*
        // value, so `now_on = !prev`. Using the atomic directly keeps the
        // CLI and `BuiltinToolRegistry` views in sync without a separate
        // mirror field that could drift.
        let prev = self.plan_mode.fetch_xor(true, Ordering::SeqCst);
        let now_on = !prev;
        if now_on {
            self.history.push(Message::text(
                "system",
                "SYSTEM: Plan mode is ON. Do not modify files or run \
                 destructive commands. Produce a plan and wait for the \
                 user to confirm before applying changes.",
            ));
            println!(
                "{} Plan mode {}",
                "✓".bright_green(),
                "enabled".bright_green()
            );
        } else {
            self.history.push(Message::text(
                "system",
                "SYSTEM: Plan mode is OFF. Normal tool use is allowed.",
            ));
            println!(
                "{} Plan mode {}",
                "✓".bright_green(),
                "disabled".bright_black()
            );
        }
    }

    fn persist_todos(&self) {
        if let Err(e) = self.todos.save() {
            eprintln!("{} Failed to persist todos: {}", "Warn:".bright_yellow(), e);
        }
    }

    fn persist_memdir(&self) {
        if let Err(e) = self.memdir.save() {
            eprintln!(
                "{} Failed to persist memdir: {}",
                "Warn:".bright_yellow(),
                e
            );
        }
    }

    /// `/rewind [n]` — removes the last `n` user-assistant exchange pairs
    /// from history (default 1). System messages are never removed.
    fn rewind(&mut self, args: &str) {
        let n: usize = if args.is_empty() {
            1
        } else {
            match args.parse::<usize>() {
                Ok(0) => {
                    println!("{} Nothing to rewind.", "ℹ".bright_blue());
                    return;
                }
                Ok(v) => v,
                Err(_) => {
                    eprintln!("{} Usage: /rewind [n]", "Error:".bright_red());
                    return;
                }
            }
        };

        // Count how many non-system messages exist.
        let non_system: Vec<usize> = self
            .history
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role != "system")
            .map(|(i, _)| i)
            .collect();

        // Each "exchange" is roughly a user + assistant pair (2 messages),
        // but tool messages add more. We remove the last `n * 2` non-system
        // messages (or all of them if n is too large).
        let to_remove = (n * 2).min(non_system.len());
        if to_remove == 0 {
            println!("{} Nothing to rewind.", "ℹ".bright_blue());
            return;
        }

        // Indices to remove (from the tail of non-system messages).
        let remove_set: std::collections::HashSet<usize> = non_system
            [non_system.len() - to_remove..]
            .iter()
            .copied()
            .collect();

        let prev_len = self.history.len();
        self.history = self
            .history
            .iter()
            .enumerate()
            .filter(|(i, _)| !remove_set.contains(i))
            .map(|(_, m)| m.clone())
            .collect();
        let removed = prev_len - self.history.len();
        // Roll back any file edits/writes recorded by the built-in
        // tools during the rewound turns. We treat each removed
        // exchange (assumed n) as one journal turn — matches how the
        // CLI opens one journal turn for each agent entry point.
        let outcome = self.journal.rewind(n);
        println!(
            "{} Rewound {} message{} ({} exchange{})",
            "✓".bright_green(),
            removed,
            if removed == 1 { "" } else { "s" },
            n.min(non_system.len() / 2),
            if n == 1 { "" } else { "s" }
        );
        if !outcome.restored.is_empty() {
            println!(
                "  {} Restored {} file{}:",
                "↺".bright_yellow(),
                outcome.restored.len(),
                if outcome.restored.len() == 1 { "" } else { "s" }
            );
            for p in &outcome.restored {
                println!("    {} {}", "•".bright_cyan(), p.display());
            }
        }
        for (path, err) in &outcome.errors {
            eprintln!(
                "  {} could not roll back {}: {}",
                "Warn:".bright_yellow(),
                path.display(),
                err
            );
        }
        self.checkpoint_session();
    }

    /// `/compact` — summarizes older conversation turns into a single
    /// system message, preserving the last few exchanges in full fidelity.
    /// Uses the model itself to generate the summary.
    ///
    /// Returns `Ok(N)` where `N` is the count of summarized messages
    /// (0 when nothing was compacted), or an error when the summarizer
    /// call failed. The count lets `agent_turn` emit a `compacted` JSON
    /// event with an accurate `summarized_messages` field.
    async fn compact(&mut self) -> Result<usize> {
        // We keep the last N non-system messages intact and summarize
        // everything before that.
        const KEEP_RECENT: usize = 6; // ~3 exchanges

        let non_system_indices: Vec<usize> = self
            .history
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role != "system")
            .map(|(i, _)| i)
            .collect();

        if non_system_indices.len() <= KEEP_RECENT {
            println!(
                "{} Conversation is too short to compact ({} non-system messages).",
                "ℹ".bright_blue(),
                non_system_indices.len()
            );
            return Ok(0);
        }

        // Split: messages to summarize vs. messages to keep.
        let cutoff_idx = non_system_indices[non_system_indices.len() - KEEP_RECENT];
        let to_summarize: Vec<&Message> = self.history[..cutoff_idx]
            .iter()
            .filter(|m| m.role != "system")
            .collect();

        if to_summarize.is_empty() {
            println!("{} Nothing to compact.", "ℹ".bright_blue());
            return Ok(0);
        }

        // Build a condensed transcript for the summarizer.
        let mut transcript = String::new();
        for msg in &to_summarize {
            let role_label = match msg.role.as_str() {
                "user" => "User",
                "assistant" => "Assistant",
                "tool" => "Tool",
                _ => &msg.role,
            };
            // Truncate very long messages for the summarizer (char-safe).
            let content = if msg.content.chars().count() > 500 {
                let truncated: String = msg.content.chars().take(500).collect();
                format!("{}…", truncated)
            } else {
                msg.content.clone()
            };
            transcript.push_str(&format!("{}: {}\n", role_label, content));
        }

        println!(
            "{} Compacting {} messages into a summary…",
            "⚙".bright_blue(),
            to_summarize.len()
        );

        let summary_prompt = vec![
            Message::text(
                "system",
                "You are a summarizer. Produce a concise bullet-point summary of the \
                 following conversation. Preserve key decisions, facts, and outcomes. \
                 Do NOT include conversational filler. Output only the summary.",
            ),
            Message::text("user", transcript),
        ];

        match self.executor.chat(summary_prompt).await {
            Ok(summary) => {
                let summarized_count = to_summarize.len();
                // Rebuild history: keep all messages at their original positions,
                // but replace old non-system messages before cutoff with a single
                // summary. Messages at/after cutoff are preserved as-is.
                let mut new_history: Vec<Message> = Vec::new();

                // Keep system messages that appeared before the cutoff
                // (this includes PINNED_SYSTEM_TAG entries, which must
                // survive compaction so the user's pinned context isn't
                // silently dropped along with the summarized turns).
                for msg in &self.history[..cutoff_idx] {
                    if msg.role == "system" {
                        new_history.push(msg.clone());
                    }
                }

                // Insert the compacted summary.
                new_history.push(Message::text(
                    "system",
                    format!(
                        "SYSTEM: Compacted summary of earlier conversation:\n\n{}",
                        summary
                    ),
                ));

                // Append all messages from cutoff onward (recent messages kept intact).
                new_history.extend(self.history[cutoff_idx..].iter().cloned());

                self.history = new_history;

                println!(
                    "{} Compacted. History now has {} messages.",
                    "✓".bright_green(),
                    self.history.len()
                );

                let extract_prompt = vec![
                    Message::text(
                        "system",
                        "Extract 3-5 concise factual bullets from the following conversation summary that would be useful to remember in future sessions. Each bullet starts with '- '. Output ONLY the bullets, nothing else.",
                    ),
                    Message::text("user", &summary),
                ];
                if let Ok(extracted) = self.executor.chat(extract_prompt).await {
                    if extracted.contains("- ") {
                        let bullets: Vec<_> = extracted
                            .lines()
                            .map(str::trim)
                            .filter_map(|line| line.strip_prefix("- "))
                            .map(str::trim)
                            .filter(|line| !line.is_empty())
                            .collect();
                        if !bullets.is_empty() {
                            for bullet in &bullets {
                                self.memdir.add(bullet, Some("auto-extracted"));
                            }
                            self.persist_memdir();
                            self.inject_memdir();
                            println!(
                                "{} Extracted {} memories to memdir.",
                                "ℹ".bright_blue(),
                                bullets.len()
                            );
                        }
                    }
                }
                self.checkpoint_session();
                Ok(summarized_count)
            }
            Err(e) => Err(e),
        }
    }

    /// Pre-flight check: if the estimated prompt tokens would exceed
    /// the active model's context window, refuse the LLM request.
    ///
    /// Returns `Ok(true)` when the caller should stop the turn (the
    /// REPL path keeps the user's input in history so they can
    /// `/compact`/`/pin` and retry; headless returns an `AppExit`
    /// error). Returns `Ok(false)` when the request is within budget
    /// or the model's window is unknown (in which case we can't make
    /// an informed decision and let the request through — the provider
    /// will return its own error).
    pub(super) fn check_token_budget(&self, _turn_start: usize) -> Result<bool> {
        let model = self.executor.get_model().to_string();
        let Some(window) = crate::llm::context_window_for_model(&model) else {
            return Ok(false);
        };
        let needed = crate::llm::estimate_conversation_tokens(&self.history);
        if needed <= window {
            return Ok(false);
        }
        let json_enabled = self.json_enabled && self.headless_mode;
        if json_enabled {
            Self::emit_json_event(crate::json_events::budget_error(needed, window, &model));
            return Err(exit_code::err(
                ExitCode::Budget,
                format!(
                    "cubi: would exceed context window ({} tokens estimated, model window is {})",
                    needed, window
                ),
            ));
        }
        if self.headless_mode {
            return Err(exit_code::err(
                ExitCode::Budget,
                format!(
                    "cubi: refusing — prompt ({} est. tokens) exceeds {} window of {} tokens. \
                     Run /compact or shorten the prompt and retry.",
                    needed, model, window
                ),
            ));
        }
        eprintln!(
            "{} prompt would exceed the model's context window ({} estimated tokens > {} window for {}).",
            "[budget]".bright_red(),
            needed,
            window,
            model.bright_cyan()
        );
        eprintln!(
            "  {} run {} to summarize older turns, or {} / {} to drop pinned context, then resend.",
            "↳".bright_black(),
            "/compact".bright_cyan(),
            "/pins".bright_cyan(),
            "/unpin <idx>".bright_cyan()
        );
        Ok(true)
    }

    /// Adds `text` as a persistent pinned context item. Returns the
    /// new item's 1-based index (matching `/pins` output). The tagged
    /// system message is inserted near the head of `history` (after
    /// any pre-existing system preamble) so the LLM sees pins before
    /// the conversation transcript on every turn.
    fn pin(&mut self, text: &str) -> usize {
        let text = text.trim().to_string();
        self.pinned.push(text.clone());
        let msg = Message::text("system", format!("{} {}", PINNED_SYSTEM_TAG, text));
        // Insert right after the leading run of system messages so
        // pins live with the preamble rather than at the tail.
        let mut insert_at = 0usize;
        for (i, m) in self.history.iter().enumerate() {
            if m.role == "system" {
                insert_at = i + 1;
            } else {
                break;
            }
        }
        self.history.insert(insert_at, msg);
        self.checkpoint_session();
        self.pinned.len()
    }

    /// Removes the pinned item at the given 1-based index. Returns the
    /// removed text, or `None` if the index is out of range. The
    /// matching `PINNED_SYSTEM_TAG` system message is removed from
    /// `history` as well so the LLM stops seeing it on the next turn.
    fn unpin(&mut self, idx_1based: usize) -> Option<String> {
        if idx_1based == 0 || idx_1based > self.pinned.len() {
            return None;
        }
        let removed = self.pinned.remove(idx_1based - 1);
        // Drop only the *first* matching tagged system message so
        // duplicate pins (same text) are handled in order. The wire
        // form mirrors what `pin` injects.
        let needle = format!("{} {}", PINNED_SYSTEM_TAG, removed);
        if let Some(pos) = self
            .history
            .iter()
            .position(|m| m.role == "system" && m.content == needle)
        {
            self.history.remove(pos);
        }
        self.checkpoint_session();
        Some(removed)
    }

    /// `/pins` — render the pinned list with 1-based indices.
    fn show_pins(&self) {
        if self.pinned.is_empty() {
            println!("{} No pinned items.", "ℹ".bright_blue());
            return;
        }
        println!("{}", "Pinned context:".bright_yellow().bold());
        for (i, text) in self.pinned.iter().enumerate() {
            println!("  {}. {}", (i + 1).to_string().bright_cyan(), text);
        }
    }

    /// Inspects the current history against the active model's context
    /// window. If `auto_compact` is enabled and the estimated prompt
    /// tokens cross `compact_threshold_pct`, invoke [`compact`] before
    /// the next LLM call. No-op when the model's window is unknown
    /// (we'd be guessing) or when `auto_compact` is disabled.
    pub(super) async fn maybe_auto_compact(&mut self) {
        if !self.app_config.auto_compact {
            return;
        }
        let model = self.executor.get_model().to_string();
        let Some(window) = crate::llm::context_window_for_model(&model) else {
            return;
        };
        let threshold =
            crate::onboarding::clamp_compact_threshold(self.app_config.compact_threshold_pct)
                as usize;
        let estimated = crate::llm::estimate_conversation_tokens(&self.history);
        // window * threshold / 100 — order chosen to keep precision and
        // stay inside usize on tiny windows.
        let cutoff = window.saturating_mul(threshold) / 100;
        if estimated < cutoff {
            return;
        }
        let json_enabled = self.json_enabled && self.headless_mode;
        match self.compact().await {
            Ok(0) => {
                tracing::debug!(
                    target: "cubi::cli",
                    estimated,
                    cutoff,
                    "auto-compact skipped: nothing to summarize",
                );
            }
            Ok(n) => {
                if !json_enabled {
                    println!(
                        "{} auto-compacted {} earlier turn(s) into a summary",
                        "⚙".bright_blue(),
                        n
                    );
                }
                Self::emit_json_event_if(json_enabled, crate::json_events::compacted(n, window));
            }
            Err(e) => {
                tracing::warn!(
                    target: "cubi::cli",
                    error = %e,
                    "auto-compact failed; continuing with full history",
                );
            }
        }
    }

    /// Records a clarifying question from the user. Until the model can
    /// invoke `ask_user` itself, this command lets the user front-load a
    /// pointed question that the next turn's system context highlights.
    ///
    /// The injected system message is tagged with [`SINGLE_TURN_SYSTEM_TAG`]
    /// so it is removed after the next assistant response and doesn't
    /// re-emphasize the same question on every subsequent turn.
    fn ask_user(&mut self, question: &str) {
        self.history.push(Message::text(
            "system",
            format!(
                "{} The user has a clarifying question they want addressed \
                 directly and concisely on the next turn:\n\n{}",
                SINGLE_TURN_SYSTEM_TAG, question
            ),
        ));
        println!(
            "{} Question recorded. It will be highlighted on the next turn only.",
            "✓".bright_green()
        );
    }

    /// Removes any system messages tagged as single-turn (see
    /// [`SINGLE_TURN_SYSTEM_TAG`]) from the history.
    fn strip_single_turn_system_messages(&mut self) {
        self.history
            .retain(|m| !(m.role == "system" && m.content.starts_with(SINGLE_TURN_SYSTEM_TAG)));
    }

    fn run_init(&self) {
        match project_memory::write_starter_if_absent() {
            Ok(true) => println!(
                "{} Wrote starter {} in current directory",
                "✓".bright_green(),
                project_memory::MEMORY_FILENAME.bright_cyan()
            ),
            Ok(false) => println!(
                "{} {} already exists; left untouched",
                "ℹ".bright_blue(),
                project_memory::MEMORY_FILENAME.bright_cyan()
            ),
            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
        }
    }

    fn show_memory(&self) {
        match project_memory::read_memory_with_path() {
            Ok(Some((path, contents))) => {
                println!(
                    "\n{} ({}):",
                    "Project memory".bright_yellow().bold(),
                    path.display().to_string().bright_cyan()
                );
                println!("{}", "-".repeat(60).bright_black());
                println!("{}", contents);
                println!("{}\n", "-".repeat(60).bright_black());
            }
            Ok(None) => println!(
                "{} No {} found. Run /init to create one.",
                "ℹ".bright_blue(),
                project_memory::MEMORY_FILENAME.bright_cyan()
            ),
            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
        }
    }

    /// Removes any previously injected project-memory system message and
    /// re-reads `CUBI.md` (walking up the directory tree) into history.
    ///
    /// The reloaded memory is inserted **at the front** of `history` (after
    /// any earlier system messages, but before user/assistant turns) so that
    /// `/memory-reload` produces the same model-weighting as the initial
    /// `ChatCLI::new` injection. Appending at the tail meant the refreshed
    /// context got the most recency weight, which is usually the opposite of
    /// what the user wants from "long-lived project context".
    fn inject_project_memory(&mut self) {
        // Drop any prior project-memory entries so callers can use this as
        // both an initial inject and a reload.
        self.history
            .retain(|m| !(m.role == "system" && m.content.starts_with(PROJECT_MEMORY_PREFIX)));

        let Ok(Some((path, memory))) = project_memory::read_memory_with_path() else {
            return;
        };

        let msg = Message::text(
            "system",
            format!(
                "{} {}:\n\n{}",
                PROJECT_MEMORY_PREFIX,
                path.display(),
                memory.trim()
            ),
        );

        // Find the boundary: the first non-system message. We want the
        // project-memory entry to sit with the other system messages,
        // ahead of any user/assistant turns.
        let insert_at = self
            .history
            .iter()
            .position(|m| m.role != "system")
            .unwrap_or(self.history.len());
        self.history.insert(insert_at, msg);
    }

    /// Prefix used to locate previously injected memdir system messages.
    const MEMDIR_PREFIX: &'static str = "SYSTEM: Cross-session memories";

    /// Injects memdir context into the system messages (or removes it if
    /// the memdir is now empty). Safe to call multiple times.
    fn inject_memdir(&mut self) {
        // Drop any prior memdir entries.
        self.history
            .retain(|m| !(m.role == "system" && m.content.starts_with(Self::MEMDIR_PREFIX)));

        let Some(ctx) = self.memdir.as_context_string() else {
            return;
        };

        // `as_context_string()` already includes the header, so use the prefix
        // only as a tag for future removal — the actual content comes from ctx.
        let msg = Message::text(
            "system",
            format!("{} (from ~/.cubi/memdir/):\n{}", Self::MEMDIR_PREFIX, ctx),
        );

        let insert_at = self
            .history
            .iter()
            .position(|m| m.role != "system")
            .unwrap_or(self.history.len());
        self.history.insert(insert_at, msg);
    }

    /// Cleanly shuts down any owned MCP connections. Call this from an
    /// async context **before** dropping the CLI; the `Drop` impl is only a
    /// best-effort safety net.
    pub async fn shutdown(&mut self) {
        if let Some(mcp) = &mut self.mcp_manager {
            mcp.shutdown().await;
        }
    }

    fn export_markdown(&self, filename: &str, force: bool) -> Result<()> {
        check_overwrite_allowed(filename, force, "/export")?;
        let mut out = String::new();
        out.push_str("# cubi conversation\n\n");
        out.push_str(&format!("- model: `{}`\n", self.executor.get_model()));
        out.push_str(&format!("- messages: {}\n\n", self.history.len()));
        out.push_str("---\n\n");
        for msg in &self.history {
            let heading = match msg.role.as_str() {
                "user" => "## You",
                "assistant" => "## AI",
                "system" => "## System",
                other => {
                    out.push_str(&format!("## {}\n\n", other));
                    out.push_str(&msg.content);
                    out.push_str("\n\n");
                    continue;
                }
            };
            out.push_str(heading);
            out.push_str("\n\n");
            out.push_str(&msg.content);
            out.push_str("\n\n");
        }
        fs::write(filename, out)?;
        Ok(())
    }

    // -------- New slash command handlers (v0.2.0) --------

    /// `/init-verifiers` — scan the cwd for well-known build/test/lint
    /// manifests and print (and save to `.cubi-verifiers.json`) the
    /// inferred verifier commands.
    fn run_init_verifiers(&self) {
        let cwd = match std::env::current_dir() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{} {}", "Error:".bright_red(), e);
                return;
            }
        };
        let mut verifiers: Vec<(&str, &str)> = Vec::new();
        if cwd.join("Cargo.toml").exists() {
            verifiers.push(("build", "cargo build"));
            verifiers.push(("test", "cargo test"));
            verifiers.push(("lint", "cargo clippy --all-targets -- -D warnings"));
            verifiers.push(("fmt", "cargo fmt --all -- --check"));
        }
        if cwd.join("package.json").exists() {
            verifiers.push(("build", "npm run build"));
            verifiers.push(("test", "npm test"));
            verifiers.push(("lint", "npm run lint"));
        }
        if cwd.join("pyproject.toml").exists() || cwd.join("setup.py").exists() {
            verifiers.push(("test", "pytest"));
            verifiers.push(("lint", "ruff check ."));
        }
        if cwd.join("go.mod").exists() {
            verifiers.push(("build", "go build ./..."));
            verifiers.push(("test", "go test ./..."));
            verifiers.push(("lint", "go vet ./..."));
        }
        if cwd.join("Makefile").exists() {
            verifiers.push(("make", "make"));
        }
        println!("\n{}", "Detected verifiers:".bright_yellow().bold());
        if verifiers.is_empty() {
            println!(
                "  {} No known build manifest found in {}",
                "ℹ".bright_blue(),
                cwd.display()
            );
            return;
        }
        for (k, v) in &verifiers {
            println!("  {} {}: {}", "•".bright_cyan(), k.bright_cyan(), v);
        }
        let out_path = cwd.join(".cubi-verifiers.json");
        let payload: Vec<serde_json::Value> = verifiers
            .iter()
            .map(|(k, v)| serde_json::json!({ "kind": k, "command": v }))
            .collect();
        let serialized = match serde_json::to_string_pretty(&payload) {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "  {} Could not serialize {}: {}",
                    "✗".bright_red(),
                    out_path.display(),
                    e
                );
                return;
            }
        };
        match fs::write(&out_path, serialized) {
            Ok(()) => println!(
                "\n  {} Saved to {}",
                "✓".bright_green(),
                out_path.display().to_string().bright_cyan()
            ),
            Err(e) => eprintln!(
                "  {} Could not write {}: {}",
                "✗".bright_red(),
                out_path.display(),
                e
            ),
        }
        println!();
    }

    /// `/pr_comments` — shells out to `gh pr view --comments` if installed.
    fn run_pr_comments(&self, args: &str) {
        let arg_pr = args.trim();
        let mut cmd = std::process::Command::new("gh");
        cmd.arg("pr").arg("view").arg("--comments");
        if !arg_pr.is_empty() {
            cmd.arg(arg_pr);
        }
        match cmd.output() {
            Ok(out) => {
                std::io::Write::write_all(&mut std::io::stdout(), &out.stdout).ok();
                if !out.status.success() {
                    eprintln!(
                        "{} gh pr view exited {} ({})",
                        "Error:".bright_red(),
                        out.status.code().unwrap_or(-1),
                        String::from_utf8_lossy(&out.stderr).trim()
                    );
                }
            }
            Err(e) => eprintln!(
                "{} `gh` not available ({}). Install GitHub CLI: https://cli.github.com",
                "Error:".bright_red(),
                e
            ),
        }
    }

    /// `/security-review` — like `/review`, but prompts the model to focus
    /// on security-relevant issues in the current `git diff`.
    async fn run_security_review(&mut self) {
        let diff = match git_cmds::diff_for_review() {
            Ok(out) if !out.stdout.trim().is_empty() => out.stdout,
            Ok(_) => {
                println!(
                    "{} Working tree is clean — nothing to review.",
                    "ℹ".bright_blue()
                );
                return;
            }
            Err(e) => {
                eprintln!("{} {}", "Error:".bright_red(), e);
                return;
            }
        };
        let prompt = format!(
            "Please perform a focused security review of the following `git diff`. \
             Highlight any potential vulnerabilities (injection, path traversal, \
             auth, secret handling, supply-chain, race conditions, deserialization, \
             etc.), explain the impact, and suggest a remediation. If the diff is \
             security-clean, say so explicitly.\n\n```diff\n{diff}\n```"
        );
        self.history.push(Message::text("user", prompt));
        match self.executor.chat(self.history.clone()).await {
            Ok(reply) => {
                println!("{} {}", "AI:".bright_cyan().bold(), reply);
                self.history.push(Message::text("assistant", reply));
            }
            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
        }
    }

    /// `/autofix-pr` — fetch PR comments and ask the model to propose fixes.
    async fn run_autofix_pr(&mut self, args: &str) {
        let arg_pr = args.trim();
        let mut cmd = std::process::Command::new("gh");
        cmd.arg("pr").arg("view").arg("--comments");
        if !arg_pr.is_empty() {
            cmd.arg(arg_pr);
        }
        let body = match cmd.output() {
            Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).to_string(),
            Ok(out) => {
                eprintln!(
                    "{} gh pr view failed: {}",
                    "Error:".bright_red(),
                    String::from_utf8_lossy(&out.stderr).trim()
                );
                return;
            }
            Err(e) => {
                eprintln!("{} `gh` not available: {}", "Error:".bright_red(), e);
                return;
            }
        };
        let prompt = format!(
            "Here are the review comments on the current pull request. Propose \
             concrete code changes that address each substantive comment. Group \
             suggestions by file and explain reasoning. If a comment is unclear, \
             flag it.\n\n```\n{body}\n```"
        );
        self.history.push(Message::text("user", prompt));
        match self.executor.chat(self.history.clone()).await {
            Ok(reply) => {
                println!("{} {}", "AI:".bright_cyan().bold(), reply);
                self.history.push(Message::text("assistant", reply));
            }
            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
        }
    }

    /// `/agents` — list active background agent sessions (from the session
    /// store). Each session is a candidate "agent" worker.
    fn show_agents(&self) {
        println!("\n{}", "Agents / sessions:".bright_yellow().bold());
        let Some(store) = self.session_store.as_ref() else {
            println!(
                "  {} Sessions disabled (no home dir resolved).",
                "ℹ".bright_blue()
            );
            return;
        };
        match store.list() {
            Ok(list) if list.is_empty() => println!("  {} No sessions yet.", "ℹ".bright_blue()),
            Ok(list) => {
                for s in list {
                    println!(
                        "  {} {} ({} msgs)",
                        "•".bright_cyan(),
                        s.id.bright_cyan(),
                        s.message_count
                    );
                }
            }
            Err(e) => eprintln!("  {} {}", "✗".bright_red(), e),
        }
        println!();
    }

    /// `/teleport <path>` — change cwd to a trusted directory.
    fn run_teleport(&self, args: &str) {
        let path = args.trim();
        if path.is_empty() {
            println!("{} Usage: /teleport <path>", "Info:".bright_yellow());
            return;
        }
        let candidate = PathBuf::from(path);
        if let Err(e) = self.permissions.lock().unwrap().check_exec(&candidate) {
            eprintln!("{} {}", "Error:".bright_red(), e);
            return;
        }
        match std::env::set_current_dir(path) {
            Ok(()) => println!("{} cwd is now {}", "✓".bright_green(), path.bright_cyan()),
            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
        }
    }

    /// `/passes` and `/effort` — show or set the agent-loop pass budget. We
    /// stash it in the `EffortConfig` config field. The agent loop reads
    /// `AGENT_PASSES_OVERRIDE` env var if set; we set that here so the
    /// change takes effect mid-session without rebuilding the loop.
    fn handle_effort(&self, cmd: Cmd, args: &str) {
        let arg = args.trim();
        if arg.is_empty() {
            let current = std::env::var("AGENT_PASSES_OVERRIDE")
                .unwrap_or_else(|_| format!("{}", agent_loop::MAX_AGENT_STEPS));
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
        unsafe { std::env::set_var("AGENT_PASSES_OVERRIDE", n.to_string()) };
        println!("{} max passes set to {}", "✓".bright_green(), n);
    }

    /// Loads the on-disk AppConfig, applies `mutate`, persists. Errors
    /// are surfaced but never fatal — handlers fall back to "in-memory
    /// only" (still affects the env vars for the current process) so a
    /// read-only $HOME doesn't break the slash commands.
    fn update_config(&self, mutate: impl FnOnce(&mut AppConfig)) -> Result<()> {
        let mut cfg = AppConfig::load();
        mutate(&mut cfg);
        cfg.save()
    }

    fn handle_theme(&self, args: &str) {
        let arg = args.trim().to_ascii_lowercase();
        if arg.is_empty() {
            println!(
                "{} theme: {} (set with /theme [{}])",
                "ℹ".bright_blue(),
                std::env::var("CUBI_THEME")
                    .unwrap_or_else(|_| "auto".to_string())
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
        // SAFETY: handled on the readline thread, no race.
        unsafe { std::env::set_var("CUBI_THEME", &arg) };
        if let Err(e) = self.update_config(|c| c.theme = Some(arg.clone())) {
            eprintln!(
                "{} theme set in this session but not persisted: {}",
                "Warn:".bright_yellow(),
                e
            );
        }
        println!("{} theme set to {}", "✓".bright_green(), arg);
    }

    fn handle_color(&self, args: &str) {
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
                crate::style::set_color_override(true);
                if let Err(e) = self.update_config(|c| c.color = Some("on".to_string())) {
                    eprintln!("{} not persisted: {}", "Warn:".bright_yellow(), e);
                }
                println!("{} colored output ON", "✓".bright_green());
            }
            "off" => {
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

    fn handle_output_style(&self, args: &str) {
        let arg = args.trim().to_ascii_lowercase();
        if arg.is_empty() {
            println!(
                "{} output style: {} (set with /output-style [{}])",
                "ℹ".bright_blue(),
                std::env::var("CUBI_OUTPUT_STYLE")
                    .unwrap_or_else(|_| output_styles::DEFAULT_STYLE.to_string())
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
        unsafe { std::env::set_var("CUBI_OUTPUT_STYLE", &arg) };
        if let Err(e) = self.update_config(|c| c.output_style = Some(arg.clone())) {
            eprintln!(
                "{} style set in this session but not persisted: {}",
                "Warn:".bright_yellow(),
                e
            );
        }
        println!("{} output style set to {}", "✓".bright_green(), arg);
    }

    fn show_statusline(&self) {
        let plan = if self.plan_mode.load(Ordering::SeqCst) {
            "plan"
        } else {
            "live"
        };
        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "<unknown>".to_string());
        println!(
            "{} [{}] [{}] {} msgs | todos {}/{} | memdir {} | model {}",
            "statusline:".bright_yellow().bold(),
            plan.bright_cyan(),
            cwd.bright_cyan(),
            self.history.len(),
            self.todos.pending(),
            self.todos.len(),
            self.memdir.len(),
            self.executor.get_model().bright_cyan(),
        );
    }

    fn show_keybindings(&self) {
        println!("\n{}", "Keybindings:".bright_yellow().bold());
        let pairs: &[(&str, &str)] = &[
            (
                "Up / Down",
                "history navigation (persisted in ~/.cubi/history)",
            ),
            ("Ctrl-A / Ctrl-E", "beginning / end of line (emacs)"),
            ("Ctrl-W", "delete previous word"),
            ("Ctrl-U", "kill to start of line"),
            ("Ctrl-K", "kill to end of line"),
            ("Ctrl-R", "reverse-i-search through history"),
            ("Ctrl-C", "cancel current input"),
            ("Ctrl-D", "exit on empty line"),
            ("Tab", "rustyline autocomplete (where available)"),
        ];
        for (k, v) in pairs {
            println!("  {} {}", k.bright_cyan(), v);
        }
        println!();
    }

    fn handle_vim(&self, args: &str) {
        let arg = args.trim().to_ascii_lowercase();
        match arg.as_str() {
            "" => println!(
                "{} vim mode: {} (toggle with /vim on|off; takes effect next session)",
                "ℹ".bright_blue(),
                std::env::var("CUBI_VIM_MODE")
                    .unwrap_or_else(|_| "off".to_string())
                    .bright_cyan()
            ),
            "on" | "off" => {
                unsafe { std::env::set_var("CUBI_VIM_MODE", &arg) };
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

    fn handle_login(&self, args: &str) {
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
        unsafe { std::env::set_var(&env_var, &parsed.access_token) };
        println!(
            "{} OAuth token saved for '{}' and loaded into {}",
            "✓".bright_green(),
            parsed.provider.bright_cyan(),
            env_var.bright_cyan()
        );
    }

    fn handle_logout(&self, args: &str) {
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
        unsafe { std::env::remove_var(&var) };
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

    fn show_oauth_refresh(&self, args: &str) {
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
            unsafe { std::env::set_var(&env_var, &token.access_token) };
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

    fn handle_privacy_settings(&self, args: &str) {
        let arg = args.trim();
        if arg.is_empty() {
            let telemetry = std::env::var("CUBI_TELEMETRY").unwrap_or_else(|_| "off".to_string());
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
                unsafe { std::env::set_var("CUBI_TELEMETRY", *v) };
                println!("{} telemetry set to {}", "✓".bright_green(), v);
            }
            _ => eprintln!(
                "{} Usage: /privacy-settings telemetry on|off",
                "Error:".bright_red()
            ),
        }
    }

    fn show_mcp_status(&self) {
        println!("\n{}", "MCP status:".bright_yellow().bold());
        match &self.mcp_manager {
            Some(m) => {
                let tools = m.list_tools();
                println!("  {} {} tool(s) available", "•".bright_cyan(), tools.len());
                for t in tools.iter().take(20) {
                    println!("    - {}", t.name.bright_cyan());
                }
                if tools.len() > 20 {
                    println!("    … {} more (see /mcp-tools)", tools.len() - 20);
                }
            }
            None => println!("  {} No MCP manager loaded.", "ℹ".bright_blue()),
        }
        println!();
    }

    fn show_plugins(&self) {
        println!("\n{}", "Plugins:".bright_yellow().bold());
        let Some(dir) = plugins::plugins_dir() else {
            println!("  {} No home directory.", "ℹ".bright_blue());
            return;
        };
        if self.plugins.is_empty() {
            println!(
                "  {} No plugins discovered in {}. Drop a Markdown file at \n     {}/<plugin>/commands/<name>.md to register one.",
                "ℹ".bright_blue(),
                dir.display(),
                dir.display()
            );
            println!();
            return;
        }
        for p in &self.plugins {
            println!(
                "  {} {} v{} ({})",
                "•".bright_cyan(),
                p.name.bright_cyan(),
                p.version.bright_magenta(),
                p.root.display().to_string().bright_black()
            );
            if p.commands.is_empty() {
                println!("      {} (no commands)", "ℹ".bright_blue());
                continue;
            }
            for c in &p.commands {
                println!(
                    "      {} {} — {}",
                    "›".bright_green(),
                    c.trigger(&p.name).bright_white(),
                    c.description
                );
            }
        }
        println!();
    }

    fn reload_plugins(&mut self) {
        self.skills = skills::load_skills();
        let before = std::mem::take(&mut self.plugins);
        self.plugins = plugins::load_plugins();
        print!("{} ", "✓".bright_green());
        plugins::print_reload_summary(&before, &self.plugins, self.skills.len());
    }

    fn show_cost(&self) {
        let tokens = self.history.iter().map(|m| m.content.len()).sum::<usize>() / 4;
        println!(
            "\n{}\n  estimated tokens: ~{}\n  estimated cost:   $0.00 (local Ollama)\n",
            "Cost:".bright_yellow().bold(),
            tokens
        );
    }

    fn show_perf_issue_url(&self, args: &str) {
        let title = if args.trim().is_empty() {
            "Performance issue".to_string()
        } else {
            args.trim().to_string()
        };
        let body = format!(
            "## Symptom\n<!-- What was slow? -->\n\n## Environment\n- cubi: v{}\n- os: {} ({})\n- model: {}\n",
            env!("CARGO_PKG_VERSION"),
            std::env::consts::OS,
            std::env::consts::ARCH,
            self.executor.get_model(),
        );
        let url = format!(
            "https://github.com/peterchoi1014/cubi/issues/new?labels=performance&title={}&body={}",
            url_encode(&title),
            url_encode(&body),
        );
        println!(
            "\n{} {}\n",
            "Perf issue URL:".bright_yellow().bold(),
            url.bright_cyan()
        );
    }

    fn show_heap_info(&self) {
        println!("\n{}", "Process info:".bright_yellow().bold());
        #[cfg(target_os = "linux")]
        {
            match fs::read_to_string("/proc/self/status") {
                Ok(s) => {
                    for line in s.lines() {
                        if line.starts_with("VmPeak")
                            || line.starts_with("VmRSS")
                            || line.starts_with("VmSize")
                            || line.starts_with("VmData")
                            || line.starts_with("Threads")
                        {
                            println!("  {}", line);
                        }
                    }
                }
                Err(e) => println!(
                    "  {} /proc/self/status unavailable: {}",
                    "ℹ".bright_blue(),
                    e
                ),
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            println!(
                "  {} No portable heap-info backend; use an external profiler.",
                "ℹ".bright_blue()
            );
        }
        println!();
    }

    fn handle_debug_tool_call(&self, args: &str) {
        let arg = args.trim().to_ascii_lowercase();
        match arg.as_str() {
            "" => println!(
                "{} debug-tool-call: {} (toggle with /debug-tool-call on|off)",
                "ℹ".bright_blue(),
                std::env::var("CUBI_DEBUG_TOOL_CALL")
                    .unwrap_or_else(|_| "off".to_string())
                    .bright_cyan()
            ),
            "on" => {
                unsafe { std::env::set_var("CUBI_DEBUG_TOOL_CALL", "on") };
                println!("{} debug-tool-call ON", "✓".bright_green());
            }
            "off" => {
                unsafe { std::env::remove_var("CUBI_DEBUG_TOOL_CALL") };
                println!("debug-tool-call OFF");
            }
            other => eprintln!(
                "{} Unknown value '{}'. Expected on|off.",
                "Error:".bright_red(),
                other
            ),
        }
    }

    fn show_upgrade(&self) {
        println!(
            "\n{}\n  cd ~/code/cubi && git pull && cargo install --path . --locked\n  \
             or download a release binary from\n  \
             https://github.com/peterchoi1014/cubi/releases\n",
            "Upgrade cubi:".bright_yellow().bold()
        );
    }

    fn show_install(&self) {
        println!(
            "\n{}\n  1. install Rust: https://rustup.rs/\n  \
             2. install Ollama: https://ollama.ai/\n  \
             3. `ollama pull llama3.2:1b`\n  \
             4. `cargo install --path .` from a clone of\n     https://github.com/peterchoi1014/cubi\n",
            "Install cubi:".bright_yellow().bold()
        );
    }

    fn show_install_github_app(&self) {
        println!(
            "{} cubi does not ship a GitHub App. Use the GitHub CLI (`gh`) \
             instead for PRs/issues; see /commit-push-pr and /pr_comments.",
            "ℹ".bright_blue()
        );
    }

    fn show_install_slack_app(&self) {
        println!(
            "{} No Slack integration is bundled. You can drive cubi from \
             a Slack bot by piping prompts through `--batch` mode.",
            "ℹ".bright_blue()
        );
    }

    fn reset_limits(&self) {
        // The Ollama client tracks retries per-call (no persistent state), so
        // "resetting" is informational. Clear the env-var override if set so
        // backoff returns to defaults.
        unsafe { std::env::remove_var("CUBI_RATE_LIMIT_BACKOFF_MS") };
        println!(
            "{} Rate-limit / retry state cleared (Ollama retries are per-call).",
            "✓".bright_green()
        );
    }

    fn run_share(&self, args: &str) {
        let path = args.trim();
        if path.is_empty() {
            println!("{} Usage: /share <file.md>", "Info:".bright_yellow());
            return;
        }
        match self.export_markdown(path, true) {
            Ok(()) => {
                let abs = std::fs::canonicalize(path)
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|_| path.to_string());
                println!(
                    "{} Conversation written to {}\n  share it as a gist:\n  \
                     gh gist create {}",
                    "✓".bright_green(),
                    abs.bright_cyan(),
                    abs
                );
            }
            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
        }
    }

    /// `/copy` — copy the last assistant message to the clipboard via
    /// `pbcopy` (macOS), `wl-copy` / `xclip` (Linux), or `clip` (Windows).
    fn run_copy(&self) {
        let Some(last) = self
            .history
            .iter()
            .rev()
            .find(|m| m.role == "assistant")
            .map(|m| m.content.clone())
        else {
            println!("{} No assistant message in history yet.", "ℹ".bright_blue());
            return;
        };

        let (program, extra): (&str, &[&str]) = if cfg!(target_os = "macos") {
            ("pbcopy", &[])
        } else if cfg!(target_os = "windows") {
            ("clip", &[])
        } else if std::env::var("WAYLAND_DISPLAY").is_ok() {
            ("wl-copy", &[])
        } else {
            ("xclip", &["-selection", "clipboard"])
        };

        let mut child = match std::process::Command::new(program)
            .args(extra)
            .stdin(std::process::Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    "{} `{}` not available ({}). Install a clipboard tool or paste manually.",
                    "Error:".bright_red(),
                    program,
                    e
                );
                return;
            }
        };
        if let Some(mut stdin) = child.stdin.take() {
            use std::io::Write;
            let _ = stdin.write_all(last.as_bytes());
        }
        match child.wait() {
            Ok(s) if s.success() => println!(
                "{} Copied last assistant message ({} chars) via {}",
                "✓".bright_green(),
                last.len(),
                program
            ),
            Ok(s) => eprintln!(
                "{} {} exited {}",
                "Error:".bright_red(),
                program,
                s.code().unwrap_or(-1)
            ),
            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
        }
    }

    fn show_feedback_url(&self, args: &str) {
        let title = if args.trim().is_empty() {
            "Feedback".to_string()
        } else {
            args.trim().to_string()
        };
        let url = format!(
            "https://github.com/peterchoi1014/cubi/issues/new?labels=feedback&title={}",
            url_encode(&title),
        );
        println!(
            "\n{} {}\n",
            "Feedback URL:".bright_yellow().bold(),
            url.bright_cyan()
        );
    }

    fn show_release_notes(&self) {
        println!(
            "\n{} v{}\n\n\
             - Plugin system: namespaced `/<plugin>:<command>` triggers\n   \
               loaded from ~/.cubi/plugins/<name>/commands/*.md\n\
             - Themable output styles: `/theme`, `/output-style`, `/color`,\n   \
               and `/vim` now persist to ~/.cubi/config.json\n\
             - `prevent_sleep` built-in tool (caffeinate / systemd-inhibit /\n   \
               SetThreadExecutionState)\n\
             - Opt-in telemetry: tool calls log to ~/.cubi/telemetry.log\n\
             - Tip-of-the-day at startup + on-demand via `/tip`\n\
             - Admin policy overlay (~/.cubi/policy.json,\n   \
               /etc/cubi/policy.json, $CUBI_POLICY_FILE); inspect via `/policy`\n\
             - Git-backed cross-machine sync via `/settings-sync`\n\
             - File-mutation rollback on `/rewind` (edit_file / write_file)\n\
             - MCP prompts (`prompts/list` + `prompts/get`) via `/mcp-prompts`\n\
             - Versioned config migrations framework\n",
            "Release notes:".bright_yellow().bold(),
            env!("CARGO_PKG_VERSION")
        );
    }

    fn show_stickers(&self) {
        println!(
            "\n{}\n  ┌────────────┐    (\\(\\\n  │ ai-chat 🚀 │    ( -.-)\n  └────────────┘    o_(\")(\")\n",
            "Stickers:".bright_yellow().bold()
        );
    }

    /// `/settings-sync` — drive `settings_sync.rs` for cross-machine
    /// config + memdir + skills sync via git. Verbs: `init <remote>`,
    /// `push [msg]`, `pull`, `status`.
    fn handle_settings_sync(&self, args: &str) {
        let mut parts = args.splitn(2, char::is_whitespace);
        let verb = parts.next().unwrap_or("").trim();
        let rest = parts.next().unwrap_or("").trim();

        let result: Result<String> = match verb {
            "" | "status" => settings_sync::status(),
            "init" => {
                if rest.is_empty() {
                    eprintln!(
                        "{} Usage: /settings-sync init <remote-url>",
                        "Error:".bright_red()
                    );
                    return;
                }
                settings_sync::init(rest)
            }
            "push" => {
                let msg = if rest.is_empty() { "cubi: sync" } else { rest };
                settings_sync::push(msg)
            }
            "pull" => settings_sync::pull(),
            other => {
                eprintln!(
                    "{} Unknown verb '{}'. Expected init|push|pull|status.",
                    "Error:".bright_red(),
                    other
                );
                return;
            }
        };
        match result {
            Ok(msg) => println!("{} {}", "✓".bright_green(), msg),
            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
        }
    }

    fn show_policy(&self) {
        println!("\n{}", "Admin policy:".bright_yellow().bold());
        match Policy::active_path() {
            Some(p) => println!(
                "  {}: {}",
                "file".bright_cyan(),
                p.display().to_string().bright_cyan()
            ),
            None => {
                println!(
                    "  {} No policy file. Drop JSON at /etc/cubi/policy.json or \n     ~/.cubi/policy.json to enforce one.",
                    "ℹ".bright_blue()
                );
                println!();
                return;
            }
        }
        if self.policy.denied_tools.is_empty() {
            println!("  {} denied tools: (none)", "•".bright_cyan());
        } else {
            println!(
                "  {} denied tools: {}",
                "•".bright_red(),
                self.policy
                    .denied_tools
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        if let Some(note) = &self.policy.note {
            println!("  {} note: {}", "•".bright_cyan(), note);
        }
        println!();
    }

    fn show_tip(&self) {
        match crate::tips::tip_of_the_day() {
            Some(tip) => println!("\n{} {}\n", "💡 tip:".bright_yellow(), tip),
            None => println!("\n{} no tips available\n", "ℹ".bright_blue()),
        }
    }

    /// `/mcp-prompts` — list prompts exposed by configured MCP servers,
    /// or render a specific one when called as
    /// `/mcp-prompts <server>:<prompt>`.
    async fn show_mcp_prompts(&mut self, args: &str) {
        let Some(mcp) = self.mcp_manager.as_mut() else {
            println!(
                "{} No MCP servers loaded (configure ~/.cubi/mcp.json).",
                "ℹ".bright_blue()
            );
            return;
        };
        let arg = args.trim();
        if let Some((server, prompt)) = arg.split_once(':') {
            match mcp.get_prompt(server, prompt).await {
                Ok(body) => {
                    println!(
                        "\n{}",
                        format!("Prompt {server}:{prompt}").bright_yellow().bold()
                    );
                    println!("{}\n", body);
                }
                Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
            }
            return;
        }
        match mcp.list_prompts().await {
            Ok(prompts) => {
                println!("\n{}", "MCP prompts:".bright_yellow().bold());
                if prompts.is_empty() {
                    println!(
                        "  {} No prompts exposed by any configured server.",
                        "ℹ".bright_blue()
                    );
                } else {
                    for (server, name, description) in &prompts {
                        println!(
                            "  {} {}:{} — {}",
                            "•".bright_cyan(),
                            server.bright_cyan(),
                            name.bright_white(),
                            description
                        );
                    }
                    println!(
                        "\n  {} /mcp-prompts <server>:<name> renders the prompt body.",
                        "ℹ".bright_blue()
                    );
                }
                println!();
            }
            Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
        }
    }
}

// Best-effort fallback if `shutdown()` was never called. We intentionally do
// **not** spin up a fresh `tokio::runtime::Runtime` here: doing so from inside
// the outer `#[tokio::main]` runtime panics, and previously caused a crash on
// every clean exit that actually had MCP cleanup to do. Callers should
// `cli.shutdown().await` from `main` instead.
impl Drop for ChatCLI {
    fn drop(&mut self) {
        if self.mcp_manager.is_some() {
            // Try the cheap path: if there's a current Tokio runtime handle
            // available, spawn a detached cleanup task. Otherwise log and move
            // on — losing the MCP shutdown on a hard drop is preferable to
            // panicking.
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                if let Some(mut mcp) = self.mcp_manager.take() {
                    handle.spawn(async move {
                        mcp.shutdown().await;
                    });
                }
            } else {
                eprintln!(
                    "{} ChatCLI dropped without an explicit shutdown(); \
                     MCP servers may not be cleanly stopped.",
                    "Warning:".bright_yellow()
                );
            }
        }
    }
}

/// Tally returned by [`ChatCLI::process_batch_file`]. Tracked separately so
/// the caller can render a final "N ok, M failed" summary line and emit a
/// distinct color / severity when any prompts failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BatchSummary {
    ok: usize,
    failed: usize,
}

/// Parses an on/off/toggle argument for boolean slash commands. Accepts
/// `on`, `off`, `true`, `false`, `1`, `0`, `enable`, `disable`. Only the
/// first whitespace-delimited token is inspected; trailing words are
/// ignored so users can type `/stream off please` without surprise.
/// Returns `None` when the token is non-empty and unrecognized so the
/// caller can refuse the change instead of silently toggling. An empty
/// arg is treated as a toggle of the current value.
fn parse_toggle(arg: &str, current: bool) -> Option<bool> {
    let first = arg.split_whitespace().next().unwrap_or("");
    match first.to_ascii_lowercase().as_str() {
        "" => Some(!current),
        "on" | "true" | "1" | "enable" | "enabled" | "yes" => Some(true),
        "off" | "false" | "0" | "disable" | "disabled" | "no" => Some(false),
        _ => None,
    }
}

/// Prints a one-line dim footer summarizing token usage and wall time for
/// the just-completed turn. Only the fields the provider actually returned
/// are shown; missing fields are skipped to avoid printing "0 in / 0 out".
/// one place to audit the file-clobber footgun.
fn check_overwrite_allowed(filename: &str, force: bool, cmd: &str) -> Result<()> {
    if !force && Path::new(filename).exists() {
        anyhow::bail!(
            "Refusing to overwrite existing file '{}'. Re-run with {} -f <file> to force.",
            filename,
            cmd
        );
    }
    Ok(())
}

/// Parses the argument list of `/export` and `/save` into a `(force, filename)`
/// pair. Accepts the `-f` flag in either position:
///
/// * `-f foo.md`
/// * `foo.md -f`
/// * `foo.md` (no force)
///
/// Returns `None` if the argument list is empty or contains only the flag,
/// which the caller should treat as a usage error.
fn parse_force_and_filename(rest: &str) -> Option<(bool, &str)> {
    let trimmed = rest.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(after) = trimmed.strip_prefix("-f ") {
        let name = after.trim();
        if name.is_empty() {
            return None;
        }
        return Some((true, name));
    }
    if let Some(before) = trimmed.strip_suffix(" -f") {
        let name = before.trim();
        if name.is_empty() {
            return None;
        }
        return Some((true, name));
    }
    if trimmed == "-f" {
        return None;
    }
    Some((false, trimmed))
}

/// Minimal RFC 3986 percent-encoder for query/body params used by `/bug`.
/// Encodes everything except the unreserved set (`A-Z a-z 0-9 - _ . ~`).
/// Kept here to avoid pulling in the `percent-encoding` or `urlencoding`
/// crates for one URL.
fn github_repo_from_remote(remote: &str) -> Option<(String, String)> {
    let trimmed = remote.trim();
    if let Some(rest) = trimmed.strip_prefix("git@github.com:") {
        let repo = rest.strip_suffix(".git").unwrap_or(rest);
        let (owner, name) = repo.split_once('/')?;
        return Some((owner.to_string(), name.to_string()));
    }
    let https_prefix = "https://github.com/";
    if let Some(rest) = trimmed.strip_prefix(https_prefix) {
        let repo = rest.strip_suffix(".git").unwrap_or(rest);
        let (owner, name) = repo.split_once('/')?;
        return Some((owner.to_string(), name.to_string()));
    }
    None
}

fn url_encode(input: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::with_capacity(input.len());
    for &b in input.as_bytes() {
        let unreserved = b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~');
        if unreserved {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 0x0f) as usize] as char);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    fn user(s: &str) -> Message {
        Message::text("user", s)
    }

    fn assistant(s: &str) -> Message {
        Message::text("assistant", s)
    }

    fn system(s: &str) -> Message {
        Message::text("system", s)
    }

    #[test]
    fn repl_history_path_lives_under_cubi_home() {
        if let Some(path) = repl_history_path() {
            assert!(path.ends_with(Path::new(".cubi").join("history")));
        }
    }

    #[test]
    fn welcome_banner_rows_include_every_command_name() {
        let banner = welcome_banner_rows(false).join("\n");
        for name in commands::command_names() {
            assert!(banner.contains(name), "welcome banner missing {name}");
        }
        assert!(!banner.contains("Available Commands:"));
    }

    // ---- parse_force_and_filename ----

    #[test]
    fn parse_force_empty_is_none() {
        assert_eq!(parse_force_and_filename(""), None);
        assert_eq!(parse_force_and_filename("   "), None);
        assert_eq!(parse_force_and_filename("-f"), None);
        assert_eq!(parse_force_and_filename("-f "), None);
        assert_eq!(parse_force_and_filename(" -f"), None);
    }

    #[test]
    fn parse_force_plain_filename() {
        assert_eq!(
            parse_force_and_filename("conv.json"),
            Some((false, "conv.json"))
        );
    }

    #[test]
    fn parse_force_prefix() {
        assert_eq!(
            parse_force_and_filename("-f conv.json"),
            Some((true, "conv.json"))
        );
    }

    #[test]
    fn parse_force_suffix() {
        assert_eq!(
            parse_force_and_filename("conv.json -f"),
            Some((true, "conv.json"))
        );
    }

    // ---- strip_single_turn_system_messages ----

    #[test]
    fn strip_single_turn_removes_only_tagged_system_messages() {
        let mut cli_history = vec![
            system("SYSTEM: persistent context"),
            system(&format!("{} ephemeral question", SINGLE_TURN_SYSTEM_TAG)),
            user("hi"),
            assistant("hello"),
        ];
        cli_history
            .retain(|m| !(m.role == "system" && m.content.starts_with(SINGLE_TURN_SYSTEM_TAG)));
        assert_eq!(cli_history.len(), 3);
        assert_eq!(cli_history[0].content, "SYSTEM: persistent context");
        assert_eq!(cli_history[1].role, "user");
        assert_eq!(cli_history[2].role, "assistant");
    }

    // ---- url_encode ----

    #[test]
    fn url_encode_passes_through_unreserved() {
        assert_eq!(url_encode("abcXYZ-_.~012"), "abcXYZ-_.~012");
    }

    #[test]
    fn url_encode_escapes_reserved_and_unicode() {
        // Space, slash, colon, newline, and a multi-byte character all
        // round-trip via percent-encoding.
        assert_eq!(url_encode(" "), "%20");
        assert_eq!(url_encode("a/b:c"), "a%2Fb%3Ac");
        assert_eq!(url_encode("x\ny"), "x%0Ay");
        // U+00E9 (é) is 0xC3 0xA9 in UTF-8.
        assert_eq!(url_encode("é"), "%C3%A9");
    }

    // ---- overwrite guard ----

    #[test]
    fn overwrite_guard_blocks_existing_file_without_force() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("cubi-overwrite-{nanos}.txt"));
        std::fs::write(&path, "existing").unwrap();

        let p = path.to_str().unwrap();
        let err = check_overwrite_allowed(p, false, "/save").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("Refusing to overwrite"), "got: {msg}");
        assert!(
            msg.contains("/save -f"),
            "expected hint for /save -f, got: {msg}"
        );

        // With force=true the guard must pass even though the file exists.
        check_overwrite_allowed(p, true, "/save").expect("force should bypass");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn overwrite_guard_allows_missing_file() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("cubi-missing-{nanos}.txt"));
        let p = path.to_str().unwrap();
        check_overwrite_allowed(p, false, "/export").expect("missing file is fine");
    }

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn new_test_cli() -> ChatCLI {
        let executor = AIExecutor::new_from_env("test-model".to_string());
        ChatCLI::new(
            executor,
            None,
            Arc::new(Mutex::new(Permissions::default())),
            Arc::new(AtomicBool::new(false)),
            FileJournal::default(),
        )
    }

    fn temp_oauth_path(suffix: &str) -> String {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0))
            .as_nanos();
        std::env::temp_dir()
            .join(format!("cubi-{suffix}-{nanos}.json"))
            .display()
            .to_string()
    }

    #[test]
    fn login_and_logout_persist_and_remove_provider_tokens() {
        let _guard = env_lock().lock().expect("lock should not be poisoned");
        let oauth_path = temp_oauth_path("oauth");
        unsafe {
            std::env::set_var("CUBI_OAUTH_FILE", &oauth_path);
            std::env::remove_var("CUBI_GITHUB_API_KEY");
        }

        let cli = new_test_cli();
        cli.handle_login("GiTHub token123 --refresh-token refresh123 --expires-in 60");

        let after_login = oauth::OAuthStore::load();
        let token = after_login
            .get_provider("github")
            .expect("provider token should exist after /login");
        assert_eq!(token.access_token, "token123");
        assert_eq!(token.refresh_token.as_deref(), Some("refresh123"));
        assert_eq!(
            std::env::var("CUBI_GITHUB_API_KEY").ok().as_deref(),
            Some("token123")
        );

        cli.handle_logout("github");
        let after_logout = oauth::OAuthStore::load();
        assert!(after_logout.get_provider("github").is_none());
        assert!(std::env::var("CUBI_GITHUB_API_KEY").is_err());

        let _ = std::fs::remove_file(&oauth_path);
        unsafe {
            std::env::remove_var("CUBI_OAUTH_FILE");
            std::env::remove_var("CUBI_GITHUB_API_KEY");
        }
    }

    #[test]
    fn pin_inserts_tagged_system_message_and_returns_1based_index() {
        let mut cli = new_test_cli();
        let history_before = cli.history.len();
        let idx = cli.pin("remember Project X spec");
        assert_eq!(idx, 1);
        assert_eq!(cli.pinned.len(), 1);
        assert_eq!(cli.history.len(), history_before + 1);
        let pinned = cli
            .history
            .iter()
            .find(|m| m.role == "system" && m.content.starts_with(PINNED_SYSTEM_TAG))
            .expect("pinned tagged system message must be present");
        assert!(pinned.content.contains("Project X"));
        // Pinned items live in the leading system-preamble band, never
        // after a non-system message.
        let pinned_pos = cli
            .history
            .iter()
            .position(|m| m.role == "system" && m.content.starts_with(PINNED_SYSTEM_TAG))
            .unwrap();
        let first_non_system = cli.history.iter().position(|m| m.role != "system");
        if let Some(p) = first_non_system {
            assert!(
                pinned_pos < p,
                "pin should sit before any user/assistant turn"
            );
        }

        let idx2 = cli.pin("second pin");
        assert_eq!(idx2, 2);
        assert_eq!(cli.pinned, vec!["remember Project X spec", "second pin"]);
    }

    #[test]
    fn unpin_removes_pin_and_its_tagged_system_message() {
        let mut cli = new_test_cli();
        cli.pin("first");
        cli.pin("second");
        let after_pins = cli.history.len();

        let removed = cli.unpin(1).expect("first pin should exist");
        assert_eq!(removed, "first");
        assert_eq!(cli.pinned, vec!["second"]);
        assert_eq!(cli.history.len(), after_pins - 1);
        // No remaining tagged system message for "first"; "second" stays.
        assert!(
            cli.history
                .iter()
                .filter(|m| m.role == "system" && m.content.starts_with(PINNED_SYSTEM_TAG))
                .count()
                == 1
        );

        assert!(cli.unpin(99).is_none());
        assert!(cli.unpin(0).is_none());
    }

    #[test]
    fn pinned_system_messages_survive_compact_rebuild() {
        // Simulates the rebuild step from `compact()` directly: pinned
        // system messages must be retained verbatim and reappear before
        // the summary in the new history.
        let mut cli = new_test_cli();
        cli.pin("keep me through compaction");
        // Stuff some user/assistant turns to be summarized.
        for i in 0..10 {
            cli.history.push(Message::text("user", format!("u{}", i)));
            cli.history
                .push(Message::text("assistant", format!("a{}", i)));
        }
        // Replicate the rebuild rule: every system message in the
        // pre-cutoff slice survives. With KEEP_RECENT = 6, cutoff_idx
        // is at index of the 6th-from-last non-system message.
        let non_system_indices: Vec<usize> = cli
            .history
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role != "system")
            .map(|(i, _)| i)
            .collect();
        let cutoff_idx = non_system_indices[non_system_indices.len() - 6];
        let preserved: Vec<&Message> = cli.history[..cutoff_idx]
            .iter()
            .filter(|m| m.role == "system")
            .collect();
        assert!(
            preserved
                .iter()
                .any(|m| m.content.starts_with(PINNED_SYSTEM_TAG)
                    && m.content.contains("keep me through compaction")),
            "pinned system message must be preserved across compact rebuild"
        );
    }

    #[test]
    fn show_pins_with_empty_list_does_not_panic() {
        let cli = new_test_cli();
        cli.show_pins();
    }
}
