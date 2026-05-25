use anyhow::{Context, Result};
use colored::*;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use crate::executor::AIExecutor;
use crate::mcp_manager::McpManager;
use crate::ollama::Message;
use crate::project_memory;
use crate::todos::TodoList;
use std::fs;
use std::path::Path;

/// Sentinel prefix used to tag system messages that should only influence the
/// next assistant turn (e.g. `/ask`). After the model responds once, any
/// system message starting with this prefix is stripped from `history` so the
/// hint does not poison every subsequent turn.
const SINGLE_TURN_SYSTEM_TAG: &str = "SYSTEM[single-turn]:";

/// Prefix used to tag the auto-injected project-memory system message so it
/// can be located and replaced on `/memory-reload`.
const PROJECT_MEMORY_PREFIX: &str = "SYSTEM: Project memory loaded from";

pub struct ChatCLI {
    executor: AIExecutor,
    history: Vec<Message>,
    mcp_manager: Option<McpManager>,
    todos: TodoList,
    plan_mode: bool,
}

impl ChatCLI {
    pub fn new(executor: AIExecutor, mcp_manager: Option<McpManager>) -> Self {
        let mut cli = Self {
            executor,
            history: Vec::new(),
            mcp_manager,
            todos: TodoList::load_for_current_dir(),
            plan_mode: false,
        };

        // Auto-inject project memory (AICHAT.md) into context, if present.
        cli.inject_project_memory();

        // Auto-inject MCP tools into context
        if let Some(mcp) = &cli.mcp_manager
            && mcp.has_tools()
        {
            let tools = mcp.list_tools();
            let mut msg = String::from("SYSTEM: You have access to these MCP tools:\n\n");
            for t in tools {
                msg.push_str(&format!("- {}: {}\n", t.name, t.description));
            }
            msg.push_str("\nWhen relevant, tell users they can execute these with /mcp-call <tool> <args>");

            cli.history.push(Message {
                role: "system".to_string(),
                content: msg,
            });
        }
    
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

    pub async fn run(&mut self) -> Result<()> {
        self.print_welcome();

        let mut rl = DefaultEditor::new()?;

        loop {
            let prompt = if self.plan_mode {
                format!(
                    "{}{} ",
                    "[plan] ".bright_yellow().bold(),
                    "You:".bright_green().bold()
                )
            } else {
                format!("{} ", "You:".bright_green().bold())
            };

            match rl.readline(&prompt) {
                Ok(line) => {
                    let input = line.trim();
                    
                    if input.is_empty() {
                        continue;
                    }

                    // Handle commands
                    if input.starts_with('/') {
                        if !self.handle_command(input).await? {
                            break;
                        }
                        continue;
                    }

                    // Add line to readline history
                    rl.add_history_entry(input)?;

                    // Add user message to history
                    self.history.push(Message {
                        role: "user".to_string(),
                        content: input.to_string(),
                    });

                    // Get AI response
                    print!("{} ", "AI:".bright_blue().bold());
                    
                    match self.executor.chat(self.history.clone()).await {
                        Ok(response) => {
                            println!("{}\n", response.bright_white());
                            
                            // Add assistant response to history
                            self.history.push(Message {
                                role: "assistant".to_string(),
                                content: response,
                            });

                            // Drop any system messages tagged as single-turn
                            // (e.g. from `/ask`) so they don't keep nudging
                            // every subsequent turn.
                            self.strip_single_turn_system_messages();
                        }
                        Err(e) => {
                            eprintln!("{} {}\n", "Error:".bright_red().bold(), e);
                        }
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("{}",  "Use /quit to exit".yellow());
                    continue;
                }
                Err(ReadlineError::Eof) => {
                    break;
                }
                Err(err) => {
                    eprintln!("Error: {:?}", err);
                    break;
                }
            }
        }

        Ok(())
    }

    async fn handle_command(&mut self, cmd: &str) -> Result<bool> {
        match cmd {
            "/quit" | "/exit" => {
                println!("{}", "Goodbye!".bright_cyan());
                return Ok(false);
            }
            "/clear" => {
                self.history.clear();
                println!("{}", "Conversation history cleared.".yellow());
            }
            "/history" => {
                self.show_history();
            }
            "/help" => {
                self.show_help();
            }
            "/model" => {
                println!("Current model: {}", self.executor.get_model().bright_cyan());
            }
            "/mcp-tools" => {
                self.show_mcp_tools();
            }
            cmd if cmd.starts_with("/mcp-call ") => {
                if self.plan_mode {
                    println!(
                        "{} Plan mode is on — refusing /mcp-call. Toggle off with /plan first.",
                        "✗".bright_red()
                    );
                    return Ok(true);
                }
                let rest = cmd.strip_prefix("/mcp-call ").unwrap().trim();
                let parts: Vec<&str> = rest.splitn(2, ' ').collect();
                
                if parts.len() < 2 {
                    println!("{} Usage: /mcp-call <tool_name> <json_args>", 
                        "Info:".bright_yellow());
                    println!("Example: /mcp-call add {{\"a\": 5, \"b\": 3}}");
                } else {
                    let tool_name = parts[0];
                    let args_str = parts[1];
                    
                    match serde_json::from_str(args_str) {
                        Ok(args) => {
                            if let Err(e) = self.call_mcp_tool(tool_name, args).await {
                                eprintln!("{} {}", "Error:".bright_red(), e);
                            }
                        }
                        Err(e) => {
                            eprintln!("{} Invalid JSON: {}", "Error:".bright_red(), e);
                        }
                    }
                }
            }
            "/mcp-reload" => {
                if let Err(e) = self.reload_mcp().await {
                    eprintln!("{} Failed to reload MCP: {}", "Error:".bright_red(), e);
                } else {
                    println!("{} MCP configuration reloaded", "✓".bright_green());
                }
            }
            cmd if cmd.starts_with("/model ") => {
                let model = cmd.strip_prefix("/model ").unwrap().trim();
                match self.executor.switch_model(model.to_string()).await {
                    Ok(_) => {
                        println!("{} Switched to model: {}", "✓".bright_green(), model.bright_cyan());
                        self.history.clear();
                    }
                    Err(e) => {
                        eprintln!("{} {}", "Error:".bright_red(), e);
                    }
                }
            }
            cmd if cmd.starts_with("/save ") => {
                let rest = cmd.strip_prefix("/save ").unwrap().trim();
                match parse_force_and_filename(rest) {
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
            "/save" => {
                println!("{} Usage: /save [-f] <filename>", "Info:".bright_yellow());
                println!("       Pass -f to overwrite an existing file.");
                println!("Example: /save my_chat.json");
            }
            cmd if cmd.starts_with("/load ") => {
                let filename = cmd.strip_prefix("/load ").unwrap().trim();
                if filename.is_empty() {
                    println!("{} Usage: /load <filename>", "Info:".bright_yellow());
                } else if let Err(e) = self.load_conversation(filename) {
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
                        filename.bright_cyan()
                    );
                }
            }
            "/load" => {
                println!("{} Usage: /load <filename>", "Info:".bright_yellow());
                println!("Example: /load my_chat.json");
            }
            cmd if cmd.starts_with("/batch ") => {
                let filename = cmd.strip_prefix("/batch ").unwrap().trim();
                if filename.is_empty() {
                    println!("{} Usage: /batch <filename>", "Info:".bright_yellow());
                } else {
                    match self.process_batch_file(filename).await {
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
            "/batch" => {
                println!("{} Usage: /batch <filename>", "Info:".bright_yellow());
                println!("Example: /batch prompts.txt");
                println!("\nBatch file format (one prompt per line, blank lines and #-comments are skipped):");
                println!("  # warm-up");
                println!("  What is Rust?");
                println!("  Write hello world in Python");
                println!("  Explain recursion");
            }
            "/version" => {
                println!("{} {}", "ai-chat-cli".bright_cyan(), env!("CARGO_PKG_VERSION"));
            }
            "/status" => {
                self.show_status();
            }
            "/plan" => {
                self.toggle_plan_mode();
            }
            "/init" => {
                self.run_init();
            }
            "/memory" => {
                self.show_memory();
            }
            "/memory-reload" => {
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
            "/todos" => {
                self.todos.render();
            }
            cmd if cmd.starts_with("/todo-add ") => {
                let text = cmd.strip_prefix("/todo-add ").unwrap().trim();
                if text.is_empty() {
                    println!("{} Usage: /todo-add <text>", "Info:".bright_yellow());
                } else {
                    self.todos.add(text);
                    self.persist_todos();
                    println!("{} Added todo", "✓".bright_green());
                }
            }
            "/todo-add" => {
                println!("{} Usage: /todo-add <text>", "Info:".bright_yellow());
            }
            cmd if cmd.starts_with("/todo-done ") => {
                let arg = cmd.strip_prefix("/todo-done ").unwrap().trim();
                match arg.parse::<usize>() {
                    Ok(n) => {
                        if self.todos.mark_done(n) {
                            self.persist_todos();
                            println!("{} Marked todo {} as done", "✓".bright_green(), n);
                        } else {
                            eprintln!(
                                "{} No todo with index {}",
                                "Error:".bright_red(),
                                n
                            );
                        }
                    }
                    Err(_) => {
                        eprintln!(
                            "{} Usage: /todo-done <index>",
                            "Error:".bright_red()
                        );
                    }
                }
            }
            "/todo-done" => {
                println!("{} Usage: /todo-done <index>", "Info:".bright_yellow());
            }
            cmd if cmd.starts_with("/todo-rm ") => {
                let arg = cmd.strip_prefix("/todo-rm ").unwrap().trim();
                match arg.parse::<usize>() {
                    Ok(n) => {
                        if self.todos.remove(n) {
                            self.persist_todos();
                            println!("{} Removed todo {}", "✓".bright_green(), n);
                        } else {
                            eprintln!(
                                "{} No todo with index {}",
                                "Error:".bright_red(),
                                n
                            );
                        }
                    }
                    Err(_) => {
                        eprintln!("{} Usage: /todo-rm <index>", "Error:".bright_red());
                    }
                }
            }
            "/todo-rm" => {
                println!("{} Usage: /todo-rm <index>", "Info:".bright_yellow());
            }
            "/todo-clear" => {
                self.todos.clear();
                self.persist_todos();
                println!("{} Cleared todos", "✓".bright_green());
            }
            cmd if cmd.starts_with("/ask ") => {
                let question = cmd.strip_prefix("/ask ").unwrap().trim();
                if question.is_empty() {
                    println!("{} Usage: /ask <question>", "Info:".bright_yellow());
                } else {
                    self.ask_user(question);
                }
            }
            "/ask" => {
                println!("{} Usage: /ask <question>", "Info:".bright_yellow());
                println!("Records a clarifying question to be answered on the next turn.");
            }
            cmd if cmd.starts_with("/export ") => {
                let rest = cmd.strip_prefix("/export ").unwrap().trim();
                match parse_force_and_filename(rest) {
                    Some((force, filename)) => match self.export_markdown(filename, force) {
                        Ok(()) => println!(
                            "{} Conversation exported to {}",
                            "✓".bright_green(),
                            filename.bright_cyan()
                        ),
                        Err(e) => eprintln!("{} {}", "Error:".bright_red(), e),
                    },
                    None => {
                        println!("{} Usage: /export [-f] <filename.md>", "Info:".bright_yellow());
                        println!("       Pass -f to overwrite an existing file.");
                    }
                }
            }
            "/export" => {
                println!("{} Usage: /export [-f] <filename.md>", "Info:".bright_yellow());
                println!("       Pass -f to overwrite an existing file.");
            }
            _ => {
                println!("{} {}", "Unknown command:".bright_red(), cmd);
                println!("Type {} for available commands", "/help".bright_cyan());
            }
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
                .chat(vec![Message {
                    role: "user".to_string(),
                    content: prompt.clone(),
                }])
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
                    println!("\n  {} {} (from {})", 
                        "●".bright_green(), 
                        tool.name.bright_cyan(),
                        server.bright_magenta());
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

    async fn reload_mcp(&mut self) -> Result<()> {
        // Shutdown existing MCP connections
        if let Some(mcp) = &mut self.mcp_manager {
            mcp.shutdown().await;
        }
        
        // Reload configuration and reconnect
        self.mcp_manager = match McpManager::new().await {
            Ok(manager) => Some(manager),
            Err(e) => {
                eprintln!("{} {}", "Warning:".bright_yellow(), e);
                None
            }
        };
        
        Ok(())
    }
    

    /// Single source of truth for slash commands shown in `/help` and the
    /// startup banner. `(command, description)` pairs.
    fn command_help() -> &'static [(&'static str, &'static str)] {
        &[
            ("/help", "Show this help message"),
            ("/status", "Show session status"),
            ("/plan", "Toggle plan mode (read-only)"),
            ("/init", "Create starter AICHAT.md"),
            ("/memory", "Show project memory (AICHAT.md)"),
            ("/memory-reload", "Re-read AICHAT.md from disk"),
            ("/todos", "List todos"),
            ("/todo-add <text>", "Add a todo"),
            ("/todo-done <n>", "Mark todo n as done"),
            ("/todo-rm <n>", "Remove todo n"),
            ("/todo-clear", "Clear all todos"),
            ("/ask <q>", "Record a clarifying question (single-turn)"),
            ("/clear", "Clear conversation history"),
            ("/history", "Show conversation history"),
            ("/export [-f] <f.md>", "Export conversation as Markdown"),
            ("/save [-f] <f.json>", "Save conversation (-f overwrites)"),
            ("/load <f.json>", "Load conversation"),
            ("/batch <f>", "Process batch file"),
            ("/mcp-tools", "List available MCP tools"),
            ("/mcp-call <t> <a>", "Call MCP tool"),
            ("/mcp-reload", "Reload MCP configuration"),
            ("/model", "Show current model"),
            ("/model <name>", "Switch to a different model"),
            ("/version", "Show version"),
            ("/quit", "Exit the chat"),
        ]
    }

    fn print_welcome(&self) {
        println!("\n{}", "=".repeat(60).bright_cyan());
        println!("{}", "  AI Chat CLI - Powered by Repartir".bright_cyan().bold());
        println!("{}", "=".repeat(60).bright_cyan());
        println!("\n{}", "Commands:".bright_yellow().bold());
        for (cmd, desc) in Self::command_help() {
            println!("  {} - {}", cmd.bright_cyan(), desc);
        }
        println!("\n{}\n", "Start chatting! (Ctrl+C to interrupt, /quit to exit)".bright_white());
    }

    fn show_help(&self) {
        println!("\n{}", "Available Commands:".bright_yellow().bold());
        for (cmd, desc) in Self::command_help() {
            println!("  {} - {}", cmd.bright_cyan(), desc);
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
        println!("\n{}", "Status:".bright_yellow().bold());
        println!("  {}: {}", "model".bright_cyan(), self.executor.get_model());
        println!("  {}: {}", "messages".bright_cyan(), self.history.len());
        println!(
            "  {}: {}",
            "plan mode".bright_cyan(),
            if self.plan_mode { "on".bright_green() } else { "off".bright_black() }
        );
        println!(
            "  {}: {} ({} pending)",
            "todos".bright_cyan(),
            self.todos.len(),
            self.todos.pending()
        );
        println!("  {}: {}", "mcp tools".bright_cyan(), mcp_tool_count);
        println!();
    }

    fn toggle_plan_mode(&mut self) {
        self.plan_mode = !self.plan_mode;
        if self.plan_mode {
            self.history.push(Message {
                role: "system".to_string(),
                content:
                    "SYSTEM: Plan mode is ON. Do not modify files or run \
                     destructive commands. Produce a plan and wait for the \
                     user to confirm before applying changes."
                        .to_string(),
            });
            println!(
                "{} Plan mode {}",
                "✓".bright_green(),
                "enabled".bright_green()
            );
        } else {
            self.history.push(Message {
                role: "system".to_string(),
                content: "SYSTEM: Plan mode is OFF. Normal tool use is allowed."
                    .to_string(),
            });
            println!(
                "{} Plan mode {}",
                "✓".bright_green(),
                "disabled".bright_black()
            );
        }
    }

    fn persist_todos(&self) {
        if let Err(e) = self.todos.save() {
            eprintln!(
                "{} Failed to persist todos: {}",
                "Warn:".bright_yellow(),
                e
            );
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
        self.history.push(Message {
            role: "system".to_string(),
            content: format!(
                "{} The user has a clarifying question they want addressed \
                 directly and concisely on the next turn:\n\n{}",
                SINGLE_TURN_SYSTEM_TAG, question
            ),
        });
        println!(
            "{} Question recorded. It will be highlighted on the next turn only.",
            "✓".bright_green()
        );
    }

    /// Removes any system messages tagged as single-turn (see
    /// [`SINGLE_TURN_SYSTEM_TAG`]) from the history.
    fn strip_single_turn_system_messages(&mut self) {
        self.history.retain(|m| {
            !(m.role == "system" && m.content.starts_with(SINGLE_TURN_SYSTEM_TAG))
        });
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
    /// re-reads `AICHAT.md` (walking up the directory tree) into history.
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
        self.history.retain(|m| {
            !(m.role == "system" && m.content.starts_with(PROJECT_MEMORY_PREFIX))
        });

        let Ok(Some((path, memory))) = project_memory::read_memory_with_path() else {
            return;
        };

        let msg = Message {
            role: "system".to_string(),
            content: format!(
                "{} {}:\n\n{}",
                PROJECT_MEMORY_PREFIX,
                path.display(),
                memory.trim()
            ),
        };

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
        out.push_str("# ai-chat-cli conversation\n\n");
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

/// Refuses to overwrite `filename` unless `force` is true. Shared between
/// `/export` and `/save` so the error wording stays in sync and there's only
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

#[cfg(test)]
mod tests {
    use super::*;

    fn user(s: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: s.to_string(),
        }
    }

    fn assistant(s: &str) -> Message {
        Message {
            role: "assistant".to_string(),
            content: s.to_string(),
        }
    }

    fn system(s: &str) -> Message {
        Message {
            role: "system".to_string(),
            content: s.to_string(),
        }
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
            system(&format!(
                "{} ephemeral question",
                SINGLE_TURN_SYSTEM_TAG
            )),
            user("hi"),
            assistant("hello"),
        ];
        cli_history.retain(|m| {
            !(m.role == "system" && m.content.starts_with(SINGLE_TURN_SYSTEM_TAG))
        });
        assert_eq!(cli_history.len(), 3);
        assert_eq!(cli_history[0].content, "SYSTEM: persistent context");
        assert_eq!(cli_history[1].role, "user");
        assert_eq!(cli_history[2].role, "assistant");
    }

    // ---- command_help / handle_command parity ----
    //
    // Guards against the drift that bit us before this PR: a command added
    // to `handle_command` but not to `command_help()` (or vice versa). We
    // assert every command listed in the help table is mentioned by name in
    // the handler source, and that a handful of must-have commands are in
    // the help table.

    #[test]
    fn command_help_lists_match_handler() {
        let source = include_str!("cli.rs");
        for (cmd, _desc) in ChatCLI::command_help() {
            // Strip any argument placeholder after the first space so we look
            // up just the `/foo` part — that's what appears in `handle_command`.
            let bare = cmd.split_whitespace().next().unwrap();
            assert!(
                source.contains(&format!("\"{}\"", bare))
                    || source.contains(&format!("starts_with(\"{} \")", bare)),
                "command `{}` listed in help but not handled in cli.rs",
                bare
            );
        }
    }

    #[test]
    fn command_help_covers_core_commands() {
        let cmds: Vec<&str> = ChatCLI::command_help()
            .iter()
            .map(|(c, _)| c.split_whitespace().next().unwrap())
            .collect();
        for must in [
            "/help",
            "/quit",
            "/save",
            "/load",
            "/batch",
            "/export",
            "/memory",
            "/memory-reload",
            "/todo-add",
            "/todo-rm",
        ] {
            assert!(cmds.contains(&must), "missing core command in help: {must}");
        }
    }

    // ---- overwrite guard ----

    #[test]
    fn overwrite_guard_blocks_existing_file_without_force() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("ai-chat-cli-overwrite-{nanos}.txt"));
        std::fs::write(&path, "existing").unwrap();

        let p = path.to_str().unwrap();
        let err = check_overwrite_allowed(p, false, "/save").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("Refusing to overwrite"), "got: {msg}");
        assert!(msg.contains("/save -f"), "expected hint for /save -f, got: {msg}");

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
        let path = std::env::temp_dir().join(format!("ai-chat-cli-missing-{nanos}.txt"));
        let p = path.to_str().unwrap();
        check_overwrite_allowed(p, false, "/export").expect("missing file is fine");
    }
}
