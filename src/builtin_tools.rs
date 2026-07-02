use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command as TokioCommand};
use tokio::sync::Mutex as AsyncMutex;
use tokio::time::timeout;

use crate::file_rollback::FileJournal;
use crate::permissions::Permissions;
use crate::repomap::{RepoMap, RepoMapOptions};

const READ_FILE_DEFAULT_MAX_LINES: usize = 400;
const READ_FILE_DEFAULT_MAX_BYTES: usize = 50 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuiltinTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub content: Vec<ToolContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

impl ToolResult {
    pub fn success(text: String) -> Self {
        Self {
            content: vec![ToolContent {
                content_type: "text".to_string(),
                text,
            }],
            is_error: None,
        }
    }

    pub fn error(text: String) -> Self {
        Self {
            content: vec![ToolContent {
                content_type: "text".to_string(),
                text,
            }],
            is_error: Some(true),
        }
    }
}

pub struct BuiltinToolRegistry {
    tools: Vec<BuiltinTool>,
    /// Shared trust + sandbox state. Consulted by the write/exec tools
    /// (`bash`, `edit_file`, `write_file`) before they touch the disk.
    permissions: Arc<Mutex<Permissions>>,
    /// Plan-mode flag. When set, write/exec tools refuse so the model can
    /// reason about the change without applying it. The CLI flips this in
    /// response to `/plan`; the atomic + `Arc` lets the registry observe
    /// changes without taking a lock on every tool call.
    plan_mode: Arc<AtomicBool>,
    /// Pre-image journal for `/rewind` — see `file_rollback.rs`. Cloned
    /// (cheap `Arc`) into every tool call so `edit_file`/`write_file`
    /// can capture the original bytes before mutating the disk.
    journal: FileJournal,
    /// Long-lived shell sessions owned by the REPL tool. Keyed by
    /// caller-supplied session id (the model picks a stable string and
    /// reuses it across `repl_eval` calls). Wrapped in a tokio
    /// `AsyncMutex` so it can be held across the awaits needed to read
    /// from the child process.
    repls: Arc<AsyncMutex<HashMap<String, ReplSession>>>,
    /// Long-lived headless-browser sessions. Same opaque session-id
    /// model as the REPL tools. Only present when the `browser` cargo
    /// feature is enabled — without it the field, the tool specs, and
    /// the `execute` arms are all absent.
    #[cfg(feature = "browser")]
    browsers: crate::browser_tool::BrowserManager,
}

/// One long-lived REPL backed by `bash -i`. We use a sentinel marker to
/// know when each `repl_eval` has finished — bash never sends EOF on its
/// own and reading "until idle" is racy. The marker is unique per session
/// so concurrent reads from different sessions can't be confused.
struct ReplSession {
    stdin: ChildStdin,
    /// Captures stdout *and* stderr (the child is spawned with stderr
    /// redirected to stdout) one line at a time. The reader task ends
    /// when the child exits.
    reader: BufReader<tokio::process::ChildStdout>,
    /// Sentinel suffix used to identify the end of one eval's output.
    /// We prepend a random component on `repl_start` so a model that
    /// echoes the sentinel literally can't fool us into ending early.
    sentinel: String,
    /// Kept alive so the child isn't reaped while the session is open.
    /// The wait happens implicitly when the session is dropped.
    _child: Child,
}

impl BuiltinToolRegistry {
    /// Convenience constructor that wires up a default (no-op) journal.
    /// Kept for any callers that don't care about `/rewind` integration.
    #[allow(dead_code)]
    pub fn new(permissions: Arc<Mutex<Permissions>>, plan_mode: Arc<AtomicBool>) -> Self {
        Self::with_journal(permissions, plan_mode, FileJournal::default())
    }

    /// Same as [`Self::new`] but lets the caller supply a journal so the
    /// CLI's `/rewind` can roll back any mutations recorded by
    /// `edit_file` and `write_file`. The two-constructor split keeps
    /// existing callers (and tests) working without forcing them to
    /// invent a journal they don't care about.
    pub fn with_journal(
        permissions: Arc<Mutex<Permissions>>,
        plan_mode: Arc<AtomicBool>,
        journal: FileJournal,
    ) -> Self {
        let tools = vec![
            Self::bash_tool(),
            Self::read_file_tool(),
            Self::list_files_tool(),
            Self::search_glob_tool(),
            Self::search_tools_tool(),
            Self::grep_tool(),
            Self::edit_file_tool(),
            Self::write_file_tool(),
            Self::think_tool(),
            Self::worktree_tool(),
            Self::web_fetch_tool(),
            Self::web_search_tool(),
            Self::repl_start_tool(),
            Self::repl_eval_tool(),
            Self::repl_close_tool(),
            Self::notebook_tool(),
            Self::lsp_tool(),
            Self::shell_tool(),
            Self::sleep_tool(),
            Self::schedule_tool(),
            Self::brief_tool(),
            Self::synthetic_output_tool(),
            Self::send_message_tool(),
            Self::recv_messages_tool(),
            Self::remote_trigger_tool(),
            Self::notify_tool(),
            Self::prevent_sleep_tool(),
            Self::repo_map_tool(),
            #[cfg(feature = "browser")]
            Self::browser_open_tool(),
            #[cfg(feature = "browser")]
            Self::browser_eval_tool(),
            #[cfg(feature = "browser")]
            Self::browser_screenshot_tool(),
            #[cfg(feature = "browser")]
            Self::browser_text_tool(),
            #[cfg(feature = "browser")]
            Self::browser_close_tool(),
        ];

        Self {
            tools,
            permissions,
            plan_mode,
            journal,
            repls: Arc::new(AsyncMutex::new(HashMap::new())),
            #[cfg(feature = "browser")]
            browsers: crate::browser_tool::BrowserManager::new(),
        }
    }

    /// Returns a short, model-facing refusal message when plan mode is on.
    fn plan_mode_refusal(tool: &str) -> String {
        format!(
            "Refusing `{tool}`: plan mode is ON. Produce a plan and ask the user to disable \
             plan mode (`/plan`) before retrying."
        )
    }

    pub fn list_tools(&self) -> &[BuiltinTool] {
        &self.tools
    }

    pub async fn execute(&self, name: &str, args: serde_json::Value) -> Result<ToolResult> {
        match name {
            "bash" => self.execute_bash(args).await,
            "read_file" => self.execute_read_file(args),
            "list_files" => self.execute_list_files(args),
            "search_glob" => self.execute_search_glob(args),
            "search_tools" => self.execute_search_tools(args),
            "grep" => self.execute_grep(args),
            "edit_file" => self.execute_edit_file(args),
            "write_file" => self.execute_write_file(args),
            "think" => self.execute_think(args),
            "worktree" => self.execute_worktree(args),
            "web_fetch" => self.execute_web_fetch(args).await,
            "web_search" => self.execute_web_search(args).await,
            "repl_start" => self.execute_repl_start(args).await,
            "repl_eval" => self.execute_repl_eval(args).await,
            "repl_close" => self.execute_repl_close(args).await,
            "notebook" => self.execute_notebook(args),
            "lsp" => self.execute_lsp(args).await,
            "shell" => self.execute_shell(args).await,
            "sleep" => self.execute_sleep(args).await,
            "schedule" => self.execute_schedule(args),
            "brief" => self.execute_brief(args),
            "synthetic_output" => self.execute_synthetic_output(args),
            "send_message" => self.execute_send_message(args),
            "recv_messages" => self.execute_recv_messages(args),
            "remote_trigger" => self.execute_remote_trigger(args),
            "notify" => self.execute_notify(args),
            "prevent_sleep" => self.execute_prevent_sleep(args).await,
            "repo_map" => self.execute_repo_map(args),
            #[cfg(feature = "browser")]
            "browser_open" => self.execute_browser_open(args).await,
            #[cfg(feature = "browser")]
            "browser_eval" => self.execute_browser_eval(args).await,
            #[cfg(feature = "browser")]
            "browser_screenshot" => self.execute_browser_screenshot(args).await,
            #[cfg(feature = "browser")]
            "browser_text" => self.execute_browser_text(args).await,
            #[cfg(feature = "browser")]
            "browser_close" => self.execute_browser_close(args).await,
            _ => anyhow::bail!("Unknown built-in tool: {}", name),
        }
    }

    // Tool Definitions

    fn bash_tool() -> BuiltinTool {
        BuiltinTool {
            name: "bash".to_string(),
            description: "Execute shell commands in a secure environment. Use for running CLI tools, scripts, and system commands.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 30)",
                        "default": 30
                    }
                },
                "required": ["command"]
            }),
        }
    }

    fn read_file_tool() -> BuiltinTool {
        BuiltinTool {
            name: "read_file".to_string(),
            description: "Read the contents of a file from the filesystem. Use start_line/end_line for targeted reads of large files; unbounded large-file reads are truncated.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    },
                    "start_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Starting line number (1-indexed, optional). If omitted with end_line, reads from line 1."
                    },
                    "end_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Ending line number (1-indexed, inclusive, optional). If omitted with start_line, reads through end of file."
                    }
                },
                "required": ["path"]
            }),
        }
    }

    fn repo_map_tool() -> BuiltinTool {
        BuiltinTool {
            name: "repo_map".to_string(),
            description:
                "Return a compact outline of the project's files and top-level symbols. Useful as a first step to orient in an unfamiliar codebase."
                    .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "description": "Optional directory to map (default: cwd)"
                    },
                    "max_files": {
                        "type": "integer",
                        "description": "Maximum files to include (default: 200)",
                        "default": 200
                    },
                    "max_symbols_per_file": {
                        "type": "integer",
                        "description": "Maximum symbols per file (default: 20)",
                        "default": 20
                    }
                }
            }),
        }
    }

    fn execute_repo_map(&self, args: serde_json::Value) -> Result<ToolResult> {
        let cwd = std::env::current_dir().context("Could not resolve current directory")?;
        let scope = match args["scope"].as_str() {
            Some(s) if !s.is_empty() => {
                let p = Path::new(s);
                if p.is_absolute() {
                    p.to_path_buf()
                } else {
                    cwd.join(p)
                }
            }
            _ => cwd.clone(),
        };

        // L1 trust: refuse to map directories outside the cwd unless
        // the caller has trusted the parent. The repo-map walks every
        // file's metadata, so this is the same access pattern as `read_file`.
        let canonical_scope = fs::canonicalize(&scope).unwrap_or_else(|_| scope.clone());
        let canonical_cwd = fs::canonicalize(&cwd).unwrap_or_else(|_| cwd.clone());
        if !canonical_scope.starts_with(&canonical_cwd) {
            let trusted = self
                .permissions
                .lock()
                .map(|p| {
                    p.trusted_roots()
                        .any(|root| canonical_scope.starts_with(root))
                })
                .unwrap_or(false);
            if !trusted {
                return Ok(ToolResult::error(format!(
                    "Refusing repo_map: scope {} is outside the current working directory and not in a trusted path. Add the directory with `/add-dir` or `/trust` first.",
                    canonical_scope.display()
                )));
            }
        }

        let opts = RepoMapOptions {
            scope: Some(canonical_scope.clone()),
            max_files: args["max_files"]
                .as_u64()
                .map(|n| n as usize)
                .unwrap_or(200),
            max_symbols_per_file: args["max_symbols_per_file"]
                .as_u64()
                .map(|n| n as usize)
                .unwrap_or(20),
        };
        match RepoMap::build(&canonical_scope, &opts) {
            Ok(outline) => Ok(ToolResult::success(RepoMap::render(&outline))),
            Err(e) => Ok(ToolResult::error(format!("repo_map failed: {e}"))),
        }
    }

    fn list_files_tool() -> BuiltinTool {
        BuiltinTool {
            name: "list_files".to_string(),
            description:
                "List files and directories with metadata (size, modified time, permissions)."
                    .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list (default: current directory)",
                        "default": "."
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "List recursively",
                        "default": false
                    }
                }
            }),
        }
    }

    fn search_glob_tool() -> BuiltinTool {
        BuiltinTool {
            name: "search_glob".to_string(),
            description:
                "Search for files matching a glob pattern (e.g., '**/*.rs', 'src/**/*.json')."
                    .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match"
                    },
                    "base_path": {
                        "type": "string",
                        "description": "Base directory to search from (default: current directory)",
                        "default": "."
                    }
                },
                "required": ["pattern"]
            }),
        }
    }

    fn search_tools_tool() -> BuiltinTool {
        BuiltinTool {
            name: "search_tools".to_string(),
            description: "Search available tools by name or description keyword".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    fn grep_tool() -> BuiltinTool {
        BuiltinTool {
            name: "grep".to_string(),
            description: "Search for text patterns in files using regex. For better performance, consider using 'rg' (ripgrep) via bash tool.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Search recursively in directories",
                        "default": false
                    },
                    "ignore_case": {
                        "type": "boolean",
                        "description": "Case-insensitive search",
                        "default": false
                    }
                },
                "required": ["pattern", "path"]
            }),
        }
    }

    fn edit_file_tool() -> BuiltinTool {
        BuiltinTool {
            name: "edit_file".to_string(),
            description: "Edit a file by performing exact string replacement. The old_text must match exactly.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit"
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to replace (must match exactly)"
                    },
                    "new_text": {
                        "type": "string",
                        "description": "New text to insert"
                    }
                },
                "required": ["path", "old_text", "new_text"]
            }),
        }
    }

    fn write_file_tool() -> BuiltinTool {
        BuiltinTool {
            name: "write_file".to_string(),
            description: "Write content to a file, creating it if it doesn't exist or overwriting if it does.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }),
        }
    }

    fn think_tool() -> BuiltinTool {
        BuiltinTool {
            name: "think".to_string(),
            description: "A no-operation tool for internal reasoning and planning. Use this to think through complex problems step by step.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "thoughts": {
                        "type": "string",
                        "description": "Your internal thoughts and reasoning"
                    }
                },
                "required": ["thoughts"]
            }),
        }
    }

    // Tool Implementations

    async fn execute_bash(&self, args: serde_json::Value) -> Result<ToolResult> {
        let command = args["command"]
            .as_str()
            .context("Missing 'command' parameter")?;

        let timeout_secs = args["timeout"].as_u64().unwrap_or(30);

        // Plan mode: refuse before doing anything else, so the model gets
        // a clear signal that arbitrary execution is off the table.
        if self.plan_mode.load(Ordering::SeqCst) {
            return Ok(ToolResult::error(Self::plan_mode_refusal("bash")));
        }

        // Permissions: the cwd must be a trusted project.
        let cwd = std::env::current_dir().context("Could not read cwd")?;
        if let Err(e) = self.permissions.lock().unwrap().check_exec(&cwd) {
            return Ok(ToolResult::error(format!("{}", e)));
        }

        // Security: Basic command validation (defence in depth — the
        // permissions check above is the primary gate, this just catches a
        // few well-known footguns even inside trusted projects).
        let dangerous_patterns = ["rm -rf /", "dd if=", "mkfs", "format", "> /dev/"];
        for pattern in &dangerous_patterns {
            if command.contains(pattern) {
                return Ok(ToolResult::error(format!(
                    "Command blocked for security: contains '{}'",
                    pattern
                )));
            }
        }

        let execution = async {
            let output = TokioCommand::new("sh")
                .arg("-c")
                .arg(command)
                .kill_on_drop(true)
                .output()
                .await
                .context("Failed to execute command")?;

            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();

            let mut result = String::new();
            if !stdout.is_empty() {
                result.push_str(&stdout);
            }
            if !stderr.is_empty() {
                if !result.is_empty() {
                    result.push_str("\nSTDERR:\n");
                }
                result.push_str(&stderr);
            }

            if output.status.success() {
                Ok(ToolResult::success(result))
            } else {
                Ok(ToolResult::error(format!(
                    "Command failed with exit code {}\n{}",
                    output.status.code().unwrap_or(-1),
                    result
                )))
            }
        };

        match timeout(Duration::from_secs(timeout_secs), execution).await {
            Ok(result) => result,
            Err(_) => Ok(ToolResult::error(format!(
                "Command timed out after {} seconds",
                timeout_secs
            ))),
        }
    }

    fn execute_read_file(&self, args: serde_json::Value) -> Result<ToolResult> {
        let path = args["path"].as_str().context("Missing 'path' parameter")?;

        let content = fs::read_to_string(path).context(format!("Failed to read file: {}", path))?;

        let start_line = match parse_read_file_line_arg(&args, "start_line") {
            Ok(line) => line,
            Err(error) => return Ok(ToolResult::error(error.to_string())),
        };
        let end_line = match parse_read_file_line_arg(&args, "end_line") {
            Ok(line) => line,
            Err(error) => return Ok(ToolResult::error(error.to_string())),
        };

        let result = if start_line.is_some() || end_line.is_some() {
            match read_file_line_range(&content, start_line, end_line) {
                Ok(range) => range,
                Err(message) => return Ok(ToolResult::error(message)),
            }
        } else {
            read_file_with_default_cap(&content)
        };

        Ok(ToolResult::success(result))
    }

    fn execute_list_files(&self, args: serde_json::Value) -> Result<ToolResult> {
        let path = args["path"].as_str().unwrap_or(".");
        let recursive = args["recursive"].as_bool().unwrap_or(false);

        let mut result = String::new();

        if recursive {
            Self::list_files_recursive(Path::new(path), &mut result, 0)?;
        } else {
            self.list_files_single(Path::new(path), &mut result)?;
        }

        Ok(ToolResult::success(result))
    }

    fn list_files_single(&self, path: &Path, result: &mut String) -> Result<()> {
        let entries =
            fs::read_dir(path).context(format!("Failed to read directory: {:?}", path))?;

        for entry in entries {
            let entry = entry?;
            let metadata = entry.metadata()?;
            let file_type = if metadata.is_dir() { "DIR " } else { "FILE" };
            let size = metadata.len();
            let name = entry.file_name().to_string_lossy().to_string();

            result.push_str(&format!("{} {:>10} {}\n", file_type, size, name));
        }

        Ok(())
    }

    fn list_files_recursive(path: &Path, result: &mut String, depth: usize) -> Result<()> {
        let entries =
            fs::read_dir(path).context(format!("Failed to read directory: {:?}", path))?;

        let indent = "  ".repeat(depth);

        for entry in entries {
            let entry = entry?;
            let metadata = entry.metadata()?;
            let name = entry.file_name().to_string_lossy().to_string();

            if metadata.is_dir() {
                result.push_str(&format!("{}📁 {}/\n", indent, name));
                Self::list_files_recursive(&entry.path(), result, depth + 1)?;
            } else {
                let size = metadata.len();
                result.push_str(&format!("{}📄 {} ({} bytes)\n", indent, name, size));
            }
        }

        Ok(())
    }

    fn execute_search_glob(&self, args: serde_json::Value) -> Result<ToolResult> {
        let pattern = args["pattern"]
            .as_str()
            .context("Missing 'pattern' parameter")?;
        let base_path = args["base_path"].as_str().unwrap_or(".");

        // Use glob crate for pattern matching
        let _glob_pattern = format!("{}/{}", base_path, pattern);

        // For now, use basic shell globbing via bash
        let command = format!("find {} -name '{}'", base_path, pattern.replace("**", "*"));

        let output = Command::new("sh")
            .arg("-c")
            .arg(&command)
            .output()
            .context("Failed to execute glob search")?;

        let result = String::from_utf8_lossy(&output.stdout).to_string();

        Ok(ToolResult::success(if result.is_empty() {
            format!("No files found matching pattern: {}", pattern)
        } else {
            result
        }))
    }

    fn execute_search_tools(&self, args: serde_json::Value) -> Result<ToolResult> {
        let query = args["query"].as_str().context("Missing 'query' field")?;
        let needle = query.to_ascii_lowercase();
        let mut matches = Vec::new();
        for tool in &self.tools {
            let haystack = format!("{} {}", tool.name, tool.description).to_ascii_lowercase();
            if haystack.contains(&needle) {
                matches.push(format!("- {}: {}", tool.name, tool.description));
            }
        }
        if matches.is_empty() {
            Ok(ToolResult::success(format!("No tools matched '{}'", query)))
        } else {
            Ok(ToolResult::success(matches.join("\n")))
        }
    }

    fn execute_grep(&self, args: serde_json::Value) -> Result<ToolResult> {
        let pattern = args["pattern"]
            .as_str()
            .context("Missing 'pattern' parameter")?;
        let path = args["path"].as_str().context("Missing 'path' parameter")?;
        let recursive = args["recursive"].as_bool().unwrap_or(false);
        let ignore_case = args["ignore_case"].as_bool().unwrap_or(false);

        let mut cmd_args = vec!["grep"];

        if ignore_case {
            cmd_args.push("-i");
        }
        if recursive {
            cmd_args.push("-r");
        }
        cmd_args.push("-n"); // Show line numbers
        cmd_args.push(pattern);
        cmd_args.push(path);

        let output = Command::new("grep")
            .args(&cmd_args)
            .output()
            .context("Failed to execute grep")?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if !stderr.is_empty() {
            return Ok(ToolResult::error(stderr));
        }

        Ok(ToolResult::success(if stdout.is_empty() {
            format!("No matches found for pattern: {}", pattern)
        } else {
            stdout
        }))
    }

    fn execute_edit_file(&self, args: serde_json::Value) -> Result<ToolResult> {
        let path = args["path"].as_str().context("Missing 'path' parameter")?;
        let old_text = args["old_text"]
            .as_str()
            .context("Missing 'old_text' parameter")?;
        let new_text = args["new_text"]
            .as_str()
            .context("Missing 'new_text' parameter")?;

        if self.plan_mode.load(Ordering::SeqCst) {
            return Ok(ToolResult::error(Self::plan_mode_refusal("edit_file")));
        }

        if let Err(e) = self
            .permissions
            .lock()
            .unwrap()
            .check_write(Path::new(path))
        {
            return Ok(ToolResult::error(format!("{}", e)));
        }

        let content = fs::read_to_string(path).context(format!("Failed to read file: {}", path))?;

        if !content.contains(old_text) {
            return Ok(ToolResult::error(
                "Old text not found in file. Text must match exactly.".to_string(),
            ));
        }

        let new_content = content.replace(old_text, new_text);

        fs::write(path, new_content).context(format!("Failed to write file: {}", path))?;
        // Capture the pre-image only after a successful mutation so failed
        // writes don't create phantom rollback entries.
        self.journal
            .record(PathBuf::from(path), Some(content.into_bytes()));

        Ok(ToolResult::success(format!(
            "File edited successfully: {}",
            path
        )))
    }

    fn execute_write_file(&self, args: serde_json::Value) -> Result<ToolResult> {
        let path = args["path"].as_str().context("Missing 'path' parameter")?;
        let content = args["content"]
            .as_str()
            .context("Missing 'content' parameter")?;

        if self.plan_mode.load(Ordering::SeqCst) {
            return Ok(ToolResult::error(Self::plan_mode_refusal("write_file")));
        }

        if let Err(e) = self
            .permissions
            .lock()
            .unwrap()
            .check_write(Path::new(path))
        {
            return Ok(ToolResult::error(format!("{}", e)));
        }

        // Create parent directories if needed
        if let Some(parent) = Path::new(path).parent() {
            fs::create_dir_all(parent).context("Failed to create parent directories")?;
        }

        // Capture the pre-image before writing. Distinguish true "missing"
        // from other read failures so we don't mis-journal unreadable files
        // as if they never existed.
        let previous = match fs::read(path) {
            Ok(bytes) => Some(bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e).context(format!("Failed to read file: {}", path)),
        };

        fs::write(path, content).context(format!("Failed to write file: {}", path))?;
        // Record only after a successful write so rewind tracks real
        // mutations.
        self.journal.record(PathBuf::from(path), previous);

        Ok(ToolResult::success(format!(
            "File written successfully: {} ({} bytes)",
            path,
            content.len()
        )))
    }

    fn execute_think(&self, args: serde_json::Value) -> Result<ToolResult> {
        let thoughts = args["thoughts"]
            .as_str()
            .context("Missing 'thoughts' parameter")?;

        Ok(ToolResult::success(format!(
            "💭 Internal reasoning:\n{}",
            thoughts
        )))
    }

    // ---- Worktree tool ----
    //
    // Wraps `git worktree` (list/add/remove). Add auto-trusts the new
    // worktree path so subsequent write/exec tool calls there don't fail
    // the permissions check. Remove does *not* auto-revoke trust, on the
    // theory that you might still want to write into the original cwd.

    fn worktree_tool() -> BuiltinTool {
        BuiltinTool {
            name: "worktree".to_string(),
            description: "Manage git worktrees. Subcommands: 'list' shows all worktrees; \
                 'add' creates a new worktree at the given path (optionally on a \
                 named branch) and auto-trusts it for write/exec tools; \
                 'remove' deletes a worktree by path."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "add", "remove"],
                        "description": "Operation to perform"
                    },
                    "path": {
                        "type": "string",
                        "description": "Worktree path (required for add/remove)"
                    },
                    "branch": {
                        "type": "string",
                        "description": "Branch name for `add` (optional)"
                    }
                },
                "required": ["action"]
            }),
        }
    }

    fn execute_worktree(&self, args: serde_json::Value) -> Result<ToolResult> {
        let action = args["action"]
            .as_str()
            .context("Missing 'action' parameter")?;

        if matches!(action, "add" | "remove") && self.plan_mode.load(Ordering::SeqCst) {
            return Ok(ToolResult::error(Self::plan_mode_refusal("worktree")));
        }

        // Mutating worktree operations need a trusted cwd, same as `bash`.
        if matches!(action, "add" | "remove") {
            let cwd = std::env::current_dir().context("Could not read cwd")?;
            if let Err(e) = self.permissions.lock().unwrap().check_exec(&cwd) {
                return Ok(ToolResult::error(format!("{}", e)));
            }
        }

        match action {
            "list" => {
                let out = Command::new("git")
                    .args(["worktree", "list", "--porcelain"])
                    .output()
                    .context("Failed to run git worktree list")?;
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                if out.status.success() {
                    Ok(ToolResult::success(if stdout.trim().is_empty() {
                        "(no worktrees)".to_string()
                    } else {
                        stdout
                    }))
                } else {
                    Ok(ToolResult::error(format!(
                        "git worktree list failed: {stderr}"
                    )))
                }
            }
            "add" => {
                let path = args["path"]
                    .as_str()
                    .context("Missing 'path' parameter for `add`")?;
                let mut cmd = Command::new("git");
                cmd.args(["worktree", "add", path]);
                if let Some(branch) = args["branch"].as_str() {
                    cmd.arg(branch);
                }
                let out = cmd.output().context("Failed to run git worktree add")?;
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                if !out.status.success() {
                    return Ok(ToolResult::error(format!(
                        "git worktree add failed: {}{}",
                        stdout.trim(),
                        stderr.trim()
                    )));
                }
                // Auto-trust the new path so the model can immediately edit
                // files / run commands there without a separate /trust step.
                let trusted_msg = match self.permissions.lock().unwrap().trust_dir(Path::new(path))
                {
                    Ok(true) => {
                        // Persist the new trust entry so the approval
                        // survives a CLI restart.
                        if let Err(e) = self.permissions.lock().unwrap().save() {
                            format!(" (auto-trusted in-memory but failed to persist: {e})")
                        } else {
                            " (auto-trusted)".to_string()
                        }
                    }
                    Ok(false) => " (already trusted)".to_string(),
                    Err(e) => format!(" (could not auto-trust: {e})"),
                };
                Ok(ToolResult::success(format!(
                    "Worktree created at {path}{trusted_msg}\n{}{}",
                    stdout.trim(),
                    if stderr.trim().is_empty() {
                        String::new()
                    } else {
                        format!("\n{}", stderr.trim())
                    }
                )))
            }
            "remove" => {
                let path = args["path"]
                    .as_str()
                    .context("Missing 'path' parameter for `remove`")?;
                let out = Command::new("git")
                    .args(["worktree", "remove", path])
                    .output()
                    .context("Failed to run git worktree remove")?;
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                if out.status.success() {
                    Ok(ToolResult::success(format!(
                        "Worktree removed: {path}\n{}",
                        stdout.trim()
                    )))
                } else {
                    Ok(ToolResult::error(format!(
                        "git worktree remove failed: {}{}",
                        stdout.trim(),
                        stderr.trim()
                    )))
                }
            }
            other => Ok(ToolResult::error(format!(
                "Unknown worktree action '{other}'. Use list, add, or remove."
            ))),
        }
    }

    // ---- Web tools ----
    //
    // `web_fetch` is a permission-gated HTTP GET capped at 64 KB; `web_search`
    // is a no-API-key DuckDuckGo lite-mode scrape, also capped. Both refuse
    // in plan mode (network egress is observable behavior) and depend on a
    // trusted cwd as a coarse "is this an approved project" check.

    fn web_fetch_tool() -> BuiltinTool {
        BuiltinTool {
            name: "web_fetch".to_string(),
            description: "Fetch a URL (HTTP GET only) and return the response body, capped at 64 KB. \
                          Strips HTML tags to plain text when content-type is HTML, otherwise returns \
                          the raw body. Refused in plan mode."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Absolute URL, must be http:// or https://"
                    }
                },
                "required": ["url"]
            }),
        }
    }

    async fn execute_web_fetch(&self, args: serde_json::Value) -> Result<ToolResult> {
        let url = args["url"].as_str().context("Missing 'url' parameter")?;
        if let Some(err) = self.network_preflight("web_fetch", url) {
            return Ok(err);
        }
        match http_get_text(url, MAX_WEB_BYTES).await {
            Ok(text) => Ok(ToolResult::success(text)),
            Err(e) => Ok(ToolResult::error(format!("web_fetch failed: {e}"))),
        }
    }

    fn web_search_tool() -> BuiltinTool {
        BuiltinTool {
            name: "web_search".to_string(),
            description:
                "Web search via DuckDuckGo (no API key). Returns the top results as a plain-text \
                          list of `title — snippet — url` lines. Refused in plan mode."
                    .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text search query"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn execute_web_search(&self, args: serde_json::Value) -> Result<ToolResult> {
        let query = args["query"]
            .as_str()
            .context("Missing 'query' parameter")?;
        if query.trim().is_empty() {
            return Ok(ToolResult::error("query must be non-empty".to_string()));
        }
        // Use the URL as the preflight target so the user can see what
        // host the tool is about to hit. The query gets percent-encoded
        // because DuckDuckGo expects standard form encoding.
        let encoded = percent_encode_query(query);
        let url = format!("https://lite.duckduckgo.com/lite/?q={encoded}");
        if let Some(err) = self.network_preflight("web_search", &url) {
            return Ok(err);
        }
        match http_get_text(&url, MAX_WEB_BYTES).await {
            Ok(html) => {
                let results = parse_ddg_lite_results(&html);
                if results.is_empty() {
                    Ok(ToolResult::success(
                        "No results extracted (DuckDuckGo may have throttled or rate-limited the request)."
                            .to_string(),
                    ))
                } else {
                    Ok(ToolResult::success(results.join("\n")))
                }
            }
            Err(e) => Ok(ToolResult::error(format!("web_search failed: {e}"))),
        }
    }

    /// Shared safety preflight for the network tools. Returns `Some(error)`
    /// if the call should be refused; `None` if it may proceed.
    fn network_preflight(&self, tool: &str, url: &str) -> Option<ToolResult> {
        if self.plan_mode.load(Ordering::SeqCst) {
            return Some(ToolResult::error(Self::plan_mode_refusal(tool)));
        }
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Some(ToolResult::error(format!(
                "Refusing `{tool}`: only http(s) URLs are allowed."
            )));
        }
        // Network egress is gated by cwd trust as a coarse "is this an
        // approved project" check — same model as `bash`. Lets the user
        // keep the network off in an untrusted directory.
        let cwd = match std::env::current_dir() {
            Ok(c) => c,
            Err(e) => {
                return Some(ToolResult::error(format!("Could not read cwd: {e}")));
            }
        };
        if let Err(e) = self.permissions.lock().unwrap().check_exec(&cwd) {
            return Some(ToolResult::error(format!("{e}")));
        }
        None
    }

    // ---- REPL tool (long-lived bash session) ----
    //
    // Each session is keyed by a caller-supplied id (the model picks a
    // stable string and reuses it across `repl_eval` calls). Output is
    // delimited by a per-session sentinel echoed after every eval so we
    // can tell where one command's output ends. Inherits the same
    // plan-mode + cwd-trust gate as `bash` — a REPL is just a long-lived
    // shell session.

    fn repl_start_tool() -> BuiltinTool {
        BuiltinTool {
            name: "repl_start".to_string(),
            description: "Start a long-lived bash REPL session. State (cwd, env vars, shell \
                          functions, background processes) persists across `repl_eval` calls \
                          using the same `session_id`. Refused in plan mode."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Caller-chosen stable id. Reuse the same id on subsequent `repl_eval` calls."
                    }
                },
                "required": ["session_id"]
            }),
        }
    }

    fn repl_eval_tool() -> BuiltinTool {
        BuiltinTool {
            name: "repl_eval".to_string(),
            description: "Run shell code in an existing REPL session and return its captured \
                          stdout+stderr plus the exit code of the last command. Multi-line \
                          input is supported. Refused in plan mode."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Id from a previous `repl_start` call"
                    },
                    "code": {
                        "type": "string",
                        "description": "Shell code to run"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Per-eval timeout in seconds (default 30)",
                        "default": 30
                    }
                },
                "required": ["session_id", "code"]
            }),
        }
    }

    fn repl_close_tool() -> BuiltinTool {
        BuiltinTool {
            name: "repl_close".to_string(),
            description: "Terminate a REPL session and release its resources.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": { "type": "string" }
                },
                "required": ["session_id"]
            }),
        }
    }

    async fn execute_repl_start(&self, args: serde_json::Value) -> Result<ToolResult> {
        let session_id = args["session_id"]
            .as_str()
            .context("Missing 'session_id' parameter")?
            .to_string();
        if let Some(err) = self.repl_preflight("repl_start") {
            return Ok(err);
        }

        let mut sessions = self.repls.lock().await;
        if sessions.contains_key(&session_id) {
            return Ok(ToolResult::error(format!(
                "REPL session '{session_id}' already exists. Use a different id or call `repl_close` first."
            )));
        }

        // Sentinel includes a random suffix so a model echoing the literal
        // string can't trick us into ending early. UUIDs are cheap and we
        // already depend on the crate.
        let sentinel = format!("__CUBI_REPL_DONE_{}__", uuid::Uuid::new_v4().simple());

        let mut child = TokioCommand::new("bash")
            .arg("--noprofile")
            .arg("--norc")
            // Intentionally NOT `-i`: interactive mode pulls in job control
            // and the SIGTTIN dance, which makes multiple parallel sessions
            // (e.g. inside tests) deadlock on the controlling terminal.
            // Non-interactive bash still happily accepts commands over a
            // stdin pipe and keeps its environment / cwd / functions
            // between reads, which is all the REPL feature needs.
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // We pipe stderr separately and immediately redirect it to
            // stdout *inside the shell* via `exec 2>&1` (below). That way
            // every subsequent command — including compilation errors,
            // `set -x` traces, etc. — lands on the same FD we read from
            // and is captured in the eval's output.
            .stderr(Stdio::piped())
            .spawn()
            .context("Failed to spawn bash for REPL")?;

        let mut stdin = child
            .stdin
            .take()
            .context("bash spawned without stdin pipe")?;
        let stdout = child
            .stdout
            .take()
            .context("bash spawned without stdout pipe")?;
        // Bash, once we send this, will dup stderr onto stdout for itself
        // and every child it spawns. Done as the very first input so it
        // applies to the user's first eval, too.
        stdin
            .write_all(b"exec 2>&1\n")
            .await
            .context("Failed to merge REPL stderr into stdout")?;
        stdin
            .flush()
            .await
            .context("Failed to flush REPL stderr merge")?;

        // Anything bash wrote to its own stderr *before* `exec 2>&1` took
        // effect (e.g. interactive-shell warnings on some hosts) still
        // comes out the original stderr pipe. Drain it on a background
        // task so the pipe doesn't fill up and stall the child.
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(async move {
                let mut reader = BufReader::new(stderr);
                let mut buf = String::new();
                while reader.read_line(&mut buf).await.unwrap_or(0) > 0 {
                    buf.clear();
                }
            });
        }

        let reader = BufReader::new(stdout);
        sessions.insert(
            session_id.clone(),
            ReplSession {
                stdin,
                reader,
                sentinel,
                _child: child,
            },
        );

        Ok(ToolResult::success(format!(
            "REPL session '{session_id}' started (bash)."
        )))
    }

    async fn execute_repl_eval(&self, args: serde_json::Value) -> Result<ToolResult> {
        let session_id = args["session_id"]
            .as_str()
            .context("Missing 'session_id' parameter")?
            .to_string();
        let code = args["code"].as_str().context("Missing 'code' parameter")?;
        let timeout_secs = args["timeout"].as_u64().unwrap_or(30);

        if let Some(err) = self.repl_preflight("repl_eval") {
            return Ok(err);
        }

        let mut sessions = self.repls.lock().await;
        let session = match sessions.get_mut(&session_id) {
            Some(s) => s,
            None => {
                return Ok(ToolResult::error(format!(
                    "No REPL session '{session_id}'. Call `repl_start` first."
                )));
            }
        };

        // Write the code, then a sentinel echo that includes the exit
        // status of the *last* command (`$?` after the user code). The
        // sentinel is printed on its own line so line-based matching is
        // unambiguous.
        let payload = format!(
            "{code}\nprintf '\\n{sentinel}:%s\\n' \"$?\"\n",
            code = code,
            sentinel = session.sentinel
        );
        if let Err(e) = session.stdin.write_all(payload.as_bytes()).await {
            return Ok(ToolResult::error(format!(
                "Failed to send to REPL '{session_id}': {e}"
            )));
        }
        if let Err(e) = session.stdin.flush().await {
            return Ok(ToolResult::error(format!(
                "Failed to flush REPL '{session_id}': {e}"
            )));
        }

        // Drain stdout until we see the sentinel marker (or hit the
        // timeout). We strip the sentinel line out of what we return to
        // the model so it never sees the marker.
        let needle = format!("{}:", session.sentinel);
        let read = async {
            let mut out = String::new();
            let mut buf = String::new();
            loop {
                buf.clear();
                match session.reader.read_line(&mut buf).await {
                    Ok(0) => break Err(anyhow::anyhow!("REPL stdout closed")),
                    Ok(_) => {
                        if let Some(rest) = buf.trim_end().strip_prefix(&needle) {
                            let exit_code = rest.parse::<i32>().unwrap_or(0);
                            break Ok((out, exit_code));
                        }
                        out.push_str(&buf);
                    }
                    Err(e) => break Err(anyhow::anyhow!("REPL read error: {e}")),
                }
            }
        };

        match timeout(Duration::from_secs(timeout_secs), read).await {
            Ok(Ok((out, exit_code))) => {
                let mut text = if out.is_empty() {
                    String::new()
                } else {
                    out.trim_end().to_string()
                };
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(&format!("[exit {exit_code}]"));
                if exit_code == 0 {
                    Ok(ToolResult::success(text))
                } else {
                    Ok(ToolResult::error(text))
                }
            }
            Ok(Err(e)) => Ok(ToolResult::error(format!("REPL error: {e}"))),
            Err(_) => Ok(ToolResult::error(format!(
                "REPL eval timed out after {timeout_secs}s. The session may now be in an \
                 inconsistent state; call `repl_close` and start a fresh session."
            ))),
        }
    }

    async fn execute_repl_close(&self, args: serde_json::Value) -> Result<ToolResult> {
        let session_id = args["session_id"]
            .as_str()
            .context("Missing 'session_id' parameter")?
            .to_string();
        let mut sessions = self.repls.lock().await;
        if let Some(mut session) = sessions.remove(&session_id) {
            // Best-effort exit: send `exit` to bash so it cleans up its
            // own background jobs, then let the child be dropped (which
            // closes the pipes and reaps the process).
            let _ = session.stdin.write_all(b"exit\n").await;
            let _ = session.stdin.flush().await;
            Ok(ToolResult::success(format!(
                "REPL session '{session_id}' closed."
            )))
        } else {
            Ok(ToolResult::error(format!(
                "No REPL session '{session_id}'."
            )))
        }
    }

    /// Shared plan-mode + cwd-trust check for the REPL tools.
    fn repl_preflight(&self, tool: &str) -> Option<ToolResult> {
        if self.plan_mode.load(Ordering::SeqCst) {
            return Some(ToolResult::error(Self::plan_mode_refusal(tool)));
        }
        let cwd = match std::env::current_dir() {
            Ok(c) => c,
            Err(e) => return Some(ToolResult::error(format!("Could not read cwd: {e}"))),
        };
        if let Err(e) = self.permissions.lock().unwrap().check_exec(&cwd) {
            return Some(ToolResult::error(format!("{e}")));
        }
        None
    }

    // ---- Notebook tool (.ipynb cell-level edits) ----
    //
    // Pure JSON manipulation — no Jupyter dependency. Supports list / read
    // / insert / replace / delete on cell indices. Write actions go through
    // the same plan-mode + path-trust gate as `write_file` since they
    // mutate the disk.

    fn notebook_tool() -> BuiltinTool {
        BuiltinTool {
            name: "notebook".to_string(),
            description: "Cell-level edits to Jupyter notebooks (.ipynb). \
                          Actions: 'list' (shows index/type/preview), 'read' (one cell), \
                          'insert' (new cell at index), 'replace' (overwrite source), \
                          'delete' (remove cell). Indices are 0-based. \
                          Write actions are plan-mode-aware and path-trust gated."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "read", "insert", "replace", "delete"]
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to the .ipynb file"
                    },
                    "cell_index": {
                        "type": "integer",
                        "description": "0-based cell index (required for read/insert/replace/delete; for insert, the cell is inserted *at* this index)"
                    },
                    "cell_type": {
                        "type": "string",
                        "enum": ["code", "markdown", "raw"],
                        "description": "Cell type for 'insert' (default 'code')"
                    },
                    "source": {
                        "type": "string",
                        "description": "Cell source for insert/replace"
                    }
                },
                "required": ["action", "path"]
            }),
        }
    }

    fn execute_notebook(&self, args: serde_json::Value) -> Result<ToolResult> {
        let action = args["action"]
            .as_str()
            .context("Missing 'action' parameter")?;
        let path = args["path"].as_str().context("Missing 'path' parameter")?;

        let is_write = matches!(action, "insert" | "replace" | "delete");
        if is_write && self.plan_mode.load(Ordering::SeqCst) {
            return Ok(ToolResult::error(Self::plan_mode_refusal("notebook")));
        }
        if is_write {
            if let Err(e) = self
                .permissions
                .lock()
                .unwrap()
                .check_write(Path::new(path))
            {
                return Ok(ToolResult::error(format!("{}", e)));
            }
        }

        let raw = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => return Ok(ToolResult::error(format!("Failed to read {path}: {e}"))),
        };
        let mut nb: serde_json::Value = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(e) => {
                return Ok(ToolResult::error(format!(
                    "Failed to parse {path} as JSON notebook: {e}"
                )));
            }
        };

        // Ensure top-level shape matches nbformat v4. Older v3 notebooks
        // are out of scope; we surface a clear error rather than silently
        // corrupting them.
        if nb["cells"].as_array().is_none() {
            return Ok(ToolResult::error(format!(
                "{path} is not an nbformat-4 notebook (no top-level `cells` array)"
            )));
        }

        let result = match action {
            "list" => notebook_list(&nb),
            "read" => {
                let idx = args["cell_index"]
                    .as_u64()
                    .context("Missing 'cell_index' parameter for `read`")?
                    as usize;
                notebook_read(&nb, idx)
            }
            "insert" => {
                let idx = args["cell_index"]
                    .as_u64()
                    .context("Missing 'cell_index' parameter for `insert`")?
                    as usize;
                let cell_type = args["cell_type"].as_str().unwrap_or("code").to_string();
                let source = args["source"]
                    .as_str()
                    .context("Missing 'source' parameter for `insert`")?
                    .to_string();
                let r = notebook_insert(&mut nb, idx, &cell_type, &source);
                if r.is_ok() {
                    notebook_save(&nb, path)?;
                }
                r
            }
            "replace" => {
                let idx = args["cell_index"]
                    .as_u64()
                    .context("Missing 'cell_index' parameter for `replace`")?
                    as usize;
                let source = args["source"]
                    .as_str()
                    .context("Missing 'source' parameter for `replace`")?
                    .to_string();
                let r = notebook_replace(&mut nb, idx, &source);
                if r.is_ok() {
                    notebook_save(&nb, path)?;
                }
                r
            }
            "delete" => {
                let idx = args["cell_index"]
                    .as_u64()
                    .context("Missing 'cell_index' parameter for `delete`")?
                    as usize;
                let r = notebook_delete(&mut nb, idx);
                if r.is_ok() {
                    notebook_save(&nb, path)?;
                }
                r
            }
            other => Err(anyhow::anyhow!("Unknown notebook action '{other}'")),
        };

        match result {
            Ok(text) => Ok(ToolResult::success(text)),
            Err(e) => Ok(ToolResult::error(format!("{e}"))),
        }
    }

    // ---- LSP tool ----
    //
    // Spawns a fresh LSP server per query (rust-analyzer, pyright,
    // typescript-language-server, etc.) and runs hover / definition /
    // references against a 1-based line+column. Stateless: simpler to
    // reason about, more predictable for a tool the model drives. See
    // `lsp_client.rs` for the wire-protocol details.

    fn lsp_tool() -> BuiltinTool {
        BuiltinTool {
            name: "lsp".to_string(),
            description: "Run a one-shot LSP query (hover / definition / references) against \
                          a file using an external language server. You specify the server \
                          command (e.g. 'rust-analyzer', 'pyright-langserver --stdio') so this \
                          works for any language as long as the binary is on PATH. \
                          Read-only — no plan-mode gate."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["hover", "definition", "references"]
                    },
                    "server": {
                        "type": "string",
                        "description": "Command to launch the LSP server, e.g. 'rust-analyzer'. Arguments may be included space-separated, e.g. 'pyright-langserver --stdio'."
                    },
                    "file": {
                        "type": "string",
                        "description": "Path to the file to query"
                    },
                    "line": {
                        "type": "integer",
                        "description": "1-based line number (as shown in editors)"
                    },
                    "character": {
                        "type": "integer",
                        "description": "1-based column",
                        "default": 1
                    },
                    "workspace_root": {
                        "type": "string",
                        "description": "Workspace root passed to the LSP server (default: current directory)"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Overall timeout in seconds (default 30)",
                        "default": 30
                    }
                },
                "required": ["action", "server", "file", "line"]
            }),
        }
    }

    async fn execute_lsp(&self, args: serde_json::Value) -> Result<ToolResult> {
        let action_str = args["action"]
            .as_str()
            .context("Missing 'action' parameter")?;
        let action = match crate::lsp_client::LspAction::from_str(action_str) {
            Some(a) => a,
            None => {
                return Ok(ToolResult::error(format!(
                    "Unknown LSP action '{action_str}'. Use hover, definition, or references."
                )));
            }
        };
        let server_raw = args["server"]
            .as_str()
            .context("Missing 'server' parameter")?;
        let file = args["file"].as_str().context("Missing 'file' parameter")?;
        let line = args["line"].as_u64().context("Missing 'line' parameter")? as u32;
        let character = args["character"].as_u64().unwrap_or(1) as u32;
        let timeout_secs = args["timeout"].as_u64().unwrap_or(30);

        // Read-only tool, but we still gate on cwd trust so a model
        // running in an untrusted project can't spawn arbitrary binaries.
        let cwd = std::env::current_dir().context("Could not read cwd")?;
        if let Err(e) = self.permissions.lock().unwrap().check_exec(&cwd) {
            return Ok(ToolResult::error(format!("{e}")));
        }

        // Workspace root defaults to the caller's cwd. The LSP needs
        // *some* root for many features (rust-analyzer especially).
        let workspace_root = match args["workspace_root"].as_str() {
            Some(p) => Path::new(p).to_path_buf(),
            None => cwd.clone(),
        };

        // Split the server command into argv. Simple whitespace split — no
        // shell-style quoting. Good enough for the common case
        // "pyright-langserver --stdio" and avoids pulling shellwords in.
        let mut parts = server_raw.split_whitespace();
        let server = match parts.next() {
            Some(s) => s.to_string(),
            None => return Ok(ToolResult::error("server command must not be empty".into())),
        };
        let server_args: Vec<String> = parts.map(|s| s.to_string()).collect();

        match crate::lsp_client::run_lsp_query(
            &server,
            &server_args,
            &workspace_root,
            Path::new(file),
            line,
            character,
            action,
            timeout_secs,
        )
        .await
        {
            Ok(text) => Ok(ToolResult::success(text)),
            Err(e) => Ok(ToolResult::error(format!("lsp failed: {e}"))),
        }
    }

    // ---- Cross-platform shell tool ----

    fn shell_tool() -> BuiltinTool {
        BuiltinTool {
            name: "shell".to_string(),
            description: "Run a command in the host platform's native shell: bash/sh on Unix, PowerShell (`pwsh` or `powershell`) on Windows. Subject to the same project-trust and plan-mode gates as `bash`.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute, in the syntax of the host shell."
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default 30, max 300)."
                    }
                },
                "required": ["command"]
            }),
        }
    }

    async fn execute_shell(&self, args: serde_json::Value) -> Result<ToolResult> {
        let command = args["command"]
            .as_str()
            .context("Missing 'command' parameter")?;
        let timeout_secs = args["timeout"].as_u64().unwrap_or(30).min(300);

        if self.plan_mode.load(Ordering::SeqCst) {
            return Ok(ToolResult::error(Self::plan_mode_refusal("shell")));
        }

        let cwd = std::env::current_dir().context("Could not read cwd")?;
        if let Err(e) = self.permissions.lock().unwrap().check_exec(&cwd) {
            return Ok(ToolResult::error(format!("{}", e)));
        }

        let dangerous_patterns = ["rm -rf /", "dd if=", "mkfs", "format", "> /dev/"];
        for pattern in &dangerous_patterns {
            if command.contains(pattern) {
                return Ok(ToolResult::error(format!(
                    "Command blocked for security: contains '{}'",
                    pattern
                )));
            }
        }

        let (program, flag) = host_shell();
        let command = command.to_string();
        let flag = flag.to_string();
        let program = program.to_string();
        let execution = async move {
            let child = TokioCommand::new(&program)
                .arg(&flag)
                .arg(&command)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true)
                .spawn()
                .context("Failed to execute command")?;
            let output = child
                .wait_with_output()
                .await
                .context("Failed to execute command")?;

            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let mut result = String::new();
            result.push_str(&stdout);
            if !stderr.is_empty() {
                if !result.is_empty() {
                    result.push_str("\nSTDERR:\n");
                }
                result.push_str(&stderr);
            }
            if output.status.success() {
                Ok(ToolResult::success(result))
            } else {
                Ok(ToolResult::error(format!(
                    "Command failed with exit code {} (shell: {})\n{}",
                    output.status.code().unwrap_or(-1),
                    program,
                    result
                )))
            }
        };

        match timeout(Duration::from_secs(timeout_secs), execution).await {
            Ok(r) => r,
            Err(_) => Ok(ToolResult::error(format!(
                "Command timed out after {} seconds",
                timeout_secs
            ))),
        }
    }

    // ---- Time tools ----

    fn sleep_tool() -> BuiltinTool {
        BuiltinTool {
            name: "sleep".to_string(),
            description: "Pause execution for the given number of seconds (capped at 60). Useful for letting an external process settle before re-reading its state.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "number",
                        "description": "How long to sleep, in seconds. Hard-capped at 60."
                    }
                },
                "required": ["seconds"]
            }),
        }
    }

    async fn execute_sleep(&self, args: serde_json::Value) -> Result<ToolResult> {
        // Accept ints or floats so the model can ask for 0.5s as well as 3s.
        let seconds = args["seconds"]
            .as_f64()
            .or_else(|| args["seconds"].as_u64().map(|n| n as f64))
            .context("Missing 'seconds' parameter")?;
        let Some(capped) = capped_sleep_seconds(seconds) else {
            return Ok(ToolResult::error("'seconds' must be >= 0".to_string()));
        };
        let ms = (capped * 1000.0) as u64;
        tokio::time::sleep(Duration::from_millis(ms)).await;
        Ok(ToolResult::success(format!("slept {capped:.3}s")))
    }

    fn schedule_tool() -> BuiltinTool {
        BuiltinTool {
            name: "schedule".to_string(),
            description: "Manage persistent scheduled triggers stored in `~/.cubi/schedule.json`. Actions: `list`, `add` (with `name`, `when` cron-like string, and `command`), `remove` (by `name`). The CLI itself does not run them — an external runner (cron, systemd timer, launchd) reads the file.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["list", "add", "remove"] },
                    "name": { "type": "string", "description": "Unique entry name (required for add/remove)." },
                    "when": { "type": "string", "description": "Cron-like schedule expression, e.g. '*/5 * * * *'." },
                    "command": { "type": "string", "description": "Shell command to run when the schedule fires." }
                },
                "required": ["action"]
            }),
        }
    }

    fn execute_schedule(&self, args: serde_json::Value) -> Result<ToolResult> {
        let action = args["action"]
            .as_str()
            .context("Missing 'action' parameter")?;
        let path = match schedule_path() {
            Some(p) => p,
            None => {
                return Ok(ToolResult::error(
                    "Could not resolve ~/.cubi/schedule.json".to_string(),
                ));
            }
        };
        let mut entries = read_json_array_or_empty(&path, "schedule entries")?;

        match action {
            "list" => {
                if entries.is_empty() {
                    return Ok(ToolResult::success("(no scheduled entries)".to_string()));
                }
                let mut s = String::new();
                for e in &entries {
                    s.push_str(&format!(
                        "- {} :: {} :: {}\n",
                        e["name"].as_str().unwrap_or("?"),
                        e["when"].as_str().unwrap_or("?"),
                        e["command"].as_str().unwrap_or("?"),
                    ));
                }
                Ok(ToolResult::success(s.trim_end().to_string()))
            }
            "add" => {
                if self.plan_mode.load(Ordering::SeqCst) {
                    return Ok(ToolResult::error(Self::plan_mode_refusal("schedule.add")));
                }
                let name = args["name"]
                    .as_str()
                    .context("'name' is required for add")?
                    .to_string();
                let when = args["when"]
                    .as_str()
                    .context("'when' is required for add")?
                    .to_string();
                let command = args["command"]
                    .as_str()
                    .context("'command' is required for add")?
                    .to_string();
                if !validate_cron_like(&when) {
                    return Ok(ToolResult::error(format!(
                        "Invalid cron expression: '{when}' (expected 5 whitespace-separated fields)"
                    )));
                }
                entries.retain(|e| e["name"].as_str() != Some(&name));
                entries.push(json!({"name": name, "when": when, "command": command}));
                write_json(&path, &entries)?;
                Ok(ToolResult::success(format!(
                    "scheduled entry saved ({} total)",
                    entries.len()
                )))
            }
            "remove" => {
                if self.plan_mode.load(Ordering::SeqCst) {
                    return Ok(ToolResult::error(Self::plan_mode_refusal(
                        "schedule.remove",
                    )));
                }
                let name = args["name"]
                    .as_str()
                    .context("'name' is required for remove")?;
                let before = entries.len();
                entries.retain(|e| e["name"].as_str() != Some(name));
                if entries.len() == before {
                    return Ok(ToolResult::error(format!("no entry named '{name}'")));
                }
                write_json(&path, &entries)?;
                Ok(ToolResult::success(format!("removed '{name}'")))
            }
            other => Ok(ToolResult::error(format!(
                "Unknown schedule action '{other}' (expected list|add|remove)"
            ))),
        }
    }

    // ---- Structured output helpers ----

    fn brief_tool() -> BuiltinTool {
        BuiltinTool {
            name: "brief".to_string(),
            description: "Distill a long piece of text into a short structured brief: title (first non-empty line), bullet points (one per remaining non-empty paragraph), and a one-line summary. Pure text reducer — no model call.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "The text to summarize." },
                    "max_bullets": { "type": "number", "description": "Maximum bullets to keep (default 5)." }
                },
                "required": ["text"]
            }),
        }
    }

    fn execute_brief(&self, args: serde_json::Value) -> Result<ToolResult> {
        let text = args["text"].as_str().context("Missing 'text' parameter")?;
        let max_bullets = args["max_bullets"].as_u64().unwrap_or(5) as usize;
        let brief = build_brief(text, max_bullets);
        Ok(ToolResult::success(brief.to_string()))
    }

    fn synthetic_output_tool() -> BuiltinTool {
        BuiltinTool {
            name: "synthetic_output".to_string(),
            description: "Given a JSON Schema and a free-form context string, returns a JSON object whose fields match the schema's `properties`. String fields are filled with the trimmed context, numeric/boolean fields get type-appropriate defaults, and unknown types fall back to null. Pure deterministic helper — no model call.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "schema": { "type": "object", "description": "JSON Schema with a top-level `properties` map." },
                    "context": { "type": "string", "description": "Free-form text used to fill string fields." }
                },
                "required": ["schema"]
            }),
        }
    }

    fn execute_synthetic_output(&self, args: serde_json::Value) -> Result<ToolResult> {
        let schema = &args["schema"];
        let context = args["context"].as_str().unwrap_or("").trim().to_string();
        let out = synthesize_from_schema(schema, &context);
        Ok(ToolResult::success(serde_json::to_string_pretty(&out)?))
    }

    // ---- Inter-agent messaging ----

    fn send_message_tool() -> BuiltinTool {
        BuiltinTool {
            name: "send_message".to_string(),
            description: "Append a JSON message to another agent's mailbox at `~/.cubi/messages/<recipient>.json`. The recipient reads with `recv_messages`.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "to": { "type": "string", "description": "Recipient agent name (alphanumeric, dash, underscore)." },
                    "body": { "type": "string", "description": "Message body." },
                    "from": { "type": "string", "description": "Optional sender name." }
                },
                "required": ["to", "body"]
            }),
        }
    }

    fn execute_send_message(&self, args: serde_json::Value) -> Result<ToolResult> {
        let to = args["to"].as_str().context("Missing 'to' parameter")?;
        let body = args["body"].as_str().context("Missing 'body' parameter")?;
        let from = args["from"].as_str().unwrap_or("anonymous").to_string();
        if !is_safe_agent_name(to) {
            return Ok(ToolResult::error(
                "'to' must match [A-Za-z0-9_-]+".to_string(),
            ));
        }
        let path = match messages_path(to) {
            Some(p) => p,
            None => {
                return Ok(ToolResult::error(
                    "Could not resolve ~/.cubi/messages".to_string(),
                ));
            }
        };
        let _lock = acquire_file_lock(&path)?;
        let mut msgs = read_json_array_or_empty(&path, "mailbox")?;
        msgs.push(json!({
            "from": from,
            "body": body,
            "ts": unix_timestamp(),
        }));
        write_json(&path, &msgs)?;
        Ok(ToolResult::success(format!(
            "delivered to {to} ({} pending)",
            msgs.len()
        )))
    }

    fn recv_messages_tool() -> BuiltinTool {
        BuiltinTool {
            name: "recv_messages".to_string(),
            description: "Read pending messages for the given recipient from `~/.cubi/messages/<recipient>.json`. With `drain: true`, the mailbox is emptied after reading.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "recipient": { "type": "string", "description": "Recipient agent name." },
                    "drain": { "type": "boolean", "description": "If true, empty the mailbox after reading (default false)." }
                },
                "required": ["recipient"]
            }),
        }
    }

    fn execute_recv_messages(&self, args: serde_json::Value) -> Result<ToolResult> {
        let recipient = args["recipient"]
            .as_str()
            .context("Missing 'recipient' parameter")?;
        let drain = args["drain"].as_bool().unwrap_or(false);
        if !is_safe_agent_name(recipient) {
            return Ok(ToolResult::error(
                "'recipient' must match [A-Za-z0-9_-]+".to_string(),
            ));
        }
        let path = match messages_path(recipient) {
            Some(p) => p,
            None => {
                return Ok(ToolResult::error(
                    "Could not resolve ~/.cubi/messages".to_string(),
                ));
            }
        };
        let _lock = if drain {
            Some(acquire_file_lock(&path)?)
        } else {
            None
        };
        let msgs = read_json_array_or_empty(&path, "mailbox")?;
        if drain && path.exists() {
            // Truncate rather than delete, so concurrent senders see a valid
            // (empty) array instead of falling back to defaults mid-write.
            write_json(&path, &Vec::<serde_json::Value>::new())?;
        }
        Ok(ToolResult::success(serde_json::to_string_pretty(&msgs)?))
    }

    fn remote_trigger_tool() -> BuiltinTool {
        BuiltinTool {
            name: "remote_trigger".to_string(),
            description: "Write a trigger file to `~/.cubi/triggers/<name>.json` containing a payload and timestamp. Other processes poll the directory to fire on the trigger.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Trigger name (alphanumeric, dash, underscore)." },
                    "payload": { "description": "Arbitrary JSON payload to attach to the trigger." }
                },
                "required": ["name"]
            }),
        }
    }

    fn execute_remote_trigger(&self, args: serde_json::Value) -> Result<ToolResult> {
        let name = args["name"].as_str().context("Missing 'name' parameter")?;
        if !is_safe_agent_name(name) {
            return Ok(ToolResult::error(
                "'name' must match [A-Za-z0-9_-]+".to_string(),
            ));
        }
        let payload = args.get("payload").cloned().unwrap_or(json!(null));
        let path = match triggers_path(name) {
            Some(p) => p,
            None => {
                return Ok(ToolResult::error(
                    "Could not resolve ~/.cubi/triggers".to_string(),
                ));
            }
        };
        let value = json!({ "name": name, "payload": payload, "ts": unix_timestamp() });
        write_json(&path, &value)?;
        Ok(ToolResult::success(format!(
            "trigger '{name}' written to {}",
            path.display()
        )))
    }

    // ---- OS notification ----

    fn notify_tool() -> BuiltinTool {
        BuiltinTool {
            name: "notify".to_string(),
            description: "Send a desktop notification via the host OS: `osascript` on macOS, `notify-send` on Linux, PowerShell balloon on Windows. Silently degrades to a no-op message if the underlying tool is unavailable.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "title": { "type": "string", "description": "Notification title." },
                    "message": { "type": "string", "description": "Notification body." }
                },
                "required": ["message"]
            }),
        }
    }

    fn execute_notify(&self, args: serde_json::Value) -> Result<ToolResult> {
        let title = args["title"].as_str().unwrap_or("cubi").to_string();
        let message = args["message"]
            .as_str()
            .context("Missing 'message' parameter")?
            .to_string();

        match send_os_notification(&title, &message) {
            Ok(s) => Ok(ToolResult::success(s)),
            Err(e) => Ok(ToolResult::error(format!("notify failed: {e}"))),
        }
    }

    // ---- prevent_sleep ----
    //
    // Roadmap C#8 (sleep prevention): spawns the platform's
    // sleep-inhibiting helper for the requested duration (default 5
    // minutes, hard-capped at 4 hours so a misbehaving agent can't
    // pin a laptop awake forever). The helper is detached and the
    // tool returns immediately — the model only needs to know "the
    // request was accepted" before kicking off a long-running task.

    fn prevent_sleep_tool() -> BuiltinTool {
        BuiltinTool {
            name: "prevent_sleep".to_string(),
            description: "Prevent the host from sleeping for the given number of seconds. \
                 Uses `caffeinate -dimsu` on macOS, `systemd-inhibit` on Linux, and a \
                 SetThreadExecutionState-equivalent PowerShell loop on Windows. \
                 Capped at 4 hours; the inhibitor is detached and will exit on its own."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "number",
                        "description": "How long to keep the host awake. Defaults to 300 (5 minutes), hard-capped at 14400 (4 hours)."
                    }
                }
            }),
        }
    }

    async fn execute_prevent_sleep(&self, args: serde_json::Value) -> Result<ToolResult> {
        let seconds = args["seconds"]
            .as_f64()
            .or_else(|| args["seconds"].as_u64().map(|n| n as f64))
            .unwrap_or(300.0);
        if !seconds.is_finite() || seconds < 0.0 {
            return Ok(ToolResult::error(
                "'seconds' must be a non-negative number".to_string(),
            ));
        }
        // Hard-cap at 4 hours.
        let capped = seconds.min(14_400.0) as u64;
        if capped == 0 {
            return Ok(ToolResult::success(
                "prevent_sleep: nothing to do (seconds=0)".to_string(),
            ));
        }

        // Pick the right inhibitor for this host. All branches return a
        // ready-to-spawn `Command` so the spawn-and-detach logic stays
        // unified below.
        #[cfg(target_os = "macos")]
        let mut cmd = {
            let mut c = Command::new("caffeinate");
            // -d display, -i system idle, -m disk, -s system sleep, -u user
            c.args(["-dimsu", "-t", &capped.to_string()]);
            c
        };
        #[cfg(target_os = "linux")]
        let mut cmd = {
            // systemd-inhibit holds the lock for the lifetime of its
            // child command. We use `sleep` as the held command.
            let mut c = Command::new("systemd-inhibit");
            c.args([
                "--what=idle:sleep",
                "--who=cubi",
                "--why=cubi prevent_sleep",
                "--mode=block",
                "sleep",
                &capped.to_string(),
            ]);
            c
        };
        #[cfg(target_os = "windows")]
        let mut cmd = {
            // A tiny PowerShell loop sets ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
            // periodically. Exits on its own after `capped` seconds.
            let script = format!(
                "$sig='[DllImport(\\\"kernel32.dll\\\")] public static extern uint SetThreadExecutionState(uint esFlags);'; \
                 Add-Type -MemberDefinition $sig -Name PWR -Namespace W; \
                 $end = (Get-Date).AddSeconds({secs}); \
                 while ((Get-Date) -lt $end) {{ [W.PWR]::SetThreadExecutionState(0x80000003) | Out-Null; Start-Sleep -Seconds 30 }}; \
                 [W.PWR]::SetThreadExecutionState(0x80000000) | Out-Null",
                secs = capped
            );
            let mut c = Command::new("powershell");
            c.args(["-NoProfile", "-Command", &script]);
            c
        };

        // Detach: ignore stdio + spawn so we return immediately.
        cmd.stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        match cmd.spawn() {
            Ok(_child) => Ok(ToolResult::success(format!(
                "prevent_sleep: host inhibitor started for {capped}s"
            ))),
            Err(e) => Ok(ToolResult::error(format!(
                "prevent_sleep failed to start inhibitor: {e}"
            ))),
        }
    }

    // ---- Headless-browser tools (feature = "browser") ----
    //
    // Tool surface mirrors the REPL family: caller-supplied opaque
    // session id, long-lived per-session browser owned by the
    // registry's `BrowserManager`. All five tools refuse in plan mode
    // — even the read-only ones trigger arbitrary external network and
    // JS execution which is not safe to perform while the user is
    // still reviewing a plan. `browser_screenshot` additionally goes
    // through `Permissions::check_write` because it mutates the disk.

    #[cfg(feature = "browser")]
    fn browser_open_tool() -> BuiltinTool {
        BuiltinTool {
            name: "browser_open".to_string(),
            description: "Open a URL in a headless browser session. Reuses the session if \
                          `session_id` is already open. Optionally waits for a CSS selector \
                          before returning. Refused in plan mode."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Caller-chosen stable id. Reuse on subsequent browser_* calls."
                    },
                    "url": { "type": "string", "description": "Absolute URL to open" },
                    "wait_for": {
                        "type": "string",
                        "description": "Optional CSS selector to wait for before returning"
                    }
                },
                "required": ["session_id", "url"]
            }),
        }
    }

    #[cfg(feature = "browser")]
    fn browser_eval_tool() -> BuiltinTool {
        BuiltinTool {
            name: "browser_eval".to_string(),
            description: "Execute JavaScript in the page and return the JSON-serializable \
                          result. Refused in plan mode."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": { "type": "string" },
                    "js": { "type": "string", "description": "Expression or function source to evaluate" }
                },
                "required": ["session_id", "js"]
            }),
        }
    }

    #[cfg(feature = "browser")]
    fn browser_screenshot_tool() -> BuiltinTool {
        BuiltinTool {
            name: "browser_screenshot".to_string(),
            description: "Save a full-page PNG screenshot of the session to `path`. \
                          Path-trust gated and refused in plan mode."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": { "type": "string" },
                    "path": { "type": "string", "description": "Filesystem path to write the PNG" }
                },
                "required": ["session_id", "path"]
            }),
        }
    }

    #[cfg(feature = "browser")]
    fn browser_text_tool() -> BuiltinTool {
        BuiltinTool {
            name: "browser_text".to_string(),
            description: "Extract visible text from the page (whole page when `selector` is \
                          omitted, otherwise the matched element). Refused in plan mode."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": { "type": "string" },
                    "selector": {
                        "type": "string",
                        "description": "Optional CSS selector; omit for full-page text"
                    }
                },
                "required": ["session_id"]
            }),
        }
    }

    #[cfg(feature = "browser")]
    fn browser_close_tool() -> BuiltinTool {
        BuiltinTool {
            name: "browser_close".to_string(),
            description: "Close a browser session and terminate the underlying Chrome process. \
                          Refused in plan mode."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": { "type": "string" }
                },
                "required": ["session_id"]
            }),
        }
    }

    #[cfg(feature = "browser")]
    async fn execute_browser_open(&self, args: serde_json::Value) -> Result<ToolResult> {
        if self.plan_mode.load(Ordering::SeqCst) {
            return Ok(ToolResult::error(Self::plan_mode_refusal("browser_open")));
        }
        let session_id = args["session_id"]
            .as_str()
            .context("Missing 'session_id' parameter")?;
        let url = args["url"].as_str().context("Missing 'url' parameter")?;
        let wait_for = args["wait_for"].as_str();
        match self.browsers.open(session_id, url, wait_for).await {
            Ok(msg) => Ok(ToolResult::success(msg)),
            Err(e) => Ok(ToolResult::error(format!("browser_open: {e}"))),
        }
    }

    #[cfg(feature = "browser")]
    async fn execute_browser_eval(&self, args: serde_json::Value) -> Result<ToolResult> {
        if self.plan_mode.load(Ordering::SeqCst) {
            return Ok(ToolResult::error(Self::plan_mode_refusal("browser_eval")));
        }
        let session_id = args["session_id"]
            .as_str()
            .context("Missing 'session_id' parameter")?;
        let js = args["js"].as_str().context("Missing 'js' parameter")?;
        match self.browsers.eval(session_id, js).await {
            Ok(value) => Ok(ToolResult::success(value.to_string())),
            Err(e) => Ok(ToolResult::error(format!("browser_eval: {e}"))),
        }
    }

    #[cfg(feature = "browser")]
    async fn execute_browser_screenshot(&self, args: serde_json::Value) -> Result<ToolResult> {
        if self.plan_mode.load(Ordering::SeqCst) {
            return Ok(ToolResult::error(Self::plan_mode_refusal(
                "browser_screenshot",
            )));
        }
        let session_id = args["session_id"]
            .as_str()
            .context("Missing 'session_id' parameter")?;
        let path = args["path"].as_str().context("Missing 'path' parameter")?;
        if let Err(e) = self
            .permissions
            .lock()
            .unwrap()
            .check_write(Path::new(path))
        {
            return Ok(ToolResult::error(format!("{e}")));
        }
        if let Some(parent) = Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(e) = fs::create_dir_all(parent) {
                    return Ok(ToolResult::error(format!(
                        "browser_screenshot: failed to create parent directories: {e}"
                    )));
                }
            }
        }
        match self.browsers.screenshot(session_id, Path::new(path)).await {
            Ok(()) => Ok(ToolResult::success(format!("screenshot saved to {path}"))),
            Err(e) => Ok(ToolResult::error(format!("browser_screenshot: {e}"))),
        }
    }

    #[cfg(feature = "browser")]
    async fn execute_browser_text(&self, args: serde_json::Value) -> Result<ToolResult> {
        if self.plan_mode.load(Ordering::SeqCst) {
            return Ok(ToolResult::error(Self::plan_mode_refusal("browser_text")));
        }
        let session_id = args["session_id"]
            .as_str()
            .context("Missing 'session_id' parameter")?;
        let selector = args["selector"].as_str();
        match self.browsers.text(session_id, selector).await {
            Ok(text) => Ok(ToolResult::success(text)),
            Err(e) => Ok(ToolResult::error(format!("browser_text: {e}"))),
        }
    }

    #[cfg(feature = "browser")]
    async fn execute_browser_close(&self, args: serde_json::Value) -> Result<ToolResult> {
        if self.plan_mode.load(Ordering::SeqCst) {
            return Ok(ToolResult::error(Self::plan_mode_refusal("browser_close")));
        }
        let session_id = args["session_id"]
            .as_str()
            .context("Missing 'session_id' parameter")?;
        match self.browsers.close(session_id).await {
            Ok(()) => Ok(ToolResult::success(format!(
                "browser session '{session_id}' closed"
            ))),
            Err(e) => Ok(ToolResult::error(format!("browser_close: {e}"))),
        }
    }
}

// ---- Helpers for the new tools ----

/// Returns `(program, flag)` suitable for invoking a single shell command on
/// the host OS, preferring PowerShell on Windows and POSIX `sh` elsewhere.
fn host_shell() -> (&'static str, &'static str) {
    if cfg!(windows) {
        if is_program_on_path("pwsh") {
            ("pwsh", "-Command")
        } else {
            ("powershell", "-Command")
        }
    } else {
        ("sh", "-c")
    }
}

fn schedule_path() -> Option<std::path::PathBuf> {
    Some(app_home_dir()?.join(".cubi").join("schedule.json"))
}

fn messages_path(recipient: &str) -> Option<std::path::PathBuf> {
    Some(
        app_home_dir()?
            .join(".cubi")
            .join("messages")
            .join(format!("{recipient}.json")),
    )
}

fn triggers_path(name: &str) -> Option<std::path::PathBuf> {
    Some(
        app_home_dir()?
            .join(".cubi")
            .join("triggers")
            .join(format!("{name}.json")),
    )
}

fn write_json<P: AsRef<Path>, T: Serialize>(path: P, value: &T) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let raw = serde_json::to_string_pretty(value)?;
    let parent = path
        .parent()
        .context(format!("resolve parent for {}", path.display()))?;
    let tmp_name = format!(
        ".{}.tmp-{}-{}",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("write_json"),
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );
    let tmp_path = parent.join(tmp_name);
    fs::write(&tmp_path, raw).with_context(|| format!("write {}", tmp_path.display()))?;
    #[cfg(windows)]
    if path.exists() {
        fs::remove_file(path).with_context(|| format!("remove {}", path.display()))?;
    }
    fs::rename(&tmp_path, path)
        .with_context(|| format!("rename {} -> {}", tmp_path.display(), path.display()))?;
    Ok(())
}

fn app_home_dir() -> Option<PathBuf> {
    #[cfg(test)]
    {
        use std::sync::OnceLock;
        static TEST_HOME: OnceLock<PathBuf> = OnceLock::new();
        Some(
            TEST_HOME
                .get_or_init(|| {
                    let path =
                        std::env::temp_dir().join(format!("cubi-test-home-{}", std::process::id()));
                    let _ = fs::create_dir_all(&path);
                    path
                })
                .clone(),
        )
    }
    #[cfg(not(test))]
    {
        dirs::home_dir()
    }
}

fn read_json_array_or_empty(path: &Path, context: &str) -> Result<Vec<serde_json::Value>> {
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e).with_context(|| format!("read {}", path.display())),
    };
    serde_json::from_str(&raw)
        .with_context(|| format!("parse {context} JSON at {}", path.display()))
}

fn lock_path(path: &Path) -> PathBuf {
    path.with_extension(format!(
        "{}lock",
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| format!("{ext}."))
            .unwrap_or_default()
    ))
}

struct FileLockGuard {
    path: PathBuf,
}

impl Drop for FileLockGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn acquire_file_lock(path: &Path) -> Result<FileLockGuard> {
    let lock = lock_path(path);
    if let Some(parent) = lock.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        match OpenOptions::new().write(true).create_new(true).open(&lock) {
            Ok(_) => return Ok(FileLockGuard { path: lock }),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                if Instant::now() >= deadline {
                    anyhow::bail!("timed out acquiring lock {}", lock.display());
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => return Err(e).with_context(|| format!("lock {}", lock.display())),
        }
    }
}

fn is_program_on_path(program: &str) -> bool {
    let Some(path_var) = std::env::var_os("PATH") else {
        return false;
    };
    for dir in std::env::split_paths(&path_var) {
        #[cfg(windows)]
        {
            for candidate in [program.to_string(), format!("{program}.exe")] {
                if dir.join(candidate).is_file() {
                    return true;
                }
            }
        }
        #[cfg(not(windows))]
        {
            if dir.join(program).is_file() {
                return true;
            }
        }
    }
    false
}

fn is_safe_agent_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() < 64
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn capped_sleep_seconds(seconds: f64) -> Option<f64> {
    if !seconds.is_finite() || seconds < 0.0 {
        None
    } else {
        Some(seconds.min(60.0))
    }
}

/// Validates that `expr` looks like a 5-field cron expression: minute, hour,
/// day-of-month, month, day-of-week. We do *not* fully parse each field —
/// the actual runner is external — but reject obvious shape errors so a
/// typo doesn't silently get persisted.
fn validate_cron_like(expr: &str) -> bool {
    let fields: Vec<&str> = expr.split_whitespace().collect();
    fields.len() == 5 && fields.iter().all(|f| !f.is_empty())
}

/// Produces a structured Markdown brief from `text`. The first non-empty
/// line becomes the title; subsequent non-empty paragraphs (separated by
/// blank lines) become bullets, capped at `max_bullets`. A one-line summary
/// (the first sentence of the title) is appended.
fn build_brief(text: &str, max_bullets: usize) -> serde_json::Value {
    let lines: Vec<&str> = text.lines().map(|l| l.trim()).collect();
    let title = lines
        .iter()
        .find(|l| !l.is_empty())
        .copied()
        .unwrap_or("")
        .to_string();

    // Bullets: gather distinct non-empty lines after the title.
    let mut bullets: Vec<String> = Vec::new();
    let mut seen_title = false;
    for l in &lines {
        if l.is_empty() {
            continue;
        }
        if !seen_title {
            seen_title = true;
            continue;
        }
        bullets.push((*l).to_string());
        if bullets.len() >= max_bullets {
            break;
        }
    }

    // Summary: first sentence of title (or whole title if no period).
    let summary = match title.find('.') {
        Some(p) => title[..p].trim().to_string(),
        None => title.clone(),
    };

    json!({
        "title": title,
        "bullets": bullets,
        "summary": summary,
    })
}

/// Walks a JSON Schema's top-level `properties` map and produces a JSON
/// object with one field per property. Filled values are type-appropriate
/// (strings get `context`, numbers get 0, booleans get false, arrays get
/// `[]`, objects get `{}`, everything else gets null).
fn synthesize_from_schema(schema: &serde_json::Value, context: &str) -> serde_json::Value {
    let Some(props) = schema.get("properties").and_then(|p| p.as_object()) else {
        return json!({});
    };
    let mut out = serde_json::Map::new();
    for (key, spec) in props {
        let ty = spec
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("string");
        let value = match ty {
            "string" => json!(context),
            "integer" | "number" => json!(0),
            "boolean" => json!(false),
            "array" => json!([]),
            "object" => json!({}),
            _ => json!(null),
        };
        out.insert(key.clone(), value);
    }
    serde_json::Value::Object(out)
}

/// Best-effort OS notification. Returns an Ok message describing what was
/// attempted; errors only when the OS notification tool fails to spawn.
fn send_os_notification(title: &str, message: &str) -> Result<String> {
    if cfg!(target_os = "macos") {
        let script = format!(
            "display notification \"{}\" with title \"{}\"",
            escape_applescript(message),
            escape_applescript(title)
        );
        let status = Command::new("osascript")
            .arg("-e")
            .arg(&script)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        match status {
            Ok(s) if s.success() => Ok("notification sent via osascript".to_string()),
            Ok(s) => Ok(format!("osascript exited {}", s.code().unwrap_or(-1))),
            Err(e) => Ok(format!("osascript unavailable: {e}")),
        }
    } else if cfg!(target_os = "linux") {
        let status = Command::new("notify-send")
            .arg(title)
            .arg(message)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        match status {
            Ok(s) if s.success() => Ok("notification sent via notify-send".to_string()),
            Ok(s) => Ok(format!("notify-send exited {}", s.code().unwrap_or(-1))),
            Err(e) => Ok(format!("notify-send unavailable: {e}")),
        }
    } else if cfg!(target_os = "windows") {
        // Use Windows Forms balloon tip via PowerShell. Fully self-contained,
        // no external module required.
        let ps = format!(
            "[reflection.assembly]::loadwithpartialname('System.Windows.Forms') | Out-Null; \
             $n = New-Object System.Windows.Forms.NotifyIcon; \
             $n.Icon = [System.Drawing.SystemIcons]::Information; \
             $n.Visible = $true; \
             $n.BalloonTipTitle = '{}'; $n.BalloonTipText = '{}'; \
             $n.ShowBalloonTip(5000); Start-Sleep -Seconds 1; $n.Dispose()",
            title.replace('\'', "''"),
            message.replace('\'', "''")
        );
        let status = Command::new("powershell")
            .arg("-NoProfile")
            .arg("-Command")
            .arg(&ps)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        match status {
            Ok(s) if s.success() => Ok("notification sent via powershell".to_string()),
            Ok(s) => Ok(format!("powershell exited {}", s.code().unwrap_or(-1))),
            Err(e) => Ok(format!("powershell unavailable: {e}")),
        }
    } else {
        Ok(format!(
            "no native notification backend for this OS ({})",
            std::env::consts::OS
        ))
    }
}

fn escape_applescript(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

// ---- Notebook helpers (free functions for ease of testing) ----

/// Reads a cell's `source` field, which nbformat allows to be either a
/// string or an array of strings. Returns the concatenation.
fn notebook_cell_source(cell: &serde_json::Value) -> String {
    match &cell["source"] {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v.as_str())
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// Lists all cells with a short preview. Used by the `list` action.
fn notebook_list(nb: &serde_json::Value) -> Result<String> {
    let cells = nb["cells"]
        .as_array()
        .context("notebook missing `cells` array")?;
    let mut out = String::new();
    out.push_str(&format!("{} cell(s):\n", cells.len()));
    for (i, cell) in cells.iter().enumerate() {
        let cell_type = cell["cell_type"].as_str().unwrap_or("?");
        let source = notebook_cell_source(cell);
        let preview: String = source
            .lines()
            .next()
            .unwrap_or("")
            .chars()
            .take(80)
            .collect();
        out.push_str(&format!("  [{i}] {cell_type}: {preview}\n"));
    }
    Ok(out)
}

/// Returns the full source of one cell.
fn notebook_read(nb: &serde_json::Value, idx: usize) -> Result<String> {
    let cells = nb["cells"]
        .as_array()
        .context("notebook missing `cells` array")?;
    let cell = cells
        .get(idx)
        .with_context(|| format!("cell index {idx} out of range (have {})", cells.len()))?;
    let cell_type = cell["cell_type"].as_str().unwrap_or("?");
    let source = notebook_cell_source(cell);
    Ok(format!("[{idx}] {cell_type}\n---\n{source}"))
}

/// Builds a fresh cell. We always store `source` as an array of lines (with
/// trailing newlines preserved on all but the last) to match the convention
/// Jupyter itself writes.
fn build_cell(cell_type: &str, source: &str) -> serde_json::Value {
    let lines = split_source_for_nbformat(source);
    match cell_type {
        "code" => json!({
            "cell_type": "code",
            "metadata": {},
            "source": lines,
            "outputs": [],
            "execution_count": serde_json::Value::Null,
        }),
        "markdown" => json!({
            "cell_type": "markdown",
            "metadata": {},
            "source": lines,
        }),
        "raw" => json!({
            "cell_type": "raw",
            "metadata": {},
            "source": lines,
        }),
        _ => json!({
            "cell_type": cell_type,
            "metadata": {},
            "source": lines,
        }),
    }
}

/// Splits a source blob into the per-line array nbformat expects. Each
/// non-final line keeps its trailing `\n` so reassembly is lossless.
fn split_source_for_nbformat(source: &str) -> Vec<String> {
    if source.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut start = 0;
    let bytes = source.as_bytes();
    for (i, b) in bytes.iter().enumerate() {
        if *b == b'\n' {
            out.push(source[start..=i].to_string());
            start = i + 1;
        }
    }
    if start < source.len() {
        out.push(source[start..].to_string());
    }
    out
}

fn notebook_insert(
    nb: &mut serde_json::Value,
    idx: usize,
    cell_type: &str,
    source: &str,
) -> Result<String> {
    let cells = nb["cells"]
        .as_array_mut()
        .context("notebook missing `cells` array")?;
    if idx > cells.len() {
        anyhow::bail!(
            "insert index {idx} out of range (notebook has {} cell(s); valid range 0..={})",
            cells.len(),
            cells.len()
        );
    }
    cells.insert(idx, build_cell(cell_type, source));
    Ok(format!(
        "Inserted {cell_type} cell at index {idx} (notebook now has {} cell(s))",
        cells.len()
    ))
}

fn notebook_replace(nb: &mut serde_json::Value, idx: usize, source: &str) -> Result<String> {
    let cells = nb["cells"]
        .as_array_mut()
        .context("notebook missing `cells` array")?;
    let len = cells.len();
    let cell = cells
        .get_mut(idx)
        .with_context(|| format!("cell index {idx} out of range (have {})", len))?;
    cell["source"] = json!(split_source_for_nbformat(source));
    // Reset output state when replacing a code cell so stale outputs from
    // the prior version of the cell don't mislead the next reader.
    if cell["cell_type"].as_str() == Some("code") {
        cell["outputs"] = json!([]);
        cell["execution_count"] = serde_json::Value::Null;
    }
    Ok(format!("Replaced cell {idx}"))
}

fn notebook_delete(nb: &mut serde_json::Value, idx: usize) -> Result<String> {
    let cells = nb["cells"]
        .as_array_mut()
        .context("notebook missing `cells` array")?;
    if idx >= cells.len() {
        anyhow::bail!(
            "delete index {idx} out of range (notebook has {} cell(s))",
            cells.len()
        );
    }
    cells.remove(idx);
    Ok(format!(
        "Deleted cell {idx} (notebook now has {} cell(s))",
        cells.len()
    ))
}

fn notebook_save(nb: &serde_json::Value, path: &str) -> Result<()> {
    let s = serde_json::to_string_pretty(nb)?;
    fs::write(path, s).with_context(|| format!("Failed to write {path}"))?;
    Ok(())
}

fn parse_read_file_line_arg(args: &serde_json::Value, name: &str) -> Result<Option<usize>> {
    let Some(value) = args.get(name).filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let line = value
        .as_u64()
        .with_context(|| format!("`{name}` must be a positive integer"))?;
    if line == 0 {
        anyhow::bail!("`{name}` must be 1 or greater");
    }
    usize::try_from(line)
        .with_context(|| format!("`{name}` is too large"))
        .map(Some)
}

fn read_file_line_range(
    content: &str,
    start_line: Option<usize>,
    end_line: Option<usize>,
) -> std::result::Result<String, String> {
    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();
    let start = start_line.unwrap_or(1);

    if total_lines == 0 {
        return if start == 1 && end_line.unwrap_or(1) == 1 {
            Ok(String::new())
        } else {
            Err("Requested line range is outside an empty file.".to_string())
        };
    }

    let end = end_line.unwrap_or(total_lines);
    if start > total_lines {
        return Err(format!(
            "Invalid line range: start_line ({start}) is beyond end of file ({total_lines} lines)."
        ));
    }
    if start > end {
        return Err(format!(
            "Invalid line range: start_line ({start}) must be less than or equal to end_line ({end})."
        ));
    }

    let start_idx = start - 1;
    let end_idx = end.min(total_lines);
    Ok(lines[start_idx..end_idx].join("\n"))
}

fn read_file_with_default_cap(content: &str) -> String {
    let total_bytes = content.len();
    let total_lines = content.lines().count();

    if total_bytes <= READ_FILE_DEFAULT_MAX_BYTES && total_lines <= READ_FILE_DEFAULT_MAX_LINES {
        return content.to_string();
    }

    let mut head = String::new();
    let mut shown_lines = 0usize;
    let mut truncated_mid_line = false;

    for line in content.lines().take(READ_FILE_DEFAULT_MAX_LINES) {
        let separator_bytes = usize::from(shown_lines > 0);
        let needed_bytes = separator_bytes + line.len();
        if head.len() + needed_bytes <= READ_FILE_DEFAULT_MAX_BYTES {
            if separator_bytes == 1 {
                head.push('\n');
            }
            head.push_str(line);
            shown_lines += 1;
            continue;
        }

        let remaining = READ_FILE_DEFAULT_MAX_BYTES.saturating_sub(head.len());
        if remaining > separator_bytes {
            if separator_bytes == 1 {
                head.push('\n');
            }
            let prefix = utf8_prefix(line, remaining - separator_bytes);
            head.push_str(prefix);
            shown_lines += usize::from(!prefix.is_empty());
            truncated_mid_line = true;
        }
        break;
    }

    let mut result = head;
    result.push_str(&format!(
        "\n\n[read_file output truncated: showing the first {shown_lines} of {total_lines} lines and {} of {total_bytes} bytes (limit: {READ_FILE_DEFAULT_MAX_LINES} lines or {READ_FILE_DEFAULT_MAX_BYTES} bytes). Re-run read_file with start_line/end_line for the specific range you need; use grep to find relevant line numbers first.]",
        result.len()
    ));
    if truncated_mid_line {
        result.push_str(" The last shown line was cut at the byte limit.");
    }
    result
}

fn utf8_prefix(s: &str, max_bytes: usize) -> &str {
    if max_bytes >= s.len() {
        return s;
    }
    let mut end = max_bytes;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Cap on the response body we'll buffer from a web tool. Keeps a single
/// rogue URL from blowing the model's context window or the process's RAM.
const MAX_WEB_BYTES: usize = 64 * 1024;

/// HTTP GET with a body cap. The response is streamed and we stop
/// reading once we've accumulated `max_bytes`, so an upstream sending a
/// gigabyte file doesn't blow process RAM just because the model asked
/// to look at it. If the response is HTML, the tags are stripped to
/// plain text before being returned.
async fn http_get_text(url: &str, max_bytes: usize) -> Result<String> {
    use futures_util::StreamExt;

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .user_agent("cubi/0.1")
        .build()?;
    let resp = client.get(url).send().await?;
    let status = resp.status();
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    // Stream and stop once we hit the cap. `take(max_bytes + 1)` on the
    // accumulated length lets us detect truncation cheaply (we read one
    // chunk past the cap, then trim).
    let mut buf: Vec<u8> = Vec::with_capacity(max_bytes.min(8 * 1024));
    let mut stream = resp.bytes_stream();
    let mut truncated = false;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let remaining = max_bytes.saturating_sub(buf.len());
        if chunk.len() <= remaining {
            buf.extend_from_slice(&chunk);
        } else {
            buf.extend_from_slice(&chunk[..remaining]);
            truncated = true;
            break;
        }
    }

    let body = String::from_utf8_lossy(&buf).to_string();
    let mut text = if content_type.contains("html") {
        strip_html(&body)
    } else {
        body
    };
    if truncated {
        text.push_str(&format!(
            "\n\n[response truncated at {max_bytes} bytes; full status was {}]",
            status
        ));
    }
    if !status.is_success() {
        // Surface the HTTP status as the first line so the model can react
        // (404 vs 500 vs 403 all imply different next steps).
        text.insert_str(0, &format!("HTTP {}\n\n", status));
    }
    Ok(text)
}

/// Crude HTML → text conversion: drops `<script>` / `<style>` blocks, then
/// strips every remaining tag and collapses whitespace. Good enough for
/// search-result snippets and short articles; not a real HTML parser.
///
/// Iterates by `char` so non-ASCII content (CJK, accents, emoji…) round-trips
/// intact; tag markers and the `<script>` / `<style>` lookahead are still
/// ASCII-only, which is true of all real-world HTML.
fn strip_html(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut in_tag = false;
    let mut skip_until: Option<&'static str> = None;
    let lower = input.to_ascii_lowercase();
    let mut iter = input.char_indices().peekable();
    while let Some((i, ch)) = iter.next() {
        if let Some(end_tag) = skip_until {
            // Look for the closing tag (case-insensitive ASCII) at the
            // current byte offset; advance past it on match.
            if lower[i..].starts_with(end_tag) {
                // Skip past the end tag in the iterator too.
                let mut to_skip = end_tag.len() - ch.len_utf8();
                while to_skip > 0 {
                    if let Some((_, c)) = iter.next() {
                        to_skip = to_skip.saturating_sub(c.len_utf8());
                    } else {
                        break;
                    }
                }
                skip_until = None;
            }
            continue;
        }
        if ch == '<' {
            // Detect <script ...> and <style ...> openings to drop their bodies.
            let rest = &lower[i..];
            if rest.starts_with("<script") {
                skip_until = Some("</script>");
                continue;
            }
            if rest.starts_with("<style") {
                skip_until = Some("</style>");
                continue;
            }
            in_tag = true;
            continue;
        }
        if ch == '>' {
            in_tag = false;
            // Tag boundaries act as whitespace so adjacent words don't
            // glue together when we strip the markup.
            if !out.ends_with(' ') && !out.is_empty() {
                out.push(' ');
            }
            continue;
        }
        if !in_tag {
            out.push(ch);
        }
    }
    // Decode the handful of HTML entities that crop up in DDG output.
    let decoded = out
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ");
    // Collapse runs of whitespace so the output is readable.
    let mut collapsed = String::with_capacity(decoded.len());
    let mut prev_ws = false;
    for ch in decoded.chars() {
        if ch.is_whitespace() {
            if !prev_ws {
                collapsed.push(' ');
                prev_ws = true;
            }
        } else {
            collapsed.push(ch);
            prev_ws = false;
        }
    }
    collapsed.trim().to_string()
}

/// Minimal percent-encoding for a search query: encodes any byte outside
/// the unreserved set per RFC 3986 §2.3. Avoids pulling in a separate
/// `percent-encoding` crate for one call site.
fn percent_encode_query(input: &str) -> String {
    let mut out = String::with_capacity(input.len() * 3);
    for b in input.bytes() {
        let unreserved =
            b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.' || b == b'~';
        if unreserved {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{:02X}", b));
        }
    }
    out
}

/// Pulls a best-effort list of "title — url" pairs out of DuckDuckGo's
/// "lite" search results page. The page is mostly a flat table of `<a>`
/// tags with a small consistent class. We extract every link whose href
/// looks like an outbound result and dedupe.
fn parse_ddg_lite_results(html: &str) -> Vec<String> {
    let mut results = Vec::new();
    let mut seen = std::collections::HashSet::new();
    // Strategy: scan for `<a ... href="..." ...>TITLE</a>` and keep the
    // ones whose href starts with http(s) and isn't a DDG-internal link.
    let mut i = 0;
    while let Some(rel) = html[i..].find("<a ") {
        let start = i + rel;
        let after_a = start + 3;
        // Find href attribute.
        let tag_end = match html[after_a..].find('>') {
            Some(p) => after_a + p,
            None => break,
        };
        let attrs = &html[after_a..tag_end];
        let href = attrs
            .find("href=\"")
            .and_then(|p| {
                let q = p + 6;
                attrs[q..].find('"').map(|e| &attrs[q..q + e])
            })
            .unwrap_or("");
        // Find content between `<a ...>` and `</a>`.
        let content_start = tag_end + 1;
        let content_end = match html[content_start..].find("</a>") {
            Some(p) => content_start + p,
            None => break,
        };
        let title = strip_html(&html[content_start..content_end]);
        // DDG's lite redirects via /l/?kh=...&uddg=<encoded-url>. Try to
        // unwrap that so we surface the real destination.
        let resolved = unwrap_ddg_redirect(href).unwrap_or_else(|| href.to_string());
        if !title.is_empty()
            && (resolved.starts_with("http://") || resolved.starts_with("https://"))
            && !resolved.contains("duckduckgo.com")
            && seen.insert(resolved.clone())
        {
            results.push(format!("{title} — {resolved}"));
        }
        i = content_end + 4;
        // Hard cap so an enormous response doesn't generate thousands of lines.
        if results.len() >= 10 {
            break;
        }
    }
    results
}

/// Unwraps DuckDuckGo lite-mode redirect links of the form
/// `/l/?kh=...&uddg=<percent-encoded-url>` into the destination URL.
/// Returns `None` if the link isn't a DDG redirect (in which case the
/// caller should use the original `href` directly).
fn unwrap_ddg_redirect(href: &str) -> Option<String> {
    let needle = "uddg=";
    let idx = href.find(needle)?;
    let after = &href[idx + needle.len()..];
    // The encoded URL runs until the next `&` or end-of-string.
    let raw_encoded = match after.find('&') {
        Some(end) => &after[..end],
        None => after,
    };
    Some(percent_decode(raw_encoded))
}

/// Minimal percent-decoder. Like `percent_encode_query`, kept in-tree to
/// avoid pulling another crate for a single call site.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = (bytes[i + 1] as char).to_digit(16);
            let lo = (bytes[i + 2] as char).to_digit(16);
            if let (Some(h), Some(l)) = (hi, lo) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_tmp(label: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("cubi-tool-{label}-{nanos}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn unique_project_tmp(label: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::current_dir()
            .unwrap()
            .join("target")
            .join("test-work")
            .join(format!("cubi-tool-{label}-{nanos}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn registry_with_trust(dir: &Path, plan_on: bool) -> BuiltinToolRegistry {
        let mut perms = Permissions::default();
        perms.trust_dir(dir).unwrap();
        BuiltinToolRegistry::new(
            Arc::new(Mutex::new(perms)),
            Arc::new(AtomicBool::new(plan_on)),
        )
    }

    #[tokio::test]
    async fn read_file_returns_full_small_file_without_range() {
        let dir = unique_project_tmp("read-small");
        let path = dir.join("small.txt");
        let content = "one\ntwo\nthree\n";
        fs::write(&path, content).unwrap();
        let registry = registry_with_trust(&dir, false);

        let result = registry
            .execute("read_file", json!({ "path": path.to_str().unwrap() }))
            .await
            .expect("call ok");

        assert!(result.is_error.is_none(), "got {:?}", result);
        assert_eq!(result.content[0].text, content);
        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn read_file_supports_inclusive_and_open_ended_ranges() {
        let dir = unique_project_tmp("read-range");
        let path = dir.join("range.txt");
        fs::write(&path, "one\ntwo\nthree\nfour\n").unwrap();
        let registry = registry_with_trust(&dir, false);

        let middle = registry
            .execute(
                "read_file",
                json!({ "path": path.to_str().unwrap(), "start_line": 2, "end_line": 3 }),
            )
            .await
            .expect("call ok");
        assert_eq!(middle.content[0].text, "two\nthree");

        let from_start = registry
            .execute(
                "read_file",
                json!({ "path": path.to_str().unwrap(), "end_line": 2 }),
            )
            .await
            .expect("call ok");
        assert_eq!(from_start.content[0].text, "one\ntwo");

        let through_end = registry
            .execute(
                "read_file",
                json!({ "path": path.to_str().unwrap(), "start_line": 3 }),
            )
            .await
            .expect("call ok");
        assert_eq!(through_end.content[0].text, "three\nfour");

        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn read_file_invalid_ranges_return_errors_instead_of_panicking() {
        let dir = unique_project_tmp("read-invalid-range");
        let path = dir.join("range.txt");
        fs::write(&path, "one\ntwo\nthree\n").unwrap();
        let registry = registry_with_trust(&dir, false);

        let reversed = registry
            .execute(
                "read_file",
                json!({ "path": path.to_str().unwrap(), "start_line": 3, "end_line": 2 }),
            )
            .await
            .expect("call ok");
        assert_eq!(reversed.is_error, Some(true));
        assert!(reversed.content[0].text.contains("start_line (3)"));

        let past_end = registry
            .execute(
                "read_file",
                json!({ "path": path.to_str().unwrap(), "start_line": 99 }),
            )
            .await
            .expect("call ok");
        assert_eq!(past_end.is_error, Some(true));
        assert!(past_end.content[0].text.contains("beyond end of file"));

        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn read_file_truncates_unbounded_large_files_with_range_guidance() {
        let dir = unique_project_tmp("read-truncate");
        let path = dir.join("large.txt");
        let mut content = String::new();
        for i in 1..=(READ_FILE_DEFAULT_MAX_LINES + 5) {
            content.push_str(&format!("line {i}\n"));
        }
        fs::write(&path, content).unwrap();
        let registry = registry_with_trust(&dir, false);

        let result = registry
            .execute("read_file", json!({ "path": path.to_str().unwrap() }))
            .await
            .expect("call ok");

        assert!(result.is_error.is_none(), "got {:?}", result);
        let text = &result.content[0].text;
        assert!(text.contains("line 1"));
        assert!(text.contains(&format!("line {READ_FILE_DEFAULT_MAX_LINES}")));
        assert!(!text.contains(&format!("line {}", READ_FILE_DEFAULT_MAX_LINES + 1)));
        assert!(text.contains("read_file output truncated"));
        assert!(text.contains("start_line/end_line"));
        assert!(text.contains("grep"));

        fs::remove_dir_all(&dir).ok();
    }

    /// Registry whose trust store also covers the current working
    /// directory. Used by tests for tools whose preflight check inspects
    /// the cwd (REPL, web tools) rather than a specific path argument.
    fn registry_trusting_cwd(plan_on: bool) -> BuiltinToolRegistry {
        let mut perms = Permissions::default();
        perms.trust_dir(&std::env::current_dir().unwrap()).unwrap();
        BuiltinToolRegistry::new(
            Arc::new(Mutex::new(perms)),
            Arc::new(AtomicBool::new(plan_on)),
        )
    }

    #[tokio::test]
    async fn plan_mode_blocks_write_file() {
        let dir = unique_tmp("plan-write");
        let registry = registry_with_trust(&dir, true);
        let path = dir.join("plan.txt");

        let result = registry
            .execute(
                "write_file",
                json!({ "path": path.to_str().unwrap(), "content": "x" }),
            )
            .await
            .expect("call ok");
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("plan mode is ON"));
        assert!(!path.exists(), "plan mode must not create the file");

        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn plan_mode_blocks_edit_file() {
        let dir = unique_tmp("plan-edit");
        let path = dir.join("plan-edit.txt");
        fs::write(&path, "hello").unwrap();
        let registry = registry_with_trust(&dir, true);

        let result = registry
            .execute(
                "edit_file",
                json!({
                    "path": path.to_str().unwrap(),
                    "old_text": "hello",
                    "new_text": "world",
                }),
            )
            .await
            .expect("call ok");
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("plan mode is ON"));
        assert_eq!(fs::read_to_string(&path).unwrap(), "hello");

        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn plan_mode_blocks_bash() {
        let dir = unique_tmp("plan-bash");
        // Trust the dir so the permissions gate would otherwise allow it.
        let registry = registry_with_trust(&dir, true);
        let result = registry
            .execute("bash", json!({ "command": "echo hi", "timeout": 5 }))
            .await
            .expect("call ok");
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("plan mode is ON"));
        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn permissions_block_write_outside_trusted_root() {
        let trusted = unique_tmp("perm-trusted");
        let outside = unique_tmp("perm-outside");
        let registry = registry_with_trust(&trusted, false);

        let target = outside.join("escape.txt");
        let result = registry
            .execute(
                "write_file",
                json!({ "path": target.to_str().unwrap(), "content": "nope" }),
            )
            .await
            .expect("call ok");
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("outside any trusted root"));
        assert!(!target.exists());

        fs::remove_dir_all(&trusted).ok();
        fs::remove_dir_all(&outside).ok();
    }

    #[tokio::test]
    async fn write_file_inside_trusted_root_succeeds() {
        let dir = unique_tmp("perm-ok");
        let registry = registry_with_trust(&dir, false);

        let target = dir.join("ok.txt");
        let result = registry
            .execute(
                "write_file",
                json!({ "path": target.to_str().unwrap(), "content": "yay" }),
            )
            .await
            .expect("call ok");
        assert!(result.is_error.is_none(), "got {:?}", result);
        assert_eq!(fs::read_to_string(&target).unwrap(), "yay");

        fs::remove_dir_all(&dir).ok();
    }

    // ---- Web tool helpers ----

    #[test]
    fn percent_encode_passes_unreserved_bytes() {
        assert_eq!(percent_encode_query("abcXYZ123-_.~"), "abcXYZ123-_.~");
    }

    #[test]
    fn percent_encode_escapes_spaces_and_symbols() {
        assert_eq!(percent_encode_query("a b/c?d"), "a%20b%2Fc%3Fd");
    }

    #[test]
    fn percent_decode_roundtrips_common_chars() {
        assert_eq!(percent_decode("a%20b%2Fc"), "a b/c");
        // Malformed escapes leave the bytes intact rather than panicking.
        assert_eq!(percent_decode("a%2"), "a%2");
        assert_eq!(percent_decode("%ZZ"), "%ZZ");
    }

    #[test]
    fn strip_html_removes_tags_and_collapses_whitespace() {
        let input = "<p>Hello   <b>world</b>!</p>";
        assert_eq!(strip_html(input), "Hello world !");
    }

    #[test]
    fn strip_html_drops_script_and_style_bodies() {
        let input = "<style>body{color:red}</style>before<script>alert(1)</script>after";
        let out = strip_html(input);
        assert!(!out.contains("alert"), "got: {out}");
        assert!(!out.contains("color"), "got: {out}");
        assert!(out.contains("before"));
        assert!(out.contains("after"));
    }

    #[test]
    fn strip_html_decodes_basic_entities() {
        let out = strip_html("<p>5 &lt; 10 &amp; 20 &gt; 15</p>");
        assert!(out.contains("5 < 10 & 20 > 15"), "got: {out}");
    }

    #[test]
    fn strip_html_preserves_non_ascii_utf8() {
        // Mix of CJK, accented Latin, and emoji — must round-trip
        // through the tag stripper without being mangled into mojibake.
        let out = strip_html("<p>こんにちは <b>café</b> 🎉 — naïve</p>");
        assert!(out.contains("こんにちは"), "got: {out}");
        assert!(out.contains("café"), "got: {out}");
        assert!(out.contains("🎉"), "got: {out}");
        assert!(out.contains("naïve"), "got: {out}");
    }

    #[test]
    fn strip_html_drops_script_with_non_ascii_body() {
        let out = strip_html("before<script>let x = 'こんにちは'; alert(x);</script>after");
        assert!(!out.contains("alert"), "got: {out}");
        assert!(!out.contains("こんにちは"), "got: {out}");
        assert!(out.contains("before"));
        assert!(out.contains("after"));
    }

    #[test]
    fn unwrap_ddg_redirect_returns_destination() {
        let href = "/l/?kh=-1&uddg=https%3A%2F%2Fexample.com%2Fpage&rut=foo";
        assert_eq!(
            unwrap_ddg_redirect(href).as_deref(),
            Some("https://example.com/page")
        );
    }

    #[test]
    fn unwrap_ddg_redirect_returns_none_for_plain_link() {
        assert!(unwrap_ddg_redirect("https://example.com/").is_none());
    }

    #[test]
    fn parse_ddg_lite_results_extracts_unique_external_links() {
        // Cut-down lite-mode-shaped HTML with one redirect link, one
        // direct external link, one DDG-internal link, and one repeat.
        let html = r#"
            <html><body>
            <a href="/l/?uddg=https%3A%2F%2Fexample.com%2Fone&rut=x">First Title</a>
            <a href="https://example.org/two">Second Title</a>
            <a href="https://duckduckgo.com/internal">Internal</a>
            <a href="/l/?uddg=https%3A%2F%2Fexample.com%2Fone&rut=x">First Title duplicate</a>
            </body></html>
        "#;
        let results = parse_ddg_lite_results(html);
        assert_eq!(results.len(), 2, "got: {results:?}");
        assert!(results[0].contains("First Title"));
        assert!(results[0].contains("https://example.com/one"));
        assert!(results[1].contains("Second Title"));
        assert!(results[1].contains("https://example.org/two"));
    }

    // ---- Worktree tool ----

    #[tokio::test]
    async fn worktree_list_runs_in_a_git_repo() {
        // We're inside the cubi repo, so `git worktree list` works.
        let dir = std::env::current_dir().unwrap();
        let registry = registry_with_trust(&dir, false);
        let result = registry
            .execute("worktree", json!({ "action": "list" }))
            .await
            .expect("call ok");
        assert!(result.is_error.is_none(), "got {:?}", result);
        // The porcelain output always at least includes a `worktree ` line.
        assert!(
            result.content[0].text.contains("worktree ")
                || result.content[0].text == "(no worktrees)",
            "got: {}",
            result.content[0].text
        );
    }

    #[tokio::test]
    async fn worktree_add_refused_in_plan_mode() {
        let dir = unique_tmp("worktree-plan");
        let registry = registry_with_trust(&dir, true);
        let result = registry
            .execute(
                "worktree",
                json!({ "action": "add", "path": "/tmp/should-not-be-created" }),
            )
            .await
            .expect("call ok");
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("plan mode is ON"));
        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn worktree_rejects_unknown_action() {
        let dir = unique_tmp("worktree-unknown");
        let registry = registry_with_trust(&dir, false);
        let result = registry
            .execute("worktree", json!({ "action": "destroy" }))
            .await
            .expect("call ok");
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("Unknown worktree action"));
        fs::remove_dir_all(&dir).ok();
    }

    // ---- Web tool refusals ----

    #[tokio::test]
    async fn web_fetch_refused_in_plan_mode() {
        let dir = unique_tmp("web-plan");
        let registry = registry_with_trust(&dir, true);
        let result = registry
            .execute("web_fetch", json!({ "url": "https://example.com/" }))
            .await
            .expect("call ok");
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("plan mode is ON"));
        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn web_fetch_rejects_non_http_schemes() {
        let dir = unique_tmp("web-scheme");
        let registry = registry_with_trust(&dir, false);
        let result = registry
            .execute("web_fetch", json!({ "url": "file:///etc/passwd" }))
            .await
            .expect("call ok");
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("http(s)"));
        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn web_search_rejects_empty_query() {
        let dir = unique_tmp("web-empty");
        let registry = registry_with_trust(&dir, false);
        let result = registry
            .execute("web_search", json!({ "query": "" }))
            .await
            .expect("call ok");
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("non-empty"));
        fs::remove_dir_all(&dir).ok();
    }

    // ---- REPL tool ----

    #[cfg(unix)]
    #[tokio::test]
    async fn repl_full_lifecycle_preserves_state_across_evals() {
        let registry = registry_trusting_cwd(false);

        // Start the session.
        let start = registry
            .execute("repl_start", json!({ "session_id": "s1" }))
            .await
            .expect("call ok");
        assert!(start.is_error.is_none(), "got {:?}", start);

        // First eval: set a variable.
        let e1 = registry
            .execute(
                "repl_eval",
                json!({ "session_id": "s1", "code": "FOO=bar", "timeout": 10 }),
            )
            .await
            .expect("call ok");
        assert!(e1.is_error.is_none(), "got {:?}", e1);
        assert!(e1.content[0].text.contains("[exit 0]"));

        // Second eval: read it back. State must have persisted.
        let e2 = registry
            .execute(
                "repl_eval",
                json!({ "session_id": "s1", "code": "echo \"V=$FOO\"", "timeout": 10 }),
            )
            .await
            .expect("call ok");
        assert!(e2.is_error.is_none(), "got {:?}", e2);
        assert!(
            e2.content[0].text.contains("V=bar"),
            "expected `V=bar` in output, got: {}",
            e2.content[0].text
        );

        // Third eval: nonzero exit must surface as a tool error.
        let e3 = registry
            .execute(
                "repl_eval",
                json!({ "session_id": "s1", "code": "false", "timeout": 10 }),
            )
            .await
            .expect("call ok");
        assert_eq!(e3.is_error, Some(true));
        assert!(e3.content[0].text.contains("[exit 1]"));

        // Close.
        let close = registry
            .execute("repl_close", json!({ "session_id": "s1" }))
            .await
            .expect("call ok");
        assert!(close.is_error.is_none(), "got {:?}", close);

        // Closing again must error.
        let close2 = registry
            .execute("repl_close", json!({ "session_id": "s1" }))
            .await
            .expect("call ok");
        assert_eq!(close2.is_error, Some(true));
    }

    #[tokio::test]
    async fn repl_eval_unknown_session_errors() {
        let registry = registry_trusting_cwd(false);
        let r = registry
            .execute(
                "repl_eval",
                json!({ "session_id": "nope", "code": "echo hi" }),
            )
            .await
            .expect("call ok");
        assert_eq!(r.is_error, Some(true));
        assert!(r.content[0].text.contains("No REPL session"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn repl_eval_captures_stderr_in_band() {
        // Reviewer concern: stderr was being silently drained. After the
        // fix, `exec 2>&1` is injected so anything a command writes to
        // its stderr appears in the eval output.
        let registry = registry_trusting_cwd(false);
        let _ = registry
            .execute("repl_start", json!({ "session_id": "err" }))
            .await
            .expect("start ok");
        let r = registry
            .execute(
                "repl_eval",
                json!({
                    "session_id": "err",
                    "code": "echo to-stderr 1>&2",
                    "timeout": 10
                }),
            )
            .await
            .expect("eval ok");
        assert!(r.is_error.is_none(), "got {:?}", r);
        assert!(
            r.content[0].text.contains("to-stderr"),
            "expected stderr-merged output, got: {}",
            r.content[0].text
        );
        let _ = registry
            .execute("repl_close", json!({ "session_id": "err" }))
            .await;
    }

    #[tokio::test]
    async fn repl_start_refused_in_plan_mode() {
        let dir = unique_tmp("repl-plan");
        let registry = registry_with_trust(&dir, true);
        let r = registry
            .execute("repl_start", json!({ "session_id": "x" }))
            .await
            .expect("call ok");
        assert_eq!(r.is_error, Some(true));
        assert!(r.content[0].text.contains("plan mode is ON"));
        fs::remove_dir_all(&dir).ok();
    }

    // ---- Notebook tool ----

    fn empty_notebook() -> serde_json::Value {
        json!({
            "cells": [],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5
        })
    }

    fn write_notebook(dir: &Path, nb: &serde_json::Value) -> std::path::PathBuf {
        let p = dir.join("nb.ipynb");
        fs::write(&p, serde_json::to_string(nb).unwrap()).unwrap();
        p
    }

    #[test]
    fn split_source_preserves_newlines_on_intermediate_lines() {
        assert_eq!(split_source_for_nbformat(""), Vec::<String>::new());
        assert_eq!(
            split_source_for_nbformat("a\nb\nc"),
            vec!["a\n".to_string(), "b\n".to_string(), "c".to_string()]
        );
        assert_eq!(
            split_source_for_nbformat("a\nb\n"),
            vec!["a\n".to_string(), "b\n".to_string()]
        );
    }

    #[test]
    fn notebook_cell_source_handles_string_and_array_forms() {
        let s = json!({ "source": "hello" });
        assert_eq!(notebook_cell_source(&s), "hello");
        let a = json!({ "source": ["line 1\n", "line 2"] });
        assert_eq!(notebook_cell_source(&a), "line 1\nline 2");
    }

    #[tokio::test]
    async fn notebook_list_and_insert_and_read_roundtrip() {
        let dir = unique_tmp("nb-list");
        let registry = registry_with_trust(&dir, false);
        let path = write_notebook(&dir, &empty_notebook());
        let p = path.to_str().unwrap().to_string();

        // List on an empty notebook.
        let r0 = registry
            .execute("notebook", json!({ "action": "list", "path": p }))
            .await
            .expect("ok");
        assert!(r0.is_error.is_none(), "got {:?}", r0);
        assert!(r0.content[0].text.starts_with("0 cell(s)"));

        // Insert a markdown cell at the top.
        let r1 = registry
            .execute(
                "notebook",
                json!({
                    "action": "insert",
                    "path": p,
                    "cell_index": 0,
                    "cell_type": "markdown",
                    "source": "# Title\nSubtitle",
                }),
            )
            .await
            .expect("ok");
        assert!(r1.is_error.is_none(), "got {:?}", r1);

        // Insert a code cell at the end.
        let r2 = registry
            .execute(
                "notebook",
                json!({
                    "action": "insert",
                    "path": p,
                    "cell_index": 1,
                    "cell_type": "code",
                    "source": "print('hi')",
                }),
            )
            .await
            .expect("ok");
        assert!(r2.is_error.is_none(), "got {:?}", r2);

        // Read cell 0.
        let r3 = registry
            .execute(
                "notebook",
                json!({ "action": "read", "path": p, "cell_index": 0 }),
            )
            .await
            .expect("ok");
        assert!(r3.is_error.is_none());
        assert!(r3.content[0].text.contains("markdown"));
        assert!(r3.content[0].text.contains("Title"));

        // List after inserts.
        let r4 = registry
            .execute("notebook", json!({ "action": "list", "path": p }))
            .await
            .expect("ok");
        assert!(r4.content[0].text.starts_with("2 cell(s)"));
        assert!(r4.content[0].text.contains("markdown"));
        assert!(r4.content[0].text.contains("code"));

        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn notebook_replace_resets_outputs_on_code_cells() {
        let dir = unique_tmp("nb-replace");
        let registry = registry_with_trust(&dir, false);
        let mut nb = empty_notebook();
        nb["cells"] = json!([{
            "cell_type": "code",
            "source": ["old"],
            "metadata": {},
            "outputs": [{"output_type": "stream", "name": "stdout", "text": ["stale"]}],
            "execution_count": 7
        }]);
        let path = write_notebook(&dir, &nb);
        let p = path.to_str().unwrap().to_string();

        let r = registry
            .execute(
                "notebook",
                json!({
                    "action": "replace",
                    "path": p,
                    "cell_index": 0,
                    "source": "new\nsource",
                }),
            )
            .await
            .expect("ok");
        assert!(r.is_error.is_none(), "got {:?}", r);

        // Reload the file and confirm the outputs were cleared.
        let reloaded: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(reloaded["cells"][0]["outputs"].as_array().unwrap().len(), 0);
        assert!(reloaded["cells"][0]["execution_count"].is_null());
        let src = notebook_cell_source(&reloaded["cells"][0]);
        assert_eq!(src, "new\nsource");

        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn notebook_delete_out_of_range_errors() {
        let dir = unique_tmp("nb-delete");
        let registry = registry_with_trust(&dir, false);
        let path = write_notebook(&dir, &empty_notebook());
        let p = path.to_str().unwrap().to_string();

        let r = registry
            .execute(
                "notebook",
                json!({ "action": "delete", "path": p, "cell_index": 5 }),
            )
            .await
            .expect("ok");
        assert_eq!(r.is_error, Some(true));
        assert!(r.content[0].text.contains("out of range"));
        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn notebook_write_refused_in_plan_mode() {
        let dir = unique_tmp("nb-plan");
        let registry = registry_with_trust(&dir, true);
        let path = write_notebook(&dir, &empty_notebook());
        let p = path.to_str().unwrap().to_string();
        let r = registry
            .execute(
                "notebook",
                json!({
                    "action": "insert",
                    "path": p,
                    "cell_index": 0,
                    "source": "x",
                }),
            )
            .await
            .expect("ok");
        assert_eq!(r.is_error, Some(true));
        assert!(r.content[0].text.contains("plan mode is ON"));
        fs::remove_dir_all(&dir).ok();
    }

    // ---- New tool helpers (shell / sleep / schedule / brief /
    // synthetic_output / send_message / recv_messages / remote_trigger /
    // notify) ----

    #[test]
    fn host_shell_matches_target_os() {
        let (program, flag) = host_shell();
        if cfg!(windows) {
            assert!(program == "pwsh" || program == "powershell");
            assert_eq!(flag, "-Command");
        } else {
            assert_eq!(program, "sh");
            assert_eq!(flag, "-c");
        }
    }

    #[test]
    fn is_safe_agent_name_accepts_basic_identifiers() {
        assert!(is_safe_agent_name("alice"));
        assert!(is_safe_agent_name("alice_2"));
        assert!(is_safe_agent_name("alice-bob_2"));
        assert!(!is_safe_agent_name(""));
        assert!(!is_safe_agent_name("alice/../etc"));
        assert!(!is_safe_agent_name("alice bob"));
        assert!(!is_safe_agent_name("alice.json"));
        // Cap at 63 characters.
        assert!(!is_safe_agent_name(&"a".repeat(64)));
    }

    #[test]
    fn validate_cron_like_requires_five_fields() {
        assert!(validate_cron_like("* * * * *"));
        assert!(validate_cron_like("*/5 * * * *"));
        assert!(validate_cron_like("0 0 1 1 0"));
        assert!(!validate_cron_like(""));
        assert!(!validate_cron_like("* * * *"));
        assert!(!validate_cron_like("* * * * * *"));
    }

    #[test]
    fn build_brief_extracts_title_bullets_and_summary() {
        let text = "Title line. With more detail.\n\nFirst point\nSecond point\nThird point\nFourth point\nFifth point\nSixth point";
        let brief = build_brief(text, 3);
        assert_eq!(brief["title"], "Title line. With more detail.");
        assert_eq!(brief["summary"], "Title line");
        let bullets = brief["bullets"].as_array().unwrap();
        assert_eq!(bullets.len(), 3);
        assert_eq!(bullets[0], "First point");
        assert_eq!(bullets[2], "Third point");
    }

    #[test]
    fn build_brief_handles_empty_input() {
        let brief = build_brief("", 5);
        assert_eq!(brief["title"], "");
        assert_eq!(brief["bullets"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn synthesize_from_schema_fills_typed_defaults() {
        let schema = json!({
            "properties": {
                "name":    { "type": "string" },
                "count":   { "type": "integer" },
                "ratio":   { "type": "number" },
                "ok":      { "type": "boolean" },
                "tags":    { "type": "array" },
                "meta":    { "type": "object" },
                "blob":    {}
            }
        });
        let out = synthesize_from_schema(&schema, "hello");
        assert_eq!(out["name"], "hello");
        assert_eq!(out["count"], 0);
        assert_eq!(out["ratio"], 0);
        assert_eq!(out["ok"], false);
        assert!(out["tags"].as_array().unwrap().is_empty());
        assert!(out["meta"].as_object().unwrap().is_empty());
        // No type → string field omitted defaults to string→context.
        assert_eq!(out["blob"], "hello");
    }

    #[test]
    fn synthesize_from_schema_handles_missing_properties() {
        let out = synthesize_from_schema(&json!({}), "x");
        assert!(out.as_object().unwrap().is_empty());
    }

    #[tokio::test]
    async fn sleep_blocks_for_requested_duration() {
        let registry = registry_trusting_cwd(false);
        let start = std::time::Instant::now();
        let r = registry
            .execute("sleep", json!({ "seconds": 0.1 }))
            .await
            .expect("ok");
        let elapsed = start.elapsed();
        assert!(r.is_error.is_none(), "got {:?}", r);
        assert!(
            elapsed >= Duration::from_millis(90),
            "sleep returned too fast: {elapsed:?}"
        );
        assert!(r.content[0].text.contains("slept"));
    }

    #[test]
    fn sleep_caps_at_sixty_seconds() {
        assert_eq!(capped_sleep_seconds(90.0), Some(60.0));
        assert_eq!(capped_sleep_seconds(0.0), Some(0.0));
        assert_eq!(capped_sleep_seconds(-1.0), None);
    }

    #[tokio::test]
    async fn sleep_rejects_negative_seconds() {
        let registry = registry_trusting_cwd(false);
        let r = registry
            .execute("sleep", json!({ "seconds": -1 }))
            .await
            .expect("ok");
        assert_eq!(r.is_error, Some(true));
    }

    #[tokio::test]
    async fn brief_tool_returns_structured_json() {
        let registry = registry_trusting_cwd(false);
        let r = registry
            .execute(
                "brief",
                json!({ "text": "Headline.\n\nfact a\nfact b", "max_bullets": 2 }),
            )
            .await
            .expect("ok");
        assert!(r.is_error.is_none());
        let parsed: serde_json::Value = serde_json::from_str(&r.content[0].text).unwrap();
        assert_eq!(parsed["title"], "Headline.");
        let bullets = parsed["bullets"].as_array().unwrap();
        assert_eq!(bullets.len(), 2);
    }

    #[tokio::test]
    async fn synthetic_output_tool_emits_object() {
        let registry = registry_trusting_cwd(false);
        let r = registry
            .execute(
                "synthetic_output",
                json!({
                    "schema": {
                        "properties": {
                            "title": { "type": "string" },
                            "count": { "type": "integer" }
                        }
                    },
                    "context": "hi"
                }),
            )
            .await
            .expect("ok");
        assert!(r.is_error.is_none());
        let parsed: serde_json::Value = serde_json::from_str(&r.content[0].text).unwrap();
        assert_eq!(parsed["title"], "hi");
        assert_eq!(parsed["count"], 0);
    }

    #[tokio::test]
    async fn send_message_rejects_invalid_recipient() {
        let registry = registry_trusting_cwd(false);
        let r = registry
            .execute(
                "send_message",
                json!({ "to": "alice/../etc", "body": "ping" }),
            )
            .await
            .expect("ok");
        assert_eq!(r.is_error, Some(true));
        assert!(r.content[0].text.contains("[A-Za-z0-9_-]+"));
    }

    #[tokio::test]
    async fn recv_messages_with_drain_clears_mailbox() {
        // Use a unique recipient name so we don't collide with the user's
        // real mailbox.
        let recipient = format!(
            "test-{}",
            unix_timestamp().to_string() + &uuid::Uuid::new_v4().to_string()[..8]
        );
        let safe: String = recipient
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '-'
                }
            })
            .collect();
        let registry = registry_trusting_cwd(false);

        let send = registry
            .execute(
                "send_message",
                json!({ "to": safe, "body": "hi there", "from": "tester" }),
            )
            .await
            .expect("ok");
        assert!(send.is_error.is_none(), "got {:?}", send);

        let recv = registry
            .execute("recv_messages", json!({ "recipient": safe, "drain": true }))
            .await
            .expect("ok");
        assert!(recv.is_error.is_none(), "got {:?}", recv);
        let msgs: Vec<serde_json::Value> = serde_json::from_str(&recv.content[0].text).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["body"], "hi there");

        // Mailbox should now be empty.
        let again = registry
            .execute("recv_messages", json!({ "recipient": safe }))
            .await
            .expect("ok");
        let msgs2: Vec<serde_json::Value> = serde_json::from_str(&again.content[0].text).unwrap();
        assert!(msgs2.is_empty());

        // Cleanup mailbox file we created in $HOME.
        if let Some(p) = messages_path(&safe) {
            fs::remove_file(&p).ok();
        }
    }

    #[tokio::test]
    async fn schedule_add_then_remove_round_trips() {
        let registry = registry_trusting_cwd(false);
        let unique_name = format!("test-sched-{}", &uuid::Uuid::new_v4().to_string()[..8]);

        let add = registry
            .execute(
                "schedule",
                json!({
                    "action": "add",
                    "name": unique_name,
                    "when": "*/5 * * * *",
                    "command": "echo hi"
                }),
            )
            .await
            .expect("ok");
        assert!(add.is_error.is_none(), "got {:?}", add);

        let list = registry
            .execute("schedule", json!({ "action": "list" }))
            .await
            .expect("ok");
        assert!(list.is_error.is_none());
        assert!(list.content[0].text.contains(&unique_name));

        let rm = registry
            .execute(
                "schedule",
                json!({ "action": "remove", "name": unique_name }),
            )
            .await
            .expect("ok");
        assert!(rm.is_error.is_none(), "got {:?}", rm);
    }

    #[tokio::test]
    async fn schedule_rejects_bad_cron_expression() {
        let registry = registry_trusting_cwd(false);
        let r = registry
            .execute(
                "schedule",
                json!({
                    "action": "add",
                    "name": "x",
                    "when": "every minute",
                    "command": "echo hi"
                }),
            )
            .await
            .expect("ok");
        assert_eq!(r.is_error, Some(true));
        assert!(r.content[0].text.contains("Invalid cron"));
    }

    #[tokio::test]
    async fn shell_executes_simple_command_on_unix() {
        if cfg!(windows) {
            return;
        }
        let registry = registry_trusting_cwd(false);
        let r = registry
            .execute("shell", json!({ "command": "echo from-shell" }))
            .await
            .expect("ok");
        assert!(r.is_error.is_none(), "got {:?}", r);
        assert!(r.content[0].text.contains("from-shell"));
    }

    #[tokio::test]
    async fn shell_blocked_in_plan_mode() {
        let registry = registry_trusting_cwd(true);
        let r = registry
            .execute("shell", json!({ "command": "echo no" }))
            .await
            .expect("ok");
        assert_eq!(r.is_error, Some(true));
        assert!(r.content[0].text.contains("plan mode is ON"));
    }

    #[tokio::test]
    async fn remote_trigger_writes_payload_file() {
        let registry = registry_trusting_cwd(false);
        let name = format!("test-trigger-{}", &uuid::Uuid::new_v4().to_string()[..8]);
        let r = registry
            .execute(
                "remote_trigger",
                json!({ "name": name, "payload": { "x": 1 } }),
            )
            .await
            .expect("ok");
        assert!(r.is_error.is_none(), "got {:?}", r);

        let path = triggers_path(&name).unwrap();
        assert!(path.exists());
        let raw = fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(parsed["payload"]["x"], 1);
        fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn notify_returns_message_not_an_error() {
        let registry = registry_trusting_cwd(false);
        let r = registry
            .execute("notify", json!({ "title": "t", "message": "hello" }))
            .await
            .expect("ok");
        // We can't guarantee a notifier exists in CI, but the tool must
        // degrade gracefully (success or descriptive non-fatal message)
        // rather than crash.
        assert!(r.is_error.is_none(), "got {:?}", r);
    }
}
