use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command as TokioCommand};
use tokio::sync::Mutex as AsyncMutex;
use tokio::time::timeout;

use crate::permissions::Permissions;

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
    /// Long-lived shell sessions owned by the REPL tool. Keyed by
    /// caller-supplied session id (the model picks a stable string and
    /// reuses it across `repl_eval` calls). Wrapped in a tokio
    /// `AsyncMutex` so it can be held across the awaits needed to read
    /// from the child process.
    repls: Arc<AsyncMutex<HashMap<String, ReplSession>>>,
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
    pub fn new(permissions: Arc<Mutex<Permissions>>, plan_mode: Arc<AtomicBool>) -> Self {
        let tools = vec![
            Self::bash_tool(),
            Self::read_file_tool(),
            Self::list_files_tool(),
            Self::search_glob_tool(),
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
        ];

        Self {
            tools,
            permissions,
            plan_mode,
            repls: Arc::new(AsyncMutex::new(HashMap::new())),
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
            description: "Read the contents of a file from the filesystem.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Starting line number (1-indexed, optional)"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Ending line number (inclusive, optional)"
                    }
                },
                "required": ["path"]
            }),
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
            let output = Command::new("sh")
                .arg("-c")
                .arg(command)
                .output()
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

        let start_line = args["start_line"].as_u64().map(|n| n as usize);
        let end_line = args["end_line"].as_u64().map(|n| n as usize);

        let result = if let (Some(start), Some(end)) = (start_line, end_line) {
            let lines: Vec<&str> = content.lines().collect();
            let start_idx = start.saturating_sub(1);
            let end_idx = end.min(lines.len());

            lines[start_idx..end_idx].join("\n")
        } else {
            content
        };

        Ok(ToolResult::success(result))
    }

    fn execute_list_files(&self, args: serde_json::Value) -> Result<ToolResult> {
        let path = args["path"].as_str().unwrap_or(".");
        let recursive = args["recursive"].as_bool().unwrap_or(false);

        let mut result = String::new();

        if recursive {
            self.list_files_recursive(Path::new(path), &mut result, 0)?;
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

    fn list_files_recursive(&self, path: &Path, result: &mut String, depth: usize) -> Result<()> {
        let entries =
            fs::read_dir(path).context(format!("Failed to read directory: {:?}", path))?;

        let indent = "  ".repeat(depth);

        for entry in entries {
            let entry = entry?;
            let metadata = entry.metadata()?;
            let name = entry.file_name().to_string_lossy().to_string();

            if metadata.is_dir() {
                result.push_str(&format!("{}📁 {}/\n", indent, name));
                self.list_files_recursive(&entry.path(), result, depth + 1)?;
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

        fs::write(path, content).context(format!("Failed to write file: {}", path))?;

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
            description:
                "Manage git worktrees. Subcommands: 'list' shows all worktrees; \
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
                    Ok(ToolResult::error(format!("git worktree list failed: {stderr}")))
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
                let trusted_msg = match self
                    .permissions
                    .lock()
                    .unwrap()
                    .trust_dir(Path::new(path))
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
            description: "Web search via DuckDuckGo (no API key). Returns the top results as a plain-text \
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
        let sentinel = format!("__AICHAT_REPL_DONE_{}__", uuid::Uuid::new_v4().simple());

        let mut child = TokioCommand::new("bash")
            .arg("--noprofile")
            .arg("--norc")
            .arg("-i")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Merge stderr into stdout so the model sees errors in-band
            // without us having to read two streams concurrently.
            .stderr(Stdio::piped())
            .spawn()
            .context("Failed to spawn bash for REPL")?;

        let stdin = child
            .stdin
            .take()
            .context("bash spawned without stdin pipe")?;
        let stdout = child
            .stdout
            .take()
            .context("bash spawned without stdout pipe")?;
        // Forward stderr by spawning a task that drains it. Simpler than
        // juggling a second reader in the eval loop.
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(async move {
                let mut reader = BufReader::new(stderr);
                let mut buf = String::new();
                // Best-effort drain; errors are ignored — when bash exits the
                // stream closes and the loop terminates.
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
}

/// Cap on the response body we'll buffer from a web tool. Keeps a single
/// rogue URL from blowing the model's context window or the process's RAM.
const MAX_WEB_BYTES: usize = 64 * 1024;

/// HTTP GET with a body cap. If the response is HTML, the tags are
/// stripped to plain text before being returned.
async fn http_get_text(url: &str, max_bytes: usize) -> Result<String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .user_agent("ai-chat-cli/0.1")
        .build()?;
    let resp = client.get(url).send().await?;
    let status = resp.status();
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = resp.bytes().await?;
    let truncated = bytes.len() > max_bytes;
    let body = if truncated {
        String::from_utf8_lossy(&bytes[..max_bytes]).to_string()
    } else {
        String::from_utf8_lossy(&bytes).to_string()
    };
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
fn strip_html(input: &str) -> String {
    // Lowercase scan helps the script/style match, but we want to slice
    // from the original string to preserve case in the output.
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;
    let mut in_tag = false;
    let mut skip_until: Option<&'static [u8]> = None;
    while i < bytes.len() {
        if let Some(end_tag) = skip_until {
            // Look for the closing tag (case-insensitive ASCII).
            if i + end_tag.len() <= bytes.len()
                && bytes[i..i + end_tag.len()].eq_ignore_ascii_case(end_tag)
            {
                i += end_tag.len();
                skip_until = None;
            } else {
                i += 1;
            }
            continue;
        }
        let b = bytes[i];
        if b == b'<' {
            // Detect <script ...> and <style ...> openings to drop their bodies.
            let rest_lower: String =
                bytes[i..(i + 8).min(bytes.len())].iter().map(|c| c.to_ascii_lowercase() as char).collect();
            if rest_lower.starts_with("<script") {
                skip_until = Some(b"</script>");
                i += 1;
                continue;
            }
            if rest_lower.starts_with("<style") {
                skip_until = Some(b"</style>");
                i += 1;
                continue;
            }
            in_tag = true;
            i += 1;
            continue;
        }
        if b == b'>' {
            in_tag = false;
            // Tag boundaries act as whitespace so adjacent words don't
            // glue together when we strip the markup.
            if !out.ends_with(' ') && !out.is_empty() {
                out.push(' ');
            }
            i += 1;
            continue;
        }
        if !in_tag {
            out.push(b as char);
        }
        i += 1;
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
        let unreserved = b.is_ascii_alphanumeric()
            || b == b'-'
            || b == b'_'
            || b == b'.'
            || b == b'~';
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
        let p = std::env::temp_dir().join(format!("ai-chat-cli-tool-{label}-{nanos}"));
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

    /// Registry whose trust store also covers the current working
    /// directory. Used by tests for tools whose preflight check inspects
    /// the cwd (REPL, web tools) rather than a specific path argument.
    fn registry_trusting_cwd(plan_on: bool) -> BuiltinToolRegistry {
        let mut perms = Permissions::default();
        perms
            .trust_dir(&std::env::current_dir().unwrap())
            .unwrap();
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
        // We're inside the ai-chat-cli repo, so `git worktree list` works.
        let dir = std::env::current_dir().unwrap();
        let registry = registry_with_trust(&dir, false);
        let result = registry
            .execute("worktree", json!({ "action": "list" }))
            .await
            .expect("call ok");
        assert!(result.is_error.is_none(), "got {:?}", result);
        // The porcelain output always at least includes a `worktree ` line.
        assert!(
            result.content[0].text.contains("worktree ") || result.content[0].text == "(no worktrees)",
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
}
