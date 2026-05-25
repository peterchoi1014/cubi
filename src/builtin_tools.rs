use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
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
        ];

        Self {
            tools,
            permissions,
            plan_mode,
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
}
