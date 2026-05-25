//! Hook system for cubi.
//!
//! Hooks allow users to run custom logic at defined lifecycle points:
//!
//! * **PreToolUse** — fires before a built-in or MCP tool executes.
//!   Can approve or deny the call (via exit code).
//! * **PostToolUse** — fires after a tool returns. Can inspect/log the result.
//! * **SessionStart** — fires when a new chat session begins.
//! * **Stop** — fires when the session ends (user quits or EOF).
//!
//! ## Hook definition
//!
//! Hooks are defined in `~/.cubi/hooks.json` (global) or
//! `.cubi/hooks.json` (project-local, overrides global). Each hook
//! is a shell command that receives context via environment variables.
//! The hook's exit code determines behavior:
//!
//! * Exit 0 → proceed normally
//! * Exit 1 → for PreToolUse: deny the tool call (the model sees a refusal)
//! * Exit 2+ → ignored (logged as warning)
//!
//! Environment variables provided to hooks:
//! * `HOOK_EVENT` — the lifecycle event name
//! * `HOOK_TOOL_NAME` — (PreToolUse/PostToolUse) the tool being invoked
//! * `HOOK_TOOL_ARGS` — (PreToolUse) JSON-encoded tool arguments
//! * `HOOK_TOOL_RESULT` — (PostToolUse) the tool's output
//! * `HOOK_TOOL_ERROR` — (PostToolUse) "true" if the tool errored
//! * `HOOK_MODEL` — (SessionStart) the model name
//! * `HOOK_CWD` — (SessionStart) the working directory
//!
//! ## Configuration format
//!
//! ```json
//! {
//!   "hooks": [
//!     {
//!       "event": "PreToolUse",
//!       "match_tool": "bash",
//!       "command": "echo $HOOK_TOOL_NAME >> /tmp/tool-audit.log"
//!     },
//!     {
//!       "event": "PostToolUse",
//!       "command": "notify-send 'Tool done: $HOOK_TOOL_NAME'"
//!     },
//!     {
//!       "event": "SessionStart",
//!       "command": "echo 'session started' >> /tmp/sessions.log"
//!     },
//!     {
//!       "event": "Stop",
//!       "command": "echo 'session ended' >> /tmp/sessions.log"
//!     }
//!   ]
//! }
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Hook lifecycle event types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HookEvent {
    /// Before a tool is executed.
    PreToolUse,
    /// After a tool has finished.
    PostToolUse,
    /// When the chat session starts.
    SessionStart,
    /// When the session ends (quit/EOF).
    Stop,
}

impl HookEvent {
    #[allow(dead_code)]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PreToolUse => "PreToolUse",
            Self::PostToolUse => "PostToolUse",
            Self::SessionStart => "SessionStart",
            Self::Stop => "Stop",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "pretooluse" | "pre-tool-use" | "pre_tool_use" => Some(Self::PreToolUse),
            "posttooluse" | "post-tool-use" | "post_tool_use" => Some(Self::PostToolUse),
            "sessionstart" | "session-start" | "session_start" => Some(Self::SessionStart),
            "stop" => Some(Self::Stop),
            _ => None,
        }
    }
}

/// Result of running a PreToolUse hook.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HookDecision {
    /// Allow the tool call to proceed.
    Allow,
    /// Deny the tool call (hook exited with code 1).
    Deny(String),
}

/// A single hook definition from the config file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookDef {
    /// Which event triggers this hook.
    pub event: HookEvent,
    /// Optional tool name filter. If set, only fires for matching tool names.
    /// Supports exact match or glob-like `*` prefix/suffix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub match_tool: Option<String>,
    /// Shell command to execute.
    pub command: String,
}

/// Top-level hooks configuration file shape.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HooksConfig {
    #[serde(default)]
    pub hooks: Vec<HookDef>,
}

/// The runtime hook registry.
#[derive(Debug, Clone)]
pub struct HookRegistry {
    hooks: Vec<HookDef>,
}

impl HookRegistry {
    /// Loads hooks from project-local and global config files.
    /// Project-local hooks are prepended (run first).
    pub fn load() -> Self {
        let mut hooks = Vec::new();

        // Project-local: .cubi/hooks.json in cwd
        // If a local config exists, it overrides the global config entirely.
        let mut found_local = false;
        if let Ok(cwd) = std::env::current_dir() {
            let local_path = cwd.join(".cubi").join("hooks.json");
            if let Some(cfg) = Self::load_file(&local_path) {
                hooks.extend(cfg.hooks);
                found_local = true;
            }
        }

        // Global: ~/.cubi/hooks.json (only used when no local config exists)
        if !found_local && let Some(home) = dirs::home_dir() {
            let global_path = home.join(".cubi").join("hooks.json");
            if let Some(cfg) = Self::load_file(&global_path) {
                hooks.extend(cfg.hooks);
            }
        }

        Self { hooks }
    }

    /// Creates a registry with specific hooks (for testing).
    #[cfg(test)]
    pub fn with_hooks(hooks: Vec<HookDef>) -> Self {
        Self { hooks }
    }

    /// Creates an empty registry.
    #[allow(dead_code)]
    pub fn empty() -> Self {
        Self { hooks: Vec::new() }
    }

    /// Returns `true` if any hooks are registered.
    #[allow(dead_code)]
    pub fn has_hooks(&self) -> bool {
        !self.hooks.is_empty()
    }

    /// Returns the number of registered hooks.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.hooks.len()
    }

    #[allow(dead_code)]
    pub fn hooks(&self) -> &[HookDef] {
        &self.hooks
    }

    /// Fires PreToolUse hooks for the given tool. Returns `Allow` if all
    /// hooks pass (exit 0) or no hooks match. Returns `Deny` if any hook
    /// exits with code 1.
    pub fn fire_pre_tool_use(&self, tool_name: &str, args: &serde_json::Value) -> HookDecision {
        let matching: Vec<&HookDef> = self
            .hooks
            .iter()
            .filter(|h| h.event == HookEvent::PreToolUse && Self::matches_tool(h, tool_name))
            .collect();

        for hook in matching {
            let mut env = HashMap::new();
            env.insert("HOOK_EVENT".to_string(), "PreToolUse".to_string());
            env.insert("HOOK_TOOL_NAME".to_string(), tool_name.to_string());
            env.insert(
                "HOOK_TOOL_ARGS".to_string(),
                serde_json::to_string(args).unwrap_or_default(),
            );

            match Self::run_hook_command(&hook.command, &env) {
                Ok(1) => {
                    return HookDecision::Deny(format!(
                        "PreToolUse hook denied `{}`: {}",
                        tool_name, hook.command
                    ));
                }
                Ok(_) => {} // 0 or 2+ = proceed
                Err(e) => {
                    eprintln!("Warning: PreToolUse hook failed: {} ({})", hook.command, e);
                }
            }
        }

        HookDecision::Allow
    }

    /// Fires PostToolUse hooks for the given tool.
    pub fn fire_post_tool_use(&self, tool_name: &str, result: &str, is_error: bool) {
        let matching: Vec<&HookDef> = self
            .hooks
            .iter()
            .filter(|h| h.event == HookEvent::PostToolUse && Self::matches_tool(h, tool_name))
            .collect();

        for hook in matching {
            let mut env = HashMap::new();
            env.insert("HOOK_EVENT".to_string(), "PostToolUse".to_string());
            env.insert("HOOK_TOOL_NAME".to_string(), tool_name.to_string());
            env.insert("HOOK_TOOL_RESULT".to_string(), result.to_string());
            env.insert("HOOK_TOOL_ERROR".to_string(), is_error.to_string());

            if let Err(e) = Self::run_hook_command(&hook.command, &env) {
                eprintln!("Warning: PostToolUse hook failed: {} ({})", hook.command, e);
            }
        }
    }

    /// Fires SessionStart hooks.
    pub fn fire_session_start(&self, model: &str) {
        let matching: Vec<&HookDef> = self
            .hooks
            .iter()
            .filter(|h| h.event == HookEvent::SessionStart)
            .collect();

        for hook in matching {
            let mut env = HashMap::new();
            env.insert("HOOK_EVENT".to_string(), "SessionStart".to_string());
            env.insert("HOOK_MODEL".to_string(), model.to_string());
            if let Ok(cwd) = std::env::current_dir() {
                env.insert("HOOK_CWD".to_string(), cwd.display().to_string());
            }

            if let Err(e) = Self::run_hook_command(&hook.command, &env) {
                eprintln!(
                    "Warning: SessionStart hook failed: {} ({})",
                    hook.command, e
                );
            }
        }
    }

    /// Fires Stop hooks.
    pub fn fire_stop(&self) {
        let matching: Vec<&HookDef> = self
            .hooks
            .iter()
            .filter(|h| h.event == HookEvent::Stop)
            .collect();

        for hook in matching {
            let mut env = HashMap::new();
            env.insert("HOOK_EVENT".to_string(), "Stop".to_string());

            if let Err(e) = Self::run_hook_command(&hook.command, &env) {
                eprintln!("Warning: Stop hook failed: {} ({})", hook.command, e);
            }
        }
    }

    // ─── Helpers ────────────────────────────────────────────────────────

    fn load_file(path: &Path) -> Option<HooksConfig> {
        let raw = fs::read_to_string(path).ok()?;
        serde_json::from_str(&raw).ok()
    }

    /// Returns true if the hook's `match_tool` filter matches the given tool.
    fn matches_tool(hook: &HookDef, tool_name: &str) -> bool {
        let Some(pattern) = &hook.match_tool else {
            return true; // no filter = matches all
        };
        if pattern == "*" {
            return true;
        }
        if let Some(suffix) = pattern.strip_prefix('*') {
            return tool_name.ends_with(suffix);
        }
        if let Some(prefix) = pattern.strip_suffix('*') {
            return tool_name.starts_with(prefix);
        }
        pattern == tool_name
    }

    /// Executes a hook command with the given environment variables.
    /// Returns the exit code.
    fn run_hook_command(command: &str, env: &HashMap<String, String>) -> Result<i32> {
        let output = Command::new("sh")
            .arg("-c")
            .arg(command)
            .envs(env)
            .output()?;
        Ok(output.status.code().unwrap_or(2))
    }
}

/// Writes the global hooks config at `~/.cubi/hooks.json`.
pub fn save_global(hooks: &[HookDef]) -> Result<()> {
    let path =
        global_hooks_path().ok_or_else(|| anyhow::anyhow!("Could not resolve home directory"))?;
    let mut cfg = if path.exists() {
        let raw = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read hooks file: {}", path.display()))?;
        serde_json::from_str::<HooksConfig>(&raw)
            .with_context(|| format!("Failed to parse hooks file: {}", path.display()))?
    } else {
        HooksConfig::default()
    };
    cfg.hooks = hooks.to_vec();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, serde_json::to_string_pretty(&cfg)?)?;
    Ok(())
}

/// Path to the global hooks config.
#[allow(dead_code)]
pub fn global_hooks_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".cubi").join("hooks.json"))
}

/// Path to the project-local hooks config.
#[allow(dead_code)]
pub fn local_hooks_path() -> Option<PathBuf> {
    std::env::current_dir()
        .ok()
        .map(|c| c.join(".cubi").join("hooks.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_tool_exact() {
        let hook = HookDef {
            event: HookEvent::PreToolUse,
            match_tool: Some("bash".to_string()),
            command: "true".to_string(),
        };
        assert!(HookRegistry::matches_tool(&hook, "bash"));
        assert!(!HookRegistry::matches_tool(&hook, "write_file"));
    }

    #[test]
    fn matches_tool_wildcard() {
        let hook = HookDef {
            event: HookEvent::PreToolUse,
            match_tool: Some("*".to_string()),
            command: "true".to_string(),
        };
        assert!(HookRegistry::matches_tool(&hook, "anything"));
    }

    #[test]
    fn matches_tool_prefix_glob() {
        let hook = HookDef {
            event: HookEvent::PreToolUse,
            match_tool: Some("web_*".to_string()),
            command: "true".to_string(),
        };
        assert!(HookRegistry::matches_tool(&hook, "web_fetch"));
        assert!(HookRegistry::matches_tool(&hook, "web_search"));
        assert!(!HookRegistry::matches_tool(&hook, "bash"));
    }

    #[test]
    fn matches_tool_suffix_glob() {
        let hook = HookDef {
            event: HookEvent::PreToolUse,
            match_tool: Some("*_file".to_string()),
            command: "true".to_string(),
        };
        assert!(HookRegistry::matches_tool(&hook, "read_file"));
        assert!(HookRegistry::matches_tool(&hook, "write_file"));
        assert!(!HookRegistry::matches_tool(&hook, "bash"));
    }

    #[test]
    fn matches_tool_none_matches_all() {
        let hook = HookDef {
            event: HookEvent::PreToolUse,
            match_tool: None,
            command: "true".to_string(),
        };
        assert!(HookRegistry::matches_tool(&hook, "anything"));
    }

    #[test]
    fn fire_pre_tool_use_allows_when_no_hooks() {
        let reg = HookRegistry::empty();
        let decision = reg.fire_pre_tool_use("bash", &serde_json::json!({}));
        assert_eq!(decision, HookDecision::Allow);
    }

    #[test]
    fn fire_pre_tool_use_allows_on_exit_0() {
        let reg = HookRegistry::with_hooks(vec![HookDef {
            event: HookEvent::PreToolUse,
            match_tool: None,
            command: "true".to_string(), // exit 0
        }]);
        let decision = reg.fire_pre_tool_use("bash", &serde_json::json!({}));
        assert_eq!(decision, HookDecision::Allow);
    }

    #[test]
    fn fire_pre_tool_use_denies_on_exit_1() {
        let reg = HookRegistry::with_hooks(vec![HookDef {
            event: HookEvent::PreToolUse,
            match_tool: None,
            command: "exit 1".to_string(),
        }]);
        let decision = reg.fire_pre_tool_use("bash", &serde_json::json!({}));
        assert!(matches!(decision, HookDecision::Deny(_)));
    }

    #[test]
    fn fire_pre_tool_use_skips_non_matching_tools() {
        let reg = HookRegistry::with_hooks(vec![HookDef {
            event: HookEvent::PreToolUse,
            match_tool: Some("bash".to_string()),
            command: "exit 1".to_string(),
        }]);
        // Should allow because the hook only matches "bash", not "write_file"
        let decision = reg.fire_pre_tool_use("write_file", &serde_json::json!({}));
        assert_eq!(decision, HookDecision::Allow);
    }

    #[test]
    fn fire_session_start_runs_without_error() {
        let reg = HookRegistry::with_hooks(vec![HookDef {
            event: HookEvent::SessionStart,
            match_tool: None,
            command: "true".to_string(),
        }]);
        // Should not panic.
        reg.fire_session_start("llama3.2:1b");
    }

    #[test]
    fn fire_stop_runs_without_error() {
        let reg = HookRegistry::with_hooks(vec![HookDef {
            event: HookEvent::Stop,
            match_tool: None,
            command: "true".to_string(),
        }]);
        reg.fire_stop();
    }

    #[test]
    fn fire_post_tool_use_runs_without_error() {
        let reg = HookRegistry::with_hooks(vec![HookDef {
            event: HookEvent::PostToolUse,
            match_tool: None,
            command: "true".to_string(),
        }]);
        reg.fire_post_tool_use("bash", "output", false);
    }

    #[test]
    fn load_from_json() {
        let json = r#"{
            "hooks": [
                {
                    "event": "PreToolUse",
                    "match_tool": "bash",
                    "command": "echo test"
                },
                {
                    "event": "SessionStart",
                    "command": "echo started"
                }
            ]
        }"#;
        let config: HooksConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.hooks.len(), 2);
        assert_eq!(config.hooks[0].event, HookEvent::PreToolUse);
        assert_eq!(config.hooks[0].match_tool.as_deref(), Some("bash"));
        assert_eq!(config.hooks[1].event, HookEvent::SessionStart);
        assert!(config.hooks[1].match_tool.is_none());
    }

    #[test]
    fn empty_registry_has_no_hooks() {
        let reg = HookRegistry::empty();
        assert!(!reg.has_hooks());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn hook_event_as_str() {
        assert_eq!(HookEvent::PreToolUse.as_str(), "PreToolUse");
        assert_eq!(HookEvent::PostToolUse.as_str(), "PostToolUse");
        assert_eq!(HookEvent::SessionStart.as_str(), "SessionStart");
        assert_eq!(HookEvent::Stop.as_str(), "Stop");
    }
}
