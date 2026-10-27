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
use std::process::Stdio;
use std::time::Duration;

const HOOK_TIMEOUT: Duration = Duration::from_secs(5);
const HOOK_TERMINATION_TIMEOUT: Duration = Duration::from_secs(2);

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
    skipped_untrusted_project: bool,
}

impl HookRegistry {
    /// Loads hooks from project-local and global config files.
    /// Project-local hooks are prepended (run first).
    pub fn load(permissions: &crate::permissions::Permissions) -> Self {
        let mut hooks = Vec::new();

        // Project-local: .cubi/hooks.json in cwd
        // If a local config exists, it overrides the global config entirely.
        let mut found_local = false;
        let mut skipped_untrusted_project = false;
        if let Ok(cwd) = std::env::current_dir() {
            let local_path = cwd.join(".cubi").join("hooks.json");
            if local_path.exists() {
                if permissions.contains(&cwd) {
                    if let Some(cfg) = Self::load_file(&local_path) {
                        hooks.extend(cfg.hooks);
                        found_local = true;
                    }
                } else {
                    skipped_untrusted_project = true;
                }
            }
        }

        // Global: ~/.cubi/hooks.json (only used when no local config exists)
        if !found_local {
            if let Some(global_path) = global_hooks_path() {
                if let Some(cfg) = Self::load_file(&global_path) {
                    hooks.extend(cfg.hooks);
                }
            }
        }

        Self {
            hooks,
            skipped_untrusted_project,
        }
    }

    /// Creates a registry with specific hooks (for testing).
    #[cfg(test)]
    pub fn with_hooks(hooks: Vec<HookDef>) -> Self {
        Self {
            hooks,
            skipped_untrusted_project: false,
        }
    }

    /// Creates an empty registry.
    #[allow(dead_code)]
    pub fn empty() -> Self {
        Self {
            hooks: Vec::new(),
            skipped_untrusted_project: false,
        }
    }

    pub fn skipped_untrusted_project(&self) -> bool {
        self.skipped_untrusted_project
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
    pub async fn fire_pre_tool_use(
        &self,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> HookDecision {
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

            match Self::run_hook_command(&hook.command, &env).await {
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
    pub async fn fire_post_tool_use(&self, tool_name: &str, result: &str, is_error: bool) {
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

            if let Err(e) = Self::run_hook_command(&hook.command, &env).await {
                eprintln!("Warning: PostToolUse hook failed: {} ({})", hook.command, e);
            }
        }
    }

    /// Fires SessionStart hooks.
    pub async fn fire_session_start(&self, model: &str) {
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

            if let Err(e) = Self::run_hook_command(&hook.command, &env).await {
                eprintln!(
                    "Warning: SessionStart hook failed: {} ({})",
                    hook.command, e
                );
            }
        }
    }

    /// Fires Stop hooks.
    pub async fn fire_stop(&self) {
        let matching: Vec<&HookDef> = self
            .hooks
            .iter()
            .filter(|h| h.event == HookEvent::Stop)
            .collect();

        for hook in matching {
            let mut env = HashMap::new();
            env.insert("HOOK_EVENT".to_string(), "Stop".to_string());

            if let Err(e) = Self::run_hook_command(&hook.command, &env).await {
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
    async fn run_hook_command(command: &str, env: &HashMap<String, String>) -> Result<i32> {
        Self::run_hook_command_with_timeout(command, env, HOOK_TIMEOUT).await
    }

    async fn run_hook_command_with_timeout(
        command: &str,
        env: &HashMap<String, String>,
        timeout: Duration,
    ) -> Result<i32> {
        let mut child_command = crate::config::async_command("sh");
        child_command
            .arg("-c")
            .arg(command)
            .envs(env)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            child_command.as_std_mut().process_group(0);
        }
        let mut child = child_command.spawn()?;
        let pid = child
            .id()
            .context("hook process exited before its PID was available")?;
        match tokio::time::timeout(timeout, child.wait()).await {
            Ok(status) => Ok(status?.code().unwrap_or(2)),
            Err(_) => {
                if let Err(error) = terminate_hook_process_tree(&mut child, pid).await {
                    return Err(anyhow::anyhow!(
                        "hook timed out after {} ms; process-tree cleanup failed: {}",
                        timeout.as_millis(),
                        error
                    ));
                }
                Err(anyhow::anyhow!(
                    "hook timed out after {} ms",
                    timeout.as_millis()
                ))
            }
        }
    }
}

#[cfg(unix)]
async fn terminate_hook_process_tree(child: &mut tokio::process::Child, pid: u32) -> Result<()> {
    let process_group = i32::try_from(pid).context("hook PID does not fit a Unix process ID")?;
    // SAFETY: `kill` receives the negated PID of the child configured above as a process-group
    // leader. SIGKILL does not dereference Rust memory, and ESRCH is handled as an exit race.
    let result = unsafe { libc::kill(-process_group, libc::SIGKILL) };
    if result == -1 {
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::ESRCH) {
            return Err(error).context("failed to kill hook process group");
        }
    }
    tokio::time::timeout(HOOK_TERMINATION_TIMEOUT, child.wait())
        .await
        .context("timed out waiting for killed hook process group")??;
    Ok(())
}

#[cfg(windows)]
async fn terminate_hook_process_tree(child: &mut tokio::process::Child, pid: u32) -> Result<()> {
    let mut taskkill = crate::config::async_command("taskkill");
    taskkill
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    let status = tokio::time::timeout(HOOK_TERMINATION_TIMEOUT, taskkill.status())
        .await
        .context("timed out running taskkill for hook process tree")??;
    if !status.success() {
        anyhow::bail!("taskkill failed with status {}", status);
    }
    tokio::time::timeout(HOOK_TERMINATION_TIMEOUT, child.wait())
        .await
        .context("timed out waiting for killed hook process tree")??;
    Ok(())
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
    crate::sessions::cubi_dir().map(|d| d.join("hooks.json"))
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
    fn global_hooks_path_uses_cubi_home() {
        crate::compat::test_env::with_cubi_home(|cubi_home, other_home| {
            let path = global_hooks_path().expect("global hooks path");
            assert_eq!(path, cubi_home.join(".cubi").join("hooks.json"));
            assert!(!path.starts_with(other_home));
        });
    }

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

    #[tokio::test]
    async fn fire_pre_tool_use_allows_when_no_hooks() {
        let reg = HookRegistry::empty();
        let decision = reg.fire_pre_tool_use("bash", &serde_json::json!({})).await;
        assert_eq!(decision, HookDecision::Allow);
    }

    #[tokio::test]
    async fn fire_pre_tool_use_allows_on_exit_0() {
        let reg = HookRegistry::with_hooks(vec![HookDef {
            event: HookEvent::PreToolUse,
            match_tool: None,
            command: "true".to_string(), // exit 0
        }]);
        let decision = reg.fire_pre_tool_use("bash", &serde_json::json!({})).await;
        assert_eq!(decision, HookDecision::Allow);
    }

    #[tokio::test]
    async fn fire_pre_tool_use_denies_on_exit_1() {
        let reg = HookRegistry::with_hooks(vec![HookDef {
            event: HookEvent::PreToolUse,
            match_tool: None,
            command: "exit 1".to_string(),
        }]);
        let decision = reg.fire_pre_tool_use("bash", &serde_json::json!({})).await;
        assert!(matches!(decision, HookDecision::Deny(_)));
    }

    #[tokio::test]
    async fn fire_pre_tool_use_skips_non_matching_tools() {
        let reg = HookRegistry::with_hooks(vec![HookDef {
            event: HookEvent::PreToolUse,
            match_tool: Some("bash".to_string()),
            command: "exit 1".to_string(),
        }]);
        // Should allow because the hook only matches "bash", not "write_file"
        let decision = reg
            .fire_pre_tool_use("write_file", &serde_json::json!({}))
            .await;
        assert_eq!(decision, HookDecision::Allow);
    }

    #[tokio::test]
    async fn fire_session_start_runs_without_error() {
        let reg = HookRegistry::with_hooks(vec![HookDef {
            event: HookEvent::SessionStart,
            match_tool: None,
            command: "true".to_string(),
        }]);
        // Should not panic.
        reg.fire_session_start("llama3.2:1b").await;
    }

    #[tokio::test]
    async fn fire_stop_runs_without_error() {
        let reg = HookRegistry::with_hooks(vec![HookDef {
            event: HookEvent::Stop,
            match_tool: None,
            command: "true".to_string(),
        }]);
        reg.fire_stop().await;
    }

    #[tokio::test]
    async fn fire_post_tool_use_runs_without_error() {
        let reg = HookRegistry::with_hooks(vec![HookDef {
            event: HookEvent::PostToolUse,
            match_tool: None,
            command: "true".to_string(),
        }]);
        reg.fire_post_tool_use("bash", "output", false).await;
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

    #[test]
    fn stalled_hook_is_terminated_at_timeout() {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "hooks::tests::stalled_hook_process_tree_helper",
                "--exact",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("CUBI_STALLED_HOOK_HELPER", "1")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            // Captured so a CI failure shows the helper's assertion. Hook
            // descendants get null stdio, so none of them can hold this pipe.
            .stderr(std::process::Stdio::piped());
        configure_outer_process_group(&mut command);
        let child = command.spawn().unwrap();
        let child_pid = child.id();
        let (status_tx, status_rx) = std::sync::mpsc::sync_channel(1);
        std::thread::spawn(move || {
            let _ = status_tx.send(child.wait_with_output());
        });

        match status_rx.recv_timeout(Duration::from_secs(15)) {
            Ok(output) => {
                let output = output.unwrap();
                assert!(
                    output.status.success(),
                    "stalled hook helper failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            Err(error) => {
                terminate_outer_process_tree(child_pid);
                let _ = status_rx.recv_timeout(Duration::from_secs(5));
                panic!("stalled hook test process exceeded outer deadline: {error}");
            }
        }
    }

    #[test]
    #[ignore]
    fn stalled_hook_process_tree_helper() {
        if std::env::var_os("CUBI_STALLED_HOOK_HELPER").is_none() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let pid_file = dir.path().join("grandchild.pid");
        let pid_path = pid_file.to_string_lossy().replace('\\', "/");
        let mut env = HashMap::new();
        env.insert("HOOK_PID_FILE".to_string(), pid_path);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let started = std::time::Instant::now();
        let result = runtime.block_on(HookRegistry::run_hook_command_with_timeout(
            r#"sh -c 'echo $$ > "$HOOK_PID_FILE"; while :; do :; done' & while [ ! -s "$HOOK_PID_FILE" ]; do :; done; while :; do :; done"#,
            &env,
            Duration::from_secs(2),
        ));

        assert!(result.is_err());
        assert!(started.elapsed() < Duration::from_secs(10));
        let grandchild_pid = fs::read_to_string(pid_file)
            .unwrap()
            .trim()
            .parse::<u32>()
            .unwrap();
        assert_process_is_gone(grandchild_pid);
    }

    #[cfg(unix)]
    fn configure_outer_process_group(command: &mut std::process::Command) {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }

    #[cfg(windows)]
    fn configure_outer_process_group(_command: &mut std::process::Command) {}

    #[cfg(unix)]
    fn terminate_outer_process_tree(pid: u32) {
        let pid = i32::try_from(pid).unwrap();
        // SAFETY: `kill` is called with a negated, live child PID that was configured as the
        // process-group leader; SIGKILL does not dereference Rust memory.
        let result = unsafe { libc::kill(-pid, libc::SIGKILL) };
        assert_eq!(result, 0, "failed to kill outer test process group");
    }

    #[cfg(windows)]
    fn terminate_outer_process_tree(pid: u32) {
        let status = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap();
        assert!(status.success(), "failed to kill outer test process tree");
    }

    #[cfg(unix)]
    fn assert_process_is_gone(pid: u32) {
        // The killed grandchild is reparented to init and can remain a zombie
        // until init reaps it, which `kill(pid, 0)` still reports as alive. A
        // zombie has already terminated, so accept "not listed" or state Z.
        let output = std::process::Command::new("ps")
            .args(["-o", "stat=", "-p", &pid.to_string()])
            .output()
            .unwrap();
        let stat = String::from_utf8_lossy(&output.stdout);
        let stat = stat.trim();
        assert!(
            stat.is_empty() || stat.starts_with('Z'),
            "grandchild process {} is still running (ps stat {:?})",
            pid,
            stat
        );
    }

    #[cfg(windows)]
    fn assert_process_is_gone(pid: u32) {
        let output = std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
            .output()
            .unwrap();
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            !stdout.contains(&format!("\"{pid}\"")),
            "grandchild process {pid} is still alive: {stdout}"
        );
    }
}
