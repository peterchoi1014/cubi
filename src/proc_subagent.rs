#![allow(dead_code)]

//! Isolated subprocess runner for consensus subagents.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::Value;
use tokio::io::{AsyncRead, AsyncReadExt};

const CHILD_KILL_GRACE: Duration = Duration::from_secs(5);
const CHILD_COOPERATIVE_TIMEOUT_GRACE: Duration = Duration::from_secs(2);
const STDOUT_DRAIN_GRACE: Duration = Duration::from_secs(5);
const CHILD_STDOUT_LIMIT: usize = 1024 * 1024;
const DIAGNOSTIC_LIMIT: usize = 4 * 1024;
pub(crate) const INTERNAL_SUBAGENT_FLAG: &str = "--internal-subagent";
pub(crate) const INTERNAL_MAX_STEPS_FLAG: &str = "--internal-max-steps";
pub(crate) const INTERNAL_TIME_CAP_MS_FLAG: &str = "--internal-time-cap-ms";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcSubagentResult {
    pub output: String,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    /// Number of `tool_call` events observed in the subprocess JSONL stream.
    pub tool_calls: usize,
    /// Token usage parsed from the subprocess's final `"done"` event.
    /// Zero when the subprocess never emitted `done`.
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Number of agent-loop steps reported by the subprocess JSON stream.
    /// Modern children emit an exact top-level `steps_used` on `done`; legacy
    /// streams fall back to one `tool_call` event per intermediate step plus a
    /// final step for the report.
    pub steps_used: usize,
    /// Stderr is intentionally discarded for isolated subprocesses, so this
    /// remains empty for subprocess-backed subagents.
    pub stderr: String,
    /// Structured failure parsed from `"error"` / `"budget_error"` JSON
    /// events in the subprocess stream.
    pub error: Option<String>,
}

pub fn resolve_cubi_binary() -> PathBuf {
    resolve_cubi_binary_inner(std::env::current_exe().ok(), cargo_bin_from_env())
}

fn resolve_cubi_binary_inner(current_exe: Option<PathBuf>, cargo_bin: Option<PathBuf>) -> PathBuf {
    if let Some(path) = cargo_bin.as_deref().filter(|path| path.is_file()) {
        return path.to_path_buf();
    }

    if let Some(current) = current_exe.as_deref() {
        if is_named_cubi_binary(current) && !is_cargo_test_harness(current) && current.is_file() {
            return current.to_path_buf();
        }
        if is_cargo_test_harness(current) {
            if let Some(path) =
                cubi_binary_next_to_test_harness(current).filter(|path| path.is_file())
            {
                return path;
            }
            return PathBuf::from(cubi_binary_name());
        }
    }

    PathBuf::from(cubi_binary_name())
}

fn cargo_bin_from_env() -> Option<PathBuf> {
    std::env::var_os("CARGO_BIN_EXE_cubi")
        .map(PathBuf::from)
        .or_else(|| option_env!("CARGO_BIN_EXE_cubi").map(PathBuf::from))
        .filter(|path| path.is_file())
}

fn cubi_binary_name() -> &'static str {
    if cfg!(windows) { "cubi.exe" } else { "cubi" }
}

fn is_cargo_test_harness(path: &Path) -> bool {
    let Some(parent_name) = path.parent().and_then(Path::file_name) else {
        return false;
    };
    if parent_name != OsStr::new("deps") {
        return false;
    }

    let Some(name) = file_name_without_exe_suffix(path) else {
        return false;
    };
    name.starts_with("cubi-") && name.len() > "cubi-".len()
}

fn is_named_cubi_binary(path: &Path) -> bool {
    file_name_without_exe_suffix(path) == Some("cubi")
}

fn cubi_binary_next_to_test_harness(current: &Path) -> Option<PathBuf> {
    let deps_dir = current.parent()?;
    let profile_dir = deps_dir.parent()?;
    Some(profile_dir.join(cubi_binary_name()))
}

fn file_name_without_exe_suffix(path: &Path) -> Option<&str> {
    let name = path.file_name()?.to_str()?;
    if cfg!(windows) {
        Some(name.strip_suffix(".exe").unwrap_or(name))
    } else {
        Some(name)
    }
}

pub async fn run_isolated_subagent(
    model: &str,
    goal: &str,
    workdir: &Path,
    time_cap: Duration,
) -> Result<ProcSubagentResult> {
    run_isolated_subagent_with_max_steps(
        model,
        goal,
        workdir,
        time_cap,
        crate::agent_loop::SUBAGENT_DEFAULT_STEPS,
    )
    .await
}

pub(crate) async fn run_isolated_subagent_with_max_steps(
    model: &str,
    goal: &str,
    workdir: &Path,
    time_cap: Duration,
    max_steps: usize,
) -> Result<ProcSubagentResult> {
    let cubi_bin = resolve_cubi_binary();
    run_isolated_subagent_with_binary(&cubi_bin, model, goal, workdir, time_cap, max_steps).await
}

pub(crate) async fn run_isolated_subagent_with_binary(
    cubi_bin: &Path,
    model: &str,
    goal: &str,
    workdir: &Path,
    time_cap: Duration,
    max_steps: usize,
) -> Result<ProcSubagentResult> {
    run_isolated_subagent_with_binary_and_trust_root(
        cubi_bin, model, goal, workdir, workdir, time_cap, max_steps,
    )
    .await
}

pub(crate) async fn run_isolated_subagent_with_binary_and_trust_root(
    cubi_bin: &Path,
    model: &str,
    goal: &str,
    workdir: &Path,
    trusted_root: &Path,
    time_cap: Duration,
    max_steps: usize,
) -> Result<ProcSubagentResult> {
    let home_dir = tempfile::Builder::new()
        .prefix("cubi-subagent-home-")
        .tempdir()
        .context("create isolated home tempdir")?;
    let permissions = current_tool_permission_snapshot();
    seed_trust(home_dir.path(), trusted_root, &permissions).context("seed trusted root")?;
    seed_config(home_dir.path(), model).context("seed non-interactive cubi config")?;
    let max_steps_arg = max_steps.max(1).to_string();
    let time_cap_arg = duration_millis_arg(time_cap);
    let parent_time_cap = time_cap.saturating_add(CHILD_COOPERATIVE_TIMEOUT_GRACE);

    let raw = run_binary_isolated(
        cubi_bin,
        &[
            INTERNAL_SUBAGENT_FLAG,
            INTERNAL_MAX_STEPS_FLAG,
            max_steps_arg.as_str(),
            INTERNAL_TIME_CAP_MS_FLAG,
            time_cap_arg.as_str(),
            "-p",
            goal,
            "--json",
            "--no-stream",
            "--no-banner",
            "--quiet",
        ],
        &[
            ("CUBI_MODEL", model),
            ("CUBI_NO_BANNER", "1"),
            ("CUBI_NO_ONBOARD", "1"),
            ("CUBI_QUIET", "1"),
            ("CUBI_NO_SPINNER", "1"),
            ("CUBI_COLOR", "off"),
            ("CUBI_EXPLAIN_TOOLS", "0"),
            (crate::agent_loop::DISABLE_META_TOOLS_ENV, "1"),
            ("NO_COLOR", "1"),
        ],
        home_dir.path(),
        workdir,
        parent_time_cap,
    )
    .await?;

    Ok(match raw {
        RawRunResult::TimedOut { stdout, stderr } => {
            let parsed = parse_subagent_stream(&stdout);
            ProcSubagentResult {
                output: parsed.output,
                exit_code: None,
                timed_out: true,
                tool_calls: parsed.tool_calls,
                prompt_tokens: parsed.prompt_tokens,
                completion_tokens: parsed.completion_tokens,
                steps_used: parsed.steps_used,
                stderr,
                error: parsed.error,
            }
        }
        RawRunResult::Completed {
            stdout,
            stderr,
            exit_code,
        } => {
            let parsed = parse_subagent_stream(&stdout);
            ProcSubagentResult {
                output: parsed.output,
                exit_code,
                timed_out: false,
                tool_calls: parsed.tool_calls,
                prompt_tokens: parsed.prompt_tokens,
                completion_tokens: parsed.completion_tokens,
                steps_used: parsed.steps_used,
                stderr,
                error: parsed.error,
            }
        }
    })
}

fn duration_millis_arg(duration: Duration) -> String {
    let millis = duration.as_millis().clamp(1, u128::from(u64::MAX));
    millis.to_string()
}

impl ProcSubagentResult {
    pub fn diagnostics(&self) -> String {
        let mut parts = Vec::new();
        if let Some(error) = self
            .error
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            parts.push(error.to_string());
        }
        let stderr = self.stderr.trim();
        if !stderr.is_empty() {
            parts.push(format!("stderr: {stderr}"));
        }
        truncate_diagnostic(&parts.join("; "))
    }
}

#[derive(Debug)]
enum RawRunResult {
    TimedOut {
        stdout: String,
        stderr: String,
    },
    Completed {
        stdout: String,
        stderr: String,
        exit_code: Option<i32>,
    },
}

async fn run_binary_isolated(
    binary: &Path,
    args: &[&str],
    extra_envs: &[(&str, &str)],
    home: &Path,
    workdir: &Path,
    time_cap: Duration,
) -> Result<RawRunResult> {
    let seeded_policy_file = seed_policy(home).context("seed active cubi policy")?;
    let mut cmd = tokio::process::Command::new(binary);
    cmd.args(args)
        .env("HOME", home)
        .env("USERPROFILE", home)
        .env(crate::sessions::CUBI_HOME_ENV, home)
        .env_remove("CUBI_EVENTS")
        .env_remove("CUBI_RECEIPTS")
        .env_remove("CUBI_TRACE_TOOLS")
        .env_remove("CUBI_POLICY_FILE")
        .current_dir(workdir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    cmd.kill_on_drop(true);
    for (key, value) in extra_envs {
        if *key == "CUBI_POLICY_FILE" {
            continue;
        }
        cmd.env(key, value);
    }
    if let Some(policy_file) = seeded_policy_file.as_deref() {
        cmd.env("CUBI_POLICY_FILE", policy_file);
    }

    let mut child = cmd.spawn().with_context(|| {
        format!(
            "spawn subprocess {} in {}",
            binary.display(),
            workdir.display()
        )
    })?;
    let stdout = child.stdout.take().context("capture child stdout")?;
    let stdout_task = tokio::spawn(async move {
        read_bounded(stdout, CHILD_STDOUT_LIMIT, "stdout")
            .await
            .context("read child stdout")
    });
    let status = match tokio::time::timeout(time_cap, child.wait()).await {
        Ok(Ok(status)) => status,
        Ok(Err(e)) => {
            kill_child(&mut child).await;
            abort_reader_task(stdout_task).await;
            return Err(e).with_context(|| format!("wait for subprocess {}", binary.display()));
        }
        Err(_) => {
            kill_child(&mut child).await;
            let stdout = collect_lossy_or_empty(stdout_task, STDOUT_DRAIN_GRACE).await;
            return Ok(RawRunResult::TimedOut {
                stdout,
                stderr: String::new(),
            });
        }
    };

    let stdout = collect_reader(stdout_task, STDOUT_DRAIN_GRACE, "stdout").await?;
    Ok(RawRunResult::Completed {
        stdout: String::from_utf8_lossy(&stdout).into_owned(),
        stderr: String::new(),
        exit_code: status.code(),
    })
}

async fn kill_child(child: &mut tokio::process::Child) {
    let _ = child.start_kill();
    let _ = tokio::time::timeout(CHILD_KILL_GRACE, child.wait()).await;
}

async fn collect_reader(
    mut reader_task: tokio::task::JoinHandle<Result<Vec<u8>>>,
    grace: Duration,
    name: &str,
) -> Result<Vec<u8>> {
    match tokio::time::timeout(grace, &mut reader_task).await {
        Ok(joined) => joined.with_context(|| format!("join {name} reader"))?,
        Err(_) => {
            reader_task.abort();
            let _ = reader_task.await;
            anyhow::bail!("{name} reader did not finish within {:?}", grace);
        }
    }
}

async fn collect_lossy_or_empty(
    mut reader_task: tokio::task::JoinHandle<Result<Vec<u8>>>,
    grace: Duration,
) -> String {
    match tokio::time::timeout(grace, &mut reader_task).await {
        Ok(Ok(Ok(bytes))) => String::from_utf8_lossy(&bytes).into_owned(),
        Ok(Ok(Err(_))) | Ok(Err(_)) | Err(_) => {
            reader_task.abort();
            let _ = reader_task.await;
            String::new()
        }
    }
}

async fn abort_reader_task(reader_task: tokio::task::JoinHandle<Result<Vec<u8>>>) {
    reader_task.abort();
    let _ = reader_task.await;
}

async fn read_bounded<R>(mut reader: R, limit: usize, label: &str) -> Result<Vec<u8>>
where
    R: AsyncRead + Unpin,
{
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let mut truncated = false;
    loop {
        let n = reader.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        if buf.len() < limit {
            let remaining = limit - buf.len();
            let take = remaining.min(n);
            buf.extend_from_slice(&chunk[..take]);
            if take < n {
                truncated = true;
            }
        } else {
            truncated = true;
        }
    }
    if truncated {
        if label == "stdout" {
            buf.extend_from_slice(b"\n{\"type\":\"error\",\"message\":\"stdout truncated\"}\n");
        } else {
            let marker = format!("\n[{label} truncated]\n");
            buf.extend_from_slice(marker.as_bytes());
        }
    }
    Ok(buf)
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ParsedSubagentStream {
    output: String,
    prompt_tokens: u64,
    completion_tokens: u64,
    tool_calls: usize,
    steps_used: usize,
    error: Option<String>,
}

pub fn parse_final_output(stdout: &str) -> String {
    parse_subagent_stream(stdout).output
}

pub fn count_tool_calls(stdout: &str) -> usize {
    parse_subagent_stream(stdout).tool_calls
}

fn parse_subagent_stream(stdout: &str) -> ParsedSubagentStream {
    let mut out = String::new();
    let mut prompt_tokens = 0u64;
    let mut completion_tokens = 0u64;
    let mut tool_calls = 0usize;
    let mut saw_json_event = false;
    let mut explicit_steps_used = None;
    let mut diagnostics = Vec::new();

    for line in stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        saw_json_event = true;
        if let Some(steps) = token_stat(&v, &["steps_used"]) {
            explicit_steps_used = Some(usize::try_from(steps).unwrap_or(usize::MAX));
        }
        match v.get("type").and_then(Value::as_str) {
            Some("token") => {
                if let Some(token) = event_text(&v) {
                    out.push_str(token);
                }
            }
            Some("assistant" | "final" | "output") => {
                if let Some(text) = event_text(&v) {
                    out.clear();
                    out.push_str(text);
                }
            }
            Some("tool_call") => {
                tool_calls += 1;
                out.clear();
            }
            Some("done") => {
                if out.is_empty() {
                    if let Some(text) = event_text(&v) {
                        out.push_str(text);
                    }
                }
                if let Some(value) =
                    token_stat(&v, &["prompt_tokens", "input_tokens", "prompt_eval_count"])
                {
                    prompt_tokens = value;
                }
                if let Some(value) =
                    token_stat(&v, &["completion_tokens", "output_tokens", "eval_count"])
                {
                    completion_tokens = value;
                }
            }
            Some("error") => {
                let message = event_diagnostic_text(&v).unwrap_or("subprocess emitted error event");
                push_diagnostic(&mut diagnostics, format!("error: {message}"));
            }
            Some("budget_error") => {
                push_diagnostic(&mut diagnostics, budget_error_diagnostic(&v));
            }
            _ => {}
        }
    }

    ParsedSubagentStream {
        output: out,
        prompt_tokens,
        completion_tokens,
        tool_calls,
        steps_used: explicit_steps_used
            .unwrap_or_else(|| if saw_json_event { tool_calls + 1 } else { 0 }),
        error: if diagnostics.is_empty() {
            None
        } else {
            Some(truncate_diagnostic(&diagnostics.join("; ")))
        },
    }
}

fn event_text(v: &Value) -> Option<&str> {
    v.get("value")
        .or_else(|| v.get("text"))
        .or_else(|| v.get("output"))
        .or_else(|| v.get("content"))
        .and_then(Value::as_str)
        .or_else(|| {
            v.get("message")
                .and_then(|message| message.get("content"))
                .and_then(Value::as_str)
        })
}

fn event_diagnostic_text(v: &Value) -> Option<&str> {
    event_text(v).or_else(|| v.get("message").and_then(Value::as_str))
}

fn budget_error_diagnostic(v: &Value) -> String {
    let needed = v
        .get("needed")
        .and_then(Value::as_u64)
        .map(|n| n.to_string())
        .unwrap_or_else(|| "?".to_string());
    let window = v
        .get("window")
        .and_then(Value::as_u64)
        .map(|n| n.to_string())
        .unwrap_or_else(|| "?".to_string());
    let model = v.get("model").and_then(Value::as_str).unwrap_or("?");
    format!("budget_error: needed {needed} tokens exceeds window {window} for model `{model}`")
}

fn push_diagnostic(diagnostics: &mut Vec<String>, diagnostic: String) {
    if diagnostics.join("; ").chars().count() >= DIAGNOSTIC_LIMIT {
        return;
    }
    diagnostics.push(truncate_diagnostic(&diagnostic));
}

fn truncate_diagnostic(s: &str) -> String {
    if s.chars().count() <= DIAGNOSTIC_LIMIT {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(DIAGNOSTIC_LIMIT).collect();
        out.push('…');
        out
    }
}

fn token_stat(event: &Value, names: &[&str]) -> Option<u64> {
    event
        .get("stats")
        .into_iter()
        .chain(event.get("usage"))
        .chain(std::iter::once(event))
        .find_map(|stats| {
            names.iter().find_map(|name| {
                stats.get(*name).and_then(|value| {
                    value
                        .as_u64()
                        .or_else(|| value.as_str().and_then(|raw| raw.parse().ok()))
                })
            })
        })
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ToolPermissionSnapshot {
    allowed_tools: Vec<String>,
    denied_tools: Vec<String>,
}

fn current_tool_permission_snapshot() -> ToolPermissionSnapshot {
    let permissions = crate::permissions::Permissions::load();
    ToolPermissionSnapshot {
        allowed_tools: permissions.allowed_tools().cloned().collect(),
        denied_tools: permissions.denied_tools().cloned().collect(),
    }
}

fn seed_trust(
    home: &Path,
    trusted_root: &Path,
    permissions: &ToolPermissionSnapshot,
) -> Result<()> {
    let canonical = std::fs::canonicalize(trusted_root)
        .with_context(|| format!("canonicalize {}", trusted_root.display()))?;
    if !canonical.is_dir() {
        anyhow::bail!("trusted root is not a directory: {}", canonical.display());
    }
    let cubi_dir = home.join(".cubi");
    std::fs::create_dir_all(&cubi_dir).with_context(|| format!("create {}", cubi_dir.display()))?;
    let trust = serde_json::json!({
        "trusted_roots": [canonical],
        "allowed_tools": &permissions.allowed_tools,
        "denied_tools": &permissions.denied_tools,
    });
    let trust_path = cubi_dir.join("trusted_dirs.json");
    let serialized = serde_json::to_string_pretty(&trust).context("serialize trusted_dirs.json")?;
    std::fs::write(&trust_path, serialized)
        .with_context(|| format!("write {}", trust_path.display()))?;
    Ok(())
}

fn seed_config(home: &Path, model: &str) -> Result<()> {
    let cubi_dir = home.join(".cubi");
    std::fs::create_dir_all(&cubi_dir).with_context(|| format!("create {}", cubi_dir.display()))?;
    let config = serde_json::json!({
        "default_model": model,
        "onboarded": true,
        "config_version": 1,
    });
    let config_path = cubi_dir.join("config.json");
    let serialized = serde_json::to_string_pretty(&config).context("serialize config.json")?;
    std::fs::write(&config_path, serialized)
        .with_context(|| format!("write {}", config_path.display()))?;
    Ok(())
}

fn seed_policy(home: &Path) -> Result<Option<PathBuf>> {
    let Some(active_policy) = crate::policy::Policy::active_path() else {
        return Ok(None);
    };

    let cubi_dir = home.join(".cubi");
    std::fs::create_dir_all(&cubi_dir).with_context(|| format!("create {}", cubi_dir.display()))?;
    let policy_bytes = std::fs::read(&active_policy)
        .with_context(|| format!("read active policy {}", active_policy.display()))?;
    let policy_path = cubi_dir.join("policy.json");
    std::fs::write(&policy_path, policy_bytes)
        .with_context(|| format!("write {}", policy_path.display()))?;
    std::fs::canonicalize(&policy_path)
        .with_context(|| format!("canonicalize {}", policy_path.display()))
        .map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvVarGuard {
        key: &'static str,
        old: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn set_path(key: &'static str, value: &Path) -> Self {
            let guard = Self {
                key,
                old: std::env::var_os(key),
            };
            // SAFETY: Tests that mutate process-wide environment variables
            // hold ENV_LOCK until the spawned helper has inherited the desired
            // values and the original value is restored by EnvVarGuard::drop.
            unsafe { std::env::set_var(key, value) };
            guard
        }

        fn remove(key: &'static str) -> Self {
            let guard = Self {
                key,
                old: std::env::var_os(key),
            };
            // SAFETY: See EnvVarGuard::set_path; the same ENV_LOCK guard
            // serializes this temporary process-wide environment mutation.
            unsafe { std::env::remove_var(key) };
            guard
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            // SAFETY: EnvVarGuard values are only used while ENV_LOCK is held,
            // so restoring the previous process-wide value is serialized with
            // the corresponding test mutation.
            unsafe {
                match &self.old {
                    Some(value) => std::env::set_var(self.key, value),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    #[test]
    fn parse_final_output_concatenates_tokens_in_order() {
        let stdout = r#"{"type":"token","value":"hello"}"#.to_string()
            + "\n"
            + r#"{"type":"token","value":" "}"#
            + "\n"
            + r#"{"type":"token","value":"world"}"#;

        assert_eq!(parse_final_output(&stdout), "hello world");
    }

    #[test]
    fn parse_final_output_ignores_non_token_events() {
        let stdout = r#"{"type":"tool_call","name":"grep","arguments":{}}"#.to_string()
            + "\n"
            + r#"{"type":"token","value":"answer"}"#
            + "\n"
            + r#"{"type":"tool_result","name":"grep","content":"ok"}"#
            + "\n"
            + r#"{"type":"done","stats":{}}"#
            + "\n"
            + r#"{"type":"error","message":"ignored"}"#;

        assert_eq!(parse_final_output(&stdout), "answer");
    }

    #[test]
    fn parse_final_output_skips_malformed_and_blank_lines() {
        let stdout = "\n  \nnot-json\n{\"type\":\"token\",\"value\":\"ok\"}\n{";

        assert_eq!(parse_final_output(stdout), "ok");
    }

    #[test]
    fn parse_final_output_empty_input_is_empty() {
        assert_eq!(parse_final_output(""), "");
    }

    #[tokio::test]
    async fn read_bounded_limits_stdout_capture() {
        use tokio::io::AsyncWriteExt;

        let (mut writer, reader) = tokio::io::duplex(16);
        let writer_task = tokio::spawn(async move {
            writer.write_all(b"abcdef").await.unwrap();
        });

        let bytes = read_bounded(reader, 3, "stdout").await.unwrap();
        writer_task.await.unwrap();

        let raw = String::from_utf8_lossy(&bytes);
        assert_eq!(
            raw,
            "abc\n{\"type\":\"error\",\"message\":\"stdout truncated\"}\n"
        );
        let parsed = parse_subagent_stream(&raw);
        assert!(
            parsed
                .error
                .as_deref()
                .is_some_and(|err| err.contains("stdout truncated")),
            "expected structured truncation error, got: {parsed:?}"
        );
    }

    #[test]
    fn parse_subagent_stream_empty_input_has_no_steps() {
        let parsed = parse_subagent_stream("");

        assert_eq!(parsed.output, "");
        assert_eq!(parsed.tool_calls, 0);
        assert_eq!(parsed.steps_used, 0);
    }

    #[test]
    fn parse_subagent_stream_recovers_stats_steps_and_tokens() {
        let stdout = r#"{"type":"token","value":"intermediate rationale"}"#.to_string()
            + "\n"
            + r#"{"type":"tool_call","name":"read_file","arguments":{"path":"a"}}"#
            + "\n"
            + r#"{"type":"tool_result","name":"read_file","content":"ok"}"#
            + "\n"
            + "not-json"
            + "\n"
            + r#"{"type":"unknown","value":"ignored"}"#
            + "\n"
            + r#"{"type":"done","stats":{"prompt_tokens":1,"completion_tokens":2}}"#
            + "\n"
            + r#"{"type":"token","value":"final"}"#
            + "\n"
            + r#"{"type":"tool_call","name":"grep","arguments":{"pattern":"x"}}"#
            + "\n"
            + r#"{"type":"tool_result","name":"grep","content":"ok"}"#
            + "\n"
            + r#"{"type":"token","value":" answer"}"#
            + "\n"
            + r#"{"type":"done","stats":{"prompt_tokens":12,"completion_tokens":34}}"#;

        let parsed = parse_subagent_stream(&stdout);

        assert_eq!(parsed.output, " answer");
        assert_eq!(parsed.prompt_tokens, 12);
        assert_eq!(parsed.completion_tokens, 34);
        assert_eq!(parsed.tool_calls, 2);
        assert_eq!(parsed.steps_used, 3);
    }

    #[test]
    fn parse_subagent_stream_accepts_done_stats_aliases_and_string_counts() {
        let stdout = r#"{"type":"token","text":"done text"}"#.to_string()
            + "\n"
            + r#"{"type":"done","usage":{"prompt_eval_count":"5","eval_count":"8"}}"#;

        let parsed = parse_subagent_stream(&stdout);

        assert_eq!(parsed.output, "done text");
        assert_eq!(parsed.prompt_tokens, 5);
        assert_eq!(parsed.completion_tokens, 8);
        assert_eq!(parsed.steps_used, 1);
    }

    #[test]
    fn parse_subagent_stream_prefers_explicit_done_steps_over_tool_call_heuristic() {
        let stdout = r#"{"type":"tool_call","name":"read_file","arguments":{}}"#.to_string()
            + "\n"
            + r#"{"type":"tool_call","name":"grep","arguments":{}}"#
            + "\n"
            + r#"{"type":"token","value":"done"}"#
            + "\n"
            + r#"{"type":"done","stats":{"prompt_tokens":5,"completion_tokens":8},"steps_used":1}"#;

        let parsed = parse_subagent_stream(&stdout);

        assert_eq!(parsed.output, "done");
        assert_eq!(parsed.tool_calls, 2);
        assert_eq!(parsed.prompt_tokens, 5);
        assert_eq!(parsed.completion_tokens, 8);
        assert_eq!(parsed.steps_used, 1);
    }

    #[test]
    fn parse_subagent_stream_counts_tool_calls_plus_final_step() {
        let stdout = r#"{"type":"tool_call","name":"bash","arguments":{}}"#.to_string()
            + "\n"
            + r#"{"type":"tool_result","name":"bash","content":"ok"}"#
            + "\n"
            + r#"{"type":"tool_timeout","name":"grep","secs":1}"#
            + "\n"
            + r#"{"type":"tool_call","name":"read_file","arguments":{}}"#;

        let parsed = parse_subagent_stream(&stdout);

        assert_eq!(parsed.tool_calls, 2);
        assert_eq!(parsed.steps_used, 3);
    }

    #[test]
    fn parse_subagent_stream_records_structured_errors_and_explicit_steps() {
        let stdout = r#"{"type":"tool_call","name":"bash","arguments":{}}"#.to_string()
            + "\n"
            + r#"{"type":"error","message":"step cap reached","steps_used":1}"#
            + "\n"
            + r#"{"type":"budget_error","needed":99,"window":42,"model":"tiny"}"#
            + "\n"
            + r#"{"type":"final","value":"partial diagnostic","steps_used":1}"#
            + "\n"
            + r#"{"type":"done","stats":{"prompt_tokens":5,"completion_tokens":6}}"#;

        let parsed = parse_subagent_stream(&stdout);

        assert_eq!(parsed.output, "partial diagnostic");
        assert_eq!(parsed.steps_used, 1);
        assert_eq!(parsed.prompt_tokens, 5);
        assert_eq!(parsed.completion_tokens, 6);
        let error = parsed.error.as_deref().unwrap_or_default();
        assert!(error.contains("step cap reached"), "error: {error}");
        assert!(error.contains("budget_error"), "error: {error}");
    }

    #[test]
    fn seed_trust_writes_canonical_workdir_path() {
        let home = tempfile::tempdir().unwrap();
        let workdir = tempfile::tempdir().unwrap();

        seed_trust(
            home.path(),
            workdir.path(),
            &ToolPermissionSnapshot::default(),
        )
        .unwrap();

        let raw =
            std::fs::read_to_string(home.path().join(".cubi").join("trusted_dirs.json")).unwrap();
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let canonical = std::fs::canonicalize(workdir.path()).unwrap();
        let roots = v["trusted_roots"].as_array().unwrap();

        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].as_str().unwrap(), canonical.to_str().unwrap());
        assert_eq!(v["allowed_tools"].as_array().unwrap().len(), 0);
        assert_eq!(v["denied_tools"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn seed_trust_copies_parent_tool_allow_and_deny_lists() {
        let home = tempfile::tempdir().unwrap();
        let workdir = tempfile::tempdir().unwrap();
        let permissions = ToolPermissionSnapshot {
            allowed_tools: vec!["bash".to_string(), "read_file".to_string()],
            denied_tools: vec!["write_file".to_string()],
        };

        seed_trust(home.path(), workdir.path(), &permissions).unwrap();

        let raw =
            std::fs::read_to_string(home.path().join(".cubi").join("trusted_dirs.json")).unwrap();
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let allowed: Vec<&str> = v["allowed_tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect();
        let denied: Vec<&str> = v["denied_tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect();

        assert_eq!(allowed, vec!["bash", "read_file"]);
        assert_eq!(denied, vec!["write_file"]);
    }

    #[test]
    fn seed_config_marks_home_onboarded_with_model() {
        let home = tempfile::tempdir().unwrap();

        seed_config(home.path(), "model-x").unwrap();

        let raw = std::fs::read_to_string(home.path().join(".cubi").join("config.json")).unwrap();
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();

        assert_eq!(v["default_model"], "model-x");
        assert_eq!(v["onboarded"], true);
        assert_eq!(v["config_version"], 1);
    }

    #[test]
    fn seed_policy_copies_active_policy_into_isolated_home() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let source_dir = tempfile::tempdir().unwrap();
        let policy_path = source_dir.path().join("policy.json");
        let policy_json = r#"{"denied_tools":["dangerous-tool"],"note":"parent"}"#;
        std::fs::write(&policy_path, policy_json).unwrap();
        let _policy_file_guard = EnvVarGuard::set_path("CUBI_POLICY_FILE", &policy_path);
        let isolated_home = tempfile::tempdir().unwrap();

        let copied = seed_policy(isolated_home.path()).unwrap().unwrap();

        let expected_path = isolated_home.path().join(".cubi").join("policy.json");
        assert_eq!(copied, std::fs::canonicalize(&expected_path).unwrap());
        assert!(copied.is_absolute());
        assert_ne!(copied, policy_path);
        assert_eq!(std::fs::read_to_string(copied).unwrap(), policy_json);
    }

    #[test]
    fn seed_policy_copies_relative_active_policy_to_absolute_isolated_path() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let _cubi_home_guard = EnvVarGuard::remove(crate::sessions::CUBI_HOME_ENV);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let source_dir = PathBuf::from("target")
            .join("test-proc-subagent-policy")
            .join(format!("relative-{nanos}"));
        std::fs::create_dir_all(&source_dir).unwrap();
        let policy_path = source_dir.join("policy.json");
        assert!(!policy_path.is_absolute());
        let policy_json = r#"{"denied_tools":["relative-tool"],"note":"parent"}"#;
        std::fs::write(&policy_path, policy_json).unwrap();
        let _policy_file_guard = EnvVarGuard::set_path("CUBI_POLICY_FILE", &policy_path);
        let isolated_home = tempfile::tempdir().unwrap();

        let copied = seed_policy(isolated_home.path()).unwrap().unwrap();

        let expected_path = isolated_home.path().join(".cubi").join("policy.json");
        assert_eq!(copied, std::fs::canonicalize(&expected_path).unwrap());
        assert!(copied.is_absolute());
        assert_eq!(std::fs::read_to_string(copied).unwrap(), policy_json);
        std::fs::remove_dir_all(source_dir).ok();
    }

    #[test]
    fn seed_policy_errors_when_active_policy_cannot_be_read() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let unreadable_policy = tempfile::tempdir().unwrap();
        let _policy_file_guard =
            EnvVarGuard::set_path("CUBI_POLICY_FILE", unreadable_policy.path());
        let isolated_home = tempfile::tempdir().unwrap();

        let error = seed_policy(isolated_home.path()).unwrap_err().to_string();

        assert!(
            error.contains("read active policy"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn cubi_binary_next_to_test_harness_points_to_profile_binary() {
        let harness = PathBuf::from("target")
            .join("debug")
            .join("deps")
            .join(if cfg!(windows) {
                "cubi-abc123.exe"
            } else {
                "cubi-abc123"
            });

        assert!(is_cargo_test_harness(&harness));
        assert_eq!(
            cubi_binary_next_to_test_harness(&harness).unwrap(),
            PathBuf::from("target")
                .join("debug")
                .join(cubi_binary_name())
        );
        assert!(!is_cargo_test_harness(
            &PathBuf::from("target")
                .join("debug")
                .join(cubi_binary_name())
        ));
    }

    #[test]
    fn resolve_cubi_binary_inner_prefers_cargo_bin_over_real_current_exe() {
        let dir = tempfile::tempdir().unwrap();
        let current = dir.path().join(cubi_binary_name());
        let cargo_bin = dir.path().join("other-cubi");
        std::fs::write(&current, "").unwrap();
        std::fs::write(&cargo_bin, "").unwrap();

        assert_eq!(
            resolve_cubi_binary_inner(Some(current), Some(cargo_bin.clone())),
            cargo_bin
        );
    }

    #[test]
    fn resolve_cubi_binary_inner_uses_cargo_bin_for_test_harness() {
        let dir = tempfile::tempdir().unwrap();
        let deps_dir = dir.path().join("debug").join("deps");
        std::fs::create_dir_all(&deps_dir).unwrap();
        let harness = deps_dir.join(if cfg!(windows) {
            "cubi-abc123.exe"
        } else {
            "cubi-abc123"
        });
        let cargo_bin = dir.path().join(cubi_binary_name());
        std::fs::write(&harness, "").unwrap();
        std::fs::write(&cargo_bin, "").unwrap();

        assert_eq!(
            resolve_cubi_binary_inner(Some(harness), Some(cargo_bin.clone())),
            cargo_bin
        );
    }

    #[test]
    fn resolve_cubi_binary_inner_maps_test_harness_to_existing_profile_binary() {
        let dir = tempfile::tempdir().unwrap();
        let profile_dir = dir.path().join("debug");
        let deps_dir = profile_dir.join("deps");
        std::fs::create_dir_all(&deps_dir).unwrap();
        let harness = deps_dir.join(if cfg!(windows) {
            "cubi-abc123.exe"
        } else {
            "cubi-abc123"
        });
        let profile_bin = profile_dir.join(cubi_binary_name());
        std::fs::write(&harness, "").unwrap();
        std::fs::write(&profile_bin, "").unwrap();

        assert_eq!(resolve_cubi_binary_inner(Some(harness), None), profile_bin);
    }

    #[test]
    fn resolve_cubi_binary_inner_falls_back_to_platform_name() {
        let dir = tempfile::tempdir().unwrap();
        let harness = dir
            .path()
            .join("debug")
            .join("deps")
            .join(if cfg!(windows) {
                "cubi-abc123.exe"
            } else {
                "cubi-abc123"
            });
        std::fs::create_dir_all(harness.parent().unwrap()).unwrap();
        std::fs::write(&harness, "").unwrap();

        assert_eq!(
            resolve_cubi_binary_inner(Some(harness), None),
            PathBuf::from(cubi_binary_name())
        );
    }

    #[tokio::test]
    async fn run_binary_isolated_returns_timeout_and_partial_stdout() {
        let exe = std::env::current_exe().unwrap();
        let home = tempfile::tempdir().unwrap();
        let workdir = tempfile::tempdir().unwrap();
        let args = helper_args("proc_subagent::tests::timeout_helper_sleeps_when_requested");

        let result = run_binary_isolated(
            &exe,
            &args,
            &[("CUBI_PROC_SUBAGENT_SLEEP_HELPER", "1")],
            home.path(),
            workdir.path(),
            Duration::from_millis(100),
        )
        .await
        .unwrap();

        let RawRunResult::TimedOut { stdout, .. } = result else {
            panic!("helper did not time out");
        };
        assert!(stdout.contains("sleep-helper-started"), "stdout:\n{stdout}");
    }

    #[tokio::test]
    async fn run_binary_isolated_sets_env_home_userprofile_and_workdir() {
        let exe = std::env::current_exe().unwrap();
        let home = tempfile::tempdir().unwrap();
        let parent_home = tempfile::tempdir().unwrap();
        let workdir = tempfile::tempdir().unwrap();
        let home_s = home.path().to_string_lossy().into_owned();
        let parent_home_s = parent_home.path().to_string_lossy().into_owned();
        let workdir_s = std::fs::canonicalize(workdir.path())
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let args = helper_args("proc_subagent::tests::env_helper_checks_isolated_process_context");
        let envs = [
            ("CUBI_PROC_SUBAGENT_ENV_HELPER", "1"),
            ("CUBI_PROC_SUBAGENT_EXPECTED_HOME", home_s.as_str()),
            ("CUBI_PROC_SUBAGENT_PARENT_HOME", parent_home_s.as_str()),
            ("CUBI_PROC_SUBAGENT_EXPECTED_WORKDIR", workdir_s.as_str()),
            ("CUBI_MODEL", "test-model"),
            ("CUBI_NO_BANNER", "1"),
            ("CUBI_NO_ONBOARD", "1"),
        ];

        let result = run_binary_isolated(
            &exe,
            &args,
            &envs,
            home.path(),
            workdir.path(),
            Duration::from_secs(10),
        )
        .await
        .unwrap();

        let RawRunResult::Completed {
            stdout, exit_code, ..
        } = result
        else {
            panic!("helper timed out");
        };
        assert_eq!(exit_code, Some(0), "stdout:\n{stdout}");
        assert!(
            stdout.contains("env-helper-ok"),
            "helper stdout did not include marker:\n{stdout}"
        );
        assert!(
            !parent_home
                .path()
                .join(".cubi")
                .join("trusted_dirs.json")
                .exists(),
            "isolated child must not write Cubi trust state under parent home"
        );
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn run_binary_isolated_seeds_active_policy_file_while_isolating_home() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let _cubi_home_guard = EnvVarGuard::remove(crate::sessions::CUBI_HOME_ENV);
        let parent_home = tempfile::tempdir().unwrap();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let policy_dir = PathBuf::from("target")
            .join("test-proc-subagent-policy")
            .join(format!("child-env-{nanos}"));
        std::fs::create_dir_all(&policy_dir).unwrap();
        let policy_path = policy_dir.join("policy.json");
        assert!(!policy_path.is_absolute());
        let policy_json = r#"{"denied_tools":["dangerous-tool"]}"#;
        std::fs::write(&policy_path, policy_json).unwrap();
        let _policy_file_guard = EnvVarGuard::set_path("CUBI_POLICY_FILE", &policy_path);
        let _home_guard = EnvVarGuard::set_path("HOME", parent_home.path());
        let _userprofile_guard = EnvVarGuard::set_path("USERPROFILE", parent_home.path());

        let exe = std::env::current_exe().unwrap();
        let isolated_home = tempfile::tempdir().unwrap();
        let workdir = tempfile::tempdir().unwrap();
        let isolated_home_s = isolated_home.path().to_string_lossy().into_owned();
        let seeded_policy = isolated_home.path().join(".cubi").join("policy.json");
        let expected_seeded_policy = isolated_home
            .path()
            .canonicalize()
            .unwrap()
            .join(".cubi")
            .join("policy.json");
        let seeded_policy_s = expected_seeded_policy.to_string_lossy().into_owned();
        let bogus_policy_s = parent_home
            .path()
            .join("missing-policy.json")
            .to_string_lossy()
            .into_owned();
        let args =
            helper_args("proc_subagent::tests::policy_env_helper_checks_forwarded_policy_file");
        let envs = [
            ("CUBI_PROC_SUBAGENT_POLICY_ENV_HELPER", "1"),
            ("CUBI_POLICY_FILE", bogus_policy_s.as_str()),
            (
                "CUBI_PROC_SUBAGENT_EXPECTED_POLICY_FILE",
                seeded_policy_s.as_str(),
            ),
            ("CUBI_PROC_SUBAGENT_EXPECTED_HOME", isolated_home_s.as_str()),
        ];

        let result = run_binary_isolated(
            &exe,
            &args,
            &envs,
            isolated_home.path(),
            workdir.path(),
            Duration::from_secs(10),
        )
        .await
        .unwrap();

        let RawRunResult::Completed {
            stdout, exit_code, ..
        } = result
        else {
            panic!("helper timed out");
        };
        assert_eq!(exit_code, Some(0), "stdout:\n{stdout}");
        assert!(
            stdout.contains("policy-env-helper-ok"),
            "helper stdout did not include marker:\n{stdout}"
        );
        assert_eq!(std::fs::read_to_string(seeded_policy).unwrap(), policy_json);
        std::fs::remove_dir_all(policy_dir).ok();
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn run_binary_isolated_drops_unseeded_policy_env_from_child() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let _cubi_home_guard = EnvVarGuard::remove(crate::sessions::CUBI_HOME_ENV);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let missing_policy = PathBuf::from("target")
            .join("test-proc-subagent-policy")
            .join(format!("missing-{nanos}"))
            .join("policy.json");
        assert!(!missing_policy.exists());
        let _policy_file_guard = EnvVarGuard::set_path("CUBI_POLICY_FILE", &missing_policy);

        let exe = std::env::current_exe().unwrap();
        let isolated_home = tempfile::tempdir().unwrap();
        let workdir = tempfile::tempdir().unwrap();
        let workdir_policy = workdir.path().join(&missing_policy);
        std::fs::create_dir_all(workdir_policy.parent().unwrap()).unwrap();
        std::fs::write(&workdir_policy, r#"{"denied_tools":["workdir-leak"]}"#).unwrap();
        let missing_policy_s = missing_policy.to_string_lossy().into_owned();
        let args = helper_args("proc_subagent::tests::policy_env_helper_checks_policy_file_absent");
        let envs = [
            ("CUBI_PROC_SUBAGENT_POLICY_ABSENT_HELPER", "1"),
            ("CUBI_POLICY_FILE", missing_policy_s.as_str()),
        ];

        let result = run_binary_isolated(
            &exe,
            &args,
            &envs,
            isolated_home.path(),
            workdir.path(),
            Duration::from_secs(10),
        )
        .await
        .unwrap();

        let RawRunResult::Completed {
            stdout, exit_code, ..
        } = result
        else {
            panic!("helper timed out");
        };
        assert_eq!(exit_code, Some(0), "stdout:\n{stdout}");
        assert!(
            stdout.contains("policy-env-absent-helper-ok"),
            "helper stdout did not include marker:\n{stdout}"
        );
    }

    #[tokio::test]
    async fn run_binary_isolated_drains_large_stdout_without_deadlock() {
        let exe = std::env::current_exe().unwrap();
        let home = tempfile::tempdir().unwrap();
        let workdir = tempfile::tempdir().unwrap();
        let args = helper_args("proc_subagent::tests::large_stdout_helper_writes_and_exits");

        let result = run_binary_isolated(
            &exe,
            &args,
            &[("CUBI_PROC_SUBAGENT_LARGE_STDOUT_HELPER", "1")],
            home.path(),
            workdir.path(),
            Duration::from_secs(10),
        )
        .await
        .unwrap();

        let RawRunResult::Completed {
            stdout, exit_code, ..
        } = result
        else {
            panic!("helper timed out");
        };
        assert_eq!(exit_code, Some(0), "stdout length {}", stdout.len());
        assert!(stdout.contains("large-stdout-helper-end"));
    }

    fn helper_args(test_name: &'static str) -> [&'static str; 5] {
        [
            test_name,
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ]
    }

    #[test]
    #[ignore]
    fn timeout_helper_sleeps_when_requested() {
        if std::env::var_os("CUBI_PROC_SUBAGENT_SLEEP_HELPER").is_none() {
            return;
        }
        use std::io::Write;

        println!("sleep-helper-started");
        std::io::stdout().flush().unwrap();
        std::thread::sleep(Duration::from_secs(30));
    }

    #[test]
    #[ignore]
    fn env_helper_checks_isolated_process_context() {
        if std::env::var_os("CUBI_PROC_SUBAGENT_ENV_HELPER").is_none() {
            return;
        }

        let expected_home = std::env::var("CUBI_PROC_SUBAGENT_EXPECTED_HOME").unwrap();
        let expected_workdir =
            std::fs::canonicalize(std::env::var("CUBI_PROC_SUBAGENT_EXPECTED_WORKDIR").unwrap())
                .unwrap();
        assert_eq!(std::env::var("HOME").unwrap(), expected_home);
        assert_eq!(std::env::var("USERPROFILE").unwrap(), expected_home);
        assert_eq!(
            std::env::var(crate::sessions::CUBI_HOME_ENV).unwrap(),
            expected_home
        );
        assert_eq!(std::env::var("CUBI_MODEL").unwrap(), "test-model");
        assert_eq!(std::env::var("CUBI_NO_BANNER").unwrap(), "1");
        assert_eq!(std::env::var("CUBI_NO_ONBOARD").unwrap(), "1");

        let cwd = std::env::current_dir().unwrap();
        assert_eq!(std::fs::canonicalize(cwd).unwrap(), expected_workdir);
        if let Ok(parent_home) = std::env::var("CUBI_PROC_SUBAGENT_PARENT_HOME") {
            let mut permissions = crate::permissions::Permissions::default();
            permissions.allow_tool("isolated-helper-tool");
            permissions.save().unwrap();

            let isolated_trust = std::path::PathBuf::from(&expected_home)
                .join(".cubi")
                .join("trusted_dirs.json");
            assert!(
                std::fs::read_to_string(&isolated_trust)
                    .unwrap()
                    .contains("isolated-helper-tool"),
                "helper did not write trust state under isolated CUBI_HOME"
            );
            let parent_trust = std::path::PathBuf::from(parent_home)
                .join(".cubi")
                .join("trusted_dirs.json");
            assert!(
                !parent_trust.exists(),
                "helper wrote trust state under parent home: {}",
                parent_trust.display()
            );
        }
        println!("env-helper-ok");
    }

    #[test]
    #[ignore]
    fn policy_env_helper_checks_forwarded_policy_file() {
        if std::env::var_os("CUBI_PROC_SUBAGENT_POLICY_ENV_HELPER").is_none() {
            return;
        }

        let expected_policy = std::env::var("CUBI_PROC_SUBAGENT_EXPECTED_POLICY_FILE").unwrap();
        let expected_home = std::env::var("CUBI_PROC_SUBAGENT_EXPECTED_HOME").unwrap();
        let policy_file = std::env::var("CUBI_POLICY_FILE").unwrap();
        assert_eq!(policy_file, expected_policy);
        assert!(
            PathBuf::from(&policy_file).is_absolute(),
            "child CUBI_POLICY_FILE must be absolute, got {policy_file}"
        );
        assert_eq!(std::env::var("HOME").unwrap(), expected_home);
        assert_eq!(std::env::var("USERPROFILE").unwrap(), expected_home);
        assert_eq!(
            std::env::var(crate::sessions::CUBI_HOME_ENV).unwrap(),
            expected_home
        );
        println!("policy-env-helper-ok");
    }

    #[test]
    #[ignore]
    fn policy_env_helper_checks_policy_file_absent() {
        if std::env::var_os("CUBI_PROC_SUBAGENT_POLICY_ABSENT_HELPER").is_none() {
            return;
        }

        assert!(
            std::env::var_os("CUBI_POLICY_FILE").is_none(),
            "child inherited CUBI_POLICY_FILE={:?}",
            std::env::var_os("CUBI_POLICY_FILE")
        );
        println!("policy-env-absent-helper-ok");
    }

    #[test]
    #[ignore]
    fn large_stdout_helper_writes_and_exits() {
        if std::env::var_os("CUBI_PROC_SUBAGENT_LARGE_STDOUT_HELPER").is_none() {
            return;
        }
        use std::io::Write;

        let payload = vec![b'x'; 256 * 1024];
        std::io::stdout().write_all(&payload).unwrap();
        println!("large-stdout-helper-end");
        std::io::stdout().flush().unwrap();
    }

    /// Writes a tiny `/bin/sh` script that joins its argv with `|||` and
    /// emits it as a single JSON `assistant` event, so `parse_subagent_stream`
    /// hands it straight back as `ProcSubagentResult::output`. Standing in
    /// for the real `cubi` binary, this lets us prove — through the actual
    /// `run_isolated_subagent_with_binary`/`run_binary_isolated` spawn path,
    /// not a stubbed double — exactly which argv a caller's `max_steps`
    /// turns into on the wire.
    #[cfg(unix)]
    fn write_argv_echo_script(dir: &Path) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let path = dir.join("argv-echo.sh");
        let script = "#!/bin/sh\n\
             out=\"\"\n\
             first=1\n\
             for a in \"$@\"; do\n\
             \x20\x20if [ \"$first\" -eq 0 ]; then out=\"$out|||\"; fi\n\
             \x20\x20first=0\n\
             \x20\x20out=\"$out$a\"\n\
             done\n\
             printf '{\"type\":\"assistant\",\"value\":\"%s\"}\\n' \"$out\"\n";
        std::fs::write(&path, script).unwrap();
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).unwrap();
        path
    }

    #[cfg(unix)]
    fn write_json_error_script(dir: &Path) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let path = dir.join("json-error.sh");
        let script = "#!/bin/sh\n\
             printf 'child stderr diagnostic\\n' >&2\n\
             printf '{\"type\":\"tool_call\",\"name\":\"bash\",\"arguments\":{}}\\n'\n\
             printf '{\"type\":\"error\",\"message\":\"structured failure\",\"steps_used\":1}\\n'\n\
             printf '{\"type\":\"done\",\"stats\":{\"prompt_tokens\":3,\"completion_tokens\":4}}\\n'\n\
             exit 7\n";
        std::fs::write(&path, script).unwrap();
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).unwrap();
        path
    }

    #[cfg(unix)]
    fn write_context_echo_script(dir: &Path) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let path = dir.join("context-echo.sh");
        let script = "#!/bin/sh\n\
             cwd=$(pwd)\n\
             root=$(dirname \"$cwd\")\n\
             trust=$(tr -d '\\n' < \"$CUBI_HOME/.cubi/trusted_dirs.json\")\n\
             root_needle=\"\\\"$root\\\"\"\n\
             cwd_needle=\"\\\"$cwd\\\"\"\n\
             case \"$trust\" in\n\
             \x20\x20*\"$root_needle\"*) marker=root ;;\n\
             \x20\x20*\"$cwd_needle\"*) marker=cwd ;;\n\
             \x20\x20*) marker=missing ;;\n\
             esac\n\
             printf '{\"type\":\"assistant\",\"value\":\"cwd=%s trust_marker=%s\"}\\n' \"$cwd\" \"$marker\"\n";
        std::fs::write(&path, script).unwrap();
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).unwrap();
        path
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_isolated_subagent_with_binary_nulls_stderr_and_json_error() {
        let bin_dir = tempfile::tempdir().unwrap();
        let fake_bin = write_json_error_script(bin_dir.path());
        let workdir = tempfile::tempdir().unwrap();

        let result = run_isolated_subagent_with_binary(
            &fake_bin,
            "test-model",
            "the goal",
            workdir.path(),
            Duration::from_secs(10),
            3,
        )
        .await
        .unwrap();

        assert_eq!(result.exit_code, Some(7));
        assert_eq!(result.steps_used, 1);
        assert_eq!(result.prompt_tokens, 3);
        assert_eq!(result.completion_tokens, 4);
        assert!(
            result
                .error
                .as_deref()
                .is_some_and(|error| error.contains("structured failure")),
            "structured error missing: {:?}",
            result.error
        );
        assert_eq!(result.stderr, "");
        assert!(
            !result.diagnostics().contains("child stderr diagnostic"),
            "diagnostics should not include nulled stderr: {}",
            result.diagnostics()
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_isolated_subagent_with_binary_forwards_caps_into_child_argv() {
        let bin_dir = tempfile::tempdir().unwrap();
        let fake_bin = write_argv_echo_script(bin_dir.path());
        let workdir = tempfile::tempdir().unwrap();

        let result = run_isolated_subagent_with_binary(
            &fake_bin,
            "test-model",
            "the goal",
            workdir.path(),
            Duration::from_secs(10),
            3,
        )
        .await
        .unwrap();

        let argv: Vec<&str> = result.output.split("|||").collect();
        assert_eq!(
            argv,
            vec![
                INTERNAL_SUBAGENT_FLAG,
                INTERNAL_MAX_STEPS_FLAG,
                "3",
                INTERNAL_TIME_CAP_MS_FLAG,
                "10000",
                "-p",
                "the goal",
                "--json",
                "--no-stream",
                "--no-banner",
                "--quiet",
            ],
            "run_isolated_subagent_with_binary must forward the requested \
             max_steps and time_cap to the spawned child"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_isolated_subagent_with_binary_clamps_zero_max_steps_before_child_argv() {
        let bin_dir = tempfile::tempdir().unwrap();
        let fake_bin = write_argv_echo_script(bin_dir.path());
        let workdir = tempfile::tempdir().unwrap();

        let result = run_isolated_subagent_with_binary(
            &fake_bin,
            "test-model",
            "the goal",
            workdir.path(),
            Duration::from_secs(10),
            0,
        )
        .await
        .unwrap();

        let argv: Vec<&str> = result.output.split("|||").collect();
        let flag_index = argv
            .iter()
            .position(|arg| *arg == INTERNAL_MAX_STEPS_FLAG)
            .expect("missing max steps flag in child argv");
        assert_eq!(
            argv.get(flag_index + 1),
            Some(&"1"),
            "zero max_steps must be clamped before spawning child: {argv:?}"
        );
        assert!(
            !argv
                .windows(2)
                .any(|window| window == [INTERNAL_MAX_STEPS_FLAG, "0"]),
            "child argv must not receive {INTERNAL_MAX_STEPS_FLAG} 0: {argv:?}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_isolated_subagent_with_binary_can_trust_root_separate_from_execution_cwd() {
        let bin_dir = tempfile::tempdir().unwrap();
        let fake_bin = write_context_echo_script(bin_dir.path());
        let worktree_root = tempfile::tempdir().unwrap();
        let child_workdir = worktree_root.path().join("nested");
        std::fs::create_dir_all(&child_workdir).unwrap();

        let result = run_isolated_subagent_with_binary_and_trust_root(
            &fake_bin,
            "test-model",
            "the goal",
            &child_workdir,
            worktree_root.path(),
            Duration::from_secs(10),
            3,
        )
        .await
        .unwrap();

        let canonical_workdir = std::fs::canonicalize(&child_workdir)
            .unwrap()
            .to_string_lossy()
            .into_owned();
        assert!(
            result.output.contains(&format!("cwd={canonical_workdir}")),
            "child did not execute in requested cwd: {}",
            result.output
        );
        assert!(
            result.output.contains("trust_marker=root"),
            "child trust store did not trust the separate root: {}",
            result.output
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_isolated_subagent_public_api_defaults_to_subagent_default_steps() {
        let bin_dir = tempfile::tempdir().unwrap();
        let fake_bin = write_argv_echo_script(bin_dir.path());
        let workdir = tempfile::tempdir().unwrap();

        // The old two-arg-plus-caps public API (no explicit max_steps)
        // must still default to `SUBAGENT_DEFAULT_STEPS` rather than
        // silently dropping the cap.
        let result = run_isolated_subagent_with_binary(
            &fake_bin,
            "test-model",
            "goal text",
            workdir.path(),
            Duration::from_secs(10),
            crate::agent_loop::SUBAGENT_DEFAULT_STEPS,
        )
        .await
        .unwrap();

        let argv: Vec<&str> = result.output.split("|||").collect();
        assert_eq!(
            argv[1], INTERNAL_MAX_STEPS_FLAG,
            "expected {INTERNAL_MAX_STEPS_FLAG} flag at argv[1]"
        );
        assert_eq!(
            argv[2],
            crate::agent_loop::SUBAGENT_DEFAULT_STEPS.to_string(),
            "default max_steps must equal SUBAGENT_DEFAULT_STEPS"
        );
        assert_eq!(
            argv[3], INTERNAL_TIME_CAP_MS_FLAG,
            "expected {INTERNAL_TIME_CAP_MS_FLAG} flag at argv[3]"
        );
    }
}
