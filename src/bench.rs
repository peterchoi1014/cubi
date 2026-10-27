//! `cubi bench` — curated regression suite runner.
//!
//! Reads task definitions from `bench/tasks/<id>/`, copies each task's
//! starting repo into a temp dir, drives the Cubi binary headlessly as
//! a subprocess, then runs `verify.sh` to score the result. Aggregates
//! per-task results into a `summary.json` so CI (and humans) can track
//! pass rates against a local model over time.
//!
//! SWE-bench-Lite integration is explicitly out of scope; this module
//! exists so the Cubi project has a reproducible, network-free
//! regression metric that runs nightly against `qwen3.5:4b`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt};

const DIAGNOSTIC_TAIL_BYTES: usize = 4 * 1024;
const DEFAULT_VERIFY_TIMEOUT_SECONDS: u64 = 120;

/// Parsed `task.toml` describing one bench task.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct TaskDef {
    pub id: String,
    pub difficulty: Difficulty,
    pub description: String,
    pub prompt: String,
    pub time_cap_seconds: u64,
    pub step_cap: u32,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
}

/// Outcome of running a single task. Recorded to disk (one per task)
/// and aggregated into the summary.
#[derive(Debug, Clone, Serialize)]
pub struct TaskResult {
    pub task_id: String,
    pub model: String,
    pub status: TaskStatus,
    pub elapsed_seconds: f64,
    pub steps_used: Option<u32>,
    pub verify_exit_code: Option<i32>,
    pub cubi_exit_code: Option<i32>,
    pub tokens_in: Option<u64>,
    pub tokens_out: Option<u64>,
    pub events_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_stderr_tail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verifier_stdout_tail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verifier_stderr_tail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TaskStatus {
    Pass,
    Fail,
    Timeout,
    Error,
}

#[derive(Debug)]
pub(crate) struct CapturedProcess {
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub stdout_tail: String,
    pub stderr_tail: String,
}

pub(crate) fn configure_isolated_child_env(command: &mut tokio::process::Command, home: &Path) {
    command
        .env("HOME", home)
        .env("USERPROFILE", home)
        .env(crate::sessions::CUBI_HOME_ENV, home)
        .env_remove("CUBI_EVENTS")
        .env_remove("CUBI_RECEIPTS")
        .env_remove("CUBI_TRACE_TOOLS")
        .env_remove("CUBI_POLICY_FILE");

    let parent_home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from);
    match std::env::var_os("RUSTUP_HOME").or_else(|| {
        parent_home
            .as_ref()
            .map(|p| p.join(".rustup").into_os_string())
    }) {
        Some(value) => {
            command.env("RUSTUP_HOME", value);
        }
        None => {
            command.env_remove("RUSTUP_HOME");
        }
    }
    match std::env::var_os("CARGO_HOME").or_else(|| {
        parent_home
            .as_ref()
            .map(|p| p.join(".cargo").into_os_string())
    }) {
        Some(value) => {
            command.env("CARGO_HOME", value);
        }
        None => {
            command.env_remove("CARGO_HOME");
        }
    }
    match std::env::var_os("RUSTUP_TOOLCHAIN") {
        Some(value) => {
            command.env("RUSTUP_TOOLCHAIN", value);
        }
        None => {
            command.env_remove("RUSTUP_TOOLCHAIN");
        }
    }
    // Every bench/swebench task is a copy of the same crate, so a shared
    // parent CARGO_TARGET_DIR lets one task reuse another task's build
    // artifacts (e.g. a negative control picking up a fixed library).
    // Give each isolated home its own target directory instead.
    command.env("CARGO_TARGET_DIR", home.join("cargo-target"));
}

pub(crate) async fn capture_child(
    mut child: tokio::process::Child,
    timeout: Duration,
) -> std::io::Result<CapturedProcess> {
    let stdout_task = child
        .stdout
        .take()
        .map(|stdout| tokio::spawn(read_tail(stdout)));
    let stderr_task = child
        .stderr
        .take()
        .map(|stderr| tokio::spawn(read_tail(stderr)));

    let (exit_code, timed_out) = match tokio::time::timeout(timeout, child.wait()).await {
        Ok(Ok(status)) => (status.code(), false),
        Ok(Err(error)) => {
            let _ = child.start_kill();
            return Err(error);
        }
        Err(_) => {
            let _ = child.start_kill();
            let _ = child.wait().await;
            (None, true)
        }
    };

    Ok(CapturedProcess {
        exit_code,
        timed_out,
        stdout_tail: join_tail(stdout_task).await?,
        stderr_tail: join_tail(stderr_task).await?,
    })
}

async fn read_tail<R>(mut reader: R) -> std::io::Result<Vec<u8>>
where
    R: AsyncRead + Unpin,
{
    let mut tail = Vec::with_capacity(DIAGNOSTIC_TAIL_BYTES);
    let mut chunk = [0_u8; 1024];
    loop {
        let count = reader.read(&mut chunk).await?;
        if count == 0 {
            break;
        }
        tail.extend_from_slice(&chunk[..count]);
        if tail.len() > DIAGNOSTIC_TAIL_BYTES {
            let excess = tail.len() - DIAGNOSTIC_TAIL_BYTES;
            tail.drain(..excess);
        }
    }
    Ok(tail)
}

async fn join_tail(
    task: Option<tokio::task::JoinHandle<std::io::Result<Vec<u8>>>>,
) -> std::io::Result<String> {
    let Some(task) = task else {
        return Ok(String::new());
    };
    let bytes = task
        .await
        .map_err(|error| std::io::Error::other(format!("diagnostic reader failed: {error}")))??;
    Ok(bound_diagnostic(&String::from_utf8_lossy(&bytes)))
}

fn bound_diagnostic(value: &str) -> String {
    if value.len() <= DIAGNOSTIC_TAIL_BYTES {
        return value.to_string();
    }
    let mut start = value.len() - DIAGNOSTIC_TAIL_BYTES;
    while !value.is_char_boundary(start) {
        start += 1;
    }
    value[start..].to_string()
}

fn diagnostic(value: String) -> Option<String> {
    (!value.is_empty()).then_some(value)
}

fn verifier_timeout() -> Duration {
    let seconds = std::env::var("CUBI_BENCH_VERIFY_TIMEOUT_SECS")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .unwrap_or(DEFAULT_VERIFY_TIMEOUT_SECONDS);
    Duration::from_secs(seconds)
}

pub(crate) fn infrastructure_exit(exit_code: Option<i32>) -> bool {
    matches!(exit_code, Some(2 | 10 | 13))
}

/// Aggregated summary written to `<output>/summary.json`. Stable
/// schema; CI parses these files to track regressions.
#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub started_at: String,
    pub model: String,
    pub suite: String,
    pub n_tasks: usize,
    pub n_pass: usize,
    pub score_pct: f64,
    pub wall_time_seconds: f64,
    pub p50_elapsed: f64,
    pub p90_elapsed: f64,
    pub total_tokens_in: u64,
    pub total_tokens_out: u64,
    pub per_task: Vec<TaskResult>,
}

/// CLI arguments parsed in `main.rs`.
#[derive(Debug, Clone)]
pub struct BenchArgs {
    pub suite: Suite,
    pub model: Option<String>,
    pub task: Option<String>,
    pub output: Option<PathBuf>,
    pub step_cap: Option<u32>,
    pub keep_workdir: bool,
    pub json: bool,
    /// Override `bench/tasks/` root (defaults to `<cwd>/bench/tasks`).
    pub tasks_root: Option<PathBuf>,
    /// Scales every task's `time_cap_seconds` by this factor. Task
    /// definitions state caps that suit GPU-class inference; CPU-only hosts
    /// (notably GitHub-hosted CI runners) need several times longer for the
    /// same work, and without headroom every task merely times out. Scaling
    /// keeps the relative per-task budgets intact rather than flattening
    /// them to one number.
    pub time_cap_multiplier: Option<f64>,
}

/// Applies `multiplier` to a task's configured cap, rounding up so the result
/// is never *below* the requested scale (a multiplier < 1 does shorten the cap
/// — that is the caller's intent — but rounding never shortens it further).
/// Falls back to the unscaled cap when the multiplier is absent or not a
/// usable positive number, and never returns 0 (a zero cap would time every
/// task out instantly).
fn scaled_time_cap(base_seconds: u64, multiplier: Option<f64>) -> u64 {
    let Some(m) = multiplier.filter(|m| m.is_finite() && *m > 0.0) else {
        return base_seconds;
    };
    let scaled = (base_seconds as f64 * m).ceil();
    // Saturate rather than wrap on an absurd multiplier.
    if scaled >= u64::MAX as f64 {
        return u64::MAX;
    }
    (scaled as u64).max(1)
}

#[derive(Debug, Clone, Copy)]
pub enum Suite {
    Quick,
    All,
}

impl Suite {
    pub fn parse(s: &str) -> Option<Suite> {
        match s {
            "quick" => Some(Suite::Quick),
            "all" => Some(Suite::All),
            _ => None,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Suite::Quick => "quick",
            Suite::All => "all",
        }
    }
}

impl Default for BenchArgs {
    fn default() -> Self {
        Self {
            suite: Suite::Quick,
            model: None,
            task: None,
            output: None,
            step_cap: None,
            keep_workdir: false,
            json: false,
            tasks_root: None,
            time_cap_multiplier: None,
        }
    }
}

/// Discover tasks under `tasks_root`, ordered by task id. Each
/// immediate subdirectory containing a `task.toml` is loaded.
pub fn discover_tasks(tasks_root: &Path) -> Result<Vec<TaskDef>> {
    if !tasks_root.exists() {
        bail!(
            "bench tasks directory not found at {}. Run from the repo root, \
             or pass `--tasks-root <dir>`.",
            tasks_root.display()
        );
    }
    let mut out: BTreeMap<String, TaskDef> = BTreeMap::new();
    for entry in std::fs::read_dir(tasks_root)
        .with_context(|| format!("read_dir {}", tasks_root.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let toml_path = path.join("task.toml");
        if !toml_path.exists() {
            continue;
        }
        let raw = std::fs::read_to_string(&toml_path)
            .with_context(|| format!("read {}", toml_path.display()))?;
        let def: TaskDef =
            toml::from_str(&raw).with_context(|| format!("parse {}", toml_path.display()))?;
        let dir_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        if def.id != dir_name {
            bail!(
                "task id `{}` does not match directory name `{}` at {}",
                def.id,
                dir_name,
                toml_path.display()
            );
        }
        out.insert(def.id.clone(), def);
    }
    Ok(out.into_values().collect())
}

/// Filter discovered tasks by suite (`quick` = easy only) and an
/// optional single-task override (for `--task <id>`).
pub fn select_tasks(all: Vec<TaskDef>, suite: Suite, only: Option<&str>) -> Result<Vec<TaskDef>> {
    if let Some(id) = only {
        let picked = all.into_iter().find(|t| t.id == id);
        return picked
            .map(|t| vec![t])
            .ok_or_else(|| anyhow!("no bench task with id `{}`", id));
    }
    let filtered: Vec<TaskDef> = match suite {
        Suite::Quick => all
            .into_iter()
            .filter(|t| t.difficulty == Difficulty::Easy)
            .collect(),
        Suite::All => all,
    };
    Ok(filtered)
}

/// Compute the percentile (0.0..=1.0) of a sorted slice of floats
/// using linear interpolation. Returns 0.0 for empty input. Used for
/// p50/p90 elapsed-time stats in the summary.
pub fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let q = q.clamp(0.0, 1.0);
    let idx = q * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Aggregate per-task results into the structured summary written to
/// `summary.json`. Pure function for ease of unit testing.
pub fn summarize(
    results: Vec<TaskResult>,
    model: &str,
    suite: Suite,
    started_at: &str,
    wall_time_seconds: f64,
) -> Summary {
    let n_tasks = results.len();
    let n_pass = results
        .iter()
        .filter(|r| r.status == TaskStatus::Pass)
        .count();
    let score_pct = if n_tasks == 0 {
        0.0
    } else {
        (n_pass as f64 / n_tasks as f64) * 100.0
    };
    let mut elapsed: Vec<f64> = results.iter().map(|r| r.elapsed_seconds).collect();
    elapsed.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let total_tokens_in: u64 = results.iter().filter_map(|r| r.tokens_in).sum();
    let total_tokens_out: u64 = results.iter().filter_map(|r| r.tokens_out).sum();
    Summary {
        started_at: started_at.to_string(),
        model: model.to_string(),
        suite: suite.as_str().to_string(),
        n_tasks,
        n_pass,
        score_pct,
        wall_time_seconds,
        p50_elapsed: percentile(&elapsed, 0.5),
        p90_elapsed: percentile(&elapsed, 0.9),
        total_tokens_in,
        total_tokens_out,
        per_task: results,
    }
}

/// Default location for run artifacts: `bench/results/<unix-ts>/`.
fn default_output_dir() -> PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    PathBuf::from("bench/results").join(format!("{ts}"))
}

/// RFC3339-ish timestamp using only stdlib. Format: `YYYY-MM-DDTHH:MM:SSZ`.
fn now_iso() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Convert Unix seconds to UTC y/m/d/h/m/s by hand.
    let (year, month, day, hour, min, sec) = unix_to_ymdhms(secs);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}Z")
}

fn unix_to_ymdhms(secs: u64) -> (i32, u32, u32, u32, u32, u32) {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let hour = (rem / 3600) as u32;
    let min = ((rem % 3600) / 60) as u32;
    let sec = (rem % 60) as u32;
    // Civil-from-days algorithm (Howard Hinnant).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m as u32, d as u32, hour, min, sec)
}

/// Resolve the absolute path of the running `cubi` binary so the
/// subprocess invocation works regardless of cwd. Falls back to the
/// bare name `cubi` (PATH lookup) if `current_exe` fails.
fn cubi_binary() -> PathBuf {
    std::env::current_exe().unwrap_or_else(|_| PathBuf::from("cubi"))
}

/// Recursive copy of `src` into `dst` (which is created). Mirrors what
/// the cp-r equivalent does but is portable. Symlinks are followed.
fn copy_dir_all(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ft = entry.file_type()?;
        let to = dst.join(entry.file_name());
        if ft.is_dir() {
            copy_dir_all(&entry.path(), &to)?;
        } else if ft.is_file() {
            std::fs::copy(entry.path(), &to)?;
        } else if ft.is_symlink() {
            // Best-effort: read the link and re-create it.
            let target = std::fs::read_link(entry.path())?;
            #[cfg(unix)]
            std::os::unix::fs::symlink(&target, &to)?;
            #[cfg(windows)]
            {
                if target.is_dir() {
                    std::os::windows::fs::symlink_dir(&target, &to)?;
                } else {
                    std::os::windows::fs::symlink_file(&target, &to)?;
                }
            }
        }
    }
    Ok(())
}

/// Public entry point invoked by `main.rs`. Returns process exit code:
/// 0 if all tasks passed, 2 if any task failed/timed-out/errored, or
/// a usage-style 2 for setup errors (printed to stderr).
pub async fn run(args: BenchArgs) -> Result<i32> {
    let model = args
        .model
        .clone()
        .or_else(|| std::env::var("CUBI_MODEL").ok())
        .unwrap_or_else(|| crate::DEFAULT_MODEL.to_string());

    let tasks_root = args
        .tasks_root
        .clone()
        .unwrap_or_else(|| PathBuf::from("bench/tasks"));
    let all = discover_tasks(&tasks_root)?;
    // Resolve to an absolute path. `run_task` runs `verify.sh` with the
    // agent's throwaway workdir as cwd, so the default *relative*
    // `bench/tasks` would not resolve there: `sh` exits 2 ("can't open") and
    // every task scores `fail` no matter how capable the model is. The
    // existence check inside `discover_tasks` above still reports the nicer
    // error when the root is simply missing.
    let tasks_root = std::fs::canonicalize(&tasks_root)
        .with_context(|| format!("canonicalize bench tasks root {}", tasks_root.display()))?;
    let tasks = select_tasks(all, args.suite, args.task.as_deref())?;
    if tasks.is_empty() {
        bail!(
            "no bench tasks selected (suite={}, task={:?})",
            args.suite.as_str(),
            args.task
        );
    }

    let output_dir = args.output.clone().unwrap_or_else(default_output_dir);
    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("create bench output dir {}", output_dir.display()))?;
    // Absolute for the same reason as `tasks_root`: the per-task events path
    // is handed to the spawned agent via `--events`, and the agent runs with
    // the throwaway workdir as cwd. Left relative, each run's event log is
    // written *inside* the workdir — polluting what `verify.sh` sees and
    // vanishing when the workdir is cleaned up.
    let output_dir = std::fs::canonicalize(&output_dir)
        .with_context(|| format!("canonicalize bench output dir {}", output_dir.display()))?;

    let started_at = now_iso();
    let wall_start = Instant::now();
    let cubi_bin = cubi_binary();

    if !args.json {
        eprintln!(
            "cubi bench: suite={} model={} tasks={} output={}",
            args.suite.as_str(),
            model,
            tasks.len(),
            output_dir.display()
        );
    }

    let mut results: Vec<TaskResult> = Vec::with_capacity(tasks.len());
    for task in &tasks {
        let result = run_one_task(
            task,
            &model,
            &cubi_bin,
            &tasks_root,
            &output_dir,
            args.step_cap,
            args.keep_workdir,
            args.json,
            args.time_cap_multiplier,
        )
        .await;
        let result = match result {
            Ok(r) => r,
            Err(e) => TaskResult {
                task_id: task.id.clone(),
                model: model.clone(),
                status: TaskStatus::Error,
                elapsed_seconds: 0.0,
                steps_used: None,
                verify_exit_code: None,
                cubi_exit_code: None,
                tokens_in: None,
                tokens_out: None,
                events_path: None,
                agent_stderr_tail: None,
                verifier_stdout_tail: None,
                verifier_stderr_tail: None,
                error: Some(format!("{e:#}")),
            },
        };
        if !args.json {
            eprintln!(
                "  [{}] {} ({:.1}s)",
                status_glyph(result.status),
                result.task_id,
                result.elapsed_seconds
            );
        }
        results.push(result);
    }

    let wall_time_seconds = wall_start.elapsed().as_secs_f64();
    let summary = summarize(results, &model, args.suite, &started_at, wall_time_seconds);

    let summary_path = output_dir.join("summary.json");
    let summary_json = serde_json::to_string_pretty(&summary)?;
    std::fs::write(&summary_path, &summary_json)
        .with_context(|| format!("write {}", summary_path.display()))?;

    if args.json {
        println!("{summary_json}");
    } else {
        eprintln!(
            "cubi bench: {}/{} passed ({:.1}%) in {:.1}s — summary at {}",
            summary.n_pass,
            summary.n_tasks,
            summary.score_pct,
            summary.wall_time_seconds,
            summary_path.display()
        );
    }

    Ok(if summary.n_pass == summary.n_tasks {
        0
    } else {
        2
    })
}

fn status_glyph(s: TaskStatus) -> &'static str {
    match s {
        TaskStatus::Pass => "PASS",
        TaskStatus::Fail => "FAIL",
        TaskStatus::Timeout => "TIME",
        TaskStatus::Error => "ERR ",
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_one_task(
    task: &TaskDef,
    model: &str,
    cubi_bin: &Path,
    tasks_root: &Path,
    output_dir: &Path,
    step_cap_override: Option<u32>,
    keep_workdir: bool,
    quiet_stdout: bool,
    time_cap_multiplier: Option<f64>,
) -> Result<TaskResult> {
    let src_repo = tasks_root.join(&task.id).join("repo");
    let verify_script = tasks_root.join(&task.id).join("verify.sh");
    if !src_repo.exists() {
        bail!(
            "missing repo/ for task {} at {}",
            task.id,
            src_repo.display()
        );
    }
    if !verify_script.exists() {
        bail!(
            "missing verify.sh for task {} at {}",
            task.id,
            verify_script.display()
        );
    }

    let workdir = tempfile::Builder::new()
        .prefix(&format!("cubi-bench-{}-", task.id))
        .tempdir()
        .context("create bench tempdir")?;
    copy_dir_all(&src_repo, workdir.path())?;

    // Sibling tempdir used as the subprocess's HOME so that any
    // `~/.cubi/` state (sessions, telemetry, policy, todos, …) lands
    // here instead of the developer's real home. Kept separate from the
    // workdir so verify.sh can list workdir contents without seeing the
    // dotfile tree.
    let home_dir = tempfile::Builder::new()
        .prefix(&format!("cubi-bench-home-{}-", task.id))
        .tempdir()
        .context("create bench home tempdir")?;

    // Cubi refuses writes outside a trusted root, and headless mode denies
    // instead of prompting — so without pre-trusting the throwaway workdir
    // every `edit_file`/`write_file` call comes back "[tool denied]" and the
    // suite scores 0% no matter how capable the model is. Seed the isolated
    // HOME's trust store the same way `cubi swebench` does.
    if let Err(e) = crate::permissions::seed_trust_file(home_dir.path(), workdir.path()) {
        return Ok(TaskResult {
            task_id: task.id.clone(),
            model: model.to_string(),
            status: TaskStatus::Error,
            elapsed_seconds: 0.0,
            steps_used: None,
            verify_exit_code: None,
            cubi_exit_code: None,
            tokens_in: None,
            tokens_out: None,
            events_path: None,
            agent_stderr_tail: None,
            verifier_stdout_tail: None,
            verifier_stderr_tail: None,
            error: Some(format!("seed trust: {e:#}")),
        });
    }

    let events_path = output_dir.join(format!("{}.events.jsonl", task.id));
    let _step_cap = step_cap_override.unwrap_or(task.step_cap); // reserved: cubi has no public --max-steps flag yet

    let start = Instant::now();
    let mut cmd = tokio::process::Command::new(cubi_bin);
    cmd.arg("-p")
        .arg(&task.prompt)
        .arg("--json")
        .arg("--no-stream")
        .arg("--no-banner")
        .arg("--quiet")
        .arg("--events")
        .arg(&events_path)
        .env("CUBI_MODEL", model)
        .env("CUBI_NO_BANNER", "1")
        .env("CUBI_NO_ONBOARD", "1")
        .current_dir(workdir.path())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    configure_isolated_child_env(&mut cmd, home_dir.path());
    let _ = quiet_stdout; // reserved: a future verbose mode may stream cubi output through.

    let child = cmd
        .spawn()
        .with_context(|| format!("spawn cubi for task {}", task.id))?;

    let time_cap_seconds = scaled_time_cap(task.time_cap_seconds, time_cap_multiplier);
    let timeout = Duration::from_secs(time_cap_seconds);
    let agent = match capture_child(child, timeout).await {
        Ok(output) => output,
        Err(error) => {
            let (steps_used, tokens_in, tokens_out) = parse_events_metrics(&events_path);
            return Ok(TaskResult {
                task_id: task.id.clone(),
                model: model.to_string(),
                status: TaskStatus::Error,
                elapsed_seconds: start.elapsed().as_secs_f64(),
                steps_used,
                verify_exit_code: None,
                cubi_exit_code: None,
                tokens_in,
                tokens_out,
                events_path: Some(events_path.display().to_string()),
                agent_stderr_tail: None,
                verifier_stdout_tail: None,
                verifier_stderr_tail: None,
                error: Some(format!("agent wait error: {error}")),
            });
        }
    };
    if agent.timed_out {
        // Parse whatever the agent managed to write before the cap. A
        // timeout that reports `steps_used: null` is indistinguishable
        // between "never got started" and "was one step from done", which
        // is precisely the number needed to size `--time-cap-multiplier`.
        // The event log is appended per event, so the partial file left by
        // the killed child is still valid JSONL up to the last whole line.
        let (steps_used, tokens_in, tokens_out) = parse_events_metrics(&events_path);
        return Ok(TaskResult {
            task_id: task.id.clone(),
            model: model.to_string(),
            status: TaskStatus::Timeout,
            elapsed_seconds: start.elapsed().as_secs_f64(),
            steps_used,
            verify_exit_code: None,
            cubi_exit_code: None,
            tokens_in,
            tokens_out,
            events_path: Some(events_path.display().to_string()),
            agent_stderr_tail: diagnostic(agent.stderr_tail),
            verifier_stdout_tail: None,
            verifier_stderr_tail: None,
            // Report the cap actually enforced, not the unscaled one from
            // task.toml, so a scaled CI run isn't misleading.
            error: Some(format!("exceeded time_cap_seconds = {}", time_cap_seconds)),
        });
    }
    let cubi_exit_code = agent.exit_code;
    let agent_stderr_tail = diagnostic(agent.stderr_tail);
    if infrastructure_exit(cubi_exit_code) {
        let (steps_used, tokens_in, tokens_out) = parse_events_metrics(&events_path);
        return Ok(TaskResult {
            task_id: task.id.clone(),
            model: model.to_string(),
            status: TaskStatus::Error,
            elapsed_seconds: start.elapsed().as_secs_f64(),
            steps_used,
            verify_exit_code: None,
            cubi_exit_code,
            tokens_in,
            tokens_out,
            events_path: Some(events_path.display().to_string()),
            agent_stderr_tail,
            verifier_stdout_tail: None,
            verifier_stderr_tail: None,
            error: Some(format!(
                "agent infrastructure failure (exit code {:?})",
                cubi_exit_code
            )),
        });
    }

    // Run verify.sh from inside the workdir. `verify.sh` is documented
    // as POSIX sh, so invoke `sh` rather than `bash`.
    let mut verify_cmd = tokio::process::Command::new("sh");
    verify_cmd
        .arg(&verify_script)
        .current_dir(workdir.path())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    configure_isolated_child_env(&mut verify_cmd, home_dir.path());
    let verify_child = verify_cmd
        .spawn()
        .with_context(|| format!("spawn verify.sh for task {}", task.id))?;
    let verify_out = capture_child(verify_child, verifier_timeout())
        .await
        .with_context(|| format!("run verify.sh for task {}", task.id))?;
    if verify_out.timed_out {
        let (steps_used, tokens_in, tokens_out) = parse_events_metrics(&events_path);
        return Ok(TaskResult {
            task_id: task.id.clone(),
            model: model.to_string(),
            status: TaskStatus::Error,
            elapsed_seconds: start.elapsed().as_secs_f64(),
            steps_used,
            verify_exit_code: None,
            cubi_exit_code,
            tokens_in,
            tokens_out,
            events_path: Some(events_path.display().to_string()),
            agent_stderr_tail,
            verifier_stdout_tail: diagnostic(verify_out.stdout_tail),
            verifier_stderr_tail: diagnostic(verify_out.stderr_tail),
            error: Some(format!(
                "verifier timed out after {}s",
                verifier_timeout().as_secs()
            )),
        });
    }
    let verify_exit_code = verify_out.exit_code;
    let final_status = if verify_exit_code == Some(0) {
        TaskStatus::Pass
    } else {
        TaskStatus::Fail
    };

    // Best-effort: count agent steps + tokens from the events file.
    let (steps_used, tokens_in, tokens_out) = parse_events_metrics(&events_path);

    if !keep_workdir {
        // TempDir drops on scope exit; this is here to make it explicit
        // and to enable the `--keep-workdir` toggle later.
        drop(workdir);
        drop(home_dir);
    } else {
        // Leak the TempDirs so the directories survive. Print the paths
        // so the user can inspect them.
        let path = workdir.keep();
        let home_path = home_dir.keep();
        eprintln!("cubi bench: kept workdir at {}", path.display());
        eprintln!("cubi bench: kept home at {}", home_path.display());
    }

    Ok(TaskResult {
        task_id: task.id.clone(),
        model: model.to_string(),
        status: final_status,
        elapsed_seconds: start.elapsed().as_secs_f64(),
        steps_used,
        verify_exit_code,
        cubi_exit_code,
        tokens_in,
        tokens_out,
        events_path: Some(events_path.display().to_string()),
        agent_stderr_tail,
        verifier_stdout_tail: diagnostic(verify_out.stdout_tail),
        verifier_stderr_tail: diagnostic(verify_out.stderr_tail),
        error: None,
    })
}

/// Walk the JSONL events file produced by `cubi --events` and pull out
/// a few coarse-grained metrics. Returns `(steps, tokens_in, tokens_out)`.
/// All are best-effort: if the file is missing or fields aren't present,
/// the corresponding entries come back as `None`.
/// Event names and payload shape must match what the agent actually writes to
/// the `--events` sink (`src/cli/agent.rs`): `turn_start`, `tool_call_start`,
/// `tool_call_complete`, `turn_end`, with per-turn token counts nested under
/// `usage`. An earlier version of this function matched `tool_call` / `turn`
/// and read *top-level* `prompt_tokens`, none of which are ever emitted — so it
/// silently reported `steps_used: null` and zero tokens for every task,
/// including passing ones. The tests below pin the real schema; don't relax
/// them without checking the emitter.
///
/// `steps` counts `tool_call_start` events (tool invocations attempted). A run
/// that produced recognizable events but called no tools yields `Some(0)`
/// rather than `None`, since "started but never called a tool" is a materially
/// different signal from "no event log at all".
fn parse_events_metrics(path: &Path) -> (Option<u32>, Option<u64>, Option<u64>) {
    let Ok(raw) = std::fs::read_to_string(path) else {
        return (None, None, None);
    };
    let mut steps: u32 = 0;
    let mut tokens_in: u64 = 0;
    let mut tokens_out: u64 = 0;
    let mut saw_tokens = false;
    let mut saw_activity = false;
    for line in raw.lines() {
        // A partial trailing line (the child was killed mid-write on a
        // timeout) simply fails to parse and is skipped.
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        match v.get("type").and_then(|t| t.as_str()) {
            Some("tool_call_start") => {
                steps = steps.saturating_add(1);
                saw_activity = true;
            }
            Some("turn_start" | "tool_call_complete" | "turn_end") => saw_activity = true,
            _ => {}
        }
        if let Some(usage) = v.get("usage") {
            if let Some(n) = usage.get("prompt_tokens").and_then(|n| n.as_u64()) {
                tokens_in = tokens_in.saturating_add(n);
                saw_tokens = true;
            }
            if let Some(n) = usage.get("completion_tokens").and_then(|n| n.as_u64()) {
                tokens_out = tokens_out.saturating_add(n);
                saw_tokens = true;
            }
        }
    }
    (
        if saw_activity { Some(steps) } else { None },
        if saw_tokens { Some(tokens_in) } else { None },
        if saw_tokens { Some(tokens_out) } else { None },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolated_child_env_uses_a_per_home_cargo_target_dir() {
        let home = std::path::PathBuf::from("/tmp/cubi-bench-home-for-test");
        let mut command = tokio::process::Command::new("true");
        configure_isolated_child_env(&mut command, &home);
        let target_dir = command
            .as_std()
            .get_envs()
            .find(|(key, _)| *key == "CARGO_TARGET_DIR")
            .and_then(|(_, value)| value)
            .map(std::path::PathBuf::from);
        assert_eq!(target_dir, Some(home.join("cargo-target")));
    }

    fn mk_result(id: &str, status: TaskStatus, elapsed: f64) -> TaskResult {
        TaskResult {
            task_id: id.to_string(),
            model: "qwen3:8b".to_string(),
            status,
            elapsed_seconds: elapsed,
            steps_used: Some(3),
            verify_exit_code: Some(if status == TaskStatus::Pass { 0 } else { 1 }),
            cubi_exit_code: Some(0),
            tokens_in: Some(100),
            tokens_out: Some(50),
            events_path: None,
            agent_stderr_tail: None,
            verifier_stdout_tail: None,
            verifier_stderr_tail: None,
            error: None,
        }
    }

    #[test]
    fn percentile_handles_empty_and_single() {
        assert_eq!(percentile(&[], 0.5), 0.0);
        assert_eq!(percentile(&[7.0], 0.9), 7.0);
    }

    #[test]
    fn percentile_interpolates() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&v, 0.5) - 3.0).abs() < 1e-9);
        // p90 with 5 elements: idx = 0.9*4 = 3.6 -> 4 + 0.6*(5-4) = 4.6
        assert!((percentile(&v, 0.9) - 4.6).abs() < 1e-9);
    }

    #[test]
    fn summarize_computes_score_and_percentiles() {
        let results = vec![
            mk_result("a", TaskStatus::Pass, 1.0),
            mk_result("b", TaskStatus::Pass, 2.0),
            mk_result("c", TaskStatus::Fail, 3.0),
            mk_result("d", TaskStatus::Pass, 4.0),
            mk_result("e", TaskStatus::Timeout, 5.0),
        ];
        let s = summarize(
            results,
            "qwen3:8b",
            Suite::Quick,
            "2025-01-01T00:00:00Z",
            15.0,
        );
        assert_eq!(s.n_tasks, 5);
        assert_eq!(s.n_pass, 3);
        assert!((s.score_pct - 60.0).abs() < 1e-9);
        assert!((s.p50_elapsed - 3.0).abs() < 1e-9);
        assert!((s.p90_elapsed - 4.6).abs() < 1e-9);
        assert_eq!(s.total_tokens_in, 500);
        assert_eq!(s.total_tokens_out, 250);
        assert_eq!(s.suite, "quick");
    }

    #[test]
    fn summarize_empty_input_is_zero_score() {
        let s = summarize(vec![], "m", Suite::All, "t", 0.0);
        assert_eq!(s.n_tasks, 0);
        assert_eq!(s.n_pass, 0);
        assert_eq!(s.score_pct, 0.0);
        assert_eq!(s.p50_elapsed, 0.0);
        assert_eq!(s.p90_elapsed, 0.0);
    }

    #[test]
    fn discover_and_parse_shipped_quick_tasks() {
        // Run from the repo root (cargo test default cwd).
        let root = Path::new("bench/tasks");
        if !root.exists() {
            return; // not in repo root; skip silently
        }
        let tasks = discover_tasks(root).expect("discover");
        assert!(!tasks.is_empty(), "expected at least one shipped task");
        for t in &tasks {
            assert!(!t.id.is_empty());
            assert!(!t.prompt.trim().is_empty());
            assert!(t.time_cap_seconds > 0);
            assert!(t.step_cap > 0);
        }
        let quick = select_tasks(tasks.clone(), Suite::Quick, None).unwrap();
        assert!(
            !quick.is_empty(),
            "quick suite must include at least one easy task"
        );
        let all = select_tasks(tasks, Suite::All, None).unwrap();
        assert!(all.len() >= quick.len());
    }

    #[test]
    fn unix_to_ymdhms_matches_known_epoch() {
        // 1700000000 = 2023-11-14T22:13:20 UTC
        let (y, mo, d, h, mi, s) = unix_to_ymdhms(1_700_000_000);
        assert_eq!((y, mo, d, h, mi, s), (2023, 11, 14, 22, 13, 20));
    }

    #[test]
    fn scaled_time_cap_applies_multiplier() {
        assert_eq!(scaled_time_cap(120, Some(3.0)), 360);
        assert_eq!(scaled_time_cap(180, Some(3.0)), 540);
        // Rounds up, so the cap is never below the requested scale.
        assert_eq!(scaled_time_cap(120, Some(2.5)), 300);
        assert_eq!(scaled_time_cap(121, Some(1.5)), 182);
        // A multiplier below 1 deliberately shortens the cap.
        assert_eq!(scaled_time_cap(120, Some(0.5)), 60);
    }

    #[test]
    fn scaled_time_cap_passes_through_without_multiplier() {
        assert_eq!(scaled_time_cap(120, None), 120);
        assert_eq!(scaled_time_cap(120, Some(1.0)), 120);
    }

    #[test]
    fn scaled_time_cap_ignores_unusable_multipliers() {
        // A zero/negative/NaN scale must not collapse the cap to 0, which
        // would time every task out instantly.
        for bad in [0.0, -2.0, f64::NAN, f64::INFINITY] {
            assert_eq!(scaled_time_cap(120, Some(bad)), 120, "bad = {bad}");
        }
    }

    #[test]
    fn scaled_time_cap_saturates_instead_of_wrapping() {
        assert_eq!(scaled_time_cap(u64::MAX, Some(2.0)), u64::MAX);
    }

    /// Event lines copied from what `src/cli/agent.rs` actually emits to the
    /// `--events` sink. If the emitter's names or payload shape change, this
    /// test is the tripwire.
    const REAL_EVENTS: &str = concat!(
        r#"{"model":"m","ts":"t","turn":1,"type":"turn_start"}"#,
        "\n",
        r#"{"args":{"path":"src/lib.rs"},"tool":"edit_file","ts":"t","type":"tool_call_start"}"#,
        "\n",
        r#"{"ok":true,"result_chars":9,"tool":"edit_file","ts":"t","type":"tool_call_complete"}"#,
        "\n",
        r#"{"model":"m","ts":"t","type":"turn_end","usage":{"completion_tokens":34,"elapsed_ms":5,"prompt_tokens":12}}"#,
        "\n",
    );

    fn write_events(contents: &str) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        f.write_all(contents.as_bytes()).expect("write events");
        f.flush().expect("flush events");
        f
    }

    #[test]
    fn parse_events_metrics_reads_real_event_schema() {
        let f = write_events(REAL_EVENTS);
        let (steps, tin, tout) = parse_events_metrics(f.path());
        assert_eq!(steps, Some(1), "one tool_call_start");
        assert_eq!(tin, Some(12), "prompt_tokens live under `usage`");
        assert_eq!(tout, Some(34), "completion_tokens live under `usage`");
    }

    #[test]
    fn parse_events_metrics_skips_truncated_trailing_line() {
        // A timeout kills the child mid-write, leaving a partial last line.
        let truncated = format!("{REAL_EVENTS}{{\"type\":\"tool_call_st");
        let f = write_events(&truncated);
        let (steps, tin, _) = parse_events_metrics(f.path());
        assert_eq!(steps, Some(1), "partial line must be ignored, not counted");
        assert_eq!(tin, Some(12));
    }

    #[test]
    fn parse_events_metrics_reports_zero_steps_when_no_tool_was_called() {
        // Distinguishable from "no event log": the agent ran but called nothing.
        let f = write_events(concat!(
            r#"{"turn":1,"type":"turn_start"}"#,
            "\n",
            r#"{"type":"turn_end","usage":{"prompt_tokens":7,"completion_tokens":1}}"#,
            "\n",
        ));
        let (steps, tin, tout) = parse_events_metrics(f.path());
        assert_eq!(steps, Some(0));
        assert_eq!((tin, tout), (Some(7), Some(1)));
    }

    #[test]
    fn parse_events_metrics_returns_none_for_missing_file() {
        let (steps, tin, tout) = parse_events_metrics(Path::new("/nonexistent/events.jsonl"));
        assert_eq!((steps, tin, tout), (None, None, None));
    }

    #[test]
    fn parse_events_metrics_sums_tokens_across_turns() {
        let two_turns = format!("{REAL_EVENTS}{REAL_EVENTS}");
        let (steps, tin, tout) = parse_events_metrics(write_events(&two_turns).path());
        assert_eq!(steps, Some(2));
        assert_eq!((tin, tout), (Some(24), Some(68)));
    }
}
