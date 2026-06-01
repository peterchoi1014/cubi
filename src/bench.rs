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
//! regression metric that runs nightly against `qwen3:8b`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};

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
        // Avoid the binary writing into the developer's real
        // `~/.cubi/sessions/` while running the bench.
        .env("CUBI_NO_BANNER", "1")
        .current_dir(workdir.path())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let _ = quiet_stdout; // reserved: a future verbose mode may stream cubi output through.

    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawn cubi for task {}", task.id))?;

    let timeout = Duration::from_secs(task.time_cap_seconds);
    let cubi_exit_code: Option<i32>;
    let status: Option<TaskStatus>;
    match tokio::time::timeout(timeout, child.wait()).await {
        Ok(Ok(exit)) => {
            cubi_exit_code = exit.code();
            status = None; // determined by verify step
        }
        Ok(Err(e)) => {
            cubi_exit_code = None;
            status = Some(TaskStatus::Error);
            // best-effort cleanup
            let _ = child.kill().await;
            return Ok(TaskResult {
                task_id: task.id.clone(),
                model: model.to_string(),
                status: status.unwrap(),
                elapsed_seconds: start.elapsed().as_secs_f64(),
                steps_used: None,
                verify_exit_code: None,
                cubi_exit_code,
                tokens_in: None,
                tokens_out: None,
                events_path: Some(events_path.display().to_string()),
                error: Some(format!("wait error: {e}")),
            });
        }
        Err(_) => {
            let _ = child.kill().await;
            return Ok(TaskResult {
                task_id: task.id.clone(),
                model: model.to_string(),
                status: TaskStatus::Timeout,
                elapsed_seconds: start.elapsed().as_secs_f64(),
                steps_used: None,
                verify_exit_code: None,
                cubi_exit_code: None,
                tokens_in: None,
                tokens_out: None,
                events_path: Some(events_path.display().to_string()),
                error: Some(format!(
                    "exceeded time_cap_seconds = {}",
                    task.time_cap_seconds
                )),
            });
        }
    }

    // Run verify.sh from inside the workdir.
    let verify_out = tokio::process::Command::new("bash")
        .arg(&verify_script)
        .current_dir(workdir.path())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .with_context(|| format!("run verify.sh for task {}", task.id))?;
    let verify_exit_code = verify_out.status.code();
    let final_status = if verify_exit_code == Some(0) {
        TaskStatus::Pass
    } else {
        TaskStatus::Fail
    };
    let final_status = status.unwrap_or(final_status);

    // Best-effort: count agent steps + tokens from the events file.
    let (steps_used, tokens_in, tokens_out) = parse_events_metrics(&events_path);

    if !keep_workdir {
        // TempDir drops on scope exit; this is here to make it explicit
        // and to enable the `--keep-workdir` toggle later.
        drop(workdir);
    } else {
        // Leak the TempDir so the directory survives. Print the path
        // so the user can inspect it.
        let path = workdir.keep();
        eprintln!("cubi bench: kept workdir at {}", path.display());
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
        error: None,
    })
}

/// Walk the JSONL events file produced by `cubi --events` and pull out
/// a few coarse-grained metrics. Returns `(steps, tokens_in, tokens_out)`.
/// All are best-effort: if the file is missing or fields aren't present,
/// the corresponding entries come back as `None`.
fn parse_events_metrics(path: &Path) -> (Option<u32>, Option<u64>, Option<u64>) {
    let Ok(raw) = std::fs::read_to_string(path) else {
        return (None, None, None);
    };
    let mut steps: u32 = 0;
    let mut tokens_in: u64 = 0;
    let mut tokens_out: u64 = 0;
    let mut saw_tokens = false;
    let mut saw_step = false;
    for line in raw.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if let Some(t) = v.get("type").and_then(|t| t.as_str()) {
            if matches!(t, "tool_call" | "tool_start" | "turn") {
                steps = steps.saturating_add(1);
                saw_step = true;
            }
        }
        if let Some(n) = v.get("prompt_tokens").and_then(|n| n.as_u64()) {
            tokens_in = tokens_in.saturating_add(n);
            saw_tokens = true;
        }
        if let Some(n) = v.get("completion_tokens").and_then(|n| n.as_u64()) {
            tokens_out = tokens_out.saturating_add(n);
            saw_tokens = true;
        }
    }
    (
        if saw_step { Some(steps) } else { None },
        if saw_tokens { Some(tokens_in) } else { None },
        if saw_tokens { Some(tokens_out) } else { None },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
