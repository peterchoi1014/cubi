//! `cubi swebench` — SWE-bench-Lite prediction generator + local scorer.
//!
//! Unlike the curated [`crate::bench`] suite (network-free, hand-written
//! tasks), this module drives Cubi against the real
//! [SWE-bench-Lite](https://www.swebench.com/) dataset: 300 GitHub issues
//! drawn from popular Python projects. Each instance provides a repo, a
//! `base_commit`, an issue `problem_statement`, and the test node ids that
//! encode the fix (`FAIL_TO_PASS`) plus the tests that must keep passing
//! (`PASS_TO_PASS`).
//!
//! The canonical resolved-rate number comes from the upstream Dockerized
//! harness (`python -m swebench.harness.run_evaluation`), which pins each
//! instance's Python environment. Reproducing that environment matrix in
//! Rust is out of scope, so this module focuses on the two artifacts the
//! ecosystem actually consumes:
//!
//! 1. **`predictions.jsonl`** — one record per instance in the official
//!    schema (`instance_id`, `model_name_or_path`, `model_patch`). Feed
//!    this straight into the upstream harness for the canonical score.
//! 2. **`summary.json`** — Cubi-local aggregation. With `--score`, this
//!    module also applies each instance's `test_patch` and runs the
//!    `FAIL_TO_PASS`/`PASS_TO_PASS` node ids with `pytest` for a fast,
//!    local (best-effort) resolved rate that requires no Docker.
//!
//! For each selected instance the runner: checks out `base_commit` in a
//! clean working tree, drives the Cubi binary headlessly with the issue
//! text, captures `git diff` as the model patch, and (optionally) scores.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize};

/// One SWE-bench-Lite instance, parsed from a dataset JSONL line.
///
/// The official dataset stores `FAIL_TO_PASS`/`PASS_TO_PASS` as
/// JSON-encoded strings (e.g. `"[\"a\", \"b\"]"`); [`string_or_seq`]
/// accepts either that form or a native JSON array.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct Instance {
    pub instance_id: String,
    pub repo: String,
    pub base_commit: String,
    #[serde(default)]
    pub problem_statement: String,
    #[serde(default)]
    pub patch: String,
    #[serde(default)]
    pub test_patch: String,
    #[serde(rename = "FAIL_TO_PASS", default, deserialize_with = "string_or_seq")]
    pub fail_to_pass: Vec<String>,
    #[serde(rename = "PASS_TO_PASS", default, deserialize_with = "string_or_seq")]
    pub pass_to_pass: Vec<String>,
    #[serde(default)]
    pub environment_setup_commit: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
}

/// Deserialize a `Vec<String>` from either a native JSON array or a
/// string containing a JSON-encoded array (the dataset's on-disk form).
fn string_or_seq<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Raw {
        Seq(Vec<String>),
        Str(String),
    }
    match Raw::deserialize(deserializer)? {
        Raw::Seq(v) => Ok(v),
        Raw::Str(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                return Ok(Vec::new());
            }
            serde_json::from_str(trimmed).map_err(de::Error::custom)
        }
    }
}

/// A prediction record in the official SWE-bench schema. Serialized one
/// per line into `predictions.jsonl`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Prediction {
    pub instance_id: String,
    /// The model/agent identifier reported to the upstream harness.
    #[serde(rename = "model_name_or_path")]
    pub model_name_or_path: String,
    /// Unified diff the agent produced. Empty string means "no edit".
    pub model_patch: String,
}

/// Whether an instance's tests resolved the issue (local scorer only).
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Resolution {
    /// All FAIL_TO_PASS pass and all PASS_TO_PASS still pass.
    Resolved,
    /// At least one required test did not pass.
    Unresolved,
    /// Scoring was not attempted (generation-only run).
    NotScored,
}

/// Outcome of a single pytest node id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestOutcome {
    Passed,
    Failed,
    Errored,
}

/// Per-instance result recorded to disk and aggregated into the summary.
#[derive(Debug, Clone, Serialize)]
pub struct InstanceResult {
    pub instance_id: String,
    pub model: String,
    pub resolution: Resolution,
    /// Number of lines in the generated patch (0 = no edit produced).
    pub patch_lines: usize,
    pub elapsed_seconds: f64,
    pub cubi_exit_code: Option<i32>,
    /// FAIL_TO_PASS node ids that passed / total (scorer only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fail_to_pass_passed: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fail_to_pass_total: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pass_to_pass_passed: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pass_to_pass_total: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Aggregated run summary written to `<output>/summary.json`.
#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub started_at: String,
    pub model: String,
    pub dataset: String,
    pub n_instances: usize,
    /// Instances for which a non-empty patch was produced.
    pub n_with_patch: usize,
    /// Instances marked Resolved by the local scorer (0 when not scoring).
    pub n_resolved: usize,
    pub scored: bool,
    pub resolved_pct: f64,
    pub wall_time_seconds: f64,
    pub per_instance: Vec<InstanceResult>,
}

/// CLI arguments parsed in `main.rs`.
#[derive(Debug, Clone)]
pub struct SweBenchArgs {
    /// Path to the dataset JSONL (one instance per line).
    pub dataset: Option<PathBuf>,
    pub model: Option<String>,
    /// Run a single instance by id.
    pub instance: Option<String>,
    /// Cap the number of instances (after id filter). `None` = all.
    pub limit: Option<usize>,
    pub output: Option<PathBuf>,
    /// Directory of pre-cloned repos (named `<owner>__<name>` or the repo
    /// slug). When absent, repos are cloned from GitHub on demand.
    pub repos_root: Option<PathBuf>,
    /// Apply `test_patch` and run pytest to compute a local resolved rate.
    pub score: bool,
    /// Per-instance wall-clock cap for the Cubi subprocess.
    pub time_cap_seconds: u64,
    pub keep_workdir: bool,
    pub json: bool,
}

impl Default for SweBenchArgs {
    fn default() -> Self {
        Self {
            dataset: None,
            model: None,
            instance: None,
            limit: None,
            output: None,
            repos_root: None,
            score: false,
            time_cap_seconds: 900,
            keep_workdir: false,
            json: false,
        }
    }
}

/// Parse a dataset JSONL string into instances, skipping blank lines.
/// Line numbers are 1-based in error messages.
pub fn parse_dataset(raw: &str) -> Result<Vec<Instance>> {
    let mut out = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let inst: Instance =
            serde_json::from_str(line).with_context(|| format!("parse dataset line {}", i + 1))?;
        out.push(inst);
    }
    Ok(out)
}

/// Select instances by optional single-id filter, then an optional limit.
pub fn select_instances(
    all: Vec<Instance>,
    only: Option<&str>,
    limit: Option<usize>,
) -> Result<Vec<Instance>> {
    let mut selected: Vec<Instance> = match only {
        Some(id) => {
            let picked = all.into_iter().find(|inst| inst.instance_id == id);
            match picked {
                Some(inst) => vec![inst],
                None => bail!("no SWE-bench instance with id `{}`", id),
            }
        }
        None => all,
    };
    if let Some(n) = limit {
        selected.truncate(n);
    }
    Ok(selected)
}

/// Map the local pytest owner slug for a repo, e.g.
/// `astropy/astropy` -> `astropy__astropy`. The upstream harness uses this
/// underscore form for cached repo directories.
pub fn repo_dir_name(repo: &str) -> String {
    repo.replace('/', "__")
}

/// Parse `pytest -rA` output into a node-id → outcome map. Recognizes the
/// short-summary lines pytest prints with `-rA` that carry a full node id:
/// `PASSED test::id`, `FAILED test::id`, and `ERROR test::id`. (`SKIPPED`
/// lines report `file:line: reason` rather than a node id, so they are
/// ignored — a required test that is skipped simply never records a
/// `Passed`, which correctly counts as not-resolved.) Later outcomes for
/// the same id win.
pub fn parse_pytest_outcomes(stdout: &str) -> BTreeMap<String, TestOutcome> {
    let mut map = BTreeMap::new();
    for line in stdout.lines() {
        let line = line.trim();
        let (outcome, rest) = if let Some(r) = line.strip_prefix("PASSED ") {
            (TestOutcome::Passed, r)
        } else if let Some(r) = line.strip_prefix("FAILED ") {
            (TestOutcome::Failed, r)
        } else if let Some(r) = line.strip_prefix("ERROR ") {
            (TestOutcome::Errored, r)
        } else {
            continue;
        };
        // Format is `OUTCOME nodeid` optionally followed by ` - reason`.
        // The node id is the first whitespace-delimited token.
        let node = rest.split_whitespace().next().unwrap_or("");
        if !node.is_empty() {
            map.insert(node.to_string(), outcome);
        }
    }
    map
}

/// Count how many of `required` node ids passed in `outcomes`.
pub fn count_passed(required: &[String], outcomes: &BTreeMap<String, TestOutcome>) -> usize {
    required
        .iter()
        .filter(|id| matches!(outcomes.get(*id), Some(TestOutcome::Passed)))
        .count()
}

/// Decide whether an instance is resolved: every FAIL_TO_PASS must pass and
/// every PASS_TO_PASS must still pass. Empty FAIL_TO_PASS is treated as
/// unresolved (nothing to prove the fix), matching the upstream harness.
pub fn score_instance(
    fail_to_pass: &[String],
    pass_to_pass: &[String],
    outcomes: &BTreeMap<String, TestOutcome>,
) -> Resolution {
    if fail_to_pass.is_empty() {
        return Resolution::Unresolved;
    }
    let f2p_ok = fail_to_pass
        .iter()
        .all(|id| matches!(outcomes.get(id), Some(TestOutcome::Passed)));
    let p2p_ok = pass_to_pass
        .iter()
        .all(|id| matches!(outcomes.get(id), Some(TestOutcome::Passed)));
    if f2p_ok && p2p_ok {
        Resolution::Resolved
    } else {
        Resolution::Unresolved
    }
}

/// Aggregate per-instance results into the summary. Pure for testability.
pub fn summarize(
    results: Vec<InstanceResult>,
    model: &str,
    dataset: &str,
    scored: bool,
    started_at: &str,
    wall_time_seconds: f64,
) -> Summary {
    let n_instances = results.len();
    let n_with_patch = results.iter().filter(|r| r.patch_lines > 0).count();
    let n_resolved = results
        .iter()
        .filter(|r| r.resolution == Resolution::Resolved)
        .count();
    let resolved_pct = if n_instances == 0 {
        0.0
    } else {
        (n_resolved as f64 / n_instances as f64) * 100.0
    };
    Summary {
        started_at: started_at.to_string(),
        model: model.to_string(),
        dataset: dataset.to_string(),
        n_instances,
        n_with_patch,
        n_resolved,
        scored,
        resolved_pct,
        wall_time_seconds,
        per_instance: results,
    }
}

fn default_output_dir() -> PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    PathBuf::from("bench/swebench-results").join(format!("{ts}"))
}

fn now_iso() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (year, month, day, hour, min, sec) = unix_to_ymdhms(secs);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}Z")
}

fn unix_to_ymdhms(secs: u64) -> (i32, u32, u32, u32, u32, u32) {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let hour = (rem / 3600) as u32;
    let min = ((rem % 3600) / 60) as u32;
    let sec = (rem % 60) as u32;
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

fn cubi_binary() -> PathBuf {
    std::env::current_exe().unwrap_or_else(|_| PathBuf::from("cubi"))
}

/// Build the prompt handed to Cubi for one instance: an explicit
/// explore-then-edit workflow followed by the raw issue text.
///
/// SWE-bench instances live in large, unfamiliar repos, so the biggest
/// failure mode for a local model is *guessing* file paths (e.g.
/// `settings.py`, `~/.cubi/settings.py`) instead of locating the real
/// file. The directive below forces the model to ground itself — build a
/// repo map / search for the relevant symbol and read the file — before
/// editing, and to verify with the exact `old_text` an edit needs.
pub fn build_prompt(problem_statement: &str) -> String {
    format!(
        "You are fixing a bug in a real software project checked out in the \
         current working directory. Work methodically:\n\
         1. EXPLORE FIRST. Do not guess file paths. Use `repo_map`, and \
         `grep`/`bash` (e.g. `grep -rn <symbol> .`, `ls`) to locate the \
         file(s) that actually implement the behavior described in the \
         issue. Paths are relative to the current directory.\n\
         2. READ the target file before editing so your `edit_file` \
         `old_text` matches the real source exactly.\n\
         3. EDIT the source to resolve the issue. Make the smallest change \
         that fixes it. If an edit fails because `old_text` didn't match, \
         re-read the file and try again with the exact text.\n\
         4. Do NOT edit or add test files — the graders supply their own \
         tests. Ensure the code still imports/compiles cleanly.\n\n\
         ---\n\n{problem_statement}"
    )
}

/// Public entry point invoked by `main.rs`. Returns a process exit code:
/// 0 on clean completion, 2 on a setup error (dataset missing, no
/// instances selected, etc.).
pub async fn run(args: SweBenchArgs) -> Result<i32> {
    let dataset_path = match args.dataset.clone() {
        Some(p) => p,
        None => bail!(
            "cubi swebench requires --dataset <path.jsonl>. Export the \
             princeton-nlp/SWE-bench_Lite dataset to JSONL first."
        ),
    };
    if !dataset_path.exists() {
        bail!("dataset file not found at {}", dataset_path.display());
    }
    let raw = std::fs::read_to_string(&dataset_path)
        .with_context(|| format!("read dataset {}", dataset_path.display()))?;
    let all = parse_dataset(&raw)?;
    let instances = select_instances(all, args.instance.as_deref(), args.limit)?;
    if instances.is_empty() {
        bail!("no SWE-bench instances selected");
    }

    let model = args
        .model
        .clone()
        .or_else(|| std::env::var("CUBI_MODEL").ok())
        .unwrap_or_else(|| crate::DEFAULT_MODEL.to_string());

    let output_dir = args.output.clone().unwrap_or_else(default_output_dir);
    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("create swebench output dir {}", output_dir.display()))?;

    let started_at = now_iso();
    let wall_start = Instant::now();
    let cubi_bin = cubi_binary();

    if !args.json {
        eprintln!(
            "cubi swebench: dataset={} model={} instances={} score={} output={}",
            dataset_path.display(),
            model,
            instances.len(),
            args.score,
            output_dir.display()
        );
    }

    let mut predictions: Vec<Prediction> = Vec::with_capacity(instances.len());
    let mut results: Vec<InstanceResult> = Vec::with_capacity(instances.len());

    for inst in &instances {
        let (prediction, result) =
            run_one_instance(inst, &model, &cubi_bin, &args, &output_dir).await;
        if !args.json {
            eprintln!(
                "  [{}] {} ({} patch lines, {:.1}s)",
                resolution_glyph(result.resolution),
                result.instance_id,
                result.patch_lines,
                result.elapsed_seconds
            );
        }
        predictions.push(prediction);
        results.push(result);
    }

    // Write predictions in the official schema (one JSON object per line).
    let predictions_path = output_dir.join("predictions.jsonl");
    {
        let mut buf = String::new();
        for p in &predictions {
            buf.push_str(&serde_json::to_string(p)?);
            buf.push('\n');
        }
        std::fs::write(&predictions_path, &buf)
            .with_context(|| format!("write {}", predictions_path.display()))?;
    }

    let wall_time_seconds = wall_start.elapsed().as_secs_f64();
    let summary = summarize(
        results,
        &model,
        &dataset_path.display().to_string(),
        args.score,
        &started_at,
        wall_time_seconds,
    );
    let summary_path = output_dir.join("summary.json");
    let summary_json = serde_json::to_string_pretty(&summary)?;
    std::fs::write(&summary_path, &summary_json)
        .with_context(|| format!("write {}", summary_path.display()))?;

    if args.json {
        println!("{summary_json}");
    } else if args.score {
        eprintln!(
            "cubi swebench: {}/{} resolved ({:.1}%), {} with a patch, in {:.1}s\n\
             predictions: {}\nsummary: {}",
            summary.n_resolved,
            summary.n_instances,
            summary.resolved_pct,
            summary.n_with_patch,
            summary.wall_time_seconds,
            predictions_path.display(),
            summary_path.display()
        );
    } else {
        eprintln!(
            "cubi swebench: generated {} predictions ({} with a patch) in {:.1}s\n\
             predictions: {}\nScore them with the upstream harness:\n  \
             python -m swebench.harness.run_evaluation \\\n    \
             --predictions_path {} --dataset_name princeton-nlp/SWE-bench_Lite",
            summary.n_instances,
            summary.n_with_patch,
            summary.wall_time_seconds,
            predictions_path.display(),
            predictions_path.display()
        );
    }

    Ok(0)
}

fn resolution_glyph(r: Resolution) -> &'static str {
    match r {
        Resolution::Resolved => "RSLV",
        Resolution::Unresolved => "unrs",
        Resolution::NotScored => "gen ",
    }
}

/// Prepare a clean checkout of the instance repo at `base_commit`,
/// returning the working directory. Uses a copy of the pre-cloned repo
/// under `repos_root` when available, else clones from GitHub.
async fn prepare_repo(inst: &Instance, repos_root: Option<&Path>, dest: &Path) -> Result<()> {
    if let Some(root) = repos_root {
        // Try `<root>/<owner>__<name>` then `<root>/<name>`.
        let candidates = [
            root.join(repo_dir_name(&inst.repo)),
            root.join(inst.repo.rsplit('/').next().unwrap_or(&inst.repo)),
        ];
        let src = candidates.iter().find(|p| p.join(".git").exists());
        match src {
            Some(src) => {
                run_git(
                    dest.parent().unwrap_or(dest),
                    &[
                        "clone",
                        "--quiet",
                        &src.display().to_string(),
                        &dest.display().to_string(),
                    ],
                )
                .await
                .context("clone from repos-root cache")?;
            }
            None => bail!(
                "no pre-cloned repo for {} under {} (looked for {} / {})",
                inst.repo,
                root.display(),
                repo_dir_name(&inst.repo),
                inst.repo.rsplit('/').next().unwrap_or(&inst.repo),
            ),
        }
    } else {
        let url = format!("https://github.com/{}.git", inst.repo);
        run_git(
            dest.parent().unwrap_or(dest),
            &["clone", "--quiet", &url, &dest.display().to_string()],
        )
        .await
        .with_context(|| format!("git clone {url}"))?;
    }

    // Clean, detached checkout at the base commit.
    run_git(dest, &["checkout", "--quiet", "--force", &inst.base_commit])
        .await
        .with_context(|| format!("git checkout {}", inst.base_commit))?;
    run_git(dest, &["reset", "--hard", "--quiet", &inst.base_commit]).await?;
    run_git(dest, &["clean", "-fdq"]).await?;
    Ok(())
}

/// Run `git` in `dir`, returning an error if it exits non-zero.
async fn run_git(dir: &Path, args: &[&str]) -> Result<()> {
    let out = tokio::process::Command::new("git")
        .args(args)
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .with_context(|| format!("spawn git {}", args.join(" ")))?;
    if !out.status.success() {
        bail!(
            "git {} failed ({}): {}",
            args.join(" "),
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(())
}

/// Capture the working-tree diff (the agent's edit) as a unified patch.
async fn capture_diff(dir: &Path) -> Result<String> {
    // Stage everything so new files show up, then emit a diff against HEAD.
    run_git(dir, &["add", "-A"]).await?;
    let out = tokio::process::Command::new("git")
        .args(["diff", "--cached", "--no-color"])
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .context("git diff --cached")?;
    if !out.status.success() {
        bail!(
            "git diff failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

#[allow(clippy::too_many_arguments)]
async fn run_one_instance(
    inst: &Instance,
    model: &str,
    cubi_bin: &Path,
    args: &SweBenchArgs,
    output_dir: &Path,
) -> (Prediction, InstanceResult) {
    let start = Instant::now();
    let empty_pred = |patch: String| Prediction {
        instance_id: inst.instance_id.clone(),
        model_name_or_path: model.to_string(),
        model_patch: patch,
    };
    let err_result = |msg: String, elapsed: f64, code: Option<i32>| InstanceResult {
        instance_id: inst.instance_id.clone(),
        model: model.to_string(),
        resolution: Resolution::NotScored,
        patch_lines: 0,
        elapsed_seconds: elapsed,
        cubi_exit_code: code,
        fail_to_pass_passed: None,
        fail_to_pass_total: None,
        pass_to_pass_passed: None,
        pass_to_pass_total: None,
        error: Some(msg),
    };

    let workdir = match tempfile::Builder::new()
        .prefix(&format!("cubi-swe-{}-", sanitize(&inst.instance_id)))
        .tempdir()
    {
        Ok(d) => d,
        Err(e) => {
            return (
                empty_pred(String::new()),
                err_result(
                    format!("create tempdir: {e}"),
                    start.elapsed().as_secs_f64(),
                    None,
                ),
            );
        }
    };
    let repo_dir = workdir.path().join("repo");

    if let Err(e) = prepare_repo(inst, args.repos_root.as_deref(), &repo_dir).await {
        return (
            empty_pred(String::new()),
            err_result(
                format!("prepare repo: {e:#}"),
                start.elapsed().as_secs_f64(),
                None,
            ),
        );
    }

    // Drive Cubi headlessly inside the repo, HOME-isolated like `bench`.
    let home_dir = match tempfile::Builder::new().prefix("cubi-swe-home-").tempdir() {
        Ok(d) => d,
        Err(e) => {
            return (
                empty_pred(String::new()),
                err_result(
                    format!("create home tempdir: {e}"),
                    start.elapsed().as_secs_f64(),
                    None,
                ),
            );
        }
    };
    let events_path = output_dir.join(format!("{}.events.jsonl", sanitize(&inst.instance_id)));
    let prompt = build_prompt(&inst.problem_statement);

    // Cubi refuses writes outside a trusted root, so the agent can't
    // produce a patch unless the throwaway checkout is trusted. Seed a
    // `trusted_dirs.json` in the isolated HOME rather than prompting.
    if let Err(e) = seed_trust(home_dir.path(), &repo_dir) {
        return (
            empty_pred(String::new()),
            err_result(
                format!("seed trust: {e:#}"),
                start.elapsed().as_secs_f64(),
                None,
            ),
        );
    }

    let mut cmd = tokio::process::Command::new(cubi_bin);
    cmd.arg("-p")
        .arg(&prompt)
        .arg("--json")
        .arg("--no-stream")
        .arg("--no-banner")
        .arg("--quiet")
        .arg("--events")
        .arg(&events_path)
        .env("CUBI_MODEL", model)
        .env("HOME", home_dir.path())
        .env("USERPROFILE", home_dir.path())
        .env("CUBI_NO_BANNER", "1")
        .env("CUBI_NO_ONBOARD", "1")
        .current_dir(&repo_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    let cubi_exit_code: Option<i32> = match cmd.spawn() {
        Ok(mut child) => {
            let timeout = Duration::from_secs(args.time_cap_seconds);
            match tokio::time::timeout(timeout, child.wait()).await {
                Ok(Ok(exit)) => exit.code(),
                Ok(Err(e)) => {
                    let _ = child.kill().await;
                    return (
                        empty_pred(String::new()),
                        err_result(
                            format!("cubi wait error: {e}"),
                            start.elapsed().as_secs_f64(),
                            None,
                        ),
                    );
                }
                Err(_) => {
                    let _ = child.kill().await;
                    return (
                        empty_pred(String::new()),
                        err_result(
                            format!("exceeded time cap {}s", args.time_cap_seconds),
                            start.elapsed().as_secs_f64(),
                            None,
                        ),
                    );
                }
            }
        }
        Err(e) => {
            return (
                empty_pred(String::new()),
                err_result(
                    format!("spawn cubi: {e}"),
                    start.elapsed().as_secs_f64(),
                    None,
                ),
            );
        }
    };

    // Capture the agent's edit BEFORE any test patch is applied.
    let patch = match capture_diff(&repo_dir).await {
        Ok(p) => p,
        Err(e) => {
            return (
                empty_pred(String::new()),
                err_result(
                    format!("capture diff: {e:#}"),
                    start.elapsed().as_secs_f64(),
                    cubi_exit_code,
                ),
            );
        }
    };
    let patch_lines = patch.lines().count();

    let mut result = InstanceResult {
        instance_id: inst.instance_id.clone(),
        model: model.to_string(),
        resolution: Resolution::NotScored,
        patch_lines,
        elapsed_seconds: start.elapsed().as_secs_f64(),
        cubi_exit_code,
        fail_to_pass_passed: None,
        fail_to_pass_total: None,
        pass_to_pass_passed: None,
        pass_to_pass_total: None,
        error: None,
    };

    if args.score {
        match score_locally(inst, &repo_dir).await {
            Ok((resolution, f2p_ok, p2p_ok)) => {
                result.resolution = resolution;
                result.fail_to_pass_passed = Some(f2p_ok);
                result.fail_to_pass_total = Some(inst.fail_to_pass.len());
                result.pass_to_pass_passed = Some(p2p_ok);
                result.pass_to_pass_total = Some(inst.pass_to_pass.len());
            }
            Err(e) => result.error = Some(format!("score: {e:#}")),
        }
    }

    if args.keep_workdir {
        let kept = workdir.keep();
        let kept_home = home_dir.keep();
        eprintln!("cubi swebench: kept workdir at {}", kept.display());
        eprintln!("cubi swebench: kept home at {}", kept_home.display());
    }

    (empty_pred(patch), result)
}

/// Apply the instance `test_patch`, run the required node ids with pytest,
/// and score. Returns `(resolution, fail_to_pass_passed, pass_to_pass_passed)`.
async fn score_locally(inst: &Instance, repo_dir: &Path) -> Result<(Resolution, usize, usize)> {
    if !inst.test_patch.is_empty() {
        apply_patch(repo_dir, &inst.test_patch)
            .await
            .context("apply test_patch")?;
    }

    let mut node_ids: Vec<&str> = Vec::new();
    node_ids.extend(inst.fail_to_pass.iter().map(|s| s.as_str()));
    node_ids.extend(inst.pass_to_pass.iter().map(|s| s.as_str()));
    if node_ids.is_empty() {
        return Ok((Resolution::Unresolved, 0, 0));
    }

    let mut pytest_args: Vec<String> = vec![
        "-rA".to_string(),
        "-p".to_string(),
        "no:cacheprovider".to_string(),
        "--no-header".to_string(),
        "-q".to_string(),
    ];
    pytest_args.extend(node_ids.iter().map(|s| s.to_string()));

    let interpreter = python_interpreter();
    let out = tokio::process::Command::new(&interpreter)
        .arg("-m")
        .arg("pytest")
        .args(&pytest_args)
        .current_dir(repo_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .with_context(|| format!("run {interpreter} -m pytest"))?;
    let mut combined = String::from_utf8_lossy(&out.stdout).to_string();
    combined.push_str(&String::from_utf8_lossy(&out.stderr));
    let outcomes = parse_pytest_outcomes(&combined);

    let f2p_passed = count_passed(&inst.fail_to_pass, &outcomes);
    let p2p_passed = count_passed(&inst.pass_to_pass, &outcomes);
    let resolution = score_instance(&inst.fail_to_pass, &inst.pass_to_pass, &outcomes);
    Ok((resolution, f2p_passed, p2p_passed))
}

/// Resolve the Python interpreter for the local scorer. Honors
/// `CUBI_SWEBENCH_PYTHON`, then prefers `python3` (common on dev machines),
/// falling back to `python` (the alias inside the pinned Docker envs).
fn python_interpreter() -> String {
    if let Ok(p) = std::env::var("CUBI_SWEBENCH_PYTHON") {
        if !p.trim().is_empty() {
            return p;
        }
    }
    for cand in ["python3", "python"] {
        if std::process::Command::new(cand)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
        {
            return cand.to_string();
        }
    }
    "python".to_string()
}

/// Apply a unified diff via `git apply` (falls back to `patch -p1`).
async fn apply_patch(dir: &Path, patch: &str) -> Result<()> {
    let patch_file = dir.join(".cubi-test.patch");
    std::fs::write(&patch_file, patch).context("write patch file")?;
    let git_ok = tokio::process::Command::new("git")
        .args(["apply", "--whitespace=nowarn"])
        .arg(&patch_file)
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .map(|s| s.success())
        .unwrap_or(false);
    if !git_ok {
        let status = tokio::process::Command::new("patch")
            .args(["-p1", "-i"])
            .arg(&patch_file)
            .current_dir(dir)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .context("patch -p1")?;
        if !status.success() {
            bail!("failed to apply patch with git apply and patch -p1");
        }
    }
    let _ = std::fs::remove_file(&patch_file);
    Ok(())
}

/// Make an instance id safe for use as a filename / tempdir prefix.
fn sanitize(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

/// Seed a `trusted_dirs.json` in the isolated HOME so the headless Cubi
/// run may edit files in `repo_dir` without an interactive trust prompt.
/// Cubi refuses writes outside a trusted root; the path is canonicalized to
/// match Cubi's own trust comparison (which canonicalizes before checking).
fn seed_trust(home: &Path, repo_dir: &Path) -> Result<()> {
    let canonical = std::fs::canonicalize(repo_dir)
        .with_context(|| format!("canonicalize {}", repo_dir.display()))?;
    let cubi_dir = home.join(".cubi");
    std::fs::create_dir_all(&cubi_dir).with_context(|| format!("create {}", cubi_dir.display()))?;
    let trust = serde_json::json!({
        "trusted_roots": [canonical],
        "allowed_tools": [],
        "denied_tools": [],
    });
    std::fs::write(
        cubi_dir.join("trusted_dirs.json"),
        serde_json::to_string_pretty(&trust)?,
    )
    .context("write trusted_dirs.json")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{"instance_id":"astropy__astropy-12345","repo":"astropy/astropy","base_commit":"abc123","problem_statement":"Fix the bug in foo","patch":"","test_patch":"","FAIL_TO_PASS":"[\"tests/test_foo.py::test_a\"]","PASS_TO_PASS":"[\"tests/test_foo.py::test_b\", \"tests/test_foo.py::test_c\"]"}
{"instance_id":"django__django-999","repo":"django/django","base_commit":"def456","problem_statement":"Another issue","FAIL_TO_PASS":["t::x"],"PASS_TO_PASS":[]}"#;

    #[test]
    fn parse_dataset_accepts_string_and_array_forms() {
        let insts = parse_dataset(SAMPLE).unwrap();
        assert_eq!(insts.len(), 2);
        assert_eq!(insts[0].instance_id, "astropy__astropy-12345");
        assert_eq!(insts[0].fail_to_pass, vec!["tests/test_foo.py::test_a"]);
        assert_eq!(
            insts[0].pass_to_pass,
            vec!["tests/test_foo.py::test_b", "tests/test_foo.py::test_c"]
        );
        // Native-array form on the second line.
        assert_eq!(insts[1].fail_to_pass, vec!["t::x"]);
        assert!(insts[1].pass_to_pass.is_empty());
    }

    #[test]
    fn parse_dataset_skips_blank_lines() {
        let raw = format!("\n{}\n\n", SAMPLE.lines().next().unwrap());
        let insts = parse_dataset(&raw).unwrap();
        assert_eq!(insts.len(), 1);
    }

    #[test]
    fn select_by_instance_id() {
        let insts = parse_dataset(SAMPLE).unwrap();
        let picked = select_instances(insts, Some("django__django-999"), None).unwrap();
        assert_eq!(picked.len(), 1);
        assert_eq!(picked[0].repo, "django/django");
    }

    #[test]
    fn select_unknown_instance_errors() {
        let insts = parse_dataset(SAMPLE).unwrap();
        assert!(select_instances(insts, Some("nope"), None).is_err());
    }

    #[test]
    fn select_limit_truncates() {
        let insts = parse_dataset(SAMPLE).unwrap();
        let picked = select_instances(insts, None, Some(1)).unwrap();
        assert_eq!(picked.len(), 1);
    }

    #[test]
    fn repo_dir_name_uses_double_underscore() {
        assert_eq!(repo_dir_name("astropy/astropy"), "astropy__astropy");
        assert_eq!(repo_dir_name("psf/requests"), "psf__requests");
    }

    #[test]
    fn parse_pytest_outcomes_reads_short_summary() {
        let out = "\
=== short test summary info ===
PASSED tests/test_foo.py::test_a
FAILED tests/test_foo.py::test_b - assert 1 == 2
ERROR tests/test_foo.py::test_c
SKIPPED [1] tests/test_foo.py:12: needs network
";
        let m = parse_pytest_outcomes(out);
        assert_eq!(
            m.get("tests/test_foo.py::test_a"),
            Some(&TestOutcome::Passed)
        );
        assert_eq!(
            m.get("tests/test_foo.py::test_b"),
            Some(&TestOutcome::Failed)
        );
        assert_eq!(
            m.get("tests/test_foo.py::test_c"),
            Some(&TestOutcome::Errored)
        );
        // SKIPPED lines report file:line, not a node id, so they're ignored.
        assert_eq!(m.get("tests/test_foo.py::test_d"), None);
    }

    #[test]
    fn score_resolved_when_all_required_pass() {
        let mut m = BTreeMap::new();
        m.insert("f2p::a".to_string(), TestOutcome::Passed);
        m.insert("p2p::b".to_string(), TestOutcome::Passed);
        let r = score_instance(&["f2p::a".into()], &["p2p::b".into()], &m);
        assert_eq!(r, Resolution::Resolved);
    }

    #[test]
    fn score_unresolved_when_fail_to_pass_fails() {
        let mut m = BTreeMap::new();
        m.insert("f2p::a".to_string(), TestOutcome::Failed);
        m.insert("p2p::b".to_string(), TestOutcome::Passed);
        let r = score_instance(&["f2p::a".into()], &["p2p::b".into()], &m);
        assert_eq!(r, Resolution::Unresolved);
    }

    #[test]
    fn score_unresolved_when_regression_in_pass_to_pass() {
        let mut m = BTreeMap::new();
        m.insert("f2p::a".to_string(), TestOutcome::Passed);
        m.insert("p2p::b".to_string(), TestOutcome::Failed);
        let r = score_instance(&["f2p::a".into()], &["p2p::b".into()], &m);
        assert_eq!(r, Resolution::Unresolved);
    }

    #[test]
    fn score_unresolved_when_required_test_missing() {
        let m = BTreeMap::new();
        let r = score_instance(&["f2p::a".into()], &[], &m);
        assert_eq!(r, Resolution::Unresolved);
    }

    #[test]
    fn empty_fail_to_pass_is_unresolved() {
        let m = BTreeMap::new();
        assert_eq!(score_instance(&[], &[], &m), Resolution::Unresolved);
    }

    #[test]
    fn count_passed_counts_only_passing_required() {
        let mut m = BTreeMap::new();
        m.insert("a".to_string(), TestOutcome::Passed);
        m.insert("b".to_string(), TestOutcome::Failed);
        let n = count_passed(&["a".into(), "b".into(), "c".into()], &m);
        assert_eq!(n, 1);
    }

    #[test]
    fn prediction_serializes_official_schema() {
        let p = Prediction {
            instance_id: "x-1".into(),
            model_name_or_path: "qwen3:8b".into(),
            model_patch: "diff --git a b".into(),
        };
        let j = serde_json::to_string(&p).unwrap();
        assert!(j.contains("\"instance_id\":\"x-1\""));
        assert!(j.contains("\"model_name_or_path\":\"qwen3:8b\""));
        assert!(j.contains("\"model_patch\":\"diff --git a b\""));
    }

    #[test]
    fn summarize_computes_resolved_pct() {
        let mk = |id: &str, res: Resolution, lines: usize| InstanceResult {
            instance_id: id.into(),
            model: "m".into(),
            resolution: res,
            patch_lines: lines,
            elapsed_seconds: 1.0,
            cubi_exit_code: Some(0),
            fail_to_pass_passed: None,
            fail_to_pass_total: None,
            pass_to_pass_passed: None,
            pass_to_pass_total: None,
            error: None,
        };
        let results = vec![
            mk("a", Resolution::Resolved, 10),
            mk("b", Resolution::Unresolved, 4),
            mk("c", Resolution::Resolved, 6),
            mk("d", Resolution::NotScored, 0),
        ];
        let s = summarize(results, "m", "ds.jsonl", true, "2026-01-01T00:00:00Z", 12.0);
        assert_eq!(s.n_instances, 4);
        assert_eq!(s.n_with_patch, 3);
        assert_eq!(s.n_resolved, 2);
        assert!((s.resolved_pct - 50.0).abs() < 1e-9);
        assert!(s.scored);
    }

    #[test]
    fn build_prompt_includes_issue_and_directives() {
        let p = build_prompt("The widget crashes on None");
        assert!(p.contains("The widget crashes on None"));
        assert!(p.to_lowercase().contains("do not edit"));
        // The explore-first workflow is the key anti-path-guessing directive.
        assert!(p.to_lowercase().contains("explore"));
        assert!(p.contains("repo_map"));
    }

    #[test]
    fn sanitize_replaces_unsafe_chars() {
        assert_eq!(sanitize("astropy/astropy-12"), "astropy-astropy-12");
        assert_eq!(sanitize("a b:c"), "a-b-c");
    }

    #[test]
    fn seed_trust_writes_canonical_repo_path() {
        let home = tempfile::tempdir().unwrap();
        let repo = tempfile::tempdir().unwrap();
        seed_trust(home.path(), repo.path()).unwrap();
        let raw =
            std::fs::read_to_string(home.path().join(".cubi").join("trusted_dirs.json")).unwrap();
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let canonical = std::fs::canonicalize(repo.path()).unwrap();
        let roots = v["trusted_roots"].as_array().unwrap();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].as_str().unwrap(), canonical.to_str().unwrap());
    }
}
