//! Integration smoke tests for the `cubi bench` subcommand. These
//! spawn the Cubi binary via `assert_cmd`, but do **not** run any task
//! end-to-end against a live model. They just verify task discovery +
//! CLI plumbing.
//!
//! End-to-end runs against a real model live in the nightly
//! `.github/workflows/bench.yml` job.

use assert_cmd::Command;
use predicates::str::contains;

#[test]
fn bench_help_prints_usage() {
    Command::cargo_bin("cubi")
        .unwrap()
        .args(["bench", "--help"])
        .assert()
        .success()
        .stdout(contains("cubi bench"))
        .stdout(contains("--suite"))
        .stdout(contains("--task"));
}

#[test]
fn bench_rejects_unknown_suite() {
    Command::cargo_bin("cubi")
        .unwrap()
        .args(["bench", "--suite", "bogus"])
        .assert()
        .failure()
        .stderr(contains("--suite"));
}

#[test]
fn bench_rejects_unknown_flag() {
    Command::cargo_bin("cubi")
        .unwrap()
        .args(["bench", "--nope"])
        .assert()
        .failure()
        .stderr(contains("unexpected argument"));
}

/// End-to-end scoring run driven by the `Fake` backend, so it needs no local
/// model — only `cargo` (already required to build Cubi) for `verify.sh`.
///
/// This is a regression guard for three defects that each independently
/// pinned the suite at 0% for *every* model, making the nightly score
/// meaningless:
///
/// 1. the spawned agent got an isolated `HOME` with an empty trust store, so
///    headless mode auto-denied every `edit_file` call;
/// 2. `verify.sh` was invoked through the default *relative* `bench/tasks`
///    path while cwd was the throwaway workdir, so `sh` exited 2 without
///    ever running the tests;
/// 3. `--events` was likewise relative, so event logs were written inside
///    the workdir and discarded with it.
///
/// A scripted `edit_file` applies the real one-token fix for
/// `rust-typo-fix`, so a healthy harness must score it 100%.
///
/// Unix-only: `verify.sh` is POSIX sh (see `bench/README.md`) and shells out
/// to `cargo` through it. On Windows that lookup fails with 127, scoring the
/// task `fail` for a reason unrelated to what this test asserts — and Cubi's
/// bench suite targets macOS/Linux, so gate it rather than pretend otherwise.
#[cfg(not(windows))]
#[test]
fn bench_scores_a_pass_when_the_agent_fixes_the_task() {
    let leaked_cubi_home = tempfile::tempdir().unwrap();
    Command::cargo_bin("cubi")
        .unwrap()
        .args([
            "bench",
            "--task",
            "rust-typo-fix",
            "--model",
            "fake-model",
            "--json",
        ])
        .env("CUBI_HOME", leaked_cubi_home.path())
        .env("CUBI_FAKE_LLM", "1")
        .env(
            "CUBI_FAKE_LLM_TOOL_CALL",
            r#"{"id":"call_1","type":"function","function":{"name":"edit_file",
                "arguments":{"path":"src/lib.rs","old_text":"prnitln!","new_text":"println!"}}}"#,
        )
        .assert()
        .success()
        .stdout(contains("\"status\": \"pass\""))
        .stdout(contains("\"score_pct\": 100.0"));
}

#[cfg(not(windows))]
#[test]
fn bench_reports_verifier_timeout_as_infrastructure_error() {
    let tasks = tempfile::tempdir().unwrap();
    let task = tasks.path().join("timeout-task");
    std::fs::create_dir_all(task.join("repo")).unwrap();
    std::fs::write(
        task.join("task.toml"),
        r#"id = "timeout-task"
difficulty = "easy"
description = "timeout fixture"
prompt = "finish"
time_cap_seconds = 10
step_cap = 1
"#,
    )
    .unwrap();
    std::fs::write(task.join("verify.sh"), "#!/bin/sh\nwhile :; do :; done\n").unwrap();

    Command::cargo_bin("cubi")
        .unwrap()
        .args(["bench", "--task", "timeout-task", "--tasks-root"])
        .arg(tasks.path())
        .args(["--model", "fake-model", "--json"])
        .env("CUBI_FAKE_LLM", "1")
        .env("CUBI_FAKE_LLM_RESPONSE", "done")
        .env("CUBI_BENCH_VERIFY_TIMEOUT_SECS", "1")
        .assert()
        .code(2)
        .stdout(contains("\"status\": \"error\""))
        .stdout(contains("verifier timed out"));
}

#[cfg(not(windows))]
#[test]
fn bench_bounds_captured_verifier_diagnostics() {
    let tasks = tempfile::tempdir().unwrap();
    let task = tasks.path().join("diagnostic-task");
    std::fs::create_dir_all(task.join("repo")).unwrap();
    std::fs::write(
        task.join("task.toml"),
        r#"id = "diagnostic-task"
difficulty = "easy"
description = "diagnostic fixture"
prompt = "finish"
time_cap_seconds = 10
step_cap = 1
"#,
    )
    .unwrap();
    std::fs::write(
        task.join("verify.sh"),
        "#!/bin/sh\n\
         awk 'BEGIN { for (i = 0; i < 5000; i++) printf \"o\"; print \"OUT_END\" }'\n\
         awk 'BEGIN { for (i = 0; i < 5000; i++) printf \"e\"; print \"ERR_END\" }' >&2\n\
         exit 1\n",
    )
    .unwrap();

    let output = Command::cargo_bin("cubi")
        .unwrap()
        .args(["bench", "--task", "diagnostic-task", "--tasks-root"])
        .arg(tasks.path())
        .args(["--model", "fake-model", "--json"])
        .env("CUBI_FAKE_LLM", "1")
        .env("CUBI_FAKE_LLM_RESPONSE", "done")
        .assert()
        .code(2)
        .get_output()
        .stdout
        .clone();
    let summary: serde_json::Value = serde_json::from_slice(&output).unwrap();
    let result = &summary["per_task"][0];
    let stdout = result["verifier_stdout_tail"].as_str().unwrap();
    let stderr = result["verifier_stderr_tail"].as_str().unwrap();
    assert!(
        stdout.len() <= 4096,
        "stdout tail was {} bytes",
        stdout.len()
    );
    assert!(
        stderr.len() <= 4096,
        "stderr tail was {} bytes",
        stderr.len()
    );
    assert!(stdout.ends_with("OUT_END\n"));
    assert!(stderr.ends_with("ERR_END\n"));
}

/// Negative control for the test above: an agent that only *reads* the file
/// must still score 0%, and the recorded `verify_exit_code` must be the real
/// `cargo test` failure (101) rather than a harness-level path error. Without
/// this, a harness that scored everything as a pass would look "fixed".
///
/// Unix-only for the same reason as the test above: on Windows the task fails
/// with 127 (no `cargo` under POSIX sh), which would satisfy the "fail"
/// assertion for the wrong reason and assert nothing about the harness.
#[cfg(not(windows))]
#[test]
fn bench_scores_a_fail_when_the_agent_does_not_fix_the_task() {
    Command::cargo_bin("cubi")
        .unwrap()
        .args([
            "bench",
            "--task",
            "rust-typo-fix",
            "--model",
            "fake-model",
            "--json",
        ])
        .env("CUBI_FAKE_LLM", "1")
        .env(
            "CUBI_FAKE_LLM_TOOL_CALL",
            r#"{"id":"call_1","type":"function","function":{"name":"read_file",
                "arguments":{"path":"src/lib.rs"}}}"#,
        )
        .assert()
        // `cubi bench` exits 2 when any task fails — see `bench::run`.
        .code(2)
        .stdout(contains("\"status\": \"fail\""))
        .stdout(contains("\"score_pct\": 0.0"))
        .stdout(contains("\"verify_exit_code\": 101"));
}
