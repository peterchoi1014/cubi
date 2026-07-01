//! Integration smoke tests for the `cubi swebench` subcommand. These
//! spawn the Cubi binary via `assert_cmd` and only verify CLI plumbing
//! (help text, argument validation, dataset handling). They do **not**
//! run any instance end-to-end against a live model — that requires
//! Ollama and network access to clone repos.

use assert_cmd::Command;
use predicates::str::contains;

#[test]
fn swebench_help_prints_usage() {
    Command::cargo_bin("cubi")
        .unwrap()
        .args(["swebench", "--help"])
        .assert()
        .success()
        .stdout(contains("cubi swebench"))
        .stdout(contains("--dataset"))
        .stdout(contains("--score"));
}

#[test]
fn swebench_requires_dataset() {
    Command::cargo_bin("cubi")
        .unwrap()
        .args(["swebench"])
        .assert()
        .failure()
        .stderr(contains("--dataset"));
}

#[test]
fn swebench_rejects_missing_dataset_file() {
    Command::cargo_bin("cubi")
        .unwrap()
        .args(["swebench", "--dataset", "/no/such/dataset.jsonl"])
        .assert()
        .failure()
        .stderr(contains("dataset file not found"));
}

#[test]
fn swebench_rejects_bad_limit() {
    Command::cargo_bin("cubi")
        .unwrap()
        .args(["swebench", "--dataset", "x.jsonl", "--limit", "notanumber"])
        .assert()
        .failure()
        .stderr(contains("--limit"));
}

#[test]
fn swebench_rejects_unknown_flag() {
    Command::cargo_bin("cubi")
        .unwrap()
        .args(["swebench", "--nope"])
        .assert()
        .failure()
        .stderr(contains("unexpected argument"));
}

#[test]
fn swebench_reports_no_instances_for_empty_dataset() {
    let dir = tempfile::tempdir().unwrap();
    let ds = dir.path().join("empty.jsonl");
    std::fs::write(&ds, "\n\n").unwrap();
    Command::cargo_bin("cubi")
        .unwrap()
        .args(["swebench", "--dataset"])
        .arg(&ds)
        .assert()
        .failure()
        .stderr(contains("no SWE-bench instances selected"));
}
