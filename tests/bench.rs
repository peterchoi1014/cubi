//! Integration smoke tests for the `cubi bench` subcommand. These do
//! **not** spawn the Cubi binary (that would require a live local
//! model). They just verify task discovery + CLI plumbing.
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
