use assert_cmd::Command;
use predicates::prelude::*;
use std::path::Path;
use tempfile::tempdir;

fn cubi(home: &Path) -> Command {
    let mut cmd = Command::cargo_bin("cubi").expect("cubi binary is built by cargo test");
    cmd.env("HOME", home)
        .env("USERPROFILE", home)
        .env("CUBI_NO_ONBOARD", "1")
        .env("CUBI_COLOR", "off");
    cmd
}

#[test]
fn empty_prompt_exits_with_usage_error() {
    let home = tempdir().unwrap();

    cubi(home.path())
        .args(["-p", ""])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains(
            "cubi: --prompt/-p requires non-empty inline prompt text.",
        ));
}

#[test]
fn prompt_and_resume_are_mutually_exclusive() {
    let home = tempdir().unwrap();

    cubi(home.path())
        .args(["--prompt=", "--resume", "foo"])
        .assert()
        .failure()
        .code(2);
}

#[test]
fn whitespace_stdin_exits_with_usage_error() {
    let home = tempdir().unwrap();

    cubi(home.path())
        .write_stdin("  \n\t\n")
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("cubi: stdin prompt was empty."));
}

#[test]
fn version_prints_package_version() {
    let home = tempdir().unwrap();

    cubi(home.path())
        .arg("--version")
        .assert()
        .success()
        .stdout("cubi 0.3.0\n");
}

#[test]
fn help_mentions_prompt_and_resume() {
    let home = tempdir().unwrap();

    cubi(home.path())
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("--prompt").and(predicate::str::contains("--resume")));
}

#[test]
fn short_version_alias_matches_long_version() {
    let home = tempdir().unwrap();

    cubi(home.path())
        .arg("-v")
        .assert()
        .success()
        .stdout("cubi 0.3.0\n");
}

#[test]
fn list_sessions_succeeds_with_clean_home() {
    let home = tempdir().unwrap();

    cubi(home.path())
        .arg("--list-sessions")
        .assert()
        .success()
        .stdout(
            predicate::str::contains("ID").and(predicate::str::contains("(no sessions saved yet)")),
        );
}

#[test]
fn delete_nonexistent_session_reports_not_found() {
    let home = tempdir().unwrap();

    cubi(home.path())
        .args(["--delete-session", "nonexistent-id"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "no session matches 'nonexistent-id'",
        ));
}

#[test]
fn bash_completions_emit_function() {
    let home = tempdir().unwrap();

    cubi(home.path())
        .args(["completions", "bash"])
        .assert()
        .success()
        .stdout(predicate::str::contains("_cubi"));
}

#[test]
fn unknown_completion_shell_exits_with_usage_error() {
    let home = tempdir().unwrap();

    cubi(home.path())
        .args(["completions", "tcsh"])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains(
            "cubi: completions requires one of: bash, zsh, fish.",
        ));
}
