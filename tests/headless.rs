use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
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
        .args(["--prompt=hello", "--resume", "foo"])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains(
            "cubi: --prompt cannot be combined with --resume, --list-sessions, or --delete-session.",
        ));
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
fn list_sessions_json_outputs_session_shape() {
    let home = tempdir().unwrap();
    let bucket = home.path().join(".cubi").join("sessions").join("bucket");
    fs::create_dir_all(&bucket).unwrap();
    fs::write(
        bucket.join("20250101-000000-abcd.json"),
        r#"{
  "id": "20250101-000000-abcd",
  "started_at": 1735689600,
  "cwd": "/work/project",
  "model": "qwen3:4b",
  "history": [
    {"role":"user","content":"hello json"},
    {"role":"assistant","content":"hi"}
  ]
}"#,
    )
    .unwrap();

    let output = cubi(home.path())
        .args(["--list-sessions", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
    let first = &value.as_array().unwrap()[0];
    assert_eq!(first["id"], "20250101-000000-abcd");
    assert_eq!(first["model"], "qwen3:4b");
    assert_eq!(first["started_at"], 1735689600);
    assert_eq!(first["message_count"], 2);
    assert_eq!(first["cwd"], "/work/project");
    assert_eq!(first["preview"], "hello json");
    assert!(first.get("modified_at").is_some());
    assert!(first.get("path").is_none());
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
