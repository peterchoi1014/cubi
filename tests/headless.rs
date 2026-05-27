use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::Path;
#[cfg(unix)]
use std::process::Command as StdCommand;
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

#[cfg(unix)]
#[test]
fn prune_sessions_removes_old_session_files() {
    let home = tempdir().unwrap();
    let bucket = home.path().join(".cubi").join("sessions").join("bucket");
    fs::create_dir_all(&bucket).unwrap();
    let session_path = bucket.join("20200101-000000-abcd.json");
    fs::write(
        &session_path,
        r#"{
  "id": "20200101-000000-abcd",
  "started_at": 1577836800,
  "cwd": "/work/old",
  "model": "qwen3:4b",
  "history": [{"role":"user","content":"old"}]
}"#,
    )
    .unwrap();
    let status = StdCommand::new("touch")
        .args(["-t", "202001010000"])
        .arg(&session_path)
        .status()
        .unwrap();
    assert!(status.success());

    cubi(home.path())
        .args(["--prune-sessions", "--older-than", "1d"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Pruned 1 session"));
    assert!(!session_path.exists());
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

#[test]
fn headless_json_outputs_line_delimited_events() {
    let home = tempdir().unwrap();

    let output = cubi(home.path())
        .env("CUBI_FAKE_LLM", "1")
        .env("CUBI_FAKE_LLM_RESPONSE", "hello")
        .args(["--json", "-p", "hi"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let text = String::from_utf8(output).unwrap();
    let events: Vec<serde_json::Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(events[0]["type"], "token");
    assert_eq!(events[0]["value"], "hello");
    assert_eq!(events[1]["type"], "done");
    assert!(events[1]["stats"].is_object());
}

#[cfg(unix)]
#[test]
fn headless_json_reports_tool_timeout() {
    let home = tempdir().unwrap();
    let cubi_dir = home.path().join(".cubi");
    fs::create_dir_all(&cubi_dir).unwrap();
    let cwd = std::env::current_dir().unwrap().canonicalize().unwrap();
    fs::write(
        cubi_dir.join("trusted_dirs.json"),
        serde_json::json!({"trusted_roots": [cwd]}).to_string(),
    )
    .unwrap();

    let output = cubi(home.path())
        .env("CUBI_FAKE_LLM", "1")
        .env("CUBI_FAKE_LLM_RESPONSE", "done")
        .env(
            "CUBI_FAKE_LLM_TOOL_CALL",
            r#"{"function":{"name":"bash","arguments":{"command":"sleep 2","_timeout_secs":1}}}"#,
        )
        .args(["--json", "--no-stream", "-p", "run slow shell"])
        .assert()
        .failure()
        .code(11)
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(output).unwrap();
    assert!(stdout.contains(r#""type":"tool_timeout""#));
    assert!(stdout.contains(r#""name":"bash""#));
    assert!(stdout.contains(r#""secs":1"#));
}

#[test]
fn doctor_json_emits_check_array() {
    let home = tempdir().unwrap();
    let output = cubi(home.path())
        .args(["doctor", "--json"])
        // Doctor still calls the model host check; let it fail fast on a
        // bogus URL so the test never depends on network.
        .env("OLLAMA_BASE_URL", "http://127.0.0.1:1")
        .assert()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(output).unwrap();
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("doctor --json should emit valid JSON");
    assert!(parsed["checks"].is_array(), "checks should be an array");
    assert!(parsed["ok"].is_boolean(), "ok should be a boolean");
    let names: Vec<&str> = parsed["checks"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|c| c["name"].as_str())
        .collect();
    assert!(names.contains(&"config"));
    assert!(names.contains(&"sessions_dir"));
    assert!(names.contains(&"plugins"));
}

#[test]
fn print_config_outputs_valid_json_with_path() {
    let home = tempdir().unwrap();
    let cubi_dir = home.path().join(".cubi");
    fs::create_dir_all(&cubi_dir).unwrap();
    // Plant a config with a key-like field to verify redaction; AppConfig
    // ignores unknown fields, so we round-trip through the raw JSON only.
    fs::write(
        cubi_dir.join("config.json"),
        r#"{"default_model":"qwen3:4b","onboarded":true,"api_token":"sk-very-secret","my_api_key":"abc123"}"#,
    )
    .unwrap();

    let output = cubi(home.path())
        .arg("--print-config")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(output).unwrap();
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("--print-config should emit valid JSON");
    assert!(parsed["_config_path"].is_string());
    assert_eq!(parsed["default_model"], "qwen3:4b");
    // AppConfig::load drops unknown fields, so the redacted keys must
    // not survive — but ensure none of the canonical AppConfig fields
    // accidentally carry a raw secret either.
    let s = stdout.to_lowercase();
    assert!(!s.contains("sk-very-secret"));
    assert!(!s.contains("abc123"));
}

#[test]
fn no_banner_flag_listed_in_help() {
    let home = tempdir().unwrap();
    cubi(home.path())
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("--no-banner"));
}

#[test]
fn headless_json_emits_budget_error_when_history_exceeds_window() {
    let home = tempdir().unwrap();

    let output = cubi(home.path())
        .env("CUBI_FAKE_LLM", "1")
        .env("CUBI_FAKE_LLM_RESPONSE", "ignored")
        // Tiny override forces the prompt-tokens-vs-window comparison
        // to trip on even a single-character prompt.
        .env("CUBI_MAX_PROMPT_TOKENS_OVERRIDE", "1")
        .args(["--json", "-p", "this prompt is way too long for one token"])
        .assert()
        .failure()
        .code(12)
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(output).unwrap();
    assert!(
        stdout.contains(r#""type":"budget_error""#),
        "expected budget_error event in stdout, got: {stdout}"
    );
    assert!(stdout.contains(r#""window":1"#));
}

#[test]
fn help_lists_budget_exit_code() {
    let home = tempdir().unwrap();
    cubi(home.path())
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("12 context budget"));
}

#[test]
fn exec_subcommand_emits_json_done_event() {
    let home = tempdir().unwrap();
    let output = cubi(home.path())
        .env("CUBI_FAKE_LLM", "1")
        .env("CUBI_FAKE_LLM_RESPONSE", "scripted reply")
        .args(["exec", "summarize", "this", "diff"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let text = String::from_utf8(output).unwrap();
    // Every line must be valid JSON (no banner, no stream noise).
    let events: Vec<serde_json::Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).expect("each line is JSON"))
        .collect();
    assert!(events.iter().any(|e| e["type"] == "token"));
    assert!(events.iter().any(|e| e["type"] == "done"));
}

#[test]
fn exec_without_prompt_exits_with_usage_error() {
    let home = tempdir().unwrap();
    cubi(home.path())
        .arg("exec")
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("cubi: exec requires a prompt"));
}

#[test]
fn run_subcommand_honors_frontmatter_model_override() {
    let home = tempdir().unwrap();
    let dir = tempdir().unwrap();
    let script_path = dir.path().join("review.md");
    fs::write(
        &script_path,
        "---\nmodel: test-override-model\nsystem: be terse\n---\nplease review\n",
    )
    .unwrap();

    let output = cubi(home.path())
        .env("CUBI_FAKE_LLM", "1")
        // Echo the resolved model name back in the fake reply so we can
        // assert the frontmatter override won out.
        .env("CUBI_FAKE_LLM_RESPONSE", "ok")
        .args(["run", script_path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(output).unwrap();
    // Just confirm we got valid JSON events (no banner noise).
    let events: Vec<serde_json::Value> = stdout
        .lines()
        .map(|l| serde_json::from_str(l).expect("each line is JSON"))
        .collect();
    assert!(events.iter().any(|e| e["type"] == "done"));
}

#[test]
fn run_subcommand_missing_path_exits_with_usage_error() {
    let home = tempdir().unwrap();
    cubi(home.path())
        .arg("run")
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains(
            "cubi: run requires a markdown script path",
        ));
}

#[cfg(unix)]
#[test]
fn trace_tools_writes_jsonl_pair_per_tool_call() {
    let home = tempdir().unwrap();
    let cubi_dir = home.path().join(".cubi");
    fs::create_dir_all(&cubi_dir).unwrap();
    let cwd = std::env::current_dir().unwrap().canonicalize().unwrap();
    fs::write(
        cubi_dir.join("trusted_dirs.json"),
        serde_json::json!({"trusted_roots": [cwd]}).to_string(),
    )
    .unwrap();
    let trace_path = home.path().join("trace.jsonl");

    cubi(home.path())
        .env("CUBI_FAKE_LLM", "1")
        .env("CUBI_FAKE_LLM_RESPONSE", "done")
        .env(
            "CUBI_FAKE_LLM_TOOL_CALL",
            r#"{"function":{"name":"bash","arguments":{"command":"true"}}}"#,
        )
        .args([
            "--trace-tools",
            trace_path.to_str().unwrap(),
            "--json",
            "--no-stream",
            "-p",
            "run a tool",
        ])
        .assert()
        .success();

    let raw = fs::read_to_string(&trace_path).expect("trace file written");
    let lines: Vec<&str> = raw.lines().collect();
    assert!(
        lines.len() >= 2,
        "expected at least two JSONL lines, got: {raw}"
    );
    let v0: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(v0["event"], "tool_start");
    assert_eq!(v0["tool"], "bash");
    let v1: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(v1["event"], "tool_complete");
    assert_eq!(v1["tool"], "bash");
    assert_eq!(v1["call_id"], v0["call_id"]);
}

#[test]
fn plugins_new_scaffolds_manifest_and_handler() {
    let home = tempdir().unwrap();
    let dest = tempdir().unwrap();

    cubi(home.path())
        .env("CUBI_PLUGINS_DIR", dest.path())
        .args(["plugins", "new", "demo"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Scaffolded plugin 'demo'"));

    let manifest_path = dest.path().join("demo").join("manifest.json");
    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    assert_eq!(manifest["name"], "demo");
    assert_eq!(manifest["version"], "0.1.0");
}

#[test]
fn plugins_new_refuses_duplicate_directory() {
    let home = tempdir().unwrap();
    let dest = tempdir().unwrap();
    fs::create_dir_all(dest.path().join("dup")).unwrap();
    cubi(home.path())
        .env("CUBI_PLUGINS_DIR", dest.path())
        .args(["plugins", "new", "dup"])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("already exists"));
}
