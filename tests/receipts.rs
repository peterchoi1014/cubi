//! Smoke tests for `--receipts` and `cubi verify-receipts`. The
//! module-level unit tests in `src/receipts.rs` cover the chain math
//! directly; these exercise the CLI end-to-end via the fake LLM
//! backend so the wiring in `main.rs` and `src/cli/agent.rs` can't
//! regress silently.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

fn cubi(home: &Path) -> Command {
    let mut cmd = Command::cargo_bin("cubi").expect("cubi binary is built by cargo test");
    cmd.env("HOME", home)
        .env("USERPROFILE", home)
        .env("CUBI_NO_ONBOARD", "1")
        .env("CUBI_COLOR", "off");
    cmd
}

fn trust_cwd(home: &Path) {
    let cubi_dir = home.join(".cubi");
    fs::create_dir_all(&cubi_dir).unwrap();
    let cwd = std::env::current_dir().unwrap().canonicalize().unwrap();
    fs::write(
        cubi_dir.join("trusted_dirs.json"),
        serde_json::json!({"trusted_roots": [cwd]}).to_string(),
    )
    .unwrap();
}

fn sidecar_dir(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap().to_os_string();
    name.push(".payloads");
    path.parent().unwrap().join(name)
}

#[test]
fn receipts_flag_writes_chain_and_verify_returns_ok() {
    let home = tempdir().unwrap();
    trust_cwd(home.path());
    let receipts_path = home.path().join("r.jsonl");

    cubi(home.path())
        .env("CUBI_FAKE_LLM", "1")
        .env("CUBI_FAKE_LLM_RESPONSE", "done")
        .env(
            "CUBI_FAKE_LLM_TOOL_CALL",
            r#"{"function":{"name":"list_files","arguments":{"path":"."}}}"#,
        )
        .args([
            "--receipts",
            receipts_path.to_str().unwrap(),
            "--no-stream",
            "--no-banner",
            "-p",
            "hi",
        ])
        .assert()
        .success();

    let raw = fs::read_to_string(&receipts_path).expect("receipts file exists");
    let lines: Vec<&str> = raw.lines().collect();
    assert!(lines.len() >= 4, "expected ≥4 receipts, got: {raw}");
    let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(first["event"], "session_start");
    assert_eq!(first["seq"], 1);
    let last: serde_json::Value = serde_json::from_str(lines.last().unwrap()).unwrap();
    assert_eq!(last["event"], "session_end");

    // Tool call + result must be present.
    assert!(
        lines
            .iter()
            .any(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["event"] == "tool_call"),
        "expected a tool_call entry, got: {raw}"
    );
    assert!(
        lines.iter().any(
            |l| serde_json::from_str::<serde_json::Value>(l).unwrap()["event"] == "tool_result"
        ),
        "expected a tool_result entry, got: {raw}"
    );

    // verify-receipts must return 0 on a clean chain.
    cubi(home.path())
        .args(["verify-receipts", receipts_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("OK:"));
}

#[test]
fn verify_receipts_detects_payload_tampering() {
    let home = tempdir().unwrap();
    trust_cwd(home.path());
    let receipts_path = home.path().join("r.jsonl");

    cubi(home.path())
        .env("CUBI_FAKE_LLM", "1")
        .env("CUBI_FAKE_LLM_RESPONSE", "done")
        .env(
            "CUBI_FAKE_LLM_TOOL_CALL",
            r#"{"function":{"name":"list_files","arguments":{"path":"."}}}"#,
        )
        .args([
            "--receipts",
            receipts_path.to_str().unwrap(),
            "--no-stream",
            "--no-banner",
            "-p",
            "hi",
        ])
        .assert()
        .success();

    // Mutate one payload sidecar.
    let dir = sidecar_dir(&receipts_path);
    let entry = fs::read_dir(&dir).unwrap().next().unwrap().unwrap();
    fs::write(entry.path(), r#"{"tampered": true}"#).unwrap();

    cubi(home.path())
        .args(["verify-receipts", receipts_path.to_str().unwrap()])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("seq="));
}

#[test]
fn verify_receipts_json_output_shape() {
    let home = tempdir().unwrap();
    trust_cwd(home.path());
    let receipts_path = home.path().join("r.jsonl");

    cubi(home.path())
        .env("CUBI_FAKE_LLM", "1")
        .env("CUBI_FAKE_LLM_RESPONSE", "done")
        .args([
            "--receipts",
            receipts_path.to_str().unwrap(),
            "--no-stream",
            "--no-banner",
            "-p",
            "hello",
        ])
        .assert()
        .success();

    let out = cubi(home.path())
        .args(["verify-receipts", receipts_path.to_str().unwrap(), "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let parsed: serde_json::Value = serde_json::from_slice(&out).expect("valid JSON");
    assert_eq!(parsed["ok"], true);
    assert!(parsed["entries"].as_u64().unwrap() >= 3);
}

#[test]
fn keys_init_generates_keypair_and_signs_subsequent_receipts() {
    let home = tempdir().unwrap();
    trust_cwd(home.path());

    cubi(home.path())
        .args(["keys", "init"])
        .assert()
        .success()
        .stdout(predicate::str::contains("ssh-ed25519"));

    let priv_path = home.path().join(".cubi/keys/ed25519.priv");
    let pub_path = home.path().join(".cubi/keys/ed25519.pub");
    assert!(priv_path.exists(), "private key missing");
    assert!(pub_path.exists(), "public key missing");

    // Refusal without --force.
    cubi(home.path())
        .args(["keys", "init"])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("already exists"));

    let receipts_path = home.path().join("r.jsonl");
    cubi(home.path())
        .env("CUBI_FAKE_LLM", "1")
        .env("CUBI_FAKE_LLM_RESPONSE", "ok")
        .args([
            "--receipts",
            receipts_path.to_str().unwrap(),
            "--no-stream",
            "--no-banner",
            "-p",
            "signed?",
        ])
        .assert()
        .success();

    // Every entry must carry a `sig` field once a key exists.
    let raw = fs::read_to_string(&receipts_path).unwrap();
    for line in raw.lines() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        assert!(v.get("sig").is_some(), "missing sig in {line}");
    }

    // verify-receipts --pub-key must accept the bundled public key.
    cubi(home.path())
        .args([
            "verify-receipts",
            receipts_path.to_str().unwrap(),
            "--pub-key",
            pub_path.to_str().unwrap(),
        ])
        .assert()
        .success();
}
