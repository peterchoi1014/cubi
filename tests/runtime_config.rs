use assert_cmd::Command;
use serde_json::{Value, json};
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn cubi(home: &Path) -> Command {
    let mut command = Command::cargo_bin("cubi").expect("cubi binary is built by cargo test");
    for (name, _) in std::env::vars_os() {
        if name.to_str().is_some_and(|name| {
            name.starts_with("CUBI_")
                || name.starts_with("AICHAT_")
                || name.starts_with("AI_CHAT_CLI_")
        }) {
            command.env_remove(name);
        }
    }
    command
        .current_dir(home)
        .env("HOME", home)
        .env("USERPROFILE", home)
        .env("CUBI_HOME", home)
        .env("CUBI_NO_ONBOARD", "1")
        .env("CUBI_COLOR", "off")
        .env("CUBI_FAKE_LLM", "")
        .env("CUBI_FAKE_LLM_RESPONSE", "fallback-response")
        .env(
            "CUBI_FAKE_LLM_MODEL_RESPONSES",
            json!({
                "script-model": "script-response",
                "env-model": "env-response",
                "legacy-model": "legacy-response",
                "saved-model": "saved-response",
                "qwen3.5:9b": "default-response",
            })
            .to_string(),
        )
        .args(["--json", "--no-stream", "--no-banner"]);
    command
}

fn response(command: &mut Command) -> String {
    let output = command.assert().success().get_output().stdout.clone();
    let events: Vec<Value> = String::from_utf8(output)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).expect("each output line is a JSON event"))
        .collect();
    assert_eq!(events.last().unwrap()["type"], "done");
    events
        .iter()
        .filter(|event| event["type"] == "token")
        .map(|event| event["value"].as_str().expect("token text"))
        .collect()
}

#[test]
fn startup_model_precedence_reaches_the_provider() {
    for (script_model, env_model, legacy_model, saved_model, expected) in [
        (
            Some("script-model"),
            Some("env-model"),
            Some("legacy-model"),
            Some("saved-model"),
            "script-response",
        ),
        (
            None,
            Some("env-model"),
            Some("legacy-model"),
            Some("saved-model"),
            "env-response",
        ),
        (
            None,
            None,
            Some("legacy-model"),
            Some("saved-model"),
            "legacy-response",
        ),
        (None, None, None, Some("saved-model"), "saved-response"),
        (None, None, None, None, "default-response"),
        (
            None,
            Some(""),
            Some("legacy-model"),
            Some("saved-model"),
            "saved-response",
        ),
    ] {
        let home = tempdir().unwrap();
        fs::create_dir(home.path().join(".cubi")).unwrap();
        fs::write(
            home.path().join(".cubi/config.json"),
            json!({"default_model": saved_model, "onboarded": true, "config_version": 1})
                .to_string(),
        )
        .unwrap();
        let mut command = cubi(home.path());
        if let Some(model) = env_model {
            command.env("CUBI_MODEL", model);
        }
        if let Some(model) = legacy_model {
            command.env("AI_CHAT_CLI_MODEL", model);
        }
        if let Some(model) = script_model {
            let script = home.path().join("run.md");
            fs::write(&script, format!("---\nmodel: {}\n---\nhello\n", model)).unwrap();
            command.arg("run").arg(script);
        } else {
            command.args(["-p", "hello"]);
        }

        assert_eq!(response(&mut command), expected);
    }
}

#[test]
fn malformed_fake_response_map_uses_global_response() {
    let home = tempdir().unwrap();
    let mut command = cubi(home.path());
    command
        .env("CUBI_MODEL", "env-model")
        .env("CUBI_FAKE_LLM_MODEL_RESPONSES", r#"{"env-model":17}"#)
        .args(["-p", "hello"]);

    assert_eq!(response(&mut command), "fallback-response");
}

#[test]
fn receipts_precedence_selects_only_the_effective_startup_path() {
    for (flag, env, persisted, expected) in [
        (true, true, true, Some("flag.jsonl")),
        (false, true, true, Some("env.jsonl")),
        (false, false, true, Some("saved.jsonl")),
        (false, false, false, None),
    ] {
        let home = tempdir().unwrap();
        fs::create_dir(home.path().join(".cubi")).unwrap();
        fs::write(
            home.path().join(".cubi/config.json"),
            json!({
                "onboarded": true,
                "config_version": 1,
                "receipts": persisted.then(|| home.path().join("saved.jsonl")),
            })
            .to_string(),
        )
        .unwrap();
        let mut command = cubi(home.path());
        command.args(["-p", "hello"]);
        if flag {
            command
                .arg("--receipts")
                .arg(home.path().join("flag.jsonl"));
        }
        if env {
            command.env("CUBI_RECEIPTS", home.path().join("env.jsonl"));
        }

        assert_eq!(response(&mut command), "default-response");
        for candidate in ["flag.jsonl", "env.jsonl", "saved.jsonl"] {
            let path = home.path().join(candidate);
            assert_eq!(path.exists(), expected == Some(candidate), "{}", candidate);
            if expected == Some(candidate) {
                let receipts = fs::read_to_string(path).unwrap();
                assert!(!receipts.is_empty());
                for line in receipts.lines() {
                    serde_json::from_str::<Value>(line).expect("receipt JSON");
                }
            }
        }
    }
}
