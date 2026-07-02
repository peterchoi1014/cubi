//! Opt-in telemetry / debug log.
//!
//! Roadmap item C#10. Off by default — flip
//! [`crate::onboarding::AppConfig::telemetry`] to `true` (or set the
//! `CUBI_TELEMETRY=1` environment variable) and structured events
//! append to `~/.cubi/telemetry.log`.
//!
//! The log is intentionally append-only, line-delimited JSON so external
//! tooling (`jq`, splunk, ...) can read it without parsing. Failures to
//! write are swallowed — telemetry must never break the chat loop.

use serde::Serialize;
use serde_json::{Value, json};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

static ENABLED: OnceLock<bool> = OnceLock::new();

/// Cache the on/off resolution once per process. Subsequent flips to the
/// env var (or the config) take effect on the next restart, mirroring
/// how the rest of the CLI treats startup-time settings.
pub fn init(config_enabled: bool) {
    let env_on = matches!(
        std::env::var("CUBI_TELEMETRY").as_deref(),
        Ok("1") | Ok("true") | Ok("on") | Ok("yes")
    );
    let _ = ENABLED.set(config_enabled || env_on);
}

pub fn is_enabled() -> bool {
    *ENABLED.get().unwrap_or(&false)
}

fn log_path() -> Option<PathBuf> {
    Some(crate::sessions::cubi_dir()?.join("telemetry.log"))
}

/// Append one event. Silently no-ops when telemetry is disabled or the
/// log file can't be opened.
pub fn record_event(kind: &str, payload: Value) {
    if !is_enabled() {
        return;
    }
    let Some(path) = log_path() else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let line = json!({
        "ts": ts,
        "kind": kind,
        "payload": payload,
    });
    let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) else {
        return;
    };
    let _ = writeln!(f, "{line}");
}

/// Typed convenience wrapper for tool-call events — the most common
/// telemetry consumer in this CLI.
#[derive(Debug, Serialize)]
pub struct ToolCallEvent<'a> {
    pub tool: &'a str,
    pub ok: bool,
    pub duration_ms: u64,
}

pub fn record_tool_call(ev: ToolCallEvent<'_>) {
    record_event(
        "tool_call",
        json!({
            "tool": ev.tool,
            "ok": ev.ok,
            "duration_ms": ev.duration_ms,
        }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_path_uses_cubi_home() {
        crate::compat::test_env::with_cubi_home(|cubi_home, other_home| {
            let path = log_path().expect("log path");
            assert_eq!(path, cubi_home.join(".cubi").join("telemetry.log"));
            assert!(!path.starts_with(other_home));
        });
    }

    #[test]
    fn record_event_is_silent_when_disabled() {
        // ENABLED defaults to false until init() is called. record_event
        // must be a no-op in that state and must not panic.
        record_event("noop", json!({"foo": "bar"}));
        assert!(!is_enabled());
    }
}
