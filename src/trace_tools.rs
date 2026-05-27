//! `--trace-tools <path>` JSONL audit log.
//!
//! When the flag (or `CUBI_TRACE_TOOLS` env var) is set, every tool
//! invocation appends two lines to the file:
//!
//! ```jsonl
//! {"ts":"2025-01-02T03:04:05Z","event":"tool_start","tool":"bash",
//!  "args_redacted":{...},"call_id":"1"}
//! {"ts":"...","event":"tool_complete","tool":"bash","call_id":"1",
//!  "ok":true,"duration_ms":42,"result_chars":128}
//! ```
//!
//! Append-only and best-effort: a write failure is logged via
//! `tracing::warn!` but never aborts the tool call. Parent directories
//! are created on first write.

use serde_json::{Value, json};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Holds the configured trace-log path and an atomic call-id counter.
/// Cheap to clone (the counter and path live behind `Arc`-like
/// indirection inside `Mutex`).
#[derive(Debug)]
pub struct ToolTracer {
    path: PathBuf,
    next_id: AtomicU64,
    // Serialize writes so two concurrent tool calls don't interleave
    // half lines into the file.
    write_lock: Mutex<()>,
}

impl ToolTracer {
    /// Resolves the trace path from (in order) the explicit CLI flag,
    /// then the `CUBI_TRACE_TOOLS` env var. Returns `None` when neither
    /// is set.
    pub fn from_args(flag_path: Option<&str>) -> Option<Self> {
        let path = flag_path
            .map(PathBuf::from)
            .or_else(|| std::env::var("CUBI_TRACE_TOOLS").ok().map(PathBuf::from))?;
        Some(Self::new(path))
    }

    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            next_id: AtomicU64::new(1),
            write_lock: Mutex::new(()),
        }
    }

    /// Allocate a fresh call-id (monotonic per-process counter). The
    /// id is returned as a string so the JSONL stays human-readable.
    pub fn next_call_id(&self) -> String {
        self.next_id.fetch_add(1, Ordering::SeqCst).to_string()
    }

    /// Emit a tool_start record. `args` is redacted via
    /// [`redact_secrets`] before serialization.
    pub fn log_start(&self, tool: &str, call_id: &str, args: &Value) {
        let mut redacted = args.clone();
        redact_secrets(&mut redacted);
        let line = json!({
            "ts": now_rfc3339(),
            "event": "tool_start",
            "tool": tool,
            "call_id": call_id,
            "args_redacted": redacted,
        });
        self.append(&line);
    }

    /// Emit a tool_complete record.
    pub fn log_complete(
        &self,
        tool: &str,
        call_id: &str,
        ok: bool,
        duration_ms: u128,
        result_chars: usize,
    ) {
        let line = json!({
            "ts": now_rfc3339(),
            "event": "tool_complete",
            "tool": tool,
            "call_id": call_id,
            "ok": ok,
            "duration_ms": duration_ms,
            "result_chars": result_chars,
        });
        self.append(&line);
    }

    fn append(&self, line: &Value) {
        let _guard = self.write_lock.lock();
        if let Err(e) = self.append_inner(line) {
            tracing::warn!(
                target: "cubi::trace_tools",
                error = %e,
                path = %self.path.display(),
                "tool trace write failed; continuing",
            );
        }
    }

    fn append_inner(&self, line: &Value) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let mut file: File = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        writeln!(file, "{}", line)?;
        Ok(())
    }
}

/// Recursively replace string values under keys that look like
/// secrets. Mirrors [`crate::main::redact_secrets`] so the trace log
/// honors the same redaction rules as `--print-config`.
pub fn redact_secrets(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for (k, v) in map.iter_mut() {
                let lower = k.to_ascii_lowercase();
                if lower.contains("key")
                    || lower.contains("token")
                    || lower.contains("secret")
                    || lower.contains("password")
                {
                    *v = Value::String("<redacted>".to_string());
                } else {
                    redact_secrets(v);
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                redact_secrets(item);
            }
        }
        _ => {}
    }
}

/// Returns the current time as a minimal RFC3339 / ISO-8601 string in
/// UTC, e.g. `2025-01-02T03:04:05Z`. Hand-formatted so we don't pull
/// in a date crate for one line.
fn now_rfc3339() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let (y, m, d, hour, minute, second) = crate::sessions::civil_from_unix(now);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y, m, d, hour, minute, second
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(suffix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("cubi-trace-{suffix}-{nanos}.jsonl"))
    }

    #[test]
    fn redact_replaces_secret_like_keys() {
        let mut v = json!({
            "user": "alice",
            "api_key": "sk-abc",
            "nested": {"auth_token": "xyz", "ok": true}
        });
        redact_secrets(&mut v);
        assert_eq!(v["user"], "alice");
        assert_eq!(v["api_key"], "<redacted>");
        assert_eq!(v["nested"]["auth_token"], "<redacted>");
        assert_eq!(v["nested"]["ok"], true);
    }

    #[test]
    fn tracer_writes_two_lines_per_call_appending() {
        let path = temp_path("happy");
        let _ = std::fs::remove_file(&path);
        let tracer = ToolTracer::new(path.clone());
        let id = tracer.next_call_id();
        tracer.log_start("bash", &id, &json!({"command": "ls"}));
        tracer.log_complete("bash", &id, true, 42, 7);
        // Second invocation must keep id-counter monotonic.
        let id2 = tracer.next_call_id();
        assert_eq!(id, "1");
        assert_eq!(id2, "2");
        tracer.log_start("read_file", &id2, &json!({"path": "/etc/hosts"}));
        tracer.log_complete("read_file", &id2, false, 1, 0);

        let raw = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = raw.lines().collect();
        assert_eq!(lines.len(), 4);
        let l0: Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(l0["event"], "tool_start");
        assert_eq!(l0["tool"], "bash");
        assert_eq!(l0["call_id"], "1");
        assert_eq!(l0["args_redacted"]["command"], "ls");
        let l1: Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(l1["event"], "tool_complete");
        assert_eq!(l1["ok"], true);
        assert_eq!(l1["duration_ms"], 42);
        assert_eq!(l1["result_chars"], 7);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn tracer_redacts_secret_args() {
        let path = temp_path("redact");
        let _ = std::fs::remove_file(&path);
        let tracer = ToolTracer::new(path.clone());
        let id = tracer.next_call_id();
        tracer.log_start(
            "http_get",
            &id,
            &json!({"url": "https://api", "auth_token": "shh"}),
        );
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains("<redacted>"));
        assert!(!raw.contains("shh"));
        let _ = std::fs::remove_file(&path);
    }
}
