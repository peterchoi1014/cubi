//! Structured-event tap for `--events <path>` (and `CUBI_EVENTS` env).
//!
//! The tap captures *every* internal event of interest — turn
//! boundaries, tool calls, rationales, errors, retries, MCP server
//! transitions — as line-delimited JSON appended to a file.
//!
//! The existing `--trace-tools` audit log is a strict subset (only
//! `tool_start` / `tool_complete`). It remains operational with its
//! original record shape for back-compat, but tools that want a unified
//! event stream should use `--events`.
//!
//! Best-effort: a write failure is logged via `tracing::warn!` but
//! never aborts the caller. Parent directories are created on first
//! write. Concurrent writers are serialized through an internal mutex
//! so JSONL lines never interleave.

use serde_json::{Value, json};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

/// Open-append JSONL sink wrapping a single configured path.
#[derive(Debug)]
pub struct EventSink {
    path: PathBuf,
    write_lock: Mutex<()>,
}

impl EventSink {
    /// Resolves the configured path from (in order) the explicit CLI
    /// flag, then the `CUBI_EVENTS` env var. Returns `None` when neither
    /// is set or both are empty.
    pub fn from_args(flag_path: Option<&str>) -> Option<Self> {
        let raw = flag_path
            .map(|s| s.to_string())
            .or_else(|| std::env::var("CUBI_EVENTS").ok())?;
        if raw.is_empty() {
            return None;
        }
        Some(Self::new(PathBuf::from(raw)))
    }

    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            write_lock: Mutex::new(()),
        }
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// Best-effort append. Returns the file-system error to the caller
    /// only when used through [`Self::probe`]; the general `emit` path
    /// swallows IO errors and logs them.
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

    /// Try to open (create) the configured path with a zero-length
    /// touch so startup can surface a clear `UserError::Config` rather
    /// than discovering the failure on the first event.
    pub fn probe(&self) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        Ok(())
    }

    /// Append one event. `kind` is inserted as `"type": kind` if the
    /// value object does not already carry one. `ts` is added when
    /// missing. Both keys are convenience for callers that build the
    /// payload imperatively.
    pub fn emit(&self, kind: &str, mut payload: Value) {
        if let Some(obj) = payload.as_object_mut() {
            obj.entry("type".to_string())
                .or_insert_with(|| Value::String(kind.to_string()));
            obj.entry("ts".to_string())
                .or_insert_with(|| Value::String(now_rfc3339()));
        } else {
            payload = json!({"type": kind, "ts": now_rfc3339(), "value": payload});
        }
        let _guard = self.write_lock.lock();
        if let Err(e) = self.append_inner(&payload) {
            tracing::warn!(
                target: "cubi::event_sink",
                error = %e,
                path = %self.path.display(),
                "event sink write failed; continuing",
            );
        }
    }
}

/// Minimal RFC3339 UTC timestamp; shared with `trace_tools::now_rfc3339`
/// in shape but inlined to avoid a circular dep.
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
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("cubi-events-{label}-{nanos}.jsonl"))
    }

    #[test]
    fn sink_appends_one_line_per_event_with_ts_and_type() {
        let path = temp_path("simple");
        let _ = std::fs::remove_file(&path);
        let sink = EventSink::new(path.clone());

        sink.emit("turn_start", json!({"turn": 1}));
        sink.emit(
            "tool_call_start",
            json!({"tool": "bash", "args": {"command": "true"}}),
        );
        sink.emit("turn_end", json!({"turn": 1, "ok": true}));

        let raw = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = raw.lines().collect();
        assert_eq!(lines.len(), 3);
        for line in &lines {
            let v: Value = serde_json::from_str(line).expect("valid JSON");
            assert!(v.get("type").is_some(), "type field missing");
            assert!(v.get("ts").is_some(), "ts field missing");
        }
        let v0: Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(v0["type"], "turn_start");
        assert_eq!(v0["turn"], 1);
        let v1: Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(v1["type"], "tool_call_start");
        assert_eq!(v1["tool"], "bash");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn from_args_prefers_flag_over_env() {
        // Save & clear env so we don't leak between tests.
        let prev = std::env::var("CUBI_EVENTS").ok();
        // SAFETY: tests in this module run serially because they
        // already touch the global env one at a time.
        unsafe { std::env::set_var("CUBI_EVENTS", "/env/path") };
        let sink = EventSink::from_args(Some("/flag/path")).unwrap();
        assert_eq!(sink.path(), std::path::Path::new("/flag/path"));
        match prev {
            Some(v) => unsafe { std::env::set_var("CUBI_EVENTS", v) },
            None => unsafe { std::env::remove_var("CUBI_EVENTS") },
        }
    }

    #[test]
    fn probe_creates_parent_dir() {
        let base = temp_path("probe");
        let nested = base.join("nested/dir/events.jsonl");
        let sink = EventSink::new(nested.clone());
        sink.probe().expect("probe must succeed");
        assert!(nested.exists());
        let _ = std::fs::remove_dir_all(&base);
    }
}
