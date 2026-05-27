//! User-facing error formatter.
//!
//! Wraps the underlying `anyhow::Error` chain in a small, opinionated
//! struct that carries an [`ErrorKind`], a stable [`ExitCode`], a
//! one-line `summary`, and an optional one-line `hint`. The full cause
//! chain is preserved on `cause` and only printed when the user opted
//! into debug output (via `--debug`, `CUBI_DEBUG=1`, or any non-empty
//! `RUST_BACKTRACE`).
//!
//! In headless `--json` mode we emit a structured `error` event via
//! [`crate::json_events`] instead of human-readable text. The existing
//! event keeps its `message` field for backward compatibility; the new
//! `kind`, `exit_code`, and `hint` fields are additive.

use crate::exit_code::ExitCode;
use crate::style::CubiStyle;

/// Coarse error classification. Each variant maps to a stable
/// [`ExitCode`] via [`ErrorKind::exit_code`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    Config,
    Auth,
    Quota,
    RateLimited,
    ConnectRefused,
    Dns,
    Tls,
    Timeout,
    ServerError,
    BadRequest,
    Cancelled,
    Tool,
    Budget,
    Other,
}

impl ErrorKind {
    /// Stable serialization tag, used for the JSON `error` event.
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorKind::Config => "config",
            ErrorKind::Auth => "auth",
            ErrorKind::Quota => "quota",
            ErrorKind::RateLimited => "rate_limited",
            ErrorKind::ConnectRefused => "connect_refused",
            ErrorKind::Dns => "dns",
            ErrorKind::Tls => "tls",
            ErrorKind::Timeout => "timeout",
            ErrorKind::ServerError => "server_error",
            ErrorKind::BadRequest => "bad_request",
            ErrorKind::Cancelled => "cancelled",
            ErrorKind::Tool => "tool",
            ErrorKind::Budget => "budget",
            ErrorKind::Other => "other",
        }
    }

    /// Default exit code for this kind.
    pub fn exit_code(self) -> ExitCode {
        match self {
            // User-actionable configuration / argument problems.
            ErrorKind::Config | ErrorKind::BadRequest => ExitCode::Usage,
            // Network-class failures share the new `Network` slot so
            // scripts can distinguish them from generic model/API
            // failures (which keep `Model = 10`).
            ErrorKind::ConnectRefused | ErrorKind::Dns | ErrorKind::Tls => ExitCode::Network,
            // Model / provider problems.
            ErrorKind::Auth
            | ErrorKind::Quota
            | ErrorKind::RateLimited
            | ErrorKind::Timeout
            | ErrorKind::ServerError
            | ErrorKind::Other => ExitCode::Model,
            ErrorKind::Tool => ExitCode::Tool,
            ErrorKind::Budget => ExitCode::Budget,
            ErrorKind::Cancelled => ExitCode::Cancelled,
        }
    }
}

/// A user-facing error: small enum + summary + optional hint, with the
/// full cause chain available for `--debug`.
pub struct UserError {
    pub kind: ErrorKind,
    pub exit_code: ExitCode,
    pub summary: String,
    pub hint: Option<String>,
    pub cause: Option<anyhow::Error>,
}

impl UserError {
    pub fn new(kind: ErrorKind, summary: impl Into<String>) -> Self {
        Self {
            kind,
            exit_code: kind.exit_code(),
            summary: summary.into(),
            hint: None,
            cause: None,
        }
    }

    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }

    #[allow(dead_code)]
    pub fn with_hint_opt(mut self, hint: Option<String>) -> Self {
        self.hint = hint;
        self
    }

    pub fn with_cause(mut self, cause: anyhow::Error) -> Self {
        self.cause = Some(cause);
        self
    }

    #[allow(dead_code)]
    pub fn with_exit_code(mut self, code: ExitCode) -> Self {
        self.exit_code = code;
        self
    }

    /// Convenience: build a generic `Other`-kind UserError from any
    /// `anyhow::Error`, preserving the underlying chain. Useful at
    /// top-level fall-back call sites where we have not (yet)
    /// classified the failure.
    pub fn from_anyhow(err: anyhow::Error) -> Self {
        let summary = format!("{}", err);
        Self::new(ErrorKind::Other, summary).with_cause(err)
    }
}

/// Returns true when the user has opted into debug output. Reads
/// process env at call time; tests should use [`debug_mode_with`]
/// to avoid touching the global environment.
pub fn debug_mode() -> bool {
    debug_mode_with(|k| std::env::var(k).ok(), DEBUG_FLAG.load_relaxed())
}

/// Pure helper for [`debug_mode`]. Visible for unit tests so we can
/// vary the environment and CLI flag independently of the process.
pub fn debug_mode_with<F>(env: F, debug_flag: bool) -> bool
where
    F: Fn(&str) -> Option<String>,
{
    if debug_flag {
        return true;
    }
    if let Some(v) = env("CUBI_DEBUG") {
        if !v.is_empty() && v != "0" {
            return true;
        }
    }
    if let Some(v) = env("RUST_BACKTRACE") {
        if !v.is_empty() {
            return true;
        }
    }
    false
}

/// Process-wide `--debug` switch. Set once during argv parsing.
mod debug_flag {
    use std::sync::atomic::{AtomicBool, Ordering};
    pub struct DebugFlag(AtomicBool);
    impl DebugFlag {
        pub const fn new() -> Self {
            Self(AtomicBool::new(false))
        }
        pub fn set(&self, v: bool) {
            self.0.store(v, Ordering::Relaxed);
        }
        pub fn load_relaxed(&self) -> bool {
            self.0.load(Ordering::Relaxed)
        }
    }
}
static DEBUG_FLAG: debug_flag::DebugFlag = debug_flag::DebugFlag::new();

pub fn set_debug_flag(enabled: bool) {
    DEBUG_FLAG.set(enabled);
}

/// Pure formatter: returns the multi-line string we would print to
/// stderr in human (non-JSON) mode. Keeping this separate makes it
/// testable without capturing stderr.
pub fn format_user_error(err: &UserError, debug: bool) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let label = "error:".bright_red().bold();
    let summary = err.summary.bright_red().bold();
    let code_tag = format!("[code {}]", err.exit_code.as_i32()).bright_black();
    let _ = writeln!(out, "{} {} {}", label, summary, code_tag);
    if let Some(hint) = &err.hint {
        let prefix = "  hint:".bright_black();
        let body = hint.bright_black();
        let _ = writeln!(out, "{} {}", prefix, body);
    }
    if debug {
        if let Some(cause) = &err.cause {
            let _ = writeln!(out);
            let _ = writeln!(out, "caused by:");
            for c in cause.chain() {
                let _ = writeln!(out, "  - {}", c);
            }
        }
    }
    out
}

/// Prints `err` to stderr (human mode). In `--json` headless mode the
/// caller should use [`emit_user_error_json`] instead.
pub fn print_user_error(err: &UserError, debug: bool) {
    let s = format_user_error(err, debug);
    eprint!("{}", s);
}

/// Builds the JSON `error` event payload for this error. Keeps the
/// legacy `message` field; adds `kind`, `exit_code`, and (optional)
/// `hint`.
pub fn user_error_json(err: &UserError) -> serde_json::Value {
    let mut v = crate::json_events::error(&err.summary);
    if let Some(obj) = v.as_object_mut() {
        obj.insert(
            "kind".to_string(),
            serde_json::Value::String(err.kind.as_str().to_string()),
        );
        obj.insert(
            "exit_code".to_string(),
            serde_json::Value::Number(err.exit_code.as_i32().into()),
        );
        if let Some(hint) = &err.hint {
            obj.insert("hint".to_string(), serde_json::Value::String(hint.clone()));
        }
    }
    v
}

/// Emit the JSON error event to stdout (line-delimited) when enabled.
pub fn emit_user_error_json(enabled: bool, err: &UserError) {
    let v = user_error_json(err);
    crate::json_events::emit(enabled, &v);
}

/// Top-level dispatcher: routes to JSON or human output based on
/// `json_mode`. Returns the exit code so callers can `exit_code::exit`.
pub fn report_user_error(err: &UserError, json_mode: bool, debug: bool) -> ExitCode {
    if json_mode {
        emit_user_error_json(true, err);
    } else {
        print_user_error(err, debug);
    }
    err.exit_code
}

/// Lightweight notice printer for non-fatal conditions (e.g. an MCP
/// server degrading mid-session). Always goes to stderr and is
/// suppressed in `--json` mode to keep JSONL output parseable.
pub fn print_user_warning(summary: &str, hint: Option<&str>, json_mode: bool) {
    if json_mode {
        return;
    }
    let label = "warning:".bright_yellow().bold();
    eprintln!("{} {}", label, summary);
    if let Some(h) = hint {
        let prefix = "  hint:".bright_black();
        let body = h.bright_black();
        eprintln!("{} {}", prefix, body);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;

    fn strip_ansi(s: &str) -> String {
        // Cheap ANSI escape stripper good enough for these tests:
        // strip CSI sequences `\x1b[...m`.
        let mut out = String::with_capacity(s.len());
        let bytes = s.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
                i += 2;
                while i < bytes.len() && bytes[i] != b'm' {
                    i += 1;
                }
                if i < bytes.len() {
                    i += 1;
                }
            } else {
                out.push(bytes[i] as char);
                i += 1;
            }
        }
        out
    }

    fn force_color_off() {
        crate::style::set_color_override(false);
    }

    #[test]
    fn kind_maps_to_expected_exit_codes() {
        assert_eq!(ErrorKind::Auth.exit_code(), ExitCode::Model);
        assert_eq!(ErrorKind::Tool.exit_code(), ExitCode::Tool);
        assert_eq!(ErrorKind::Budget.exit_code(), ExitCode::Budget);
        assert_eq!(ErrorKind::Cancelled.exit_code(), ExitCode::Cancelled);
        assert_eq!(ErrorKind::ConnectRefused.exit_code(), ExitCode::Network);
        assert_eq!(ErrorKind::Dns.exit_code(), ExitCode::Network);
        assert_eq!(ErrorKind::Tls.exit_code(), ExitCode::Network);
        assert_eq!(ErrorKind::Config.exit_code(), ExitCode::Usage);
        assert_eq!(ErrorKind::BadRequest.exit_code(), ExitCode::Usage);
    }

    #[test]
    fn format_with_hint_no_debug_omits_cause_chain() {
        force_color_off();
        let err = UserError::new(ErrorKind::Auth, "missing API key")
            .with_hint("set CUBI_OPENAI_API_KEY")
            .with_cause(anyhow!("HTTP 401").context("during /chat/completions"));
        let s = strip_ansi(&format_user_error(&err, false));
        assert!(s.starts_with("error: missing API key"), "got: {s}");
        assert!(s.contains("[code 10]"));
        assert!(s.contains("hint: set CUBI_OPENAI_API_KEY"));
        assert!(!s.contains("caused by:"));
    }

    #[test]
    fn format_with_debug_renders_cause_chain() {
        force_color_off();
        let err = UserError::new(ErrorKind::Auth, "missing API key")
            .with_cause(anyhow!("HTTP 401").context("during /chat/completions"));
        let s = strip_ansi(&format_user_error(&err, true));
        assert!(s.contains("caused by:"));
        assert!(s.contains("during /chat/completions"));
        assert!(s.contains("HTTP 401"));
    }

    #[test]
    fn format_without_hint_skips_hint_line() {
        force_color_off();
        let err = UserError::new(ErrorKind::Other, "something failed");
        let s = strip_ansi(&format_user_error(&err, false));
        assert!(s.contains("error: something failed"));
        assert!(!s.contains("hint:"));
    }

    #[test]
    fn json_payload_includes_kind_exit_code_and_hint() {
        let err =
            UserError::new(ErrorKind::RateLimited, "429 from provider").with_hint("retry in ~7s");
        let v = user_error_json(&err);
        assert_eq!(v["type"], "error");
        assert_eq!(v["message"], "429 from provider");
        assert_eq!(v["kind"], "rate_limited");
        assert_eq!(v["exit_code"], 10);
        assert_eq!(v["hint"], "retry in ~7s");
    }

    #[test]
    fn json_payload_without_hint_omits_hint_field() {
        let err = UserError::new(ErrorKind::Tool, "tool blew up");
        let v = user_error_json(&err);
        assert_eq!(v["kind"], "tool");
        assert_eq!(v["exit_code"], 11);
        assert!(v.get("hint").is_none());
    }

    #[test]
    fn no_color_strips_ansi_from_output() {
        // With color forced off, the formatter must not embed ANSI
        // escape codes — `strip_ansi` and the raw string should match.
        force_color_off();
        let err = UserError::new(ErrorKind::Other, "x").with_hint("y");
        let s = format_user_error(&err, false);
        let stripped = strip_ansi(&s);
        assert_eq!(s, stripped, "output should be plain when NO_COLOR active");
    }

    #[test]
    fn debug_mode_with_flag_true_returns_true() {
        let env = |_: &str| None;
        assert!(debug_mode_with(env, true));
    }

    #[test]
    fn debug_mode_with_env_cubi_debug_enables() {
        let env = |k: &str| match k {
            "CUBI_DEBUG" => Some("1".to_string()),
            _ => None,
        };
        assert!(debug_mode_with(env, false));
    }

    #[test]
    fn debug_mode_with_cubi_debug_zero_disabled() {
        let env = |k: &str| match k {
            "CUBI_DEBUG" => Some("0".to_string()),
            _ => None,
        };
        assert!(!debug_mode_with(env, false));
    }

    #[test]
    fn debug_mode_with_rust_backtrace_any_value_enables() {
        let env = |k: &str| match k {
            "RUST_BACKTRACE" => Some("full".to_string()),
            _ => None,
        };
        assert!(debug_mode_with(env, false));
    }

    #[test]
    fn debug_mode_off_by_default() {
        let env = |_: &str| None;
        assert!(!debug_mode_with(env, false));
    }
}
