//! Centralised JSON schema helpers.
//!
//! Roadmap item C#21: most built-in tools currently consume their
//! arguments as `serde_json::Value` and reach for fields ad-hoc, which
//! makes the error messages inconsistent ("Missing 'path'" vs
//! "'path' is required" vs panics on the wrong type). This module
//! gives the registry a uniform vocabulary for argument extraction so
//! every tool refuses bad input the same way.
//!
//! The helpers are intentionally tiny — we don't want a full JSON
//! Schema validator in the binary, just enough shared phrasing that
//! future tools can drop into the same pattern.
//!
//! Marked `#[allow(dead_code)]` because the existing tools predate
//! this module — they'll be migrated incrementally and we want the
//! helpers available + tested as a stable target for that work.
#![allow(dead_code)]

use anyhow::{Context, Result, bail};
use serde_json::Value;

/// Pulls a required string field. Errors with a uniform, model-facing
/// message when the field is missing or not a string.
pub fn require_str<'a>(args: &'a Value, field: &str) -> Result<&'a str> {
    args.get(field)
        .and_then(|v| v.as_str())
        .with_context(|| format!("Missing or non-string '{field}' parameter"))
}

/// Pulls an optional string field, returning `None` for both
/// "key absent" and "key present but explicitly null".
pub fn optional_str<'a>(args: &'a Value, field: &str) -> Option<&'a str> {
    args.get(field).and_then(|v| v.as_str())
}

/// Pulls a required numeric field as `f64`. Accepts ints and floats so
/// the model can hand us either `5` or `5.0`.
pub fn require_number(args: &Value, field: &str) -> Result<f64> {
    if let Some(v) = args.get(field) {
        if let Some(n) = v.as_f64() {
            return Ok(n);
        }
        if let Some(n) = v.as_i64() {
            return Ok(n as f64);
        }
        if let Some(n) = v.as_u64() {
            return Ok(n as f64);
        }
    }
    bail!("Missing or non-numeric '{field}' parameter")
}

/// Pulls a required boolean field.
pub fn require_bool(args: &Value, field: &str) -> Result<bool> {
    args.get(field)
        .and_then(|v| v.as_bool())
        .with_context(|| format!("Missing or non-boolean '{field}' parameter"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn require_str_returns_value_or_uniform_error() {
        let v = json!({"path": "/tmp/x"});
        assert_eq!(require_str(&v, "path").unwrap(), "/tmp/x");

        let err = format!("{}", require_str(&v, "missing").unwrap_err());
        assert!(err.contains("Missing or non-string"));
    }

    #[test]
    fn optional_str_returns_none_for_missing_or_null() {
        assert_eq!(optional_str(&json!({}), "x"), None);
        assert_eq!(optional_str(&json!({"x": null}), "x"), None);
        assert_eq!(optional_str(&json!({"x": "yes"}), "x"), Some("yes"));
    }

    #[test]
    fn require_number_accepts_int_and_float() {
        assert_eq!(require_number(&json!({"n": 5}), "n").unwrap(), 5.0);
        assert_eq!(require_number(&json!({"n": 5.5}), "n").unwrap(), 5.5);
        assert!(require_number(&json!({"n": "no"}), "n").is_err());
    }

    #[test]
    fn require_bool_rejects_non_bools() {
        assert!(require_bool(&json!({"b": true}), "b").unwrap());
        assert!(require_bool(&json!({"b": "true"}), "b").is_err());
    }
}
