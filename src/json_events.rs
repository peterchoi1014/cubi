//! Headless JSON event emission.
//!
//! Until this module existed, ad-hoc `serde_json::json!({...})` +
//! `println!` calls were scattered across `cli/agent.rs` and `main.rs`.
//! That made it hard to (a) keep the wire format consistent, (b) flush
//! stdout reliably so each event is observable in real time when piped,
//! and (c) unit-test the payload shapes without spinning up the real
//! agent loop.
//!
//! All emit helpers take an `enabled: bool` so the cli surface can pass
//! `self.json_enabled && self.headless_mode` once and forget about it.
//! Helpers also return the built `serde_json::Value` so call sites can
//! reuse it (e.g. for logging) without serializing twice.

use std::io::Write;

use serde_json::{Value, json};

use crate::ollama::ChatStats;

/// Prints one JSON event as a single line to stdout (line-delimited JSON,
/// JSONL) and flushes immediately. Flushing matters because callers pipe
/// these events into tools that read one event at a time; without an
/// explicit flush, libc may hold output until the line buffer is full.
///
/// No-op when `enabled` is false so call sites can stay branch-free.
pub fn emit(enabled: bool, event: &Value) {
    if !enabled {
        return;
    }
    println!("{}", event);
    let _ = std::io::stdout().flush();
}

pub fn token(value: &str) -> Value {
    json!({ "type": "token", "value": value })
}

#[allow(dead_code)]
pub fn emit_token(enabled: bool, value: &str) -> Value {
    let v = token(value);
    emit(enabled, &v);
    v
}

pub fn done(stats: &ChatStats) -> Value {
    json!({ "type": "done", "stats": stats })
}

#[allow(dead_code)]
pub fn emit_done(enabled: bool, stats: &ChatStats) -> Value {
    let v = done(stats);
    emit(enabled, &v);
    v
}

pub fn tool_call(name: &str, arguments: &Value) -> Value {
    json!({
        "type": "tool_call",
        "name": name,
        "arguments": arguments,
    })
}

#[allow(dead_code)]
pub fn emit_tool_call(enabled: bool, name: &str, arguments: &Value) -> Value {
    let v = tool_call(name, arguments);
    emit(enabled, &v);
    v
}

pub fn tool_result(name: &str, content: &str) -> Value {
    json!({
        "type": "tool_result",
        "name": name,
        "content": content,
    })
}

#[allow(dead_code)]
pub fn emit_tool_result(enabled: bool, name: &str, content: &str) -> Value {
    let v = tool_result(name, content);
    emit(enabled, &v);
    v
}

pub fn tool_timeout(name: &str, secs: u64) -> Value {
    json!({
        "type": "tool_timeout",
        "name": name,
        "secs": secs,
    })
}

#[allow(dead_code)]
pub fn emit_tool_timeout(enabled: bool, name: &str, secs: u64) -> Value {
    let v = tool_timeout(name, secs);
    emit(enabled, &v);
    v
}

pub fn error(message: &str) -> Value {
    json!({ "type": "error", "message": message })
}

pub fn emit_error(enabled: bool, message: &str) -> Value {
    let v = error(message);
    emit(enabled, &v);
    v
}

pub fn compacted(summarized_messages: usize, window: usize) -> Value {
    json!({
        "type": "compacted",
        "summarized_messages": summarized_messages,
        "window": window,
    })
}

#[allow(dead_code)]
pub fn emit_compacted(enabled: bool, summarized_messages: usize, window: usize) -> Value {
    let v = compacted(summarized_messages, window);
    emit(enabled, &v);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_payload_shape() {
        let v = token("hello");
        assert_eq!(v["type"], "token");
        assert_eq!(v["value"], "hello");
    }

    #[test]
    fn done_payload_includes_stats() {
        let stats = ChatStats {
            prompt_tokens: 12,
            completion_tokens: 34,
            elapsed_ms: 56,
        };
        let v = done(&stats);
        assert_eq!(v["type"], "done");
        assert_eq!(v["stats"]["prompt_tokens"], 12);
        assert_eq!(v["stats"]["completion_tokens"], 34);
        assert_eq!(v["stats"]["elapsed_ms"], 56);
    }

    #[test]
    fn tool_call_payload_shape() {
        let args = json!({"command": "ls"});
        let v = tool_call("bash", &args);
        assert_eq!(v["type"], "tool_call");
        assert_eq!(v["name"], "bash");
        assert_eq!(v["arguments"]["command"], "ls");
    }

    #[test]
    fn tool_result_payload_shape() {
        let v = tool_result("bash", "ok");
        assert_eq!(v["type"], "tool_result");
        assert_eq!(v["name"], "bash");
        assert_eq!(v["content"], "ok");
    }

    #[test]
    fn tool_timeout_payload_shape() {
        let v = tool_timeout("bash", 30);
        assert_eq!(v["type"], "tool_timeout");
        assert_eq!(v["name"], "bash");
        assert_eq!(v["secs"], 30);
    }

    #[test]
    fn error_payload_shape() {
        let v = error("bad");
        assert_eq!(v["type"], "error");
        assert_eq!(v["message"], "bad");
    }

    #[test]
    fn emit_is_noop_when_disabled() {
        // emit() with enabled=false must not touch stdout. We can't
        // capture stdout here cheaply, but at minimum confirm it doesn't
        // panic and returns nothing observable from the value side.
        emit(false, &token("ignored"));
    }
}
