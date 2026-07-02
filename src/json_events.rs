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

/// Extended variant of [`done`] that includes the active model's
/// context window and the prompt-vs-window utilization (in percent).
/// When `window` is `None` the original `done` shape is returned so
/// existing consumers keep parsing.
pub fn done_with_window(stats: &ChatStats, window: Option<usize>) -> Value {
    let mut v = done(stats);
    if let Some(w) = window {
        let pct = if w == 0 {
            0u32
        } else {
            let pct = stats.prompt_tokens.saturating_mul(100) / (w as u64);
            u32::try_from(pct).unwrap_or(u32::MAX)
        };
        if let Some(obj) = v.as_object_mut() {
            obj.insert("window".to_string(), json!(w));
            obj.insert("utilization_pct".to_string(), json!(pct));
        }
    }
    v
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

#[allow(dead_code)]
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

pub fn budget_error(needed: usize, window: usize, model: &str) -> Value {
    json!({
        "type": "budget_error",
        "needed": needed,
        "window": window,
        "model": model,
    })
}

/// Emitted at the start of a `consensus_run` invocation, before any
/// subagent is dispatched. Mirrored to both stdout JSON and the
/// `--events` tap.
pub fn consensus_start(
    goal: &str,
    models: &[String],
    strategy: &str,
    max_steps_per_subagent: usize,
) -> Value {
    json!({
        "type": "consensus_start",
        "goal": goal,
        "models": models,
        "strategy": strategy,
        "max_steps_per_subagent": max_steps_per_subagent,
    })
}

/// Emitted once per subagent when it finishes (successful or not).
/// `ok` is true iff the subagent produced a non-error output. Token
/// counts are zero when the subagent errored.
pub fn consensus_subagent_result(
    model: &str,
    ok: bool,
    steps_used: usize,
    tool_calls: usize,
    stats: ChatStats,
    error: Option<&str>,
) -> Value {
    let mut v = json!({
        "type": "consensus_subagent_result",
        "model": model,
        "ok": ok,
        "steps_used": steps_used,
        "tool_calls": tool_calls,
        "elapsed_ms": stats.elapsed_ms,
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.completion_tokens,
    });
    if let Some(err) = error {
        if let Some(obj) = v.as_object_mut() {
            obj.insert("error".to_string(), json!(err));
        }
    }
    v
}

/// Emitted after arbitration with the chosen winner and a free-form
/// `decision_reason` describing why that subagent won.
pub fn consensus_decision(winner_model: &str, decision_reason: &str) -> Value {
    json!({
        "type": "consensus_decision",
        "winner_model": winner_model,
        "decision_reason": decision_reason,
    })
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
    fn done_with_window_attaches_utilization() {
        let stats = ChatStats {
            prompt_tokens: 250,
            completion_tokens: 0,
            elapsed_ms: 0,
        };
        let v = done_with_window(&stats, Some(1000));
        assert_eq!(v["window"], 1000);
        assert_eq!(v["utilization_pct"], 25);
        // None should leave the payload unmodified beyond `done`.
        let v2 = done_with_window(&stats, None);
        assert!(v2.get("window").is_none());
        assert!(v2.get("utilization_pct").is_none());
    }

    #[test]
    fn done_with_window_zero_window_is_zero_pct() {
        let stats = ChatStats {
            prompt_tokens: 250,
            completion_tokens: 0,
            elapsed_ms: 0,
        };
        let v = done_with_window(&stats, Some(0));
        assert_eq!(v["window"], 0);
        assert_eq!(v["utilization_pct"], 0);
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

    #[test]
    fn consensus_start_includes_models_and_strategy() {
        let v = consensus_start("pick", &["m1".into(), "m2".into()], "vote", 8);
        assert_eq!(v["type"], "consensus_start");
        assert_eq!(v["strategy"], "vote");
        assert_eq!(v["models"][1], "m2");
        assert_eq!(v["max_steps_per_subagent"], 8);
    }

    #[test]
    fn consensus_subagent_result_omits_error_when_ok() {
        let v = consensus_subagent_result(
            "m1",
            true,
            1,
            2,
            ChatStats {
                prompt_tokens: 10,
                completion_tokens: 20,
                elapsed_ms: 42,
            },
            None,
        );
        assert_eq!(v["type"], "consensus_subagent_result");
        assert_eq!(v["ok"], true);
        assert_eq!(v["tool_calls"], 2);
        assert_eq!(v["prompt_tokens"], 10);
        assert!(v.get("error").is_none());
    }

    #[test]
    fn consensus_subagent_result_includes_error_when_failed() {
        let v = consensus_subagent_result(
            "m1",
            false,
            0,
            0,
            ChatStats {
                prompt_tokens: 0,
                completion_tokens: 0,
                elapsed_ms: 5,
            },
            Some("boom"),
        );
        assert_eq!(v["ok"], false);
        assert_eq!(v["error"], "boom");
    }

    #[test]
    fn consensus_decision_payload_shape() {
        let v = consensus_decision("m2", "judge picked");
        assert_eq!(v["type"], "consensus_decision");
        assert_eq!(v["winner_model"], "m2");
        assert_eq!(v["decision_reason"], "judge picked");
    }
}
