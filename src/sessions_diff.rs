//! Pure diff helper for two `SessionFile`s.
//!
//! Separated from `sessions.rs` because the comparison logic has its
//! own surface (preview formatting, multiset diff on pinned, semantic
//! message equality including tool fields) and is exercised
//! independently by `--diff-sessions`.

use serde::Serialize;
use std::collections::HashMap;

use crate::ollama::Message;
use crate::sessions::SessionFile;

/// Structured diff between two persisted sessions. Designed so the
/// pretty renderer and the `--json` renderer share a single source of
/// truth.
#[derive(Debug, Clone, Serialize)]
pub struct SessionDiff {
    pub id_a: String,
    pub id_b: String,
    pub model_a: String,
    pub model_b: String,
    pub model_drift: bool,
    pub started_at_a: u64,
    pub started_at_b: u64,
    pub count_a: usize,
    pub count_b: usize,
    pub common_prefix_len: usize,
    pub divergence: Option<Divergence>,
    /// Pins added on `b` vs `a` (multiset). Each entry is the pin text
    /// repeated once per extra occurrence on `b`.
    pub pinned_added: Vec<String>,
    /// Pins missing from `b` vs `a` (multiset).
    pub pinned_removed: Vec<String>,
    /// True when `id_a == id_b` post-resolution.
    pub identical_id: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Divergence {
    pub index: usize,
    pub a_role: String,
    pub b_role: String,
    pub a_preview: String,
    pub b_preview: String,
}

/// Computes a structured diff between `a` and `b`.
pub fn diff(a: &SessionFile, b: &SessionFile) -> SessionDiff {
    let common_prefix_len = a
        .history
        .iter()
        .zip(b.history.iter())
        .take_while(|(am, bm)| messages_equal(am, bm))
        .count();

    let divergence = if common_prefix_len < a.history.len() || common_prefix_len < b.history.len() {
        let ai = a.history.get(common_prefix_len);
        let bi = b.history.get(common_prefix_len);
        Some(Divergence {
            index: common_prefix_len,
            a_role: ai.map(|m| m.role.clone()).unwrap_or_else(|| "<end>".into()),
            b_role: bi.map(|m| m.role.clone()).unwrap_or_else(|| "<end>".into()),
            a_preview: ai.map(message_preview).unwrap_or_else(|| "<end>".into()),
            b_preview: bi.map(message_preview).unwrap_or_else(|| "<end>".into()),
        })
    } else {
        None
    };

    let (pinned_added, pinned_removed) = pinned_multiset_diff(&a.pinned, &b.pinned);

    SessionDiff {
        id_a: a.id.clone(),
        id_b: b.id.clone(),
        model_a: a.model.clone(),
        model_b: b.model.clone(),
        model_drift: a.model != b.model,
        started_at_a: a.started_at,
        started_at_b: b.started_at,
        count_a: a.history.len(),
        count_b: b.history.len(),
        common_prefix_len,
        divergence,
        pinned_added,
        pinned_removed,
        identical_id: a.id == b.id,
    }
}

/// Semantic message equality: role + content + tool_name + tool_calls
/// (compared by JSON value). Two messages with identical text but
/// different tool-call arguments must NOT count as equal — otherwise
/// the diff would mask a meaningful divergence.
pub(crate) fn messages_equal(a: &Message, b: &Message) -> bool {
    if a.role != b.role || a.content != b.content || a.tool_name != b.tool_name {
        return false;
    }
    match (&a.tool_calls, &b.tool_calls) {
        (None, None) => true,
        (Some(ax), Some(bx)) => {
            // Compare via JSON value to ignore irrelevant struct-field
            // ordering and to tolerate the `serde_json::Value`
            // arguments shape used inside `ToolCallFunction`.
            serde_json::to_value(ax).ok() == serde_json::to_value(bx).ok()
        }
        _ => false,
    }
}

/// Builds an 80-char single-line preview of a message that is safe to
/// dump on stderr/stdout. Strips control characters and collapses
/// internal whitespace so a multi-line code block doesn't break the
/// caller's layout.
pub(crate) fn message_preview(m: &Message) -> String {
    let body = if m.content.is_empty() {
        if let Some(tc) = &m.tool_calls {
            if let Some(first) = tc.first() {
                format!("<tool_call:{}>", first.function.name)
            } else {
                "<empty>".to_string()
            }
        } else {
            "<empty>".to_string()
        }
    } else {
        m.content.clone()
    };
    let cleaned: String = body
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    truncate_chars(&cleaned, 80)
}

fn truncate_chars(s: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i + 1 > max_chars {
            out.push('…');
            return out;
        }
        out.push(c);
    }
    out
}

/// Multiset diff on `Vec<String>`: respects duplicates and ignores
/// order (pins are an unordered curated list semantically).
fn pinned_multiset_diff(a: &[String], b: &[String]) -> (Vec<String>, Vec<String>) {
    let mut ca: HashMap<&str, isize> = HashMap::new();
    let mut cb: HashMap<&str, isize> = HashMap::new();
    for s in a {
        *ca.entry(s.as_str()).or_insert(0) += 1;
    }
    for s in b {
        *cb.entry(s.as_str()).or_insert(0) += 1;
    }
    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut keys: Vec<&str> = ca.keys().chain(cb.keys()).copied().collect();
    keys.sort_unstable();
    keys.dedup();
    for k in keys {
        let delta = cb.get(k).copied().unwrap_or(0) - ca.get(k).copied().unwrap_or(0);
        if delta > 0 {
            for _ in 0..delta {
                added.push(k.to_string());
            }
        } else if delta < 0 {
            for _ in 0..(-delta) {
                removed.push(k.to_string());
            }
        }
    }
    (added, removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ollama::{Message, ToolCall, ToolCallFunction};

    fn msg(role: &str, content: &str) -> Message {
        Message::text(role, content)
    }

    fn session(id: &str, model: &str, history: Vec<Message>, pinned: Vec<String>) -> SessionFile {
        SessionFile {
            id: id.to_string(),
            started_at: 0,
            cwd: "/".to_string(),
            model: model.to_string(),
            history,
            pinned,
        }
    }

    #[test]
    fn identical_sessions_have_full_common_prefix_and_no_divergence() {
        let h = vec![msg("user", "hi"), msg("assistant", "hello")];
        let a = session("a", "m", h.clone(), vec![]);
        let b = session("b", "m", h, vec![]);
        let d = diff(&a, &b);
        assert_eq!(d.common_prefix_len, 2);
        assert!(d.divergence.is_none());
        assert!(!d.model_drift);
        assert!(!d.identical_id);
    }

    #[test]
    fn diverging_content_reports_index_and_previews() {
        let a = session(
            "a",
            "m",
            vec![msg("user", "hi"), msg("assistant", "x")],
            vec![],
        );
        let b = session(
            "b",
            "m",
            vec![msg("user", "hi"), msg("assistant", "y")],
            vec![],
        );
        let d = diff(&a, &b);
        assert_eq!(d.common_prefix_len, 1);
        let div = d.divergence.expect("must diverge");
        assert_eq!(div.index, 1);
        assert_eq!(div.a_preview, "x");
        assert_eq!(div.b_preview, "y");
    }

    #[test]
    fn one_side_longer_diverges_at_shorter_end() {
        let a = session("a", "m", vec![msg("user", "hi")], vec![]);
        let b = session(
            "b",
            "m",
            vec![msg("user", "hi"), msg("assistant", "ack")],
            vec![],
        );
        let d = diff(&a, &b);
        assert_eq!(d.common_prefix_len, 1);
        let div = d.divergence.expect("end-shift divergence");
        assert_eq!(div.a_role, "<end>");
        assert_eq!(div.b_role, "assistant");
    }

    #[test]
    fn model_drift_detected() {
        let h = vec![msg("user", "hi")];
        let a = session("a", "model-x", h.clone(), vec![]);
        let b = session("b", "model-y", h, vec![]);
        assert!(diff(&a, &b).model_drift);
    }

    #[test]
    fn tool_call_argument_drift_breaks_equality() {
        let mut a_msg = Message::text("assistant", "");
        a_msg.tool_calls = Some(vec![ToolCall {
            id: None,
            call_type: None,
            function: ToolCallFunction {
                name: "shell".into(),
                arguments: serde_json::json!({"cmd": "ls"}),
            },
        }]);
        let mut b_msg = Message::text("assistant", "");
        b_msg.tool_calls = Some(vec![ToolCall {
            id: None,
            call_type: None,
            function: ToolCallFunction {
                name: "shell".into(),
                arguments: serde_json::json!({"cmd": "rm -rf /"}),
            },
        }]);
        assert!(!messages_equal(&a_msg, &b_msg));
        let a = session("a", "m", vec![a_msg], vec![]);
        let b = session("b", "m", vec![b_msg], vec![]);
        let d = diff(&a, &b);
        assert_eq!(d.common_prefix_len, 0);
        let div = d.divergence.expect("tool-call drift");
        assert!(div.a_preview.contains("tool_call"));
    }

    #[test]
    fn pinned_multiset_diff_respects_duplicates() {
        let a = session("a", "m", vec![], vec!["x".into(), "x".into(), "y".into()]);
        let b = session("b", "m", vec![], vec!["x".into(), "z".into()]);
        let d = diff(&a, &b);
        // a has 2x x, b has 1x x → 1 removed. y removed. z added.
        assert_eq!(d.pinned_removed, vec!["x".to_string(), "y".to_string()]);
        assert_eq!(d.pinned_added, vec!["z".to_string()]);
    }

    #[test]
    fn message_preview_collapses_whitespace_and_truncates() {
        let m = Message::text("user", "line one\nline\ttwo  is  long ".repeat(10));
        let p = message_preview(&m);
        assert!(!p.contains('\n'));
        assert!(!p.contains('\t'));
        assert!(p.chars().count() <= 81); // 80 plus ellipsis
    }

    #[test]
    fn identical_id_flag_set_when_same_id() {
        let h = vec![msg("user", "hi")];
        let a = session("same", "m", h.clone(), vec![]);
        let b = session("same", "m", h, vec![]);
        assert!(diff(&a, &b).identical_id);
    }
}
