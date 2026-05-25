//! Native tool-calling agent loop.
//!
//! Ollama's `/api/chat` accepts a `tools` array that tool-capable models
//! (llama3.1+, qwen2.5, mistral 0.3, etc.) consult when deciding whether to
//! emit `tool_calls` instead of (or in addition to) free-form text. This
//! module turns that into an honest agent loop:
//!
//! 1. Forward the conversation + tool list to the model.
//! 2. If the model returns plain `content`, we're done — print and persist.
//! 3. If the model returns `tool_calls`, execute each one through the
//!    [`McpManager`] (which routes between built-in and external MCP tools),
//!    append a `role:"tool"` result message, and loop.
//!
//! The loop has a fixed step cap so a misbehaving model can't burn forever.
//! Tool execution failures become tool messages (so the model can react and
//! recover) rather than aborting the whole turn.
//!
//! Older models simply ignore the `tools` field and the loop collapses to a
//! single iteration of streaming chat — no behavioral regression for users
//! who haven't pulled a tool-capable model yet.

use crate::mcp_manager::McpManager;
use crate::ollama::{ToolFunction, ToolSpec};

/// Maximum number of "model → tool → model" iterations per user turn before
/// the loop bails out with a diagnostic. Generous enough for realistic
/// multi-step plans (write file → run tests → grep output → edit again),
/// small enough that a runaway model gives up well before the user does.
pub const MAX_AGENT_STEPS: usize = 12;

/// Builds the Ollama-shaped `tools` list from every tool the MCP manager
/// knows about (built-in + each connected external server). Returns `None`
/// when there are no tools at all so the caller can skip the field and let
/// older Ollama versions handle the payload identically to before.
pub fn build_tool_specs(mcp: &McpManager) -> Option<Vec<ToolSpec>> {
    let tools = mcp.list_tools();
    if tools.is_empty() {
        return None;
    }
    Some(
        tools
            .into_iter()
            .map(|t| ToolSpec {
                tool_type: "function".to_string(),
                function: ToolFunction {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.input_schema.clone(),
                },
            })
            .collect(),
    )
}

/// Renders a tool's `ToolCallResult` content into a single plain-text blob
/// suitable for feeding back to the model as a `role:"tool"` message.
/// Non-text content blocks are summarized rather than dropped so the model
/// at least learns they existed.
pub fn render_tool_result(result: &crate::mcp_client::ToolCallResult) -> String {
    let mut buf = String::new();
    for c in &result.content {
        if c.content_type == "text" {
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(&c.text);
        } else {
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(&format!("[non-text content: {}]", c.content_type));
        }
    }
    if let Some(true) = result.is_error {
        // Tag errors so the model can tell apart "tool ran and returned an
        // empty string" from "tool failed" — important when deciding whether
        // to retry.
        if buf.is_empty() {
            buf.push_str("[tool reported error with no message]");
        } else {
            buf.insert_str(0, "[tool error] ");
        }
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp_client::{Content, ToolCallResult};

    #[test]
    fn render_tool_result_joins_text_blocks() {
        let r = ToolCallResult {
            content: vec![
                Content {
                    content_type: "text".into(),
                    text: "line 1".into(),
                },
                Content {
                    content_type: "text".into(),
                    text: "line 2".into(),
                },
            ],
            is_error: None,
        };
        assert_eq!(render_tool_result(&r), "line 1\nline 2");
    }

    #[test]
    fn render_tool_result_tags_errors() {
        let r = ToolCallResult {
            content: vec![Content {
                content_type: "text".into(),
                text: "boom".into(),
            }],
            is_error: Some(true),
        };
        assert!(render_tool_result(&r).starts_with("[tool error]"));
    }

    #[test]
    fn render_tool_result_empty_error_still_signals() {
        let r = ToolCallResult {
            content: vec![],
            is_error: Some(true),
        };
        assert!(render_tool_result(&r).contains("[tool reported error"));
    }

    #[test]
    fn render_tool_result_describes_non_text_content() {
        let r = ToolCallResult {
            content: vec![Content {
                content_type: "image".into(),
                text: "<binary>".into(),
            }],
            is_error: None,
        };
        let out = render_tool_result(&r);
        assert!(out.contains("[non-text content: image]"), "got: {out}");
    }
}
