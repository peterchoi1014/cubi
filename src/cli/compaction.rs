use crate::ollama::Message;

pub(super) const COMPACTED_SUMMARY_PREFIX: &str =
    "SYSTEM: Compacted summary of earlier conversation:\n\n";

pub(super) struct CompactionPlan {
    pub summarized_count: usize,
    pub retained_count: usize,
    pub transcript: String,
    preserved_system: Vec<Message>,
    retained_tail: Vec<Message>,
}

impl CompactionPlan {
    pub fn retained_tokens(&self) -> usize {
        self.preserved_system
            .iter()
            .chain(self.retained_tail.iter())
            .map(|message| 4 + crate::llm::estimate_tokens(&message.content))
            .sum()
    }

    pub fn rebuild(self, summary: &str) -> Vec<Message> {
        let mut history =
            Vec::with_capacity(self.preserved_system.len() + self.retained_tail.len() + 1);
        history.extend(self.preserved_system);
        history.push(Message::text(
            "system",
            format!("{}{}", COMPACTED_SUMMARY_PREFIX, summary),
        ));
        history.extend(self.retained_tail);
        history
    }
}

pub(super) fn plan_compaction(
    history: &[Message],
    keep_recent_turns: usize,
) -> Option<CompactionPlan> {
    let user_indices: Vec<usize> = history
        .iter()
        .enumerate()
        .filter_map(|(index, message)| (message.role == "user").then_some(index))
        .collect();
    if user_indices.len() <= keep_recent_turns {
        return None;
    }

    let cutoff = user_indices[user_indices.len() - keep_recent_turns];
    let summarized: Vec<&Message> = history[..cutoff]
        .iter()
        .filter(|message| message.role != "system" || is_compacted_summary(message))
        .collect();
    if summarized.is_empty() {
        return None;
    }
    let preserved_system = history[..cutoff]
        .iter()
        .filter(|message| message.role == "system" && !is_compacted_summary(message))
        .cloned()
        .collect();
    let retained_tail = history[cutoff..].to_vec();
    let retained_count = retained_tail
        .iter()
        .filter(|message| message.role != "system")
        .count();
    let summarized_count = summarized
        .iter()
        .filter(|message| message.role != "system")
        .count();

    Some(CompactionPlan {
        summarized_count,
        retained_count,
        transcript: transcript_for(&summarized),
        preserved_system,
        retained_tail,
    })
}

fn transcript_for(messages: &[&Message]) -> String {
    let mut transcript = String::new();
    for message in messages {
        let role = match message.role.as_str() {
            "user" => "User",
            "assistant" => "Assistant",
            "tool" => "Tool",
            "system" if is_compacted_summary(message) => "Prior summary",
            other => other,
        };
        if !message.content.is_empty() {
            transcript.push_str(role);
            transcript.push_str(": ");
            transcript.push_str(&truncate_chars(&message.content, 2_000));
            transcript.push('\n');
        }
        if let Some(calls) = &message.tool_calls {
            for call in calls {
                transcript.push_str("Assistant tool call ");
                transcript.push_str(&call.function.name);
                transcript.push_str(": ");
                let arguments = serde_json::to_string(&call.function.arguments)
                    .unwrap_or_else(|_| "{}".to_string());
                transcript.push_str(&truncate_chars(&arguments, 2_000));
                transcript.push('\n');
            }
        }
    }
    transcript
}

fn truncate_chars(value: &str, limit: usize) -> String {
    if value.chars().count() <= limit {
        value.to_string()
    } else {
        let mut truncated: String = value.chars().take(limit).collect();
        truncated.push('…');
        truncated
    }
}

pub(super) fn bound_summary(summary: &str, token_budget: usize) -> String {
    truncate_chars(summary, token_budget.saturating_mul(4))
}

pub(super) fn is_compacted_summary(message: &Message) -> bool {
    message.role == "system" && message.content.starts_with(COMPACTED_SUMMARY_PREFIX)
}

pub(super) fn memory_extraction_enabled() -> bool {
    crate::config::with(|config| config.compact_extract_memories)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ollama::{Message, ToolCall, ToolCallFunction};
    use std::collections::HashSet;

    fn tool_call(id: &str, name: &str, argument: &str) -> Message {
        let mut message = Message::text("assistant", "");
        message.tool_calls = Some(vec![ToolCall {
            id: Some(id.to_string()),
            call_type: Some("function".to_string()),
            function: ToolCallFunction {
                name: name.to_string(),
                arguments: serde_json::json!({"value": argument}),
            },
        }]);
        message
    }

    fn turn(id: &str) -> Vec<Message> {
        vec![
            Message::text("user", format!("user-{id}")),
            tool_call(id, "read_file", &format!("arg-{id}")),
            Message::tool_result("read_file", format!("result-{id}"), Some(id.to_string())),
            Message::text("assistant", format!("final-{id}")),
        ]
    }

    fn assert_tool_pairs(history: &[Message]) {
        let calls: HashSet<&str> = history
            .iter()
            .filter_map(|message| message.tool_calls.as_ref())
            .flatten()
            .filter_map(|call| call.id.as_deref())
            .collect();
        let results: HashSet<&str> = history
            .iter()
            .filter_map(|message| message.tool_call_id.as_deref())
            .collect();
        assert_eq!(calls, results);
    }

    #[test]
    fn planner_keeps_complete_recent_turns_and_summarizes_tool_arguments() {
        let mut history = vec![Message::text("system", "base")];
        history.extend(turn("call-a"));
        history.extend(turn("call-b"));
        history.extend(turn("call-c"));

        let plan = plan_compaction(&history, 2).unwrap();

        assert!(plan.transcript.contains("read_file"));
        assert!(plan.transcript.contains("arg-call-a"));
        assert!(!plan.transcript.contains("arg-call-b"));
        let rebuilt = plan.rebuild("summary");
        assert_tool_pairs(&rebuilt);
        assert!(
            rebuilt
                .iter()
                .any(|message| message.content == "user-call-b")
        );
        assert!(
            !rebuilt
                .iter()
                .any(|message| message.content == "user-call-a")
        );
    }

    #[test]
    fn repeated_compaction_folds_prior_summary_and_preserves_pairs() {
        let mut history = vec![Message::text("system", "base")];
        history.extend(turn("call-a"));
        history.extend(turn("call-b"));
        history.extend(turn("call-c"));
        history = plan_compaction(&history, 2)
            .unwrap()
            .rebuild("first summary");
        history.extend(turn("call-d"));

        let plan = plan_compaction(&history, 2).unwrap();
        assert!(plan.transcript.contains("first summary"));
        let rebuilt = plan.rebuild("second summary");

        assert_eq!(
            rebuilt
                .iter()
                .filter(|message| is_compacted_summary(message))
                .count(),
            1
        );
        assert_tool_pairs(&rebuilt);
    }

    #[test]
    fn bounded_summary_truncates_by_token_budget() {
        let bounded = bound_summary(&"x".repeat(8_000), 100);
        assert!(crate::llm::estimate_tokens(&bounded) <= 101);
    }
}
