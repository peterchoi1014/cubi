use crate::ollama::{Message, ToolSpec};

const DEFAULT_OLLAMA_NUM_CTX_CAP: usize = 32_768;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ContextBudget {
    pub window: usize,
    pub completion_reserve: usize,
    pub prompt_limit: usize,
}

pub(super) fn effective_budget(model: &str, provider: &str) -> Option<ContextBudget> {
    let (ollama_override, openai_override) =
        crate::config::with(|config| (config.generation.num_ctx, config.generation.context_window));
    effective_budget_with_overrides(model, provider, ollama_override, openai_override)
}

fn effective_budget_with_overrides(
    model: &str,
    provider: &str,
    ollama_override: Option<usize>,
    openai_override: Option<usize>,
) -> Option<ContextBudget> {
    let nominal = crate::llm::context_window_for_model(model);
    let configured = match provider {
        "ollama" => Some(ollama_override.unwrap_or(DEFAULT_OLLAMA_NUM_CTX_CAP)),
        "openai" | "fake" => openai_override,
        _ => None,
    };
    let window = match (nominal, configured) {
        (Some(nominal), Some(configured)) => nominal.min(configured),
        (Some(nominal), None) => nominal,
        (None, Some(configured)) => configured,
        (None, None) => return None,
    };
    let completion_reserve = (window / 10).clamp(512, 4_096).min(window);
    Some(ContextBudget {
        window,
        completion_reserve,
        prompt_limit: window.saturating_sub(completion_reserve),
    })
}

pub(super) fn estimate_request_tokens(messages: &[Message], tools: Option<&[ToolSpec]>) -> usize {
    let message_tokens = messages
        .iter()
        .map(|message| {
            let mut tokens = 4 + crate::llm::estimate_tokens(&message.content);
            if let Some(calls) = &message.tool_calls {
                tokens += serialized_tokens(calls);
            }
            if let Some(name) = &message.tool_name {
                tokens += crate::llm::estimate_tokens(name);
            }
            if let Some(id) = &message.tool_call_id {
                tokens += crate::llm::estimate_tokens(id);
            }
            tokens
        })
        .sum::<usize>();
    message_tokens + tools.map(serialized_tokens).unwrap_or(0)
}

fn serialized_tokens(value: &(impl serde::Serialize + ?Sized)) -> usize {
    serde_json::to_string(value)
        .map(|json| crate::llm::estimate_tokens(&json))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ollama::{Message, ToolCall, ToolCallFunction, ToolFunction, ToolSpec};

    #[test]
    fn ollama_effective_window_uses_runtime_cap() {
        let budget = effective_budget_with_overrides("qwen3.5:9b", "ollama", None, None).unwrap();

        assert_eq!(budget.window, 32_768);
        assert!(budget.prompt_limit < budget.window);
    }

    #[test]
    fn configured_openai_window_is_clamped_to_nominal_window() {
        let budget =
            effective_budget_with_overrides("qwen3:8b", "openai", None, Some(65_536)).unwrap();

        assert_eq!(budget.window, 32_768);
    }

    #[test]
    fn request_estimate_includes_tool_schemas_and_call_arguments() {
        let mut assistant = Message::text("assistant", "");
        assistant.tool_calls = Some(vec![ToolCall {
            id: Some("call-1".to_string()),
            call_type: Some("function".to_string()),
            function: ToolCallFunction {
                name: "read_file".to_string(),
                arguments: serde_json::json!({"path": "x".repeat(4_000)}),
            },
        }]);
        let messages = vec![Message::text("user", "small"), assistant];
        let without_tools = estimate_request_tokens(&messages, None);
        let tools = vec![ToolSpec {
            tool_type: "function".to_string(),
            function: ToolFunction {
                name: "read_file".to_string(),
                description: "d".repeat(4_000),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"path": {"type": "string"}}
                }),
            },
        }];

        let with_tools = estimate_request_tokens(&messages, Some(&tools));

        assert!(without_tools > 1_000, "tool-call arguments must be counted");
        assert!(
            with_tools > without_tools + 1_000,
            "serialized schemas must be counted"
        );
    }
}
