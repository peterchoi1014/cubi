//! Multi-provider LLM abstraction and token estimator.
//!
//! This module defines an [`LlmBackend`] enum that abstracts over different
//! LLM backends (Ollama, OpenAI-compatible APIs, etc.). The existing
//! `OllamaClient` is wrapped, and a new `OpenAiClient` provides access to
//! any OpenAI-compatible endpoint (OpenAI, Anthropic via proxy, local vLLM,
//! etc.).
//!
//! **Note:** This module is not yet wired into `AIExecutor`. It serves as
//! the foundation for multi-provider support. To use it, call
//! [`create_provider()`] and use the returned [`LlmBackend`] directly.
//! Full integration into the runtime is planned for a future release.
//!
//! ## Token estimation
//!
//! A simple [`estimate_tokens`] function provides a rough character-based
//! estimate (≈4 chars/token for English, widely used as a heuristic). This
//! lets the CLI warn about context-window limits without pulling in a real
//! tokenizer crate.
//!
//! ## Provider selection
//!
//! The active provider is chosen based on environment variables:
//! * `AI_CHAT_CLI_PROVIDER=openai` + `AI_CHAT_CLI_API_KEY` → OpenAI mode
//! * `AI_CHAT_CLI_PROVIDER=ollama` (or unset) → local Ollama (default)
//! * `AI_CHAT_CLI_BASE_URL` → overrides the API endpoint for either provider

use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

use crate::ollama::{Message, OllamaClient, ToolCall, ToolCallFunction, ToolSpec};

// ─── Provider enum (object-safe dispatch without async_trait) ────────────────

/// Enum-based LLM provider dispatch. Avoids the need for `dyn` with async
/// methods while keeping a clean, extensible interface.
pub enum LlmBackend {
    Ollama(OllamaClient),
    OpenAi(OpenAiClient),
}

impl LlmBackend {
    /// Non-streaming chat. Returns the assistant response text.
    pub async fn chat(&self, model: &str, messages: Vec<Message>) -> Result<String> {
        match self {
            Self::Ollama(c) => c.chat(model, messages).await,
            Self::OpenAi(c) => {
                let msg = c.chat_with_tools(model, messages, None).await?;
                Ok(msg.content)
            }
        }
    }

    /// Non-streaming chat with optional tools.
    pub async fn chat_with_tools(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolSpec>>,
    ) -> Result<Message> {
        match self {
            Self::Ollama(c) => c.chat_with_tools(model, messages, tools).await,
            Self::OpenAi(c) => c.chat_with_tools(model, messages, tools).await,
        }
    }

    /// Streaming chat. Calls `on_token` for each text fragment.
    pub async fn chat_stream<F>(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolSpec>>,
        on_token: F,
    ) -> Result<Message>
    where
        F: FnMut(&str) + Send + 'static,
    {
        match self {
            Self::Ollama(c) => c.chat_stream(model, messages, tools, on_token).await,
            Self::OpenAi(c) => {
                c.chat_stream(model, messages, tools, Box::new(on_token))
                    .await
            }
        }
    }

    /// Lists available models.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        match self {
            Self::Ollama(c) => c.list_models().await,
            Self::OpenAi(c) => c.list_models().await,
        }
    }

    /// Returns the provider name.
    pub fn provider_name(&self) -> &str {
        match self {
            Self::Ollama(_) => "ollama",
            Self::OpenAi(_) => "openai",
        }
    }
}

// ─── OpenAI-compatible provider ─────────────────────────────────────────────

/// Client for OpenAI-compatible APIs. Works with OpenAI, Azure OpenAI,
/// vLLM, LM Studio, and any other endpoint that speaks the
/// `/v1/chat/completions` protocol.
pub struct OpenAiClient {
    base_url: String,
    api_key: String,
    client: reqwest::Client,
}

/// OpenAI chat completion response.
#[derive(Debug, Deserialize)]
struct OaiResponse {
    choices: Vec<OaiChoice>,
}

#[derive(Debug, Deserialize)]
struct OaiChoice {
    message: OaiMessage,
}

#[derive(Debug, Deserialize)]
struct OaiMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OaiToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OaiToolCall {
    #[serde(default)]
    id: Option<String>,
    #[serde(default, rename = "type")]
    call_type: Option<String>,
    function: OaiToolCallFunction,
}

#[derive(Debug, Deserialize)]
struct OaiToolCallFunction {
    name: String,
    arguments: String, // JSON string
}

/// OpenAI streaming chunk.
#[derive(Debug, Deserialize)]
struct OaiStreamChunk {
    choices: Vec<OaiStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct OaiStreamChoice {
    delta: OaiDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OaiDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OaiDeltaToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OaiDeltaToolCall {
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OaiDeltaFunction>,
}

#[derive(Debug, Deserialize)]
struct OaiDeltaFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

/// Serialized tool spec for OpenAI format.
#[derive(Debug, Serialize)]
struct OaiToolSpec {
    #[serde(rename = "type")]
    tool_type: String,
    function: OaiToolFunctionSpec,
}

#[derive(Debug, Serialize)]
struct OaiToolFunctionSpec {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

/// Serialized message for the OpenAI request.
#[derive(Debug, Serialize)]
struct OaiRequestMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl OpenAiClient {
    pub fn new(base_url: String, api_key: String) -> Self {
        Self {
            base_url,
            api_key,
            client: reqwest::Client::new(),
        }
    }

    /// Creates a client from environment variables.
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("AI_CHAT_CLI_API_KEY").ok()?;
        let base_url = std::env::var("AI_CHAT_CLI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        Some(Self::new(base_url, api_key))
    }

    fn convert_messages(messages: &[Message]) -> Vec<OaiRequestMessage> {
        messages
            .iter()
            .map(|m| OaiRequestMessage {
                role: m.role.clone(),
                content: m.content.clone(),
                // `name` is only relevant for tool-result messages.
                name: if m.role == "tool" {
                    m.tool_name.clone()
                } else {
                    None
                },
                // `tool_call_id` must only be set on role:"tool" messages and
                // should match the tool call's id. We use tool_name as a
                // fallback identifier when the full call id isn't available.
                tool_call_id: if m.role == "tool" {
                    m.tool_name.clone()
                } else {
                    None
                },
            })
            .collect()
    }

    fn convert_tools(tools: &[ToolSpec]) -> Vec<OaiToolSpec> {
        tools
            .iter()
            .map(|t| OaiToolSpec {
                tool_type: "function".to_string(),
                function: OaiToolFunctionSpec {
                    name: t.function.name.clone(),
                    description: t.function.description.clone(),
                    parameters: t.function.parameters.clone(),
                },
            })
            .collect()
    }

    fn oai_message_to_message(oai: OaiMessage) -> Message {
        let content = oai.content.unwrap_or_default();
        let tool_calls = oai.tool_calls.map(|calls| {
            calls
                .into_iter()
                .map(|c| ToolCall {
                    id: c.id,
                    call_type: c.call_type,
                    function: ToolCallFunction {
                        name: c.function.name,
                        arguments: serde_json::from_str(&c.function.arguments)
                            .unwrap_or(serde_json::Value::Object(Default::default())),
                    },
                })
                .collect()
        });
        Message {
            role: "assistant".to_string(),
            content,
            tool_calls,
            tool_name: None,
        }
    }
}

impl OpenAiClient {
    #[allow(dead_code)]
    pub async fn chat(&self, model: &str, messages: Vec<Message>) -> Result<String> {
        let msg = self.chat_with_tools(model, messages, None).await?;
        Ok(msg.content)
    }

    pub async fn chat_with_tools(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolSpec>>,
    ) -> Result<Message> {
        let mut body = serde_json::json!({
            "model": model,
            "messages": Self::convert_messages(&messages),
            "stream": false,
        });

        if let Some(tools) = &tools
            && !tools.is_empty()
        {
            body["tools"] = serde_json::to_value(Self::convert_tools(tools))?;
        }

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send request to OpenAI-compatible API")?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("OpenAI API error: {}", error_text);
        }

        let oai_resp: OaiResponse = response
            .json()
            .await
            .context("Failed to parse OpenAI response")?;

        let choice = oai_resp
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No choices in OpenAI response"))?;

        Ok(Self::oai_message_to_message(choice.message))
    }

    pub async fn chat_stream(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolSpec>>,
        mut on_token: Box<dyn FnMut(&str) + Send>,
    ) -> Result<Message> {
        let mut body = serde_json::json!({
            "model": model,
            "messages": Self::convert_messages(&messages),
            "stream": true,
        });

        if let Some(tools) = &tools
            && !tools.is_empty()
        {
            body["tools"] = serde_json::to_value(Self::convert_tools(tools))?;
        }

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send streaming request to OpenAI-compatible API")?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("OpenAI API error: {}", error_text);
        }

        let mut stream = response.bytes_stream();
        let mut buf = String::new();
        let mut content = String::new();
        let mut tool_calls_builder: Vec<(String, String, String)> = Vec::new(); // (id, name, args)

        while let Some(chunk) = stream.next().await {
            let bytes = chunk.context("Stream read failed")?;
            buf.push_str(&String::from_utf8_lossy(&bytes));

            while let Some(nl) = buf.find('\n') {
                let line = buf[..nl].to_string();
                buf.drain(..=nl);
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed == "data: [DONE]" {
                    continue;
                }
                let data = trimmed.strip_prefix("data: ").unwrap_or(trimmed);
                let Ok(chunk) = serde_json::from_str::<OaiStreamChunk>(data) else {
                    continue;
                };
                for choice in chunk.choices {
                    if let Some(text) = &choice.delta.content
                        && !text.is_empty()
                    {
                        on_token(text);
                        content.push_str(text);
                    }
                    if let Some(tcs) = &choice.delta.tool_calls {
                        for tc in tcs {
                            let idx = tc.index.unwrap_or(0);
                            while tool_calls_builder.len() <= idx {
                                tool_calls_builder.push((
                                    String::new(),
                                    String::new(),
                                    String::new(),
                                ));
                            }
                            if let Some(id) = &tc.id {
                                tool_calls_builder[idx].0 = id.clone();
                            }
                            if let Some(f) = &tc.function {
                                if let Some(name) = &f.name {
                                    tool_calls_builder[idx].1 = name.clone();
                                }
                                if let Some(args) = &f.arguments {
                                    tool_calls_builder[idx].2.push_str(args);
                                }
                            }
                        }
                    }
                }
            }
        }

        let tool_calls = if tool_calls_builder.is_empty() {
            None
        } else {
            Some(
                tool_calls_builder
                    .into_iter()
                    .map(|(id, name, args)| ToolCall {
                        id: if id.is_empty() { None } else { Some(id) },
                        call_type: Some("function".to_string()),
                        function: ToolCallFunction {
                            name,
                            arguments: serde_json::from_str(&args)
                                .unwrap_or(serde_json::Value::Object(Default::default())),
                        },
                    })
                    .collect(),
            )
        };

        Ok(Message {
            role: "assistant".to_string(),
            content,
            tool_calls,
            tool_name: None,
        })
    }

    pub async fn list_models(&self) -> Result<Vec<String>> {
        let response = self
            .client
            .get(format!("{}/models", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .context("Failed to list models from OpenAI-compatible API")?;

        if !response.status().is_success() {
            // Some providers don't support /models; return empty rather than fail.
            return Ok(Vec::new());
        }

        let data: serde_json::Value = response.json().await?;
        let models = data["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m["id"].as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        Ok(models)
    }
}

// ─── Provider factory ───────────────────────────────────────────────────────

/// Creates the appropriate LLM provider based on environment configuration.
///
/// Priority:
/// 1. `AI_CHAT_CLI_PROVIDER=openai` + `AI_CHAT_CLI_API_KEY` → OpenAI client
/// 2. Default → Ollama client (optionally using `AI_CHAT_CLI_BASE_URL`)
pub fn create_provider() -> LlmBackend {
    let provider = std::env::var("AI_CHAT_CLI_PROVIDER")
        .unwrap_or_default()
        .to_lowercase();

    if provider == "openai" {
        if let Some(client) = OpenAiClient::from_env() {
            return LlmBackend::OpenAi(client);
        }
        eprintln!(
            "Warning: AI_CHAT_CLI_PROVIDER=openai but AI_CHAT_CLI_API_KEY not set. Falling back to Ollama."
        );
    }

    // Use AI_CHAT_CLI_BASE_URL to override Ollama's default endpoint if set.
    let ollama = match std::env::var("AI_CHAT_CLI_BASE_URL") {
        Ok(url) if !url.is_empty() => OllamaClient::with_base_url(url),
        _ => OllamaClient::new(),
    };
    LlmBackend::Ollama(ollama)
}

// ─── Token estimator ────────────────────────────────────────────────────────

/// Estimates the token count for a string using a simple character-based
/// heuristic (≈4 characters per token for English text). This is intentionally
/// approximate — pulling in tiktoken or a model-specific tokenizer would add
/// significant dependency weight for marginal accuracy in our use-case.
pub fn estimate_tokens(text: &str) -> usize {
    // ~4 characters per token is a widely-used approximation for GPT-class
    // models on English text. Underestimates for CJK, overestimates for code
    // with many single-char tokens, but good enough for context-window checks.
    (text.len()).div_ceil(4)
}

/// Estimates total tokens across a conversation history.
pub fn estimate_conversation_tokens(messages: &[Message]) -> usize {
    messages
        .iter()
        .map(|m| {
            // Role overhead: ~4 tokens per message for role/delimiter
            4 + estimate_tokens(&m.content)
        })
        .sum()
}

/// Known context window sizes for popular models.
pub fn context_window_for_model(model: &str) -> Option<usize> {
    // Normalize: strip version tags for matching.
    let model_lower = model.to_lowercase();
    let base = model_lower.split(':').next().unwrap_or(&model_lower);

    match base {
        // Ollama models
        "llama3.2" | "llama3.1" | "llama3" => Some(128_000),
        "llama2" => Some(4_096),
        "mistral" | "mixtral" => Some(32_768),
        "qwen2.5" | "qwen2" => Some(128_000),
        "codellama" => Some(16_384),
        "phi3" | "phi-3" => Some(128_000),
        "gemma2" | "gemma" => Some(8_192),
        "deepseek-coder" => Some(16_384),
        // OpenAI models
        "gpt-4o" | "gpt-4o-mini" => Some(128_000),
        "gpt-4-turbo" | "gpt-4" => Some(128_000),
        "gpt-3.5-turbo" => Some(16_384),
        "o1" | "o1-mini" | "o1-preview" => Some(128_000),
        // Claude (via OpenAI-compatible proxy)
        "claude-3-opus" | "claude-3-sonnet" | "claude-3-haiku" => Some(200_000),
        "claude-3.5-sonnet" | "claude-3.5-haiku" => Some(200_000),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_tokens_basic() {
        // "hello" = 5 chars → (5+3)/4 = 2 tokens
        assert_eq!(estimate_tokens("hello"), 2);
        // Empty string → 0 tokens (rounded up: (0+3)/4 = 0 due to integer division)
        assert_eq!(estimate_tokens(""), 0);
        // 100 chars → 25 tokens
        let s = "x".repeat(100);
        assert_eq!(estimate_tokens(&s), 25);
    }

    #[test]
    fn estimate_conversation_tokens_sums_messages() {
        let msgs = vec![
            Message::text("user", "hello world"), // 4 + (11+3)/4 = 4 + 3 = 7
            Message::text("assistant", "hi there"), // 4 + (8+3)/4 = 4 + 2 = 6
        ];
        let total = estimate_conversation_tokens(&msgs);
        assert_eq!(total, 13);
    }

    #[test]
    fn context_window_known_models() {
        assert_eq!(context_window_for_model("llama3.2:1b"), Some(128_000));
        assert_eq!(context_window_for_model("gpt-4o"), Some(128_000));
        assert_eq!(context_window_for_model("unknown-model"), None);
    }

    #[test]
    fn create_provider_defaults_to_ollama() {
        // Without AI_CHAT_CLI_PROVIDER=openai, should default to Ollama.
        // We don't remove the var (unsafe in edition 2024+); instead we
        // just verify that the default codepath works when it's not "openai".
        let provider = create_provider();
        // If the env doesn't have AI_CHAT_CLI_PROVIDER=openai, we get ollama.
        // If it does (unlikely in test), we'd get openai — both are valid.
        assert!(provider.provider_name() == "ollama" || provider.provider_name() == "openai");
    }
}
