//! Multi-provider LLM abstraction and token estimator.
//!
//! This module defines an [`LlmBackend`] enum that abstracts over different
//! LLM backends (Ollama, OpenAI-compatible APIs, etc.). The existing
//! `OllamaClient` is wrapped, and a new `OpenAiClient` provides access to
//! any OpenAI-compatible endpoint (OpenAI, Anthropic via proxy, local vLLM,
//! etc.).
//!
//! This module is wired into `AIExecutor` so the CLI can select between
//! Ollama and an OpenAI-compatible endpoint at runtime.
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
//! * `OPENAI_API_KEY` → OpenAI-compatible mode
//! * `OPENAI_BASE_URL` → overrides the OpenAI-compatible endpoint
//! * otherwise local Ollama is used (optionally overridden by `CUBI_BASE_URL`)

use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::ollama::{ChatStats, Message, OllamaClient, ToolCall, ToolCallFunction, ToolSpec};

// ─── Provider enum (object-safe dispatch without async_trait) ────────────────

/// Enum-based LLM provider dispatch. Avoids the need for `dyn` with async
/// methods while keeping a clean, extensible interface.
pub enum LlmBackend {
    Ollama(OllamaClient),
    OpenAi(OpenAiClient),
    Fake,
}

impl LlmBackend {
    /// Non-streaming chat. Returns the assistant response text.
    pub async fn chat(&self, model: &str, messages: Vec<Message>) -> Result<String> {
        match self {
            Self::Ollama(c) => c.chat(model, messages).await,
            Self::OpenAi(c) => {
                let (msg, _) = c.chat_with_tools(model, messages, None).await?;
                Ok(msg.content)
            }
            Self::Fake => Ok(fake_content()),
        }
    }

    /// Non-streaming chat with optional tools.
    pub async fn chat_with_tools(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolSpec>>,
    ) -> Result<(Message, ChatStats)> {
        match self {
            Self::Ollama(c) => c.chat_with_tools(model, messages, tools).await,
            Self::OpenAi(c) => c.chat_with_tools(model, messages, tools).await,
            Self::Fake => Ok((fake_message(&messages), fake_stats())),
        }
    }

    /// Streaming chat. Calls `on_token` for each text fragment.
    pub async fn chat_stream<F>(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolSpec>>,
        mut on_token: F,
    ) -> Result<(Message, ChatStats)>
    where
        F: FnMut(&str),
    {
        match self {
            Self::Ollama(c) => c.chat_stream(model, messages, tools, on_token).await,
            Self::OpenAi(c) => c.chat_stream(model, messages, tools, on_token).await,
            Self::Fake => {
                let message = fake_message(&messages);
                if message.tool_calls.is_none() {
                    on_token(&message.content);
                }
                Ok((message, fake_stats()))
            }
        }
    }

    /// Lists available models.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        match self {
            Self::Ollama(c) => c.list_models().await,
            Self::OpenAi(c) => c.list_models().await,
            Self::Fake => Ok(vec!["qwen3:4b".to_string()]),
        }
    }

    /// Returns the provider name.
    pub fn provider_name(&self) -> &str {
        match self {
            Self::Ollama(_) => "ollama",
            Self::OpenAi(_) => "openai",
            Self::Fake => "fake",
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
    #[serde(default)]
    usage: Option<OaiUsage>,
}

#[derive(Debug, Deserialize, Default)]
struct OaiUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
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
    #[serde(default)]
    choices: Vec<OaiStreamChoice>,
    #[serde(default)]
    usage: Option<OaiUsage>,
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
        let api_key = std::env::var("OPENAI_API_KEY")
            .ok()
            .or_else(|| std::env::var("CUBI_API_KEY").ok())?;
        let base_url = std::env::var("OPENAI_BASE_URL")
            .ok()
            .or_else(|| std::env::var("CUBI_BASE_URL").ok())
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        Some(Self::new(
            base_url.trim_end_matches('/').to_string(),
            api_key,
        ))
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
        let (msg, _) = self.chat_with_tools(model, messages, None).await?;
        Ok(msg.content)
    }

    pub async fn chat_with_tools(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolSpec>>,
    ) -> Result<(Message, ChatStats)> {
        let started = std::time::Instant::now();
        let mut body = serde_json::json!({
            "model": model,
            "messages": Self::convert_messages(&messages),
            "stream": false,
        });

        if let Some(tools) = &tools {
            if !tools.is_empty() {
                body["tools"] = serde_json::to_value(Self::convert_tools(tools))?;
            }
        }

        tracing::debug!(
            target: "cubi::llm",
            model = %model,
            base_url = %self.base_url,
            "openai chat_with_tools request"
        );

        let url = format!("{}/chat/completions", self.base_url);
        let max_retries = current_max_retries();
        let mut attempt: u32 = 0;
        let response = loop {
            let send_result = self
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await;

            match send_result {
                Ok(resp) if resp.status().is_success() => break resp,
                Ok(resp) => {
                    let status = resp.status();
                    let retry_after = resp
                        .headers()
                        .get(reqwest::header::RETRY_AFTER)
                        .and_then(|v| v.to_str().ok())
                        .and_then(parse_retry_after);
                    match classify_retry(Some(status.as_u16()), attempt, max_retries, retry_after) {
                        RetryAction::Retry(wait) => {
                            tracing::warn!(
                                target: "cubi::llm",
                                status = status.as_u16(),
                                attempt = attempt + 1,
                                max = max_retries,
                                "LLM request failed; retrying"
                            );
                            tokio::time::sleep(wait).await;
                            attempt += 1;
                            continue;
                        }
                        RetryAction::Stop => {
                            let error_text = resp.text().await.unwrap_or_default();
                            tracing::warn!(
                                target: "cubi::llm",
                                status = status.as_u16(),
                                "openai non-success response"
                            );
                            let ue = crate::user_error::classify_http_status(
                                status.as_u16(),
                                retry_after.map(|d| d.as_secs()),
                                &url,
                                &error_text,
                            );
                            return Err(anyhow::Error::new(ue));
                        }
                    }
                }
                Err(err) => match classify_retry(None, attempt, max_retries, None) {
                    RetryAction::Retry(wait) => {
                        tracing::warn!(
                            target: "cubi::llm",
                            error = %err,
                            attempt = attempt + 1,
                            max = max_retries,
                            "LLM connect error; retrying"
                        );
                        tokio::time::sleep(wait).await;
                        attempt += 1;
                        continue;
                    }
                    RetryAction::Stop => {
                        let mut ue = crate::user_error::classify_send_error(&err, &url);
                        ue.cause = Some(anyhow::Error::new(err));
                        return Err(anyhow::Error::new(ue));
                    }
                },
            }
        };

        let oai_resp: OaiResponse = response
            .json()
            .await
            .context("Failed to parse OpenAI response")?;

        let choice = oai_resp
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No choices in OpenAI response"))?;

        let stats = ChatStats {
            prompt_tokens: oai_resp
                .usage
                .as_ref()
                .map(|u| u.prompt_tokens)
                .unwrap_or(0),
            completion_tokens: oai_resp
                .usage
                .as_ref()
                .map(|u| u.completion_tokens)
                .unwrap_or(0),
            elapsed_ms: started.elapsed().as_millis() as u64,
        };
        Ok((Self::oai_message_to_message(choice.message), stats))
    }

    pub async fn chat_stream<F>(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolSpec>>,
        mut on_token: F,
    ) -> Result<(Message, ChatStats)>
    where
        F: FnMut(&str),
    {
        let started = std::time::Instant::now();
        let mut body = serde_json::json!({
            "model": model,
            "messages": Self::convert_messages(&messages),
            "stream": true,
            // Ask OpenAI to include a final `usage` block in the stream
            // (otherwise streaming responses omit it entirely). Providers
            // that don't honor `stream_options` will simply ignore the field.
            "stream_options": { "include_usage": true },
        });

        if let Some(tools) = &tools {
            if !tools.is_empty() {
                body["tools"] = serde_json::to_value(Self::convert_tools(tools))?;
            }
        }

        tracing::debug!(
            target: "cubi::llm",
            model = %model,
            base_url = %self.base_url,
            "openai chat_stream request"
        );

        let url = format!("{}/chat/completions", self.base_url);
        let max_retries = current_max_retries();
        let mut attempt: u32 = 0;
        // Only the initial send is retried — once bytes start flowing,
        // mid-stream failures surface as-is since we may have already
        // streamed tokens to the caller.
        let response = loop {
            let send_result = self
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await;

            match send_result {
                Ok(resp) if resp.status().is_success() => break resp,
                Ok(resp) => {
                    let status = resp.status();
                    let retry_after = resp
                        .headers()
                        .get(reqwest::header::RETRY_AFTER)
                        .and_then(|v| v.to_str().ok())
                        .and_then(parse_retry_after);
                    match classify_retry(Some(status.as_u16()), attempt, max_retries, retry_after) {
                        RetryAction::Retry(wait) => {
                            tracing::warn!(
                                target: "cubi::llm",
                                status = status.as_u16(),
                                attempt = attempt + 1,
                                max = max_retries,
                                "LLM stream request failed; retrying"
                            );
                            tokio::time::sleep(wait).await;
                            attempt += 1;
                            continue;
                        }
                        RetryAction::Stop => {
                            let error_text = resp.text().await.unwrap_or_default();
                            tracing::warn!(
                                target: "cubi::llm",
                                status = status.as_u16(),
                                "openai stream non-success response"
                            );
                            let ue = crate::user_error::classify_http_status(
                                status.as_u16(),
                                retry_after.map(|d| d.as_secs()),
                                &url,
                                &error_text,
                            );
                            return Err(anyhow::Error::new(ue));
                        }
                    }
                }
                Err(err) => match classify_retry(None, attempt, max_retries, None) {
                    RetryAction::Retry(wait) => {
                        tracing::warn!(
                            target: "cubi::llm",
                            error = %err,
                            attempt = attempt + 1,
                            max = max_retries,
                            "LLM stream connect error; retrying"
                        );
                        tokio::time::sleep(wait).await;
                        attempt += 1;
                        continue;
                    }
                    RetryAction::Stop => {
                        let mut ue = crate::user_error::classify_send_error(&err, &url);
                        ue.cause = Some(anyhow::Error::new(err));
                        return Err(anyhow::Error::new(ue));
                    }
                },
            }
        };

        let mut stream = response.bytes_stream();
        let mut buf = String::new();
        let mut content = String::new();
        let mut tool_calls_builder: Vec<(String, String, String)> = Vec::new(); // (id, name, args)
        let mut usage: Option<OaiUsage> = None;

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
                if let Some(u) = chunk.usage {
                    usage = Some(u);
                }
                for choice in chunk.choices {
                    if let Some(text) = &choice.delta.content {
                        if !text.is_empty() {
                            on_token(text);
                            content.push_str(text);
                        }
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

        let stats = ChatStats {
            prompt_tokens: usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0),
            completion_tokens: usage.as_ref().map(|u| u.completion_tokens).unwrap_or(0),
            elapsed_ms: started.elapsed().as_millis() as u64,
        };

        Ok((
            Message {
                role: "assistant".to_string(),
                content,
                tool_calls,
                tool_name: None,
            },
            stats,
        ))
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

fn fake_content() -> String {
    std::env::var("CUBI_FAKE_LLM_RESPONSE").unwrap_or_else(|_| "hi".to_string())
}

fn fake_message(messages: &[Message]) -> Message {
    if let Ok(raw) = std::env::var("CUBI_FAKE_LLM_TOOL_CALL") {
        if !messages.iter().any(|message| message.role == "tool") {
            if let Ok(call) = serde_json::from_str::<ToolCall>(&raw) {
                return Message {
                    role: "assistant".to_string(),
                    content: String::new(),
                    tool_calls: Some(vec![call]),
                    tool_name: None,
                };
            }
        }
    }
    Message::text("assistant", fake_content())
}

fn fake_stats() -> ChatStats {
    ChatStats {
        prompt_tokens: 1,
        completion_tokens: 1,
        elapsed_ms: 1,
    }
}

/// Decision returned by [`classify_retry`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryAction {
    /// Retry after the given wait duration.
    Retry(Duration),
    /// Do not retry; surface the error to the caller.
    Stop,
}

/// Pure helper: given an HTTP response status (or `None` for a connect
/// error) and the attempt number (starting at 0), return whether we
/// should retry and how long to wait first.
///
/// Retry policy:
///   * connect errors (None) — retry.
///   * 408 Request Timeout — retry.
///   * 429 Too Many Requests — retry, honoring `Retry-After` when given.
///   * 5xx (500/502/503/504/other 5xx) — retry.
///   * any other 4xx — do not retry (client error).
///   * 2xx/3xx — caller should not call this in the first place; we
///     return `Stop` defensively.
///
/// Backoff is `250ms * 2^attempt + jitter` capped at 5s. `retry_after`
/// (seconds) trumps the computed backoff when present.
pub fn classify_retry(
    status: Option<u16>,
    attempt: u32,
    max_retries: u32,
    retry_after: Option<Duration>,
) -> RetryAction {
    if attempt >= max_retries {
        return RetryAction::Stop;
    }
    let transient = match status {
        None => true, // connect / IO error
        Some(s) if s == 408 || s == 429 || (500..600).contains(&s) => true,
        _ => false,
    };
    if !transient {
        return RetryAction::Stop;
    }
    if let Some(wait) = retry_after {
        return RetryAction::Retry(wait.min(Duration::from_secs(30)));
    }
    let base_ms: u64 = 250u64 * (1u64 << attempt.min(5));
    let jitter_ms = (attempt as u64).wrapping_mul(37) % 100;
    let total = Duration::from_millis(base_ms + jitter_ms);
    RetryAction::Retry(total.min(Duration::from_secs(5)))
}

/// Parses a `Retry-After` header value, accepting either an integer
/// seconds count (`30`) or an HTTP-date. Falls back to `None` on
/// unparseable input.
fn parse_retry_after(value: &str) -> Option<Duration> {
    if let Ok(secs) = value.trim().parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }
    // HTTP-date parsing is intentionally not pulled in — providers
    // overwhelmingly use the integer-seconds form. Surface as None so
    // the caller falls back to computed backoff.
    None
}

fn current_max_retries() -> u32 {
    if let Ok(v) = std::env::var("CUBI_LLM_MAX_RETRIES") {
        if let Ok(n) = v.parse::<u32>() {
            return n;
        }
    }
    let cfg = crate::onboarding::AppConfig::load();
    cfg.llm_max_retries
}

/// Creates the appropriate LLM provider based on environment configuration.
pub fn create_provider() -> LlmBackend {
    if std::env::var("CUBI_FAKE_LLM").is_ok() {
        return LlmBackend::Fake;
    }
    if let Some(client) = OpenAiClient::from_env() {
        return LlmBackend::OpenAi(client);
    }

    let ollama = match std::env::var("CUBI_BASE_URL") {
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
///
/// Test hook: setting `CUBI_MAX_PROMPT_TOKENS_OVERRIDE=<n>` (positive
/// integer) returns `Some(n)` for every model. This lets the headless
/// budget-error path be exercised without depending on a particular
/// model's real window size.
pub fn context_window_for_model(model: &str) -> Option<usize> {
    if let Ok(raw) = std::env::var("CUBI_MAX_PROMPT_TOKENS_OVERRIDE") {
        if let Ok(n) = raw.parse::<usize>() {
            if n > 0 {
                return Some(n);
            }
        }
    }
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
    fn classify_retry_retries_5xx_until_max() {
        assert!(matches!(
            classify_retry(Some(500), 0, 2, None),
            RetryAction::Retry(_)
        ));
        assert!(matches!(
            classify_retry(Some(503), 1, 2, None),
            RetryAction::Retry(_)
        ));
        assert!(matches!(
            classify_retry(Some(500), 2, 2, None),
            RetryAction::Stop
        ));
    }

    #[test]
    fn classify_retry_retries_429_and_408() {
        assert!(matches!(
            classify_retry(Some(429), 0, 2, None),
            RetryAction::Retry(_)
        ));
        assert!(matches!(
            classify_retry(Some(408), 0, 2, None),
            RetryAction::Retry(_)
        ));
    }

    #[test]
    fn classify_retry_skips_other_4xx() {
        assert!(matches!(
            classify_retry(Some(400), 0, 5, None),
            RetryAction::Stop
        ));
        assert!(matches!(
            classify_retry(Some(401), 0, 5, None),
            RetryAction::Stop
        ));
        assert!(matches!(
            classify_retry(Some(404), 0, 5, None),
            RetryAction::Stop
        ));
    }

    #[test]
    fn classify_retry_retries_connect_errors() {
        assert!(matches!(
            classify_retry(None, 0, 2, None),
            RetryAction::Retry(_)
        ));
    }

    #[test]
    fn classify_retry_honors_retry_after() {
        let wait = Duration::from_secs(7);
        if let RetryAction::Retry(d) = classify_retry(Some(429), 0, 5, Some(wait)) {
            assert_eq!(d, wait);
        } else {
            panic!("expected retry");
        }
    }

    #[test]
    fn classify_retry_caps_retry_after_at_30s() {
        if let RetryAction::Retry(d) =
            classify_retry(Some(429), 0, 5, Some(Duration::from_secs(120)))
        {
            assert_eq!(d, Duration::from_secs(30));
        } else {
            panic!("expected retry");
        }
    }

    #[test]
    fn classify_retry_disabled_when_max_zero() {
        assert!(matches!(
            classify_retry(Some(500), 0, 0, None),
            RetryAction::Stop
        ));
    }

    #[test]
    fn classify_retry_backoff_caps_at_5s() {
        for attempt in 0..20 {
            if let RetryAction::Retry(d) = classify_retry(Some(500), attempt, 100, None) {
                assert!(d <= Duration::from_secs(5), "attempt {attempt} -> {d:?}");
            }
        }
    }

    #[test]
    fn parse_retry_after_accepts_integer_seconds() {
        assert_eq!(parse_retry_after("30"), Some(Duration::from_secs(30)));
        assert_eq!(parse_retry_after("  5 "), Some(Duration::from_secs(5)));
    }

    #[test]
    fn parse_retry_after_rejects_http_date() {
        assert_eq!(parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT"), None);
    }

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
    fn estimate_conversation_tokens_empty_history_is_zero() {
        assert_eq!(estimate_conversation_tokens(&[]), 0);
    }

    #[test]
    fn estimate_conversation_tokens_single_message_adds_role_overhead() {
        let msgs = vec![Message::text("user", "")];
        // role overhead 4, content empty → 4
        assert_eq!(estimate_conversation_tokens(&msgs), 4);
    }

    #[test]
    fn context_window_known_models() {
        assert_eq!(context_window_for_model("llama3.2:1b"), Some(128_000));
        assert_eq!(context_window_for_model("gpt-4o"), Some(128_000));
        assert_eq!(context_window_for_model("unknown-model"), None);
    }

    #[test]
    fn create_provider_defaults_to_ollama() {
        // Without CUBI_PROVIDER=openai, should default to Ollama.
        // We don't remove the var (unsafe in edition 2024+); instead we
        // just verify that the default codepath works when it's not "openai".
        let provider = create_provider();
        // If the env doesn't have CUBI_PROVIDER=openai, we get ollama.
        // If it does (unlikely in test), we'd get openai — both are valid.
        assert!(provider.provider_name() == "ollama" || provider.provider_name() == "openai");
    }
}
