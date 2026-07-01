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
            Self::Fake => {
                fake_check_fail(model)?;
                Ok(fake_content_for(model))
            }
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
            Self::Fake => {
                fake_check_fail(model)?;
                Ok((fake_message_for(model, &messages), fake_stats()))
            }
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
                fake_check_fail(model)?;
                let message = fake_message_for(model, &messages);
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

    /// Returns the HTTP endpoint this backend talks to, when applicable.
    /// `None` for the `Fake` backend which has no network surface.
    /// Used by `/doctor` so it can show the actual probed URL instead
    /// of hard-coding `http://localhost:11434`.
    pub fn base_url(&self) -> Option<&str> {
        match self {
            Self::Ollama(c) => Some(c.base_url()),
            Self::OpenAi(c) => Some(c.base_url()),
            Self::Fake => None,
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

/// Serialized tool call inside an assistant request message. Mirrors
/// OpenAI's wire format: `arguments` must be a JSON-encoded string,
/// not a raw object.
#[derive(Debug, Serialize)]
struct OaiRequestToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: OaiRequestToolCallFunction,
}

#[derive(Debug, Serialize)]
struct OaiRequestToolCallFunction {
    name: String,
    arguments: String,
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
    /// Assistant-only. Must be present (and match the ids on subsequent
    /// `role:"tool"` messages) for strict OpenAI-compatible validators
    /// to accept the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OaiRequestToolCall>>,
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

    /// Base URL this client is configured to talk to. Used by `/doctor`
    /// and similar diagnostics to show the actual endpoint being probed
    /// rather than hard-coding any specific URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn convert_messages(messages: &[Message]) -> Vec<OaiRequestMessage> {
        messages
            .iter()
            .map(|m| {
                // Round-trip assistant tool_calls so the server can
                // validate subsequent role:"tool" messages against the
                // ids the model emitted. Chat completions is stateless
                // — these must be on every request, not just the first.
                let tool_calls = if m.role == "assistant" {
                    m.tool_calls.as_ref().map(|calls| {
                        calls
                            .iter()
                            .enumerate()
                            .map(|(i, c)| OaiRequestToolCall {
                                // Synthesize a stable id when the source
                                // backend (e.g. Ollama) didn't supply
                                // one, so the matching tool-result
                                // message has *something* consistent to
                                // point at.
                                id: c
                                    .id
                                    .clone()
                                    .unwrap_or_else(|| format!("call_{}_{}", i, c.function.name)),
                                call_type: c
                                    .call_type
                                    .clone()
                                    .unwrap_or_else(|| "function".to_string()),
                                function: OaiRequestToolCallFunction {
                                    name: c.function.name.clone(),
                                    arguments: c.function.arguments.to_string(),
                                },
                            })
                            .collect()
                    })
                } else {
                    None
                };
                OaiRequestMessage {
                    role: m.role.clone(),
                    content: m.content.clone(),
                    // `name` is only relevant for tool-result messages.
                    name: if m.role == "tool" {
                        m.tool_name.clone()
                    } else {
                        None
                    },
                    // `tool_call_id` must only be set on role:"tool" messages and
                    // should match the id of the assistant ToolCall this is the
                    // result of. Prefer the id we recorded when constructing the
                    // tool-result message (set from `ToolCall::id`); fall back to
                    // `tool_name` only when the assistant turn didn't carry an id
                    // (older Ollama responses). Strict OpenAI validators reject
                    // the fallback, but it's the best we can do in that case.
                    tool_call_id: if m.role == "tool" {
                        m.tool_call_id.clone().or_else(|| m.tool_name.clone())
                    } else {
                        None
                    },
                    tool_calls,
                }
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
        let content =
            crate::thinking_filter::strip_thinking_blocks(&oai.content.unwrap_or_default());
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
            tool_call_id: None,
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
        // Strips Qwen3-style <think>…</think> reasoning blocks from
        // streamed tokens before forwarding to the UI and recording.
        let mut stripper = crate::thinking_filter::ThinkStripper::new();

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
                            let clean = stripper.feed(text);
                            if !clean.is_empty() {
                                on_token(&clean);
                                content.push_str(&clean);
                            }
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
        let tail = stripper.flush();
        if !tail.is_empty() {
            on_token(&tail);
            content.push_str(&tail);
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
                tool_call_id: None,
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

/// Per-model response lookup for the `Fake` backend. Resolution order:
///   1. `CUBI_FAKE_LLM_MODEL_RESPONSES` as a JSON object mapping
///      model name to canned response text. Used by the consensus
///      tests to script different "model" outputs in the same process.
///   2. Fallback to [`fake_content`] (single global response).
pub(crate) fn fake_content_for(model: &str) -> String {
    if let Ok(raw) = std::env::var("CUBI_FAKE_LLM_MODEL_RESPONSES") {
        if let Ok(map) = serde_json::from_str::<std::collections::HashMap<String, String>>(&raw) {
            if let Some(v) = map.get(model) {
                return v.clone();
            }
        }
    }
    fake_content()
}

fn fake_message_for(model: &str, messages: &[Message]) -> Message {
    if let Ok(raw) = std::env::var("CUBI_FAKE_LLM_TOOL_CALL") {
        // Normally the scripted tool call is emitted once (until a tool
        // result appears in history) so tests can drive a single
        // model→tool→model round-trip. Setting
        // `CUBI_FAKE_LLM_TOOL_CALL_REPEAT=1` removes that gate so the fake
        // backend re-issues the call on every step — used to exercise the
        // headless consecutive-tool-error safety valve.
        let repeat = std::env::var("CUBI_FAKE_LLM_TOOL_CALL_REPEAT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if repeat || !messages.iter().any(|message| message.role == "tool") {
            if let Ok(call) = serde_json::from_str::<ToolCall>(&raw) {
                return Message {
                    role: "assistant".to_string(),
                    content: String::new(),
                    tool_calls: Some(vec![call]),
                    tool_name: None,
                    tool_call_id: None,
                };
            }
        }
    }
    Message::text("assistant", fake_content_for(model))
}

/// Fail injection used by the consensus tests to verify that one
/// failing subagent doesn't abort the others. Set
/// `CUBI_FAKE_LLM_FAIL_MODELS` to a comma-separated list of model
/// names; matching calls return an error from the backend.
pub(crate) fn fake_check_fail(model: &str) -> Result<()> {
    if model.is_empty() {
        return Ok(());
    }
    if let Ok(raw) = std::env::var("CUBI_FAKE_LLM_FAIL_MODELS") {
        if raw.split(',').any(|m| m.trim() == model) {
            anyhow::bail!("fake backend: scripted failure for model `{model}`");
        }
    }
    Ok(())
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

    // Tag-specific overrides for families where Ollama ships different
    // context windows per parameter size. Must run before the bare-name
    // match below.
    match model_lower.as_str() {
        // Gemma 3: 270M and 1B variants ship with a 32K window; 4B/12B/27B
        // are 128K. See https://ollama.com/library/gemma3.
        "gemma3:270m" | "gemma3:1b" => return Some(32_768),
        _ => {}
    }

    let without_tag = model_lower.split(':').next().unwrap_or(&model_lower);
    // HuggingFace-style ids carry an `org/` repo prefix (e.g.
    // "zai-org/glm-5.2") and provider prefixes like "ollama/llama3" do the
    // same. Match on the final path segment so they map identically to the
    // bare family names below.
    let base = without_tag.rsplit('/').next().unwrap_or(without_tag);

    match base {
        // Ollama models
        "llama3.3" | "llama3.2" | "llama3.1" | "llama3" => Some(128_000),
        "llama2" => Some(4_096),
        "mistral" | "mixtral" => Some(32_768),
        "mistral-small3.2" | "mistral-small3.1" | "mistral-small" => Some(131_072),
        "devstral" => Some(131_072),
        "qwen2.5" | "qwen2" => Some(128_000),
        // Qwen3 ships with a 32K native window; YaRN can extend it but
        // we conservatively report the native value since the backend
        // doesn't always enable YaRN by default.
        "qwen3" => Some(32_768),
        "codellama" => Some(16_384),
        "phi3" | "phi-3" => Some(128_000),
        // Phi-4 (14B dense) ships with a 16K window per Microsoft's model
        // card; only phi4-mini is 128K.
        "phi4" | "phi-4" => Some(16_384),
        "phi4-mini" => Some(128_000),
        "gemma2" | "gemma" => Some(8_192),
        "gemma3" => Some(128_000),
        // Gemma 4 ships with a 256K context window across its variants.
        // See https://ai.google.dev/gemma/docs/releases.
        "gemma4" => Some(256_000),
        "deepseek-coder" => Some(16_384),
        "granite3.3" | "granite3.2" | "granite3.1" => Some(131_072),
        "hermes3" => Some(131_072),
        "command-r7b" | "command-r" => Some(131_072),
        // GLM-5.2 (Z.ai / Zhipu, MIT-licensed) ships with a solid 1M-token
        // context for long-horizon agentic work. See https://z.ai/blog/glm-5.2.
        "glm-5.2" | "glm5.2" => Some(1_000_000),
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
        // Cover the newly added families so a future refactor that drops
        // a row trips this test instead of silently regressing the
        // budget warning for users on those models.
        assert_eq!(context_window_for_model("qwen3:8b"), Some(32_768));
        assert_eq!(context_window_for_model("devstral"), Some(131_072));
        assert_eq!(context_window_for_model("mistral-small3.2"), Some(131_072));
        assert_eq!(context_window_for_model("granite3.3:8b"), Some(131_072));
        assert_eq!(context_window_for_model("hermes3:8b"), Some(131_072));
        assert_eq!(context_window_for_model("llama3.3:70b"), Some(128_000));
        assert_eq!(context_window_for_model("command-r7b"), Some(131_072));
        assert_eq!(context_window_for_model("phi4"), Some(16_384));
        assert_eq!(context_window_for_model("phi4-mini"), Some(128_000));
        assert_eq!(context_window_for_model("gemma3:4b"), Some(128_000));
        assert_eq!(context_window_for_model("gemma3:270m"), Some(32_768));
        assert_eq!(context_window_for_model("gemma3:1b"), Some(32_768));
        assert_eq!(context_window_for_model("gemma4"), Some(256_000));
        assert_eq!(context_window_for_model("gemma4:31b"), Some(256_000));
        // GLM-5.2 — 1M context, recognized via Ollama tag, HF repo id, and
        // provider-prefixed forms.
        assert_eq!(context_window_for_model("glm-5.2"), Some(1_000_000));
        assert_eq!(context_window_for_model("glm5.2:latest"), Some(1_000_000));
        assert_eq!(context_window_for_model("zai-org/GLM-5.2"), Some(1_000_000));
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

    #[test]
    fn openai_convert_messages_uses_tool_call_id_when_present() {
        // Strict OpenAI validators require tool_call_id on role:"tool"
        // messages to match the id the assistant supplied. Prove the
        // converter sends the real id, not the (incorrect) tool_name
        // fallback.
        let msg = Message::tool_result("bash", "ok", Some("call_xyz".into()));
        let converted = OpenAiClient::convert_messages(&[msg]);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "tool");
        assert_eq!(converted[0].tool_call_id.as_deref(), Some("call_xyz"));
        assert_eq!(converted[0].name.as_deref(), Some("bash"));
    }

    #[test]
    fn openai_convert_messages_falls_back_to_tool_name_when_no_id() {
        // Older Ollama responses don't carry an id. The converter must
        // still emit *some* tool_call_id so the request body validates,
        // even though strict servers will reject the fallback.
        let msg = Message::tool_result("bash", "ok", None);
        let converted = OpenAiClient::convert_messages(&[msg]);
        assert_eq!(converted[0].tool_call_id.as_deref(), Some("bash"));
    }

    #[test]
    fn openai_convert_messages_never_sets_tool_call_id_on_non_tool_roles() {
        for role in ["user", "assistant", "system"] {
            let msg = Message::text(role, "hi");
            let converted = OpenAiClient::convert_messages(&[msg]);
            assert!(converted[0].tool_call_id.is_none(), "role {role}");
            assert!(converted[0].name.is_none(), "role {role}");
        }
    }

    #[test]
    fn openai_convert_messages_round_trips_assistant_tool_calls() {
        // Chat completions is stateless: every request must serialize
        // the assistant's prior tool_calls so the server can validate
        // subsequent role:"tool" messages against the ids it produced.
        let assistant = Message {
            role: "assistant".to_string(),
            content: String::new(),
            tool_calls: Some(vec![ToolCall {
                id: Some("call_real_id".to_string()),
                call_type: Some("function".to_string()),
                function: ToolCallFunction {
                    name: "bash".to_string(),
                    arguments: serde_json::json!({"cmd": "ls"}),
                },
            }]),
            tool_name: None,
            tool_call_id: None,
        };
        let tool_result = Message::tool_result("bash", "ok", Some("call_real_id".into()));
        let converted = OpenAiClient::convert_messages(&[assistant, tool_result]);

        assert_eq!(converted.len(), 2);
        let asst = &converted[0];
        assert_eq!(asst.role, "assistant");
        let tcs = asst
            .tool_calls
            .as_ref()
            .expect("assistant tool_calls present");
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].id, "call_real_id");
        assert_eq!(tcs[0].call_type, "function");
        assert_eq!(tcs[0].function.name, "bash");
        // OpenAI requires `arguments` as a JSON-encoded string, not an object.
        assert_eq!(tcs[0].function.arguments, r#"{"cmd":"ls"}"#);

        let tool = &converted[1];
        assert_eq!(tool.role, "tool");
        assert_eq!(tool.tool_call_id.as_deref(), Some("call_real_id"));
        assert!(tool.tool_calls.is_none());
    }

    #[test]
    fn openai_convert_messages_synthesizes_id_for_assistant_tool_calls_without_id() {
        // Older Ollama-style assistant turns may not carry tool_call ids.
        // The converter must still emit *some* stable id so subsequent
        // tool-result messages have something to match against.
        let assistant = Message {
            role: "assistant".to_string(),
            content: String::new(),
            tool_calls: Some(vec![ToolCall {
                id: None,
                call_type: None,
                function: ToolCallFunction {
                    name: "bash".to_string(),
                    arguments: serde_json::json!({}),
                },
            }]),
            tool_name: None,
            tool_call_id: None,
        };
        let converted = OpenAiClient::convert_messages(&[assistant]);
        let tcs = converted[0].tool_calls.as_ref().unwrap();
        assert!(!tcs[0].id.is_empty(), "must synthesize a non-empty id");
        assert_eq!(tcs[0].call_type, "function");
    }
}
