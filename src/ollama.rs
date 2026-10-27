use crate::thinking_filter::{ThinkStripper, strip_thinking_blocks};
use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::time::{Duration, sleep};

const DEFAULT_OLLAMA_NUM_CTX_CAP: usize = 32_768;

/// Normalize assistant content before handing it back to the host.
/// Some Ollama versions expose a separate `thinking` field that already
/// strips `<think>...</think>` server-side; in that case this is a no-op.
pub(crate) fn normalize_assistant_message(
    mut m: Message,
    advertised_tools: Option<&[ToolSpec]>,
) -> Message {
    m.content = strip_thinking_blocks(&m.content);
    synthesize_advertised_tool_call_from_content(&mut m, advertised_tools);
    m
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub stream: bool,
    /// Optional native tool list. When `Some`, Ollama returns `tool_calls`
    /// on the assistant message instead of (or in addition to) `content`.
    /// Only supported on tool-capable models (llama3.1+, qwen2.5, etc.);
    /// older models will silently ignore the field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolSpec>>,
    /// Runtime options for Ollama's `/api/chat` endpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<ChatOptions>,
    /// Native reasoning control. Omitted unless `CUBI_THINK` is set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think: Option<bool>,
    /// Native model residency control. Omitted unless `CUBI_KEEP_ALIVE` is set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u64>,
}

/// Native tool definition forwarded to Ollama. Mirrors the OpenAI-compatible
/// shape Ollama accepts on `/api/chat`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    #[serde(rename = "type")]
    pub tool_type: String, // always "function"
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    /// Tool calls the model wants the host to execute. Only present on
    /// assistant turns when native tool-calling is in use.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// When this message is a tool result (`role:"tool"`), Ollama wants the
    /// name of the tool so it can correlate it back to the call. We always
    /// pass it for tool messages and omit it everywhere else.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_name: Option<String>,
    /// Echoes the `id` of the assistant `ToolCall` this message is the
    /// result of. OpenAI-compatible servers (strict ones especially)
    /// require `tool_call_id` on every `role:"tool"` message and reject
    /// anything that doesn't match a prior call's id. Optional because
    /// the Ollama path correlates on `tool_name` and may not have an id
    /// at all.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_call_id: Option<String>,
}

impl Message {
    /// Convenience constructor for user / assistant / system messages that
    /// carry only text content (the overwhelming majority).
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            tool_calls: None,
            tool_name: None,
            tool_call_id: None,
        }
    }

    /// Constructor for a tool-result message produced by the host after
    /// executing a model-requested tool call. Pass `tool_call_id` when
    /// the assistant's `ToolCall` carried an `id` (OpenAI-compatible
    /// backends require this for correlation); pass `None` only when the
    /// backend didn't supply one (older Ollama responses).
    pub fn tool_result(
        tool_name: impl Into<String>,
        content: impl Into<String>,
        tool_call_id: Option<String>,
    ) -> Self {
        Self {
            role: "tool".to_string(),
            content: content.into(),
            tool_calls: None,
            tool_name: Some(tool_name.into()),
            tool_call_id,
        }
    }
}

/// A single tool invocation the model wants the host to perform.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Ollama doesn't always set an id; tolerate its absence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// Ollama returns arguments as a JSON object; older clients sometimes
    /// stringify it. Accept both via `serde_json::Value`.
    pub arguments: serde_json::Value,
}

fn chat_options_for_model(model: &str) -> Result<Option<ChatOptions>> {
    let num_ctx = resolve_num_ctx(model, current_num_ctx_override());
    let num_predict = crate::llm::current_max_tokens();
    let sampling = crate::model_profile::sampling_for_model(model)?;
    Ok((num_ctx.is_some()
        || num_predict.is_some()
        || sampling.temperature.is_some()
        || sampling.top_p.is_some()
        || sampling.top_k.is_some())
    .then_some(ChatOptions {
        num_ctx,
        num_predict,
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        top_k: sampling.top_k,
    }))
}

fn current_num_ctx_override() -> Option<usize> {
    crate::config::with(|config| config.generation.num_ctx)
}

fn current_think() -> Option<bool> {
    crate::config::with(|config| config.generation.think)
}

fn current_keep_alive() -> Option<String> {
    crate::config::with(|config| config.generation.keep_alive.clone())
}

fn resolve_num_ctx(model: &str, env_override: Option<usize>) -> Option<usize> {
    let env_override = env_override.filter(|value| *value > 0);
    let cap = env_override.unwrap_or(DEFAULT_OLLAMA_NUM_CTX_CAP);
    match crate::llm::context_window_for_model(model) {
        Some(model_window) => Some(model_window.min(cap)),
        None => env_override,
    }
}

#[cfg(test)]
fn synthesize_tool_call_from_content(message: &mut Message) -> bool {
    if message.role != "assistant"
        || message
            .tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty())
    {
        return false;
    }

    let Some(tool_call) = parse_tool_call_content(&message.content) else {
        return false;
    };

    message.content.clear();
    message.tool_calls = Some(vec![tool_call]);
    true
}

fn synthesize_advertised_tool_call_from_content(
    message: &mut Message,
    advertised_tools: Option<&[ToolSpec]>,
) -> bool {
    if message.role != "assistant"
        || message
            .tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty())
    {
        return false;
    }

    let Some(tool_call) = parse_tool_call_content(&message.content) else {
        return false;
    };
    let is_advertised = advertised_tools.is_some_and(|tools| {
        tools
            .iter()
            .any(|tool| tool.function.name == tool_call.function.name)
    });
    if !is_advertised {
        return false;
    }

    message.content.clear();
    message.tool_calls = Some(vec![tool_call]);
    true
}

fn parse_tool_call_content(content: &str) -> Option<ToolCall> {
    let json = extract_tool_call_json(content)?.trim();
    if !(json.starts_with('{') && json.ends_with('}')) {
        return None;
    }

    let value: serde_json::Value = serde_json::from_str(json).ok()?;
    let object = value.as_object()?;

    let (name, arguments) = if let Some(function) = object.get("function") {
        let function = function.as_object()?;
        (
            non_empty_string(function.get("name")?)?,
            parse_tool_arguments(function.get("arguments")?)?,
        )
    } else {
        (
            non_empty_string(object.get("name")?)?,
            parse_tool_arguments(object.get("arguments")?)?,
        )
    };

    Some(ToolCall {
        id: object
            .get("id")
            .and_then(|value| value.as_str())
            .map(ToOwned::to_owned),
        call_type: Some(
            object
                .get("type")
                .and_then(|value| value.as_str())
                .filter(|value| !value.trim().is_empty())
                .unwrap_or("function")
                .to_string(),
        ),
        function: ToolCallFunction { name, arguments },
    })
}

fn extract_tool_call_json(content: &str) -> Option<&str> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return None;
    }
    if !trimmed.starts_with("```") {
        return Some(trimmed);
    }

    let without_closing = trimmed.strip_suffix("```")?;
    let (opening, body) = without_closing.split_once('\n')?;
    if opening.trim_end_matches('\r').trim() != "```json" || body.contains("```") {
        return None;
    }
    Some(body.trim())
}

fn non_empty_string(value: &serde_json::Value) -> Option<String> {
    let value = value.as_str()?.trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn parse_tool_arguments(value: &serde_json::Value) -> Option<serde_json::Value> {
    match value {
        serde_json::Value::Object(_) => Some(value.clone()),
        serde_json::Value::String(raw) => {
            let parsed: serde_json::Value = serde_json::from_str(raw).ok()?;
            parsed.is_object().then_some(parsed)
        }
        _ => None,
    }
}

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub message: Message,
    // Note: 'done' field exists in API but we don't need it for non-streaming
    #[allow(dead_code)]
    pub done: bool,
    #[serde(default)]
    pub done_reason: Option<String>,
    /// Number of tokens in the prompt (Ollama only sends this on the final
    /// chunk, or on the single non-streaming response).
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
    /// Number of tokens generated.
    #[serde(default)]
    pub eval_count: Option<u64>,
    /// End-to-end duration of the request, in nanoseconds. Includes load,
    /// prompt eval, and generation.
    #[serde(default)]
    pub total_duration: Option<u64>,
    /// Time spent loading the model, in nanoseconds.
    #[serde(default)]
    pub load_duration: Option<u64>,
    /// Time spent evaluating the prompt, in nanoseconds.
    #[serde(default)]
    pub prompt_eval_duration: Option<u64>,
    /// Time spent generating the completion, in nanoseconds.
    #[serde(default)]
    pub eval_duration: Option<u64>,
}

/// One streaming chunk from `/api/chat` when `stream: true`. Each NDJSON
/// line decodes to one of these. The final chunk has `done: true` and a
/// fully-populated `message` (token deltas are in `message.content` on
/// intermediate chunks).
#[derive(Debug, Deserialize)]
pub struct ChatStreamChunk {
    pub message: Message,
    pub done: bool,
    #[serde(default)]
    pub done_reason: Option<String>,
    /// Token counts. Present only on the final chunk; intermediate chunks
    /// have these as `None`.
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
    #[serde(default)]
    pub eval_count: Option<u64>,
    /// End-to-end duration in nanoseconds. Final chunk only.
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub load_duration: Option<u64>,
    #[serde(default)]
    pub prompt_eval_duration: Option<u64>,
    #[serde(default)]
    pub eval_duration: Option<u64>,
}

struct StreamToolCall {
    index: Option<usize>,
    call: ToolCall,
}

impl StreamToolCall {
    fn has_same_identity(&self, other: &Self) -> bool {
        if self.index.is_some()
            || self.call.id.is_some()
            || other.index.is_some()
            || other.call.id.is_some()
        {
            self.index == other.index && self.call.id == other.call.id
        } else {
            self.call == other.call
        }
    }
}

fn stream_tool_call_indices(line: &str) -> Vec<Option<usize>> {
    serde_json::from_str::<serde_json::Value>(line)
        .ok()
        .and_then(|value| {
            value
                .get("message")?
                .get("tool_calls")?
                .as_array()
                .map(|calls| {
                    calls
                        .iter()
                        .map(|call| {
                            call.get("index")
                                .or_else(|| call.get("function")?.get("index"))
                                .and_then(serde_json::Value::as_u64)
                                .and_then(|index| usize::try_from(index).ok())
                        })
                        .collect()
                })
        })
        .unwrap_or_default()
}

/// Per-call inference statistics. Populated from whatever the provider
/// returned (Ollama: native counts/durations; OpenAI: `usage`). Existing
/// token and elapsed fields remain additive counters. Provider-specific
/// values stay `None` when they were not reported.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ChatStats {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub elapsed_ms: u64,
    pub time_to_first_token_ms: Option<u64>,
    pub load_duration_ms: Option<u64>,
    pub prompt_eval_duration_ms: Option<u64>,
    pub eval_duration_ms: Option<u64>,
    pub cached_prompt_tokens: Option<u64>,
    pub request_count: u64,
    pub wall_clock_ms: Option<u64>,
    pub last_prompt_tokens: Option<u64>,
}

impl ChatStats {
    #[allow(dead_code)]
    pub fn total_tokens(&self) -> u64 {
        self.prompt_tokens + self.completion_tokens
    }

    /// Accumulates `other` into `self`. Used for session-level totals.
    pub fn add(&mut self, other: &Self) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(other.prompt_tokens);
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(other.completion_tokens);
        self.elapsed_ms = self.elapsed_ms.saturating_add(other.elapsed_ms);
        add_optional(
            &mut self.time_to_first_token_ms,
            other.time_to_first_token_ms,
        );
        add_optional(&mut self.load_duration_ms, other.load_duration_ms);
        add_optional(
            &mut self.prompt_eval_duration_ms,
            other.prompt_eval_duration_ms,
        );
        add_optional(&mut self.eval_duration_ms, other.eval_duration_ms);
        add_optional(&mut self.cached_prompt_tokens, other.cached_prompt_tokens);
        self.request_count = self.request_count.saturating_add(other.request_count);
        add_optional(&mut self.wall_clock_ms, other.wall_clock_ms);
        if other.last_prompt_tokens.is_some() {
            self.last_prompt_tokens = other.last_prompt_tokens;
        }
    }

    /// Records a request that ended before provider statistics were available.
    pub fn add_unreported_request(&mut self) {
        self.request_count = self.request_count.saturating_add(1);
    }

    /// True when no field was populated. Used to decide whether to print a
    /// footer (skip if the provider returned nothing useful).
    pub fn is_empty(&self) -> bool {
        self.prompt_tokens == 0
            && self.completion_tokens == 0
            && self.elapsed_ms == 0
            && self.time_to_first_token_ms.is_none()
            && self.load_duration_ms.is_none()
            && self.prompt_eval_duration_ms.is_none()
            && self.eval_duration_ms.is_none()
            && self.cached_prompt_tokens.is_none()
            && self.request_count == 0
            && self.wall_clock_ms.is_none()
    }
}

fn add_optional(total: &mut Option<u64>, value: Option<u64>) {
    if let Some(value) = value {
        *total = Some(total.unwrap_or(0).saturating_add(value));
    }
}

fn nanos_to_millis(value: Option<u64>) -> Option<u64> {
    value.map(|nanoseconds| nanoseconds / 1_000_000)
}

pub struct OllamaClient {
    base_url: String,
    client: reqwest::Client,
}

async fn with_retry<F, Fut, T>(max_attempts: u32, mut f: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let mut attempt = 0;
    loop {
        attempt += 1;
        match f().await {
            Ok(value) => return Ok(value),
            Err(err) if attempt < max_attempts && is_network_error(&err) => {
                sleep(Duration::from_secs(1_u64 << (attempt - 1))).await;
            }
            Err(err) => return Err(err),
        }
    }
}

fn is_network_error(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        cause
            .downcast_ref::<reqwest::Error>()
            .map(|e| e.is_connect() || e.is_timeout() || e.is_request() || e.is_body())
            .unwrap_or(false)
    })
}

impl OllamaClient {
    pub fn new() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            client: crate::llm::build_http_client(),
        }
    }

    /// Creates an OllamaClient with a custom base URL.
    pub fn with_base_url(base_url: String) -> Self {
        Self {
            base_url,
            client: crate::llm::build_http_client(),
        }
    }

    /// Base URL this client is configured to talk to. Used by `/doctor`
    /// and similar diagnostics so they can show the actual probed
    /// endpoint instead of hard-coding `http://localhost:11434`.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub async fn chat(&self, model: &str, messages: Vec<Message>) -> Result<String> {
        let (msg, _) = self.chat_with_tools(model, messages, None).await?;
        Ok(msg.content)
    }

    /// Non-streaming chat that optionally forwards a native tool list. Returns
    /// the full assistant [`Message`] plus per-call [`ChatStats`] so the
    /// caller can inspect `tool_calls` and run an agent loop on top.
    pub async fn chat_with_tools(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolSpec>>,
    ) -> Result<(Message, ChatStats)> {
        let request = ChatRequest {
            model: model.to_string(),
            messages,
            stream: false,
            tools,
            options: chat_options_for_model(model)?,
            think: current_think(),
            keep_alive: current_keep_alive(),
        };
        with_retry(3, || {
            let request = request.clone();
            async move {
                tracing::debug!(target: "cubi::llm::ollama", model = %model, "ollama chat request");
                let response = self
                    .client
                    .post(format!("{}/api/chat", self.base_url))
                    .json(&request)
                    .send()
                    .await
                    .context("Failed to send request to Ollama")?;

                if !response.status().is_success() {
                    let status = response.status();
                    let error_text = response.text().await.unwrap_or_default();
                    tracing::warn!(target: "cubi::llm::ollama", status = status.as_u16(), "ollama non-success");
                    anyhow::bail!("Ollama API error: {}", error_text);
                }

                let chat_response: ChatResponse = response
                    .json()
                    .await
                    .context("Failed to parse Ollama response")?;
                if chat_response.done_reason.as_deref() == Some("length") {
                    anyhow::bail!(
                        "Ollama response was truncated by the output-token limit; partial content: {}",
                        chat_response.message.content
                    );
                }

                let stats = ChatStats {
                    prompt_tokens: chat_response.prompt_eval_count.unwrap_or(0),
                    completion_tokens: chat_response.eval_count.unwrap_or(0),
                    elapsed_ms: nanos_to_millis(chat_response.total_duration).unwrap_or(0),
                    load_duration_ms: nanos_to_millis(chat_response.load_duration),
                    prompt_eval_duration_ms: nanos_to_millis(
                        chat_response.prompt_eval_duration,
                    ),
                    eval_duration_ms: nanos_to_millis(chat_response.eval_duration),
                    request_count: 1,
                    last_prompt_tokens: chat_response.prompt_eval_count,
                    ..ChatStats::default()
                };

                Ok((
                    normalize_assistant_message(chat_response.message, request.tools.as_deref()),
                    stats,
                ))
            }
        })
        .await
    }

    /// Streaming chat. `on_token` is invoked with each text fragment as it
    /// arrives (typically a few characters at a time). The fully-assembled
    /// final assistant [`Message`] is returned when the stream completes, so
    /// the caller can both stream tokens *and* inspect any `tool_calls` the
    /// model returned on the final chunk.
    ///
    /// Note: when `tools` are provided, Ollama buffers the message until the
    /// model decides whether to call a tool — most tool-capable models emit
    /// the full content (or `tool_calls`) on the final chunk rather than
    /// token-by-token. We still call `on_token` for any intermediate text so
    /// the UX is consistent regardless of model behavior.
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
        let request = ChatRequest {
            model: model.to_string(),
            messages,
            stream: true,
            tools: tools.clone(),
            options: chat_options_for_model(model)?,
            think: current_think(),
            keep_alive: current_keep_alive(),
        };
        let started = std::time::Instant::now();
        let response = with_retry(3, || {
            let request = request.clone();
            async move {
                let response = self
                    .client
                    .post(format!("{}/api/chat", self.base_url))
                    .json(&request)
                    .send()
                    .await
                    .context("Failed to send streaming request to Ollama")?;

                if !response.status().is_success() {
                    let error_text = response.text().await.unwrap_or_default();
                    anyhow::bail!("Ollama API error: {}", error_text);
                }

                Ok(response)
            }
        })
        .await?;

        // NDJSON: each chunk arrives as bytes; we split on newlines and
        // parse one JSON object per non-empty line. `reqwest`'s
        // `bytes_stream` does not guarantee per-line framing, so we buffer.
        let mut stream = response.bytes_stream();
        let mut buf = Vec::new();
        let mut content = String::new();
        let mut final_msg: Option<Message> = None;
        let mut tool_calls: Vec<StreamToolCall> = Vec::new();
        let mut stats = ChatStats::default();
        let mut time_to_first_token_ms = None;
        let mut done_reason: Option<String> = None;
        // Strips Qwen3-style <think>…</think> reasoning blocks from
        // both the tokens we forward to the UI and the accumulated
        // content we record in history.
        let mut stripper = ThinkStripper::new();

        // Local closure so the trailing-buffer case after the read loop can
        // reuse the same parse-and-dispatch logic without duplication.
        let mut handle_line = |line: &[u8],
                               content: &mut String,
                               final_msg: &mut Option<Message>,
                               tool_calls: &mut Vec<StreamToolCall>,
                               stats: &mut ChatStats,
                               time_to_first_token_ms: &mut Option<u64>,
                               done_reason: &mut Option<String>,
                               stripper: &mut ThinkStripper|
         -> Result<()> {
            let line = std::str::from_utf8(line).context("Ollama stream returned invalid UTF-8")?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                return Ok(());
            }
            let parsed: ChatStreamChunk = match serde_json::from_str(trimmed) {
                Ok(c) => c,
                Err(_) => return Ok(()), // tolerate a malformed mid-stream line
            };
            let tool_call_indices = stream_tool_call_indices(trimmed);
            let mut message = parsed.message;
            if !message.content.is_empty() {
                let clean = stripper.feed(&message.content);
                if !clean.is_empty() {
                    time_to_first_token_ms
                        .get_or_insert_with(|| started.elapsed().as_millis() as u64);
                    on_token(&clean);
                    content.push_str(&clean);
                }
            }
            if let Some(calls) = message.tool_calls.take() {
                for (position, call) in calls.into_iter().enumerate() {
                    let call = StreamToolCall {
                        index: tool_call_indices.get(position).copied().flatten(),
                        call,
                    };
                    if !tool_calls
                        .iter()
                        .any(|existing| existing.has_same_identity(&call))
                    {
                        tool_calls.push(call);
                    }
                }
            }
            if parsed.done {
                *done_reason = parsed.done_reason;
                // Capture provider stats from the terminal chunk.
                stats.prompt_tokens = parsed.prompt_eval_count.unwrap_or(0);
                stats.completion_tokens = parsed.eval_count.unwrap_or(0);
                stats.elapsed_ms = nanos_to_millis(parsed.total_duration).unwrap_or(0);
                stats.load_duration_ms = nanos_to_millis(parsed.load_duration);
                stats.prompt_eval_duration_ms = nanos_to_millis(parsed.prompt_eval_duration);
                stats.eval_duration_ms = nanos_to_millis(parsed.eval_duration);
                stats.request_count = 1;
                stats.last_prompt_tokens = parsed.prompt_eval_count;
                let tail = stripper.flush();
                if !tail.is_empty() {
                    on_token(&tail);
                    content.push_str(&tail);
                }
                // Replace fragmentary content with the accumulated text so
                // callers don't have to reconstruct it.
                message.content = content.clone();
                message.tool_calls = (!tool_calls.is_empty()).then(|| {
                    std::mem::take(tool_calls)
                        .into_iter()
                        .map(|tool_call| tool_call.call)
                        .collect()
                });
                *final_msg = Some(message);
            }
            Ok(())
        };

        while let Some(chunk) = stream.next().await {
            let bytes = chunk.context("Stream read failed")?;
            buf.extend_from_slice(&bytes);
            while let Some(nl) = buf.iter().position(|byte| *byte == b'\n') {
                let mut line: Vec<u8> = buf.drain(..=nl).collect();
                line.pop();
                handle_line(
                    &line,
                    &mut content,
                    &mut final_msg,
                    &mut tool_calls,
                    &mut stats,
                    &mut time_to_first_token_ms,
                    &mut done_reason,
                    &mut stripper,
                )?;
            }
        }

        // NDJSON isn't required to end with a trailing newline. If the
        // server sent the final `done:true` object without `\n`, it would
        // otherwise sit in `buf` forever and we'd report "ended without
        // done chunk". Flush whatever's left.
        if !buf.iter().all(|byte| byte.is_ascii_whitespace()) {
            let remaining = std::mem::take(&mut buf);
            handle_line(
                &remaining,
                &mut content,
                &mut final_msg,
                &mut tool_calls,
                &mut stats,
                &mut time_to_first_token_ms,
                &mut done_reason,
                &mut stripper,
            )?;
        }

        let msg = normalize_assistant_message(
            final_msg
                .ok_or_else(|| anyhow::anyhow!("Ollama stream ended without a `done` chunk"))?,
            tools.as_deref(),
        );
        if done_reason.as_deref() == Some("length") {
            anyhow::bail!(
                "Ollama response was truncated by the output-token limit; partial content: {}",
                msg.content
            );
        }
        stats.time_to_first_token_ms = time_to_first_token_ms;
        Ok((msg, stats))
    }

    pub async fn list_models(&self) -> Result<Vec<String>> {
        let response = self
            .client
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await
            .context("Failed to list models")?;

        let data: serde_json::Value = response.json().await?;

        let models = data["models"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m["name"].as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        Ok(models)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    struct EnvGuard(Vec<(&'static str, Option<OsString>)>);

    impl EnvGuard {
        fn set(values: &[(&'static str, Option<&str>)]) -> Self {
            let previous = values
                .iter()
                .map(|(name, value)| {
                    let previous = std::env::var_os(name);
                    // SAFETY: env-mutating tests are serialized by `crate::llm::ENV_LOCK`.
                    unsafe {
                        match value {
                            Some(value) => std::env::set_var(name, value),
                            None => std::env::remove_var(name),
                        }
                    }
                    (*name, previous)
                })
                .collect();
            Self(previous)
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (name, value) in self.0.drain(..) {
                // SAFETY: the guard is dropped while its test still holds the env lock.
                unsafe {
                    match value {
                        Some(value) => std::env::set_var(name, value),
                        None => std::env::remove_var(name),
                    }
                }
            }
        }
    }

    async fn serve_chunked(parts: Vec<Vec<u8>>) -> (String, tokio::task::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut header_end = None;
            let mut content_length = 0;
            loop {
                let mut chunk = [0_u8; 4096];
                let count = socket.read(&mut chunk).await.unwrap();
                if count == 0 {
                    break;
                }
                request.extend_from_slice(&chunk[..count]);
                if header_end.is_none()
                    && let Some(index) = request.windows(4).position(|window| window == b"\r\n\r\n")
                {
                    let end = index + 4;
                    let headers = String::from_utf8_lossy(&request[..end]);
                    content_length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().unwrap())
                        })
                        .unwrap_or(0);
                    header_end = Some(end);
                }
                if header_end.is_some_and(|end| request.len() >= end + content_length) {
                    break;
                }
            }

            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\n\
                      Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
                )
                .await
                .unwrap();
            for part in parts {
                socket
                    .write_all(format!("{:x}\r\n", part.len()).as_bytes())
                    .await
                    .unwrap();
                socket.write_all(&part).await.unwrap();
                socket.write_all(b"\r\n").await.unwrap();
                socket.flush().await.unwrap();
            }
            socket.write_all(b"0\r\n\r\n").await.unwrap();

            let body_start = header_end.unwrap();
            String::from_utf8(request[body_start..body_start + content_length].to_vec()).unwrap()
        });
        (format!("http://{}", address), handle)
    }

    fn bash_tool_call() -> serde_json::Value {
        serde_json::json!({
            "function": {
                "name": "bash",
                "arguments": {"command": "pwd"}
            }
        })
    }

    fn indexed_bash_tool_call(index: usize) -> serde_json::Value {
        serde_json::json!({
            "function": {
                "index": index,
                "name": "bash",
                "arguments": {"command": "pwd"}
            }
        })
    }

    #[tokio::test]
    async fn stream_preserves_tool_call_from_non_final_chunk() {
        let _lock = crate::llm::ENV_LOCK.lock().await;
        let early = serde_json::json!({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [bash_tool_call()]
            },
            "done": false
        });
        let final_chunk = serde_json::json!({
            "message": {"role": "assistant", "content": ""},
            "done": true
        });
        let body = format!("{}\n{}\n", early, final_chunk).into_bytes();
        let (base_url, server) = serve_chunked(vec![body]).await;

        let client = OllamaClient::with_base_url(base_url);
        let (message, _) = client
            .chat_stream("test-model", vec![], None, |_| {})
            .await
            .unwrap();
        server.await.unwrap();

        let calls = message.tool_calls.expect("early tool call preserved");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "bash");
        assert_eq!(calls[0].function.arguments["command"], "pwd");
    }

    #[tokio::test]
    async fn response_stats_preserve_native_timings_when_reported() {
        let _lock = crate::llm::ENV_LOCK.lock().await;
        let response = serde_json::json!({
            "message": {"role": "assistant", "content": "ok"},
            "done": true,
            "prompt_eval_count": 12,
            "eval_count": 5,
            "total_duration": 9_000_000,
            "load_duration": 1_000_000,
            "prompt_eval_duration": 2_000_000,
            "eval_duration": 3_000_000
        });
        let (base_url, server) = serve_chunked(vec![response.to_string().into_bytes()]).await;
        let client = OllamaClient::with_base_url(base_url);

        let (_, stats) = client
            .chat_with_tools("test-model", vec![], None)
            .await
            .unwrap();
        server.await.unwrap();

        assert_eq!(stats.prompt_tokens, 12);
        assert_eq!(stats.completion_tokens, 5);
        assert_eq!(stats.elapsed_ms, 9);
        assert_eq!(stats.load_duration_ms, Some(1));
        assert_eq!(stats.prompt_eval_duration_ms, Some(2));
        assert_eq!(stats.eval_duration_ms, Some(3));
        assert_eq!(stats.request_count, 1);
        assert_eq!(stats.last_prompt_tokens, Some(12));
    }

    #[tokio::test]
    async fn response_stats_leave_unreported_native_timings_unknown() {
        let _lock = crate::llm::ENV_LOCK.lock().await;
        let response = serde_json::json!({
            "message": {"role": "assistant", "content": "ok"},
            "done": true
        });
        let (base_url, server) = serve_chunked(vec![response.to_string().into_bytes()]).await;
        let client = OllamaClient::with_base_url(base_url);

        let (_, stats) = client
            .chat_with_tools("test-model", vec![], None)
            .await
            .unwrap();
        server.await.unwrap();

        assert_eq!(stats.load_duration_ms, None);
        assert_eq!(stats.prompt_eval_duration_ms, None);
        assert_eq!(stats.eval_duration_ms, None);
        assert_eq!(stats.last_prompt_tokens, None);
    }

    #[tokio::test]
    async fn streaming_stats_measure_time_to_first_token_client_side() {
        let _lock = crate::llm::ENV_LOCK.lock().await;
        let body = serde_json::json!({
            "message": {"role": "assistant", "content": "ok"},
            "done": true
        });
        let (base_url, server) = serve_chunked(vec![format!("{}\n", body).into_bytes()]).await;
        let client = OllamaClient::with_base_url(base_url);

        let (_, stats) = client
            .chat_stream("test-model", vec![], None, |_| {})
            .await
            .unwrap();
        server.await.unwrap();

        assert!(stats.time_to_first_token_ms.is_some());
    }

    #[tokio::test]
    async fn stream_deduplicates_tool_call_repeated_on_final_chunk() {
        let _lock = crate::llm::ENV_LOCK.lock().await;
        let call = bash_tool_call();
        let early = serde_json::json!({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [call.clone()]
            },
            "done": false
        });
        let final_chunk = serde_json::json!({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [call]
            },
            "done": true
        });
        let body = format!("{}\n{}\n", early, final_chunk).into_bytes();
        let (base_url, server) = serve_chunked(vec![body]).await;

        let client = OllamaClient::with_base_url(base_url);
        let (message, _) = client
            .chat_stream("test-model", vec![], None, |_| {})
            .await
            .unwrap();
        server.await.unwrap();

        assert_eq!(message.tool_calls.expect("tool call preserved").len(), 1);
    }

    #[tokio::test]
    async fn stream_preserves_distinct_equal_tool_calls_by_index() {
        let _lock = crate::llm::ENV_LOCK.lock().await;
        let calls = [indexed_bash_tool_call(0), indexed_bash_tool_call(1)];
        let early = serde_json::json!({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": calls
            },
            "done": false
        });
        let final_chunk = serde_json::json!({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": calls
            },
            "done": true
        });
        let body = format!("{}\n{}\n", early, final_chunk).into_bytes();
        let (base_url, server) = serve_chunked(vec![body]).await;

        let client = OllamaClient::with_base_url(base_url);
        let (message, _) = client
            .chat_stream("test-model", vec![], None, |_| {})
            .await
            .unwrap();
        server.await.unwrap();

        let calls = message.tool_calls.expect("tool calls preserved");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.arguments, calls[1].function.arguments);
    }

    #[tokio::test]
    async fn buffered_response_preserves_distinct_equal_tool_calls() {
        let _lock = crate::llm::ENV_LOCK.lock().await;
        let response = serde_json::json!({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [indexed_bash_tool_call(0), indexed_bash_tool_call(1)]
            },
            "done": true
        });
        let (base_url, server) = serve_chunked(vec![response.to_string().into_bytes()]).await;
        let client = OllamaClient::with_base_url(base_url);

        let (message, _) = client
            .chat_with_tools("test-model", vec![], None)
            .await
            .unwrap();
        server.await.unwrap();

        let calls = message.tool_calls.expect("tool calls preserved");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.arguments, calls[1].function.arguments);
    }

    #[tokio::test]
    async fn stream_preserves_multibyte_utf8_split_across_writes() {
        let _lock = crate::llm::ENV_LOCK.lock().await;
        let line = serde_json::json!({
            "message": {"role": "assistant", "content": "café"},
            "done": true
        })
        .to_string();
        let bytes = format!("{}\n", line).into_bytes();
        let split = bytes
            .windows("é".len())
            .position(|window| window == "é".as_bytes())
            .unwrap()
            + 1;
        let parts = vec![bytes[..split].to_vec(), bytes[split..].to_vec()];
        let (base_url, server) = serve_chunked(parts).await;
        let mut streamed = String::new();

        let client = OllamaClient::with_base_url(base_url);
        let (message, _) = client
            .chat_stream("test-model", vec![], None, |token| streamed.push_str(token))
            .await
            .unwrap();
        server.await.unwrap();

        assert_eq!(streamed, "café");
        assert_eq!(message.content, "café");
    }

    #[tokio::test]
    async fn stream_reports_length_limited_response_as_truncated() {
        let _lock = crate::llm::ENV_LOCK.lock().await;
        let body = serde_json::json!({
            "message": {"role": "assistant", "content": "partial"},
            "done": true,
            "done_reason": "length"
        });
        let (base_url, server) = serve_chunked(vec![format!("{}\n", body).into_bytes()]).await;

        let client = OllamaClient::with_base_url(base_url);
        let error = client
            .chat_stream("test-model", vec![], None, |_| {})
            .await
            .unwrap_err();
        server.await.unwrap();

        assert!(error.to_string().contains("truncated"));
        assert!(error.to_string().contains("partial"));
    }

    #[tokio::test]
    async fn request_includes_generation_controls_only_when_configured() {
        let _lock = crate::llm::ENV_LOCK.lock().await;
        let _env = EnvGuard::set(&[
            ("CUBI_MAX_TOKENS", Some("123")),
            ("CUBI_THINK", Some("off")),
            ("CUBI_KEEP_ALIVE", Some("10m")),
            ("CUBI_NUM_CTX", None),
        ]);
        let response = serde_json::json!({
            "message": {"role": "assistant", "content": "ok"},
            "done": true
        });
        let (base_url, server) = serve_chunked(vec![response.to_string().into_bytes()]).await;
        let client = OllamaClient::with_base_url(base_url);

        client
            .chat_with_tools("unknown-model", vec![], None)
            .await
            .unwrap();
        let request = server.await.unwrap();
        let body: serde_json::Value = serde_json::from_str(&request).unwrap();

        assert_eq!(body["options"]["num_predict"], 123);
        assert_eq!(body["think"], false);
        assert_eq!(body["keep_alive"], "10m");
    }

    #[tokio::test]
    async fn request_omits_generation_controls_when_unset() {
        let _lock = crate::llm::ENV_LOCK.lock().await;
        let _env = EnvGuard::set(&[
            ("CUBI_MAX_TOKENS", None),
            ("CUBI_THINK", None),
            ("CUBI_KEEP_ALIVE", None),
            ("CUBI_NUM_CTX", None),
        ]);
        let response = serde_json::json!({
            "message": {"role": "assistant", "content": "ok"},
            "done": true
        });
        let (base_url, server) = serve_chunked(vec![response.to_string().into_bytes()]).await;
        let client = OllamaClient::with_base_url(base_url);

        client
            .chat_with_tools("unknown-model", vec![], None)
            .await
            .unwrap();
        let request = server.await.unwrap();
        let body: serde_json::Value = serde_json::from_str(&request).unwrap();

        assert!(body.get("options").is_none());
        assert!(body.get("think").is_none());
        assert!(body.get("keep_alive").is_none());
    }

    #[tokio::test]
    async fn request_sampling_uses_env_then_profile_then_omits() {
        let _lock = crate::llm::ENV_LOCK.lock().await;
        let cases = [
            (
                "Ternary-Bonsai-27B",
                Some(("0.2", "0.8", "7")),
                Some((0.2, 0.8, 7)),
            ),
            ("Ternary-Bonsai-27B", None, Some((0.7, 0.95, 20))),
            ("unknown-model", None, None),
        ];

        for (model, overrides, expected) in cases {
            let values = match overrides {
                Some((temperature, top_p, top_k)) => [
                    ("CUBI_TEMPERATURE", Some(temperature)),
                    ("CUBI_TOP_P", Some(top_p)),
                    ("CUBI_TOP_K", Some(top_k)),
                    ("CUBI_NUM_CTX", None),
                    ("CUBI_MAX_TOKENS", None),
                ],
                None => [
                    ("CUBI_TEMPERATURE", None),
                    ("CUBI_TOP_P", None),
                    ("CUBI_TOP_K", None),
                    ("CUBI_NUM_CTX", None),
                    ("CUBI_MAX_TOKENS", None),
                ],
            };
            let _env = EnvGuard::set(&values);
            let response = serde_json::json!({
                "message": {"role": "assistant", "content": "ok"},
                "done": true
            });
            let (base_url, server) = serve_chunked(vec![response.to_string().into_bytes()]).await;
            let client = OllamaClient::with_base_url(base_url);

            client.chat_with_tools(model, vec![], None).await.unwrap();
            let request = server.await.unwrap();
            let body: serde_json::Value = serde_json::from_str(&request).unwrap();

            match expected {
                Some((temperature, top_p, top_k)) => {
                    assert_eq!(body["options"]["temperature"], temperature);
                    assert_eq!(body["options"]["top_p"], top_p);
                    assert_eq!(body["options"]["top_k"], top_k);
                }
                None => assert!(body.get("options").is_none()),
            }
        }
    }

    #[tokio::test]
    async fn request_rejects_invalid_sampling_override() {
        let _lock = crate::llm::ENV_LOCK.lock().await;
        let _env = EnvGuard::set(&[
            ("CUBI_TEMPERATURE", Some("-0.1")),
            ("CUBI_TOP_P", None),
            ("CUBI_TOP_K", None),
            ("CUBI_NUM_CTX", None),
        ]);
        let client = OllamaClient::with_base_url("http://127.0.0.1:1".to_string());

        let error = client
            .chat_with_tools("unknown-model", vec![], None)
            .await
            .unwrap_err();

        assert!(error.to_string().contains("CUBI_TEMPERATURE"));
        assert!(error.to_string().contains("non-negative"));
    }

    #[test]
    fn message_text_constructor_omits_tool_fields() {
        let m = Message::text("user", "hello");
        assert_eq!(m.role, "user");
        assert_eq!(m.content, "hello");
        assert!(m.tool_calls.is_none());
        assert!(m.tool_name.is_none());

        // The serialized form must not contain `tool_calls` / `tool_name`
        // for plain user/assistant messages — Ollama rejects unknown fields
        // on older versions and we want to stay backward-compatible.
        let s = serde_json::to_string(&m).unwrap();
        assert!(!s.contains("tool_calls"), "got: {s}");
        assert!(!s.contains("tool_name"), "got: {s}");
    }

    #[test]
    fn message_tool_result_serializes_with_role_and_name() {
        let m = Message::tool_result("bash", "ok", None);
        let s = serde_json::to_string(&m).unwrap();
        assert!(s.contains(r#""role":"tool""#), "got: {s}");
        assert!(s.contains(r#""tool_name":"bash""#), "got: {s}");
        assert!(s.contains(r#""content":"ok""#), "got: {s}");
        // No id supplied → don't emit `tool_call_id` at all (the field is
        // serde-skip'd when None).
        assert!(!s.contains("tool_call_id"), "got: {s}");
    }

    #[test]
    fn message_tool_result_round_trips_tool_call_id_when_present() {
        let m = Message::tool_result("bash", "ok", Some("call_abc123".into()));
        let s = serde_json::to_string(&m).unwrap();
        assert!(s.contains(r#""tool_call_id":"call_abc123""#), "got: {s}");
        let back: Message = serde_json::from_str(&s).unwrap();
        assert_eq!(back.tool_call_id.as_deref(), Some("call_abc123"));
        assert_eq!(back.tool_name.as_deref(), Some("bash"));
    }

    #[test]
    fn message_deserializes_old_payloads_without_tool_fields() {
        // Sessions saved before the schema grew did not include the
        // new fields. We must still be able to load them.
        let raw = r#"{"role":"user","content":"hi"}"#;
        let m: Message = serde_json::from_str(raw).unwrap();
        assert_eq!(m.role, "user");
        assert!(m.tool_calls.is_none());
    }

    #[test]
    fn stream_chunk_parses_intermediate_and_final() {
        let mid = r#"{"message":{"role":"assistant","content":"he"},"done":false}"#;
        let end = r#"{"message":{"role":"assistant","content":"llo"},"done":true}"#;
        let mid: ChatStreamChunk = serde_json::from_str(mid).unwrap();
        let end: ChatStreamChunk = serde_json::from_str(end).unwrap();
        assert!(!mid.done);
        assert_eq!(mid.message.content, "he");
        assert!(end.done);
        assert_eq!(end.message.content, "llo");
    }

    #[test]
    fn stream_chunk_parses_tool_calls_on_final_message() {
        let raw = r#"{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "bash", "arguments": {"command": "ls"}}}
                ]
            },
            "done": true
        }"#;
        let chunk: ChatStreamChunk = serde_json::from_str(raw).unwrap();
        let calls = chunk.message.tool_calls.expect("tool_calls present");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "bash");
        assert_eq!(calls[0].function.arguments["command"], "ls");
    }

    #[test]
    fn resolve_num_ctx_uses_model_window_when_below_cap() {
        assert_eq!(resolve_num_ctx("llama2", None), Some(4_096));
    }

    #[test]
    fn resolve_num_ctx_clamps_large_model_to_default_cap() {
        assert_eq!(
            resolve_num_ctx("qwen3-coder:30b", None),
            Some(DEFAULT_OLLAMA_NUM_CTX_CAP)
        );
    }

    #[test]
    fn resolve_num_ctx_respects_env_override_cap() {
        assert_eq!(
            resolve_num_ctx("qwen3-coder:30b", Some(65_536)),
            Some(65_536)
        );
        assert_eq!(resolve_num_ctx("llama2", Some(2_048)), Some(2_048));
    }

    #[test]
    fn resolve_num_ctx_uses_override_for_unknown_model_only_when_present() {
        assert_eq!(resolve_num_ctx("unknown-model", None), None);
        assert_eq!(resolve_num_ctx("unknown-model", Some(8_192)), Some(8_192));
        assert_eq!(resolve_num_ctx("unknown-model", Some(0)), None);
    }

    #[test]
    fn chat_request_serializes_num_ctx_options() {
        let request = ChatRequest {
            model: "llama2".to_string(),
            messages: vec![],
            stream: false,
            tools: None,
            options: Some(ChatOptions {
                num_ctx: Some(4_096),
                num_predict: None,
                temperature: None,
                top_p: None,
                top_k: None,
            }),
            think: None,
            keep_alive: None,
        };
        let json = serde_json::to_value(request).unwrap();
        assert_eq!(json["options"]["num_ctx"], 4_096);
    }

    #[test]
    fn bare_tool_call_json_synthesizes_tool_call() {
        let mut message = Message::text(
            "assistant",
            r#"{"name":"write_file","arguments":{"path":"a.txt","content":"hi"}}"#,
        );

        assert!(synthesize_tool_call_from_content(&mut message));
        assert_eq!(message.content, "");
        let calls = message.tool_calls.as_ref().expect("tool call synthesized");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "write_file");
        assert_eq!(calls[0].function.arguments["path"], "a.txt");
        assert_eq!(calls[0].function.arguments["content"], "hi");
    }

    #[test]
    fn function_tool_call_json_synthesizes_tool_call() {
        let mut message = Message::text(
            "assistant",
            r#"{"function":{"name":"read_file","arguments":{"path":"src/lib.rs"}}}"#,
        );

        assert!(synthesize_tool_call_from_content(&mut message));
        let calls = message.tool_calls.as_ref().expect("tool call synthesized");
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(calls[0].function.arguments["path"], "src/lib.rs");
        assert_eq!(message.content, "");
    }

    #[test]
    fn tool_call_json_string_arguments_are_parsed() {
        let content = serde_json::json!({
            "name": "write_file",
            "arguments": "{\"path\":\"a.txt\",\"content\":\"hi\"}"
        })
        .to_string();
        let mut message = Message::text("assistant", content);

        assert!(synthesize_tool_call_from_content(&mut message));
        let calls = message.tool_calls.as_ref().expect("tool call synthesized");
        assert_eq!(calls[0].function.name, "write_file");
        assert_eq!(calls[0].function.arguments["path"], "a.txt");
        assert_eq!(calls[0].function.arguments["content"], "hi");
    }

    #[test]
    fn fenced_tool_call_json_synthesizes_tool_call() {
        let mut message = Message::text(
            "assistant",
            "```json\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"src/main.rs\"}}\n```",
        );

        assert!(synthesize_tool_call_from_content(&mut message));
        let calls = message.tool_calls.as_ref().expect("tool call synthesized");
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(calls[0].function.arguments["path"], "src/main.rs");
    }

    #[test]
    fn normal_prose_does_not_synthesize_tool_call() {
        let content = "I should inspect the file before making changes.";
        let mut message = Message::text("assistant", content);

        assert!(!synthesize_tool_call_from_content(&mut message));
        assert_eq!(message.content, content);
        assert!(message.tool_calls.is_none());
    }

    #[test]
    fn prose_that_mentions_json_does_not_synthesize_tool_call() {
        let content = r#"Use this JSON object: {"name":"write_file","arguments":{"path":"a.txt"}}"#;
        let mut message = Message::text("assistant", content);

        assert!(!synthesize_tool_call_from_content(&mut message));
        assert_eq!(message.content, content);
        assert!(message.tool_calls.is_none());
    }

    #[test]
    fn existing_structured_tool_calls_are_untouched() {
        let content = r#"{"name":"write_file","arguments":{"path":"a.txt"}}"#;
        let mut message = Message {
            role: "assistant".to_string(),
            content: content.to_string(),
            tool_calls: Some(vec![ToolCall {
                id: Some("call_existing".to_string()),
                call_type: Some("function".to_string()),
                function: ToolCallFunction {
                    name: "bash".to_string(),
                    arguments: serde_json::json!({"command": "pwd"}),
                },
            }]),
            tool_name: None,
            tool_call_id: None,
        };

        assert!(!synthesize_tool_call_from_content(&mut message));
        assert_eq!(message.content, content);
        let calls = message
            .tool_calls
            .as_ref()
            .expect("existing call preserved");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id.as_deref(), Some("call_existing"));
        assert_eq!(calls[0].function.name, "bash");
        assert_eq!(calls[0].function.arguments["command"], "pwd");
    }
}
