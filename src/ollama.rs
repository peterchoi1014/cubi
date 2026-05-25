use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::time::{Duration, sleep};

#[derive(Debug, Serialize)]
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
        }
    }

    /// Constructor for a tool-result message produced by the host after
    /// executing a model-requested tool call.
    pub fn tool_result(tool_name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: content.into(),
            tool_calls: None,
            tool_name: Some(tool_name.into()),
        }
    }
}

/// A single tool invocation the model wants the host to perform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Ollama doesn't always set an id; tolerate its absence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// Ollama returns arguments as a JSON object; older clients sometimes
    /// stringify it. Accept both via `serde_json::Value`.
    pub arguments: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub message: Message,
    // Note: 'done' field exists in API but we don't need it for non-streaming
    #[allow(dead_code)]
    pub done: bool,
}

/// One streaming chunk from `/api/chat` when `stream: true`. Each NDJSON
/// line decodes to one of these. The final chunk has `done: true` and a
/// fully-populated `message` (token deltas are in `message.content` on
/// intermediate chunks).
#[derive(Debug, Deserialize)]
pub struct ChatStreamChunk {
    pub message: Message,
    pub done: bool,
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
            client: reqwest::Client::new(),
        }
    }

    /// Creates an OllamaClient with a custom base URL.
    pub fn with_base_url(base_url: String) -> Self {
        Self {
            base_url,
            client: reqwest::Client::new(),
        }
    }

    pub async fn chat(&self, model: &str, messages: Vec<Message>) -> Result<String> {
        let msg = self.chat_with_tools(model, messages, None).await?;
        Ok(msg.content)
    }

    /// Non-streaming chat that optionally forwards a native tool list. Returns
    /// the full assistant [`Message`] so the caller can inspect `tool_calls`
    /// and run an agent loop on top.
    pub async fn chat_with_tools(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolSpec>>,
    ) -> Result<Message> {
        with_retry(3, || {
            let request = ChatRequest {
                model: model.to_string(),
                messages: messages.clone(),
                stream: false,
                tools: tools.clone(),
            };
            async move {
                let response = self
                    .client
                    .post(format!("{}/api/chat", self.base_url))
                    .json(&request)
                    .send()
                    .await
                    .context("Failed to send request to Ollama")?;

                if !response.status().is_success() {
                    let error_text = response.text().await.unwrap_or_default();
                    anyhow::bail!("Ollama API error: {}", error_text);
                }

                let chat_response: ChatResponse = response
                    .json()
                    .await
                    .context("Failed to parse Ollama response")?;

                Ok(chat_response.message)
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
    ) -> Result<Message>
    where
        F: FnMut(&str),
    {
        let response = with_retry(3, || {
            let request = ChatRequest {
                model: model.to_string(),
                messages: messages.clone(),
                stream: true,
                tools: tools.clone(),
            };
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
        let mut buf = String::new();
        let mut content = String::new();
        let mut final_msg: Option<Message> = None;

        // Local closure so the trailing-buffer case after the read loop can
        // reuse the same parse-and-dispatch logic without duplication.
        let mut handle_line =
            |line: &str, content: &mut String, final_msg: &mut Option<Message>| {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    return;
                }
                let parsed: ChatStreamChunk = match serde_json::from_str(trimmed) {
                    Ok(c) => c,
                    Err(_) => return, // tolerate a malformed mid-stream line
                };
                if !parsed.message.content.is_empty() {
                    on_token(&parsed.message.content);
                    content.push_str(&parsed.message.content);
                }
                if parsed.done {
                    let mut m = parsed.message;
                    // Replace fragmentary content with the accumulated text so
                    // callers don't have to reconstruct it.
                    m.content = content.clone();
                    *final_msg = Some(m);
                }
            };

        while let Some(chunk) = stream.next().await {
            let bytes = chunk.context("Stream read failed")?;
            buf.push_str(&String::from_utf8_lossy(&bytes));
            while let Some(nl) = buf.find('\n') {
                let line = buf[..nl].to_string();
                buf.drain(..=nl);
                handle_line(&line, &mut content, &mut final_msg);
            }
        }

        // NDJSON isn't required to end with a trailing newline. If the
        // server sent the final `done:true` object without `\n`, it would
        // otherwise sit in `buf` forever and we'd report "ended without
        // done chunk". Flush whatever's left.
        if !buf.trim().is_empty() {
            let remaining = std::mem::take(&mut buf);
            handle_line(&remaining, &mut content, &mut final_msg);
        }

        final_msg.ok_or_else(|| anyhow::anyhow!("Ollama stream ended without a `done` chunk"))
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
        let m = Message::tool_result("bash", "ok");
        let s = serde_json::to_string(&m).unwrap();
        assert!(s.contains(r#""role":"tool""#), "got: {s}");
        assert!(s.contains(r#""tool_name":"bash""#), "got: {s}");
        assert!(s.contains(r#""content":"ok""#), "got: {s}");
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
}
