use crate::ollama::{Message, OllamaClient, ToolSpec};
use anyhow::Result;

pub struct AIExecutor {
    ollama: OllamaClient,
    model: String,
}

impl AIExecutor {
    pub async fn new(model: String, _cpu_workers: usize) -> Result<Self> {
        // Note: In a production distributed system, you would initialize
        // Repartir pool here and use it to distribute AI inference tasks
        // across multiple workers/machines. For this demo, we're focusing
        // on the local Ollama integration.

        let ollama = OllamaClient::new();

        Ok(Self { ollama, model })
    }

    pub async fn chat(&self, messages: Vec<Message>) -> Result<String> {
        // Execute AI inference through Ollama
        let response = self.ollama.chat(&self.model, messages).await?;
        Ok(response)
    }

    /// Streaming chat. `on_token` fires for each text fragment as it
    /// arrives. Returns the final assembled assistant [`Message`] (which
    /// may also carry `tool_calls` when tools are in play).
    pub async fn chat_stream<F>(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolSpec>>,
        on_token: F,
    ) -> Result<Message>
    where
        F: FnMut(&str),
    {
        self.ollama
            .chat_stream(&self.model, messages, tools, on_token)
            .await
    }

    /// One-shot non-streaming chat with native tool calling. Returns the
    /// full message including any `tool_calls`.
    #[allow(dead_code)] // Wired up by the agent loop in the next commit.
    pub async fn chat_with_tools(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolSpec>>,
    ) -> Result<Message> {
        self.ollama
            .chat_with_tools(&self.model, messages, tools)
            .await
    }

    pub fn get_model(&self) -> &str {
        &self.model
    }

    pub async fn switch_model(&mut self, model: String) -> Result<()> {
        // Verify model exists before switching
        let models = self.ollama.list_models().await?;
        if !models.iter().any(|m| m.starts_with(&model)) {
            anyhow::bail!("Model '{}' not found. Available: {:?}", model, models);
        }
        self.model = model;
        Ok(())
    }
}
