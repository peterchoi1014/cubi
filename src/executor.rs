use crate::llm::{LlmBackend, create_provider};
use crate::ollama::{ChatStats, Message, ToolSpec};
use anyhow::Result;

pub struct AIExecutor {
    backend: LlmBackend,
    model: String,
}

impl AIExecutor {
    pub async fn new(model: String, _cpu_workers: usize) -> Result<Self> {
        Ok(Self::new_from_env(model))
    }

    pub fn new_from_env(model: String) -> Self {
        Self {
            backend: create_provider(),
            model,
        }
    }

    pub async fn chat(&self, messages: Vec<Message>) -> Result<String> {
        self.backend.chat(&self.model, messages).await
    }

    pub async fn chat_stream<F>(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolSpec>>,
        on_token: F,
    ) -> Result<(Message, ChatStats)>
    where
        F: FnMut(&str),
    {
        self.backend
            .chat_stream(&self.model, messages, tools, on_token)
            .await
    }

    pub async fn chat_with_tools(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolSpec>>,
    ) -> Result<(Message, ChatStats)> {
        self.backend
            .chat_with_tools(&self.model, messages, tools)
            .await
    }

    pub fn get_model(&self) -> &str {
        &self.model
    }

    pub fn provider_name(&self) -> &str {
        self.backend.provider_name()
    }

    pub async fn switch_model(&mut self, model: String) -> Result<()> {
        let models = self.backend.list_models().await?;
        if !models.is_empty() && !models.iter().any(|m| m.starts_with(&model) || m == &model) {
            anyhow::bail!("Model '{}' not found. Available: {:?}", model, models);
        }
        self.model = model;
        Ok(())
    }
}
