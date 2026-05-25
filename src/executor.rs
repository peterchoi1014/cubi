use crate::ollama::{Message, OllamaClient};
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
