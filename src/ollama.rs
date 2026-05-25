use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub message: Message,
    // Note: 'done' field exists in API but we don't need it for non-streaming
    #[allow(dead_code)]
    pub done: bool,
}

pub struct OllamaClient {
    base_url: String,
    client: reqwest::Client,
}

impl OllamaClient {
    pub fn new() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub async fn chat(&self, model: &str, messages: Vec<Message>) -> Result<String> {
        let request = ChatRequest {
            model: model.to_string(),
            messages,
            stream: false,
        };

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

        Ok(chat_response.message.content)
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
