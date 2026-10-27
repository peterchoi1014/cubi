use crate::llm::{LlmBackend, create_provider};
use crate::ollama::{ChatStats, Message, ToolSpec};
use anyhow::Result;

pub struct AIExecutor {
    backend: LlmBackend,
    model: String,
    config: crate::config::SharedConfig,
}

impl AIExecutor {
    pub async fn new(model: String) -> Result<Self> {
        Ok(Self::new_from_env(model))
    }

    pub fn new_from_env(model: String) -> Self {
        Self {
            backend: create_provider(),
            model,
            config: crate::config::shared(),
        }
    }

    /// Test-only constructor that bypasses env-var-driven provider
    /// selection. Used by the consensus unit tests so they don't
    /// pollute the process env (which is global across the test
    /// binary's parallel runners).
    #[cfg(test)]
    pub fn with_backend(backend: LlmBackend, model: String) -> Self {
        Self {
            backend,
            model,
            config: crate::config::shared(),
        }
    }

    pub(crate) fn runtime_config(&self) -> crate::config::SharedConfig {
        self.config.clone()
    }

    pub async fn chat(&self, messages: Vec<Message>) -> Result<String> {
        crate::context_budget::ensure_request_fits(
            &self.model,
            self.backend.provider_name(),
            &messages,
            None,
        )?;
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
        crate::context_budget::ensure_request_fits(
            &self.model,
            self.backend.provider_name(),
            &messages,
            tools.as_deref(),
        )?;
        self.backend
            .chat_stream(&self.model, messages, tools, on_token)
            .await
    }

    pub async fn chat_with_tools(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolSpec>>,
    ) -> Result<(Message, ChatStats)> {
        crate::context_budget::ensure_request_fits(
            &self.model,
            self.backend.provider_name(),
            &messages,
            tools.as_deref(),
        )?;
        self.backend
            .chat_with_tools(&self.model, messages, tools)
            .await
    }

    /// Variant of [`chat_with_tools`] that overrides the executor's
    /// stored default model for a single call. Used by the consensus
    /// dispatcher to run the same goal under several models without
    /// mutating the executor's global `model` field (which would race
    /// with concurrent subagents).
    pub async fn chat_with_model(
        &self,
        model: &str,
        messages: Vec<Message>,
    ) -> Result<(Message, ChatStats)> {
        crate::context_budget::ensure_request_fits(
            model,
            self.backend.provider_name(),
            &messages,
            None,
        )?;
        self.backend.chat_with_tools(model, messages, None).await
    }

    /// Variant of [`chat_with_model`] that forwards a tool list to the
    /// backend while still overriding the executor's default model for
    /// just this call.
    pub async fn chat_with_model_and_tools(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolSpec>>,
    ) -> Result<(Message, ChatStats)> {
        crate::context_budget::ensure_request_fits(
            model,
            self.backend.provider_name(),
            &messages,
            tools.as_deref(),
        )?;
        self.backend.chat_with_tools(model, messages, tools).await
    }

    pub fn get_model(&self) -> &str {
        &self.model
    }

    pub fn provider_name(&self) -> &str {
        self.backend.provider_name()
    }

    /// Backend endpoint URL, or `None` for the fake test backend.
    pub fn base_url(&self) -> Option<&str> {
        self.backend.base_url()
    }

    pub fn is_local_backend(&self) -> bool {
        self.backend.is_local()
    }

    /// Lists models the configured backend reports. Used by `/doctor`
    /// so it can probe whichever backend is actually selected (Ollama
    /// or any OpenAI-compatible local server like `llama-server` or
    /// LM Studio) instead of always pinging `:11434`.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        self.backend.list_models().await
    }

    pub async fn switch_model(&mut self, model: String) -> Result<()> {
        let models = self.backend.list_models().await?;
        if !models.is_empty() && !models.iter().any(|m| m.starts_with(&model) || m == &model) {
            anyhow::bail!("Model '{}' not found. Available: {:?}", model, models);
        }
        self.config.update(|config| config.model = model.clone());
        self.model = model;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

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

    #[tokio::test]
    async fn ollama_request_rejects_output_cap_that_fills_context_window() {
        let _lock = crate::llm::ENV_LOCK.lock().await;
        let _env = EnvGuard::set(&[
            ("CUBI_NUM_CTX", Some("8192")),
            ("CUBI_CONTEXT_WINDOW", None),
            ("CUBI_MAX_TOKENS", Some("8192")),
            ("CUBI_LLM_MAX_RETRIES", Some("0")),
        ]);
        let executor = AIExecutor::with_backend(
            LlmBackend::Ollama(crate::ollama::OllamaClient::with_base_url(
                "http://127.0.0.1:1".to_string(),
            )),
            "unknown-model".to_string(),
        );

        let error = executor
            .chat_with_tools(vec![Message::text("user", "x")], None)
            .await
            .unwrap_err();

        assert!(error.to_string().contains("refusing model request"));
        assert!(error.to_string().contains("8192 reserved for completion"));
    }

    #[tokio::test]
    async fn openai_request_rejects_output_cap_that_fills_context_window() {
        let _lock = crate::llm::ENV_LOCK.lock().await;
        let _env = EnvGuard::set(&[
            ("CUBI_NUM_CTX", None),
            ("CUBI_CONTEXT_WINDOW", Some("8192")),
            ("CUBI_MAX_TOKENS", Some("8192")),
            ("CUBI_LLM_MAX_RETRIES", Some("0")),
        ]);
        let executor = AIExecutor::with_backend(
            LlmBackend::OpenAi(crate::llm::OpenAiClient::new(
                "http://127.0.0.1:1/v1".to_string(),
                String::new(),
            )),
            "unknown-model".to_string(),
        );

        let error = executor
            .chat_with_tools(vec![Message::text("user", "x")], None)
            .await
            .unwrap_err();

        assert!(error.to_string().contains("refusing model request"));
        assert!(error.to_string().contains("8192 reserved for completion"));
    }
}
