use super::{RuntimeConfig, ToolProfile};
use anyhow::Context;
use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::path::Path;

pub(crate) fn command(program: impl AsRef<OsStr>) -> std::process::Command {
    let mut command = std::process::Command::new(program);
    super::with(|config| config.apply_to_child(&mut command));
    command
}

pub(crate) fn async_command(program: impl AsRef<OsStr>) -> tokio::process::Command {
    command(program).into()
}

impl RuntimeConfig {
    /// Replace Cubi's inherited configuration with an explicit snapshot. The
    /// caller then applies isolation policy (home, audit paths, meta-tool caps).
    /// Command::env changes only the child's environment, never the process.
    pub(crate) fn apply_to_child(&self, command: &mut std::process::Command) {
        for (name, _) in std::env::vars_os() {
            if is_managed_key_with_case(&name, cfg!(windows)) {
                command.env_remove(name);
            }
        }
        command.envs(self.child_env());
    }

    pub(crate) fn child_env(&self) -> BTreeMap<OsString, OsString> {
        let mut values = BTreeMap::new();
        let mut put = |key: &str, value: &dyn std::fmt::Display| {
            values.insert(OsString::from(key), OsString::from(value.to_string()));
        };
        put("CUBI_MODEL", &self.model);
        for (key, value) in [
            ("CUBI_BASE_URL", &self.provider.base_url),
            ("OPENAI_BASE_URL", &self.provider.openai_base_url),
            ("OPENAI_API_KEY", &self.provider.openai_api_key),
            ("CUBI_API_KEY", &self.provider.api_key),
            ("CUBI_PROVIDER", &self.provider.provider),
            ("OLLAMA_BASE_URL", &self.provider.ollama_base_url),
            ("CUBI_KEEP_ALIVE", &self.generation.keep_alive),
            ("CUBI_COLOR", &self.ui.color),
            ("AGENT_PASSES_OVERRIDE", &self.passes),
            ("CUBI_DEBUG_TOOL_CALL", &self.debug_tool_call),
            ("CUBI_TELEMETRY", &self.telemetry),
            ("CUBI_LOG", &self.log_filter),
            ("CUBI_SHELL", &self.shell),
            ("CUBI_SWEBENCH_PYTHON", &self.swebench_python),
        ] {
            if let Some(value) = value {
                put(key, value);
            }
        }
        for (key, value) in [
            ("CUBI_NUM_CTX", self.generation.num_ctx),
            ("CUBI_CONTEXT_WINDOW", self.generation.context_window),
            (
                "CUBI_MAX_PROMPT_TOKENS_OVERRIDE",
                self.generation.prompt_tokens_override,
            ),
        ] {
            if let Some(value) = value {
                put(key, &value);
            }
        }
        for (key, value) in [
            ("CUBI_MAX_TOKENS", self.generation.max_tokens),
            ("CUBI_RATE_LIMIT_BACKOFF_MS", self.rate_limit_backoff_ms),
        ] {
            if let Some(value) = value {
                put(key, &value);
            }
        }
        if let Some(think) = self.generation.think {
            put("CUBI_THINK", &if think { "on" } else { "off" });
        }
        put(
            "CUBI_HTTP_TIMEOUT_SECS",
            &self.generation.http_timeout.as_secs(),
        );
        put("CUBI_LLM_MAX_RETRIES", &self.generation.max_retries);
        put("CUBI_THEME", &self.ui.theme);
        put("CUBI_OUTPUT_STYLE", &self.ui.output_style);
        put("CUBI_VIM_MODE", &self.ui.vim_mode);
        put("CUBI_HISTORY_PAGE", &self.ui.history_page_size);
        put("CUBI_EDITOR", &self.editor);
        put("CUBI_READ_FILE_MAX_LINES", &self.read_file_max_lines);
        put("CUBI_READ_FILE_MAX_BYTES", &self.read_file_max_bytes);
        put(
            "CUBI_BENCH_VERIFY_TIMEOUT_SECS",
            &self.bench_verify_timeout.as_secs(),
        );
        put(
            "CUBI_TOOL_PROFILE",
            &match self.tool_profile {
                ToolProfile::Full => "full",
                ToolProfile::Coding => "coding",
            },
        );
        for (key, enabled) in [
            ("CUBI_QUIET", self.ui.quiet),
            ("CUBI_NO_BANNER", self.ui.no_banner),
            ("CUBI_TUI", self.ui.tui),
            ("CUBI_CLASSIC", self.ui.classic),
            ("CUBI_EXPLAIN_TOOLS", self.ui.explain_tools),
            ("CUBI_NO_SPINNER", self.ui.no_spinner),
            ("CUBI_NO_COLOR", self.ui.cubi_no_color),
            ("CUBI_NO_ONBOARD", self.no_onboard),
            ("CUBI_DISABLE_META_TOOLS", self.disable_meta_tools),
            ("CUBI_KEEP_THINKING", self.keep_thinking),
            (
                "CUBI_COMPACT_EXTRACT_MEMORIES",
                self.compact_extract_memories,
            ),
            ("CUBI_DEBUG", self.debug),
            ("CUBI_FAKE_LLM", self.fake.enabled),
            ("CUBI_FAKE_LLM_TOOL_CALL_REPEAT", self.fake.tool_call_repeat),
        ] {
            if enabled {
                put(key, &"1");
            }
        }
        put("CUBI_FAKE_LLM_RESPONSE", &self.fake.response);
        // A map of strings and ToolCall's serde_json::Value fields cannot fail
        // JSON serialization. Do not silently drop a configured test hook.
        put(
            "CUBI_FAKE_LLM_MODEL_RESPONSES",
            &serde_json::to_string(&self.fake.model_responses)
                .expect("string response map is JSON-serializable"),
        );
        put(
            "CUBI_FAKE_LLM_FAIL_MODELS",
            &self.fake.fail_models.join(","),
        );
        if let Some(call) = &self.fake.tool_call {
            put(
                "CUBI_FAKE_LLM_TOOL_CALL",
                &serde_json::to_string(call).expect("tool-call JSON fields are serializable"),
            );
        }
        for (key, setting) in [
            ("CUBI_TEMPERATURE", &self.generation.temperature),
            ("CUBI_TOP_P", &self.generation.top_p),
        ] {
            if let Ok(Some(value)) = setting.value {
                put(key, &value);
            }
        }
        if let Ok(Some(value)) = self.generation.top_k.value {
            put("CUBI_TOP_K", &value);
        }
        for (key, raw) in [
            (
                "CUBI_TEMPERATURE",
                &self.generation.temperature.invalid_input,
            ),
            ("CUBI_TOP_P", &self.generation.top_p.invalid_input),
            ("CUBI_TOP_K", &self.generation.top_k.invalid_input),
        ] {
            if let Some(raw) = raw {
                values.insert(OsString::from(key), raw.clone());
            }
        }
        for (key, path) in [
            ("CUBI_HOME", &self.paths.home),
            ("CUBI_POLICY_FILE", &self.paths.policy_file),
            ("CUBI_OAUTH_FILE", &self.paths.oauth_file),
            ("CUBI_PLUGINS_DIR", &self.paths.plugins_dir),
            ("CUBI_TRACE_TOOLS", &self.paths.trace_tools),
            ("CUBI_EVENTS", &self.paths.events),
            ("CUBI_RECEIPTS", &self.paths.receipts),
        ] {
            if let Some(path) = path {
                values.insert(OsString::from(key), path.as_os_str().to_owned());
            }
        }
        values.extend(
            self.provider
                .keys
                .iter()
                .map(|(name, value)| (OsStr::new(name).to_owned(), value.clone())),
        );
        values
    }

    /// Fields that have no env interface travel in the isolated child's own
    /// config file. UI values already travel in child_env; leaving those
    /// persisted slots unset lets the caller's headless overrides win.
    pub(crate) fn write_child_config(&self, home: &Path, model: &str) -> anyhow::Result<()> {
        let mut app = self.app.clone();
        app.default_model = Some(model.to_string());
        app.onboarded = true;
        app.config_version = crate::migrations::CURRENT_CONFIG_VERSION;
        app.theme = None;
        app.output_style = None;
        app.color = None;
        app.vim_mode = None;
        app.receipts = None;
        let directory = home.join(".cubi");
        std::fs::create_dir_all(&directory)
            .with_context(|| format!("create {}", directory.display()))?;
        let path = directory.join("config.json");
        std::fs::write(&path, serde_json::to_string_pretty(&app)?)
            .with_context(|| format!("write {}", path.display()))
    }
}

fn is_managed_key_with_case(name: &OsStr, case_insensitive: bool) -> bool {
    let normalized = super::env_key::normalize_with_case(name, case_insensitive);
    normalized.to_str().is_some_and(|name| {
        name.starts_with("CUBI_")
            || name.starts_with("AICHAT_")
            || name.starts_with("AI_CHAT_CLI_")
            || matches!(
                name,
                "OPENAI_API_KEY" | "OPENAI_BASE_URL" | "OLLAMA_BASE_URL" | "AGENT_PASSES_OVERRIDE"
            )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn managed_child_keys_select_platform_case_semantics() {
        assert!(!is_managed_key_with_case(OsStr::new("cubi_model"), false));
        assert!(is_managed_key_with_case(OsStr::new("cubi_model"), true));
        assert!(is_managed_key_with_case(OsStr::new("openai_api_key"), true));
        assert!(!is_managed_key_with_case(OsStr::new("unrelated"), true));
    }
}
