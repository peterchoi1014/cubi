//! Startup configuration and the session's shared, typed runtime settings.
//!
//! Only `Environment::capture` reads process configuration. `main` resolves it
//! before starting Tokio; consumers borrow the installed settings, and ChatCLI
//! holds the same handle for interactive updates. Tests can construct private
//! configurations without installing process-wide state.

mod child_env;
mod environment;
mod generation;
#[cfg(test)]
mod tests;

pub(crate) use child_env::{async_command, command};
pub(crate) use environment::Environment;
pub(crate) use generation::GenerationConfig;

use crate::onboarding::AppConfig;
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolProfile {
    Full,
    Coding,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct StartupOverrides {
    pub model: Option<String>,
    pub trace_tools: Option<String>,
    pub events: Option<String>,
    pub receipts: Option<String>,
    pub quiet: bool,
    pub no_banner: bool,
    pub tui: bool,
    pub classic: bool,
    pub explain_tools: bool,
    pub debug: bool,
    /// Historically persisted UI preferences apply only to chat sessions.
    pub chat_session: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct ProviderConfig {
    pub base_url: Option<String>,
    pub openai_base_url: Option<String>,
    pub openai_api_key: Option<String>,
    pub api_key: Option<String>,
    pub provider: Option<String>,
    pub ollama_base_url: Option<String>,
    pub keys: BTreeMap<String, OsString>,
}

#[derive(Debug, Clone)]
pub(crate) struct UiConfig {
    pub theme: String,
    pub output_style: String,
    pub color: Option<String>,
    pub vim_mode: String,
    pub quiet: bool,
    pub no_banner: bool,
    pub tui: bool,
    pub classic: bool,
    pub explain_tools: bool,
    pub no_spinner: bool,
    pub no_color: bool,
    pub cubi_no_color: bool,
    pub force_color: bool,
    pub clicolor_off: bool,
    pub history_page_size: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct PathConfig {
    pub home: Option<PathBuf>,
    pub policy_file: Option<PathBuf>,
    pub oauth_file: Option<PathBuf>,
    pub plugins_dir: Option<PathBuf>,
    pub trace_tools: Option<PathBuf>,
    pub events: Option<PathBuf>,
    pub receipts: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub(crate) struct FakeConfig {
    pub enabled: bool,
    pub response: String,
    pub model_responses: BTreeMap<String, String>,
    pub tool_call: Option<crate::ollama::ToolCall>,
    pub tool_call_repeat: bool,
    pub fail_models: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct RuntimeConfig {
    pub app: AppConfig,
    pub model: String,
    /// Keep the raw optional override: bench historically accepts an empty
    /// model, whereas interactive model resolution skips it.
    pub model_override: Option<String>,
    pub provider: ProviderConfig,
    pub generation: GenerationConfig,
    pub ui: UiConfig,
    pub paths: PathConfig,
    pub fake: FakeConfig,
    pub tool_profile: ToolProfile,
    pub disable_meta_tools: bool,
    pub read_file_max_lines: usize,
    pub read_file_max_bytes: usize,
    pub compact_extract_memories: bool,
    pub keep_thinking: bool,
    pub no_onboard: bool,
    // These settings have historically been displayed verbatim by slash
    // commands, including unrecognized values. Preserve that representation.
    pub passes: Option<String>,
    pub debug_tool_call: Option<String>,
    pub telemetry: Option<String>,
    pub rate_limit_backoff_ms: Option<u64>,
    pub debug: bool,
    pub log_filter: Option<String>,
    pub editor: String,
    pub shell: Option<String>,
    pub swebench_python: Option<String>,
    pub bench_verify_timeout: Duration,
}

impl RuntimeConfig {
    pub(crate) fn resolve(env: &Environment, app: AppConfig, flags: &StartupOverrides) -> Self {
        let preference = |persisted: &Option<String>, name, default: &str| {
            persisted
                .as_ref()
                .filter(|_| flags.chat_session)
                .cloned()
                .or_else(|| env.string(name))
                .unwrap_or_else(|| default.to_string())
        };
        let model_override = flags.model.clone().or_else(|| env.string("CUBI_MODEL"));
        let model = Self::resolve_model(model_override.as_deref(), &app, crate::DEFAULT_MODEL);
        let quiet = flags.quiet || env.flag("CUBI_QUIET");
        let color = app
            .color
            .as_ref()
            .filter(|_| flags.chat_session)
            .cloned()
            .or_else(|| env.string("CUBI_COLOR"));
        let generation = GenerationConfig::resolve(env, app.llm_max_retries);
        Self {
            model,
            model_override,
            provider: ProviderConfig {
                base_url: env.string("CUBI_BASE_URL"),
                openai_base_url: env.string("OPENAI_BASE_URL"),
                openai_api_key: env.string("OPENAI_API_KEY"),
                api_key: env.string("CUBI_API_KEY"),
                provider: env.string("CUBI_PROVIDER"),
                ollama_base_url: env.string("OLLAMA_BASE_URL"),
                keys: env.provider_keys(),
            },
            ui: UiConfig {
                theme: preference(&app.theme, "CUBI_THEME", "auto"),
                output_style: preference(
                    &app.output_style,
                    "CUBI_OUTPUT_STYLE",
                    crate::output_styles::DEFAULT_STYLE,
                ),
                color,
                vim_mode: preference(&app.vim_mode, "CUBI_VIM_MODE", "off"),
                quiet,
                no_banner: flags.no_banner || quiet || env.flag("CUBI_NO_BANNER"),
                tui: flags.tui || env.flag("CUBI_TUI"),
                classic: flags.classic || env.flag("CUBI_CLASSIC"),
                explain_tools: flags.explain_tools || env.flag("CUBI_EXPLAIN_TOOLS"),
                no_spinner: quiet || env.value("CUBI_NO_SPINNER").is_some(),
                no_color: env.value("NO_COLOR").is_some(),
                cubi_no_color: env.value("CUBI_NO_COLOR").is_some(),
                force_color: env.flag("CLICOLOR_FORCE"),
                clicolor_off: env.value("CLICOLOR") == Some("0"),
                history_page_size: env
                    .value("CUBI_HISTORY_PAGE")
                    .and_then(|raw| raw.parse::<usize>().ok())
                    .filter(|size| (1..=500).contains(size))
                    .unwrap_or(10),
            },
            paths: PathConfig {
                home: env.home_dir(),
                policy_file: env.string("CUBI_POLICY_FILE").map(PathBuf::from),
                oauth_file: env
                    .string("CUBI_OAUTH_FILE")
                    .filter(|value| !value.trim().is_empty())
                    .map(PathBuf::from),
                plugins_dir: env
                    .string("CUBI_PLUGINS_DIR")
                    .filter(|value| !value.is_empty())
                    .map(PathBuf::from),
                trace_tools: flags
                    .trace_tools
                    .clone()
                    .or_else(|| env.string("CUBI_TRACE_TOOLS"))
                    .map(PathBuf::from),
                events: flags
                    .events
                    .clone()
                    .or_else(|| env.string("CUBI_EVENTS"))
                    .filter(|value| !value.is_empty())
                    .map(PathBuf::from),
                receipts: flags
                    .receipts
                    .clone()
                    .or_else(|| env.string("CUBI_RECEIPTS").filter(|v| !v.is_empty()))
                    .or_else(|| app.receipts.clone().filter(|v| !v.is_empty()))
                    .map(PathBuf::from),
            },
            fake: FakeConfig {
                enabled: env.value("CUBI_FAKE_LLM").is_some(),
                response: env
                    .string("CUBI_FAKE_LLM_RESPONSE")
                    .unwrap_or_else(|| "hi".to_string()),
                model_responses: env
                    .value("CUBI_FAKE_LLM_MODEL_RESPONSES")
                    .and_then(|raw| serde_json::from_str(raw).ok())
                    .unwrap_or_default(),
                tool_call: env
                    .value("CUBI_FAKE_LLM_TOOL_CALL")
                    .and_then(|raw| serde_json::from_str(raw).ok()),
                tool_call_repeat: env.one_or_true("CUBI_FAKE_LLM_TOOL_CALL_REPEAT"),
                fail_models: env
                    .value("CUBI_FAKE_LLM_FAIL_MODELS")
                    .map(|raw| raw.split(',').map(|s| s.trim().to_string()).collect())
                    .unwrap_or_default(),
            },
            tool_profile: match env.value(crate::agent_loop::TOOL_PROFILE_ENV) {
                Some(value) if value.eq_ignore_ascii_case("coding") => ToolProfile::Coding,
                _ => ToolProfile::Full,
            },
            disable_meta_tools: env.os_value("CUBI_DISABLE_META_TOOLS").is_some(),
            read_file_max_lines: env
                .positive("CUBI_READ_FILE_MAX_LINES")
                .unwrap_or(crate::builtin_tools::READ_FILE_DEFAULT_MAX_LINES),
            read_file_max_bytes: env
                .positive("CUBI_READ_FILE_MAX_BYTES")
                .unwrap_or(crate::builtin_tools::READ_FILE_DEFAULT_MAX_BYTES),
            compact_extract_memories: env.value("CUBI_COMPACT_EXTRACT_MEMORIES") == Some("1"),
            keep_thinking: env.one_or_true("CUBI_KEEP_THINKING"),
            no_onboard: env.flag("CUBI_NO_ONBOARD"),
            passes: env.string("AGENT_PASSES_OVERRIDE"),
            debug_tool_call: env.string("CUBI_DEBUG_TOOL_CALL"),
            telemetry: env.string("CUBI_TELEMETRY"),
            rate_limit_backoff_ms: env
                .value("CUBI_RATE_LIMIT_BACKOFF_MS")
                .and_then(|raw| raw.parse().ok()),
            debug: crate::user_error::debug_mode_with(|name| env.string(name), flags.debug),
            log_filter: env.string("CUBI_LOG"),
            editor: ["CUBI_EDITOR", "VISUAL", "EDITOR"]
                .into_iter()
                .filter_map(|name| env.value(name))
                .map(str::trim)
                .find(|value| !value.is_empty())
                .unwrap_or(if cfg!(windows) { "notepad.exe" } else { "vi" })
                .to_string(),
            shell: env.string("CUBI_SHELL").filter(|v| !v.is_empty()),
            swebench_python: env
                .string("CUBI_SWEBENCH_PYTHON")
                .filter(|v| !v.trim().is_empty()),
            bench_verify_timeout: Duration::from_secs(
                env.value("CUBI_BENCH_VERIFY_TIMEOUT_SECS")
                    .and_then(|raw| raw.parse::<u64>().ok())
                    .filter(|seconds| *seconds > 0)
                    .unwrap_or(120),
            ),
            generation,
            app,
        }
    }

    pub(crate) fn resolve_model(
        override_model: Option<&str>,
        app: &AppConfig,
        fallback: &str,
    ) -> String {
        override_model
            .filter(|model| !model.is_empty())
            .or_else(|| {
                app.default_model
                    .as_deref()
                    .filter(|model| !model.is_empty())
            })
            .unwrap_or(fallback)
            .to_string()
    }

    pub(crate) fn apply_onboarding(&mut self, app: AppConfig) {
        self.model =
            Self::resolve_model(self.model_override.as_deref(), &app, crate::DEFAULT_MODEL);
        self.app = app;
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SharedConfig(Arc<RwLock<RuntimeConfig>>);

impl SharedConfig {
    pub(crate) fn new(config: RuntimeConfig) -> Self {
        Self(Arc::new(RwLock::new(config)))
    }

    pub(crate) fn read<T>(&self, read: impl FnOnce(&RuntimeConfig) -> T) -> T {
        read(
            &self
                .0
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        )
    }

    pub(crate) fn update(&self, update: impl FnOnce(&mut RuntimeConfig)) {
        update(
            &mut self
                .0
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
    }

    pub(crate) fn snapshot(&self) -> RuntimeConfig {
        self.read(Clone::clone)
    }
}

static CONFIG: OnceLock<SharedConfig> = OnceLock::new();

pub(crate) fn install(config: RuntimeConfig) -> anyhow::Result<SharedConfig> {
    let shared = SharedConfig::new(config);
    CONFIG
        .set(shared.clone())
        .map_err(|_| anyhow::anyhow!("runtime configuration was already initialized"))?;
    Ok(shared)
}

pub(crate) fn shared() -> SharedConfig {
    #[cfg(test)]
    if CONFIG.get().is_none() {
        // The test harness does not run main. Env fixtures retain their existing
        // locks, while each test gets a private handle instead of a sticky global.
        let env = Environment::capture();
        let app = AppConfig::load_from_home(env.home_dir().as_deref());
        return SharedConfig::new(RuntimeConfig::resolve(
            &env,
            app,
            &StartupOverrides::default(),
        ));
    }
    // main installs configuration synchronously before running any consumer.
    CONFIG
        .get()
        .expect("main initialized runtime configuration")
        .clone()
}

pub(crate) fn with<T>(read: impl FnOnce(&RuntimeConfig) -> T) -> T {
    shared().read(read)
}
