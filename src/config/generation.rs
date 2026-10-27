use super::Environment;
use crate::model_profile::Sampling;
use std::ffi::OsString;
use std::str::FromStr;
use std::time::Duration;

/// Invalid sampling values used to fail on the first model request, not on
/// `--help` or startup. Parse once, retain that deferred error and its original
/// bytes for a child that must observe the same error.
#[derive(Debug, Clone)]
pub(super) struct SamplingValue<T> {
    pub value: Result<Option<T>, String>,
    pub invalid_input: Option<OsString>,
}

impl<T: FromStr + Copy> SamplingValue<T> {
    fn parse(
        env: &Environment,
        name: &str,
        parse_error: &str,
        valid: impl FnOnce(T) -> bool,
        range_error: &str,
    ) -> Self {
        let Some(raw) = env.os_value(name) else {
            return Self {
                value: Ok(None),
                invalid_input: None,
            };
        };
        let value = match raw.to_str() {
            None => Err(format!("{} must contain valid UTF-8", name)),
            Some(raw) => match raw.trim().parse::<T>() {
                Ok(value) if valid(value) => Ok(Some(value)),
                Ok(_) => Err(range_error.to_string()),
                Err(_) => Err(parse_error.to_string()),
            },
        };
        let invalid_input = value.is_err().then(|| raw.to_os_string());
        Self {
            value,
            invalid_input,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GenerationConfig {
    pub num_ctx: Option<usize>,
    pub context_window: Option<usize>,
    pub max_tokens: Option<u64>,
    pub think: Option<bool>,
    pub keep_alive: Option<String>,
    pub http_timeout: Duration,
    pub max_retries: u32,
    pub prompt_tokens_override: Option<usize>,
    pub(super) temperature: SamplingValue<f64>,
    pub(super) top_p: SamplingValue<f64>,
    pub(super) top_k: SamplingValue<u64>,
}

impl GenerationConfig {
    pub(super) fn resolve(env: &Environment, persisted_retries: u32) -> Self {
        Self {
            num_ctx: env.positive("CUBI_NUM_CTX"),
            context_window: env.positive("CUBI_CONTEXT_WINDOW"),
            max_tokens: env.positive("CUBI_MAX_TOKENS"),
            think: env.value("CUBI_THINK").and_then(|raw| {
                match raw.trim().to_ascii_lowercase().as_str() {
                    "on" => Some(true),
                    "off" => Some(false),
                    _ => None,
                }
            }),
            keep_alive: env
                .value("CUBI_KEEP_ALIVE")
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .map(str::to_string),
            http_timeout: Duration::from_secs(
                env.positive("CUBI_HTTP_TIMEOUT_SECS").unwrap_or(300),
            ),
            max_retries: env
                .value("CUBI_LLM_MAX_RETRIES")
                .and_then(|raw| raw.parse().ok())
                .unwrap_or(persisted_retries),
            prompt_tokens_override: env
                .value("CUBI_MAX_PROMPT_TOKENS_OVERRIDE")
                .and_then(|raw| raw.parse::<usize>().ok())
                .filter(|value| *value > 0),
            temperature: SamplingValue::parse(
                env,
                "CUBI_TEMPERATURE",
                "CUBI_TEMPERATURE must be a non-negative number",
                |v: f64| v.is_finite() && v >= 0.0,
                "CUBI_TEMPERATURE must be a finite, non-negative number",
            ),
            top_p: SamplingValue::parse(
                env,
                "CUBI_TOP_P",
                "CUBI_TOP_P must be a number in (0, 1]",
                |v: f64| v.is_finite() && v > 0.0 && v <= 1.0,
                "CUBI_TOP_P must be a finite number in (0, 1]",
            ),
            top_k: SamplingValue::parse(
                env,
                "CUBI_TOP_K",
                "CUBI_TOP_K must be a positive integer",
                |v: u64| v > 0,
                "CUBI_TOP_K must be a positive integer",
            ),
        }
    }

    pub(crate) fn sampling_for_model(&self, model: &str) -> anyhow::Result<Sampling> {
        let defaults = crate::model_profile::lookup(model)
            .map(|profile| profile.default_sampling)
            .unwrap_or_default();
        Ok(Sampling {
            temperature: self
                .temperature
                .value
                .clone()
                .map_err(anyhow::Error::msg)?
                .or(defaults.temperature),
            top_p: self
                .top_p
                .value
                .clone()
                .map_err(anyhow::Error::msg)?
                .or(defaults.top_p),
            top_k: self
                .top_k
                .value
                .clone()
                .map_err(anyhow::Error::msg)?
                .or(defaults.top_k),
        })
    }
}
