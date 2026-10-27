use anyhow::{Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Locality {
    Local,
    Remote,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Sampling {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelPrice {
    pub prompt_per_1k: f64,
    pub completion_per_1k: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelProfile {
    pub nominal_context_window: Option<usize>,
    pub tool_calling: bool,
    pub locality: Locality,
    pub default_sampling: Sampling,
    pub pricing: Option<ModelPrice>,
}

pub fn lookup(model_id: &str) -> Option<ModelProfile> {
    let id = model_id.to_ascii_lowercase();
    if id.contains("bonsai") {
        return Some(ModelProfile {
            nominal_context_window: Some(262_144),
            tool_calling: true,
            locality: Locality::Local,
            default_sampling: Sampling {
                temperature: Some(0.7),
                top_p: Some(0.95),
                top_k: Some(20),
            },
            pricing: Some(ModelPrice {
                prompt_per_1k: 0.0,
                completion_per_1k: 0.0,
            }),
        });
    }

    let nominal_context_window = context_window(&id);
    let non_tool_capable = is_non_tool_capable(&id);
    let pricing = pricing(&id);
    if nominal_context_window.is_none() && !non_tool_capable && pricing.is_none() {
        return None;
    }

    Some(ModelProfile {
        nominal_context_window,
        tool_calling: !non_tool_capable,
        locality: locality(&id),
        default_sampling: Sampling::default(),
        pricing,
    })
}

pub fn sampling_for_model(model_id: &str) -> Result<Sampling> {
    let mut sampling = lookup(model_id)
        .map(|profile| profile.default_sampling)
        .unwrap_or_default();

    if let Some(raw) = env_value("CUBI_TEMPERATURE")? {
        let value = raw
            .trim()
            .parse::<f64>()
            .map_err(|_| anyhow::anyhow!("CUBI_TEMPERATURE must be a non-negative number"))?;
        if !value.is_finite() || value < 0.0 {
            bail!("CUBI_TEMPERATURE must be a finite, non-negative number");
        }
        sampling.temperature = Some(value);
    }
    if let Some(raw) = env_value("CUBI_TOP_P")? {
        let value = raw
            .trim()
            .parse::<f64>()
            .map_err(|_| anyhow::anyhow!("CUBI_TOP_P must be a number in (0, 1]"))?;
        if !(value.is_finite() && 0.0 < value && value <= 1.0) {
            bail!("CUBI_TOP_P must be a finite number in (0, 1]");
        }
        sampling.top_p = Some(value);
    }
    if let Some(raw) = env_value("CUBI_TOP_K")? {
        let value = raw
            .trim()
            .parse::<u64>()
            .map_err(|_| anyhow::anyhow!("CUBI_TOP_K must be a positive integer"))?;
        if value == 0 {
            bail!("CUBI_TOP_K must be a positive integer");
        }
        sampling.top_k = Some(value);
    }

    Ok(sampling)
}

fn env_value(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            bail!("{} must contain valid UTF-8", name)
        }
    }
}

fn context_window(id: &str) -> Option<usize> {
    match id {
        "gemma3:270m" | "gemma3:1b" => return Some(32_768),
        _ => {}
    }

    let without_tag = id.split(':').next().unwrap_or(id);
    let base = without_tag.rsplit('/').next().unwrap_or(without_tag);
    match base {
        "llama3.3" | "llama3.2" | "llama3.1" | "llama3" => Some(128_000),
        "llama2" => Some(4_096),
        "mistral" | "mixtral" => Some(32_768),
        "mistral-small3.2" | "mistral-small3.1" | "mistral-small" => Some(131_072),
        "devstral" => Some(131_072),
        "qwen2.5" | "qwen2" | "qwen2.5-coder" => Some(128_000),
        "qwen3" => Some(32_768),
        "qwen3.5" | "qwen3-coder" => Some(256_000),
        "codellama" => Some(16_384),
        "phi3" | "phi-3" => Some(128_000),
        "phi4" | "phi-4" => Some(16_384),
        "phi4-mini" => Some(128_000),
        "gemma2" | "gemma" => Some(8_192),
        "gemma3" => Some(128_000),
        "gemma4" => Some(256_000),
        "deepseek-coder" => Some(16_384),
        "granite3.3" | "granite3.2" | "granite3.1" => Some(131_072),
        "hermes3" => Some(131_072),
        "command-r7b" | "command-r" => Some(131_072),
        "glm-5.2" | "glm5.2" => Some(1_000_000),
        "gpt-4o" | "gpt-4o-mini" | "gpt-4-turbo" | "gpt-4" => Some(128_000),
        "gpt-3.5-turbo" => Some(16_384),
        "o1" | "o1-mini" | "o1-preview" => Some(128_000),
        "claude-3-opus" | "claude-3-sonnet" | "claude-3-haiku" => Some(200_000),
        "claude-3.5-sonnet" | "claude-3.5-haiku" => Some(200_000),
        _ => None,
    }
}

fn is_non_tool_capable(id: &str) -> bool {
    let family = id.split(':').next().unwrap_or(id);
    if family == "llama3.2"
        && let Some(tag) = id.split(':').nth(1)
        && tag.starts_with("1b")
    {
        return true;
    }
    matches!(
        family,
        "tinyllama" | "smollm" | "smollm2" | "gemma" | "gemma2" | "gemma3" | "phi" | "phi3"
    ) || family.starts_with("orca-mini")
}

fn pricing(id: &str) -> Option<ModelPrice> {
    let full = pricing_prefix(id);
    if full.is_some() {
        return full;
    }
    let stripped = id.rsplit('/').next().unwrap_or(id);
    (stripped != id).then(|| pricing_prefix(stripped)).flatten()
}

fn pricing_prefix(id: &str) -> Option<ModelPrice> {
    const LOCAL: ModelPrice = ModelPrice {
        prompt_per_1k: 0.0,
        completion_per_1k: 0.0,
    };
    const TABLE: &[(&str, ModelPrice)] = &[
        (
            "gpt-4o-mini",
            ModelPrice {
                prompt_per_1k: 0.00015,
                completion_per_1k: 0.0006,
            },
        ),
        (
            "gpt-4o",
            ModelPrice {
                prompt_per_1k: 0.0025,
                completion_per_1k: 0.01,
            },
        ),
        (
            "gpt-4.1-mini",
            ModelPrice {
                prompt_per_1k: 0.0004,
                completion_per_1k: 0.0016,
            },
        ),
        (
            "gpt-4.1",
            ModelPrice {
                prompt_per_1k: 0.002,
                completion_per_1k: 0.008,
            },
        ),
        (
            "claude-3-5-sonnet",
            ModelPrice {
                prompt_per_1k: 0.003,
                completion_per_1k: 0.015,
            },
        ),
        (
            "claude-3-5-haiku",
            ModelPrice {
                prompt_per_1k: 0.001,
                completion_per_1k: 0.005,
            },
        ),
        (
            "claude-3-opus",
            ModelPrice {
                prompt_per_1k: 0.015,
                completion_per_1k: 0.075,
            },
        ),
        ("ollama/", LOCAL),
        ("llama", LOCAL),
        ("qwen", LOCAL),
        ("phi", LOCAL),
        ("mistral", LOCAL),
        ("gemma", LOCAL),
        ("glm", LOCAL),
    ];

    TABLE
        .iter()
        .find_map(|(prefix, price)| id.starts_with(prefix).then_some(*price))
}

fn locality(id: &str) -> Locality {
    if id.starts_with("ollama/") {
        return Locality::Local;
    }
    let without_tag = id.split(':').next().unwrap_or(id);
    let base = without_tag.rsplit('/').next().unwrap_or(without_tag);
    let remote = base.starts_with("gpt-") || base.starts_with("o1") || base.starts_with("claude-");
    if remote {
        Locality::Remote
    } else {
        Locality::Local
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bonsai_identifier_variants_resolve_to_local_tool_profile() {
        for model in [
            "Ternary-Bonsai-27B",
            "ternary-bonsai-27b:latest",
            "local/BONSAI-Q4_K_M",
        ] {
            let profile = lookup(model).unwrap_or_else(|| panic!("missing profile for {model}"));

            assert_eq!(profile.nominal_context_window, Some(262_144));
            assert!(profile.tool_calling);
            assert_eq!(profile.locality, Locality::Local);
            assert_eq!(
                profile.default_sampling,
                Sampling {
                    temperature: Some(0.7),
                    top_p: Some(0.95),
                    top_k: Some(20),
                }
            );
            assert_eq!(
                profile.pricing,
                Some(ModelPrice {
                    prompt_per_1k: 0.0,
                    completion_per_1k: 0.0,
                })
            );
        }
    }
}
