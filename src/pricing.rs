//! Built-in token pricing table for the `/usage` slash command and the
//! optional per-turn usage footer.
//!
//! Lookup is case-insensitive and prefix-aware so e.g.
//! `claude-3-5-sonnet-20240620` resolves to the same rate as the bare
//! `claude-3-5-sonnet` entry. Anything matched as a local model (Ollama,
//! `llama`, `qwen`, `phi`, ...) returns a zero-cost row so the formatter
//! can still attribute it as `$0.00 (local)` rather than `—`.
//!
//! Prices are USD per 1k tokens and reflect the public list price at the
//! time of writing; they are not pulled live, on purpose, so `/usage`
//! never makes a network call. Update by hand when needed — these are
//! load-bearing only for cost estimates, not billing.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelPricing {
    pub prompt_per_1k: f64,
    pub completion_per_1k: f64,
    /// `true` when the row was produced by the "local model" heuristic
    /// rather than a built-in priced entry; formatters use this to
    /// label cost as `$0.00 (local)`.
    pub local: bool,
}

impl ModelPricing {
    pub const fn priced(prompt_per_1k: f64, completion_per_1k: f64) -> Self {
        Self {
            prompt_per_1k,
            completion_per_1k,
            local: false,
        }
    }

    pub const fn local() -> Self {
        Self {
            prompt_per_1k: 0.0,
            completion_per_1k: 0.0,
            local: true,
        }
    }

    /// USD cost for the supplied token counts.
    pub fn cost_usd(&self, prompt_tokens: u64, completion_tokens: u64) -> f64 {
        let p = (prompt_tokens as f64 / 1000.0) * self.prompt_per_1k;
        let c = (completion_tokens as f64 / 1000.0) * self.completion_per_1k;
        p + c
    }
}

/// Built-in (prefix, pricing) table. Order matters — the first matching
/// prefix wins, so list more-specific entries before less-specific ones.
const TABLE: &[(&str, ModelPricing)] = &[
    // OpenAI (USD / 1k tokens, public list, late 2024)
    ("gpt-4o-mini", ModelPricing::priced(0.00015, 0.0006)),
    ("gpt-4o", ModelPricing::priced(0.0025, 0.01)),
    ("gpt-4.1-mini", ModelPricing::priced(0.0004, 0.0016)),
    ("gpt-4.1", ModelPricing::priced(0.002, 0.008)),
    // Anthropic
    ("claude-3-5-sonnet", ModelPricing::priced(0.003, 0.015)),
    ("claude-3-5-haiku", ModelPricing::priced(0.001, 0.005)),
    ("claude-3-opus", ModelPricing::priced(0.015, 0.075)),
    // Local-model prefixes — return zero-cost entries so the formatter
    // can show "$0.00 (local)".
    ("ollama/", ModelPricing::local()),
    ("llama", ModelPricing::local()),
    ("qwen", ModelPricing::local()),
    ("phi", ModelPricing::local()),
    ("mistral", ModelPricing::local()),
    ("gemma", ModelPricing::local()),
];

/// Returns the [`ModelPricing`] for `model_id` if any built-in prefix
/// matches, else `None`. Comparison is case-insensitive.
pub fn lookup(model_id: &str) -> Option<ModelPricing> {
    let id = model_id.to_ascii_lowercase();
    for (prefix, pricing) in TABLE {
        if id.starts_with(prefix) {
            return Some(*pricing);
        }
    }
    None
}

/// Formats a USD cost as `$X.XXXX` or `$X.XX (local)`. Returns `—` when
/// the pricing is unknown.
pub fn format_cost(pricing: Option<ModelPricing>, prompt: u64, completion: u64) -> String {
    match pricing {
        None => "—".to_string(),
        Some(p) if p.local => "$0.00 (local)".to_string(),
        Some(p) => format!("${:.4}", p.cost_usd(prompt, completion)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_exact_known_models() {
        assert!(lookup("gpt-4o").is_some());
        assert!(lookup("gpt-4o-mini").is_some());
        assert!(lookup("gpt-4.1").is_some());
    }

    #[test]
    fn lookup_prefix_match_for_versioned_anthropic_ids() {
        let p = lookup("claude-3-5-sonnet-20240620").expect("prefix match");
        assert!((p.prompt_per_1k - 0.003).abs() < 1e-9);
        assert!((p.completion_per_1k - 0.015).abs() < 1e-9);
        assert!(!p.local);
    }

    #[test]
    fn lookup_local_prefixes_return_zero_cost() {
        for id in ["ollama/llama3", "llama3.2:1b", "qwen3:4b", "phi4-mini"] {
            let p = lookup(id).unwrap_or_else(|| panic!("missing local prefix for {id}"));
            assert!(p.local, "{id} should be local");
            assert_eq!(p.cost_usd(1_000_000, 1_000_000), 0.0);
        }
    }

    #[test]
    fn lookup_unknown_returns_none() {
        assert!(lookup("totally-made-up-model-9000").is_none());
    }

    #[test]
    fn lookup_is_case_insensitive() {
        assert!(lookup("GPT-4o").is_some());
        assert!(lookup("Claude-3-5-Sonnet").is_some());
    }

    #[test]
    fn cost_usd_scales_linearly() {
        let p = ModelPricing::priced(0.002, 0.008);
        // 1000 prompt + 1000 completion → 0.002 + 0.008 = 0.010
        assert!((p.cost_usd(1000, 1000) - 0.010).abs() < 1e-9);
        // 500 prompt + 0 completion → 0.001
        assert!((p.cost_usd(500, 0) - 0.001).abs() < 1e-9);
    }

    #[test]
    fn format_cost_unknown_is_em_dash() {
        assert_eq!(format_cost(None, 100, 100), "—");
    }

    #[test]
    fn format_cost_local_is_zero_local() {
        let s = format_cost(Some(ModelPricing::local()), 100, 100);
        assert_eq!(s, "$0.00 (local)");
    }

    #[test]
    fn format_cost_priced_is_dollar_amount() {
        let p = ModelPricing::priced(0.002, 0.008);
        let s = format_cost(Some(p), 1000, 1000);
        assert_eq!(s, "$0.0100");
    }
}
