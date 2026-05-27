//! `cubi run <file.md>` markdown script parser.
//!
//! Reads a markdown file and produces a [`RunScript`] containing
//! optional frontmatter overrides and a list of pre-rendered chat
//! messages. The format is intentionally minimal so prompt packs stay
//! reviewable in plain `cat`:
//!
//! ```text
//! ---
//! model: qwen3:4b
//! system: You are a pedantic reviewer.
//! tools: false
//! ---
//! First user turn.
//!
//! ---
//!
//! Assistant reply to seed the transcript.
//!
//! ---
//!
//! Follow-up user turn (this one is sent to the model).
//! ```
//!
//! YAML is hand-parsed (one key:value per line) so we don't have to
//! pull in a 50KLOC YAML crate for three keys. Body segments are split
//! on `\n\n---\n\n` and alternate user/assistant by index.
//!
//! Trailing assistant turns are dropped — the file always ends with a
//! user prompt that the runner sends through the normal one-shot path.

use anyhow::{Context, Result, anyhow};
use std::path::Path;

use crate::ollama::Message;

/// Parsed script ready to be handed to the CLI.
#[derive(Debug, Default, Clone)]
pub struct RunScript {
    /// Frontmatter `model:` override; honored by the runner via env
    /// before `AppConfig::load`.
    pub model: Option<String>,
    /// Frontmatter `system:` override; injected as the leading system
    /// message at CLI construction time.
    pub system: Option<String>,
    /// Frontmatter `tools: false` disables MCP tool dispatch for the
    /// run. `None` means "honor the default", `Some(true)` is explicit
    /// opt-in (already the default), `Some(false)` disables.
    pub tools: Option<bool>,
    /// Pre-rendered transcript messages to seed history with. The
    /// final user message is moved to [`Self::prompt`] so the runner
    /// can send it via `run_one_shot`.
    pub prefill: Vec<Message>,
    /// Required: the final user prompt to send. A script that ends
    /// with an assistant turn (or has no user turn at all) is rejected
    /// at parse time — there's nothing to send.
    pub prompt: String,
}

/// Parses `text` according to the format documented at the module
/// level. Returns an error if the body has no user turn to send.
pub fn parse(text: &str) -> Result<RunScript> {
    let (frontmatter, body) = split_frontmatter(text);
    let (model, system, tools) = if let Some(fm) = frontmatter {
        parse_frontmatter(fm)?
    } else {
        (None, None, None)
    };

    let segments: Vec<String> = body
        .split("\n\n---\n\n")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if segments.is_empty() {
        return Err(anyhow!("script body has no user prompt"));
    }

    // Alternating user / assistant by index. Trailing assistant turns
    // are dropped because we have no new user prompt to send.
    let mut messages: Vec<Message> = segments
        .iter()
        .enumerate()
        .map(|(i, content)| {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            Message::text(role, content)
        })
        .collect();
    // Drop trailing assistant turn(s), if any.
    while messages
        .last()
        .map(|m| m.role == "assistant")
        .unwrap_or(false)
    {
        messages.pop();
    }
    let prompt_msg = messages
        .pop()
        .ok_or_else(|| anyhow!("script has no user turn to send"))?;

    Ok(RunScript {
        model,
        system,
        tools,
        prefill: messages,
        prompt: prompt_msg.content,
    })
}

/// Loads `path` and parses it. `@` prefixes are tolerated for
/// ergonomics (`cubi run @prompts/review.md`).
pub fn load(path_arg: &str) -> Result<RunScript> {
    let p = path_arg.strip_prefix('@').unwrap_or(path_arg);
    let text = std::fs::read_to_string(Path::new(p))
        .with_context(|| format!("failed to read run script '{}'", p))?;
    parse(&text)
}

/// Returns `(frontmatter_text, body_text)` when `text` starts with a
/// `---` line followed by another `---` line. The frontmatter is
/// everything in between (exclusive); body is everything after the
/// closing fence (with one trailing newline stripped for tidiness).
/// When no frontmatter is detected the entire input is the body.
fn split_frontmatter(text: &str) -> (Option<&str>, &str) {
    // Accept BOM and leading whitespace-only lines? Keep it strict so
    // accidental indentation doesn't silently disable overrides.
    let trimmed_start = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut lines = trimmed_start.split_inclusive('\n');
    let first = match lines.next() {
        Some(l) => l,
        None => return (None, text),
    };
    if first.trim_end() != "---" {
        return (None, text);
    }
    let after_first = &trimmed_start[first.len()..];
    // Find next `---` line.
    let mut acc = 0usize;
    for line in after_first.split_inclusive('\n') {
        if line.trim_end() == "---" {
            let fm = &after_first[..acc];
            let body_start = acc + line.len();
            let body = &after_first[body_start..];
            return (Some(fm), body.trim_start_matches('\n'));
        }
        acc += line.len();
    }
    (None, text)
}

fn parse_frontmatter(fm: &str) -> Result<(Option<String>, Option<String>, Option<bool>)> {
    let mut model = None;
    let mut system = None;
    let mut tools = None;
    for raw in fm.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            return Err(anyhow!("invalid frontmatter line: {:?}", raw));
        };
        let key = key.trim();
        let mut value = value.trim().to_string();
        // Strip surrounding quotes (single or double) for ergonomics.
        if (value.starts_with('"') && value.ends_with('"') && value.len() >= 2)
            || (value.starts_with('\'') && value.ends_with('\'') && value.len() >= 2)
        {
            value = value[1..value.len() - 1].to_string();
        }
        match key {
            "model" => model = Some(value),
            "system" => system = Some(value),
            "tools" => match value.as_str() {
                "true" | "yes" | "on" | "1" => tools = Some(true),
                "false" | "no" | "off" | "0" => tools = Some(false),
                _ => return Err(anyhow!("invalid `tools` value: {:?}", value)),
            },
            other => {
                return Err(anyhow!("unknown frontmatter key: {:?}", other));
            }
        }
    }
    Ok((model, system, tools))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_only_yields_single_user_prompt() {
        let s = parse("hello world\n").unwrap();
        assert_eq!(s.model, None);
        assert_eq!(s.system, None);
        assert_eq!(s.tools, None);
        assert!(s.prefill.is_empty());
        assert_eq!(s.prompt, "hello world");
    }

    #[test]
    fn frontmatter_overrides_are_parsed() {
        let s = parse("---\nmodel: qwen3:4b\nsystem: be terse\ntools: false\n---\nhi\n").unwrap();
        assert_eq!(s.model.as_deref(), Some("qwen3:4b"));
        assert_eq!(s.system.as_deref(), Some("be terse"));
        assert_eq!(s.tools, Some(false));
        assert_eq!(s.prompt, "hi");
    }

    #[test]
    fn alternating_segments_preload_history_user_first() {
        let body = "u1\n\n---\n\na1\n\n---\n\nu2";
        let s = parse(body).unwrap();
        assert_eq!(s.prefill.len(), 2);
        assert_eq!(s.prefill[0].role, "user");
        assert_eq!(s.prefill[0].content, "u1");
        assert_eq!(s.prefill[1].role, "assistant");
        assert_eq!(s.prefill[1].content, "a1");
        assert_eq!(s.prompt, "u2");
    }

    #[test]
    fn trailing_assistant_segment_is_dropped() {
        let body = "u1\n\n---\n\na1";
        let s = parse(body).unwrap();
        assert!(s.prefill.is_empty());
        assert_eq!(s.prompt, "u1");
    }

    #[test]
    fn unknown_frontmatter_key_errors() {
        assert!(parse("---\nfoo: bar\n---\nhi\n").is_err());
    }

    #[test]
    fn invalid_tools_value_errors() {
        assert!(parse("---\ntools: maybe\n---\nhi\n").is_err());
    }

    #[test]
    fn empty_body_errors() {
        assert!(parse("---\nmodel: x\n---\n").is_err());
    }

    #[test]
    fn quoted_frontmatter_values_are_unwrapped() {
        let s = parse("---\nsystem: \"be terse\"\n---\nhi\n").unwrap();
        assert_eq!(s.system.as_deref(), Some("be terse"));
    }
}
