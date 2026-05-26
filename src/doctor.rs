//! `cubi doctor` preflight subcommand.
//!
//! Runs a fixed battery of checks (config parse, sessions dir writable,
//! default-model host reachable, `ollama` on PATH, MCP server commands
//! resolvable, plugin parse) and prints a human-readable report or
//! machine-readable JSON. Exits 0 when nothing failed, 2 otherwise.
//!
//! The in-REPL `/doctor` slash command (defined in `cli/mod.rs`) is a
//! separate, looser surface; this module is the headless preflight that
//! CI / install scripts can rely on.

use std::path::PathBuf;
use std::time::Duration;

use serde::Serialize;
use serde_json::json;

use crate::onboarding::AppConfig;
use crate::style::CubiStyle;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckStatus {
    Ok,
    Warn,
    Fail,
}

#[derive(Debug, Clone, Serialize)]
pub struct CheckResult {
    pub name: &'static str,
    pub status: CheckStatus,
    pub message: String,
}

impl CheckResult {
    fn ok(name: &'static str, message: impl Into<String>) -> Self {
        Self {
            name,
            status: CheckStatus::Ok,
            message: message.into(),
        }
    }
    fn warn(name: &'static str, message: impl Into<String>) -> Self {
        Self {
            name,
            status: CheckStatus::Warn,
            message: message.into(),
        }
    }
    fn fail(name: &'static str, message: impl Into<String>) -> Self {
        Self {
            name,
            status: CheckStatus::Fail,
            message: message.into(),
        }
    }
}

/// Public entry point. Runs all checks, prints output, returns
/// `true` when there were no failures.
pub async fn run(json: bool) -> bool {
    let results = run_checks().await;
    if json {
        print_json(&results);
    } else {
        print_human(&results);
    }
    !results.iter().any(|r| r.status == CheckStatus::Fail)
}

async fn run_checks() -> Vec<CheckResult> {
    let mut results = Vec::new();
    results.push(check_config());
    results.push(check_sessions_dir());
    results.push(check_model_host().await);
    results.push(check_ollama_binary());
    results.extend(check_mcp_servers());
    results.push(check_plugins_dir());
    results
}

fn check_config() -> CheckResult {
    match AppConfig::storage_path() {
        Some(path) if path.exists() => match std::fs::read_to_string(&path) {
            Ok(raw) => match serde_json::from_str::<serde_json::Value>(&raw) {
                Ok(_) => CheckResult::ok("config", format!("parsed {}", path.display())),
                Err(e) => CheckResult::fail(
                    "config",
                    format!("failed to parse {}: {}", path.display(), e),
                ),
            },
            Err(e) => CheckResult::fail(
                "config",
                format!("failed to read {}: {}", path.display(), e),
            ),
        },
        Some(path) => CheckResult::ok("config", format!("no config yet at {}", path.display())),
        None => CheckResult::warn("config", "could not resolve home directory"),
    }
}

fn check_sessions_dir() -> CheckResult {
    let Some(dir) = crate::sessions::sessions_root() else {
        return CheckResult::warn("sessions_dir", "could not resolve home directory");
    };
    if let Err(e) = std::fs::create_dir_all(&dir) {
        return CheckResult::fail(
            "sessions_dir",
            format!("cannot create {}: {}", dir.display(), e),
        );
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let probe = dir.join(format!(".doctor-probe-{}-{}", std::process::id(), nanos));
    match std::fs::write(&probe, b"ok") {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            CheckResult::ok("sessions_dir", format!("writable: {}", dir.display()))
        }
        Err(e) => CheckResult::fail(
            "sessions_dir",
            format!("not writable ({}): {}", dir.display(), e),
        ),
    }
}

async fn check_model_host() -> CheckResult {
    let config = AppConfig::load();
    let model = crate::onboarding::resolve_model(&config, "qwen3:4b");

    // Resolve provider + URL the same way `executor`/`main` do.
    let openai_base = std::env::var("OPENAI_BASE_URL")
        .ok()
        .or_else(|| std::env::var("CUBI_BASE_URL").ok());
    let has_openai_key = std::env::var("OPENAI_API_KEY")
        .ok()
        .filter(|v| !v.is_empty())
        .is_some();

    let (provider, url) = if has_openai_key || openai_base.is_some() {
        let base = openai_base.unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        ("openai", format!("{}/models", base.trim_end_matches('/')))
    } else {
        let base = std::env::var("OLLAMA_BASE_URL")
            .ok()
            .unwrap_or_else(|| "http://localhost:11434".to_string());
        ("ollama", format!("{}/api/tags", base.trim_end_matches('/')))
    };

    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            return CheckResult::fail("model_host", format!("http client init failed: {}", e));
        }
    };

    match client.get(&url).send().await {
        Ok(resp) => {
            let status = resp.status();
            if status.is_success() {
                CheckResult::ok(
                    "model_host",
                    format!("{} reachable at {} (model={})", provider, url, model),
                )
            } else if status.as_u16() == 404 {
                CheckResult::warn(
                    "model_host",
                    format!("{} responded 404 at {} (endpoint not found)", provider, url),
                )
            } else {
                CheckResult::warn(
                    "model_host",
                    format!("{} responded HTTP {} at {}", provider, status.as_u16(), url),
                )
            }
        }
        Err(e) => CheckResult::fail(
            "model_host",
            format!("{} unreachable at {}: {}", provider, url, e),
        ),
    }
}

fn check_ollama_binary() -> CheckResult {
    match which_on_path("ollama") {
        Some(path) => CheckResult::ok("ollama_binary", format!("found at {}", path.display())),
        None => CheckResult::warn("ollama_binary", "`ollama` not on PATH (informational)"),
    }
}

fn check_mcp_servers() -> Vec<CheckResult> {
    let config = match crate::mcp_config::McpConfig::load() {
        Ok(c) => c,
        Err(e) => {
            return vec![CheckResult::warn(
                "mcp_servers",
                format!("failed to load mcp.json: {}", e),
            )];
        }
    };
    if config.mcp_servers.is_empty() {
        return vec![CheckResult::ok("mcp_servers", "no MCP servers configured")];
    }
    let mut out = Vec::new();
    for (name, server) in &config.mcp_servers {
        if let Some(cmd) = server.command.as_deref() {
            match which_on_path(cmd) {
                Some(path) => out.push(CheckResult::ok(
                    "mcp_server",
                    format!("'{}' command '{}' -> {}", name, cmd, path.display()),
                )),
                None => out.push(CheckResult::warn(
                    "mcp_server",
                    format!("'{}' command '{}' not on PATH", name, cmd),
                )),
            }
        } else if server.http_url.is_some() {
            out.push(CheckResult::ok(
                "mcp_server",
                format!("'{}' is http (no PATH check)", name),
            ));
        } else {
            out.push(CheckResult::warn(
                "mcp_server",
                format!("'{}' has neither command nor httpUrl", name),
            ));
        }
    }
    out
}

fn check_plugins_dir() -> CheckResult {
    let plugins = crate::plugins::load_plugins();
    let count = plugins.len();
    CheckResult::ok("plugins", format!("{} plugin(s) discovered", count))
}

/// Lightweight `which`-equivalent without a dependency: walks `PATH`
/// honoring `PATHEXT` on Windows.
fn which_on_path(cmd: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    let exts: Vec<String> = if cfg!(windows) {
        std::env::var("PATHEXT")
            .unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string())
            .split(';')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    } else {
        vec![String::new()]
    };
    for dir in std::env::split_paths(&path_var) {
        for ext in &exts {
            let candidate = dir.join(format!("{}{}", cmd, ext));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn print_json(results: &[CheckResult]) {
    let summary = json!({
        "ok": !results.iter().any(|r| r.status == CheckStatus::Fail),
        "checks": results,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&summary).unwrap_or_default()
    );
}

fn print_human(results: &[CheckResult]) {
    println!("{}", "Doctor:".bright_yellow().bold());
    for r in results {
        let glyph = match r.status {
            CheckStatus::Ok => "✓".bright_green().to_string(),
            CheckStatus::Warn => "!".bright_yellow().to_string(),
            CheckStatus::Fail => "✗".bright_red().to_string(),
        };
        println!("  {} [{}] {}", glyph, r.name, r.message);
    }
    let failed = results
        .iter()
        .filter(|r| r.status == CheckStatus::Fail)
        .count();
    let warned = results
        .iter()
        .filter(|r| r.status == CheckStatus::Warn)
        .count();
    println!();
    if failed == 0 {
        println!(
            "All critical checks passed ({} warning{}).",
            warned,
            if warned == 1 { "" } else { "s" }
        );
    } else {
        println!("{} check(s) failed.", failed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ok_result_renders_with_ok_status() {
        let r = CheckResult::ok("test", "everything fine");
        assert_eq!(r.status, CheckStatus::Ok);
        assert_eq!(r.name, "test");
        assert_eq!(r.message, "everything fine");
    }

    #[test]
    fn fail_result_renders_with_fail_status() {
        let r = CheckResult::fail("x", "broken");
        assert_eq!(r.status, CheckStatus::Fail);
    }

    #[test]
    fn json_serialization_includes_status_and_message() {
        let r = CheckResult::warn("foo", "minor issue");
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains("\"status\":\"warn\""));
        assert!(s.contains("\"name\":\"foo\""));
        assert!(s.contains("\"message\":\"minor issue\""));
    }

    #[test]
    fn which_finds_present_command() {
        // `cargo` is necessarily present while running `cargo test`.
        let p = which_on_path("cargo");
        assert!(
            p.is_some(),
            "cargo should be discoverable on PATH during tests"
        );
    }

    #[test]
    fn which_returns_none_for_missing_command() {
        assert!(which_on_path("definitely-not-a-real-binary-zzz-cubi").is_none());
    }
}
