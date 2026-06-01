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
    /// Set when `doctor --fix` applied (or attempted) a safe remedy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fix: Option<String>,
}

impl CheckResult {
    fn ok(name: &'static str, message: impl Into<String>) -> Self {
        Self {
            name,
            status: CheckStatus::Ok,
            message: message.into(),
            fix: None,
        }
    }
    fn warn(name: &'static str, message: impl Into<String>) -> Self {
        Self {
            name,
            status: CheckStatus::Warn,
            message: message.into(),
            fix: None,
        }
    }
    fn fail(name: &'static str, message: impl Into<String>) -> Self {
        Self {
            name,
            status: CheckStatus::Fail,
            message: message.into(),
            fix: None,
        }
    }
    fn with_fix(mut self, fix: impl Into<String>) -> Self {
        self.fix = Some(fix.into());
        self
    }
}

/// Public entry point. Runs all checks, prints output, returns
/// `true` when there were no failures. When `fix` is true, safe
/// automated remedies are applied for failing checks; the remedy line
/// is prefixed with `+` in human output (and is reflected in the
/// machine-readable `fix` field on the JSON payload).
pub async fn run(json: bool, fix: bool) -> bool {
    let results = run_checks(fix).await;
    if json {
        print_json(&results);
    } else {
        print_human(&results);
    }
    !results.iter().any(|r| r.status == CheckStatus::Fail)
}

async fn run_checks(fix: bool) -> Vec<CheckResult> {
    let mut results = Vec::new();
    results.push(check_config(fix));
    results.push(check_sessions_dir(fix));
    results.push(check_model_host().await);
    results.push(check_ollama_binary());
    results.extend(check_mcp_servers());
    results.push(check_plugins_dir());
    results.push(check_completions_installed(fix));
    results.push(check_browser().await);
    results
}

async fn check_browser() -> CheckResult {
    #[cfg(feature = "browser")]
    {
        match crate::browser_tool::doctor_probe(3).await {
            Ok(()) => CheckResult::ok("browser", "headless browser launches"),
            Err(e) => CheckResult::warn("browser", format!("{e}")),
        }
    }
    #[cfg(not(feature = "browser"))]
    {
        CheckResult::ok(
            "browser",
            "disabled (compile with --features browser to enable)",
        )
    }
}

fn check_config(fix: bool) -> CheckResult {
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
        Some(path) => {
            // Config file doesn't exist yet — this is normally an
            // informational "ok" state, but with --fix we can drop a
            // minimal stub so the next run has somewhere to grow from.
            if fix {
                match maybe_write_stub_config(&path) {
                    StubWriteResult::Wrote => {
                        return CheckResult::ok(
                            "config",
                            format!("created stub config at {}", path.display()),
                        )
                        .with_fix(format!("wrote stub config to {}", path.display()));
                    }
                    StubWriteResult::Skipped(reason) => {
                        return CheckResult::ok(
                            "config",
                            format!("no config yet at {} ({})", path.display(), reason),
                        );
                    }
                    StubWriteResult::Err(e) => {
                        return CheckResult::warn(
                            "config",
                            format!("could not write stub config: {}", e),
                        );
                    }
                }
            }
            CheckResult::ok("config", format!("no config yet at {}", path.display()))
        }
        None => CheckResult::warn("config", "could not resolve home directory"),
    }
}

enum StubWriteResult {
    Wrote,
    Skipped(&'static str),
    Err(std::io::Error),
}

fn maybe_write_stub_config(path: &std::path::Path) -> StubWriteResult {
    if path.exists() {
        return StubWriteResult::Skipped("already exists, skipping");
    }
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            return StubWriteResult::Err(e);
        }
    }
    let model = std::env::var("CUBI_MODEL")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "qwen3:8b".to_string());
    let stub = format!(
        "{{\n  \"default_model\": {}\n}}\n",
        serde_json::to_string(&model).unwrap_or_else(|_| "\"qwen3:8b\"".to_string())
    );
    match std::fs::write(path, stub) {
        Ok(()) => StubWriteResult::Wrote,
        Err(e) => StubWriteResult::Err(e),
    }
}

fn check_sessions_dir(fix: bool) -> CheckResult {
    let Some(dir) = crate::sessions::sessions_root() else {
        return CheckResult::warn("sessions_dir", "could not resolve home directory");
    };
    let existed = dir.exists();
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
            let r = CheckResult::ok("sessions_dir", format!("writable: {}", dir.display()));
            if fix && !existed {
                r.with_fix(format!("created {}", dir.display()))
            } else {
                r
            }
        }
        Err(e) => CheckResult::fail(
            "sessions_dir",
            format!("not writable ({}): {}", dir.display(), e),
        ),
    }
}

/// Optional check: ensure per-user shell completion scripts exist on
/// disk. We never edit shell rc files; we only write to
/// `~/.cubi/completions/` and print the rc-file snippet the user
/// should add. With `--fix` (and `CUBI_SHELL` / `$SHELL` indicating a
/// supported shell) we drop the file and prefix the line with `+`.
fn check_completions_installed(fix: bool) -> CheckResult {
    let Some(home) = crate::sessions::home_dir() else {
        return CheckResult::warn("completions", "could not resolve home directory");
    };
    let shell = detect_shell();
    completions_check_in_home(&home, shell.as_deref(), fix)
}

/// Inner logic for [`check_completions_installed`], factored out so
/// tests can drive it against a [`tempfile::TempDir`] without
/// mutating the process environment.
fn completions_check_in_home(
    home: &std::path::Path,
    shell: Option<&str>,
    fix: bool,
) -> CheckResult {
    let comp_dir = home.join(".cubi").join("completions");
    let (shell_name, file_name) = match shell {
        Some("zsh") => ("zsh", "_cubi"),
        Some("bash") => ("bash", "cubi.bash"),
        Some("fish") => ("fish", "cubi.fish"),
        Some(other) => {
            return CheckResult::ok(
                "completions",
                format!("unsupported shell '{}' (no completions)", other),
            );
        }
        None => {
            return CheckResult::ok(
                "completions",
                "shell not detected (set $SHELL or CUBI_SHELL to install)",
            );
        }
    };
    let target = comp_dir.join(file_name);
    if target.exists() {
        return CheckResult::ok(
            "completions",
            format!("{} completions present at {}", shell_name, target.display()),
        );
    }
    if !fix {
        return CheckResult::ok(
            "completions",
            format!(
                "{} completions not installed (run `cubi doctor --fix` to install)",
                shell_name
            ),
        );
    }
    let Some(script) = crate::completions::script(shell_name) else {
        return CheckResult::warn("completions", "no completion script for this shell");
    };
    if let Err(e) = std::fs::create_dir_all(&comp_dir) {
        return CheckResult::warn(
            "completions",
            format!("could not create {}: {}", comp_dir.display(), e),
        );
    }
    match std::fs::write(&target, script) {
        Ok(()) => CheckResult::ok(
            "completions",
            format!(
                "installed {} completions to {}",
                shell_name,
                target.display()
            ),
        )
        .with_fix(format!(
            "installed {} completions to {} — add `{}` to your shell rc",
            shell_name,
            target.display(),
            rc_snippet_for(shell_name, &comp_dir)
        )),
        Err(e) => CheckResult::warn(
            "completions",
            format!("could not write {}: {}", target.display(), e),
        ),
    }
}

fn detect_shell() -> Option<String> {
    if let Ok(s) = std::env::var("CUBI_SHELL") {
        if !s.is_empty() {
            return Some(s);
        }
    }
    let shell_path = std::env::var("SHELL").ok()?;
    let name = std::path::Path::new(&shell_path)
        .file_name()
        .and_then(|s| s.to_str())?
        .to_string();
    Some(name)
}

fn rc_snippet_for(shell: &str, dir: &std::path::Path) -> String {
    match shell {
        "zsh" => format!("fpath=({} $fpath)", dir.display()),
        "bash" => format!("source {}/cubi.bash", dir.display()),
        "fish" => format!("source {}/cubi.fish", dir.display()),
        _ => String::new(),
    }
}

async fn check_model_host() -> CheckResult {
    let config = AppConfig::load();
    let model = crate::onboarding::resolve_model(&config, "qwen3:8b");

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
        if let Some(fix) = &r.fix {
            println!("    {} {}", "+".bright_green(), fix);
        }
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

    #[test]
    fn stub_config_is_written_when_missing() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("nested").join("config.json");
        let r = maybe_write_stub_config(&path);
        assert!(matches!(r, StubWriteResult::Wrote));
        assert!(path.exists());
        // Parsed back as JSON with a default_model field.
        let raw = std::fs::read_to_string(&path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert!(v.get("default_model").is_some());
    }

    #[test]
    fn stub_config_refuses_to_overwrite_existing() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("config.json");
        std::fs::write(&path, "{\"keep\": true}").unwrap();
        let r = maybe_write_stub_config(&path);
        assert!(matches!(r, StubWriteResult::Skipped(_)));
        // Untouched.
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains("keep"));
    }

    #[test]
    fn completions_check_with_no_shell_returns_ok_without_writing() {
        let tmp = tempfile::TempDir::new().unwrap();
        let r = completions_check_in_home(tmp.path(), None, true);
        assert_eq!(r.status, CheckStatus::Ok);
        assert!(r.fix.is_none());
        assert!(!tmp.path().join(".cubi/completions").exists());
    }

    #[test]
    fn completions_check_without_fix_does_not_write() {
        let tmp = tempfile::TempDir::new().unwrap();
        let r = completions_check_in_home(tmp.path(), Some("zsh"), false);
        assert_eq!(r.status, CheckStatus::Ok);
        assert!(r.fix.is_none());
        assert!(!tmp.path().join(".cubi/completions/_cubi").exists());
    }

    #[test]
    fn completions_check_with_fix_writes_zsh_script() {
        let tmp = tempfile::TempDir::new().unwrap();
        let r = completions_check_in_home(tmp.path(), Some("zsh"), true);
        assert_eq!(r.status, CheckStatus::Ok);
        let fix = r.fix.expect("fix line should be set");
        assert!(fix.contains("installed zsh completions"));
        assert!(fix.contains("fpath="));
        let written = tmp.path().join(".cubi/completions/_cubi");
        assert!(written.exists());
        let body = std::fs::read_to_string(&written).unwrap();
        assert!(body.contains("#compdef cubi"));
    }

    #[test]
    fn completions_check_does_not_overwrite_existing() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path().join(".cubi/completions");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("_cubi");
        std::fs::write(&path, "preserved").unwrap();
        let r = completions_check_in_home(tmp.path(), Some("zsh"), true);
        assert_eq!(r.status, CheckStatus::Ok);
        assert!(r.fix.is_none(), "should not claim to have installed");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "preserved");
    }

    #[test]
    fn completions_check_with_unsupported_shell_is_ok_noop() {
        let tmp = tempfile::TempDir::new().unwrap();
        let r = completions_check_in_home(tmp.path(), Some("tcsh"), true);
        assert_eq!(r.status, CheckStatus::Ok);
        assert!(r.fix.is_none());
        assert!(r.message.contains("unsupported shell"));
    }

    #[test]
    fn rc_snippet_zsh_uses_fpath() {
        let s = rc_snippet_for("zsh", std::path::Path::new("/x/y"));
        assert_eq!(s, "fpath=(/x/y $fpath)");
    }

    #[test]
    fn check_result_serialization_omits_fix_when_none() {
        let r = CheckResult::ok("x", "ok");
        let s = serde_json::to_string(&r).unwrap();
        assert!(!s.contains("\"fix\""));
    }

    #[test]
    fn check_result_serialization_includes_fix_when_set() {
        let r = CheckResult::ok("x", "ok").with_fix("did the thing");
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains("\"fix\":\"did the thing\""));
    }
}
