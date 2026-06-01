//! CLI/REPL entry points for `cubi mcp search/install/uninstall`.
//!
//! Split out from `main.rs` so the REPL slash commands (`/mcp-search`,
//! `/mcp-install`, `/mcp-uninstall` in `src/commands.rs`) can call the
//! same logic without duplicating it. The headless `cubi mcp ...`
//! subcommands and the slash-command handlers both funnel through these
//! functions.
use crate::mcp_config::McpConfig;
use crate::mcp_manager::McpManager;
use crate::mcp_registry::{self, RegistryEntry};
use std::collections::BTreeMap;

/// Implements `cubi mcp search [<query>] [--json]`. Returns a process
/// exit code (`0` on hits, `1` on no match, `2` on usage error — we
/// have none here yet).
pub fn run_mcp_search(query: &str, json: bool) -> i32 {
    let results = mcp_registry::search(query);
    if json {
        let arr: Vec<_> = results
            .iter()
            .map(|e| {
                serde_json::json!({
                    "name": e.name,
                    "description": e.description,
                    "transport": e.transport_label(),
                    "homepage": e.homepage,
                    "license": e.license,
                    "tags": e.tags,
                })
            })
            .collect();
        println!("{}", serde_json::Value::Array(arr));
        return 0;
    }
    if results.is_empty() {
        eprintln!("cubi: no MCP servers matched '{}'.", query);
        return 1;
    }
    let name_w = results
        .iter()
        .map(|e| e.name.len())
        .max()
        .unwrap_or(4)
        .max(4);
    let tp_w = 6;
    println!(
        "{:<name_w$}  {:<tp_w$}  DESCRIPTION",
        "NAME",
        "TRANS",
        name_w = name_w,
        tp_w = tp_w
    );
    for e in &results {
        let desc = if e.description.chars().count() > 60 {
            let truncated: String = e.description.chars().take(59).collect();
            format!("{truncated}…")
        } else {
            e.description.clone()
        };
        println!(
            "{:<name_w$}  {:<tp_w$}  {}",
            e.name,
            e.transport_label(),
            desc,
            name_w = name_w,
            tp_w = tp_w
        );
        if !e.tags.is_empty() {
            println!(
                "{:<name_w$}  {:<tp_w$}    tags: {}",
                "",
                "",
                e.tags.join(", "),
                name_w = name_w,
                tp_w = tp_w
            );
        }
    }
    0
}

/// Prompt the user (interactively) for the required env vars declared by
/// `entry`. `preset` carries values supplied non-interactively via
/// `--env K=V`. Returns `None` when stdin is not a TTY and a required
/// var is still missing — callers print their own follow-up message.
fn prompt_env_vars(
    entry: &RegistryEntry,
    preset: &BTreeMap<String, String>,
) -> Option<BTreeMap<String, String>> {
    use std::io::{self, IsTerminal, Write};
    let mut out: BTreeMap<String, String> = preset.clone();
    let interactive = io::stdin().is_terminal();
    for (key, spec) in &entry.env {
        if out.contains_key(key) {
            continue;
        }
        if !spec.required {
            continue;
        }
        if !interactive {
            eprintln!(
                "cubi: missing required env '{}' for '{}' and no TTY for prompting.\n  hint: pass --env {}=<value>",
                key, entry.name, key
            );
            return None;
        }
        println!("  {} — {}", key, spec.description);
        print!("  {} = ", key);
        let _ = io::stdout().flush();
        let mut buf = String::new();
        if io::stdin().read_line(&mut buf).is_err() {
            eprintln!("cubi: could not read env value for {}.", key);
            return None;
        }
        let v = buf.trim().to_string();
        if v.is_empty() {
            eprintln!("cubi: '{}' is required; aborting.", key);
            return None;
        }
        out.insert(key.clone(), v);
    }
    Some(out)
}

/// Connect to `server` via the existing `mcp test` path and issue a single
/// `tools/list` round-trip. Returns `Ok(true)` if the server returned at
/// least one tool, `Ok(false)` if it returned zero, and `Err` on any
/// connect / RPC failure. Used by `install` to validate the entry we just
/// wrote without printing the per-tool noise `cubi mcp test` emits.
pub async fn validate_server(server: &str) -> anyhow::Result<bool> {
    let config = McpConfig::load()?;
    let server_config = config
        .mcp_servers
        .get(server)
        .ok_or_else(|| anyhow::anyhow!("no MCP server named '{server}'"))?;
    let mut client = McpManager::connect_for_test(server_config).await?;
    let tools = client.list_tools().await;
    let _ = client.shutdown().await;
    Ok(!tools?.is_empty())
}

/// Implements `cubi mcp install <name> [--force] [--env K=V]... [--json]`.
pub async fn run_mcp_install(
    name: &str,
    force: bool,
    json: bool,
    envs: &[(String, String)],
) -> i32 {
    let Some(entry) = mcp_registry::find(name) else {
        if json {
            println!(
                "{}",
                serde_json::json!({
                    "status": "failed",
                    "error": format!("unknown server '{}'", name),
                })
            );
        } else {
            eprintln!(
                "cubi: no registry entry named '{}'. Try `cubi mcp search`.",
                name
            );
        }
        return 2;
    };
    let mut config = match McpConfig::load() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("cubi: could not load MCP config: {e}");
            return 2;
        }
    };
    if config.mcp_servers.contains_key(&entry.name) && !force {
        if json {
            println!(
                "{}",
                serde_json::json!({
                    "status": "failed",
                    "error": format!("'{}' is already configured; pass --force to overwrite", entry.name),
                    "configured": true,
                })
            );
        } else {
            eprintln!(
                "cubi: '{}' is already configured in ~/.cubi/mcp.json. Re-run with --force to overwrite.",
                entry.name
            );
        }
        return 2;
    }
    let preset: BTreeMap<String, String> = envs.iter().cloned().collect();
    let Some(env_values) = prompt_env_vars(entry, &preset) else {
        if json {
            println!(
                "{}",
                serde_json::json!({"status": "failed", "error": "missing required env vars"})
            );
        }
        return 2;
    };
    let server_config = entry.to_server_config(&env_values);
    config.mcp_servers.insert(entry.name.clone(), server_config);
    if let Err(e) = config.save() {
        eprintln!("cubi: failed to save MCP config: {e}");
        return 2;
    }
    let provided: Vec<String> = env_values.keys().cloned().collect();
    if !json {
        println!(
            "✓ wrote ~/.cubi/mcp.json entry '{}'. Validating…",
            entry.name
        );
    }
    match validate_server(&entry.name).await {
        Ok(_) => {
            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "status": "ok",
                        "name": entry.name,
                        "configured": true,
                        "env_provided": provided,
                    })
                );
            } else {
                println!("✓ '{}' installed and validated.", entry.name);
            }
            0
        }
        Err(e) => {
            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "status": "failed",
                        "name": entry.name,
                        "configured": true,
                        "env_provided": provided,
                        "error": format!("validation failed: {:#}", e),
                    })
                );
            } else {
                eprintln!(
                    "cubi: '{}' validation failed: {:#}\n  entry left in ~/.cubi/mcp.json; fix env vars or args and re-run `cubi mcp test {}`.",
                    entry.name, e, entry.name
                );
            }
            11
        }
    }
}

/// Implements `cubi mcp uninstall <name> [--json]`.
pub fn run_mcp_uninstall(name: &str, json: bool) -> i32 {
    let mut config = match McpConfig::load() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("cubi: could not load MCP config: {e}");
            return 2;
        }
    };
    let removed = config.remove_server(name);
    if !removed {
        if json {
            println!(
                "{}",
                serde_json::json!({
                    "status": "failed",
                    "error": format!("'{}' not configured", name),
                })
            );
        } else {
            eprintln!("cubi: no MCP server named '{}' is configured.", name);
        }
        return 2;
    }
    if let Err(e) = config.save() {
        eprintln!("cubi: failed to save MCP config: {e}");
        return 2;
    }
    if json {
        println!("{}", serde_json::json!({"status": "ok", "removed": name}));
    } else {
        println!("✓ removed '{}' from ~/.cubi/mcp.json", name);
    }
    0
}
