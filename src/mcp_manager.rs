use crate::style::CubiStyle;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::builtin_tools::BuiltinToolRegistry;
use crate::file_rollback::FileJournal;
use crate::mcp_client::{McpClient, McpResource, McpResourceContent, Tool, ToolCallResult};
use crate::mcp_config::{McpConfig, McpServerConfig};

const MCP_STARTUP_TIMEOUT: Duration = Duration::from_millis(1500);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpHealthState {
    Ready,
    Failed(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpHealth {
    pub name: String,
    pub state: McpHealthState,
}

pub async fn with_mcp_startup_timeout<F, T>(future: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    match tokio::time::timeout(MCP_STARTUP_TIMEOUT, future).await {
        Ok(result) => result,
        Err(_) => anyhow::bail!("timed out after 1.5s"),
    }
}

pub fn format_health_line(health: &[McpHealth], color: bool) -> String {
    let ready = health
        .iter()
        .filter(|h| matches!(h.state, McpHealthState::Ready))
        .count();
    let failed: Vec<&str> = health
        .iter()
        .filter_map(|h| match &h.state {
            McpHealthState::Failed(_) => Some(h.name.as_str()),
            McpHealthState::Ready => None,
        })
        .collect();
    let failed_label = if failed.is_empty() {
        "".to_string()
    } else if failed.len() == 1 {
        format!("({})", failed[0])
    } else {
        format!("({}+{})", failed[0], failed.len() - 1)
    };

    let green = if color {
        "●".bright_green().to_string()
    } else {
        "●".to_string()
    };
    let red = if color {
        "●".bright_red().to_string()
    } else {
        "●".to_string()
    };
    let dim = if color {
        "●".bright_black().to_string()
    } else {
        "●".to_string()
    };
    let disabled = if color {
        "0(disabled)".bright_black().to_string()
    } else {
        "0(disabled)".to_string()
    };

    format!(
        "MCP: {green}{ready}  {red}{}{failed_label}  {dim}{disabled}",
        failed.len()
    )
}
use crate::oauth;
use crate::permissions::Permissions;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolTimeoutError {
    pub name: String,
    pub secs: u64,
}

impl std::fmt::Display for ToolTimeoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tool '{}' timed out after {}s", self.name, self.secs)
    }
}

impl std::error::Error for ToolTimeoutError {}

pub struct McpManager {
    clients: HashMap<String, McpClient>,
    tools: HashMap<String, (String, Tool)>, // tool_name -> (server_name, tool)
    builtin_tools: BuiltinToolRegistry,
    tool_timeout_secs: Option<u64>,
}

impl McpManager {
    pub async fn health_check_configured() -> Result<Vec<McpHealth>> {
        let config = McpConfig::load()?;
        let mut handles = Vec::new();
        for (name, server_config) in config.mcp_servers {
            handles.push(tokio::spawn(async move {
                let state = match with_mcp_startup_timeout(async {
                    let mut client = Self::connect_client(&server_config).await?;
                    client.list_tools().await?;
                    let _ = client.shutdown().await;
                    Ok(())
                })
                .await
                {
                    Ok(()) => McpHealthState::Ready,
                    Err(err) => McpHealthState::Failed(err.to_string()),
                };
                McpHealth { name, state }
            }));
        }

        let mut health = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(item) => health.push(item),
                Err(err) => health.push(McpHealth {
                    name: "unknown".to_string(),
                    state: McpHealthState::Failed(err.to_string()),
                }),
            }
        }
        health.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(health)
    }

    pub async fn new(
        permissions: Arc<Mutex<Permissions>>,
        plan_mode: Arc<AtomicBool>,
    ) -> Result<Self> {
        Self::new_with_journal(permissions, plan_mode, FileJournal::default()).await
    }

    /// Variant that wires an explicit [`FileJournal`] through to the
    /// built-in tool registry. The journal is shared with the CLI so
    /// `/rewind` can roll back `edit_file`/`write_file` mutations.
    pub async fn new_with_journal(
        permissions: Arc<Mutex<Permissions>>,
        plan_mode: Arc<AtomicBool>,
        journal: FileJournal,
    ) -> Result<Self> {
        Self::new_with_journal_quiet(permissions, plan_mode, journal, false).await
    }

    pub async fn new_with_journal_quiet(
        permissions: Arc<Mutex<Permissions>>,
        plan_mode: Arc<AtomicBool>,
        journal: FileJournal,
        quiet_stdout: bool,
    ) -> Result<Self> {
        let config = McpConfig::load()?;
        let mut manager = Self {
            clients: HashMap::new(),
            tools: HashMap::new(),
            builtin_tools: BuiltinToolRegistry::with_journal(permissions, plan_mode, journal),
            tool_timeout_secs: Some(60),
        };

        // Add built-in tools first
        for tool in manager.builtin_tools.list_tools() {
            let mcp_tool = Tool {
                name: tool.name.clone(),
                description: tool.description.clone(),
                input_schema: tool.input_schema.clone(),
            };
            manager
                .tools
                .insert(tool.name.clone(), ("builtin".to_string(), mcp_tool));
        }

        let builtins_msg = format!(
            "{} Loaded {} built-in tools",
            "✓".bright_green(),
            manager.builtin_tools.list_tools().len()
        );
        if quiet_stdout {
            eprintln!("{builtins_msg}");
        } else {
            println!("{builtins_msg}");
        }

        // Connect to configured MCP servers
        for (name, server_config) in config.mcp_servers {
            if let Err(e) =
                with_mcp_startup_timeout(manager.connect_server(&name, &server_config)).await
            {
                eprintln!(
                    "{} Failed to connect to MCP server '{}': {}",
                    "Warning:".bright_yellow(),
                    name,
                    e
                );
                continue;
            }
            let server_msg = format!(
                "{} Connected to MCP server: {}",
                "✓".bright_green(),
                name.bright_cyan()
            );
            if quiet_stdout {
                eprintln!("{server_msg}");
            } else {
                println!("{server_msg}");
            }
        }

        // Discover tools from external servers
        manager.discover_tools().await?;

        Ok(manager)
    }

    pub fn get_tools_with_server(&self) -> &HashMap<String, (String, Tool)> {
        &self.tools
    }

    pub fn set_tool_timeout_secs(&mut self, timeout_secs: Option<u64>) {
        self.tool_timeout_secs = timeout_secs;
    }

    pub async fn call_tool(
        &mut self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolCallResult> {
        let (server_name, _) = self
            .tools
            .get(name)
            .context(format!("Tool '{}' not found", name))?;

        let mut arguments = arguments;
        let override_timeout = strip_timeout_override(&mut arguments);
        let timeout_secs = override_timeout.or(self.tool_timeout_secs).unwrap_or(60);
        let timeout_duration = Duration::from_secs(timeout_secs);

        // Time the call so the opt-in telemetry log has useful numbers.
        let started = std::time::Instant::now();
        let server_name = server_name.clone();

        // Handle built-in tools
        if server_name == "builtin" {
            let result = tokio::time::timeout(
                timeout_duration,
                self.builtin_tools.execute(name, arguments),
            )
            .await;
            let result = match result {
                Ok(result) => result,
                Err(_) => {
                    crate::telemetry::record_tool_call(crate::telemetry::ToolCallEvent {
                        tool: name,
                        ok: false,
                        duration_ms: started.elapsed().as_millis() as u64,
                    });
                    return Err(ToolTimeoutError {
                        name: name.to_string(),
                        secs: timeout_secs,
                    }
                    .into());
                }
            };
            crate::telemetry::record_tool_call(crate::telemetry::ToolCallEvent {
                tool: name,
                ok: result
                    .as_ref()
                    .map(|r| r.is_error != Some(true))
                    .unwrap_or(false),
                duration_ms: started.elapsed().as_millis() as u64,
            });
            let result = result?;
            return Ok(ToolCallResult {
                content: result
                    .content
                    .into_iter()
                    .map(|c| crate::mcp_client::Content {
                        content_type: c.content_type,
                        text: c.text,
                    })
                    .collect(),
                is_error: result.is_error,
            });
        }

        // Handle external MCP server tools
        let client = self
            .clients
            .get_mut(&server_name)
            .context(format!("Server '{}' not connected", server_name))?;

        let result =
            tokio::time::timeout(timeout_duration, client.call_tool(name, arguments)).await;
        let result = match result {
            Ok(result) => result,
            Err(_) => {
                crate::telemetry::record_tool_call(crate::telemetry::ToolCallEvent {
                    tool: name,
                    ok: false,
                    duration_ms: started.elapsed().as_millis() as u64,
                });
                return Err(ToolTimeoutError {
                    name: name.to_string(),
                    secs: timeout_secs,
                }
                .into());
            }
        };
        crate::telemetry::record_tool_call(crate::telemetry::ToolCallEvent {
            tool: name,
            ok: result
                .as_ref()
                .map(|r| r.is_error != Some(true))
                .unwrap_or(false),
            duration_ms: started.elapsed().as_millis() as u64,
        });
        result
    }

    async fn connect_server(&mut self, name: &str, config: &McpServerConfig) -> Result<()> {
        let client = Self::connect_client(config).await?;
        self.clients.insert(name.to_string(), client);
        Ok(())
    }

    async fn connect_client(config: &McpServerConfig) -> Result<McpClient> {
        if config.is_stdio() {
            let command = config.command.clone().unwrap();
            let args = config.args.clone().unwrap_or_default();
            let env = config.env.clone().unwrap_or_default();

            McpClient::connect_stdio(command, args, env).await
        } else if config.is_http() {
            let url = config.http_url.clone().unwrap();
            let mut headers = config.headers.clone().unwrap_or_default();
            if let Some(provider) = config.oauth_provider.as_deref() {
                let has_authorization = headers
                    .keys()
                    .any(|k| k.eq_ignore_ascii_case("authorization"));
                if !has_authorization {
                    match oauth::bearer_header_for_provider(provider)? {
                        Some(header) => {
                            headers.insert("Authorization".to_string(), header);
                        }
                        None => {
                            eprintln!(
                                "{} No usable OAuth token for MCP provider '{}'; continuing without Authorization header.",
                                "Warning:".bright_yellow(),
                                provider
                            );
                        }
                    }
                }
            }

            McpClient::connect_http(url, headers).await
        } else {
            anyhow::bail!("Server configuration must specify either command or httpUrl");
        }
    }

    async fn discover_tools(&mut self) -> Result<()> {
        for (server_name, client) in &mut self.clients {
            match client.list_tools().await {
                Ok(tools) => {
                    for tool in tools {
                        self.tools
                            .insert(tool.name.clone(), (server_name.clone(), tool));
                    }
                }
                Err(e) => {
                    eprintln!(
                        "{} Failed to list tools from '{}': {}",
                        "Warning:".bright_yellow(),
                        server_name,
                        e
                    );
                }
            }
        }

        Ok(())
    }

    pub fn list_tools(&self) -> Vec<&Tool> {
        self.tools.values().map(|(_, tool)| tool).collect()
    }

    pub fn has_tools(&self) -> bool {
        !self.tools.is_empty()
    }

    pub async fn list_resources(&mut self) -> Result<Vec<(String, McpResource)>> {
        let mut out = Vec::new();
        for (server_name, client) in &mut self.clients {
            match client.list_resources().await {
                Ok(resources) => {
                    for resource in resources {
                        out.push((server_name.clone(), resource));
                    }
                }
                Err(e) => {
                    eprintln!(
                        "{} Failed to list resources from '{}': {}",
                        "Warning:".bright_yellow(),
                        server_name,
                        e
                    );
                }
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.uri.cmp(&b.1.uri)));
        Ok(out)
    }

    pub async fn read_resource(
        &mut self,
        server: &str,
        uri: &str,
    ) -> Result<Vec<McpResourceContent>> {
        let client = self
            .clients
            .get_mut(server)
            .context(format!("Server '{}' not connected", server))?;
        client.read_resource(uri).await
    }

    /// Aggregate `prompts/list` over every connected MCP server.
    /// Per-server failures are logged as warnings but do not abort the
    /// whole call — partial results are still useful.
    pub async fn list_prompts(&mut self) -> Result<Vec<(String, String, String)>> {
        let mut out = Vec::new();
        for (server_name, client) in &mut self.clients {
            match client.list_prompts().await {
                Ok(prompts) => {
                    for p in prompts {
                        let desc = p.description.clone().unwrap_or_default();
                        out.push((server_name.clone(), p.name, desc));
                    }
                }
                Err(e) => {
                    eprintln!(
                        "{} Failed to list prompts from '{}': {}",
                        "Warning:".bright_yellow(),
                        server_name,
                        e
                    );
                }
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        Ok(out)
    }

    pub async fn get_prompt(&mut self, server: &str, name: &str) -> Result<String> {
        let client = self
            .clients
            .get_mut(server)
            .context(format!("Server '{}' not connected", server))?;
        client.get_prompt(name).await
    }

    pub async fn shutdown(&mut self) {
        for (name, client) in &mut self.clients {
            if let Err(e) = client.shutdown().await {
                eprintln!("Failed to shutdown MCP server '{}': {}", name, e);
            }
        }
    }
}

fn strip_timeout_override(arguments: &mut serde_json::Value) -> Option<u64> {
    let object = arguments.as_object_mut()?;
    object
        .remove("_timeout_secs")
        .and_then(|value| value.as_u64())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn startup_timeout_fires_for_slow_future() {
        let started = std::time::Instant::now();
        let result = with_mcp_startup_timeout(async {
            tokio::time::sleep(Duration::from_secs(5)).await;
            Ok(())
        })
        .await;
        assert!(result.is_err());
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    #[test]
    fn strips_timeout_override_from_arguments() {
        let mut args = serde_json::json!({"command": "echo ok", "_timeout_secs": 2});
        assert_eq!(strip_timeout_override(&mut args), Some(2));
        assert_eq!(args, serde_json::json!({"command": "echo ok"}));
    }
}
