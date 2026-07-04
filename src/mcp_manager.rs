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
    /// Servers we've marked offline mid-session (reconnect attempts
    /// exhausted). Their tools are removed from [`Self::tools`] so the
    /// agent loop stops advertising them, but the entry stays around
    /// so `/mcp` can surface the failure to the user and the REPL turn
    /// can continue without killing the process.
    failed_servers: HashMap<String, String>,
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
            failed_servers: HashMap::new(),
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
        crate::out::status_line(quiet_stdout, builtins_msg);

        // Connect to configured MCP servers
        for (name, server_config) in Self::connectable_servers(&config) {
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

    /// Servers that should be connected on load: enabled ones only, sorted by
    /// name for deterministic connect order. Disabled servers are intentionally
    /// skipped and therefore never counted as connection failures.
    fn connectable_servers(config: &McpConfig) -> Vec<(String, McpServerConfig)> {
        let mut servers: Vec<(String, McpServerConfig)> = config
            .mcp_servers
            .iter()
            .filter(|(_, cfg)| cfg.enabled)
            .map(|(name, cfg)| (name.clone(), cfg.clone()))
            .collect();
        servers.sort_by(|a, b| a.0.cmp(&b.0));
        servers
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
        // `None` (or `Some(0)`) explicitly disables the wall-clock timeout
        // and lets the tool run to completion; anything > 0 wraps the
        // call with `tokio::time::timeout` and surfaces ToolTimeoutError
        // on expiry.
        let configured = override_timeout.or(self.tool_timeout_secs);
        let timeout_secs: Option<u64> = configured.filter(|&n| n > 0);

        // Time the call so the opt-in telemetry log has useful numbers.
        let started = std::time::Instant::now();
        let server_name = server_name.clone();

        // Handle built-in tools
        if server_name == "builtin" {
            let result = match timeout_secs {
                Some(secs) => match tokio::time::timeout(
                    Duration::from_secs(secs),
                    self.builtin_tools.execute(name, arguments),
                )
                .await
                {
                    Ok(r) => r,
                    Err(_) => {
                        crate::telemetry::record_tool_call(crate::telemetry::ToolCallEvent {
                            tool: name,
                            ok: false,
                            duration_ms: started.elapsed().as_millis() as u64,
                        });
                        return Err(ToolTimeoutError {
                            name: name.to_string(),
                            secs,
                        }
                        .into());
                    }
                },
                None => self.builtin_tools.execute(name, arguments).await,
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
        // Clone args so we can retry once on transport death.
        let retry_args = arguments.clone();

        let client = self
            .clients
            .get_mut(&server_name)
            .context(format!("Server '{}' not connected", server_name))?;

        let result = match timeout_secs {
            Some(secs) => match tokio::time::timeout(
                Duration::from_secs(secs),
                client.call_tool(name, arguments),
            )
            .await
            {
                Ok(r) => r,
                Err(_) => {
                    crate::telemetry::record_tool_call(crate::telemetry::ToolCallEvent {
                        tool: name,
                        ok: false,
                        duration_ms: started.elapsed().as_millis() as u64,
                    });
                    return Err(ToolTimeoutError {
                        name: name.to_string(),
                        secs,
                    }
                    .into());
                }
            },
            None => client.call_tool(name, arguments).await,
        };

        // Auto-reconnect once for stdio servers whose transport looks
        // dead (broken pipe / closed stream). Drop the dead client first
        // so the borrow on self.clients ends before we mutate. http
        // servers don't get this treatment because reqwest already
        // tears down the connection on each call.
        let result = match result {
            Ok(ok) => Ok(ok),
            Err(err) if is_transport_dead(&err) => {
                tracing::warn!(
                    target: "cubi::mcp",
                    server = %server_name,
                    error = %err,
                    "stdio MCP transport looks dead; reconnecting and retrying once"
                );
                self.clients.remove(&server_name);
                match Self::reconnect_stdio(&server_name).await {
                    Ok(new_client) => {
                        self.clients.insert(server_name.clone(), new_client);
                        let client = self.clients.get_mut(&server_name).expect("just inserted");
                        let retry_result = match timeout_secs {
                            Some(secs) => match tokio::time::timeout(
                                Duration::from_secs(secs),
                                client.call_tool(name, retry_args),
                            )
                            .await
                            {
                                Ok(r) => r,
                                Err(_) => {
                                    crate::telemetry::record_tool_call(
                                        crate::telemetry::ToolCallEvent {
                                            tool: name,
                                            ok: false,
                                            duration_ms: started.elapsed().as_millis() as u64,
                                        },
                                    );
                                    return Err(ToolTimeoutError {
                                        name: name.to_string(),
                                        secs,
                                    }
                                    .into());
                                }
                            },
                            None => client.call_tool(name, retry_args).await,
                        };
                        match retry_result {
                            Ok(ok) => Ok(ok),
                            Err(retry_err) => {
                                tracing::warn!(
                                    target: "cubi::mcp",
                                    server = %server_name,
                                    retry_error = %retry_err,
                                    "MCP retry after reconnect failed; surfacing original error"
                                );
                                Err(err)
                            }
                        }
                    }
                    Err(reconnect_err) => {
                        tracing::warn!(
                            target: "cubi::mcp",
                            server = %server_name,
                            error = %reconnect_err,
                            "MCP reconnect failed; degrading server to Failed state"
                        );
                        let reason = format!("{}", reconnect_err);
                        self.mark_server_failed(&server_name, &reason);
                        crate::user_error::print_user_warning(
                            &format!(
                                "MCP server '{}' is offline; its tools have been disabled \
                                 for this session. Restart cubi or `/mcp-reload` to retry.",
                                server_name
                            ),
                            Some(&format!("reason: {}", reason)),
                            false,
                        );
                        // Convert the kill-the-turn error into a soft
                        // tool-error result so the REPL can keep going.
                        Ok(crate::mcp_client::ToolCallResult {
                            content: vec![crate::mcp_client::Content {
                                content_type: "text".to_string(),
                                text: format!(
                                    "MCP server '{}' is offline ({}); tool '{}' is unavailable for this session.",
                                    server_name, reason, name
                                ),
                            }],
                            is_error: Some(true),
                        })
                    }
                }
            }
            Err(err) => Err(err),
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

    /// Re-loads the MCP config and reconnects the named server if its
    /// configuration is still a stdio transport. Returns an error
    /// otherwise — callers should not invoke this for http servers.
    async fn reconnect_stdio(name: &str) -> Result<McpClient> {
        let config = McpConfig::load()?;
        let server_config = config
            .mcp_servers
            .get(name)
            .with_context(|| format!("MCP server '{}' no longer in mcp.json", name))?
            .clone();
        if !server_config.is_stdio() {
            anyhow::bail!("server '{}' is not stdio; refusing to reconnect", name);
        }
        with_mcp_startup_timeout(Self::connect_client(&server_config)).await
    }

    async fn connect_server(&mut self, name: &str, config: &McpServerConfig) -> Result<()> {
        tracing::debug!(target: "cubi::mcp", server = %name, "connecting to MCP server");
        let client = Self::connect_client(config).await?;
        self.clients.insert(name.to_string(), client);
        tracing::debug!(target: "cubi::mcp", server = %name, "MCP server connected");
        Ok(())
    }

    /// Connect a one-shot client to a single configured server. Used by
    /// `cubi mcp test` to talk to a server without standing up the full
    /// manager (which would also load every other configured server and
    /// the built-in tool set).
    pub async fn connect_for_test(config: &McpServerConfig) -> Result<McpClient> {
        with_mcp_startup_timeout(Self::connect_client(config)).await
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

    /// Returns the list of MCP servers marked offline mid-session.
    /// Each entry is `(server_name, reason)`. Stable order by name.
    pub fn failed_servers(&self) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = self
            .failed_servers
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Mark an MCP server as offline for the rest of this session: the
    /// reason is stored for `/mcp` to surface and every tool that
    /// belongs to that server is removed from the active tool list so
    /// the agent loop stops advertising it.
    pub fn mark_server_failed(&mut self, server: &str, reason: &str) {
        self.failed_servers
            .insert(server.to_string(), reason.to_string());
        self.tools.retain(|_, (owner, _)| owner != server);
        self.clients.remove(server);
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
                tracing::warn!(target: "cubi::mcp", server = %name, error = %e, "failed to shutdown MCP server");
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

/// Returns `true` when the error chain looks like the underlying
/// transport went away (broken pipe / closed stream / connection lost)
/// rather than the tool itself returning a failure. Used to decide
/// whether reconnecting and retrying once is appropriate.
pub fn is_transport_dead(err: &anyhow::Error) -> bool {
    // Walk the cause chain and look for io::ErrorKind::BrokenPipe /
    // UnexpectedEof, or any sub-error whose Display contains markers
    // that mcp_client surfaces on a dead stdio peer.
    for cause in err.chain() {
        if let Some(io_err) = cause.downcast_ref::<std::io::Error>() {
            match io_err.kind() {
                std::io::ErrorKind::BrokenPipe
                | std::io::ErrorKind::UnexpectedEof
                | std::io::ErrorKind::ConnectionAborted
                | std::io::ErrorKind::ConnectionReset => return true,
                _ => {}
            }
        }
        let msg = cause.to_string().to_ascii_lowercase();
        if msg.contains("broken pipe")
            || msg.contains("connection lost")
            || msg.contains("connection closed")
            || msg.contains("stream closed")
            || msg.contains("channel closed")
            || msg.contains("unexpected end of file")
            || msg.contains("server exited")
        {
            return true;
        }
    }
    false
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

    #[test]
    fn is_transport_dead_detects_broken_pipe_io_kind() {
        let io_err = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pipe gone");
        let err: anyhow::Error = io_err.into();
        assert!(is_transport_dead(&err));
    }

    #[test]
    fn is_transport_dead_detects_unexpected_eof() {
        let io_err = std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "eof");
        let err: anyhow::Error = io_err.into();
        assert!(is_transport_dead(&err));
    }

    #[test]
    fn is_transport_dead_detects_text_markers() {
        for msg in [
            "Server exited unexpectedly",
            "broken pipe",
            "connection lost while reading frame",
            "stream closed",
        ] {
            let err: anyhow::Error = anyhow::anyhow!(msg.to_string());
            assert!(is_transport_dead(&err), "expected {msg:?} to be dead");
        }
    }

    #[test]
    fn is_transport_dead_returns_false_for_application_errors() {
        let err: anyhow::Error = anyhow::anyhow!("tool 'bash' returned non-zero exit code");
        assert!(!is_transport_dead(&err));
        let err = anyhow::anyhow!("invalid arguments: missing field 'command'");
        assert!(!is_transport_dead(&err));
    }

    #[test]
    fn connectable_servers_skips_disabled() {
        use crate::mcp_config::McpServerConfig;
        let mk = |enabled: bool| McpServerConfig {
            command: Some("echo".to_string()),
            args: None,
            env: None,
            http_url: None,
            headers: None,
            oauth_provider: None,
            enabled,
        };
        let mut servers = HashMap::new();
        servers.insert("on".to_string(), mk(true));
        servers.insert("off".to_string(), mk(false));
        let config = McpConfig {
            mcp_servers: servers,
        };
        let connectable = McpManager::connectable_servers(&config);
        let names: Vec<&str> = connectable.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["on"], "disabled server must be skipped");
    }

    /// Build an empty manager (no clients, no MCP config loaded) so
    /// the failed-server bookkeeping can be exercised in isolation.
    fn empty_manager() -> McpManager {
        let permissions = Arc::new(Mutex::new(crate::permissions::Permissions::default()));
        let plan_mode = Arc::new(AtomicBool::new(false));
        let journal = FileJournal::default();
        let builtin_tools = BuiltinToolRegistry::with_journal(permissions, plan_mode, journal);
        McpManager {
            clients: HashMap::new(),
            tools: HashMap::new(),
            failed_servers: HashMap::new(),
            builtin_tools,
            tool_timeout_secs: Some(60),
        }
    }

    fn fake_tool(name: &str) -> Tool {
        Tool {
            name: name.to_string(),
            description: format!("desc for {}", name),
            input_schema: serde_json::json!({}),
        }
    }

    #[test]
    fn mark_server_failed_removes_tools_and_records_reason() {
        let mut mgr = empty_manager();
        mgr.tools.insert(
            "toolA".to_string(),
            ("serverX".to_string(), fake_tool("toolA")),
        );
        mgr.tools.insert(
            "toolB".to_string(),
            ("serverX".to_string(), fake_tool("toolB")),
        );
        mgr.tools.insert(
            "toolC".to_string(),
            ("serverY".to_string(), fake_tool("toolC")),
        );
        assert_eq!(mgr.list_tools().len(), 3);

        mgr.mark_server_failed("serverX", "broken pipe");

        // Only serverY's tool survives.
        let names: Vec<&str> = mgr.list_tools().iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["toolC"]);

        // Failed list contains serverX with the recorded reason.
        let failed = mgr.failed_servers();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].0, "serverX");
        assert_eq!(failed[0].1, "broken pipe");
    }

    #[test]
    fn mark_server_failed_is_idempotent_and_sorted() {
        let mut mgr = empty_manager();
        mgr.mark_server_failed("zeta", "boom");
        mgr.mark_server_failed("alpha", "kaboom");
        mgr.mark_server_failed("alpha", "updated reason");
        let failed = mgr.failed_servers();
        assert_eq!(failed.len(), 2);
        assert_eq!(failed[0].0, "alpha");
        assert_eq!(failed[0].1, "updated reason");
        assert_eq!(failed[1].0, "zeta");
    }

    #[test]
    fn empty_manager_reports_no_failed_servers() {
        let mgr = empty_manager();
        assert!(mgr.failed_servers().is_empty());
    }
}
