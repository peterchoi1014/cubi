use anyhow::{Context, Result};
use colored::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::builtin_tools::BuiltinToolRegistry;
use crate::mcp_client::{McpClient, Tool, ToolCallResult};
use crate::mcp_config::{McpConfig, McpServerConfig};
use crate::permissions::Permissions;

pub struct McpManager {
    clients: HashMap<String, McpClient>,
    tools: HashMap<String, (String, Tool)>, // tool_name -> (server_name, tool)
    builtin_tools: BuiltinToolRegistry,
}

impl McpManager {
    pub async fn new(permissions: Arc<Mutex<Permissions>>) -> Result<Self> {
        let config = McpConfig::load()?;
        let mut manager = Self {
            clients: HashMap::new(),
            tools: HashMap::new(),
            builtin_tools: BuiltinToolRegistry::new(permissions),
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

        println!(
            "{} Loaded {} built-in tools",
            "✓".bright_green(),
            manager.builtin_tools.list_tools().len()
        );

        // Connect to configured MCP servers
        for (name, server_config) in config.mcp_servers {
            if let Err(e) = manager.connect_server(&name, &server_config).await {
                eprintln!(
                    "{} Failed to connect to MCP server '{}': {}",
                    "Warning:".bright_yellow(),
                    name,
                    e
                );
                continue;
            }
            println!(
                "{} Connected to MCP server: {}",
                "✓".bright_green(),
                name.bright_cyan()
            );
        }

        // Discover tools from external servers
        manager.discover_tools().await?;

        Ok(manager)
    }

    pub fn get_tools_with_server(&self) -> &HashMap<String, (String, Tool)> {
        &self.tools
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

        // Handle built-in tools
        if server_name == "builtin" {
            let result = self.builtin_tools.execute(name, arguments).await?;

            // Convert BuiltinToolResult to ToolCallResult
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
            .get_mut(server_name)
            .context(format!("Server '{}' not connected", server_name))?;

        client.call_tool(name, arguments).await
    }

    async fn connect_server(&mut self, name: &str, config: &McpServerConfig) -> Result<()> {
        let client = if config.is_stdio() {
            let command = config.command.clone().unwrap();
            let args = config.args.clone().unwrap_or_default();
            let env = config.env.clone().unwrap_or_default();

            McpClient::connect_stdio(command, args, env).await?
        } else if config.is_http() {
            let url = config.http_url.clone().unwrap();
            let headers = config.headers.clone().unwrap_or_default();

            McpClient::connect_http(url, headers).await?
        } else {
            anyhow::bail!("Server configuration must specify either command or httpUrl");
        };

        self.clients.insert(name.to_string(), client);
        Ok(())
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

    pub async fn shutdown(&mut self) {
        for (name, client) in &mut self.clients {
            if let Err(e) = client.shutdown().await {
                eprintln!("Failed to shutdown MCP server '{}': {}", name, e);
            }
        }
    }
}
