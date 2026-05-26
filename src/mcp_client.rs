use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResource {
    pub uri: String,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(rename = "mimeType", default)]
    pub mime_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResourceContent {
    pub uri: String,
    #[serde(rename = "mimeType", default)]
    pub mime_type: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPrompt {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResult {
    pub content: Vec<Content>,
    #[serde(rename = "isError", skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Content {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

#[derive(Debug)]
pub enum McpClient {
    Stdio(StdioClient),
    Http(HttpClient),
}

impl McpClient {
    pub async fn connect_stdio(
        command: String,
        args: Vec<String>,
        env: HashMap<String, String>,
    ) -> Result<Self> {
        let client = StdioClient::new(command, args, env).await?;
        Ok(McpClient::Stdio(client))
    }

    pub async fn connect_http(url: String, headers: HashMap<String, String>) -> Result<Self> {
        let client = HttpClient::new(url, headers).await?;
        Ok(McpClient::Http(client))
    }

    pub async fn list_tools(&mut self) -> Result<Vec<Tool>> {
        match self {
            McpClient::Stdio(client) => client.list_tools().await,
            McpClient::Http(client) => client.list_tools().await,
        }
    }

    pub async fn call_tool(
        &mut self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolCallResult> {
        match self {
            McpClient::Stdio(client) => client.call_tool(name, arguments).await,
            McpClient::Http(client) => client.call_tool(name, arguments).await,
        }
    }

    pub async fn list_resources(&mut self) -> Result<Vec<McpResource>> {
        match self {
            McpClient::Stdio(client) => client.list_resources().await,
            McpClient::Http(client) => client.list_resources().await,
        }
    }

    pub async fn read_resource(&mut self, uri: &str) -> Result<Vec<McpResourceContent>> {
        match self {
            McpClient::Stdio(client) => client.read_resource(uri).await,
            McpClient::Http(client) => client.read_resource(uri).await,
        }
    }

    pub async fn list_prompts(&mut self) -> Result<Vec<McpPrompt>> {
        match self {
            McpClient::Stdio(client) => client.list_prompts().await,
            McpClient::Http(client) => client.list_prompts().await,
        }
    }

    /// Render a prompt to its concatenated user-message text.
    pub async fn get_prompt(&mut self, name: &str) -> Result<String> {
        match self {
            McpClient::Stdio(client) => client.get_prompt(name).await,
            McpClient::Http(client) => client.get_prompt(name).await,
        }
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        match self {
            McpClient::Stdio(client) => client.shutdown().await,
            McpClient::Http(_) => Ok(()),
        }
    }
}

// STDIO Client Implementation
#[derive(Debug)]
pub struct StdioClient {
    process: Child,
    request_id: u64,
}

impl StdioClient {
    async fn new(command: String, args: Vec<String>, env: HashMap<String, String>) -> Result<Self> {
        let mut cmd = Command::new(&command);
        cmd.args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            // The startup health check wraps connect calls in
            // `tokio::time::timeout`; on timeout the connect future is
            // dropped. `kill_on_drop` ensures any spawned MCP server
            // child process is reaped instead of leaking.
            .kill_on_drop(true);

        for (key, value) in env {
            cmd.env(key, value);
        }

        let process = cmd
            .spawn()
            .context(format!("Failed to spawn MCP server: {}", command))?;

        let mut client = Self {
            process,
            request_id: 1,
        };

        // Initialize connection
        client.initialize().await?;

        Ok(client)
    }

    async fn initialize(&mut self) -> Result<()> {
        let init_request = json!({
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {
                    "name": "cubi",
                    "version": "0.3.0"
                }
            }
        });

        self.send_request(init_request).await?;
        self.request_id += 1;

        // Send initialized notification
        let initialized = json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        });

        self.send_notification(initialized).await?;

        Ok(())
    }

    async fn send_request(&mut self, request: serde_json::Value) -> Result<serde_json::Value> {
        let stdin = self.process.stdin.as_mut().context("Failed to get stdin")?;

        let request_str = serde_json::to_string(&request)?;
        stdin.write_all(request_str.as_bytes()).await?;
        stdin.write_all(b"\n").await?;
        stdin.flush().await?;

        // Read response
        let stdout = self
            .process
            .stdout
            .as_mut()
            .context("Failed to get stdout")?;

        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        reader.read_line(&mut line).await?;

        let response: serde_json::Value =
            serde_json::from_str(&line).context("Failed to parse MCP response")?;

        Ok(response)
    }

    async fn send_notification(&mut self, notification: serde_json::Value) -> Result<()> {
        let stdin = self.process.stdin.as_mut().context("Failed to get stdin")?;

        let notification_str = serde_json::to_string(&notification)?;
        stdin.write_all(notification_str.as_bytes()).await?;
        stdin.write_all(b"\n").await?;
        stdin.flush().await?;

        Ok(())
    }

    async fn list_tools(&mut self) -> Result<Vec<Tool>> {
        let request = json!({
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "tools/list"
        });
        self.request_id += 1;

        let response = self.send_request(request).await?;

        let tools: Vec<Tool> = serde_json::from_value(response["result"]["tools"].clone())?;
        Ok(tools)
    }

    async fn call_tool(
        &mut self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolCallResult> {
        let request = json!({
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        });
        self.request_id += 1;

        let response = self.send_request(request).await?;

        let result: ToolCallResult = serde_json::from_value(response["result"].clone())?;
        Ok(result)
    }

    async fn list_resources(&mut self) -> Result<Vec<McpResource>> {
        let request = json!({
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "resources/list",
            "params": {}
        });
        self.request_id += 1;
        let response = self.send_request(request).await?;
        Ok(serde_json::from_value(
            response["result"]["resources"].clone(),
        )?)
    }

    async fn read_resource(&mut self, uri: &str) -> Result<Vec<McpResourceContent>> {
        let request = json!({
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "resources/read",
            "params": {"uri": uri}
        });
        self.request_id += 1;
        let response = self.send_request(request).await?;
        Ok(serde_json::from_value(
            response["result"]["contents"].clone(),
        )?)
    }

    async fn list_prompts(&mut self) -> Result<Vec<McpPrompt>> {
        let request = json!({
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "prompts/list",
            "params": {}
        });
        self.request_id += 1;
        let response = self.send_request(request).await?;
        Ok(serde_json::from_value(response["result"]["prompts"].clone()).unwrap_or_default())
    }

    async fn get_prompt(&mut self, name: &str) -> Result<String> {
        let request = json!({
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "prompts/get",
            "params": {"name": name}
        });
        self.request_id += 1;
        let response = self.send_request(request).await?;
        Ok(render_prompt_messages(&response["result"]))
    }

    async fn shutdown(&mut self) -> Result<()> {
        self.process.kill().await?;
        Ok(())
    }
}

// HTTP Client Implementation
#[derive(Debug)]
pub struct HttpClient {
    url: String,
    headers: HashMap<String, String>,
    client: reqwest::Client,
}

impl HttpClient {
    async fn new(url: String, headers: HashMap<String, String>) -> Result<Self> {
        let client = reqwest::Client::new();

        let http_client = Self {
            url: url.clone(),
            headers,
            client,
        };

        // Initialize connection
        http_client.initialize().await?;

        Ok(http_client)
    }

    async fn initialize(&self) -> Result<()> {
        let init_request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {
                    "name": "cubi",
                    "version": "0.3.0"
                }
            }
        });

        self.send_request(init_request).await?;

        Ok(())
    }

    async fn send_request(&self, request: serde_json::Value) -> Result<serde_json::Value> {
        self.send_request_to(&self.url, request).await
    }

    async fn send_request_to(
        &self,
        url: &str,
        request: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let mut req = self.client.post(url).json(&request);

        for (key, value) in &self.headers {
            req = req.header(key, value);
        }

        let response = req
            .send()
            .await
            .context("Failed to send HTTP request to MCP server")?;

        if !response.status().is_success() {
            anyhow::bail!("MCP server returned error: {}", response.status());
        }

        let json: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse MCP response")?;

        Ok(json)
    }

    async fn list_tools(&self) -> Result<Vec<Tool>> {
        let request = json!({
            "jsonrpc": "2.0",
            "id": Uuid::new_v4().to_string(),
            "method": "tools/list"
        });

        let response = self.send_request(request).await?;

        let tools: Vec<Tool> = serde_json::from_value(response["result"]["tools"].clone())?;
        Ok(tools)
    }

    async fn call_tool(&self, name: &str, arguments: serde_json::Value) -> Result<ToolCallResult> {
        let request = json!({
            "jsonrpc": "2.0",
            "id": Uuid::new_v4().to_string(),
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        });

        let response = self.send_request(request).await?;

        let result: ToolCallResult = serde_json::from_value(response["result"].clone())?;
        Ok(result)
    }

    async fn list_resources(&self) -> Result<Vec<McpResource>> {
        let request = json!({
            "jsonrpc": "2.0",
            "id": Uuid::new_v4().to_string(),
            "method": "resources/list",
            "params": {}
        });
        let response = self.send_request(request).await?;
        Ok(serde_json::from_value(
            response["result"]["resources"].clone(),
        )?)
    }

    async fn read_resource(&self, uri: &str) -> Result<Vec<McpResourceContent>> {
        let request = json!({
            "jsonrpc": "2.0",
            "id": Uuid::new_v4().to_string(),
            "method": "resources/read",
            "params": {"uri": uri}
        });
        let response = self.send_request(request).await?;
        Ok(serde_json::from_value(
            response["result"]["contents"].clone(),
        )?)
    }

    async fn list_prompts(&self) -> Result<Vec<McpPrompt>> {
        let request = json!({
            "jsonrpc": "2.0",
            "id": Uuid::new_v4().to_string(),
            "method": "prompts/list",
            "params": {}
        });
        let response = self.send_request(request).await?;
        Ok(serde_json::from_value(response["result"]["prompts"].clone()).unwrap_or_default())
    }

    async fn get_prompt(&self, name: &str) -> Result<String> {
        let request = json!({
            "jsonrpc": "2.0",
            "id": Uuid::new_v4().to_string(),
            "method": "prompts/get",
            "params": {"name": name}
        });
        let response = self.send_request(request).await?;
        Ok(render_prompt_messages(&response["result"]))
    }
}

/// Flatten a `prompts/get` response into a single user-visible string.
/// We deliberately do not try to preserve message role boundaries —
/// `/mcp-prompts` is for human inspection, not direct injection into
/// the chat history.
fn render_prompt_messages(result: &serde_json::Value) -> String {
    let mut out = String::new();
    if let Some(messages) = result["messages"].as_array() {
        for m in messages {
            let role = m["role"].as_str().unwrap_or("user");
            let text = m["content"]["text"]
                .as_str()
                .or_else(|| m["content"].as_str())
                .unwrap_or("");
            if !text.is_empty() {
                out.push_str(&format!("[{role}] {text}\n"));
            }
        }
    }
    if out.is_empty() {
        if let Some(text) = result["description"].as_str() {
            out.push_str(text);
        }
    }
    out
}
