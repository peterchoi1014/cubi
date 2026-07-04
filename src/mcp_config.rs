use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    #[serde(rename = "mcpServers")]
    pub mcp_servers: HashMap<String, McpServerConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Command to run (for STDIO transport)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,

    /// Arguments for the command
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<Vec<String>>,

    /// Environment variables
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env: Option<HashMap<String, String>>,

    /// HTTP URL (for remote servers)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "httpUrl")]
    pub http_url: Option<String>,

    /// HTTP headers (for authentication)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,

    /// Optional OAuth provider key used to inject an Authorization header
    /// from ~/.cubi/oauth.json for HTTP MCP servers.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "oauthProvider")]
    pub oauth_provider: Option<String>,

    /// Whether this server should be connected. Absent in older config files,
    /// which serde treats as `true` so pre-existing `~/.cubi/mcp.json` files
    /// keep working (every configured server enabled). Skipped when `true` so
    /// saved files stay clean; only an explicit `false` is written.
    #[serde(default = "default_enabled", skip_serializing_if = "is_enabled")]
    pub enabled: bool,
}

fn default_enabled() -> bool {
    true
}

fn is_enabled(enabled: &bool) -> bool {
    *enabled
}

impl McpServerConfig {
    pub fn is_stdio(&self) -> bool {
        self.command.is_some()
    }

    pub fn is_http(&self) -> bool {
        self.http_url.is_some()
    }
}

impl McpConfig {
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path()?;

        if !config_path.exists() {
            // Create empty config
            let empty_config = McpConfig {
                mcp_servers: HashMap::new(),
            };
            empty_config.save()?;
            return Ok(empty_config);
        }

        let content =
            fs::read_to_string(&config_path).context("Failed to read MCP configuration file")?;

        let config: McpConfig =
            serde_json::from_str(&content).context("Failed to parse MCP configuration")?;

        Ok(config)
    }

    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path()?;

        // Ensure parent directory exists
        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let json = serde_json::to_string_pretty(self)?;
        fs::write(&config_path, json)?;

        Ok(())
    }

    pub fn config_path() -> Result<PathBuf> {
        let cubi_dir = crate::sessions::cubi_dir().context("Could not find home directory")?;

        Ok(cubi_dir.join("mcp.json"))
    }

    // Allow dead_code as these may be used for future CLI commands
    #[allow(dead_code)]
    pub fn add_server(&mut self, name: String, config: McpServerConfig) {
        self.mcp_servers.insert(name, config);
    }

    #[allow(dead_code)]
    pub fn remove_server(&mut self, name: &str) -> bool {
        self.mcp_servers.remove(name).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_path_uses_cubi_home() {
        crate::compat::test_env::with_cubi_home(|cubi_home, other_home| {
            let path = McpConfig::config_path().expect("config path");
            assert_eq!(path, cubi_home.join(".cubi").join("mcp.json"));
            assert!(!path.starts_with(other_home));
        });
    }

    #[test]
    fn enabled_defaults_to_true_when_absent() {
        // A server entry with no `enabled` field (as written by older cubi
        // versions) must deserialize as enabled so existing configs keep
        // connecting.
        let json = r#"{
            "mcpServers": {
                "legacy": { "command": "echo", "args": ["hi"] }
            }
        }"#;
        let config: McpConfig = serde_json::from_str(json).expect("parse legacy config");
        let server = config.mcp_servers.get("legacy").expect("legacy server");
        assert!(server.enabled, "absent enabled field must default to true");

        // An explicit false round-trips.
        let json_disabled = r#"{
            "mcpServers": {
                "off": { "command": "echo", "enabled": false }
            }
        }"#;
        let config: McpConfig = serde_json::from_str(json_disabled).expect("parse disabled config");
        assert!(!config.mcp_servers.get("off").expect("off server").enabled);
    }

    #[test]
    fn enabled_true_is_not_serialized() {
        // skip_serializing_if keeps saved files clean: an enabled server does
        // not write the field, while a disabled one does.
        let mut servers = HashMap::new();
        servers.insert(
            "on".to_string(),
            McpServerConfig {
                command: Some("echo".to_string()),
                args: None,
                env: None,
                http_url: None,
                headers: None,
                oauth_provider: None,
                enabled: true,
            },
        );
        let config = McpConfig {
            mcp_servers: servers,
        };
        let json = serde_json::to_string(&config).expect("serialize");
        assert!(
            !json.contains("enabled"),
            "enabled=true should be skipped, got: {json}"
        );
    }
}
