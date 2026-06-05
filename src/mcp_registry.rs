//! Embedded catalog of common Model Context Protocol servers.
//!
//! The catalog is a static JSON file shipped with the binary (`docs/mcp/
//! registry.json`) so `cubi mcp search` and `cubi mcp install` work offline.
//! Entries are templates: they describe how to launch a server and which
//! environment variables the user needs to provide. Installing an entry
//! materializes it into the user's `~/.cubi/mcp.json` via [`crate::mcp_config`].
//!
//! Adding a new server is a pull request that edits `docs/mcp/registry.json`
//! plus a `cargo test --quiet` run to validate the schema. See
//! `docs/mcp/registry.md` for the contributor guide.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::OnceLock;

const REGISTRY_JSON: &str = include_str!("../docs/mcp/registry.json");

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EnvVar {
    pub required: bool,
    pub description: String,
}

/// Transport-specific launch metadata. Mirrors the discriminator used in
/// `~/.cubi/mcp.json` (stdio vs. http) so installs round-trip cleanly.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "transport", rename_all = "lowercase")]
pub enum Transport {
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
    },
    Http {
        #[serde(rename = "http_url")]
        url: String,
        #[serde(default)]
        headers: BTreeMap<String, String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        oauth_provider: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RegistryEntry {
    pub name: String,
    pub description: String,
    #[serde(flatten)]
    pub transport: Transport,
    #[serde(default)]
    pub env: BTreeMap<String, EnvVar>,
    pub homepage: String,
    pub license: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

static REGISTRY: OnceLock<Vec<RegistryEntry>> = OnceLock::new();

/// Parse and return the embedded registry. Parses on first access and caches
/// the result for the lifetime of the process. Panics if the embedded JSON is
/// malformed; the schema is locked down by `tests/mcp_registry.rs` so any
/// malformed entry fails CI long before reaching a user.
pub fn load_registry() -> &'static [RegistryEntry] {
    REGISTRY
        .get_or_init(|| {
            serde_json::from_str::<Vec<RegistryEntry>>(REGISTRY_JSON)
                .expect("embedded docs/mcp/registry.json is malformed")
        })
        .as_slice()
}

/// Case-insensitive substring search across name, description, and tags.
/// Empty queries return every entry, sorted by name.
pub fn search(query: &str) -> Vec<&'static RegistryEntry> {
    let q = query.trim().to_ascii_lowercase();
    let mut out: Vec<_> = load_registry()
        .iter()
        .filter(|e| {
            if q.is_empty() {
                return true;
            }
            e.name.to_ascii_lowercase().contains(&q)
                || e.description.to_ascii_lowercase().contains(&q)
                || e.tags.iter().any(|t| t.to_ascii_lowercase().contains(&q))
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Exact, case-insensitive lookup by entry name.
pub fn find(name: &str) -> Option<&'static RegistryEntry> {
    load_registry()
        .iter()
        .find(|e| e.name.eq_ignore_ascii_case(name))
}

impl RegistryEntry {
    /// Short human-readable label for the transport column.
    pub fn transport_label(&self) -> &'static str {
        match self.transport {
            Transport::Stdio { .. } => "stdio",
            Transport::Http { .. } => "http",
        }
    }

    /// Materialize this entry into the on-disk schema used by
    /// `~/.cubi/mcp.json`. `env_values` supplies the user-provided values for
    /// the env vars declared in [`Self::env`]; only keys present in
    /// `env_values` are serialized.
    pub fn to_server_config(
        &self,
        env_values: &BTreeMap<String, String>,
    ) -> crate::mcp_config::McpServerConfig {
        use crate::mcp_config::McpServerConfig;
        match &self.transport {
            Transport::Stdio { command, args } => {
                let env_map: std::collections::HashMap<String, String> = env_values
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                McpServerConfig {
                    command: Some(command.clone()),
                    args: if args.is_empty() {
                        None
                    } else {
                        Some(args.clone())
                    },
                    env: if env_map.is_empty() {
                        None
                    } else {
                        Some(env_map)
                    },
                    http_url: None,
                    headers: None,
                    oauth_provider: None,
                }
            }
            Transport::Http {
                url,
                headers,
                oauth_provider,
            } => {
                let mut hdrs: std::collections::HashMap<String, String> = headers
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                for (k, v) in env_values {
                    hdrs.insert(k.clone(), v.clone());
                }
                McpServerConfig {
                    command: None,
                    args: None,
                    env: None,
                    http_url: Some(url.clone()),
                    headers: if hdrs.is_empty() { None } else { Some(hdrs) },
                    oauth_provider: oauth_provider.clone(),
                }
            }
        }
    }
}
