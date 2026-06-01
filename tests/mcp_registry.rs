//! Schema and content tests for the embedded MCP registry.
//!
//! The registry module itself lives inside the binary crate, so these tests
//! parse `docs/mcp/registry.json` from disk and validate the same shape the
//! runtime parser expects. This catches malformed entries (missing fields,
//! wrong transport type, empty strings) at PR time without booting the
//! whole CLI.
#![allow(dead_code)] // Deserialize fields are consumed by serde, not Rust call sites.

use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Debug, Deserialize)]
struct EnvVar {
    required: bool,
    description: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "transport", rename_all = "lowercase")]
enum Transport {
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
        #[serde(default)]
        oauth_provider: Option<String>,
    },
}

#[derive(Debug, Deserialize)]
struct RegistryEntry {
    name: String,
    description: String,
    #[serde(flatten)]
    transport: Transport,
    #[serde(default)]
    env: BTreeMap<String, EnvVar>,
    homepage: String,
    license: String,
    #[serde(default)]
    tags: Vec<String>,
}

fn load() -> Vec<RegistryEntry> {
    let text = include_str!("../docs/mcp/registry.json");
    serde_json::from_str::<Vec<RegistryEntry>>(text)
        .expect("docs/mcp/registry.json must parse as a Vec<RegistryEntry>")
}

#[test]
fn registry_has_at_least_ten_entries() {
    let entries = load();
    assert!(
        entries.len() >= 10,
        "registry should ship with at least 10 entries; got {}",
        entries.len()
    );
}

#[test]
fn every_entry_has_required_fields() {
    let entries = load();
    for e in &entries {
        assert!(!e.name.trim().is_empty(), "empty name in registry");
        assert!(
            !e.description.trim().is_empty(),
            "empty description for '{}'",
            e.name
        );
        assert!(
            !e.homepage.trim().is_empty(),
            "empty homepage for '{}'",
            e.name
        );
        assert!(
            !e.license.trim().is_empty(),
            "empty license for '{}'",
            e.name
        );
        match &e.transport {
            Transport::Stdio { command, .. } => {
                assert!(
                    !command.trim().is_empty(),
                    "stdio entry '{}' has empty command",
                    e.name
                );
            }
            Transport::Http { url, .. } => {
                assert!(
                    url.starts_with("http://") || url.starts_with("https://"),
                    "http entry '{}' has non-URL http_url: {}",
                    e.name,
                    url
                );
            }
        }
        for (k, spec) in &e.env {
            assert!(!k.trim().is_empty(), "empty env key in '{}'", e.name);
            assert!(
                !spec.description.trim().is_empty(),
                "env '{}' in '{}' has no description",
                k,
                e.name
            );
        }
    }
}

#[test]
fn names_are_unique() {
    let entries = load();
    let mut seen = std::collections::HashSet::new();
    for e in &entries {
        assert!(
            seen.insert(e.name.to_ascii_lowercase()),
            "duplicate registry entry name: {}",
            e.name
        );
    }
}

#[test]
fn github_entry_requires_token() {
    let entries = load();
    let github = entries
        .iter()
        .find(|e| e.name == "github")
        .expect("github entry must be present in the registry");
    let token = github
        .env
        .get("GITHUB_PERSONAL_ACCESS_TOKEN")
        .expect("github entry must declare GITHUB_PERSONAL_ACCESS_TOKEN");
    assert!(
        token.required,
        "GITHUB_PERSONAL_ACCESS_TOKEN must be marked required"
    );
}

#[test]
fn search_for_git_matches_github_and_gitlab() {
    let entries = load();
    let q = "git";
    let mut hits: Vec<&str> = entries
        .iter()
        .filter(|e| {
            e.name.to_ascii_lowercase().contains(q)
                || e.description.to_ascii_lowercase().contains(q)
                || e.tags.iter().any(|t| t.to_ascii_lowercase().contains(q))
        })
        .map(|e| e.name.as_str())
        .collect();
    hits.sort();
    insta::assert_debug_snapshot!("search_git_names", hits);
}
