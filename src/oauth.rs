use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
#[cfg(unix)]
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OAuthStore {
    #[serde(default)]
    pub providers: HashMap<String, OAuthToken>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthToken {
    pub access_token: String,
    #[serde(default)]
    pub refresh_token: Option<String>,
    #[serde(default)]
    pub expires_at_unix: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct LoginArgs {
    pub provider: String,
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_in_seconds: Option<u64>,
}

impl OAuthToken {
    pub fn is_expired(&self) -> bool {
        match self.expires_at_unix {
            Some(ts) => ts <= now_unix(),
            None => false,
        }
    }
}

impl OAuthStore {
    pub fn storage_path() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("CUBI_OAUTH_FILE") {
            if !p.trim().is_empty() {
                return Some(PathBuf::from(p));
            }
        }
        Some(dirs::home_dir()?.join(".cubi").join("oauth.json"))
    }

    pub fn load() -> Self {
        let Some(path) = Self::storage_path() else {
            return Self::default();
        };
        match fs::read_to_string(&path) {
            Ok(raw) => match serde_json::from_str(&raw) {
                Ok(store) => store,
                Err(err) => {
                    eprintln!(
                        "Warning: failed to parse OAuth store {}: {}",
                        path.display(),
                        err
                    );
                    Self::default()
                }
            },
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Self::default(),
            Err(err) => {
                eprintln!(
                    "Warning: failed to load OAuth store from {}: {}",
                    path.display(),
                    err
                );
                Self::default()
            }
        }
    }

    pub fn save(&self) -> Result<()> {
        let path = Self::storage_path().context("Could not resolve OAuth storage path")?;
        self.save_to_path(&path)
    }

    #[cfg(test)]
    fn load_from_path(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("Failed to read {}", path.display()))?;
        serde_json::from_str(&raw).with_context(|| format!("Failed to parse {}", path.display()))
    }

    fn save_to_path(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create {}", parent.display()))?;
        }
        let json = serde_json::to_string_pretty(self)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
            let mut file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .mode(0o600)
                .open(path)
                .with_context(|| format!("Failed to open {} for writing", path.display()))?;
            file.write_all(json.as_bytes())
                .with_context(|| format!("Failed to write {}", path.display()))?;
            file.sync_all()
                .with_context(|| format!("Failed to sync {}", path.display()))?;
            fs::set_permissions(path, fs::Permissions::from_mode(0o600)).with_context(|| {
                format!("Failed to set secure permissions on {}", path.display())
            })?;
        }
        #[cfg(not(unix))]
        {
            fs::write(path, json).with_context(|| format!("Failed to write {}", path.display()))?;
        }
        Ok(())
    }

    pub fn upsert_login(&mut self, args: &LoginArgs) {
        let expires_at_unix = args
            .expires_in_seconds
            .map(|s| now_unix().saturating_add(s));
        self.providers.insert(
            normalize_provider(&args.provider),
            OAuthToken {
                access_token: args.access_token.clone(),
                refresh_token: args.refresh_token.clone(),
                expires_at_unix,
            },
        );
    }

    pub fn remove_provider(&mut self, provider: &str) -> bool {
        self.providers
            .remove(&normalize_provider(provider))
            .is_some()
    }

    pub fn get_provider(&self, provider: &str) -> Option<&OAuthToken> {
        self.providers.get(&normalize_provider(provider))
    }
}

pub fn normalize_provider(provider: &str) -> String {
    provider.trim().to_ascii_lowercase()
}

pub fn provider_env_var(provider: &str) -> String {
    format!(
        "CUBI_{}_API_KEY",
        normalize_provider(provider).to_ascii_uppercase()
    )
}

pub fn bearer_header_for_provider(provider: &str) -> Result<Option<String>> {
    let store = OAuthStore::load();
    let Some(token) = store.get_provider(provider) else {
        return Ok(None);
    };
    if token.is_expired() {
        return Ok(None);
    }
    Ok(Some(format!("Bearer {}", token.access_token)))
}

pub fn parse_login_args(args: &str) -> Result<LoginArgs> {
    let parts: Vec<&str> = args.split_whitespace().collect();
    if parts.len() < 2 {
        anyhow::bail!(
            "Usage: /login <provider> <access-token> [--refresh-token <token>] [--expires-in <seconds>]"
        );
    }

    let provider = normalize_provider(parts[0]);
    let access_token = parts[1].to_string();
    if provider.is_empty() {
        anyhow::bail!("Provider cannot be empty");
    }
    if access_token.is_empty() {
        anyhow::bail!("Access token cannot be empty");
    }

    let mut refresh_token = None;
    let mut expires_in_seconds = None;
    let mut i = 2usize;
    while i < parts.len() {
        match parts[i] {
            "--refresh-token" => {
                let Some(v) = parts.get(i + 1) else {
                    anyhow::bail!("--refresh-token requires a value");
                };
                refresh_token = Some((*v).to_string());
                i += 2;
            }
            "--expires-in" => {
                let Some(v) = parts.get(i + 1) else {
                    anyhow::bail!("--expires-in requires a value");
                };
                expires_in_seconds = Some(
                    v.parse::<u64>()
                        .with_context(|| format!("Invalid --expires-in seconds: {v}"))?,
                );
                i += 2;
            }
            other => {
                anyhow::bail!("Unknown flag: {other}");
            }
        }
    }

    Ok(LoginArgs {
        provider,
        access_token,
        refresh_token,
        expires_in_seconds,
    })
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn parse_login_args_basic() {
        let parsed =
            parse_login_args("github tok123 --refresh-token rtok --expires-in 3600").unwrap();
        assert_eq!(parsed.provider, "github");
        assert_eq!(parsed.access_token, "tok123");
        assert_eq!(parsed.refresh_token.as_deref(), Some("rtok"));
        assert_eq!(parsed.expires_in_seconds, Some(3600));
    }

    #[test]
    fn parse_login_args_rejects_unknown_flags() {
        let err = parse_login_args("github tok123 --bogus x").unwrap_err();
        assert!(err.to_string().contains("Unknown flag"));
    }

    #[test]
    fn parse_login_args_requires_provider_and_token() {
        let err = parse_login_args("github").unwrap_err();
        assert!(err.to_string().contains("Usage: /login"));
    }

    #[test]
    fn parse_login_args_requires_flag_values() {
        let refresh_err = parse_login_args("github tok123 --refresh-token").unwrap_err();
        assert!(
            refresh_err
                .to_string()
                .contains("--refresh-token requires a value")
        );

        let expires_err = parse_login_args("github tok123 --expires-in").unwrap_err();
        assert!(
            expires_err
                .to_string()
                .contains("--expires-in requires a value")
        );
    }

    #[test]
    fn parse_login_args_rejects_invalid_expires_in() {
        let err = parse_login_args("github tok123 --expires-in not-a-number").unwrap_err();
        assert!(err.to_string().contains("Invalid --expires-in seconds"));
    }

    #[test]
    fn provider_normalization_and_env_mapping() {
        assert_eq!(normalize_provider("  GitHub  "), "github");
        assert_eq!(provider_env_var("  GitHub  "), "CUBI_GITHUB_API_KEY");
    }

    #[test]
    fn store_upsert_and_remove_provider_normalizes_key() {
        let mut store = OAuthStore::default();
        let args = LoginArgs {
            provider: "  GiTHub  ".to_string(),
            access_token: "token123".to_string(),
            refresh_token: Some("refresh123".to_string()),
            expires_in_seconds: Some(60),
        };

        store.upsert_login(&args);
        let token = store
            .get_provider("github")
            .expect("provider token should exist");
        assert_eq!(token.access_token, "token123");
        assert_eq!(token.refresh_token.as_deref(), Some("refresh123"));
        assert!(token.expires_at_unix.is_some());

        assert!(store.remove_provider(" GITHUB "));
        assert!(store.get_provider("github").is_none());
    }

    #[test]
    fn token_expiry_works() {
        let expired = OAuthToken {
            access_token: "a".to_string(),
            refresh_token: None,
            expires_at_unix: Some(now_unix().saturating_sub(1)),
        };
        let fresh = OAuthToken {
            access_token: "a".to_string(),
            refresh_token: None,
            expires_at_unix: Some(now_unix().saturating_add(600)),
        };
        assert!(expired.is_expired());
        assert!(!fresh.is_expired());
    }

    #[test]
    fn save_and_load_roundtrip() {
        let path = std::env::temp_dir().join(format!(
            "cubi-oauth-{}.json",
            now_unix().saturating_mul(1_000_000_000)
                + SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or(Duration::from_secs(0))
                    .subsec_nanos() as u64
        ));

        let mut store = OAuthStore::default();
        let args = LoginArgs {
            provider: "GitHub".to_string(),
            access_token: "token123".to_string(),
            refresh_token: Some("refresh123".to_string()),
            expires_in_seconds: Some(120),
        };
        store.upsert_login(&args);
        store.save_to_path(&path).expect("save should succeed");

        let loaded = OAuthStore::load_from_path(&path).expect("load should succeed");
        let token = loaded
            .get_provider("github")
            .expect("provider token should exist");
        assert_eq!(token.access_token, "token123");
        assert_eq!(token.refresh_token.as_deref(), Some("refresh123"));
        assert!(token.expires_at_unix.is_some());

        let _ = fs::remove_file(path);
    }
}
