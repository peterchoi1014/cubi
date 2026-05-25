//! Enterprise-managed policy overlay.
//!
//! Roadmap items C#2 (permissions) and C#23 (enterprise policy). The
//! per-user trust store in `permissions.rs` is fully writable by the
//! user — fine for solo developers, but not enough for an admin who
//! needs to push a hard tool denylist to a fleet of machines. This
//! module reads a separate, *read-only* JSON file (`policy.json`) from
//! whichever of these locations exists first:
//!
//! 1. `$AICHAT_POLICY_FILE` (escape hatch for tests / CI).
//! 2. `/etc/ai-chat-cli/policy.json` (Unix system-wide).
//! 3. `~/.ai-chat-cli/policy.json` (per-user fallback, useful for the
//!    Windows case where step 2 doesn't exist).
//!
//! The deny list it carries is checked **before** the user's allow/deny
//! list in `Permissions::check_tool_allowed`, so a malicious user
//! config can't undo it.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Policy {
    /// Tools the admin has globally forbidden. Wins over any user
    /// allow-list. `BTreeSet` so on-disk diffs stay stable.
    #[serde(default)]
    pub denied_tools: BTreeSet<String>,
    /// Free-form "why is this here" text shown in `/permissions` so a
    /// user who hits a denial knows whom to talk to.
    #[serde(default)]
    pub note: Option<String>,
}

impl Policy {
    /// Reads the policy file, returning an empty policy when no file is
    /// present anywhere on the search path. Malformed JSON is treated
    /// as "no policy" rather than fatally aborting the CLI — the file
    /// is admin-controlled and a parse error must not lock everyone
    /// out of the binary.
    pub fn load() -> Self {
        let Some(path) = Self::resolve_path() else {
            return Self::default();
        };
        let Ok(raw) = fs::read_to_string(&path) else {
            return Self::default();
        };
        serde_json::from_str::<Self>(&raw).unwrap_or_default()
    }

    /// Returns the path actually checked (useful for `/permissions` to
    /// say "policy: /etc/ai-chat-cli/policy.json").
    pub fn active_path() -> Option<PathBuf> {
        Self::resolve_path()
    }

    fn resolve_path() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("AICHAT_POLICY_FILE") {
            let path = PathBuf::from(p);
            if path.exists() {
                return Some(path);
            }
        }
        #[cfg(unix)]
        {
            let system = PathBuf::from("/etc/ai-chat-cli/policy.json");
            if system.exists() {
                return Some(system);
            }
        }
        let user = dirs::home_dir()?.join(".ai-chat-cli").join("policy.json");
        if user.exists() { Some(user) } else { None }
    }

    pub fn is_denied(&self, tool: &str) -> bool {
        self.denied_tools.contains(tool.trim())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::time::{SystemTime, UNIX_EPOCH};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn tmp_file(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("ai-chat-cli-policy-{name}-{nanos}.json"))
    }

    #[test]
    fn missing_file_yields_empty_policy() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("AICHAT_POLICY_FILE").ok();
        // Point at a path that definitely doesn't exist.
        let p = tmp_file("missing");
        unsafe {
            std::env::set_var("AICHAT_POLICY_FILE", &p);
        }
        let pol = Policy::load();
        assert!(pol.denied_tools.is_empty());
        unsafe {
            std::env::remove_var("AICHAT_POLICY_FILE");
        }
        if let Some(v) = prev {
            unsafe { std::env::set_var("AICHAT_POLICY_FILE", v) };
        }
    }

    #[test]
    fn loads_deny_list_from_env_path() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("AICHAT_POLICY_FILE").ok();
        let p = tmp_file("deny");
        fs::write(
            &p,
            r#"{"denied_tools":["bash","write_file"],"note":"corp policy"}"#,
        )
        .unwrap();
        unsafe {
            std::env::set_var("AICHAT_POLICY_FILE", &p);
        }
        let pol = Policy::load();
        assert!(pol.is_denied("bash"));
        assert!(pol.is_denied("write_file"));
        assert!(!pol.is_denied("read_file"));
        assert_eq!(pol.note.as_deref(), Some("corp policy"));
        fs::remove_file(&p).ok();
        unsafe {
            std::env::remove_var("AICHAT_POLICY_FILE");
        }
        if let Some(v) = prev {
            unsafe { std::env::set_var("AICHAT_POLICY_FILE", v) };
        }
    }

    #[test]
    fn corrupt_file_is_treated_as_empty() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("AICHAT_POLICY_FILE").ok();
        let p = tmp_file("corrupt");
        fs::write(&p, "not json {{{").unwrap();
        unsafe {
            std::env::set_var("AICHAT_POLICY_FILE", &p);
        }
        let pol = Policy::load();
        assert!(pol.denied_tools.is_empty());
        fs::remove_file(&p).ok();
        unsafe {
            std::env::remove_var("AICHAT_POLICY_FILE");
        }
        if let Some(v) = prev {
            unsafe { std::env::set_var("AICHAT_POLICY_FILE", v) };
        }
    }
}
