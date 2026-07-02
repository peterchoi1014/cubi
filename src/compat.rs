//! One-stop compatibility shim for the `ai-chat-cli` → `cubi` rebrand.
//!
//! Two jobs:
//!
//! 1. **Env var migration** — every previously documented environment
//!    variable lived under `AI_CHAT_CLI_*` or `AICHAT_*`. We renamed the
//!    canonical names to `CUBI_*` but don't want to break users who
//!    already wired the old names into shells, dotfiles, CI configs,
//!    etc. `promote_legacy_env` runs once at startup and copies any
//!    legacy values into their new names *iff* the new name is unset.
//!
//! 2. **Config directory migration** — persistent state previously lived
//!    in `~/.cubi/` (config.json, sessions, trust, telemetry,
//!    oauth tokens, memdir, …). We now use `~/.cubi/`. The first time
//!    the new binary runs on a machine with an existing
//!    `~/.cubi/` and no `~/.cubi/`, we rename the directory in
//!    place so the user keeps their settings.
//!
//! Both helpers are idempotent and safe to call unconditionally at
//! startup. They never overwrite a value the user has explicitly set
//! under the new name.

use std::fs;
use std::path::PathBuf;

/// All env vars renamed by the Cubi rebrand. Left column is the new
/// canonical name we read everywhere in the source tree; right column
/// is the legacy name we still honor for one release cycle.
const ENV_RENAMES: &[(&str, &str)] = &[
    ("CUBI_MODEL", "AI_CHAT_CLI_MODEL"),
    ("CUBI_BASE_URL", "AI_CHAT_CLI_BASE_URL"),
    ("CUBI_API_KEY", "AI_CHAT_CLI_API_KEY"),
    ("CUBI_PROVIDER", "AI_CHAT_CLI_PROVIDER"),
    ("CUBI_NO_ONBOARD", "AI_CHAT_CLI_NO_ONBOARD"),
    ("CUBI_THEME", "AICHAT_THEME"),
    ("CUBI_OUTPUT_STYLE", "AICHAT_OUTPUT_STYLE"),
    ("CUBI_COLOR", "AICHAT_COLOR"),
    ("CUBI_VIM_MODE", "AICHAT_VIM_MODE"),
    ("CUBI_TELEMETRY", "AICHAT_TELEMETRY"),
    ("CUBI_DEBUG_TOOL_CALL", "AICHAT_DEBUG_TOOL_CALL"),
    ("CUBI_POLICY_FILE", "AICHAT_POLICY_FILE"),
    ("CUBI_OAUTH_FILE", "AICHAT_OAUTH_FILE"),
    ("CUBI_RATE_LIMIT_BACKOFF_MS", "AICHAT_RATE_LIMIT_BACKOFF_MS"),
];

/// Copy any legacy env vars into their new names if the new name is
/// unset. Call once near the top of `main` before any module reads env.
pub fn promote_legacy_env() {
    for (new, old) in ENV_RENAMES {
        if std::env::var_os(new).is_some() {
            continue;
        }
        if let Some(v) = std::env::var_os(old) {
            // SAFETY: called from `main` before any threads are spawned.
            unsafe {
                std::env::set_var(new, v);
            }
        }
    }
    // Provider-keyed API key env vars are constructed dynamically (e.g.
    // `CUBI_OPENAI_API_KEY`), so they aren't in the static table above.
    // Sweep the environment for any `AICHAT_*_API_KEY` and promote.
    let keys: Vec<(String, std::ffi::OsString)> = std::env::vars_os()
        .filter_map(|(k, v)| {
            let key = k.to_str()?;
            if let Some(rest) = key.strip_prefix("AICHAT_") {
                if rest.ends_with("_API_KEY") {
                    return Some((format!("CUBI_{rest}"), v));
                }
            }
            None
        })
        .collect();
    for (new, v) in keys {
        if std::env::var_os(&new).is_none() {
            // SAFETY: called from `main` before any threads are spawned.
            unsafe {
                std::env::set_var(&new, v);
            }
        }
    }
}

/// Returns the resolved path to the new config directory (`~/.cubi/`),
/// or `None` if the home directory can't be determined.
pub fn cubi_dir() -> Option<PathBuf> {
    crate::sessions::cubi_dir()
}

/// Legacy config directory from the pre-rebrand era (`~/.ai-chat-cli/`).
/// Returned for migration purposes; production code should not read from
/// this path directly.
pub fn legacy_dir() -> Option<PathBuf> {
    Some(crate::sessions::home_dir()?.join(".ai-chat-cli"))
}

/// Rename `~/.ai-chat-cli/` → `~/.cubi/` exactly once, when the new
/// location does not yet exist. No-op if the legacy directory is
/// missing, the new directory is already populated, or the rename
/// fails (we log a warning and let downstream code create a fresh
/// `~/.cubi/`).
pub fn migrate_config_dir() {
    let (Some(new), Some(old)) = (cubi_dir(), legacy_dir()) else {
        return;
    };
    if new.exists() {
        return;
    }
    if !old.exists() {
        return;
    }
    if let Err(e) = fs::rename(&old, &new) {
        eprintln!(
            "Warn: could not migrate {} -> {}: {} (a fresh config dir will be created)",
            old.display(),
            new.display(),
            e
        );
    }
}

#[cfg(test)]
pub(crate) mod test_env {
    use std::ffi::OsString;
    use std::path::Path;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvRestore {
        old: Vec<(&'static str, Option<OsString>)>,
    }

    impl EnvRestore {
        fn capture(names: &[&'static str]) -> Self {
            Self {
                old: names
                    .iter()
                    .map(|name| (*name, std::env::var_os(name)))
                    .collect(),
            }
        }
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            for (name, value) in &self.old {
                unsafe {
                    match value {
                        Some(value) => std::env::set_var(name, value),
                        None => std::env::remove_var(name),
                    }
                }
            }
        }
    }

    pub(crate) fn with_cubi_home<T>(f: impl FnOnce(&Path, &Path) -> T) -> T {
        let _lock = ENV_LOCK.lock().expect("env lock not poisoned");
        let cubi_home = tempfile::tempdir().expect("cubi home tempdir");
        let other_home = tempfile::tempdir().expect("other home tempdir");
        let _restore = EnvRestore::capture(&[
            crate::sessions::CUBI_HOME_ENV,
            "HOME",
            "USERPROFILE",
            "CUBI_OAUTH_FILE",
            "CUBI_PLUGINS_DIR",
        ]);
        unsafe {
            std::env::set_var(crate::sessions::CUBI_HOME_ENV, cubi_home.path());
            std::env::set_var("HOME", other_home.path());
            std::env::set_var("USERPROFILE", other_home.path());
            std::env::remove_var("CUBI_OAUTH_FILE");
            std::env::remove_var("CUBI_PLUGINS_DIR");
        }

        f(cubi_home.path(), other_home.path())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_dirs_use_cubi_home_before_platform_home() {
        test_env::with_cubi_home(|cubi_home, other_home| {
            let cubi = cubi_dir().expect("cubi dir");
            let legacy = legacy_dir().expect("legacy dir");

            assert_eq!(cubi, cubi_home.join(".cubi"));
            assert_eq!(legacy, cubi_home.join(".ai-chat-cli"));
            assert!(!cubi.starts_with(other_home));
            assert!(!legacy.starts_with(other_home));
        });
    }

    #[test]
    fn rename_table_has_no_duplicates() {
        let mut news: Vec<&str> = ENV_RENAMES.iter().map(|(n, _)| *n).collect();
        news.sort();
        let len_before = news.len();
        news.dedup();
        assert_eq!(news.len(), len_before, "duplicate CUBI_ env names");
    }

    #[test]
    fn rename_table_uses_cubi_prefix() {
        for (new, _) in ENV_RENAMES {
            assert!(new.starts_with("CUBI_"), "{new} is not CUBI_-prefixed");
        }
    }
}
