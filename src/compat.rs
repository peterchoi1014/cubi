//! One-stop compatibility shim for the `ai-chat-cli` → `cubi` rebrand.
//!
//! Two jobs:
//!
//! 1. **Env var migration** — every previously documented environment
//!    variable lived under `AI_CHAT_CLI_*` or `AICHAT_*`. We renamed the
//!    canonical names to `CUBI_*` but don't want to break users who
//!    already wired the old names into shells, dotfiles, CI configs,
//!    etc. `promote_legacy_env` copies legacy values into their new names in
//!    the startup snapshot *iff* the new name is unset.
//!
//! 2. **Config directory migration** — persistent state previously lived
//!    in `~/.ai-chat-cli/` (config.json, sessions, trust, telemetry,
//!    oauth tokens, memdir, …). We now use `~/.cubi/`. The first time
//!    the new binary runs on a machine with an existing
//!    `~/.ai-chat-cli/` and no `~/.cubi/`, we rename the directory in
//!    place so the user keeps their settings.
//!
//! Both helpers are idempotent and safe to call unconditionally at
//! startup. They never overwrite a value the user has explicitly set
//! under the new name.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::path::Path;

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

/// Promote aliases in the captured startup input, never in the process env.
pub fn promote_legacy_env(values: &mut BTreeMap<OsString, OsString>) {
    promote_legacy_env_with_case(values, cfg!(windows));
}

pub(crate) fn promote_legacy_env_with_case(
    values: &mut BTreeMap<OsString, OsString>,
    case_insensitive: bool,
) {
    for (new, old) in ENV_RENAMES {
        if contains_key_with_case(values, new, case_insensitive) {
            continue;
        }
        if let Some(v) = value_with_case(values, old, case_insensitive).cloned() {
            values.insert(OsString::from(new), v);
        }
    }
    // Provider-keyed API key env vars are constructed dynamically (e.g.
    // `CUBI_OPENAI_API_KEY`), so they aren't in the static table above.
    // Sweep the environment for any `AICHAT_*_API_KEY` and promote.
    let keys: Vec<(String, std::ffi::OsString)> = values
        .iter()
        .filter_map(|(k, v)| {
            let key = crate::config::env_key::normalize_with_case(k, case_insensitive);
            let key = key.to_str()?;
            if let Some(rest) = key.strip_prefix("AICHAT_") {
                if rest.ends_with("_API_KEY") {
                    return Some((format!("CUBI_{rest}"), v.clone()));
                }
            }
            None
        })
        .collect();
    for (new, v) in keys {
        if !contains_key_with_case(values, &new, case_insensitive) {
            values.insert(OsString::from(new), v);
        }
    }
}

fn contains_key_with_case(
    values: &BTreeMap<OsString, OsString>,
    name: &str,
    case_insensitive: bool,
) -> bool {
    value_with_case(values, name, case_insensitive).is_some()
}

fn value_with_case<'a>(
    values: &'a BTreeMap<OsString, OsString>,
    name: &str,
    case_insensitive: bool,
) -> Option<&'a OsString> {
    values.iter().find_map(|(key, value)| {
        crate::config::env_key::matches_with_case(key, std::ffi::OsStr::new(name), case_insensitive)
            .then_some(value)
    })
}

/// Rename `~/.ai-chat-cli/` → `~/.cubi/` exactly once, when the new
/// location does not yet exist. No-op if the legacy directory is
/// missing, the new directory is already populated, or the rename
/// fails (we log a warning and let downstream code create a fresh
/// `~/.cubi/`).
pub fn migrate_config_dir(home: Option<&Path>) {
    let Some(home) = home else {
        return;
    };
    let new = home.join(".cubi");
    let old = home.join(".ai-chat-cli");
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

    /// Serialize every test that mutates process-global environment variables
    /// (`HOME`, `CUBI_HOME`, `USERPROFILE`, ...). Env vars are process-wide, so
    /// tests across *all* modules must share this one lock or they race under
    /// parallel `cargo test`. Poison-tolerant: a panicking test that held the
    /// lock must not cascade into failures in every other env-touching test.
    pub(crate) fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub(crate) struct EnvRestore {
        old: Vec<(&'static str, Option<OsString>)>,
    }

    impl EnvRestore {
        /// The caller holds `env_guard()` until this restore guard is dropped.
        pub(crate) fn capture(names: &[&'static str]) -> Self {
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
                // SAFETY: callers retain the shared environment lock until
                // this guard has restored every captured value.
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
        let _lock = env_guard();
        let cubi_home = tempfile::tempdir().expect("cubi home tempdir");
        let other_home = tempfile::tempdir().expect("other home tempdir");
        let _restore = EnvRestore::capture(&[
            crate::sessions::CUBI_HOME_ENV,
            "HOME",
            "USERPROFILE",
            "CUBI_OAUTH_FILE",
            "CUBI_PLUGINS_DIR",
        ]);
        // SAFETY: the shared environment lock is held through fixture use
        // and restoration, serializing tests that mutate these paths.
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
            let legacy = cubi_home.join(".ai-chat-cli");
            fs::create_dir(&legacy).unwrap();
            fs::write(
                legacy.join("config.json"),
                r#"{"default_model":"legacy-model"}"#,
            )
            .unwrap();
            let env = crate::config::Environment::capture();

            migrate_config_dir(env.home_dir().as_deref());

            let config = crate::onboarding::AppConfig::load_from_home(Some(cubi_home));
            assert_eq!(config.default_model.as_deref(), Some("legacy-model"));
            assert!(!legacy.exists());
            assert!(!other_home.join(".cubi").exists());
        });
    }

    #[test]
    fn legacy_aliases_preserve_values_and_respect_canonical_presence() {
        for (new, old) in ENV_RENAMES {
            for (canonical, expected) in [
                (None, "legacy-value"),
                (Some(""), ""),
                (Some("canonical-value"), "canonical-value"),
            ] {
                let mut values =
                    BTreeMap::from([(OsString::from(old), OsString::from("legacy-value"))]);
                if let Some(value) = canonical {
                    values.insert(OsString::from(new), OsString::from(value));
                }

                promote_legacy_env(&mut values);

                assert_eq!(
                    values.get(std::ffi::OsStr::new(new)),
                    Some(&OsString::from(expected)),
                    "{}",
                    new
                );
                assert_eq!(
                    values.get(std::ffi::OsStr::new(old)),
                    Some(&OsString::from("legacy-value"))
                );
            }
        }
    }

    #[test]
    fn legacy_aliases_select_platform_case_semantics() {
        let values = BTreeMap::from([(
            OsString::from("ai_chat_cli_model"),
            OsString::from("legacy-value"),
        )]);
        let mut sensitive = values.clone();
        let mut insensitive = values;

        promote_legacy_env_with_case(&mut sensitive, false);
        promote_legacy_env_with_case(&mut insensitive, true);

        assert!(!sensitive.contains_key(std::ffi::OsStr::new("CUBI_MODEL")));
        assert_eq!(
            insensitive.get(std::ffi::OsStr::new("CUBI_MODEL")),
            Some(&OsString::from("legacy-value"))
        );
    }

    #[test]
    fn mixed_case_canonical_name_blocks_legacy_promotion_on_windows() {
        let mut values = BTreeMap::from([
            (
                OsString::from("Cubi_Model"),
                OsString::from("canonical-value"),
            ),
            (
                OsString::from("AI_CHAT_CLI_MODEL"),
                OsString::from("legacy-value"),
            ),
        ]);

        promote_legacy_env_with_case(&mut values, true);

        assert!(!values.contains_key(std::ffi::OsStr::new("CUBI_MODEL")));
        assert_eq!(
            values.get(std::ffi::OsStr::new("Cubi_Model")),
            Some(&OsString::from("canonical-value"))
        );
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
