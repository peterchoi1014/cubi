//! Versioned config & session migrations.
//!
//! Roadmap item C#22: a thin forward-only migrator that bumps any old
//! `~/.ai-chat-cli/config.json` to the latest schema and back-fills any
//! fields that were added in newer versions. Old config files keep
//! working because [`crate::onboarding::AppConfig`] uses
//! `#[serde(default)]` everywhere; this module's job is to (a) record
//! that the migration happened by stamping `config_version` and (b)
//! give us a single place to add structural migrations later (e.g. if
//! `vim_mode: String` ever becomes `vim_mode: VimMode`).

use crate::onboarding::AppConfig;

/// Current on-disk schema version. Bump in lockstep with structural
/// changes to [`AppConfig`].
pub const CURRENT_CONFIG_VERSION: u32 = 1;

/// Returns `true` if the config was modified by migration and should be
/// re-saved. Always idempotent — running this twice in a row is a no-op
/// on the second call.
pub fn migrate_config(cfg: &mut AppConfig) -> bool {
    let mut changed = false;

    // v0 → v1: introduce `theme`, `output_style`, `color`, `vim_mode`,
    // `telemetry`, `config_version`. All have safe defaults via serde,
    // so the only state we need to touch is the version stamp itself.
    if cfg.config_version == 0 {
        cfg.config_version = 1;
        changed = true;
    }

    // Forward compat: if a *newer* binary wrote `config_version`
    // greater than we recognise, leave the data and version untouched.
    // This older binary should not "clamp" the version down, because
    // that can mask that the config may contain fields we don't know.
    if cfg.config_version > CURRENT_CONFIG_VERSION {
        // Intentionally do not lower it — that would let an older
        // binary silently drop fields it doesn't understand.
    }

    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v0_config_is_stamped_to_v1() {
        let mut cfg = AppConfig::default();
        assert_eq!(cfg.config_version, 0);
        let changed = migrate_config(&mut cfg);
        assert!(changed);
        assert_eq!(cfg.config_version, CURRENT_CONFIG_VERSION);
    }

    #[test]
    fn migration_is_idempotent() {
        let mut cfg = AppConfig {
            config_version: CURRENT_CONFIG_VERSION,
            ..AppConfig::default()
        };
        assert!(!migrate_config(&mut cfg));
        assert_eq!(cfg.config_version, CURRENT_CONFIG_VERSION);
    }

    #[test]
    fn future_version_is_preserved() {
        let mut cfg = AppConfig {
            config_version: 99,
            ..AppConfig::default()
        };
        let changed = migrate_config(&mut cfg);
        assert!(!changed);
        // We do NOT downgrade — a newer binary's data must round-trip
        // through an older one without silently losing the version.
        assert_eq!(cfg.config_version, 99);
    }
}
