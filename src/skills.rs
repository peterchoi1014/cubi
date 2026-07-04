use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub body: String,
    pub path: PathBuf,
}

/// Path to the JSON file that persists the set of disabled skill names,
/// `~/.cubi/skills-disabled.json`. Returns `None` when no home directory is
/// available (mirrors [`skills_dir`]).
pub fn disabled_path() -> Option<PathBuf> {
    crate::sessions::cubi_dir().map(|d| d.join("skills-disabled.json"))
}

/// Load the set of disabled skill names (lowercased, matching [`Skill::name`]).
/// A missing file means nothing is disabled, so callers stay backward
/// compatible with installs that predate this feature.
pub fn load_disabled() -> BTreeSet<String> {
    let Some(path) = disabled_path() else {
        return BTreeSet::new();
    };
    let Ok(content) = fs::read_to_string(&path) else {
        return BTreeSet::new();
    };
    let names: Vec<String> = serde_json::from_str(&content).unwrap_or_default();
    names.into_iter().map(|n| n.to_ascii_lowercase()).collect()
}

/// Persist the set of disabled skill names to `~/.cubi/skills-disabled.json`.
pub fn save_disabled(disabled: &BTreeSet<String>) -> Result<()> {
    let path = disabled_path().context("Could not find home directory")?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let names: Vec<&String> = disabled.iter().collect();
    let json = serde_json::to_string_pretty(&names)?;
    fs::write(&path, json)?;
    Ok(())
}

/// Returns `true` when the named skill is currently disabled.
pub fn is_disabled(name: &str) -> bool {
    load_disabled().contains(&name.to_ascii_lowercase())
}

/// Mark a skill as disabled. Returns `true` when the state actually changed
/// (i.e. the skill was previously enabled).
pub fn disable(name: &str) -> Result<bool> {
    let mut disabled = load_disabled();
    let changed = disabled.insert(name.to_ascii_lowercase());
    if changed {
        save_disabled(&disabled)?;
    }
    Ok(changed)
}

/// Mark a skill as enabled (remove it from the disabled set). Returns `true`
/// when the state actually changed (i.e. the skill was previously disabled).
pub fn enable(name: &str) -> Result<bool> {
    let mut disabled = load_disabled();
    let changed = disabled.remove(&name.to_ascii_lowercase());
    if changed {
        save_disabled(&disabled)?;
    }
    Ok(changed)
}

pub fn load_skills() -> Vec<Skill> {
    let Some(dir) = skills_dir() else {
        return Vec::new();
    };
    let Ok(entries) = fs::read_dir(&dir) else {
        return Vec::new();
    };

    let mut skills = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        let Ok(body) = fs::read_to_string(&path) else {
            continue;
        };
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        if name.is_empty() {
            continue;
        }
        let description = body
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
            .map(|line| line.trim_start_matches("# ").to_string())
            .unwrap_or_else(|| name.clone());
        skills.push(Skill {
            name,
            description,
            body,
            path,
        });
    }
    skills.sort_by(|a, b| a.name.cmp(&b.name));
    skills
}

pub fn skills_dir() -> Option<PathBuf> {
    crate::sessions::cubi_dir().map(|d| d.join("skills"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skills_dir_uses_cubi_home() {
        crate::compat::test_env::with_cubi_home(|cubi_home, other_home| {
            let path = skills_dir().expect("skills dir");
            assert_eq!(path, cubi_home.join(".cubi").join("skills"));
            assert!(!path.starts_with(other_home));
        });
    }

    #[test]
    fn disabled_store_round_trips() {
        crate::compat::test_env::with_cubi_home(|cubi_home, _other_home| {
            // Missing file = nothing disabled (backward compatible).
            assert!(load_disabled().is_empty());
            assert!(!is_disabled("demo"));

            // disable() persists and reports a real change.
            assert!(disable("Demo").expect("disable"));
            assert!(is_disabled("demo"));
            assert!(
                cubi_home
                    .join(".cubi")
                    .join("skills-disabled.json")
                    .exists()
            );
            // A fresh load reflects the persisted (lowercased) state.
            assert!(load_disabled().contains("demo"));
            // Re-disabling is idempotent (no change).
            assert!(!disable("demo").expect("disable idempotent"));

            // enable() persists the removal and reports the change.
            assert!(enable("DEMO").expect("enable"));
            assert!(!is_disabled("demo"));
            assert!(load_disabled().is_empty());
            // Re-enabling an already-enabled skill is a no-op.
            assert!(!enable("demo").expect("enable idempotent"));
        });
    }
}
