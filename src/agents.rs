//! Managed custom-agent *definitions* (`~/.cubi/agents/*.md`).
//!
//! An agent definition is a Markdown file with an optional YAML-style
//! frontmatter block delimited by `---` lines at the very top:
//!
//! ```text
//! ---
//! name: reviewer
//! description: Adversarial code reviewer
//! model: qwen3:8b
//! ---
//! You are a meticulous code reviewer. ...
//! ```
//!
//! Everything after the closing `---` is the agent's system-prompt body.
//! When a file has no frontmatter the whole file is treated as the prompt,
//! the `name` is derived from the filename stem, and the `description` is the
//! first non-empty line — mirroring [`crate::skills`].
//!
//! Parsing is done by hand (a couple of `key: value` lines); this module
//! deliberately introduces **no** YAML dependency.
//!
//! NOTE: these are definitions only. Defined agents are not runnable yet — the
//! `/agents` command manages (CRUD) them.

use anyhow::{Result, bail};
use std::fs;
use std::path::PathBuf;

/// A single parsed agent definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentDef {
    pub name: String,
    pub description: String,
    pub model: Option<String>,
    pub prompt: String,
    pub path: PathBuf,
}

/// Directory that holds agent definition files: `~/.cubi/agents`.
/// Returns `None` when no home directory is available (mirrors
/// [`crate::skills::skills_dir`]).
pub fn agents_dir() -> Option<PathBuf> {
    crate::sessions::cubi_dir().map(|d| d.join("agents"))
}

/// Returns `true` when `name` is a safe single filename component: non-empty,
/// with no path separators, no parent-directory traversal, and no NUL. Used to
/// keep `/agents` operations confined to `~/.cubi/agents`.
fn is_safe_name(name: &str) -> bool {
    let name = name.trim();
    if name.is_empty() || name == "." || name == ".." {
        return false;
    }
    if name.contains('/') || name.contains('\\') || name.contains("..") || name.contains('\0') {
        return false;
    }
    true
}

/// Resolve the on-disk path for the agent called `name` (lowercased, `.md`
/// suffix) under [`agents_dir`]. Returns `None` when the name is unsafe or no
/// home directory is available.
pub fn agent_path(name: &str) -> Option<PathBuf> {
    if !is_safe_name(name) {
        return None;
    }
    let file = format!("{}.md", name.trim().to_ascii_lowercase());
    agents_dir().map(|d| d.join(file))
}

/// Parse the raw contents of an agent file into an [`AgentDef`]. `path` is used
/// for the returned struct and to derive the fallback name. Returns `None` when
/// no usable name can be determined.
fn parse_agent(path: PathBuf, raw: &str) -> Option<AgentDef> {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    let mut name: Option<String> = None;
    let mut description: Option<String> = None;
    let mut model: Option<String> = None;
    let mut prompt = raw.to_string();

    // Detect frontmatter: the first non-empty content must start with a `---`
    // delimiter line. We tolerate a leading BOM / blank lines before it.
    let mut lines = raw.lines();
    if lines.next().map(str::trim) == Some("---") {
        let mut body_start: Option<usize> = None;
        // Re-walk with indices so we can slice the prompt after the close.
        let indexed: Vec<&str> = raw.lines().collect();
        for (idx, line) in indexed.iter().enumerate().skip(1) {
            if line.trim() == "---" {
                body_start = Some(idx + 1);
                break;
            }
            if let Some((key, value)) = line.split_once(':') {
                let key = key.trim().to_ascii_lowercase();
                let value = value.trim();
                match key.as_str() {
                    "name" if !value.is_empty() => name = Some(value.to_ascii_lowercase()),
                    "description" if !value.is_empty() => description = Some(value.to_string()),
                    "model" if !value.is_empty() => model = Some(value.to_string()),
                    _ => {}
                }
            }
        }
        if let Some(start) = body_start {
            prompt = indexed
                .get(start..)
                .map(|rest| rest.join("\n"))
                .unwrap_or_default();
        }
    }

    let name = name.unwrap_or(stem);
    if name.is_empty() {
        return None;
    }

    // Fall back to the first non-empty prompt line for the description, mirror
    // `skills.rs` (strip a leading "# " heading marker).
    let description = description.unwrap_or_else(|| {
        prompt
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
            .map(|line| line.trim_start_matches("# ").to_string())
            .unwrap_or_else(|| name.clone())
    });

    Some(AgentDef {
        name,
        description,
        model,
        prompt: prompt.trim_start_matches('\n').to_string(),
        path,
    })
}

/// Load every agent definition under `~/.cubi/agents`, sorted by name.
/// Unreadable or nameless files are skipped.
pub fn load_agents() -> Vec<AgentDef> {
    let Some(dir) = agents_dir() else {
        return Vec::new();
    };
    let Ok(entries) = fs::read_dir(&dir) else {
        return Vec::new();
    };

    let mut agents = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        let Ok(raw) = fs::read_to_string(&path) else {
            continue;
        };
        if let Some(def) = parse_agent(path, &raw) {
            agents.push(def);
        }
    }
    agents.sort_by(|a, b| a.name.cmp(&b.name));
    agents
}

/// Load a single agent definition by name, or `None` when it does not exist /
/// cannot be parsed / the name is unsafe.
pub fn load_one(name: &str) -> Option<AgentDef> {
    let path = agent_path(name)?;
    let raw = fs::read_to_string(&path).ok()?;
    parse_agent(path, &raw)
}

/// Create a new agent definition file for `name`, seeding a frontmatter block
/// and a placeholder prompt. `description`, `model`, and `prompt` override the
/// defaults when provided. Errors when the name is unsafe or the file already
/// exists.
pub fn create(
    name: &str,
    description: Option<&str>,
    model: Option<&str>,
    prompt: Option<&str>,
) -> Result<PathBuf> {
    let Some(path) = agent_path(name) else {
        bail!("invalid agent name '{name}' (no '/', '\\', '..', and it must not be empty)");
    };
    if path.exists() {
        bail!("agent '{}' already exists at {}", name, path.display());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let lower = name.trim().to_ascii_lowercase();
    let description = description.unwrap_or("TODO: describe this agent");
    let model_line = match model {
        Some(m) if !m.trim().is_empty() => format!("model: {}\n", m.trim()),
        _ => "model:\n".to_string(),
    };
    let prompt = prompt
        .unwrap_or("You are a custom agent.\n\nTODO: write this agent's system prompt here.\n");

    let contents =
        format!("---\nname: {lower}\ndescription: {description}\n{model_line}---\n{prompt}",);
    fs::write(&path, contents)?;
    Ok(path)
}

/// Delete the agent definition called `name`. Returns `true` when a file was
/// removed, `false` when it did not exist. Errors on unsafe names or IO errors.
pub fn delete(name: &str) -> Result<bool> {
    let Some(path) = agent_path(name) else {
        bail!("invalid agent name '{name}'");
    };
    if !path.exists() {
        return Ok(false);
    }
    fs::remove_file(&path)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agents_dir_uses_cubi_home() {
        crate::compat::test_env::with_cubi_home(|cubi_home, other_home| {
            let path = agents_dir().expect("agents dir");
            assert_eq!(path, cubi_home.join(".cubi").join("agents"));
            assert!(!path.starts_with(other_home));
        });
    }

    #[test]
    fn create_then_load_one_round_trips_frontmatter() {
        crate::compat::test_env::with_cubi_home(|_cubi_home, _other| {
            let path = create(
                "Reviewer",
                Some("Adversarial code reviewer"),
                Some("qwen3:8b"),
                Some("You are a meticulous reviewer.\n"),
            )
            .expect("create");
            assert!(path.exists());
            // Stored under a lowercased filename.
            assert_eq!(path.file_name().unwrap().to_str().unwrap(), "reviewer.md");

            let def = load_one("reviewer").expect("load_one");
            assert_eq!(def.name, "reviewer");
            assert_eq!(def.description, "Adversarial code reviewer");
            assert_eq!(def.model.as_deref(), Some("qwen3:8b"));
            assert_eq!(def.prompt, "You are a meticulous reviewer.");

            // Case-insensitive lookup resolves to the same file.
            assert_eq!(load_one("REVIEWER"), Some(def));
        });
    }

    #[test]
    fn create_without_model_yields_none_model() {
        crate::compat::test_env::with_cubi_home(|_cubi_home, _other| {
            create("plain", None, None, None).expect("create");
            let def = load_one("plain").expect("load_one");
            assert_eq!(def.name, "plain");
            assert_eq!(def.model, None);
            assert!(!def.prompt.is_empty());
        });
    }

    #[test]
    fn create_rejects_duplicate() {
        crate::compat::test_env::with_cubi_home(|_cubi_home, _other| {
            create("dup", None, None, None).expect("first create");
            assert!(create("dup", None, None, None).is_err());
        });
    }

    #[test]
    fn name_safety_rejects_traversal() {
        assert!(agent_path("../etc/passwd").is_none());
        assert!(agent_path("foo/bar").is_none());
        assert!(agent_path("foo\\bar").is_none());
        assert!(agent_path("").is_none());
        assert!(agent_path("..").is_none());
        assert!(agent_path("ok-name").is_some());

        // create must also refuse traversal even inside a real home.
        crate::compat::test_env::with_cubi_home(|_cubi_home, _other| {
            assert!(create("../escape", None, None, None).is_err());
        });
    }

    #[test]
    fn delete_removes_file_and_reports_change() {
        crate::compat::test_env::with_cubi_home(|_cubi_home, _other| {
            create("gone", None, None, None).expect("create");
            assert!(load_one("gone").is_some());
            assert!(delete("gone").expect("delete"));
            assert!(load_one("gone").is_none());
            // Deleting a missing agent is a no-op (false), not an error.
            assert!(!delete("gone").expect("delete missing"));
        });
    }

    #[test]
    fn load_agents_is_sorted_and_skips_non_md() {
        crate::compat::test_env::with_cubi_home(|cubi_home, _other| {
            create("zebra", None, None, None).expect("create");
            create("alpha", None, None, None).expect("create");
            // A non-.md file must be ignored.
            let dir = cubi_home.join(".cubi").join("agents");
            fs::write(dir.join("notes.txt"), "ignore me").expect("write txt");

            let names: Vec<String> = load_agents().into_iter().map(|a| a.name).collect();
            assert_eq!(names, vec!["alpha".to_string(), "zebra".to_string()]);
        });
    }

    #[test]
    fn parse_without_frontmatter_derives_name_and_description() {
        crate::compat::test_env::with_cubi_home(|cubi_home, _other| {
            let dir = cubi_home.join(".cubi").join("agents");
            fs::create_dir_all(&dir).expect("mkdir");
            fs::write(
                dir.join("raw.md"),
                "# Helpful helper\n\nBody text follows.\n",
            )
            .expect("write");

            let def = load_one("raw").expect("load_one");
            assert_eq!(def.name, "raw");
            assert_eq!(def.description, "Helpful helper");
            assert!(def.model.is_none());
            assert!(def.prompt.contains("Body text follows."));
        });
    }
}
