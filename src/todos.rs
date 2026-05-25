//! Todo list, inspired by Claude Code's `TodoWriteTool`.
//!
//! Tracks a short checklist of items the user (or, in the future, the model)
//! is working through. The list is keyed by the current working directory and
//! persisted to `~/.ai-chat-cli/todos/<cwd-key>.json` so that `/todos`
//! survives restarts within the same project.

use anyhow::{Context, Result};
use colored::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::{env, fs};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoItem {
    pub text: String,
    pub done: bool,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct TodoList {
    items: Vec<TodoItem>,
    #[serde(skip)]
    storage_path: Option<PathBuf>,
}

impl TodoList {
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn pending(&self) -> usize {
        self.items.iter().filter(|i| !i.done).count()
    }

    pub fn add(&mut self, text: impl Into<String>) {
        let text = text.into();
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            self.items.push(TodoItem {
                text: trimmed.to_string(),
                done: false,
            });
        }
    }

    /// Marks the 1-based item as done. Returns `true` if the index was valid.
    pub fn mark_done(&mut self, one_based_index: usize) -> bool {
        if one_based_index == 0 {
            return false;
        }
        let idx = one_based_index - 1;
        if let Some(item) = self.items.get_mut(idx) {
            item.done = true;
            true
        } else {
            false
        }
    }

    /// Removes the 1-based item. Returns `true` if the index was valid.
    /// Lets users drop a single bad todo without nuking the whole list with
    /// `/todo-clear`.
    pub fn remove(&mut self, one_based_index: usize) -> bool {
        if one_based_index == 0 {
            return false;
        }
        let idx = one_based_index - 1;
        if idx < self.items.len() {
            self.items.remove(idx);
            true
        } else {
            false
        }
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Loads the todo list for the given working directory, persisting any
    /// subsequent mutations to the matching file under
    /// `~/.ai-chat-cli/todos/`. If no file exists yet, returns an empty list
    /// wired up to that path so the first `save` will create it.
    pub fn load_for_cwd(cwd: &Path) -> Self {
        let storage_path = storage_path_for(cwd);
        let items = storage_path
            .as_ref()
            .and_then(|p| read_list(p).ok())
            .unwrap_or_default();
        Self {
            items,
            storage_path,
        }
    }

    /// Loads the todo list for the current working directory, falling back
    /// to an unconfigured (non-persistent) list if the cwd cannot be read.
    pub fn load_for_current_dir() -> Self {
        match env::current_dir() {
            Ok(cwd) => Self::load_for_cwd(&cwd),
            Err(_) => Self::default(),
        }
    }

    /// Writes the current list to its storage path if one is configured.
    /// Errors are surfaced so callers can decide whether to log them; the
    /// in-memory list is never mutated by a failed save.
    pub fn save(&self) -> Result<()> {
        let Some(path) = &self.storage_path else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create {}", parent.display()))?;
        }
        let json = serde_json::to_string_pretty(&self.items)?;
        fs::write(path, json).with_context(|| format!("Failed to write {}", path.display()))?;
        Ok(())
    }

    /// Renders the list to stdout. Empty lists print a short hint.
    pub fn render(&self) {
        if self.items.is_empty() {
            println!("{}", "No todos. Add one with /todo-add <text>".yellow());
            return;
        }
        println!("\n{}", "Todos:".bright_yellow().bold());
        for (i, item) in self.items.iter().enumerate() {
            let marker = if item.done {
                "[x]".bright_green().to_string()
            } else {
                "[ ]".bright_white().to_string()
            };
            let text = if item.done {
                item.text.bright_black().to_string()
            } else {
                item.text.bright_white().to_string()
            };
            println!("  {} {}. {}", marker, i + 1, text);
        }
        println!();
    }

    #[cfg(test)]
    fn with_storage_path(path: PathBuf) -> Self {
        Self {
            items: Vec::new(),
            storage_path: Some(path),
        }
    }
}

/// Computes the on-disk path for a given working directory's todos.
/// Returns `None` if the home directory cannot be resolved.
fn storage_path_for(cwd: &Path) -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    Some(
        home.join(".ai-chat-cli")
            .join("todos")
            .join(format!("{}.json", cwd_key(cwd))),
    )
}

/// Produces a filesystem-safe, collision-resistant key for a cwd.
///
/// The sanitized form (alphanumerics, `-`, `.` preserved; everything else
/// replaced by `_`) is suffixed with a stable hash of the original path so
/// that two different paths whose sanitized forms collide (for example
/// `/tmp/a/b` and `/tmp/a_b`, both of which sanitize to `_tmp_a_b`) still
/// receive distinct keys.
fn cwd_key(cwd: &Path) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let raw = cwd.to_string_lossy();
    let sanitized: String = raw
        .chars()
        .map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '.' => c,
            _ => '_',
        })
        .collect();

    let mut hasher = DefaultHasher::new();
    raw.hash(&mut hasher);
    format!("{sanitized}-{:016x}", hasher.finish())
}

fn read_list(path: &Path) -> Result<Vec<TodoItem>> {
    let raw =
        fs::read_to_string(path).with_context(|| format!("Failed to read {}", path.display()))?;
    let items: Vec<TodoItem> = serde_json::from_str(&raw)
        .with_context(|| format!("Failed to parse {}", path.display()))?;
    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_ignores_empty_and_whitespace() {
        let mut list = TodoList::default();
        list.add("");
        list.add("   ");
        list.add("real item");
        assert_eq!(list.len(), 1);
        assert_eq!(list.pending(), 1);
    }

    #[test]
    fn mark_done_uses_one_based_index() {
        let mut list = TodoList::default();
        list.add("one");
        list.add("two");

        assert!(!list.mark_done(0));
        assert!(list.mark_done(1));
        assert!(!list.mark_done(3));

        assert_eq!(list.pending(), 1);
    }

    #[test]
    fn clear_empties_list() {
        let mut list = TodoList::default();
        list.add("a");
        list.add("b");
        list.clear();
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn remove_uses_one_based_index_and_shifts() {
        let mut list = TodoList::default();
        list.add("one");
        list.add("two");
        list.add("three");

        assert!(!list.remove(0));
        assert!(!list.remove(99));
        assert!(list.remove(2));

        assert_eq!(list.len(), 2);
        // "two" was removed; "three" should now be index 2.
        assert!(list.remove(2));
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = std::env::temp_dir().join(format!("ai-chat-cli-todos-test-{}", unique_suffix()));
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("todos.json");

        let mut list = TodoList::with_storage_path(file.clone());
        list.add("write tests");
        list.add("ship it");
        list.mark_done(1);
        list.save().unwrap();

        // Round-trip through `with_storage_path`'s sibling reader: parse the
        // file directly so the test does not depend on any home-dir lookup.
        let items = read_list(&file).unwrap();
        assert_eq!(items.len(), 2);
        assert!(items[0].done);
        assert!(!items[1].done);
        assert_eq!(items[1].text, "ship it");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_without_storage_path_is_noop() {
        let mut list = TodoList::default();
        list.add("ephemeral");
        // Default `new()` has no storage_path; save should silently succeed.
        list.save().unwrap();
    }

    #[test]
    fn cwd_key_distinguishes_paths() {
        let a = cwd_key(Path::new("/tmp/project-a"));
        let b = cwd_key(Path::new("/tmp/project-b"));
        assert_ne!(a, b);
        assert!(!a.contains('/'));
        assert!(!a.contains('\\'));
        assert!(!a.is_empty());
    }

    #[test]
    fn cwd_key_distinguishes_sanitization_collisions() {
        // Both inputs sanitize to `_tmp_a_b`; the hash suffix must keep them
        // apart so two real projects never share one todo file.
        let nested = cwd_key(Path::new("/tmp/a/b"));
        let flat = cwd_key(Path::new("/tmp/a_b"));
        assert_ne!(nested, flat);
        // Stability: same input always hashes the same key.
        assert_eq!(nested, cwd_key(Path::new("/tmp/a/b")));
    }

    fn unique_suffix() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("{nanos}")
    }
}
