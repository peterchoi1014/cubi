//! Cross-session persistent memory (`memdir`).
//!
//! Implements a global memory store at `~/.cubi/memdir/` where each
//! "memory" is a short text snippet that the user (or the model) saves for
//! future sessions. Unlike `CUBI.md` (which is per-project), memdir is
//! global and accumulates facts across all projects.
//!
//! Storage: one JSON file (`memories.json`) containing a flat array of
//! [`Memory`] structs. The file is created lazily on first write.
//!
//! CLI surface:
//! * `/memdir` — list all memories
//! * `/memdir-add <text>` — add a memory
//! * `/memdir-rm <n>` — remove memory by 1-based index
//! * `/memdir-clear` — remove all memories
//!
//! The model can also use the `memdir_write` built-in tool to persist
//! memories it deems important for future sessions.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// A single persisted memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    /// The text content of the memory.
    pub text: String,
    /// Unix timestamp (seconds) when the memory was created.
    pub created_at: u64,
    /// Optional source context (e.g. project path, session id).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// The in-memory representation of the memdir store.
#[derive(Debug, Clone, Default)]
pub struct Memdir {
    memories: Vec<Memory>,
    path: Option<PathBuf>,
}

impl Memdir {
    /// Loads the global memdir from `~/.cubi/memdir/memories.json`.
    /// Returns an empty store if the file doesn't exist yet.
    pub fn load() -> Self {
        let path = match Self::storage_path() {
            Some(p) => p,
            None => return Self::default(),
        };
        let memories = if path.exists() {
            match fs::read_to_string(&path) {
                Ok(raw) => serde_json::from_str(&raw).unwrap_or_default(),
                Err(_) => Vec::new(),
            }
        } else {
            Vec::new()
        };
        Self {
            memories,
            path: Some(path),
        }
    }

    /// Creates a memdir backed by a specific path (for testing).
    #[cfg(test)]
    pub fn with_path(path: PathBuf) -> Self {
        let memories = if path.exists() {
            match fs::read_to_string(&path) {
                Ok(raw) => serde_json::from_str(&raw).unwrap_or_default(),
                Err(_) => Vec::new(),
            }
        } else {
            Vec::new()
        };
        Self {
            memories,
            path: Some(path),
        }
    }

    /// Adds a memory with the given text and optional source.
    pub fn add(&mut self, text: &str, source: Option<&str>) {
        let text = text.trim();
        if text.is_empty() {
            return;
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.memories.push(Memory {
            text: text.to_string(),
            created_at: now,
            source: source.map(|s| s.to_string()),
        });
    }

    /// Removes a memory by 1-based index. Returns `true` if an item was
    /// removed.
    pub fn remove(&mut self, one_based: usize) -> bool {
        if one_based == 0 || one_based > self.memories.len() {
            return false;
        }
        self.memories.remove(one_based - 1);
        true
    }

    /// Clears all memories.
    pub fn clear(&mut self) {
        self.memories.clear();
    }

    /// Returns the number of stored memories.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.memories.len()
    }

    /// Returns `true` if there are no memories.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }

    /// Returns all memories as a slice.
    #[allow(dead_code)]
    pub fn list(&self) -> &[Memory] {
        &self.memories
    }

    /// Renders the memory list to stdout.
    pub fn render(&self) {
        use colored::*;
        if self.memories.is_empty() {
            println!(
                "{}",
                "No memories stored. Add one with /memdir-add <text>".yellow()
            );
            return;
        }
        println!("\n{}", "Persistent Memories:".bright_yellow().bold());
        for (i, mem) in self.memories.iter().enumerate() {
            let source_info = mem
                .source
                .as_deref()
                .map(|s| format!(" (from: {})", s))
                .unwrap_or_default();
            println!(
                "  {}. {}{}",
                i + 1,
                mem.text.bright_white(),
                source_info.bright_black()
            );
        }
        println!();
    }

    /// Persists the current state to disk. No-ops if no path is set.
    pub fn save(&self) -> Result<()> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create {}", parent.display()))?;
        }
        let json = serde_json::to_string_pretty(&self.memories)?;
        fs::write(path, json).with_context(|| format!("Failed to write {}", path.display()))?;
        Ok(())
    }

    /// Returns the combined text of all memories, suitable for injection
    /// into the model's system context. Returns just the bullet list (the
    /// caller provides the header/prefix).
    pub fn as_context_string(&self) -> Option<String> {
        if self.memories.is_empty() {
            return None;
        }
        let mut out = String::new();
        for mem in &self.memories {
            out.push_str(&format!("- {}\n", mem.text));
        }
        Some(out)
    }

    /// Resolves the path to `~/.cubi/memdir/memories.json`.
    fn storage_path() -> Option<PathBuf> {
        let home = dirs::home_dir()?;
        Some(home.join(".cubi").join("memdir").join("memories.json"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn tmp_path(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("cubi-memdir-{label}-{nanos}.json"))
    }

    #[test]
    fn add_and_list() {
        let mut m = Memdir::with_path(tmp_path("add"));
        assert!(m.is_empty());
        m.add("remember this", None);
        assert_eq!(m.len(), 1);
        assert_eq!(m.list()[0].text, "remember this");
    }

    #[test]
    fn add_ignores_empty() {
        let mut m = Memdir::with_path(tmp_path("empty"));
        m.add("", None);
        m.add("   ", None);
        assert!(m.is_empty());
    }

    #[test]
    fn remove_one_based() {
        let mut m = Memdir::with_path(tmp_path("rm"));
        m.add("a", None);
        m.add("b", None);
        m.add("c", None);
        assert!(!m.remove(0));
        assert!(!m.remove(4));
        assert!(m.remove(2));
        assert_eq!(m.len(), 2);
        assert_eq!(m.list()[0].text, "a");
        assert_eq!(m.list()[1].text, "c");
    }

    #[test]
    fn save_and_reload() {
        let path = tmp_path("roundtrip");
        let mut m = Memdir::with_path(path.clone());
        m.add("fact one", Some("project-x"));
        m.add("fact two", None);
        m.save().unwrap();

        let reloaded = Memdir::with_path(path.clone());
        assert_eq!(reloaded.len(), 2);
        assert_eq!(reloaded.list()[0].text, "fact one");
        assert_eq!(reloaded.list()[0].source.as_deref(), Some("project-x"));
        assert_eq!(reloaded.list()[1].text, "fact two");

        fs::remove_file(&path).ok();
    }

    #[test]
    fn clear_and_context_string_track_len() {
        let mut m = Memdir::with_path(tmp_path("ctx"));
        m.add("remember alpha", None);
        m.add("remember beta", Some("src"));
        assert_eq!(m.len(), 2);
        assert_eq!(
            m.as_context_string().as_deref(),
            Some("- remember alpha\n- remember beta\n")
        );
        m.clear();
        assert_eq!(m.len(), 0);
        assert!(m.as_context_string().is_none());
    }

    #[test]
    fn clear_empties_list() {
        let mut m = Memdir::with_path(tmp_path("clear"));
        m.add("x", None);
        m.clear();
        assert!(m.is_empty());
    }

    #[test]
    fn as_context_string_none_when_empty() {
        let m = Memdir::with_path(tmp_path("ctx"));
        assert!(m.as_context_string().is_none());
    }

    #[test]
    fn as_context_string_formats_correctly() {
        let mut m = Memdir::with_path(tmp_path("ctx2"));
        m.add("use tabs", None);
        m.add("prefer async", None);
        let ctx = m.as_context_string().unwrap();
        assert!(ctx.contains("- use tabs"));
        assert!(ctx.contains("- prefer async"));
    }
}
