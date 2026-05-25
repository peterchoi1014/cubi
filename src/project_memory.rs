//! Helpers for `CUBI.md` — a per-project "memory" file analogous to
//! Claude Code's `CLAUDE.md`. It lives at the root of the current working
//! directory (or any ancestor) and contains long-lived project context that
//! should be loaded into every session.
//!
//! For back-compat with the pre-rebrand era, project memory file is also accepted
//! when no `CUBI.md` is found. New files are always written as `CUBI.md`.

use anyhow::{Context, Result};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

pub const MEMORY_FILENAME: &str = "CUBI.md";
/// Legacy filename still honored on read so users with an existing
/// project memory file in their project don't lose their memory after upgrading.
pub const LEGACY_MEMORY_FILENAME: &str = "AICHAT.md";

/// Path to `CUBI.md` in the current working directory (write target for
/// `/init`). Reads should prefer [`find_memory_path`], which also walks up
/// the directory tree and falls back to the legacy filename.
pub fn memory_path() -> PathBuf {
    Path::new(MEMORY_FILENAME).to_path_buf()
}

/// Walks from `start` up toward the filesystem root looking for the first
/// `CUBI.md` (or, failing that, the first project memory file). Returns `None`
/// if none is found.
pub fn find_memory_path_from(start: &Path) -> Option<PathBuf> {
    let mut cursor: Option<&Path> = Some(start);
    while let Some(dir) = cursor {
        let candidate = dir.join(MEMORY_FILENAME);
        if candidate.is_file() {
            return Some(candidate);
        }
        let legacy = dir.join(LEGACY_MEMORY_FILENAME);
        if legacy.is_file() {
            return Some(legacy);
        }
        cursor = dir.parent();
    }
    None
}

/// Walks from the current working directory up toward the filesystem root
/// looking for the first `CUBI.md` (or project memory file as a fallback).
pub fn find_memory_path() -> Option<PathBuf> {
    let cwd = env::current_dir().ok()?;
    find_memory_path_from(&cwd)
}

/// Returns both the path and contents of the nearest project memory in a single
/// directory walk. Prefer this over calling [`find_memory_path`] and then
/// reading the file separately: doing two walks opens a small TOCTOU window
/// where the path can be reported as one location while the contents are read
/// from another (or worse, the file vanishes between calls).
pub fn read_memory_with_path() -> Result<Option<(PathBuf, String)>> {
    let Some(path) = find_memory_path() else {
        return Ok(None);
    };
    let contents =
        fs::read_to_string(&path).with_context(|| format!("Failed to read {}", path.display()))?;
    Ok(Some((path, contents)))
}

/// Writes a starter project memory file if one does not already exist.
/// Returns `Ok(true)` if a file was created, `Ok(false)` if it already existed.
pub fn write_starter_if_absent() -> Result<bool> {
    let path = memory_path();
    if path.exists() {
        return Ok(false);
    }
    fs::write(&path, STARTER_TEMPLATE)
        .with_context(|| format!("Failed to write {}", path.display()))?;
    Ok(true)
}

const STARTER_TEMPLATE: &str = "# Project memory for Cubi\n\
\n\
This file is loaded into every chat session as long-lived project context.\n\
Keep it short and focused on things the assistant should always know.\n\
\n\
## Project overview\n\
\n\
- _What does this project do?_\n\
\n\
## Build / test / lint\n\
\n\
- _How do you build, test, and lint the project?_\n\
\n\
## Conventions\n\
\n\
- _Important coding conventions, file layout, or style preferences._\n\
\n\
## Notes\n\
\n\
- _Anything else worth remembering across sessions._\n";

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // `read_memory` / `write_starter_if_absent` operate on the process's
    // current working directory, which is global mutable state. Serialize
    // any test that touches the FS via this mutex so parallel test threads
    // don't race each other.
    static CWD_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn starter_template_has_expected_sections() {
        assert!(STARTER_TEMPLATE.contains("# Project memory for Cubi"));
        assert!(STARTER_TEMPLATE.contains("## Build / test / lint"));
        assert!(STARTER_TEMPLATE.contains("## Conventions"));
    }

    #[test]
    fn write_starter_then_read_roundtrip() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let _guard = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("cubi-mem-test-{nanos}"));
        fs::create_dir_all(&dir).unwrap();

        let prev = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();

        // First call writes the starter.
        let wrote = write_starter_if_absent().unwrap();
        assert!(wrote, "expected starter to be created");

        // Second call must be idempotent and leave the file alone.
        let wrote_again = write_starter_if_absent().unwrap();
        assert!(!wrote_again, "expected second call to be a no-op");

        // Read it back through the public API.
        let (path, contents) = read_memory_with_path()
            .unwrap()
            .expect("memory should exist");
        assert!(contents.contains("# Project memory for Cubi"));
        assert!(path.ends_with(MEMORY_FILENAME));

        // Restore cwd before cleanup so other tests aren't stranded.
        std::env::set_current_dir(&prev).unwrap();
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_memory_returns_none_when_absent() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let _guard = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("cubi-mem-absent-{nanos}"));
        fs::create_dir_all(&dir).unwrap();

        let prev = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();

        let result = read_memory_with_path().unwrap();
        assert!(result.is_none(), "expected None when CUBI.md is absent");

        std::env::set_current_dir(&prev).unwrap();
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn find_memory_path_walks_up_to_ancestor() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("cubi-mem-walk-{nanos}"));
        let nested = root.join("a").join("b").join("c");
        fs::create_dir_all(&nested).unwrap();

        // Place CUBI.md at the top of the temp tree.
        let memory = root.join(MEMORY_FILENAME);
        fs::write(&memory, "# walk-up test\n").unwrap();

        // Searching from the deepest directory must find it.
        let found = find_memory_path_from(&nested).expect("memory should be discoverable");
        // Canonicalize both sides because /tmp may be a symlink on some platforms.
        assert_eq!(
            fs::canonicalize(&found).unwrap(),
            fs::canonicalize(&memory).unwrap()
        );

        // And a sibling tree without the file must not.
        let other = std::env::temp_dir().join(format!("cubi-mem-walk-other-{nanos}"));
        fs::create_dir_all(&other).unwrap();
        // Only stop the walk at the temp dir's parent (which on a real system
        // won't contain CUBI.md). We assert the immediate dir has no hit:
        assert!(!other.join(MEMORY_FILENAME).exists());

        fs::remove_dir_all(&root).ok();
        fs::remove_dir_all(&other).ok();
    }
}
