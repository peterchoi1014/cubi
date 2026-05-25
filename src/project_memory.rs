//! Helpers for `AICHAT.md` — a per-project "memory" file analogous to
//! Claude Code's `CLAUDE.md`. It lives at the root of the current working
//! directory and contains long-lived project context that should be loaded
//! into every session.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

pub const MEMORY_FILENAME: &str = "AICHAT.md";

/// Path to `AICHAT.md` in the current working directory.
pub fn memory_path() -> PathBuf {
    Path::new(MEMORY_FILENAME).to_path_buf()
}

/// Returns the contents of `AICHAT.md` if it exists.
pub fn read_memory() -> Result<Option<String>> {
    let path = memory_path();
    if !path.exists() {
        return Ok(None);
    }
    let contents = fs::read_to_string(&path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    Ok(Some(contents))
}

/// Writes a starter `AICHAT.md` if one does not already exist.
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

const STARTER_TEMPLATE: &str = "# Project memory for ai-chat-cli\n\
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
        assert!(STARTER_TEMPLATE.contains("# Project memory for ai-chat-cli"));
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
        let dir = std::env::temp_dir().join(format!("ai-chat-cli-mem-test-{nanos}"));
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
        let contents = read_memory().unwrap().expect("memory should exist");
        assert!(contents.contains("# Project memory for ai-chat-cli"));

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
        let dir = std::env::temp_dir().join(format!("ai-chat-cli-mem-absent-{nanos}"));
        fs::create_dir_all(&dir).unwrap();

        let prev = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();

        let result = read_memory().unwrap();
        assert!(result.is_none(), "expected None when AICHAT.md is absent");

        std::env::set_current_dir(&prev).unwrap();
        fs::remove_dir_all(&dir).ok();
    }
}
