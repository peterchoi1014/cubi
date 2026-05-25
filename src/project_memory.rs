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
    use super::STARTER_TEMPLATE;

    #[test]
    fn starter_template_has_expected_sections() {
        assert!(STARTER_TEMPLATE.contains("# Project memory for ai-chat-cli"));
        assert!(STARTER_TEMPLATE.contains("## Build / test / lint"));
        assert!(STARTER_TEMPLATE.contains("## Conventions"));
    }
}
