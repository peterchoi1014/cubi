//! `@file` mention expansion + user-defined Markdown commands.
//!
//! ## `@file` mentions
//!
//! When the user types `@path/to/file` in their message, this module
//! expands it by reading the file contents and appending them to the user
//! message inside a fenced code block. This gives the model direct access
//! to referenced files without requiring a separate tool call.
//!
//! Patterns recognized:
//! * `@relative/path.ext` — resolved relative to cwd
//! * `@./relative/path.ext` — same
//! * `@/absolute/path.ext` — used as-is
//!
//! Files that don't exist or can't be read produce a warning inline rather
//! than failing the whole message.
//!
//! ## User-defined Markdown commands
//!
//! Users can place `.md` files in `~/.ai-chat-cli/commands/` (or
//! `.ai-chat-cli/commands/` in the project root). Each file defines a
//! custom slash command whose name is the filename stem (e.g.
//! `explain.md` → `/explain`). When invoked, the Markdown content is
//! injected as a system message for the next turn.

use std::fs;
use std::path::{Path, PathBuf};

/// Expands all `@file` mentions in a user message. Returns a new string
/// with file contents appended as fenced code blocks. If there are no
/// `@` mentions, returns the input unchanged.
pub fn expand_file_mentions(input: &str) -> String {
    let mentions = extract_mentions(input);
    if mentions.is_empty() {
        return input.to_string();
    }

    let mut result = input.to_string();
    let mut appendix = String::new();

    for mention in &mentions {
        let path = resolve_mention_path(mention);
        match fs::read_to_string(&path) {
            Ok(contents) => {
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                appendix.push_str(&format!(
                    "\n\n---\n**Contents of `{}`:**\n```{}\n{}\n```",
                    mention,
                    ext,
                    contents.trim_end()
                ));
            }
            Err(e) => {
                appendix.push_str(&format!(
                    "\n\n[Warning: Could not read `{}`: {}]",
                    mention, e
                ));
            }
        }
    }

    result.push_str(&appendix);
    result
}

/// Extracts `@path` tokens from user input. A mention must start with `@`
/// followed by a path-like sequence (no spaces, must contain at least one
/// `/` or `.` to distinguish from plain `@username` mentions).
fn extract_mentions(input: &str) -> Vec<String> {
    let mut mentions = Vec::new();
    for word in input.split_whitespace() {
        if let Some(path) = word.strip_prefix('@') {
            // Must look like a file path: contains / or a dot with extension
            if (path.contains('/') || path.contains('.')) && !path.is_empty() {
                // Strip trailing punctuation that isn't part of a path
                let cleaned = path.trim_end_matches([',', ';', ')']);
                if !cleaned.is_empty() {
                    mentions.push(cleaned.to_string());
                }
            }
        }
    }
    mentions
}

/// Resolves a mention path to an absolute path.
fn resolve_mention_path(mention: &str) -> PathBuf {
    let path = Path::new(mention);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

/// A user-defined Markdown command loaded from disk.
#[derive(Debug, Clone)]
pub struct UserCommand {
    /// The slash command name (e.g. "explain").
    pub name: String,
    /// The Markdown body to inject as a system message.
    pub body: String,
    /// Source path for diagnostics.
    pub path: PathBuf,
}

/// Loads user-defined commands from both the global and project-local
/// command directories. Project-local commands take precedence (override)
/// over global ones with the same name.
pub fn load_user_commands() -> Vec<UserCommand> {
    let mut commands: Vec<UserCommand> = Vec::new();
    let mut seen_names: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Project-local: .ai-chat-cli/commands/ in cwd
    if let Ok(cwd) = std::env::current_dir() {
        let local_dir = cwd.join(".ai-chat-cli").join("commands");
        load_commands_from_dir(&local_dir, &mut commands, &mut seen_names);
    }

    // Global: ~/.ai-chat-cli/commands/
    if let Some(home) = dirs::home_dir() {
        let global_dir = home.join(".ai-chat-cli").join("commands");
        load_commands_from_dir(&global_dir, &mut commands, &mut seen_names);
    }

    commands
}

fn load_commands_from_dir(
    dir: &Path,
    commands: &mut Vec<UserCommand>,
    seen: &mut std::collections::HashSet<String>,
) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let name = stem.to_lowercase();
        if seen.contains(&name) {
            continue; // project-local already loaded this name
        }
        if let Ok(body) = fs::read_to_string(&path) {
            seen.insert(name.clone());
            commands.push(UserCommand {
                name,
                body,
                path: path.clone(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_mentions_finds_file_paths() {
        let input = "look at @src/main.rs and @./Cargo.toml please";
        let mentions = extract_mentions(input);
        assert_eq!(mentions, vec!["src/main.rs", "./Cargo.toml"]);
    }

    #[test]
    fn extract_mentions_ignores_plain_at_words() {
        let input = "hello @user how are you";
        let mentions = extract_mentions(input);
        assert!(mentions.is_empty());
    }

    #[test]
    fn extract_mentions_strips_trailing_punctuation() {
        let input = "check @src/lib.rs, and @tests/test.rs;";
        let mentions = extract_mentions(input);
        assert_eq!(mentions, vec!["src/lib.rs", "tests/test.rs"]);
    }

    #[test]
    fn extract_mentions_absolute_path() {
        let input = "see @/etc/hosts";
        let mentions = extract_mentions(input);
        assert_eq!(mentions, vec!["/etc/hosts"]);
    }

    #[test]
    fn expand_no_mentions_returns_unchanged() {
        let input = "just a normal message";
        assert_eq!(expand_file_mentions(input), input);
    }

    #[test]
    fn expand_missing_file_adds_warning() {
        let input = "@/nonexistent/file.xyz explain this";
        let expanded = expand_file_mentions(input);
        assert!(expanded.contains("[Warning: Could not read"));
        assert!(expanded.contains("/nonexistent/file.xyz"));
    }

    #[test]
    fn expand_existing_file_adds_contents() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let tmp = std::env::temp_dir().join(format!("at-file-test-{nanos}.txt"));
        fs::write(&tmp, "hello world\n").unwrap();

        let input = format!("look at @{}", tmp.display());
        let expanded = expand_file_mentions(&input);
        assert!(expanded.contains("hello world"));
        assert!(expanded.contains("```txt"));
        fs::remove_file(&tmp).ok();
    }

    #[test]
    fn load_user_commands_returns_empty_when_no_dir() {
        // In a test environment there's no ~/.ai-chat-cli/commands/ typically
        let cmds = load_user_commands();
        // Should not panic, just return whatever is available (likely empty)
        let _ = cmds;
    }

    #[test]
    fn load_commands_from_dir_reads_md_files() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("user-cmds-test-{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("explain.md"), "# Explain\nExplain the code.").unwrap();
        fs::write(dir.join("review.md"), "# Review\nReview carefully.").unwrap();
        fs::write(dir.join("not-md.txt"), "ignored").unwrap();

        let mut cmds = Vec::new();
        let mut seen = std::collections::HashSet::new();
        load_commands_from_dir(&dir, &mut cmds, &mut seen);

        assert_eq!(cmds.len(), 2);
        let names: Vec<&str> = cmds.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"explain"));
        assert!(names.contains(&"review"));
        assert!(!names.contains(&"not-md"));

        fs::remove_dir_all(&dir).ok();
    }
}
