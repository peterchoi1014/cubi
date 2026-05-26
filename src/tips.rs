//! Tip-of-the-day surface.
//!
//! Roadmap item C#20: print a short, single-line tip when the CLI starts
//! up. Tips are drawn from a small built-in pool plus any user-supplied
//! lines in `~/.cubi/tips/*.txt` (one tip per non-empty line).
//! Selection is deterministic per day so users don't get the same tip
//! twice in a single session by rolling random twice.

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const BUILTIN_TIPS: &[&str] = &[
    "Use /plan to enter read-only mode — every write/exec tool is gated.",
    "Drop a file path with `@README.md` and it'll be inlined into your next prompt.",
    "Persist cross-session notes with /memdir-add — they ride along on every turn.",
    "Run /trust once per project to enable shell, write_file, edit_file.",
    "Use /sessions to find any prior chat in this directory; /resume <id> brings it back.",
    "Use /compact to summarize old turns and reclaim context window space.",
    "Use /rewind [n] to undo the last n exchanges (file mutations roll back too).",
    "Drop user-defined slash commands as Markdown files in ~/.cubi/plugins/<name>/commands/.",
    "Use /skills to see the reusable Markdown skill packs the agent can load on demand.",
    "Use /commit-push-pr to commit, push, and open a PR in one go.",
    "Use /diff to inspect uncommitted changes without leaving the chat.",
    "Use /worktree add <path> to spin up an isolated branch checkout.",
    "Enable opt-in debug logging with telemetry=true in ~/.cubi/config.json.",
    "Use /output-style concise|markdown|explanatory to switch reply formatting.",
];

pub fn tips_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".cubi").join("tips"))
}

/// Returns the tip for today, or `None` if no tips are available. The
/// selection rotates daily based on the system clock so cron-scheduled
/// invocations also see variety.
pub fn tip_of_the_day() -> Option<String> {
    let mut all: Vec<String> = BUILTIN_TIPS.iter().map(|s| s.to_string()).collect();
    if let Some(dir) = tips_dir() {
        if let Ok(entries) = fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) != Some("txt") {
                    continue;
                }
                let Ok(body) = fs::read_to_string(&path) else {
                    continue;
                };
                for line in body.lines().map(str::trim).filter(|l| !l.is_empty()) {
                    all.push(line.to_string());
                }
            }
        }
    }
    if all.is_empty() {
        return None;
    }
    let day = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() / 86_400)
        .unwrap_or(0);
    let idx = (day as usize) % all.len();
    Some(all[idx].clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_tip_is_returned_when_no_user_dir() {
        // Even with no $HOME/.cubi/tips, we should always get a
        // tip from the built-in pool.
        assert!(tip_of_the_day().is_some());
    }
}
