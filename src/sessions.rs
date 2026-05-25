//! Auto-saved chat sessions + `/sessions` + `/resume`.
//!
//! Until this module landed, the only persistence for a conversation was
//! `/save <file>` — and you only ever ran it after losing one. This
//! module turns persistence on by default: every successful assistant
//! turn appends the entire `history` to a per-project JSON checkpoint at
//! `~/.ai-chat-cli/sessions/<cwd-key>/<id>.json`.
//!
//! Layout chosen deliberately:
//!
//! * One file per session (not append-only NDJSON) so that overwriting
//!   the same file on every turn is fine — file size is small relative
//!   to disk + the latest snapshot is always self-contained.
//! * `<cwd-key>` is the same shared helper used by `todos.rs`, so a
//!   user who runs from `/foo/bar` and from `/foo/bar/baz` gets two
//!   separate session histories the same way they get two todo lists.
//! * `id` is a sortable `YYYYMMDD-HHMMSS-<4hex>` so `ls -la` is in
//!   chronological order and collisions within the same second are
//!   broken by a small random suffix.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::ollama::Message;
use crate::todos::cwd_key;

/// Persisted shape of a single session checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionFile {
    /// Sortable identifier, also used as the on-disk filename stem.
    pub id: String,
    /// Unix seconds at session start. Kept separately from `id` so the
    /// UI doesn't have to parse the formatted id.
    pub started_at: u64,
    /// Model in use at the time the snapshot was written.
    pub model: String,
    /// Full conversation history.
    pub history: Vec<Message>,
}

/// Lightweight listing entry for `/sessions`. Skips loading the full
/// history so the command stays cheap for long-lived projects.
#[derive(Debug, Clone)]
pub struct SessionMeta {
    pub id: String,
    pub started_at: u64,
    pub model: String,
    pub message_count: usize,
    /// Path on disk. Kept for future `/sessions delete <id>` and
    /// debugging output; `#[allow(dead_code)]` while no command uses it.
    #[allow(dead_code)]
    pub path: PathBuf,
    /// First user message, truncated to ~80 chars for the listing.
    pub preview: String,
}

/// Per-cwd session store. Cheap to clone — only carries the directory
/// path, not any cached state.
#[derive(Debug, Clone)]
pub struct SessionStore {
    dir: PathBuf,
}

impl SessionStore {
    /// Returns a store rooted at `~/.ai-chat-cli/sessions/<cwd-key>/`.
    /// `None` when neither the home directory nor the cwd can be read,
    /// in which case sessions silently degrade to "not persisted" —
    /// preferable to crashing on startup.
    pub fn for_current_dir() -> Option<Self> {
        let cwd = std::env::current_dir().ok()?;
        Self::for_cwd(&cwd)
    }

    pub fn for_cwd(cwd: &Path) -> Option<Self> {
        let home = dirs::home_dir()?;
        Some(Self {
            dir: home
                .join(".ai-chat-cli")
                .join("sessions")
                .join(cwd_key(cwd)),
        })
    }

    /// Allocates a new session checkpoint with a fresh id and empty
    /// history. Does not write to disk; the first `save` call creates
    /// the file.
    pub fn new_session(&self, model: String) -> SessionFile {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let secs = now.as_secs();
        // Sortable timestamp prefix + 4 hex from the nanos for collision
        // avoidance within the same second. No need for a real UUID
        // here: the directory is already scoped to one cwd and one user.
        let suffix = (now.subsec_nanos() & 0xffff) as u16;
        let id = format!("{}-{:04x}", format_timestamp(secs), suffix);
        SessionFile {
            id,
            started_at: secs,
            model,
            history: Vec::new(),
        }
    }

    /// Writes (or overwrites) the checkpoint file. Creates the
    /// per-cwd directory on first use.
    pub fn save(&self, session: &SessionFile) -> Result<()> {
        fs::create_dir_all(&self.dir)
            .with_context(|| format!("Failed to create {}", self.dir.display()))?;
        let path = self.dir.join(format!("{}.json", session.id));
        let json = serde_json::to_string_pretty(session)?;
        fs::write(&path, json).with_context(|| format!("Failed to write {}", path.display()))?;
        Ok(())
    }

    /// Loads a session by id (filename stem). Returns `None` if no file
    /// matches; surfaces parse errors so the user can see corrupted
    /// session files instead of silently losing them.
    pub fn load(&self, id: &str) -> Result<Option<SessionFile>> {
        let path = self.dir.join(format!("{}.json", id));
        if !path.exists() {
            return Ok(None);
        }
        let raw = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read {}", path.display()))?;
        let parsed = serde_json::from_str(&raw)
            .with_context(|| format!("Failed to parse {}", path.display()))?;
        Ok(Some(parsed))
    }

    /// Lists sessions newest-first. Unreadable individual files are
    /// skipped rather than failing the whole listing.
    pub fn list(&self) -> Result<Vec<SessionMeta>> {
        if !self.dir.is_dir() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for entry in fs::read_dir(&self.dir)
            .with_context(|| format!("Failed to read {}", self.dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let Ok(raw) = fs::read_to_string(&path) else {
                continue;
            };
            let Ok(file) = serde_json::from_str::<SessionFile>(&raw) else {
                continue;
            };
            let preview = file
                .history
                .iter()
                .find(|m| m.role == "user")
                .map(|m| truncate(&m.content, 80))
                .unwrap_or_default();
            out.push(SessionMeta {
                id: file.id,
                started_at: file.started_at,
                model: file.model,
                message_count: file.history.len(),
                path,
                preview,
            });
        }
        // Newest first.
        out.sort_by_key(|m| std::cmp::Reverse(m.started_at));
        Ok(out)
    }

    /// Convenience: returns the most recent session, or `None` if there
    /// are no checkpoints yet.
    pub fn latest(&self) -> Result<Option<SessionFile>> {
        let metas = self.list()?;
        let Some(latest) = metas.into_iter().next() else {
            return Ok(None);
        };
        self.load(&latest.id)
    }
}

fn truncate(s: &str, max_chars: usize) -> String {
    let one_line = s.replace('\n', " ");
    if one_line.chars().count() <= max_chars {
        one_line
    } else {
        let mut t: String = one_line.chars().take(max_chars).collect();
        t.push('…');
        t
    }
}

/// Formats a unix timestamp as `YYYYMMDD-HHMMSS` (UTC) for use as the
/// chronological prefix of a session id. Pure date math — pulling in a
/// chrono dependency just for this would be overkill.
fn format_timestamp(secs: u64) -> String {
    // Days / time of day decomposition.
    let days = (secs / 86400) as i64;
    let tod = secs % 86400;
    let hour = tod / 3600;
    let minute = (tod % 3600) / 60;
    let second = tod % 60;

    // Civil-from-days, courtesy of Howard Hinnant's date algorithm.
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{:04}{:02}{:02}-{:02}{:02}{:02}",
        y, m, d, hour, minute, second
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_store(label: &str) -> SessionStore {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ai-chat-cli-sess-{label}-{nanos}"));
        SessionStore { dir }
    }

    fn user(text: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: text.to_string(),
        }
    }

    fn assistant(text: &str) -> Message {
        Message {
            role: "assistant".to_string(),
            content: text.to_string(),
        }
    }

    #[test]
    fn save_then_load_roundtrip() {
        let store = tmp_store("roundtrip");
        let mut session = store.new_session("llama3.2:1b".into());
        session.history.push(user("hello"));
        session.history.push(assistant("hi there"));
        store.save(&session).unwrap();

        let reloaded = store.load(&session.id).unwrap().expect("loads");
        assert_eq!(reloaded.history.len(), 2);
        assert_eq!(reloaded.model, "llama3.2:1b");

        fs::remove_dir_all(&store.dir).ok();
    }

    #[test]
    fn list_orders_newest_first() {
        let store = tmp_store("ordering");
        let mut a = store.new_session("m".into());
        a.id = "20240101-000000-0001".to_string();
        a.started_at = 1_704_067_200;
        a.history.push(user("first"));
        store.save(&a).unwrap();

        let mut b = store.new_session("m".into());
        b.id = "20250101-000000-0001".to_string();
        b.started_at = 1_735_689_600;
        b.history.push(user("second"));
        store.save(&b).unwrap();

        let list = store.list().unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].id, "20250101-000000-0001");
        assert_eq!(list[1].id, "20240101-000000-0001");
        assert_eq!(list[0].preview, "second");
        fs::remove_dir_all(&store.dir).ok();
    }

    #[test]
    fn load_missing_returns_none() {
        let store = tmp_store("missing");
        assert!(store.load("nope").unwrap().is_none());
        fs::remove_dir_all(&store.dir).ok();
    }

    #[test]
    fn latest_returns_most_recent() {
        let store = tmp_store("latest");
        let mut a = store.new_session("m".into());
        a.id = "20240101-000000-0001".into();
        a.started_at = 1;
        store.save(&a).unwrap();

        let mut b = store.new_session("m".into());
        b.id = "20240101-000001-0001".into();
        b.started_at = 2;
        b.history.push(user("hello"));
        store.save(&b).unwrap();

        let latest = store.latest().unwrap().expect("some");
        assert_eq!(latest.id, b.id);
        fs::remove_dir_all(&store.dir).ok();
    }

    #[test]
    fn format_timestamp_known_epoch() {
        // 2024-01-15 12:34:56 UTC = 1_705_322_096.
        assert_eq!(format_timestamp(1_705_322_096), "20240115-123456");
        // Unix epoch.
        assert_eq!(format_timestamp(0), "19700101-000000");
    }

    #[test]
    fn truncate_respects_max_chars() {
        assert_eq!(truncate("short", 80), "short");
        let long = "x".repeat(200);
        let t = truncate(&long, 80);
        assert!(t.ends_with('…'));
        assert_eq!(t.chars().count(), 81);
    }
}
