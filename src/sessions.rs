//! Auto-saved chat sessions + `/sessions` + `/resume`.
//!
//! Until this module landed, the only persistence for a conversation was
//! `/save <file>` — and you only ever ran it after losing one. This
//! module turns persistence on by default: every successful assistant
//! turn appends the entire `history` to a per-project JSON checkpoint at
//! `~/.cubi/sessions/<cwd-key>/<id>.json`.
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
    /// Working directory where the session was created. Older checkpoints may
    /// not have this field; callers fall back to the on-disk bucket.
    #[serde(default)]
    pub cwd: String,
    /// Model in use at the time the snapshot was written.
    pub model: String,
    /// Full conversation history.
    pub history: Vec<Message>,
}

/// Lightweight listing entry for `/sessions`. Carries just enough to
/// render the list (preview, message count, model, timestamp) without
/// surfacing the full message history.
///
/// Note: today's on-disk format is one self-contained JSON document per
/// session — there's no sidecar index — so `SessionStore::list` still has
/// to read each file. To keep listing cheap on long-lived projects it
/// deserializes into [`SessionMetaFile`] (which only walks the `history`
/// array's length, not its message bodies) rather than into the full
/// [`SessionFile`].
#[derive(Debug, Clone)]
pub struct SessionMeta {
    pub id: String,
    pub model: String,
    pub message_count: usize,
    /// Unix seconds from the checkpoint file's modification time.
    pub modified_at: u64,
    /// Working directory where the session was created, or a best-effort
    /// bucket name for older checkpoints.
    pub cwd: String,
    /// Path on disk. Used for deletion and diagnostics.
    pub path: PathBuf,
    /// First user message, truncated to ~80 chars for the listing.
    pub preview: String,
}

/// On-disk shape used by `SessionStore::list` to extract metadata
/// cheaply: each cell in `history` is parsed as a minimal `PreviewMessage`
/// so per-message content is only retained until we've decided whether
/// it's the preview. Reusing the full `SessionFile` here would force
/// every listing to clone every assistant token + tool result into RAM.
#[derive(Debug, Deserialize)]
struct SessionMetaFile {
    id: String,
    started_at: u64,
    #[serde(default)]
    cwd: String,
    model: String,
    history: Vec<PreviewMessage>,
}

/// Stripped-down `Message` used during listing. Only carries `role` so
/// `message_count` is honest, plus `content` so we can pick the first
/// user message as a preview. Tool-call payloads and the like are
/// discarded by serde because they aren't in the struct.
#[derive(Debug, Deserialize)]
struct PreviewMessage {
    role: String,
    #[serde(default)]
    content: String,
}

/// Per-cwd session store. Cheap to clone — only carries the directory
/// path, not any cached state.
#[derive(Debug, Clone)]
pub struct SessionStore {
    dir: PathBuf,
    cwd: PathBuf,
}

impl SessionStore {
    /// Returns a store rooted at `~/.cubi/sessions/<cwd-key>/`.
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
            dir: home.join(".cubi").join("sessions").join(cwd_key(cwd)),
            cwd: cwd.to_path_buf(),
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
            cwd: self.cwd.display().to_string(),
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

    /// Returns `true` if a checkpoint file for this id is on disk.
    /// Cheap stat — used to confirm a session is actually resumable
    /// before pointing the user at it.
    pub fn exists(&self, id: &str) -> bool {
        self.dir.join(format!("{}.json", id)).exists()
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
            let Ok(file) = serde_json::from_str::<SessionMetaFile>(&raw) else {
                continue;
            };
            let modified_at = modified_secs(&path).unwrap_or(file.started_at);
            let cwd = if file.cwd.is_empty() {
                self.cwd.display().to_string()
            } else {
                file.cwd.clone()
            };
            let preview = file
                .history
                .iter()
                .find(|m| m.role == "user")
                .map(|m| truncate(&m.content, 80))
                .unwrap_or_default();
            out.push(SessionMeta {
                id: file.id,
                model: file.model,
                message_count: file.history.len(),
                modified_at,
                cwd,
                path,
                preview,
            });
        }
        // Newest first. Tie-break on id so that bursts of checkpoints
        // sharing the same mtime (coarse-resolution filesystems) still
        // produce a deterministic ordering.
        out.sort_by(|a, b| {
            b.modified_at
                .cmp(&a.modified_at)
                .then_with(|| b.id.cmp(&a.id))
        });
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

    /// Lists every checkpoint under `~/.cubi/sessions`, newest first.
    pub fn list_all() -> Result<Vec<SessionMeta>> {
        let Some(root) = sessions_root() else {
            return Ok(Vec::new());
        };
        if !root.is_dir() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for bucket in
            fs::read_dir(&root).with_context(|| format!("Failed to read {}", root.display()))?
        {
            let bucket = bucket?;
            let dir = bucket.path();
            if !dir.is_dir() {
                continue;
            }
            let bucket_name = dir
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            for entry in
                fs::read_dir(&dir).with_context(|| format!("Failed to read {}", dir.display()))?
            {
                let entry = entry?;
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) != Some("json") {
                    continue;
                }
                let Ok(raw) = fs::read_to_string(&path) else {
                    continue;
                };
                let Ok(file) = serde_json::from_str::<SessionMetaFile>(&raw) else {
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
                    model: file.model,
                    message_count: file.history.len(),
                    modified_at: modified_secs(&path).unwrap_or(file.started_at),
                    cwd: if file.cwd.is_empty() {
                        bucket_name.clone()
                    } else {
                        file.cwd
                    },
                    path,
                    preview,
                });
            }
        }
        out.sort_by(|a, b| {
            b.modified_at
                .cmp(&a.modified_at)
                .then_with(|| b.id.cmp(&a.id))
        });
        Ok(out)
    }

    /// Latest session in the current cwd, falling back to global newest.
    pub fn latest_for_current_dir_preferred(&self) -> Result<Option<SessionFile>> {
        // Fast path: the per-cwd bucket already lives on disk under
        // `self.dir`, so `self.latest()` only reads checkpoints for the
        // current project. Avoid scanning every bucket under
        // `~/.cubi/sessions` unless we genuinely have no local session.
        if let Some(session) = self.latest()? {
            return Ok(Some(session));
        }
        let Some(meta) = Self::list_all()?.into_iter().next() else {
            return Ok(None);
        };
        load_from_path(&meta.path).map(Some)
    }

    /// Resolves a session by full id or unique prefix without deleting it.
    pub fn find_by_prefix(prefix: &str) -> Result<FindSessionResult> {
        let matches: Vec<SessionMeta> = Self::list_all()?
            .into_iter()
            .filter(|m| m.id == prefix || m.id.starts_with(prefix))
            .collect();
        match matches.as_slice() {
            [] => Ok(FindSessionResult::NotFound),
            [meta] => Ok(FindSessionResult::Found(meta.clone())),
            _ => Ok(FindSessionResult::Ambiguous(matches)),
        }
    }

    /// Deletes a session by full id or unique prefix. Returns matching
    /// candidates when the prefix is missing or ambiguous.
    pub fn delete_by_prefix(prefix: &str) -> Result<DeleteSessionResult> {
        match Self::find_by_prefix(prefix)? {
            FindSessionResult::Found(meta) => {
                fs::remove_file(&meta.path)
                    .with_context(|| format!("Failed to delete {}", meta.path.display()))?;
                Ok(DeleteSessionResult::Deleted(meta))
            }
            FindSessionResult::NotFound => Ok(DeleteSessionResult::NotFound),
            FindSessionResult::Ambiguous(matches) => Ok(DeleteSessionResult::Ambiguous(matches)),
        }
    }
}

#[derive(Debug, Clone)]
pub enum FindSessionResult {
    Found(SessionMeta),
    NotFound,
    Ambiguous(Vec<SessionMeta>),
}

#[derive(Debug, Clone)]
pub enum DeleteSessionResult {
    Deleted(SessionMeta),
    NotFound,
    Ambiguous(Vec<SessionMeta>),
}

fn sessions_root() -> Option<PathBuf> {
    Some(dirs::home_dir()?.join(".cubi").join("sessions"))
}

fn load_from_path(path: &Path) -> Result<SessionFile> {
    let raw =
        fs::read_to_string(path).with_context(|| format!("Failed to read {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("Failed to parse {}", path.display()))
}

fn modified_secs(path: &Path) -> Option<u64> {
    path.metadata()
        .ok()?
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs())
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
    let (y, m, d, hour, minute, second) = civil_from_unix(secs);
    format!(
        "{:04}{:02}{:02}-{:02}{:02}{:02}",
        y, m, d, hour, minute, second
    )
}

/// UTC civil-time decomposition shared by session-id formatting and the
/// `/sessions` listing in `main`. Returns `(year, month, day, hour,
/// minute, second)`. Implements Howard Hinnant's civil-from-days
/// algorithm — keep a single copy here so any rounding/leap-year bug is
/// fixed once, not twice.
pub(crate) fn civil_from_unix(secs: u64) -> (i64, u64, u64, u64, u64, u64) {
    let days = (secs / 86_400) as i64;
    let tod = secs % 86_400;
    let hour = tod / 3_600;
    let minute = (tod % 3_600) / 60;
    let second = tod % 60;

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d, hour, minute, second)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_store(label: &str) -> SessionStore {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let dir = project_root
            .join("target")
            .join("test-sessions")
            .join(format!("cubi-sess-{label}-{nanos}"));
        SessionStore {
            dir,
            cwd: project_root,
        }
    }

    fn user(text: &str) -> Message {
        Message::text("user", text)
    }

    fn assistant(text: &str) -> Message {
        Message::text("assistant", text)
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
