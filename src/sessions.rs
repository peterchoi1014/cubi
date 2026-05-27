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
use std::collections::BTreeMap;
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
    /// User-curated pinned items injected as persistent system messages
    /// that survive `/compact`. Persisted so `/resume` keeps them.
    #[serde(default)]
    pub pinned: Vec<String>,
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
#[derive(Debug, Clone, Serialize)]
pub struct SessionMeta {
    pub id: String,
    pub model: String,
    pub started_at: u64,
    pub message_count: usize,
    /// Unix seconds from the checkpoint file's modification time.
    pub modified_at: u64,
    /// Working directory where the session was created, or a best-effort
    /// bucket name for older checkpoints.
    pub cwd: String,
    /// Path on disk. Used for deletion and diagnostics.
    #[serde(skip)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionIndex {
    pub version: u32,
    pub sessions: Vec<IndexSession>,
    #[serde(default)]
    pub last_used_per_cwd: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexSession {
    pub id: String,
    pub path: PathBuf,
    pub cwd: String,
    pub model: String,
    pub started_at: u64,
    pub modified_at: u64,
    pub message_count: usize,
    pub preview: String,
}

impl From<SessionMeta> for IndexSession {
    fn from(meta: SessionMeta) -> Self {
        Self {
            id: meta.id,
            path: meta.path,
            cwd: meta.cwd,
            model: meta.model,
            started_at: meta.started_at,
            modified_at: meta.modified_at,
            message_count: meta.message_count,
            preview: meta.preview,
        }
    }
}

impl From<IndexSession> for SessionMeta {
    fn from(entry: IndexSession) -> Self {
        Self {
            id: entry.id,
            model: entry.model,
            started_at: entry.started_at,
            message_count: entry.message_count,
            modified_at: entry.modified_at,
            cwd: entry.cwd,
            path: entry.path,
            preview: entry.preview,
        }
    }
}

/// Per-cwd session store. Cheap to clone — only carries the directory
/// path, not any cached state.
#[derive(Debug, Clone)]
pub struct SessionStore {
    dir: PathBuf,
    cwd: PathBuf,
}

#[derive(Debug, Clone)]
pub struct PruneItem {
    pub id: String,
    pub path: PathBuf,
    pub bytes: u64,
}

#[derive(Debug, Clone, Default)]
pub struct PruneReport {
    pub items: Vec<PruneItem>,
    pub bytes: u64,
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
        let home = home_dir()?;
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
            pinned: Vec::new(),
        }
    }

    /// Writes (or overwrites) the checkpoint file. Creates the
    /// per-cwd directory on first use.
    pub fn save(&self, session: &SessionFile) -> Result<()> {
        fs::create_dir_all(&self.dir)
            .with_context(|| format!("Failed to create {}", self.dir.display()))?;
        let path = self.dir.join(format!("{}.json", session.id));
        let json = serde_json::to_string_pretty(session)?;
        atomic_write(&path, json.as_bytes())?;
        tracing::debug!(target: "cubi::sessions", id = %session.id, path = %path.display(), "session saved");
        if is_managed_session_dir(&self.dir) {
            upsert_index_entry(meta_from_session_file(session, path)?)?;
        }
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
        if !is_managed_session_dir(&self.dir) {
            return scan_session_dir(&self.dir, &self.cwd.display().to_string());
        }
        let index = load_or_rebuild_index()?;
        let mut out: Vec<SessionMeta> = index
            .sessions
            .into_iter()
            .filter(|entry| entry.path.parent() == Some(self.dir.as_path()))
            .filter(|entry| entry.path.exists())
            .map(Into::into)
            .collect();
        sort_metas(&mut out);
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
        let index = load_or_rebuild_index()?;
        let mut out: Vec<SessionMeta> = index
            .sessions
            .into_iter()
            .filter(|entry| entry.path.exists())
            .map(Into::into)
            .collect();
        sort_metas(&mut out);
        Ok(out)
    }
}

fn scan_session_dir(dir: &Path, fallback_cwd: &str) -> Result<Vec<SessionMeta>> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("Failed to read {}", dir.display()))? {
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
            started_at: file.started_at,
            message_count: file.history.len(),
            modified_at: modified_secs(&path).unwrap_or(file.started_at),
            cwd: if file.cwd.is_empty() {
                fallback_cwd.to_string()
            } else {
                file.cwd
            },
            path,
            preview,
        });
    }
    sort_metas(&mut out);
    Ok(out)
}

fn scan_all_sessions() -> Result<Vec<SessionMeta>> {
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
                started_at: file.started_at,
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

impl SessionStore {
    /// Latest session in the current cwd, falling back to global newest.
    pub fn latest_for_current_dir_preferred(&self) -> Result<Option<SessionFile>> {
        let cwd = self.cwd.display().to_string();
        if let Ok(index) = load_or_rebuild_index() {
            if let Some(id) = index.last_used_per_cwd.get(&cwd) {
                if let Some(entry) = index
                    .sessions
                    .iter()
                    .find(|entry| &entry.id == id && entry.path.exists())
                {
                    return load_from_path(&entry.path).map(Some);
                }
            }
        }

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
                remove_index_entry(&meta.id)?;
                Ok(DeleteSessionResult::Deleted(meta))
            }
            FindSessionResult::NotFound => Ok(DeleteSessionResult::NotFound),
            FindSessionResult::Ambiguous(matches) => Ok(DeleteSessionResult::Ambiguous(matches)),
        }
    }

    pub fn prune_older_than(cutoff_secs: u64, dry_run: bool) -> Result<PruneReport> {
        let candidates: Vec<_> = Self::list_all()?
            .into_iter()
            .filter(|meta| meta.modified_at < cutoff_secs)
            .collect();
        let mut report = PruneReport::default();
        for meta in candidates {
            let bytes = meta.path.metadata().map(|m| m.len()).unwrap_or(0);
            report.bytes = report.bytes.saturating_add(bytes);
            report.items.push(PruneItem {
                id: meta.id.clone(),
                path: meta.path.clone(),
                bytes,
            });
            if !dry_run {
                fs::remove_file(&meta.path)
                    .with_context(|| format!("Failed to delete {}", meta.path.display()))?;
            }
        }
        if !dry_run && !report.items.is_empty() {
            rebuild_index()?;
        }
        Ok(report)
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

/// Returns the home directory, preferring the `HOME` environment variable
/// (and `USERPROFILE` on Windows) over the platform lookup.
///
/// On Windows, `dirs::home_dir()` uses `SHGetKnownFolderPath`, which ignores
/// the process environment. Checking the env vars first lets integration tests
/// redirect session storage to a temporary directory by setting `HOME` /
/// `USERPROFILE` on the child process.
pub(crate) fn home_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("HOME") {
        let path = PathBuf::from(p);
        if path.is_absolute() {
            return Some(path);
        }
    }
    #[cfg(windows)]
    if let Ok(p) = std::env::var("USERPROFILE") {
        let path = PathBuf::from(p);
        if path.is_absolute() {
            return Some(path);
        }
    }
    dirs::home_dir()
}

pub(crate) fn sessions_root() -> Option<PathBuf> {
    Some(home_dir()?.join(".cubi").join("sessions"))
}

fn is_managed_session_dir(dir: &Path) -> bool {
    sessions_root()
        .map(|root| dir.starts_with(root))
        .unwrap_or(false)
}

fn index_path() -> Option<PathBuf> {
    Some(sessions_root()?.join("index.json"))
}

fn load_or_rebuild_index() -> Result<SessionIndex> {
    let Some(path) = index_path() else {
        return Ok(empty_index());
    };
    if let Ok(raw) = fs::read_to_string(&path) {
        if let Ok(index) = serde_json::from_str::<SessionIndex>(&raw) {
            if index.version == 1 && !index_is_stale(&index)? {
                return Ok(index);
            }
        }
    }
    rebuild_index()
}

fn rebuild_index() -> Result<SessionIndex> {
    let mut sessions: Vec<IndexSession> =
        scan_all_sessions()?.into_iter().map(Into::into).collect();
    sort_index_sessions(&mut sessions);
    let index = SessionIndex {
        version: 1,
        sessions,
        last_used_per_cwd: BTreeMap::new(),
    };
    write_index(&index)?;
    Ok(index)
}

fn upsert_index_entry(meta: SessionMeta) -> Result<()> {
    let mut index = load_or_rebuild_index()?;
    index.sessions.retain(|entry| entry.id != meta.id);
    index
        .last_used_per_cwd
        .insert(meta.cwd.clone(), meta.id.clone());
    index.sessions.push(meta.into());
    sort_index_sessions(&mut index.sessions);
    write_index(&index)
}

fn remove_index_entry(id: &str) -> Result<()> {
    let mut index = load_or_rebuild_index()?;
    index.sessions.retain(|entry| entry.id != id);
    index
        .last_used_per_cwd
        .retain(|_, session_id| session_id != id);
    write_index(&index)
}

fn write_index(index: &SessionIndex) -> Result<()> {
    let Some(path) = index_path() else {
        return Ok(());
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create {}", parent.display()))?;
    }
    let bytes = serde_json::to_string_pretty(index)?;
    atomic_write(&path, bytes.as_bytes())
        .with_context(|| format!("Failed to atomically write {}", path.display()))?;
    Ok(())
}

/// Atomically writes `bytes` to `path` by writing into a unique temp
/// file in the same directory and renaming over the target. The unique
/// `.{filename}.tmp-{pid}-{nanos}` suffix avoids collisions with
/// concurrent writers (other cubi processes or other sessions sharing
/// the directory). On Windows, `rename` won't replace an existing file
/// so the destination is removed first.
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("file");
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let tmp_name = format!(".{}.tmp-{}-{}", filename, std::process::id(), nanos);
    let tmp = path.with_file_name(tmp_name);
    fs::write(&tmp, bytes).with_context(|| format!("Failed to write {}", tmp.display()))?;
    #[cfg(windows)]
    if path.exists() {
        fs::remove_file(path)
            .with_context(|| format!("Failed to remove stale {}", path.display()))?;
    }
    if let Err(err) = fs::rename(&tmp, path) {
        // Best-effort cleanup so a failed rename doesn't leak the tmp file.
        let _ = fs::remove_file(&tmp);
        return Err(err).with_context(|| {
            format!(
                "Failed to replace {} with {}",
                path.display(),
                tmp.display()
            )
        });
    }
    Ok(())
}

fn empty_index() -> SessionIndex {
    SessionIndex {
        version: 1,
        sessions: Vec::new(),
        last_used_per_cwd: BTreeMap::new(),
    }
}

fn index_is_stale(index: &SessionIndex) -> Result<bool> {
    let disk_count = count_session_files()?;
    let indexed_count = index
        .sessions
        .iter()
        .filter(|entry| entry.path.exists())
        .count();
    Ok(disk_count != indexed_count)
}

fn count_session_files() -> Result<usize> {
    let Some(root) = sessions_root() else {
        return Ok(0);
    };
    if !root.is_dir() {
        return Ok(0);
    }
    let mut count = 0;
    for bucket in
        fs::read_dir(&root).with_context(|| format!("Failed to read {}", root.display()))?
    {
        let bucket = bucket?;
        let dir = bucket.path();
        if !dir.is_dir() {
            continue;
        }
        for entry in
            fs::read_dir(&dir).with_context(|| format!("Failed to read {}", dir.display()))?
        {
            let entry = entry?;
            if entry.path().extension().and_then(|s| s.to_str()) == Some("json") {
                count += 1;
            }
        }
    }
    Ok(count)
}

fn sort_metas(metas: &mut [SessionMeta]) {
    metas.sort_by(|a, b| {
        b.modified_at
            .cmp(&a.modified_at)
            .then_with(|| b.id.cmp(&a.id))
    });
}

fn sort_index_sessions(sessions: &mut [IndexSession]) {
    sessions.sort_by(|a, b| {
        b.modified_at
            .cmp(&a.modified_at)
            .then_with(|| b.id.cmp(&a.id))
    });
}

fn meta_from_session_file(session: &SessionFile, path: PathBuf) -> Result<SessionMeta> {
    let modified_at = modified_secs(&path).unwrap_or(session.started_at);
    let preview = session
        .history
        .iter()
        .find(|m| m.role == "user")
        .map(|m| truncate(&m.content, 80))
        .unwrap_or_default();
    Ok(SessionMeta {
        id: session.id.clone(),
        model: session.model.clone(),
        started_at: session.started_at,
        message_count: session.history.len(),
        modified_at,
        cwd: session.cwd.clone(),
        path,
        preview,
    })
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
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

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

    fn restore_home(home: Option<std::ffi::OsString>, userprofile: Option<std::ffi::OsString>) {
        unsafe {
            if let Some(value) = home {
                std::env::set_var("HOME", value);
            } else {
                std::env::remove_var("HOME");
            }
            if let Some(value) = userprofile {
                std::env::set_var("USERPROFILE", value);
            } else {
                std::env::remove_var("USERPROFILE");
            }
        }
    }

    fn isolated_home(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let home = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test-home")
            .join(format!("cubi-home-{label}-{nanos}"));
        fs::create_dir_all(&home).unwrap();
        home
    }

    #[test]
    fn atomic_write_overwrites_existing_file() {
        let store = tmp_store("atomic-overwrite");
        fs::create_dir_all(&store.dir).unwrap();
        let path = store.dir.join("data.json");
        super::atomic_write(&path, b"v1").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"v1");
        super::atomic_write(&path, b"v2-longer").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"v2-longer");
        fs::remove_dir_all(&store.dir).ok();
    }

    #[test]
    fn save_does_not_leak_tmp_files() {
        let store = tmp_store("no-tmp-leak");
        let mut session = store.new_session("m".into());
        session.history.push(user("a"));
        store.save(&session).unwrap();
        session.history.push(assistant("b"));
        store.save(&session).unwrap();

        let leaked: Vec<_> = fs::read_dir(&store.dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.contains(".tmp-"))
            .collect();
        assert!(
            leaked.is_empty(),
            "expected no .tmp- files after atomic save, found: {leaked:?}"
        );
        fs::remove_dir_all(&store.dir).ok();
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
    fn rebuild_index_from_disk() {
        let _guard = ENV_LOCK.lock().unwrap();
        let sessions_root = sessions_root().expect("home directory should exist for tests");
        let bucket = sessions_root.join(format!(
            "bucket-rebuild-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let session_id = "20250101-000000-abcd";
        fs::create_dir_all(&bucket).unwrap();
        fs::write(
            bucket.join(format!("{session_id}.json")),
            r#"{
  "id": "20250101-000000-abcd",
  "started_at": 1735689600,
  "cwd": "/work/rebuild",
  "model": "m",
  "history": [{"role":"user","content":"from disk"}]
}"#,
        )
        .unwrap();

        let index = rebuild_index().unwrap();
        assert_eq!(index.version, 1);
        assert!(index.sessions.iter().any(|entry| entry.id == session_id));
        assert!(sessions_root.join("index.json").exists());
        fs::remove_dir_all(&bucket).ok();
    }

    #[test]
    fn save_then_list_uses_sidecar_index() {
        let _guard = ENV_LOCK.lock().unwrap();
        let old_home = std::env::var_os("HOME");
        let old_userprofile = std::env::var_os("USERPROFILE");
        let home = isolated_home("sidecar");
        unsafe {
            std::env::set_var("HOME", &home);
            std::env::set_var("USERPROFILE", &home);
        }
        let cwd = home.join("project");
        fs::create_dir_all(&cwd).unwrap();
        let store = SessionStore::for_cwd(&cwd).unwrap();
        let mut session = store.new_session("m".into());
        session.id = "20250101-000000-abcd".into();
        session.history.push(user("indexed preview"));
        store.save(&session).unwrap();
        fs::write(store.dir.join(format!("{}.json", session.id)), "not json").unwrap();

        let index = load_or_rebuild_index().unwrap();
        assert_eq!(
            index.last_used_per_cwd.get(&cwd.display().to_string()),
            Some(&session.id)
        );

        let list = store.list().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, session.id);
        assert_eq!(list[0].preview, "indexed preview");
        restore_home(old_home, old_userprofile);
        fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn resume_prefers_last_used_session_for_cwd() {
        let _guard = ENV_LOCK.lock().unwrap();
        let old_home = std::env::var_os("HOME");
        let old_userprofile = std::env::var_os("USERPROFILE");
        let home = isolated_home("last-used");
        unsafe {
            std::env::set_var("HOME", &home);
            std::env::set_var("USERPROFILE", &home);
        }
        let cwd = home.join("project");
        fs::create_dir_all(&cwd).unwrap();
        let store = SessionStore::for_cwd(&cwd).unwrap();

        let mut first = store.new_session("m".into());
        first.id = "20250101-000000-0001".into();
        first.history.push(user("first"));
        store.save(&first).unwrap();
        let mut second = store.new_session("m".into());
        second.id = "20250101-000001-0001".into();
        second.history.push(user("second"));
        store.save(&second).unwrap();

        let mut index = load_or_rebuild_index().unwrap();
        index
            .last_used_per_cwd
            .insert(cwd.display().to_string(), first.id.clone());
        write_index(&index).unwrap();

        let resumed = store.latest_for_current_dir_preferred().unwrap().unwrap();
        assert_eq!(resumed.id, first.id);
        restore_home(old_home, old_userprofile);
        fs::remove_dir_all(&home).ok();
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
