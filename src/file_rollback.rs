//! File-mutation journal for `/rewind`.
//!
//! Roadmap item C#17: when the user rewinds the conversation we also
//! want to undo any file edits that happened during the rewound turns,
//! so a "let's try that again" doesn't leave half-applied edits on
//! disk. This module is the journal those rollbacks read from.
//!
//! Every successful `edit_file` / `write_file` registers one
//! [`FileSnapshot`] containing the pre-image. Snapshots are grouped
//! into turns; `/rewind n` pops the last `n` turn-buckets and restores
//! each file to whichever snapshot is oldest within the bucket (i.e.
//! its state at the start of the rewound block).
//!
//! Snapshots live in memory only — the journal is intentionally
//! ephemeral so that closing the CLI is a hard commit. Crash recovery
//! is out of scope; the user can still inspect the underlying files
//! manually.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// One captured pre-image. `None` means "the file did not exist before
/// this turn", so rolling it back means deleting the file rather than
/// restoring bytes.
#[derive(Debug, Clone)]
pub struct FileSnapshot {
    pub path: PathBuf,
    pub previous_contents: Option<Vec<u8>>,
}

/// Snapshots captured during one assistant turn. Stored in insertion
/// order so the first snapshot for each path is the one we restore on
/// rewind (later writes within the same turn must defer to the earliest
/// pre-image we saw).
#[derive(Debug, Default, Clone)]
struct TurnSnapshots {
    /// `path -> first snapshot seen in this turn`. We deliberately
    /// drop subsequent writes to the same file inside the same turn
    /// because restoring the most recent pre-image would leak the
    /// intermediate state.
    by_path: HashMap<PathBuf, FileSnapshot>,
    /// Insertion order, used for stable diagnostics in `/rewind`.
    order: Vec<PathBuf>,
}

/// Shared, thread-safe journal handed to the built-in tools registry.
/// Cheap to clone (it's just an `Arc`).
#[derive(Debug, Default, Clone)]
pub struct FileJournal {
    inner: Arc<Mutex<JournalInner>>,
}

#[derive(Debug, Default)]
struct JournalInner {
    /// One entry per assistant turn. The CLI calls
    /// [`FileJournal::start_turn`] before each turn so we always have a
    /// fresh bucket on top.
    turns: Vec<TurnSnapshots>,
}

impl FileJournal {
    /// Pushes a fresh bucket so the next `record` writes land in their
    /// own turn. Call at the start of every assistant turn.
    pub fn start_turn(&self) {
        let mut g = self.inner.lock().unwrap();
        g.turns.push(TurnSnapshots::default());
    }

    /// Records a pre-image for `path`. No-op if a snapshot for that
    /// path already exists in the current turn (see comment on
    /// `TurnSnapshots::by_path`). Returns silently if no turn is
    /// active so misuse from background tasks doesn't crash.
    pub fn record(&self, path: PathBuf, previous_contents: Option<Vec<u8>>) {
        let mut g = self.inner.lock().unwrap();
        let Some(turn) = g.turns.last_mut() else {
            return;
        };
        if turn.by_path.contains_key(&path) {
            return;
        }
        turn.order.push(path.clone());
        turn.by_path.insert(
            path.clone(),
            FileSnapshot {
                path,
                previous_contents,
            },
        );
    }

    /// Rolls back the last `n` turns, restoring file contents. Returns
    /// the list of paths that were touched (in restoration order) plus
    /// any per-path error messages — partial failures don't abort the
    /// whole rewind.
    pub fn rewind(&self, n: usize) -> RewindOutcome {
        let mut g = self.inner.lock().unwrap();
        let drop_count = n.min(g.turns.len());
        let split_at = g.turns.len() - drop_count;
        let buckets: Vec<TurnSnapshots> = g.turns.split_off(split_at);
        drop(g);

        let mut outcome = RewindOutcome::default();
        // Walk newest → oldest so the oldest pre-image wins when a
        // path was touched in multiple rewound turns.
        let mut earliest: HashMap<PathBuf, FileSnapshot> = HashMap::new();
        let mut order: Vec<PathBuf> = Vec::new();
        for bucket in buckets.into_iter().rev() {
            for path in bucket.order {
                if let Some(snap) = bucket.by_path.get(&path).cloned() {
                    if !earliest.contains_key(&path) {
                        order.push(path.clone());
                    }
                    earliest.insert(path, snap);
                }
            }
        }
        for path in order {
            let snap = earliest.remove(&path).unwrap();
            match restore_one(&snap) {
                Ok(()) => outcome.restored.push(snap.path.clone()),
                Err(e) => outcome.errors.push((snap.path.clone(), e.to_string())),
            }
        }
        outcome
    }

    /// Drops the journal completely. Called after `/clear`.
    pub fn reset(&self) {
        let mut g = self.inner.lock().unwrap();
        g.turns.clear();
    }
}

#[derive(Debug, Default)]
pub struct RewindOutcome {
    pub restored: Vec<PathBuf>,
    pub errors: Vec<(PathBuf, String)>,
}

fn restore_one(snap: &FileSnapshot) -> std::io::Result<()> {
    match &snap.previous_contents {
        Some(bytes) => {
            if let Some(parent) = snap.path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&snap.path, bytes)
        }
        None => {
            // File didn't exist before the turn — delete whatever
            // landed there. Missing file is fine.
            match std::fs::remove_file(&snap.path) {
                Ok(()) => Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(e) => Err(e),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn tmp_file(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("cubi-journal-{name}-{nanos}.txt"))
    }

    #[test]
    fn rewind_restores_previous_contents() {
        let path = tmp_file("restore");
        std::fs::write(&path, b"original").unwrap();
        let journal = FileJournal::default();
        journal.start_turn();
        journal.record(path.clone(), Some(b"original".to_vec()));
        std::fs::write(&path, b"new").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
        let out = journal.rewind(1);
        assert!(out.errors.is_empty());
        assert_eq!(std::fs::read(&path).unwrap(), b"original");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rewind_deletes_files_created_in_rewound_turn() {
        let path = tmp_file("created");
        let _ = std::fs::remove_file(&path);
        let journal = FileJournal::default();
        journal.start_turn();
        journal.record(path.clone(), None);
        std::fs::write(&path, b"hello").unwrap();
        let out = journal.rewind(1);
        assert!(out.errors.is_empty());
        assert!(!path.exists());
    }

    #[test]
    fn multiple_writes_in_one_turn_collapse_to_first_pre_image() {
        let path = tmp_file("collapse");
        std::fs::write(&path, b"v0").unwrap();
        let journal = FileJournal::default();
        journal.start_turn();
        journal.record(path.clone(), Some(b"v0".to_vec()));
        std::fs::write(&path, b"v1").unwrap();
        // A second tool call inside the same turn records v1 — must be
        // dropped so the rollback restores v0, not v1.
        journal.record(path.clone(), Some(b"v1".to_vec()));
        std::fs::write(&path, b"v2").unwrap();
        let out = journal.rewind(1);
        assert!(out.errors.is_empty());
        assert_eq!(std::fs::read(&path).unwrap(), b"v0");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rewind_without_active_turn_is_noop() {
        let journal = FileJournal::default();
        let out = journal.rewind(5);
        assert!(out.restored.is_empty());
        assert!(out.errors.is_empty());
    }

    #[test]
    fn record_without_active_turn_is_silently_dropped() {
        let journal = FileJournal::default();
        // No panic, no state change.
        journal.record(PathBuf::from("/tmp/nope"), None);
        let out = journal.rewind(1);
        assert!(out.restored.is_empty());
    }
}
