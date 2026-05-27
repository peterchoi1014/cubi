//! `/edit` slash-command support: open the user's editor on a tempfile so
//! they can compose a longer prompt without fighting the readline buffer.
//!
//! The dispatch path in [`crate::cli`] resolves the editor binary, seeds
//! the tempfile (either from `/edit <text>`, an explicit draft, or the
//! last assistant message), spawns the editor, and then submits the
//! trimmed contents as the next user turn.
//!
//! The IO-heavy bits are intentionally pulled out of the spawn helper so
//! [`run_editor_session`] can be unit-tested with a closure that
//! synthesises edits instead of actually `exec`-ing `vi`.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Result of a single `/edit` round-trip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EditOutcome {
    /// Editor exited cleanly and the file contains new, non-empty text.
    /// The returned `String` is already trimmed and is the next user turn.
    Submit(String),
    /// Editor exited but the file is empty (or whitespace only) — cancel.
    Empty,
    /// Editor exited but the contents match the seed — cancel.
    Unchanged,
}

/// Resolves which editor binary to invoke. Order:
/// 1. `$CUBI_EDITOR`
/// 2. `$VISUAL`
/// 3. `$EDITOR`
/// 4. Platform fallback: `vi` on unix, `notepad.exe` on windows.
pub(crate) fn resolve_editor() -> String {
    for key in ["CUBI_EDITOR", "VISUAL", "EDITOR"] {
        if let Ok(v) = std::env::var(key) {
            let t = v.trim();
            if !t.is_empty() {
                return t.to_string();
            }
        }
    }
    if cfg!(windows) {
        "notepad.exe".to_string()
    } else {
        "vi".to_string()
    }
}

/// RAII tempfile cleanup. Best-effort: if removal fails (e.g. the editor
/// already deleted it) we swallow the error.
struct TempFileGuard {
    path: PathBuf,
}

impl TempFileGuard {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Returns a unique tempfile path under `std::env::temp_dir()` with the
/// `cubi-edit-*.md` shape. The suffix mixes wall-clock nanos, the pid,
/// and a process-wide atomic counter so parallel REPLs (and parallel
/// tests) never collide.
pub(crate) fn tempfile_path() -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let name = format!("cubi-edit-{}-{}-{}.md", pid, nanos, seq);
    std::env::temp_dir().join(name)
}

/// Core helper: seed a tempfile, hand it to `invoke`, then classify the
/// result. The closure is responsible for actually spawning the editor;
/// tests can swap in a no-op closure that writes synthetic content.
///
/// The tempfile is always deleted, even on error or panic, via a
/// [`TempFileGuard`].
pub(crate) fn run_editor_session<F>(seed: &str, invoke: F) -> io::Result<EditOutcome>
where
    F: FnOnce(&Path) -> io::Result<()>,
{
    let path = tempfile_path();
    let _guard = TempFileGuard::new(path.clone());
    fs::write(&path, seed)?;
    invoke(&path)?;
    let raw = fs::read_to_string(&path)?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(EditOutcome::Empty);
    }
    if trimmed == seed.trim() {
        return Ok(EditOutcome::Unchanged);
    }
    Ok(EditOutcome::Submit(trimmed.to_string()))
}

/// Spawns the resolved editor on `path` and waits for it to exit.
///
/// The editor command is split on whitespace so users can pass flags
/// (`EDITOR="code --wait"` is a common pattern). The path is appended via
/// [`std::process::Command::arg`] so platform-specific quoting is handled
/// by the OS — never string-interpolated into a shell command line.
pub(crate) fn spawn_editor_blocking(editor: &str, path: &Path) -> io::Result<()> {
    let mut parts = editor.split_whitespace();
    let Some(bin) = parts.next() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "editor command is empty",
        ));
    };
    let mut cmd = std::process::Command::new(bin);
    for a in parts {
        cmd.arg(a);
    }
    cmd.arg(path);
    let status = cmd.status()?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "editor exited with status {}",
            status
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn submit_when_editor_writes_new_content() {
        let outcome = run_editor_session("seed text", |p| fs::write(p, "edited content here\n"))
            .expect("session ok");
        assert_eq!(outcome, EditOutcome::Submit("edited content here".into()));
    }

    #[test]
    fn empty_when_editor_clears_file() {
        let outcome =
            run_editor_session("seed text", |p| fs::write(p, "   \n\t\n")).expect("session ok");
        assert_eq!(outcome, EditOutcome::Empty);
    }

    #[test]
    fn unchanged_when_editor_keeps_seed() {
        // Editor closure leaves the seed in place (writes the same bytes).
        let outcome =
            run_editor_session("seed text", |p| fs::write(p, "seed text")).expect("session ok");
        assert_eq!(outcome, EditOutcome::Unchanged);
    }

    #[test]
    fn tempfile_is_removed_after_session() {
        let mut captured: Option<PathBuf> = None;
        let cap = &mut captured;
        let _ = run_editor_session("x", |p| {
            *cap = Some(p.to_path_buf());
            fs::write(p, "y")
        })
        .expect("session ok");
        let path = captured.expect("path captured");
        assert!(
            !path.exists(),
            "expected tempfile to be removed: {}",
            path.display()
        );
    }

    #[test]
    fn tempfile_path_has_expected_shape() {
        let p = tempfile_path();
        let name = p.file_name().and_then(|s| s.to_str()).unwrap();
        assert!(name.starts_with("cubi-edit-"));
        assert!(name.ends_with(".md"));
    }

    #[test]
    fn resolve_editor_falls_back_to_platform_default() {
        // We can't mutate process env in parallel tests safely, so just
        // assert the fallback constant matches the current platform.
        let editor = if std::env::var("CUBI_EDITOR")
            .or_else(|_| std::env::var("VISUAL"))
            .or_else(|_| std::env::var("EDITOR"))
            .is_ok()
        {
            // Some env is set in CI — skip the fallback assertion.
            return;
        } else {
            resolve_editor()
        };
        if cfg!(windows) {
            assert_eq!(editor, "notepad.exe");
        } else {
            assert_eq!(editor, "vi");
        }
    }
}
