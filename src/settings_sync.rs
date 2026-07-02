//! Git-backed cross-machine settings sync.
//!
//! Roadmap item C#24: lets the user keep `~/.cubi/` in a private
//! git repo (typically a bare repo on GitHub Gist or a self-hosted
//! remote) so config, memdir, todos, and skills move with them between
//! machines. This is intentionally a thin wrapper around `git` rather
//! than a custom protocol — git already has the conflict-resolution,
//! auth, and history story we'd otherwise have to reinvent, and the
//! user can run any of these commands by hand if the wrapper breaks.
//!
//! The user-facing slash command (`/settings-sync`) calls into here for
//! the three meaningful verbs: `init <remote>`, `push`, and `pull`.

use anyhow::{Context, Result, bail};
use std::path::PathBuf;
use std::process::Command;

fn settings_root() -> Result<PathBuf> {
    crate::sessions::cubi_dir().context("Could not resolve home directory")
}

fn run_git(args: &[&str]) -> Result<String> {
    let root = settings_root()?;
    std::fs::create_dir_all(&root)?;
    let out = Command::new("git")
        .arg("-C")
        .arg(&root)
        .args(args)
        .output()
        .with_context(|| format!("Failed to spawn `git {}`", args.join(" ")))?;
    if !out.status.success() {
        bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Initialise `~/.cubi` as a git repository and wire up
/// `origin`. Idempotent: re-running on an already-initialised tree
/// just updates the remote.
pub fn init(remote: &str) -> Result<String> {
    let root = settings_root()?;
    std::fs::create_dir_all(&root)?;
    let dot_git = root.join(".git");
    if !dot_git.exists() {
        run_git(&["init", "-q", "-b", "main"])?;
    }
    // Always ignore the cache + log files that would create needless
    // churn on every push.
    let gitignore = root.join(".gitignore");
    if !gitignore.exists() {
        std::fs::write(
            &gitignore,
            "telemetry.log\nsessions/\ntriggers/\nmessages/\n",
        )?;
    }
    // `set-url` fails if there's no remote yet, so try `add` first and
    // fall back gracefully.
    if run_git(&["remote", "add", "origin", remote]).is_err() {
        run_git(&["remote", "set-url", "origin", remote])?;
    }
    Ok(format!("Initialised settings repo at {}", root.display()))
}

/// Commit any local changes and push to `origin/main`.
pub fn push(message: &str) -> Result<String> {
    let root = settings_root()?;
    if !root.join(".git").exists() {
        bail!("Settings sync is not initialised. Run `/settings-sync init <remote>` first.");
    }
    run_git(&["add", "-A"])?;
    // `git commit` exits non-zero with "nothing to commit" when there's
    // no diff. Treat that as success instead of erroring.
    if let Err(e) = run_git(&["commit", "-m", message]) {
        if !format!("{e}").contains("nothing to commit") {
            return Err(e);
        }
    }
    run_git(&["push", "-u", "origin", "main"])?;
    Ok("Pushed settings to origin/main".to_string())
}

/// Fetch + fast-forward `origin/main`. Aborts if a merge would be
/// non-trivial so users have to resolve by hand rather than silently
/// losing local edits.
pub fn pull() -> Result<String> {
    let root = settings_root()?;
    if !root.join(".git").exists() {
        bail!("Settings sync is not initialised. Run `/settings-sync init <remote>` first.");
    }
    run_git(&["fetch", "origin"])?;
    run_git(&["pull", "--ff-only", "origin", "main"])?;
    Ok("Pulled settings from origin/main".to_string())
}

/// Short human-readable status for `/settings-sync status`.
pub fn status() -> Result<String> {
    let root = settings_root()?;
    if !root.join(".git").exists() {
        return Ok(format!(
            "Settings sync is not initialised. Settings root: {}",
            root.display()
        ));
    }
    let remote = run_git(&["remote", "get-url", "origin"]).unwrap_or_else(|_| "(none)".to_string());
    let branch =
        run_git(&["rev-parse", "--abbrev-ref", "HEAD"]).unwrap_or_else(|_| "(none)".to_string());
    let dirty = run_git(&["status", "--porcelain"]).unwrap_or_default();
    let dirty_count = dirty.lines().filter(|l| !l.is_empty()).count();
    Ok(format!(
        "root: {}\nbranch: {}\nremote: {}\nuncommitted entries: {}",
        root.display(),
        branch,
        remote,
        dirty_count
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_root_uses_cubi_home() {
        crate::compat::test_env::with_cubi_home(|cubi_home, other_home| {
            let path = settings_root().expect("settings root");
            assert_eq!(path, cubi_home.join(".cubi"));
            assert!(!path.starts_with(other_home));
        });
    }
}
