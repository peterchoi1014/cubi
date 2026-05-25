//! Thin wrappers around `git` for the `/diff`, `/commit`, and `/review`
//! slash commands.
//!
//! These intentionally shell out to the user's installed `git` rather
//! than vendoring a libgit2 dependency: every developer machine that
//! could plausibly run this CLI already has a working `git`, and the
//! shell-out keeps the surface tiny.
//!
//! All commands inherit the CLI's cwd. They do not consult the
//! permissions sandbox (the user invoked the slash command explicitly,
//! so trust is implicit), but `/commit` is gated on plan mode at the
//! call site.

use anyhow::{Context, Result};
use std::process::Command;

/// Outcome of a `git` invocation. We keep both streams so the caller can
/// decide whether to show stderr context even on success (git often
/// prints useful informational messages on stderr).
#[derive(Debug)]
pub struct GitOutput {
    pub exit_ok: bool,
    pub stdout: String,
    pub stderr: String,
}

fn run_git(args: &[&str]) -> Result<GitOutput> {
    let output = Command::new("git")
        .args(args)
        .output()
        .context("Failed to execute `git`. Is git installed and on PATH?")?;
    Ok(GitOutput {
        exit_ok: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

/// Returns the combined diff for `git diff [path]`. When `path` is
/// empty, diffs the whole working tree; pass a path to scope it.
pub fn diff(path: &str) -> Result<GitOutput> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        run_git(&["--no-pager", "diff"])
    } else {
        run_git(&["--no-pager", "diff", "--", trimmed])
    }
}

/// Returns the combined (staged + unstaged) diff so `/review` sees
/// everything the user is about to commit, not just half of it.
pub fn diff_for_review() -> Result<GitOutput> {
    run_git(&["--no-pager", "diff", "HEAD"])
}

/// Parses the argument list of `/commit`. Accepts an optional leading
/// `-a` (or `--all`) flag to stage tracked files first, then takes the
/// remainder of the line as the message.
///
/// Returns `(stage_all, message)`, or `None` for usage errors
/// (including a stage-all flag with no following message).
pub fn parse_commit_args(rest: &str) -> Option<(bool, &str)> {
    let trimmed = rest.trim();
    if trimmed.is_empty() {
        return None;
    }

    let (stage_all, remainder) = match trimmed
        .split_once(char::is_whitespace)
        .map(|(head, tail)| (head, tail.trim()))
    {
        Some(("-a", rest)) | Some(("--all", rest)) => (true, rest),
        _ if trimmed == "-a" || trimmed == "--all" => return None,
        _ => (false, trimmed),
    };

    if remainder.is_empty() {
        return None;
    }
    Some((stage_all, remainder))
}

/// Runs `git commit -m <message>`, optionally with `-a` to stage
/// tracked files first.
pub fn commit(stage_all: bool, message: &str) -> Result<GitOutput> {
    let mut args = vec!["commit"];
    if stage_all {
        args.push("-a");
    }
    args.push("-m");
    args.push(message);
    run_git(&args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_commit_args_plain_message() {
        assert_eq!(
            parse_commit_args("fix: handle empty input"),
            Some((false, "fix: handle empty input"))
        );
    }

    #[test]
    fn parse_commit_args_dash_a() {
        assert_eq!(
            parse_commit_args("-a fix: stage everything"),
            Some((true, "fix: stage everything"))
        );
        assert_eq!(
            parse_commit_args("--all big refactor"),
            Some((true, "big refactor"))
        );
    }

    #[test]
    fn parse_commit_args_rejects_empty() {
        assert_eq!(parse_commit_args(""), None);
        assert_eq!(parse_commit_args("   "), None);
        assert_eq!(parse_commit_args("-a "), None);
        assert_eq!(parse_commit_args("--all"), None);
    }
}
