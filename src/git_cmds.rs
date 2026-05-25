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

// ----- worktree -----

/// Lists worktrees in porcelain form.
pub fn worktree_list() -> Result<GitOutput> {
    run_git(&["--no-pager", "worktree", "list", "--porcelain"])
}

/// Creates a new worktree at `path`, optionally checking out `branch`.
pub fn worktree_add(path: &str, branch: Option<&str>) -> Result<GitOutput> {
    let mut args = vec!["worktree", "add", path];
    if let Some(b) = branch {
        args.push(b);
    }
    run_git(&args)
}

/// Removes the worktree at `path`.
pub fn worktree_remove(path: &str) -> Result<GitOutput> {
    run_git(&["worktree", "remove", path])
}

/// Parses `/worktree` args into a [`WorktreeAction`]. Accepts:
/// `list`, `add <path> [branch]`, `remove <path>`. Returns `None` on
/// any parse error so the caller can print usage.
#[derive(Debug, PartialEq, Eq)]
pub enum WorktreeAction<'a> {
    List,
    Add {
        path: &'a str,
        branch: Option<&'a str>,
    },
    Remove {
        path: &'a str,
    },
}

pub fn parse_worktree_args(rest: &str) -> Option<WorktreeAction<'_>> {
    let trimmed = rest.trim();
    if trimmed.is_empty() {
        return Some(WorktreeAction::List);
    }
    let mut parts = trimmed.split_whitespace();
    let action = parts.next()?;
    match action {
        "list" | "ls" => {
            if parts.next().is_some() {
                return None;
            }
            Some(WorktreeAction::List)
        }
        "add" => {
            let path = parts.next()?;
            let branch = parts.next();
            if parts.next().is_some() {
                return None;
            }
            Some(WorktreeAction::Add { path, branch })
        }
        "remove" | "rm" => {
            let path = parts.next()?;
            if parts.next().is_some() {
                return None;
            }
            Some(WorktreeAction::Remove { path })
        }
        _ => None,
    }
}

// ----- branch -----

/// Lists local branches.
pub fn branch_list() -> Result<GitOutput> {
    run_git(&["--no-pager", "branch", "--list"])
}

/// Creates a new branch with the given name (does not check it out).
pub fn branch_create(name: &str) -> Result<GitOutput> {
    run_git(&["branch", name])
}

/// Switches to an existing branch.
pub fn branch_switch(name: &str) -> Result<GitOutput> {
    run_git(&["switch", name])
}

/// `/branch` argument shapes.
#[derive(Debug, PartialEq, Eq)]
pub enum BranchAction<'a> {
    List,
    Create { name: &'a str },
    Switch { name: &'a str },
}

pub fn parse_branch_args(rest: &str) -> Option<BranchAction<'_>> {
    let trimmed = rest.trim();
    if trimmed.is_empty() {
        return Some(BranchAction::List);
    }
    let mut parts = trimmed.split_whitespace();
    let head = parts.next()?;
    match head {
        "list" | "ls" => {
            if parts.next().is_some() {
                return None;
            }
            Some(BranchAction::List)
        }
        "create" | "new" => {
            let name = parts.next()?;
            if parts.next().is_some() {
                return None;
            }
            Some(BranchAction::Create { name })
        }
        "switch" | "checkout" | "co" => {
            let name = parts.next()?;
            if parts.next().is_some() {
                return None;
            }
            Some(BranchAction::Switch { name })
        }
        _ => None,
    }
}

// ----- tag -----

/// Lists tags.
pub fn tag_list() -> Result<GitOutput> {
    run_git(&["--no-pager", "tag", "--list"])
}

/// Creates a tag. If `message` is `Some`, creates an annotated tag.
pub fn tag_create(name: &str, message: Option<&str>) -> Result<GitOutput> {
    match message {
        Some(msg) => run_git(&["tag", "-a", name, "-m", msg]),
        None => run_git(&["tag", name]),
    }
}

/// `/tag` argument shapes.
#[derive(Debug, PartialEq, Eq)]
pub enum TagAction<'a> {
    List,
    Create {
        name: &'a str,
        message: Option<&'a str>,
    },
}

pub fn parse_tag_args(rest: &str) -> Option<TagAction<'_>> {
    let trimmed = rest.trim();
    if trimmed.is_empty() {
        return Some(TagAction::List);
    }
    // Single token = treat as `create <name>` for ergonomics, e.g. `/tag v1.0.0`.
    let (head, after_head) = trimmed
        .split_once(char::is_whitespace)
        .map(|(h, t)| (h, t.trim()))
        .unwrap_or((trimmed, ""));
    match head {
        "list" | "ls" => {
            if !after_head.is_empty() {
                return None;
            }
            Some(TagAction::List)
        }
        "create" | "new" => {
            // Split: <name> [-m <message...>]
            let (name, rest2) = after_head
                .split_once(char::is_whitespace)
                .map(|(n, t)| (n, t.trim()))
                .unwrap_or((after_head, ""));
            if name.is_empty() {
                return None;
            }
            let message = if rest2.is_empty() {
                None
            } else if let Some(stripped) = rest2.strip_prefix("-m ") {
                let m = stripped.trim();
                if m.is_empty() {
                    return None;
                }
                Some(m)
            } else if rest2 == "-m" {
                return None;
            } else {
                // Unknown trailing args
                return None;
            };
            Some(TagAction::Create { name, message })
        }
        // Bare `/tag <name>` shortcut for lightweight tag creation.
        other if !other.starts_with('-') => {
            if !after_head.is_empty() {
                return None;
            }
            Some(TagAction::Create {
                name: other,
                message: None,
            })
        }
        _ => None,
    }
}

// ----- ls-files -----

/// Lists files tracked by git in the current repo.
pub fn ls_files() -> Result<GitOutput> {
    run_git(&["--no-pager", "ls-files"])
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

    #[test]
    fn parse_worktree_args_variants() {
        assert_eq!(parse_worktree_args(""), Some(WorktreeAction::List));
        assert_eq!(parse_worktree_args("list"), Some(WorktreeAction::List));
        assert_eq!(parse_worktree_args("ls"), Some(WorktreeAction::List));
        assert_eq!(
            parse_worktree_args("add ../wt"),
            Some(WorktreeAction::Add {
                path: "../wt",
                branch: None
            })
        );
        assert_eq!(
            parse_worktree_args("add ../wt feat"),
            Some(WorktreeAction::Add {
                path: "../wt",
                branch: Some("feat")
            })
        );
        assert_eq!(
            parse_worktree_args("remove ../wt"),
            Some(WorktreeAction::Remove { path: "../wt" })
        );
        assert_eq!(
            parse_worktree_args("rm ../wt"),
            Some(WorktreeAction::Remove { path: "../wt" })
        );
        // bad inputs
        assert!(parse_worktree_args("add").is_none());
        assert!(parse_worktree_args("remove").is_none());
        assert!(parse_worktree_args("list extra").is_none());
        assert!(parse_worktree_args("unknown ../wt").is_none());
        assert!(parse_worktree_args("add ../wt b extra").is_none());
    }

    #[test]
    fn parse_branch_args_variants() {
        assert_eq!(parse_branch_args(""), Some(BranchAction::List));
        assert_eq!(parse_branch_args("list"), Some(BranchAction::List));
        assert_eq!(
            parse_branch_args("create feat"),
            Some(BranchAction::Create { name: "feat" })
        );
        assert_eq!(
            parse_branch_args("new feat"),
            Some(BranchAction::Create { name: "feat" })
        );
        assert_eq!(
            parse_branch_args("switch main"),
            Some(BranchAction::Switch { name: "main" })
        );
        assert_eq!(
            parse_branch_args("co main"),
            Some(BranchAction::Switch { name: "main" })
        );
        assert!(parse_branch_args("create").is_none());
        assert!(parse_branch_args("switch").is_none());
        assert!(parse_branch_args("create a b").is_none());
        assert!(parse_branch_args("bogus x").is_none());
    }

    #[test]
    fn parse_tag_args_variants() {
        assert_eq!(parse_tag_args(""), Some(TagAction::List));
        assert_eq!(parse_tag_args("list"), Some(TagAction::List));
        assert_eq!(
            parse_tag_args("v1.0.0"),
            Some(TagAction::Create {
                name: "v1.0.0",
                message: None
            })
        );
        assert_eq!(
            parse_tag_args("create v1.0.0"),
            Some(TagAction::Create {
                name: "v1.0.0",
                message: None
            })
        );
        assert_eq!(
            parse_tag_args("create v1.0.0 -m hello world"),
            Some(TagAction::Create {
                name: "v1.0.0",
                message: Some("hello world")
            })
        );
        assert!(parse_tag_args("create").is_none());
        assert!(parse_tag_args("create v1 -m").is_none());
        assert!(parse_tag_args("create v1 garbage").is_none());
        assert!(parse_tag_args("list extra").is_none());
    }
}
