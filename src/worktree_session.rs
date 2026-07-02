//! Ephemeral git worktree session guard.
//!
//! Provisions a temporary git worktree for an isolated checkout and tears it
//! down on drop.
#![allow(dead_code)]

use anyhow::{Context, Result, bail};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static COUNTER: AtomicU64 = AtomicU64::new(0);
const BRANCH_PREFIX: &str = "cubi-consensus";
const WORKTREE_PARENT_DIR: &str = "cubi-worktrees";
const WORKTREE_DIR_PREFIX: &str = "cubi-consensus-";
const MAX_NAME_COMPONENT_LEN: usize = 48;

#[derive(Debug)]
pub struct WorktreeSession {
    path: PathBuf,
    branch: String,
    repo_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoContext {
    pub top_level: PathBuf,
    pub relative_cwd: PathBuf,
}

pub fn create(base_ref: &str, label: &str) -> Result<WorktreeSession> {
    WorktreeSession::create(base_ref, label)
}

pub fn create_in(repo_dir: &Path, base_ref: &str, label: &str) -> Result<WorktreeSession> {
    WorktreeSession::create_in(repo_dir, base_ref, label)
}

pub fn resolve_repo_context(cwd: &Path) -> Result<RepoContext> {
    let cwd = cwd
        .canonicalize()
        .with_context(|| format!("Could not resolve cwd `{}`", cwd.display()))?;
    let top_level = resolve_git_toplevel(&cwd)?;
    let relative_cwd = cwd.strip_prefix(&top_level).with_context(|| {
        format!(
            "Resolved cwd `{}` is not under git top-level `{}`",
            cwd.display(),
            top_level.display()
        )
    })?;

    Ok(RepoContext {
        top_level,
        relative_cwd: relative_cwd.to_path_buf(),
    })
}

pub fn ensure_clean_worktree(repo_dir: &Path) -> Result<()> {
    let output = run_git(
        repo_dir,
        &[
            OsStr::new("status"),
            OsStr::new("--porcelain=v1"),
            OsStr::new("--untracked-files=normal"),
        ],
    )
    .with_context(|| {
        format!(
            "Failed to execute `git status --porcelain` from `{}`",
            repo_dir.display()
        )
    })?;

    if !output.status.success() {
        bail!(
            "git status --porcelain failed from `{}`: {}",
            repo_dir.display(),
            format_git_output(&output)
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let status = stdout.trim();
    if !status.is_empty() {
        bail!(
            "repository must be clean before isolated tool consensus can create worktrees; \
             commit, stash, or discard changes first. Dirty status:\n{}",
            truncate_for_error(status, 2000)
        );
    }

    Ok(())
}

impl WorktreeSession {
    pub fn create(base_ref: &str, label: &str) -> Result<Self> {
        let repo_dir = std::env::current_dir()
            .context("Could not read current directory for worktree creation")?;
        Self::create_in(&repo_dir, base_ref, label)
    }

    pub fn create_in(repo_dir: &Path, base_ref: &str, label: &str) -> Result<Self> {
        let base_ref = base_ref.trim();
        if base_ref.is_empty() {
            bail!("Cannot create worktree session: base_ref must not be empty");
        }

        create_in_checked(repo_dir, base_ref, label)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn branch(&self) -> &str {
        &self.branch
    }
}

fn create_in_checked(repo_dir: &Path, base_ref: &str, label: &str) -> Result<WorktreeSession> {
    let repo_dir = repo_dir.canonicalize().with_context(|| {
        format!(
            "Could not resolve repository directory `{}`",
            repo_dir.display()
        )
    })?;
    let names = session_names(label, &unique_suffix());
    let git_common_dir = resolve_git_common_dir(&repo_dir)?;
    let parent_dir = create_worktree_parent(&git_common_dir)?;
    let path = parent_dir.join(&names.worktree_dir);

    let output = run_git(
        &repo_dir,
        &[
            OsStr::new("worktree"),
            OsStr::new("add"),
            OsStr::new("--quiet"),
            OsStr::new("-b"),
            OsStr::new(names.branch.as_str()),
            path.as_os_str(),
            OsStr::new(base_ref),
        ],
    )
    .with_context(|| {
        format!(
            "Failed to execute `git worktree add` from `{}`",
            repo_dir.display()
        )
    })?;

    if !output.status.success() {
        cleanup_worktree(&repo_dir, &path, &names.branch);
        bail!(
            "git worktree add failed for branch `{}` at `{}` from `{}`: {}",
            names.branch,
            path.display(),
            base_ref,
            format_git_output(&output)
        );
    }

    Ok(WorktreeSession {
        path,
        branch: names.branch,
        repo_dir,
    })
}

impl Drop for WorktreeSession {
    fn drop(&mut self) {
        cleanup_worktree(&self.repo_dir, &self.path, &self.branch);
    }
}

struct SessionNames {
    branch: String,
    worktree_dir: String,
}

fn session_names(label: &str, suffix: &str) -> SessionNames {
    let label = sanitize_label(label);
    let stem = format!("{label}-{suffix}");

    SessionNames {
        branch: format!("{BRANCH_PREFIX}/{stem}"),
        worktree_dir: format!("{WORKTREE_DIR_PREFIX}{stem}"),
    }
}

fn cleanup_worktree(repo_dir: &Path, path: &Path, branch: &str) {
    let _ = run_git(
        repo_dir,
        &[
            OsStr::new("worktree"),
            OsStr::new("remove"),
            OsStr::new("--force"),
            path.as_os_str(),
        ],
    );
    let _ = run_git(repo_dir, &[OsStr::new("worktree"), OsStr::new("prune")]);
    let _ = run_git(
        repo_dir,
        &[OsStr::new("branch"), OsStr::new("-D"), OsStr::new(branch)],
    );
    remove_empty_worktree_dirs(path);
}

fn remove_empty_worktree_dirs(path: &Path) {
    let _ = fs::remove_dir(path);

    if let Some(parent) = path.parent() {
        if parent.file_name() == Some(OsStr::new(WORKTREE_PARENT_DIR)) {
            let _ = fs::remove_dir(parent);
        }
    }
}

fn resolve_git_toplevel(repo_dir: &Path) -> Result<PathBuf> {
    let output = run_git(
        repo_dir,
        &[OsStr::new("rev-parse"), OsStr::new("--show-toplevel")],
    )
    .with_context(|| {
        format!(
            "Failed to execute `git rev-parse --show-toplevel` from `{}`",
            repo_dir.display()
        )
    })?;

    if !output.status.success() {
        bail!(
            "git rev-parse --show-toplevel failed from `{}`: {}",
            repo_dir.display(),
            format_git_output(&output)
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let top_level = stdout.trim();
    if top_level.is_empty() {
        bail!(
            "git rev-parse --show-toplevel returned an empty path from `{}`",
            repo_dir.display()
        );
    }

    let top_level = PathBuf::from(top_level);
    let top_level = if top_level.is_absolute() {
        top_level
    } else {
        repo_dir.join(top_level)
    };

    top_level.canonicalize().with_context(|| {
        format!(
            "Could not resolve git top-level directory `{}`",
            top_level.display()
        )
    })
}

fn resolve_git_common_dir(repo_dir: &Path) -> Result<PathBuf> {
    let output = run_git(
        repo_dir,
        &[OsStr::new("rev-parse"), OsStr::new("--git-common-dir")],
    )
    .with_context(|| {
        format!(
            "Failed to execute `git rev-parse --git-common-dir` from `{}`",
            repo_dir.display()
        )
    })?;

    if !output.status.success() {
        bail!(
            "git rev-parse --git-common-dir failed from `{}`: {}",
            repo_dir.display(),
            format_git_output(&output)
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let common_dir = stdout.trim();
    if common_dir.is_empty() {
        bail!(
            "git rev-parse --git-common-dir returned an empty path from `{}`",
            repo_dir.display()
        );
    }

    let common_dir = PathBuf::from(common_dir);
    let common_dir = if common_dir.is_absolute() {
        common_dir
    } else {
        repo_dir.join(common_dir)
    };

    common_dir.canonicalize().with_context(|| {
        format!(
            "Could not resolve git common directory `{}`",
            common_dir.display()
        )
    })
}

fn create_worktree_parent(git_common_dir: &Path) -> Result<PathBuf> {
    let git_common_dir = git_common_dir.canonicalize().with_context(|| {
        format!(
            "Could not resolve git common directory `{}`",
            git_common_dir.display()
        )
    })?;
    let parent_dir = git_common_dir.join(WORKTREE_PARENT_DIR);
    fs::create_dir_all(&parent_dir).with_context(|| {
        format!(
            "Could not create worktree parent directory `{}` under git common directory `{}`",
            parent_dir.display(),
            git_common_dir.display(),
        )
    })?;

    ensure_worktree_parent_under_git_common_dir(&parent_dir, &git_common_dir)
}

fn ensure_worktree_parent_under_git_common_dir(
    parent_dir: &Path,
    git_common_dir: &Path,
) -> Result<PathBuf> {
    if parent_dir
        .components()
        .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        bail!(
            "Refusing to create worktree parent `{}` because it contains traversal components",
            parent_dir.display()
        );
    }

    let parent_dir = parent_dir.canonicalize().with_context(|| {
        format!(
            "Could not resolve worktree parent directory `{}`",
            parent_dir.display()
        )
    })?;
    let git_common_dir = git_common_dir.canonicalize().with_context(|| {
        format!(
            "Could not resolve git common directory `{}`",
            git_common_dir.display()
        )
    })?;

    let relative_parent = parent_dir.strip_prefix(&git_common_dir).with_context(|| {
        format!(
            "Refusing to create worktree parent `{}` outside git common directory `{}`",
            parent_dir.display(),
            git_common_dir.display()
        )
    })?;
    if relative_parent.as_os_str().is_empty() {
        bail!(
            "Refusing to use git common directory `{}` itself as worktree parent",
            git_common_dir.display()
        );
    }
    if !relative_parent
        .components()
        .all(|component| matches!(component, std::path::Component::Normal(_)))
    {
        bail!(
            "Refusing to create worktree parent `{}` because its relative path under git common directory `{}` contains traversal components",
            parent_dir.display(),
            git_common_dir.display()
        );
    }

    Ok(parent_dir)
}

fn run_git(repo_dir: &Path, args: &[&OsStr]) -> Result<Output> {
    Command::new("git")
        .current_dir(repo_dir)
        .args(args)
        .output()
        .with_context(|| format!("Failed to execute `git` from `{}`", repo_dir.display()))
}

fn unique_suffix() -> String {
    let nanos = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_nanos(),
        Err(_) => 0,
    };
    let pid = std::process::id();
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{nanos:x}-{pid:x}-{counter:x}")
}

fn sanitize_label(label: &str) -> String {
    sanitize_component(label, "session")
}

fn sanitize_component(input: &str, fallback: &str) -> String {
    let mut sanitized = String::new();

    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            sanitized.push(ch.to_ascii_lowercase());
        } else if !sanitized.is_empty() && !sanitized.ends_with('-') {
            sanitized.push('-');
        }

        if sanitized.len() >= MAX_NAME_COMPONENT_LEN {
            break;
        }
    }

    while sanitized.ends_with('-') {
        sanitized.pop();
    }

    if sanitized.is_empty() {
        fallback.to_string()
    } else {
        sanitized
    }
}

fn format_git_output(output: &Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = stderr.trim();
    let stdout = stdout.trim();

    format!(
        "status: {}; stdout: {}; stderr: {}",
        output.status,
        if stdout.is_empty() { "<empty>" } else { stdout },
        if stderr.is_empty() { "<empty>" } else { stderr }
    )
}

fn truncate_for_error(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max).collect();
        out.push('…');
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn create_in_provisions_and_drop_removes_worktree() {
        let repo = new_test_temp_dir();
        init_repo(repo.path());

        let session = create_in(repo.path(), "HEAD", "test").unwrap();
        let worktree_path = session.path().to_path_buf();
        let branch = session.branch().to_string();

        assert!(
            branch.starts_with("cubi-consensus/test-"),
            "unexpected temporary branch name: {branch}"
        );
        assert!(
            !branch.contains("head"),
            "base ref leaked into temporary branch name: {branch}"
        );
        assert!(
            worktree_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("cubi-consensus-test-"),
            "worktree path did not use sanitized session name: {}",
            worktree_path.display()
        );
        assert!(
            !worktree_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .contains("head"),
            "base ref leaked into temporary worktree path: {}",
            worktree_path.display()
        );
        assert!(worktree_path.is_dir());
        assert!(worktree_path.join("tracked.txt").is_file());
        assert!(branch_exists(repo.path(), &branch));
        let worktree_parent_path =
            assert_session_path_under_git_common_dir(repo.path(), &worktree_path);

        let listed = run_test_git(repo.path(), &["worktree", "list", "--porcelain"]);
        let listed = listed.replace('\\', "/");
        let expected_path = worktree_path.display().to_string().replace('\\', "/");
        assert!(
            listed.contains(&expected_path),
            "worktree list did not include {expected_path}:\n{listed}"
        );

        drop(session);
        assert!(!worktree_path.exists());
        assert!(!worktree_parent_path.exists());
        assert!(!branch_exists(repo.path(), &branch));

        let listed = run_test_git(repo.path(), &["worktree", "list", "--porcelain"]);
        let listed = listed.replace('\\', "/");
        assert!(
            !listed.contains(&expected_path),
            "worktree list still included {expected_path}:\n{listed}"
        );
        assert_no_consensus_artifacts(repo.path());
    }

    #[test]
    fn create_in_sanitizes_names_and_makes_unique_sessions() {
        let repo = new_test_temp_dir();
        init_repo(repo.path());
        run_test_git(repo.path(), &["branch", "origin/main"]);

        let first =
            WorktreeSession::create_in(repo.path(), "origin/main", " Review: model/a ").unwrap();
        let second =
            WorktreeSession::create_in(repo.path(), "origin/main", " Review: model/a ").unwrap();
        let first_path = first.path().to_path_buf();
        let second_path = second.path().to_path_buf();
        let first_branch = first.branch().to_string();
        let second_branch = second.branch().to_string();
        let first_parent = assert_session_path_under_git_common_dir(repo.path(), &first_path);
        let second_parent = assert_session_path_under_git_common_dir(repo.path(), &second_path);

        assert_ne!(first_branch, second_branch);
        assert_ne!(first_path, second_path);
        assert_eq!(first_parent, second_parent);
        assert_safe_session_name(&first_branch, "cubi-consensus/review-model-a-");
        assert_safe_session_name(&second_branch, "cubi-consensus/review-model-a-");
        assert_safe_session_name(
            &first_path.file_name().unwrap().to_string_lossy(),
            "cubi-consensus-review-model-a-",
        );
        assert_safe_session_name(
            &second_path.file_name().unwrap().to_string_lossy(),
            "cubi-consensus-review-model-a-",
        );
        assert!(
            !first_branch.contains("origin-main") && !second_branch.contains("origin-main"),
            "base ref leaked into temporary branch names: {first_branch}, {second_branch}"
        );
        assert!(
            !first_path.to_string_lossy().contains("origin-main")
                && !second_path.to_string_lossy().contains("origin-main"),
            "base ref leaked into temporary worktree paths: {}, {}",
            first_path.display(),
            second_path.display()
        );
        assert!(first_path.join("tracked.txt").is_file());
        assert!(second_path.join("tracked.txt").is_file());

        drop(first);
        assert!(!first_path.exists());
        assert!(second_path.exists());
        assert!(second_parent.exists());
        assert!(!branch_exists(repo.path(), &first_branch));

        drop(second);

        assert!(!second_path.exists());
        assert!(!first_parent.exists());
        assert!(!second_parent.exists());
        assert!(!branch_exists(repo.path(), &second_branch));
        assert_no_consensus_artifacts(repo.path());
    }

    #[test]
    fn create_in_reports_git_output_on_bad_base_ref_without_leaving_worktree() {
        let repo = new_test_temp_dir();
        init_repo(repo.path());
        let worktrees_before = worktree_list(repo.path());
        let refs_before = branch_refs(repo.path());

        let error = WorktreeSession::create_in(repo.path(), "missing-ref", "bad")
            .unwrap_err()
            .to_string();

        assert!(
            error.contains("git worktree add failed"),
            "unexpected error: {error}"
        );
        assert!(error.contains("stdout:"), "missing stdout: {error}");
        assert!(error.contains("stderr:"), "missing stderr: {error}");
        assert_eq!(worktree_list(repo.path()), worktrees_before);
        assert_eq!(branch_refs(repo.path()), refs_before);
        assert_no_consensus_artifacts(repo.path());
    }

    #[test]
    fn create_in_rejects_empty_base_ref_without_leaving_worktree() {
        let repo = new_test_temp_dir();
        init_repo(repo.path());
        let worktrees_before = worktree_list(repo.path());
        let refs_before = branch_refs(repo.path());

        let error = WorktreeSession::create_in(repo.path(), "   ", "empty")
            .unwrap_err()
            .to_string();

        assert!(
            error.contains("base_ref must not be empty"),
            "unexpected error: {error}"
        );
        assert_eq!(worktree_list(repo.path()), worktrees_before);
        assert_eq!(branch_refs(repo.path()), refs_before);
        assert_no_consensus_artifacts(repo.path());
    }

    #[test]
    fn resolve_git_toplevel_handles_nested_repo_path() {
        let repo = new_test_temp_dir();
        init_repo(repo.path());
        let nested = repo.path().join("nested").join("deeper");
        fs::create_dir_all(&nested).unwrap();

        let top_level = resolve_git_toplevel(&nested).unwrap();

        assert_eq!(top_level, repo.path().canonicalize().unwrap());
    }

    #[test]
    fn resolve_repo_context_preserves_relative_cwd() {
        let repo = new_test_temp_dir();
        init_repo(repo.path());
        let nested = repo.path().join("nested").join("deeper");
        fs::create_dir_all(&nested).unwrap();
        fs::write(nested.join("marker.txt"), "nested\n").unwrap();
        run_test_git(repo.path(), &["add", "nested/deeper/marker.txt"]);
        run_test_git(repo.path(), &["commit", "-m", "nested"]);

        let context = resolve_repo_context(&nested).unwrap();

        assert_eq!(context.top_level, repo.path().canonicalize().unwrap());
        assert_eq!(context.relative_cwd, PathBuf::from("nested/deeper"));
    }

    #[test]
    fn worktree_parent_guard_accepts_child_under_git_common_dir_from_nested_repo_path() {
        let repo = new_test_temp_dir();
        init_repo(repo.path());
        let nested = repo.path().join("nested").join("deeper");
        fs::create_dir_all(&nested).unwrap();
        let nested = nested.canonicalize().unwrap();
        let git_common_dir = resolve_git_common_dir(&nested).unwrap();
        let parent = git_common_dir.join(WORKTREE_PARENT_DIR);
        fs::create_dir_all(&parent).unwrap();

        let resolved =
            ensure_worktree_parent_under_git_common_dir(&parent, &git_common_dir).unwrap();

        assert_eq!(resolved, parent.canonicalize().unwrap());
    }

    #[test]
    fn worktree_parent_guard_rejects_git_common_dir_itself() {
        let repo = new_test_temp_dir();
        init_repo(repo.path());
        let git_common_dir = resolve_git_common_dir(repo.path()).unwrap();

        let error = ensure_worktree_parent_under_git_common_dir(&git_common_dir, &git_common_dir)
            .unwrap_err()
            .to_string();

        assert!(
            error.contains("itself as worktree parent"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn worktree_parent_guard_rejects_parent_outside_git_common_dir() {
        let repo = new_test_temp_dir();
        init_repo(repo.path());
        let git_common_dir = resolve_git_common_dir(repo.path()).unwrap();
        let forbidden_parent = repo.path().join("cubi-consensus-forbidden");
        fs::create_dir_all(&forbidden_parent).unwrap();

        let error = ensure_worktree_parent_under_git_common_dir(&forbidden_parent, &git_common_dir)
            .unwrap_err()
            .to_string();

        assert!(
            error.contains("outside git common directory"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn worktree_parent_guard_rejects_traversal_path() {
        let repo = new_test_temp_dir();
        init_repo(repo.path());
        let git_common_dir = resolve_git_common_dir(repo.path()).unwrap();
        let traversal_parent = git_common_dir.join("..").join("cubi-consensus-forbidden");

        let error = ensure_worktree_parent_under_git_common_dir(&traversal_parent, &git_common_dir)
            .unwrap_err()
            .to_string();

        assert!(
            error.contains("traversal components"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn ensure_clean_worktree_rejects_dirty_state_before_worktree_creation() {
        let repo = new_test_temp_dir();
        init_repo(repo.path());
        fs::write(repo.path().join("tracked.txt"), "dirty\n").unwrap();

        let error = ensure_clean_worktree(repo.path()).unwrap_err().to_string();

        assert!(
            error.contains("repository must be clean"),
            "unexpected error: {error}"
        );
        assert!(
            error.contains("tracked.txt"),
            "dirty status omitted file path: {error}"
        );
        assert_no_consensus_artifacts(repo.path());
    }

    #[test]
    fn sanitize_label_replaces_unsafe_chars_and_falls_back() {
        assert_eq!(sanitize_label(" Review: model/a "), "review-model-a");
        assert_eq!(sanitize_label("../@{lock}"), "lock");
        assert_eq!(sanitize_label("☃☃☃"), "session");
    }

    #[test]
    fn session_names_include_label_and_suffix_without_base_ref() {
        let names = session_names("Consensus ✓ Run", "abc123");

        assert_eq!(names.branch, "cubi-consensus/consensus-run-abc123");
        assert_eq!(names.worktree_dir, "cubi-consensus-consensus-run-abc123");
        assert!(
            !names.branch.contains("origin-main") && !names.worktree_dir.contains("origin-main"),
            "base refs must not be part of session names"
        );
    }

    #[test]
    fn format_git_output_reports_status_stdout_and_stderr() {
        let output = Command::new("git").arg("--version").output().unwrap();
        let formatted = format_git_output(&output);

        assert!(
            formatted.contains("status: "),
            "missing status in: {formatted}"
        );
        assert!(
            formatted.contains("stdout: git version"),
            "missing stdout in: {formatted}"
        );
        assert!(
            formatted.contains("stderr: <empty>"),
            "missing empty stderr marker in: {formatted}"
        );

        let non_repo = new_test_temp_dir();
        let output = Command::new("git")
            .current_dir(non_repo.path())
            .args(["rev-parse", "--git-common-dir"])
            .output()
            .unwrap();
        let formatted = format_git_output(&output);

        assert!(
            formatted.contains("stdout: <empty>"),
            "missing empty stdout marker in: {formatted}"
        );
        assert!(
            formatted.contains("stderr: ") && !formatted.contains("stderr: <empty>"),
            "missing non-empty stderr in: {formatted}"
        );
    }

    #[test]
    fn unique_suffix_changes_between_calls() {
        assert_ne!(unique_suffix(), unique_suffix());
    }

    fn new_test_temp_dir() -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix("cubi-worktree-session-test-")
            .tempdir()
            .unwrap()
    }

    fn init_repo(repo_dir: &Path) {
        run_test_git(repo_dir, &["init"]);
        run_test_git(repo_dir, &["config", "user.email", "test@example.com"]);
        run_test_git(repo_dir, &["config", "user.name", "Test User"]);
        run_test_git(repo_dir, &["config", "commit.gpgsign", "false"]);
        fs::write(repo_dir.join("tracked.txt"), "tracked\n").unwrap();
        run_test_git(repo_dir, &["add", "tracked.txt"]);
        run_test_git(repo_dir, &["commit", "-m", "initial"]);
    }

    fn worktree_list(repo_dir: &Path) -> String {
        run_test_git(repo_dir, &["worktree", "list", "--porcelain"])
    }

    fn branch_refs(repo_dir: &Path) -> String {
        run_test_git(
            repo_dir,
            &["for-each-ref", "--format=%(refname)", "refs/heads"],
        )
    }

    fn branch_exists(repo_dir: &Path, branch: &str) -> bool {
        Command::new("git")
            .current_dir(repo_dir)
            .args(["show-ref", "--verify", "--quiet"])
            .arg(format!("refs/heads/{branch}"))
            .status()
            .unwrap()
            .success()
    }

    fn assert_session_path_under_git_common_dir(repo_dir: &Path, worktree_path: &Path) -> PathBuf {
        let repo_dir = repo_dir.canonicalize().unwrap();
        let common_dir = resolve_git_common_dir(&repo_dir).unwrap();
        let expected_parent_path = common_dir.join(WORKTREE_PARENT_DIR).canonicalize().unwrap();
        let canonical_worktree_path = worktree_path.canonicalize().unwrap();
        let worktree_parent_path = worktree_path.parent().unwrap().canonicalize().unwrap();

        assert!(
            canonical_worktree_path.starts_with(&worktree_parent_path),
            "worktree path `{}` was not under worktree parent `{}`",
            canonical_worktree_path.display(),
            worktree_parent_path.display()
        );
        assert!(
            canonical_worktree_path.starts_with(&common_dir),
            "worktree path `{}` was not under git common dir `{}`",
            canonical_worktree_path.display(),
            common_dir.display()
        );
        assert!(
            worktree_parent_path.starts_with(&common_dir),
            "worktree parent `{}` was not under git common dir `{}`",
            worktree_parent_path.display(),
            common_dir.display()
        );
        assert_eq!(
            worktree_parent_path, expected_parent_path,
            "worktree parent did not use expected git-common-dir parent"
        );
        assert!(
            worktree_parent_path != common_dir,
            "worktree parent `{}` must not be the git common dir itself `{}`",
            worktree_parent_path.display(),
            common_dir.display()
        );
        assert!(
            worktree_parent_path.is_dir(),
            "worktree parent `{}` did not exist",
            worktree_parent_path.display(),
        );
        assert_eq!(
            worktree_parent_path.file_name().unwrap(),
            OsStr::new(WORKTREE_PARENT_DIR),
            "worktree parent `{}` did not use expected directory `{}`",
            worktree_parent_path.display(),
            WORKTREE_PARENT_DIR
        );

        worktree_parent_path
    }

    fn assert_no_consensus_artifacts(repo_dir: &Path) {
        let refs = branch_refs(repo_dir);
        assert!(
            !refs
                .lines()
                .any(|line| line.starts_with("refs/heads/cubi-consensus/")),
            "temporary branch was left behind:\n{refs}"
        );

        let worktrees = worktree_list(repo_dir).replace('\\', "/");
        assert!(
            !worktrees.contains("cubi-consensus-"),
            "temporary worktree was left behind:\n{worktrees}"
        );

        let common_dir = resolve_git_common_dir(repo_dir).unwrap();
        let leftover_parent_dir = common_dir.join(WORKTREE_PARENT_DIR);
        assert!(
            !leftover_parent_dir.exists(),
            "temporary worktree parent directory was left behind: `{}`",
            leftover_parent_dir.display()
        );
    }

    fn assert_safe_session_name(name: &str, expected_prefix: &str) {
        assert!(
            name.starts_with(expected_prefix),
            "name `{name}` did not start with `{expected_prefix}`"
        );
        assert!(
            name.chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '/'),
            "name `{name}` contained an unsafe character"
        );
        assert!(!name.contains(".."), "name `{name}` contained `..`");
        assert!(!name.contains("@{"), "name `{name}` contained `@{{`");
        assert!(!name.contains("//"), "name `{name}` contained `//`");
    }

    fn run_test_git(repo_dir: &Path, args: &[&str]) -> String {
        let output = Command::new("git")
            .current_dir(repo_dir)
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git {args:?} failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).into_owned()
    }
}
