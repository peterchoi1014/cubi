//! Project trust + path sandboxing for built-in tools.
//!
//! Built-in tools like `bash`, `write_file`, and `edit_file` execute
//! immediately against the user's machine. Before this module existed the
//! only guard was a hard-coded denylist of well-known footguns in
//! `execute_bash` (`rm -rf /`, `dd if=`, ...). That's both too strict (it
//! flags benign commands containing the substring) and far too lax (any
//! `bash` invocation outside that list is unconditionally allowed, even in
//! a directory the user has never approved this CLI to touch).
//!
//! The model here is intentionally simple:
//!
//! * A **trust store** at `$CUBI_HOME/.cubi/trusted_dirs.json` (falling
//!   back to `~/.cubi/trusted_dirs.json`) records the
//!   set of directory roots the user has explicitly approved. Approval is a
//!   conscious one-time act per project (`/trust` slash command, or the
//!   prompt shown by the first-run wizard).
//! * A path is **writable** iff it canonicalizes to a location inside one
//!   of those trusted roots. Edits and writes outside any trusted root are
//!   refused before they hit the disk.
//! * Shell execution is allowed iff the **current working directory** is
//!   inside a trusted root. This deliberately keeps the surface narrow:
//!   running `bash` from an untrusted cwd is the high-risk case.
//!
//! Plan mode (gated separately in commit 3) layers on top: even in a
//! trusted directory, write/exec tools refuse while plan mode is on.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// On-disk representation of [`Permissions`]. Kept as a separate struct so
/// the file format can grow (per-tool allow/deny lists, expiry timestamps,
/// ...) without churn in the in-memory API.
#[derive(Debug, Default, Serialize, Deserialize)]
struct TrustFile {
    /// Canonical absolute paths of directories the user has trusted.
    /// `BTreeSet` so the on-disk file is stable and diff-friendly.
    #[serde(default)]
    trusted_roots: BTreeSet<PathBuf>,
    #[serde(default)]
    pub allowed_tools: BTreeSet<String>,
    #[serde(default)]
    pub denied_tools: BTreeSet<String>,
}

/// In-memory permissions snapshot. Cheap to clone; persists changes
/// eagerly so a crash never loses an approval the user just granted.
#[derive(Debug, Default, Clone)]
pub struct Permissions {
    trusted_roots: BTreeSet<PathBuf>,
    allowed_tools: BTreeSet<String>,
    denied_tools: BTreeSet<String>,
}

impl Permissions {
    /// Loads `~/.cubi/trusted_dirs.json`. Missing or unreadable
    /// files yield an empty permissions set rather than an error: a
    /// well-formed absence simply means "no projects trusted yet".
    pub fn load() -> Self {
        let Some(path) = Self::storage_path() else {
            return Self::default();
        };
        match fs::read_to_string(&path) {
            Ok(raw) => match serde_json::from_str::<TrustFile>(&raw) {
                Ok(file) => Self {
                    trusted_roots: file.trusted_roots,
                    allowed_tools: file.allowed_tools,
                    denied_tools: file.denied_tools,
                },
                Err(_) => {
                    // Don't silently nuke a corrupt file — start empty in
                    // memory but leave the file in place so the user can
                    // inspect it. Saving will overwrite once they make a
                    // change.
                    Self::default()
                }
            },
            Err(_) => Self::default(),
        }
    }

    fn storage_path() -> Option<PathBuf> {
        Some(crate::sessions::cubi_dir()?.join("trusted_dirs.json"))
    }

    /// Persists the current trust set. Errors are surfaced so callers can
    /// decide whether to nag or swallow them.
    pub fn save(&self) -> Result<()> {
        let path = Self::storage_path().context("Could not resolve home directory")?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create {}", parent.display()))?;
        }
        let file = TrustFile {
            trusted_roots: self.trusted_roots.clone(),
            allowed_tools: self.allowed_tools.clone(),
            denied_tools: self.denied_tools.clone(),
        };
        let json = serde_json::to_string_pretty(&file)?;
        fs::write(&path, json).with_context(|| format!("Failed to write {}", path.display()))?;
        Ok(())
    }

    /// Adds `dir` to the trust set after canonicalization. The canonical
    /// form is stored so symlink games can't be used to fool
    /// [`Self::contains`]. Returns `true` if the directory was newly added.
    pub fn trust_dir(&mut self, dir: &Path) -> Result<bool> {
        let canonical = fs::canonicalize(dir)
            .with_context(|| format!("Failed to canonicalize {}", dir.display()))?;
        if !canonical.is_dir() {
            anyhow::bail!("Not a directory: {}", canonical.display());
        }
        Ok(self.trusted_roots.insert(canonical))
    }

    /// Removes `dir` from the trust set. Returns `true` if a matching
    /// entry was found.
    pub fn revoke_dir(&mut self, dir: &Path) -> Result<bool> {
        let canonical = fs::canonicalize(dir)
            .with_context(|| format!("Failed to canonicalize {}", dir.display()))?;
        Ok(self.trusted_roots.remove(&canonical))
    }

    /// Returns true when `path` (or any of its ancestors, after
    /// canonicalization) is in the trust set.
    pub fn contains(&self, path: &Path) -> bool {
        let Ok(canonical) = fs::canonicalize(path) else {
            return false;
        };
        canonical
            .ancestors()
            .any(|a| self.trusted_roots.contains(a))
    }

    /// Snapshot of the trusted roots in stable order. Cheap to iterate.
    pub fn trusted_roots(&self) -> impl Iterator<Item = &PathBuf> {
        self.trusted_roots.iter()
    }

    /// Verifies a write target. The path itself need not exist (we're
    /// about to create it), but its nearest existing ancestor must
    /// canonicalize to somewhere inside a trusted root. This catches both
    /// "write to untrusted dir" and `..`-escape attempts.
    pub fn check_write(&self, path: &Path) -> Result<()> {
        let absolute = absolutize(path)?;
        // Walk up until we find a real, existing directory we can
        // canonicalize. The nearest-existing-ancestor rule means a
        // brand-new file under a trusted dir is fine, but constructing
        // a path that would escape via `..` still gets caught because
        // canonicalize resolves the `..` components before we check.
        let mut cursor: &Path = &absolute;
        let canonical_root = loop {
            if let Ok(real) = fs::canonicalize(cursor) {
                break real;
            }
            match cursor.parent() {
                Some(parent) if parent != cursor => cursor = parent,
                _ => {
                    anyhow::bail!(
                        "Refusing write to '{}': no real ancestor could be canonicalized",
                        path.display()
                    );
                }
            }
        };

        if canonical_root
            .ancestors()
            .any(|a| self.trusted_roots.contains(a))
        {
            Ok(())
        } else {
            anyhow::bail!(
                "Refusing write to '{}': path is outside any trusted root. \
                 Run `/trust` in the project directory to approve it.",
                path.display()
            )
        }
    }

    /// Verifies that shell execution is permitted in `cwd`.
    pub fn check_exec(&self, cwd: &Path) -> Result<()> {
        let absolute = absolutize(cwd)?;
        if self.contains(&absolute) {
            Ok(())
        } else {
            anyhow::bail!(
                "Refusing to execute shell command: '{}' is not a trusted directory. \
                 Run `/trust` in the project directory to approve it.",
                absolute.display()
            )
        }
    }

    pub fn allow_tool(&mut self, tool: &str) {
        let name = tool.trim();
        if name.is_empty() {
            return;
        }
        self.allowed_tools.insert(name.to_string());
        self.denied_tools.remove(name);
    }

    pub fn deny_tool(&mut self, tool: &str) {
        let name = tool.trim();
        if name.is_empty() {
            return;
        }
        self.denied_tools.insert(name.to_string());
        self.allowed_tools.remove(name);
    }

    #[allow(dead_code)]
    pub fn undeny_tool(&mut self, tool: &str) {
        self.denied_tools.remove(tool.trim());
    }

    #[allow(dead_code)]
    pub fn unallow_tool(&mut self, tool: &str) {
        self.allowed_tools.remove(tool.trim());
    }

    pub fn check_tool_allowed(&self, tool: &str) -> bool {
        let name = tool.trim();
        !self.denied_tools.contains(name)
            && (self.allowed_tools.is_empty() || self.allowed_tools.contains(name))
    }

    pub fn allowed_tools(&self) -> impl Iterator<Item = &String> {
        self.allowed_tools.iter()
    }

    pub fn denied_tools(&self) -> impl Iterator<Item = &String> {
        self.denied_tools.iter()
    }

    /// Returns the number of trusted roots — useful for `/status`.
    pub fn trusted_count(&self) -> usize {
        self.trusted_roots.len()
    }
}

/// Returns an absolute, lexically-normalized path. Used as a fallback when
/// `canonicalize` can't run because the leaf doesn't exist yet.
fn absolutize(p: &Path) -> Result<PathBuf> {
    if p.is_absolute() {
        Ok(p.to_path_buf())
    } else {
        let cwd = std::env::current_dir().context("Could not read current working directory")?;
        Ok(cwd.join(p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compat::test_env::env_guard;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct EnvRestore {
        cubi_home: Option<std::ffi::OsString>,
        home: Option<std::ffi::OsString>,
        userprofile: Option<std::ffi::OsString>,
    }

    impl EnvRestore {
        fn capture() -> Self {
            Self {
                cubi_home: std::env::var_os(crate::sessions::CUBI_HOME_ENV),
                home: std::env::var_os("HOME"),
                userprofile: std::env::var_os("USERPROFILE"),
            }
        }
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            unsafe {
                if let Some(value) = &self.cubi_home {
                    std::env::set_var(crate::sessions::CUBI_HOME_ENV, value);
                } else {
                    std::env::remove_var(crate::sessions::CUBI_HOME_ENV);
                }
                if let Some(value) = &self.home {
                    std::env::set_var("HOME", value);
                } else {
                    std::env::remove_var("HOME");
                }
                if let Some(value) = &self.userprofile {
                    std::env::set_var("USERPROFILE", value);
                } else {
                    std::env::remove_var("USERPROFILE");
                }
            }
        }
    }

    fn unique_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("cubi-perm-{label}-{nanos}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn trust_dir_inserts_canonical_path() {
        let dir = unique_dir("trust");
        let mut perms = Permissions::default();
        assert!(perms.trust_dir(&dir).unwrap());
        // Second call is a no-op.
        assert!(!perms.trust_dir(&dir).unwrap());
        assert!(perms.contains(&dir));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn contains_walks_ancestors() {
        let root = unique_dir("anc");
        let nested = root.join("a").join("b");
        fs::create_dir_all(&nested).unwrap();

        let mut perms = Permissions::default();
        perms.trust_dir(&root).unwrap();

        assert!(perms.contains(&nested));
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn check_write_rejects_outside_trusted_root() {
        let trusted = unique_dir("write-trusted");
        let untrusted = unique_dir("write-untrusted");

        let mut perms = Permissions::default();
        perms.trust_dir(&trusted).unwrap();

        // Existing trusted file: allowed.
        let inside = trusted.join("new.txt");
        perms.check_write(&inside).expect("inside trusted root");

        // Outside trusted root: refused.
        let outside = untrusted.join("nope.txt");
        let err = perms.check_write(&outside).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("outside any trusted root"), "got: {msg}");

        fs::remove_dir_all(&trusted).ok();
        fs::remove_dir_all(&untrusted).ok();
    }

    #[test]
    fn cubi_home_env_overrides_platform_home_for_trust_store() {
        let _guard = env_guard();
        let _restore = EnvRestore::capture();
        let cubi_home = tempfile::tempdir().unwrap();
        let other_home = tempfile::tempdir().unwrap();
        let other_cubi = other_home.path().join(".cubi");
        fs::create_dir_all(&other_cubi).unwrap();
        fs::write(
            other_cubi.join("trusted_dirs.json"),
            r#"{"allowed_tools":["from-home"]}"#,
        )
        .unwrap();
        unsafe {
            std::env::set_var(crate::sessions::CUBI_HOME_ENV, cubi_home.path());
            std::env::set_var("HOME", other_home.path());
            std::env::set_var("USERPROFILE", other_home.path());
        }

        let mut saved = Permissions::default();
        saved.allow_tool("from-cubi-home");
        saved.save().unwrap();

        let cubi_trust_path = cubi_home.path().join(".cubi").join("trusted_dirs.json");
        assert_eq!(
            Permissions::storage_path().as_deref(),
            Some(cubi_trust_path.as_path())
        );
        assert!(cubi_trust_path.exists());
        assert!(
            fs::read_to_string(other_cubi.join("trusted_dirs.json"))
                .unwrap()
                .contains("from-home"),
            "HOME trust store should be left untouched when CUBI_HOME is set"
        );
        let loaded = Permissions::load();
        assert!(loaded.check_tool_allowed("from-cubi-home"));
        assert!(!loaded.check_tool_allowed("from-home"));
    }

    #[test]
    fn check_write_blocks_dotdot_escape() {
        let trusted = unique_dir("escape-trusted");
        let mut perms = Permissions::default();
        perms.trust_dir(&trusted).unwrap();

        // `<trusted>/../../etc/passwd` canonicalizes away from the trusted
        // root and must be refused.
        let escape = trusted.join("..").join("..").join("etc").join("passwd");
        assert!(perms.check_write(&escape).is_err());

        fs::remove_dir_all(&trusted).ok();
    }

    #[test]
    fn check_exec_requires_trusted_cwd() {
        let trusted = unique_dir("exec-trusted");
        let untrusted = unique_dir("exec-untrusted");
        let mut perms = Permissions::default();
        perms.trust_dir(&trusted).unwrap();

        perms.check_exec(&trusted).expect("trusted cwd ok");
        assert!(perms.check_exec(&untrusted).is_err());

        fs::remove_dir_all(&trusted).ok();
        fs::remove_dir_all(&untrusted).ok();
    }

    #[test]
    fn revoke_removes_entry() {
        let dir = unique_dir("revoke");
        let mut perms = Permissions::default();
        perms.trust_dir(&dir).unwrap();
        assert!(perms.contains(&dir));
        assert!(perms.revoke_dir(&dir).unwrap());
        assert!(!perms.contains(&dir));
        // Second revoke is a no-op (returns false).
        assert!(!perms.revoke_dir(&dir).unwrap());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn allow_tool_empty_or_whitespace_is_noop() {
        let mut perms = Permissions::default();
        perms.allow_tool("");
        perms.allow_tool("   ");
        assert_eq!(perms.allowed_tools().count(), 0);
    }

    #[test]
    fn deny_tool_empty_or_whitespace_is_noop() {
        let mut perms = Permissions::default();
        perms.deny_tool("");
        perms.deny_tool("   ");
        assert_eq!(perms.denied_tools().count(), 0);
    }

    #[test]
    fn check_tool_allowed_empty_allow_list_means_allow_all() {
        let perms = Permissions::default();
        // No allowed/denied entries: every tool is allowed.
        assert!(perms.check_tool_allowed("any_tool"));
        assert!(perms.check_tool_allowed("another_tool"));
    }

    #[test]
    fn allow_tool_restricts_to_listed_tools() {
        let mut perms = Permissions::default();
        perms.allow_tool("bash");
        assert!(perms.check_tool_allowed("bash"));
        assert!(!perms.check_tool_allowed("grep"));
    }

    #[test]
    fn deny_tool_blocks_tool_regardless_of_allow_list() {
        let mut perms = Permissions::default();
        // deny beats allow
        perms.allow_tool("bash");
        perms.deny_tool("bash");
        assert!(!perms.check_tool_allowed("bash"));
    }

    #[test]
    fn deny_tool_removes_from_allowed_tools() {
        let mut perms = Permissions::default();
        perms.allow_tool("bash");
        assert!(perms.check_tool_allowed("bash"));
        perms.deny_tool("bash");
        // 'bash' must no longer appear in the allowed set.
        assert!(!perms.allowed_tools().any(|t| t == "bash"));
        assert!(!perms.check_tool_allowed("bash"));
    }

    #[test]
    fn allow_tool_removes_from_denied_tools() {
        let mut perms = Permissions::default();
        perms.deny_tool("bash");
        assert!(!perms.check_tool_allowed("bash"));
        perms.allow_tool("bash");
        // 'bash' must no longer appear in the denied set.
        assert!(!perms.denied_tools().any(|t| t == "bash"));
        assert!(perms.check_tool_allowed("bash"));
    }

    #[test]
    fn tool_names_are_trimmed() {
        let mut perms = Permissions::default();
        perms.allow_tool("  bash  ");
        assert!(perms.check_tool_allowed("bash"));
        perms.deny_tool("  bash  ");
        assert!(!perms.check_tool_allowed("bash"));
    }
}
