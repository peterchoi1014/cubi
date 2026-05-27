//! User-defined plugin loader.
//!
//! Roadmap item B-foundation + C#15: discover plugin bundles under
//! `~/.cubi/plugins/` and expose any Markdown command files
//! (`<plugin>/commands/<name>.md`) as first-class user-defined slash
//! commands. The body of the Markdown file is injected verbatim as the
//! user-visible prompt for `/<plugin>:<name>` (Anthropic-style
//! namespacing) so people can ship reusable prompt packs without
//! recompiling the binary.
//!
//! Discovery is best-effort: missing directories, unreadable files, and
//! plugins with no `commands/` subtree are all silently skipped so the
//! CLI keeps starting on a fresh machine.

use std::fs;
use std::path::{Path, PathBuf};

/// A discovered plugin bundle. We keep the metadata trio (name, root
/// path, command list) cheaply cloneable so the CLI can rebuild it on
/// `/reload-plugins` without holding a long-lived borrow.
#[derive(Debug, Clone)]
pub struct Plugin {
    /// Plugin directory name (no slashes), used as the slash-command
    /// namespace, e.g. `mytools` for `/mytools:review`.
    pub name: String,
    pub version: String,
    pub root: PathBuf,
    pub commands: Vec<PluginCommand>,
}

#[derive(Debug, Clone)]
pub struct PluginCommand {
    /// Bare command name, e.g. `review` for `/mytools:review`.
    pub name: String,
    /// First non-empty line of the Markdown body (with a leading `#`
    /// stripped) — used as the `/help` summary.
    pub description: String,
    /// Full Markdown body injected as the user prompt when invoked.
    pub body: String,
    pub path: PathBuf,
}

impl PluginCommand {
    /// `/mytools:review`-style trigger.
    pub fn trigger(&self, plugin: &str) -> String {
        format!("/{plugin}:{}", self.name)
    }
}

pub fn plugins_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".cubi").join("plugins"))
}

/// Reload all plugin bundles from disk. Always returns a (possibly
/// empty) vector — errors only suppress individual entries.
pub fn load_plugins() -> Vec<Plugin> {
    let Some(dir) = plugins_dir() else {
        return Vec::new();
    };
    let Ok(entries) = fs::read_dir(&dir) else {
        tracing::debug!(target: "cubi::plugins", dir = %dir.display(), "plugins dir not readable");
        return Vec::new();
    };

    let mut plugins = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
        else {
            continue;
        };
        // Reject anything that would be ambiguous as a slash-command
        // namespace. We never want `/foo bar:baz`.
        if name.is_empty() || name.contains(char::is_whitespace) || name.contains(':') {
            tracing::warn!(target: "cubi::plugins", name = %name, "skipping plugin with invalid name");
            continue;
        }
        let version = load_version(&path).unwrap_or_else(|| "-".to_string());
        let commands = load_commands(&path.join("commands"));
        tracing::debug!(
            target: "cubi::plugins",
            name = %name,
            version = %version,
            command_count = commands.len(),
            "loaded plugin"
        );
        plugins.push(Plugin {
            name,
            version,
            root: path,
            commands,
        });
    }
    plugins.sort_by(|a, b| a.name.cmp(&b.name));
    plugins
}

fn load_version(root: &std::path::Path) -> Option<String> {
    for file in ["plugin.json", "manifest.json", "package.json"] {
        let Ok(raw) = fs::read_to_string(root.join(file)) else {
            continue;
        };
        let Ok(json) = serde_json::from_str::<serde_json::Value>(&raw) else {
            continue;
        };
        if let Some(version) = json.get("version").and_then(|v| v.as_str()) {
            if !version.trim().is_empty() {
                return Some(version.trim().to_string());
            }
        }
    }
    None
}

pub fn print_plugin_list(plugins: &[Plugin]) {
    println!("{:<24} {:<12} PATH", "NAME", "VERSION");
    if plugins.is_empty() {
        println!("(no plugins discovered)");
        return;
    }
    for plugin in plugins {
        println!(
            "{:<24} {:<12} {}",
            plugin.name,
            plugin.version,
            plugin.root.display()
        );
    }
}

pub fn print_reload_summary(before: &[Plugin], after: &[Plugin], skill_count: usize) {
    let before_names: std::collections::BTreeSet<_> =
        before.iter().map(|p| p.name.as_str()).collect();
    let after_names: std::collections::BTreeSet<_> =
        after.iter().map(|p| p.name.as_str()).collect();
    let added: Vec<_> = after_names.difference(&before_names).copied().collect();
    let removed: Vec<_> = before_names.difference(&after_names).copied().collect();
    let cmd_count: usize = after.iter().map(|p| p.commands.len()).sum();
    println!(
        "Reloaded {} skill(s) + {} plugin(s) ({} command(s))",
        skill_count,
        after.len(),
        cmd_count
    );
    if !added.is_empty() {
        println!("Added: {}", added.join(", "));
    }
    if !removed.is_empty() {
        println!("Removed: {}", removed.join(", "));
    }
    if added.is_empty() && removed.is_empty() {
        println!("No plugin bundle changes detected.");
    }
}

fn load_commands(commands_dir: &std::path::Path) -> Vec<PluginCommand> {
    let Ok(entries) = fs::read_dir(commands_dir) else {
        return Vec::new();
    };

    let mut cmds = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        let Ok(body) = fs::read_to_string(&path) else {
            continue;
        };
        let Some(name) = path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
        else {
            continue;
        };
        if name.is_empty() || name.contains(char::is_whitespace) || name.contains(':') {
            continue;
        }
        let description = body
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
            .map(|line| line.trim_start_matches('#').trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| name.clone());
        cmds.push(PluginCommand {
            name,
            description,
            body,
            path,
        });
    }
    cmds.sort_by(|a, b| a.name.cmp(&b.name));
    cmds
}

/// Look up `<plugin>:<command>` in the cached plugin list. Returns the
/// fully-rendered prompt body, ready to forward to the model.
pub fn resolve<'a>(plugins: &'a [Plugin], trigger: &str) -> Option<&'a PluginCommand> {
    let stripped = trigger.strip_prefix('/').unwrap_or(trigger);
    let (ns, cmd) = stripped.split_once(':')?;
    plugins
        .iter()
        .find(|p| p.name == ns)?
        .commands
        .iter()
        .find(|c| c.name == cmd)
}

/// Returns `true` when `name` is a syntactically valid plugin
/// directory name: ASCII alphanumeric plus `-` / `_`, length 1..=64.
pub fn is_valid_plugin_name(name: &str) -> bool {
    if name.is_empty() || name.len() > 64 {
        return false;
    }
    name.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// Scaffolds a fresh plugin bundle at `~/.cubi/plugins/<name>/` with a
/// minimal manifest, an executable handler stub, and a README. Returns
/// the plugin root on success.
pub fn scaffold_new(name: &str) -> anyhow::Result<PathBuf> {
    use anyhow::anyhow;
    let parent = scaffold_root().ok_or_else(|| anyhow!("could not resolve plugins directory"))?;
    scaffold_new_in(&parent, name)
}

/// Internal variant taking an explicit parent directory. Lets tests
/// scaffold into per-test tempdirs without racing on a shared
/// `CUBI_PLUGINS_DIR` env var.
fn scaffold_new_in(parent: &Path, name: &str) -> anyhow::Result<PathBuf> {
    use anyhow::{Context, anyhow};

    if !is_valid_plugin_name(name) {
        return Err(anyhow!(
            "invalid plugin name '{}': use ASCII alphanumeric plus '-' or '_' (≤ 64 chars)",
            name
        ));
    }
    let root = parent.join(name);
    if root.exists() {
        return Err(anyhow!(
            "{} already exists; refusing to overwrite",
            root.display()
        ));
    }
    fs::create_dir_all(&root).with_context(|| format!("failed to create {}", root.display()))?;

    // 1) manifest.json — mirrors what load_version expects.
    let entry = if cfg!(windows) {
        "handler.cmd"
    } else {
        "handler.sh"
    };
    let manifest = serde_json::json!({
        "name": name,
        "version": "0.1.0",
        "description": format!("Plugin '{}' scaffolded by `cubi plugins new`", name),
        "entry": entry,
    });
    fs::write(
        root.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )
    .with_context(|| "failed to write manifest.json")?;

    // 2) handler stub. Echoes its argv so the user sees something the
    //    first time they invoke it from the REPL.
    #[cfg(unix)]
    {
        let body = "#!/usr/bin/env bash\nset -euo pipefail\necho \"hello from $0: $*\"\n";
        let handler = root.join("handler.sh");
        fs::write(&handler, body).with_context(|| "failed to write handler.sh")?;
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&handler)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&handler, perms)?;
    }
    #[cfg(windows)]
    {
        let body = "@echo off\r\necho hello from %~nx0: %*\r\n";
        fs::write(root.join("handler.cmd"), body).with_context(|| "failed to write handler.cmd")?;
    }

    // 3) README quickstart.
    let readme = format!(
        "# {name} plugin\n\n\
         Scaffolded by `cubi plugins new`.\n\n\
         ## How cubi loads this plugin\n\n\
         - Directory: `~/.cubi/plugins/{name}/`\n\
         - `manifest.json` declares the plugin name, version, and entry script.\n\
         - Drop Markdown command files into `commands/<name>.md` to expose them\n  \
           as `/{name}:<command>` slash commands in the REPL.\n\
         - Run `cubi plugins reload` to pick up changes without restarting.\n\n\
         ## Next steps\n\n\
         1. Edit `{entry}` to wire your plugin to whatever it needs to do.\n\
         2. Create `commands/hello.md` with the prompt body for `/{name}:hello`.\n\
         3. Reload and try it from the REPL.\n",
        name = name,
        entry = entry,
    );
    fs::write(root.join("README.md"), readme).with_context(|| "failed to write README.md")?;

    Ok(root)
}

/// Resolves the plugins root for scaffolding. Honors the explicit
/// `CUBI_PLUGINS_DIR` env var (used by tests) before falling back to
/// `~/.cubi/plugins/`.
fn scaffold_root() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CUBI_PLUGINS_DIR") {
        if !p.is_empty() {
            return Some(PathBuf::from(p));
        }
    }
    plugins_dir()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("cubi-plugins-{label}-{nanos}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    /// Builds a plugin tree by hand and exercises the loader without
    /// depending on the user's real home directory.
    fn write_plugin(root: &std::path::Path, plugin: &str, command: &str, body: &str) {
        let dir = root.join(plugin).join("commands");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(format!("{command}.md")), body).unwrap();
    }

    /// Mirror of [`load_plugins`] that targets an explicit root so tests
    /// don't depend on `$HOME`.
    fn load_from(root: &std::path::Path) -> Vec<Plugin> {
        let entries = fs::read_dir(root).unwrap();
        let mut plugins = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            let commands = load_commands(&path.join("commands"));
            plugins.push(Plugin {
                name,
                version: "-".to_string(),
                root: path,
                commands,
            });
        }
        plugins.sort_by(|a, b| a.name.cmp(&b.name));
        plugins
    }

    #[test]
    fn loads_commands_with_namespaced_triggers() {
        let root = temp_root("ok");
        write_plugin(
            &root,
            "mytools",
            "review",
            "# Review code\n\nDo the review.\n",
        );
        let plugins = load_from(&root);
        assert_eq!(plugins.len(), 1);
        assert_eq!(plugins[0].name, "mytools");
        assert_eq!(plugins[0].commands.len(), 1);
        let c = &plugins[0].commands[0];
        assert_eq!(c.name, "review");
        assert_eq!(c.description, "Review code");
        assert_eq!(c.trigger("mytools"), "/mytools:review");
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn skips_non_markdown_files() {
        let root = temp_root("nonmd");
        let dir = root.join("p").join("commands");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("ignored.txt"), "nope").unwrap();
        fs::write(dir.join("ok.md"), "# Hi\n").unwrap();
        let plugins = load_from(&root);
        assert_eq!(plugins[0].commands.len(), 1);
        assert_eq!(plugins[0].commands[0].name, "ok");
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn skips_command_names_with_whitespace() {
        let root = temp_root("spaces");
        let dir = root.join("p").join("commands");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("my cmd.md"), "# should skip\n").unwrap();
        fs::write(dir.join("ok.md"), "# ok\n").unwrap();
        let plugins = load_from(&root);
        assert_eq!(plugins[0].commands.len(), 1);
        assert_eq!(plugins[0].commands[0].name, "ok");
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn resolve_matches_namespaced_trigger() {
        let root = temp_root("resolve");
        write_plugin(&root, "mytools", "review", "Body\n");
        let plugins = load_from(&root);
        assert!(resolve(&plugins, "/mytools:review").is_some());
        assert!(resolve(&plugins, "mytools:review").is_some());
        assert!(resolve(&plugins, "/mytools:missing").is_none());
        assert!(resolve(&plugins, "/missing:review").is_none());
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn description_falls_back_to_name_when_body_empty() {
        let root = temp_root("emptybody");
        write_plugin(&root, "p", "noop", "");
        let plugins = load_from(&root);
        assert_eq!(plugins[0].commands[0].description, "noop");
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn is_valid_plugin_name_rules() {
        for ok in ["a", "tools", "my-plugin", "my_plugin", "Plugin1"] {
            assert!(is_valid_plugin_name(ok), "expected ok: {ok}");
        }
        for bad in ["", "spaces here", "colon:bad", "slash/bad", "dot.bad"] {
            assert!(!is_valid_plugin_name(bad), "expected bad: {bad}");
        }
        // 64 chars ok, 65 not.
        let s64 = "a".repeat(64);
        let s65 = "a".repeat(65);
        assert!(is_valid_plugin_name(&s64));
        assert!(!is_valid_plugin_name(&s65));
    }

    #[test]
    fn scaffold_new_creates_manifest_handler_and_readme() {
        let root = temp_root("scaffold");
        let path = scaffold_new_in(&root, "myplug").expect("scaffold ok");

        let manifest: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(path.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["name"], "myplug");
        assert_eq!(manifest["version"], "0.1.0");
        assert!(manifest["description"].is_string());
        let entry = if cfg!(windows) {
            "handler.cmd"
        } else {
            "handler.sh"
        };
        assert_eq!(manifest["entry"], entry);
        assert!(path.join(entry).exists());
        assert!(path.join("README.md").exists());

        fs::remove_dir_all(&root).ok();
    }

    #[cfg(unix)]
    #[test]
    fn scaffold_new_makes_unix_handler_executable() {
        use std::os::unix::fs::PermissionsExt;
        let root = temp_root("scaffold-mode");
        let path = scaffold_new_in(&root, "modeplug").expect("scaffold ok");
        let perms = fs::metadata(path.join("handler.sh")).unwrap().permissions();
        assert_eq!(perms.mode() & 0o777, 0o755);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn scaffold_new_refuses_to_overwrite_existing_directory() {
        let root = temp_root("dup");
        scaffold_new_in(&root, "dup").expect("first scaffold ok");
        let err = scaffold_new_in(&root, "dup").expect_err("second scaffold must fail");
        assert!(format!("{err:#}").contains("already exists"));
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn scaffold_new_rejects_invalid_names() {
        let root = temp_root("badname");
        let err = scaffold_new_in(&root, "bad name").expect_err("invalid name must fail");
        assert!(format!("{err:#}").contains("invalid plugin name"));
        fs::remove_dir_all(&root).ok();
    }
}
