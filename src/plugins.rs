//! User-defined plugin loader.
//!
//! Roadmap item B-foundation + C#15: discover plugin bundles under
//! `~/.ai-chat-cli/plugins/` and expose any Markdown command files
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
use std::path::PathBuf;

/// A discovered plugin bundle. We keep the metadata trio (name, root
/// path, command list) cheaply cloneable so the CLI can rebuild it on
/// `/reload-plugins` without holding a long-lived borrow.
#[derive(Debug, Clone)]
pub struct Plugin {
    /// Plugin directory name (no slashes), used as the slash-command
    /// namespace, e.g. `mytools` for `/mytools:review`.
    pub name: String,
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
    dirs::home_dir().map(|h| h.join(".ai-chat-cli").join("plugins"))
}

/// Reload all plugin bundles from disk. Always returns a (possibly
/// empty) vector — errors only suppress individual entries.
pub fn load_plugins() -> Vec<Plugin> {
    let Some(dir) = plugins_dir() else {
        return Vec::new();
    };
    let Ok(entries) = fs::read_dir(&dir) else {
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
            continue;
        }
        let commands = load_commands(&path.join("commands"));
        plugins.push(Plugin {
            name,
            root: path,
            commands,
        });
    }
    plugins.sort_by(|a, b| a.name.cmp(&b.name));
    plugins
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
        if name.is_empty() || name.contains(':') {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("ai-chat-cli-plugins-{label}-{nanos}"));
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
}
