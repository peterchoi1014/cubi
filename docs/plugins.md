# Plugins cookbook

Plugins are prompt bundles discovered from `~/.cubi/plugins/`. Each direct child directory is a namespace, and each Markdown file under `commands/` becomes a slash command.

```text
~/.cubi/plugins/mytools/
  plugin.json
  commands/review.md
```

`plugin.json`, `manifest.json`, or `package.json` may contain `{ "version": "1.0.0" }`; the version is shown by `cubi plugins list`. Command files use the filename as the command name, so the example above creates `/mytools:review`. The first non-empty Markdown line becomes the help summary; the whole file is injected as the prompt when invoked. Extra user text is appended as `User argument: ...`.

Use `cubi plugins list` to inspect discovered bundles. Use `cubi plugins reload` after editing plugin files; it reloads skills and plugins and reports added or removed bundles.
