# Contributing to Cubi

Thanks for your interest in improving Cubi! Contributions — bug reports, fixes,
docs, and features — are welcome.

## Before you start

- **Bugs / features:** open an [issue](https://github.com/peterchoi1014/cubi/issues)
  first for anything non-trivial, so we can agree on the approach before you
  invest time.
- **Security issues:** do **not** file a public issue — see
  [SECURITY.md](SECURITY.md).
- Keep pull requests **focused**: one logical change per PR, targeting `main`.

## Development setup

- **Rust 1.88+** (this is the MSRV; CI gates on it).
- **Ollama** (or another OpenAI-compatible local server) to actually run the
  agent — see [INSTALL.md](INSTALL.md).

See **[DEVELOPMENT.md](DEVELOPMENT.md)** for the full build/test loop,
architecture, and project layout. The short version:

```bash
cargo build                                        # debug build
cargo test                                         # unit + integration suite
cargo +1.88 clippy --all-targets -- -D warnings    # lints (CI gates on 1.88)
cargo fmt --all                                    # format (edition 2024)
cargo deny check                                   # advisories / licenses / bans
```

CI runs on macOS, Linux, and Windows — make sure your change builds and passes
on all three (avoid platform-specific assumptions, or gate them with `cfg`).

## Guidelines

- **Add tests** for new or changed behavior. Prefer the smallest targeted test
  that covers the change.
- **No new dependencies** unless clearly necessary; they must pass
  `cargo deny check`.
- **The command surface is defined in [`src/commands.rs`](src/commands.rs)** —
  update it (and the completions/snapshots) when adding or changing a slash
  command.
- **Match existing conventions.** Idiomatic Rust: prefer `?` over `unwrap()`,
  handle errors, keep public items documented.

## Commit messages

Describe **what changed and why** in the imperative mood ("Add…", "Fix…",
"Refactor…"). Keep the subject concise; put rationale in the body. Do not
include tooling/process noise or references to AI assistants. See the
"Commit message conventions" section of [DEVELOPMENT.md](DEVELOPMENT.md) for
detail.

## Pull requests

- Open against `main` with a clear description and linked issue.
- Ensure `cargo fmt`, `cargo clippy -D warnings`, `cargo test`, and
  `cargo deny check` are green.
- Be ready to iterate on review feedback.

## License

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
