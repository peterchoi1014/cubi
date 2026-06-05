//! Integration tests for the repo-map builder.
//!
//! Builds a small synthetic project on disk and verifies that
//! [`cubi::repomap::RepoMap`] extracts the expected top-level symbols
//! for the languages we support. Works whether or not the `tree-sitter`
//! feature is enabled — the regex fallback recognizes the same shapes.

// NOTE: `cubi` is a `[[bin]]` crate, so its `mod repomap` isn't visible
// from integration tests as a library import. Re-running the small
// public surface via the binary's `--repomap-print` would be overkill;
// instead, this test file is wired up to compile only the bits it
// needs by including the module via a path attribute below.

#[path = "../src/repomap.rs"]
mod repomap;

use repomap::{RepoMap, RepoMapOptions, SymbolKind};
use std::fs;
use tempfile::TempDir;

fn write(dir: &std::path::Path, rel: &str, content: &str) {
    let path = dir.join(rel);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, content).unwrap();
}

#[test]
fn extracts_symbols_across_languages() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    write(root, "src/lib.rs", "pub fn foo() {}\npub struct Bar;\n");
    write(
        root,
        "src/util.py",
        "def helper():\n    return 1\n\nclass Thing:\n    pass\n",
    );
    write(
        root,
        "src/index.ts",
        "export function load() {}\nexport class Loader {}\n",
    );

    let opts = RepoMapOptions::default();
    let outline = RepoMap::build(root, &opts).expect("build outline");

    // Flatten to (name, kind) tuples for easy assertions.
    let symbols: Vec<(String, SymbolKind)> = outline
        .files
        .iter()
        .flat_map(|f| f.symbols.iter().map(|s| (s.name.clone(), s.kind)))
        .collect();

    let has = |name: &str, kind: SymbolKind| symbols.iter().any(|(n, k)| n == name && *k == kind);

    assert!(
        has("foo", SymbolKind::Function),
        "missing fn foo: {symbols:?}"
    );
    assert!(
        has("Bar", SymbolKind::Struct),
        "missing struct Bar: {symbols:?}"
    );
    assert!(
        has("helper", SymbolKind::Function),
        "missing def helper: {symbols:?}"
    );
    assert!(
        has("Thing", SymbolKind::Class),
        "missing class Thing: {symbols:?}"
    );
    assert!(
        has("load", SymbolKind::Function),
        "missing fn load: {symbols:?}"
    );
    assert!(
        has("Loader", SymbolKind::Class),
        "missing class Loader: {symbols:?}"
    );

    // Rendered output should mention each file and the symbol count.
    let rendered = RepoMap::render(&outline);
    assert!(rendered.contains("src/lib.rs"));
    assert!(rendered.contains("src/util.py"));
    assert!(rendered.contains("src/index.ts"));
    assert!(rendered.contains("foo"));
}
