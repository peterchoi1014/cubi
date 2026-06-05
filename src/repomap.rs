//! Compact, tree-sitter-backed outline of a project.
//!
//! Walks the file tree honoring `.gitignore` / `.ignore`, detects the
//! language from the file extension, and extracts a shallow list of
//! top-level symbols (functions, types, classes, etc.) so the model can
//! orient itself in an unfamiliar codebase without paying to read every
//! file.
//!
//! Two modes:
//!   * `tree-sitter` feature ON (default) — parses with the upstream
//!     grammars and emits precise symbol kinds.
//!   * `tree-sitter` feature OFF — falls back to a tiny line-based
//!     scanner that catches obvious top-level declarations. Same
//!     [`Outline`] shape, just shallower.
//!
//! Results are cached at `<cache_dir>/cubi/repomap/<hash>.json`,
//! keyed on the canonical root + option knobs + the (path, mtime) list
//! of every walked file. Any mismatch rebuilds.

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

const MAX_FILE_BYTES: u64 = 1_000_000;

pub struct RepoMap;

#[derive(Debug, Clone)]
pub struct RepoMapOptions {
    pub scope: Option<PathBuf>,
    pub max_files: usize,
    pub max_symbols_per_file: usize,
}

impl Default for RepoMapOptions {
    fn default() -> Self {
        Self {
            scope: None,
            max_files: 200,
            max_symbols_per_file: 20,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outline {
    pub root: PathBuf,
    pub files: Vec<FileOutline>,
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileOutline {
    pub path: PathBuf,
    pub language: String,
    pub line_count: usize,
    pub symbols: Vec<Symbol>,
    /// True when the file had more symbols than `max_symbols_per_file`
    /// and we dropped the tail. Surfaced in [`RepoMap::render`].
    pub symbols_truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub line: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SymbolKind {
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
    Class,
    Method,
    Module,
    Const,
    TypeAlias,
}

impl SymbolKind {
    fn label(self) -> &'static str {
        match self {
            SymbolKind::Function => "fn",
            SymbolKind::Struct => "struct",
            SymbolKind::Enum => "enum",
            SymbolKind::Trait => "trait",
            SymbolKind::Impl => "impl",
            SymbolKind::Class => "class",
            SymbolKind::Method => "method",
            SymbolKind::Module => "mod",
            SymbolKind::Const => "const",
            SymbolKind::TypeAlias => "type",
        }
    }
}

fn detect_language(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("rs") => "rust",
        Some("py") => "python",
        Some("js") | Some("mjs") | Some("cjs") => "javascript",
        Some("ts") | Some("tsx") => "typescript",
        _ => "text",
    }
}

/// Cache entry persisted under `<cache_dir>/cubi/repomap/`.
#[derive(Debug, Serialize, Deserialize)]
struct CacheEntry {
    root: PathBuf,
    max_files: usize,
    max_symbols_per_file: usize,
    /// Sorted `(relative_path, mtime_nanos)` of every walked file.
    /// Used as the invalidation key so any add/remove/edit triggers
    /// a rebuild.
    fingerprint: Vec<(PathBuf, u128)>,
    outline: Outline,
}

impl RepoMap {
    pub fn build(root: &Path, opts: &RepoMapOptions) -> Result<Outline> {
        let scope = opts.scope.as_deref().unwrap_or(root);
        let canonical = fs::canonicalize(scope)
            .with_context(|| format!("Could not canonicalize scope: {}", scope.display()))?;

        let walked = walk_files(&canonical, opts.max_files)?;
        let fingerprint = make_fingerprint(&canonical, &walked);

        if let Some(cached) = load_cache(&canonical, opts, &fingerprint) {
            return Ok(cached);
        }

        let (truncated, walked) = if walked.len() > opts.max_files {
            (true, walked.into_iter().take(opts.max_files).collect())
        } else {
            (false, walked)
        };

        let mut files = Vec::with_capacity(walked.len());
        for entry in walked {
            match outline_one(&canonical, &entry, opts.max_symbols_per_file) {
                Ok(Some(fo)) => files.push(fo),
                Ok(None) => {}
                Err(_) => {
                    // Per-file errors (unreadable, parse failure) are
                    // non-fatal — the outline is best-effort.
                }
            }
        }

        let outline = Outline {
            root: canonical.clone(),
            files,
            truncated,
        };

        let _ = store_cache(&canonical, opts, &fingerprint, &outline);
        Ok(outline)
    }

    pub fn render(outline: &Outline) -> String {
        let total_symbols: usize = outline.files.iter().map(|f| f.symbols.len()).sum();
        let mut out = String::new();
        out.push_str(&format!(
            "# Repo outline (root: {}, {} files, {} symbols{})\n\n",
            outline.root.display(),
            outline.files.len(),
            total_symbols,
            if outline.truncated { ", truncated" } else { "" }
        ));
        for file in &outline.files {
            out.push_str(&format!(
                "## {} ({}, {} LoC)\n",
                file.path.display().to_string().replace('\\', "/"),
                file.language,
                file.line_count
            ));
            for sym in &file.symbols {
                out.push_str(&format!(
                    "- {} {}:{}\n",
                    sym.kind.label(),
                    sym.name,
                    sym.line
                ));
            }
            if file.symbols_truncated {
                out.push_str("- ... (truncated)\n");
            }
            out.push('\n');
        }
        out
    }
}

fn walk_files(root: &Path, soft_cap: usize) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    // We deliberately walk a little past `soft_cap` so the caller can
    // record `truncated = true` accurately.
    let cap = soft_cap.saturating_add(1);
    let walker = WalkBuilder::new(root)
        .standard_filters(true)
        .hidden(true)
        .build();
    for dent in walker.flatten() {
        if !dent.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        if let Ok(meta) = dent.metadata() {
            if meta.len() > MAX_FILE_BYTES {
                continue;
            }
        }
        out.push(dent.into_path());
        if out.len() >= cap {
            break;
        }
    }
    out.sort();
    Ok(out)
}

fn outline_one(root: &Path, abs_path: &Path, max_symbols: usize) -> Result<Option<FileOutline>> {
    let rel = abs_path
        .strip_prefix(root)
        .unwrap_or(abs_path)
        .to_path_buf();
    let language = detect_language(abs_path);
    let source = match fs::read_to_string(abs_path) {
        Ok(s) => s,
        Err(_) => return Ok(None), // binary or unreadable
    };
    let line_count = source.lines().count();

    let mut symbols = extract_symbols(language, &source);
    let symbols_truncated = symbols.len() > max_symbols;
    if symbols_truncated {
        symbols.truncate(max_symbols);
    }

    // Skip pure-text files that produced no symbols — they would just
    // be noise in the outline.
    if language == "text" && symbols.is_empty() {
        return Ok(None);
    }

    Ok(Some(FileOutline {
        path: rel,
        language: language.to_string(),
        line_count,
        symbols,
        symbols_truncated,
    }))
}

#[cfg(feature = "tree-sitter")]
fn extract_symbols(language: &'static str, source: &str) -> Vec<Symbol> {
    use tree_sitter::Parser;
    let mut parser = Parser::new();
    let lang = match language {
        "rust" => tree_sitter_rust::LANGUAGE.into(),
        "python" => tree_sitter_python::LANGUAGE.into(),
        "javascript" => tree_sitter_javascript::LANGUAGE.into(),
        "typescript" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        _ => return Vec::new(),
    };
    if parser.set_language(&lang).is_err() {
        return Vec::new();
    }
    let Some(tree) = parser.parse(source, None) else {
        return Vec::new();
    };
    let root = tree.root_node();
    let bytes = source.as_bytes();
    let mut out = Vec::new();
    collect(root, bytes, language, &mut out);
    out
}

#[cfg(feature = "tree-sitter")]
fn collect(node: tree_sitter::Node, bytes: &[u8], lang: &str, out: &mut Vec<Symbol>) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let kind = child.kind();
        let mapped = match lang {
            "rust" => match kind {
                "function_item" => Some(SymbolKind::Function),
                "struct_item" => Some(SymbolKind::Struct),
                "enum_item" => Some(SymbolKind::Enum),
                "trait_item" => Some(SymbolKind::Trait),
                "impl_item" => Some(SymbolKind::Impl),
                "mod_item" => Some(SymbolKind::Module),
                "const_item" | "static_item" => Some(SymbolKind::Const),
                "type_item" => Some(SymbolKind::TypeAlias),
                _ => None,
            },
            "python" => match kind {
                "function_definition" => Some(SymbolKind::Function),
                "class_definition" => Some(SymbolKind::Class),
                _ => None,
            },
            "javascript" | "typescript" => match kind {
                "function_declaration" => Some(SymbolKind::Function),
                "class_declaration" => Some(SymbolKind::Class),
                "method_definition" => Some(SymbolKind::Method),
                "interface_declaration" => Some(SymbolKind::TypeAlias),
                "type_alias_declaration" => Some(SymbolKind::TypeAlias),
                _ => None,
            },
            _ => None,
        };

        if let Some(symkind) = mapped {
            if let Some(name) = symbol_name(&child, bytes, lang) {
                out.push(Symbol {
                    name,
                    kind: symkind,
                    line: child.start_position().row + 1,
                });
            }
        }

        // Recurse into JS/TS `export_statement`, `export_default_declaration`,
        // and Rust impl bodies so we still pick up exported functions /
        // methods declared inside them.
        let recurse = matches!(
            kind,
            "export_statement"
                | "export_default_declaration"
                | "lexical_declaration"
                | "variable_declaration"
                | "impl_item"
                | "decorated_definition"
        );
        if recurse {
            collect(child, bytes, lang, out);
        }
    }
}

#[cfg(feature = "tree-sitter")]
fn symbol_name(node: &tree_sitter::Node, bytes: &[u8], lang: &str) -> Option<String> {
    // Most node types put the identifier in a `name` field.
    if let Some(name_node) = node.child_by_field_name("name") {
        if let Ok(text) = name_node.utf8_text(bytes) {
            return Some(text.to_string());
        }
    }
    // For Rust `impl_item` use the "type" field as the display name.
    if lang == "rust" && node.kind() == "impl_item" {
        if let Some(ty) = node.child_by_field_name("type") {
            if let Ok(text) = ty.utf8_text(bytes) {
                return Some(text.to_string());
            }
        }
    }
    None
}

#[cfg(not(feature = "tree-sitter"))]
fn extract_symbols(language: &'static str, source: &str) -> Vec<Symbol> {
    let mut out = Vec::new();
    for (idx, raw) in source.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw.trim_start();
        match language {
            "rust" => {
                let rest = strip_prefix(line, "pub ").unwrap_or(line);
                let rest = strip_visibility(rest);
                if let Some((kw, ident)) = split_keyword_ident(rest) {
                    let kind = match kw {
                        "fn" => Some(SymbolKind::Function),
                        "struct" => Some(SymbolKind::Struct),
                        "enum" => Some(SymbolKind::Enum),
                        "trait" => Some(SymbolKind::Trait),
                        "impl" => Some(SymbolKind::Impl),
                        "mod" => Some(SymbolKind::Module),
                        "const" | "static" => Some(SymbolKind::Const),
                        "type" => Some(SymbolKind::TypeAlias),
                        _ => None,
                    };
                    if let Some(kind) = kind {
                        out.push(Symbol {
                            name: ident,
                            kind,
                            line: line_no,
                        });
                    }
                }
            }
            "python" => {
                let rest = strip_prefix(line, "async ").unwrap_or(line);
                if let Some((kw, ident)) = split_keyword_ident(rest) {
                    let kind = match kw {
                        "def" => Some(SymbolKind::Function),
                        "class" => Some(SymbolKind::Class),
                        _ => None,
                    };
                    if let Some(kind) = kind {
                        out.push(Symbol {
                            name: ident,
                            kind,
                            line: line_no,
                        });
                    }
                }
            }
            "javascript" | "typescript" => {
                let rest = strip_prefix(line, "export ").unwrap_or(line);
                let rest = strip_prefix(rest, "default ").unwrap_or(rest);
                let rest = strip_prefix(rest, "async ").unwrap_or(rest);
                if let Some((kw, ident)) = split_keyword_ident(rest) {
                    let kind = match kw {
                        "function" => Some(SymbolKind::Function),
                        "class" => Some(SymbolKind::Class),
                        "interface" | "type" => Some(SymbolKind::TypeAlias),
                        _ => None,
                    };
                    if let Some(kind) = kind {
                        out.push(Symbol {
                            name: ident,
                            kind,
                            line: line_no,
                        });
                    }
                }
            }
            _ => {}
        }
    }
    out
}

#[cfg(not(feature = "tree-sitter"))]
fn strip_prefix<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    s.strip_prefix(prefix)
}

#[cfg(not(feature = "tree-sitter"))]
fn strip_visibility(s: &str) -> &str {
    // Strip Rust `pub(crate)` / `pub(super)` / `pub(in …)` qualifiers.
    if let Some(rest) = s.strip_prefix("pub(") {
        if let Some(end) = rest.find(')') {
            return rest[end + 1..].trim_start();
        }
    }
    s
}

#[cfg(not(feature = "tree-sitter"))]
fn split_keyword_ident(s: &str) -> Option<(&str, String)> {
    let mut it = s.splitn(3, |c: char| c.is_whitespace());
    let kw = it.next()?;
    let ident_raw = it.next()?;
    // Identifier ends at the first non-`[A-Za-z0-9_]` char.
    let end = ident_raw
        .find(|c: char| !c.is_alphanumeric() && c != '_')
        .unwrap_or(ident_raw.len());
    if end == 0 {
        return None;
    }
    Some((kw, ident_raw[..end].to_string()))
}

// -------- Cache --------

fn cache_root() -> Option<PathBuf> {
    dirs::cache_dir().map(|d| d.join("cubi").join("repomap"))
}

fn cache_path(root: &Path) -> Option<PathBuf> {
    let dir = cache_root()?;
    let mut hasher = DefaultHasher::new();
    root.hash(&mut hasher);
    let key = format!("{:016x}", hasher.finish());
    Some(dir.join(format!("{}.json", key)))
}

fn make_fingerprint(root: &Path, files: &[PathBuf]) -> Vec<(PathBuf, u128)> {
    let mut out = Vec::with_capacity(files.len());
    for f in files {
        let mtime = fs::metadata(f)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let rel = f.strip_prefix(root).unwrap_or(f).to_path_buf();
        out.push((rel, mtime));
    }
    out
}

fn load_cache(
    root: &Path,
    opts: &RepoMapOptions,
    fingerprint: &[(PathBuf, u128)],
) -> Option<Outline> {
    let path = cache_path(root)?;
    let bytes = fs::read(&path).ok()?;
    let entry: CacheEntry = serde_json::from_slice(&bytes).ok()?;
    if entry.root != root
        || entry.max_files != opts.max_files
        || entry.max_symbols_per_file != opts.max_symbols_per_file
        || entry.fingerprint != fingerprint
    {
        return None;
    }
    Some(entry.outline)
}

fn store_cache(
    root: &Path,
    opts: &RepoMapOptions,
    fingerprint: &[(PathBuf, u128)],
    outline: &Outline,
) -> Result<()> {
    let Some(path) = cache_path(root) else {
        return Ok(());
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let entry = CacheEntry {
        root: root.to_path_buf(),
        max_files: opts.max_files,
        max_symbols_per_file: opts.max_symbols_per_file,
        fingerprint: fingerprint.to_vec(),
        outline: outline.clone(),
    };
    let bytes = serde_json::to_vec(&entry)?;
    fs::write(path, bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_language_basic() {
        assert_eq!(detect_language(Path::new("a.rs")), "rust");
        assert_eq!(detect_language(Path::new("a.py")), "python");
        assert_eq!(detect_language(Path::new("a.ts")), "typescript");
        assert_eq!(detect_language(Path::new("a.txt")), "text");
    }
}
