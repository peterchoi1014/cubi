//! Minimal LSP client used by the `lsp` built-in tool.
//!
//! Speaks the framed JSON-RPC protocol (Content-Length header + JSON body)
//! that every Language Server Protocol implementation accepts. We don't try
//! to be a full LSP host — there's no document syncing, change tracking,
//! diagnostics subscription, or workspace/configuration support. The model
//! drives one-shot queries:
//!
//! 1. Spawn the user-specified LSP server (`rust-analyzer`, `pyright-langserver
//!    --stdio`, `typescript-language-server --stdio`, etc).
//! 2. Send `initialize` → `initialized` → `textDocument/didOpen` with the
//!    full file content.
//! 3. Send one of `textDocument/hover` / `textDocument/definition` /
//!    `textDocument/references` and capture the response.
//! 4. Send `shutdown` → `exit` and wait briefly for the child to finish.
//!
//! This is enough for "what's at this line?" and "where is this symbol
//! defined?" use cases without trying to maintain a stateful session.

use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{ChildStderr, ChildStdin, ChildStdout, Command};
use tokio::time::timeout;

/// What to ask the LSP server for. The protocol method names are deliberately
/// kept short in the public API; this enum maps them onto the verbose LSP
/// method strings internally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LspAction {
    Hover,
    Definition,
    References,
}

impl LspAction {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "hover" => Some(Self::Hover),
            "definition" | "def" => Some(Self::Definition),
            "references" | "refs" => Some(Self::References),
            _ => None,
        }
    }

    fn method(self) -> &'static str {
        match self {
            Self::Hover => "textDocument/hover",
            Self::Definition => "textDocument/definition",
            Self::References => "textDocument/references",
        }
    }
}

/// Convert a `file://` URI string from the LSP back into a plain path so
/// our renderer can match it against the request's path. Falls back to the
/// raw string if it doesn't look like a `file:` URI. Percent-escapes are
/// decoded so a path containing spaces or other reserved characters is
/// copy-pasteable in the rendered output.
fn uri_to_path(uri: &str) -> String {
    let raw = if let Some(rest) = uri.strip_prefix("file://") {
        // The local file URI form: file:///abs/path. The first `/` after
        // the scheme is the host separator, the rest is the path.
        if let Some(stripped) = rest.strip_prefix("/") {
            format!("/{}", stripped)
        } else {
            rest.to_string()
        }
    } else {
        return uri.to_string();
    };
    percent_decode(&raw)
}

/// Minimal percent-decoder for the path portion of a `file://` URI.
/// Tolerant of malformed `%XX` sequences (passed through verbatim) so a
/// weird LSP response can never make `uri_to_path` panic. Inlined here so
/// `lsp_client` doesn't have to depend on `builtin_tools`.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = (bytes[i + 1] as char).to_digit(16);
            let lo = (bytes[i + 2] as char).to_digit(16);
            if let (Some(h), Some(l)) = (hi, lo) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

/// Render a hover-response `contents` field. LSP spec lets the server
/// return a string, a `{language, value}` object, a `{kind, value}` object,
/// or an array of those. We accept any of them and dump the textual bits.
fn render_hover_contents(contents: &Value) -> String {
    fn one(v: &Value) -> Option<String> {
        if let Some(s) = v.as_str() {
            return Some(s.to_string());
        }
        if let Some(value) = v.get("value").and_then(|x| x.as_str()) {
            return Some(value.to_string());
        }
        None
    }
    if let Some(arr) = contents.as_array() {
        arr.iter()
            .filter_map(one)
            .collect::<Vec<_>>()
            .join("\n---\n")
    } else if let Some(s) = one(contents) {
        s
    } else {
        // Unrecognized shape — fall back to a JSON dump so the user can
        // at least see what came back instead of an empty string.
        serde_json::to_string_pretty(contents).unwrap_or_default()
    }
}

/// Render a `Location` or `Location[]` response (definition / references).
fn render_locations(value: &Value) -> String {
    fn one(v: &Value) -> Option<String> {
        let uri = v.get("uri").and_then(|x| x.as_str())?;
        let range = v.get("range")?;
        let start = range.get("start")?;
        let line = start.get("line").and_then(|x| x.as_u64()).unwrap_or(0);
        let ch = start.get("character").and_then(|x| x.as_u64()).unwrap_or(0);
        // LSP positions are 0-based; humans (and most editors) think
        // 1-based. Convert so the output is copy-pasteable.
        Some(format!("{}:{}:{}", uri_to_path(uri), line + 1, ch + 1))
    }
    if value.is_null() {
        return "(no location returned)".to_string();
    }
    let lines = if let Some(arr) = value.as_array() {
        arr.iter().filter_map(one).collect::<Vec<_>>()
    } else if let Some(s) = one(value) {
        vec![s]
    } else {
        Vec::new()
    };
    if lines.is_empty() {
        // Unrecognized shape — fall back so the user can debug.
        serde_json::to_string_pretty(value).unwrap_or_default()
    } else {
        lines.join("\n")
    }
}

/// Run a single LSP query end-to-end. Returns the rendered result as a
/// human-readable string ready to feed back to the model. Caller-side
/// timeout: the full session is bounded at `total_timeout` seconds (default
/// 30) so a wedged language server can't hang the agent forever.
///
/// `line` and `character` are 1-based (the way editors show them); we
/// convert to LSP's 0-based positions internally.
#[allow(clippy::too_many_arguments)]
pub async fn run_lsp_query(
    server_command: &str,
    server_args: &[String],
    workspace_root: &Path,
    file_path: &Path,
    line: u32,
    character: u32,
    action: LspAction,
    total_timeout_secs: u64,
) -> Result<String> {
    // Read the file ourselves: the LSP needs the text via didOpen, and we
    // want to fail with a clear "file not found" early if the path is bad.
    let language_id = detect_language_id(file_path);
    let text = std::fs::read_to_string(file_path)
        .with_context(|| format!("Failed to read {}", file_path.display()))?;
    let file_uri = path_to_file_uri(file_path);
    let root_uri = path_to_file_uri(workspace_root);

    let fut = async move {
        let mut child = Command::new(server_command)
            .args(server_args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("Failed to spawn LSP server `{server_command}`"))?;
        let mut stdin = child.stdin.take().context("LSP stdin missing")?;
        let stdout = child.stdout.take().context("LSP stdout missing")?;
        let stderr = child.stderr.take().context("LSP stderr missing")?;
        let mut reader = LspReader::new(stdout, stderr);

        let mut next_id: i64 = 1;

        // initialize
        let init_params = json!({
            "processId": std::process::id(),
            "rootUri": root_uri,
            "capabilities": {
                "textDocument": {
                    "hover": { "contentFormat": ["markdown", "plaintext"] },
                    "definition": {},
                    "references": {},
                    "synchronization": {}
                }
            },
            "clientInfo": { "name": "cubi", "version": env!("CARGO_PKG_VERSION") }
        });
        let init_id = next_id;
        next_id += 1;
        write_request(&mut stdin, init_id, "initialize", init_params).await?;
        // Discard until we see init_id's response.
        let _init_resp = reader.read_response_for(init_id).await?;

        // initialized notification
        write_notification(&mut stdin, "initialized", json!({})).await?;

        // didOpen
        write_notification(
            &mut stdin,
            "textDocument/didOpen",
            json!({
                "textDocument": {
                    "uri": file_uri,
                    "languageId": language_id,
                    "version": 1,
                    "text": text,
                }
            }),
        )
        .await?;

        // Build action-specific params. LSP positions are 0-based; the
        // caller supplied 1-based.
        let zero_based_line = line.saturating_sub(1);
        let zero_based_char = character.saturating_sub(1);
        let mut params = json!({
            "textDocument": { "uri": file_uri },
            "position": { "line": zero_based_line, "character": zero_based_char }
        });
        if matches!(action, LspAction::References) {
            params["context"] = json!({ "includeDeclaration": true });
        }
        let query_id = next_id;
        next_id += 1;
        write_request(&mut stdin, query_id, action.method(), params).await?;

        let response = reader.read_response_for(query_id).await?;

        // Best-effort shutdown so the server flushes any state to disk.
        // We don't fail the tool call if this part errors — by now we
        // already have the answer the user actually wanted.
        let shutdown_id = next_id;
        let _ = write_request(&mut stdin, shutdown_id, "shutdown", json!(null)).await;
        let _ = reader.read_response_for(shutdown_id).await;
        let _ = write_notification(&mut stdin, "exit", json!(null)).await;
        let _ = child.wait().await;

        // Format depending on the action.
        let rendered = match action {
            LspAction::Hover => {
                let contents = response.get("contents").cloned().unwrap_or(Value::Null);
                if contents.is_null() {
                    "(no hover information at that position)".to_string()
                } else {
                    render_hover_contents(&contents)
                }
            }
            LspAction::Definition | LspAction::References => render_locations(&response),
        };
        Ok::<String, anyhow::Error>(rendered)
    };

    match timeout(Duration::from_secs(total_timeout_secs), fut).await {
        Ok(r) => r,
        Err(_) => anyhow::bail!(
            "LSP query timed out after {total_timeout_secs}s (server `{server_command}` may be wedged)"
        ),
    }
}

/// Convert an absolute path to a `file://` URI. On non-Unix systems this
/// would need more work; we only target Linux/macOS hosts.
pub fn path_to_file_uri(path: &Path) -> String {
    let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let s = canon.to_string_lossy();
    if s.starts_with('/') {
        format!("file://{}", s)
    } else {
        // Relative path that didn't canonicalize: still emit a URI but
        // mark it clearly so an LSP error message is debuggable.
        format!("file://{}", s)
    }
}

/// Map a file extension to the LSP `languageId` the spec defines. The
/// servers we care about (rust-analyzer, pyright, ts-server) all key on
/// this; getting it wrong usually means the server treats the file as
/// plain text and returns nothing useful.
pub fn detect_language_id(path: &Path) -> String {
    match path.extension().and_then(|e| e.to_str()) {
        Some("rs") => "rust",
        Some("py") => "python",
        Some("ts") => "typescript",
        Some("tsx") => "typescriptreact",
        Some("js") => "javascript",
        Some("jsx") => "javascriptreact",
        Some("go") => "go",
        Some("c") => "c",
        Some("cpp" | "cc" | "cxx") => "cpp",
        Some("h" | "hpp") => "cpp",
        Some("java") => "java",
        Some("rb") => "ruby",
        Some("md") => "markdown",
        Some("json") => "json",
        Some("toml") => "toml",
        Some("sh") => "shellscript",
        _ => "plaintext",
    }
    .to_string()
}

async fn write_request(stdin: &mut ChildStdin, id: i64, method: &str, params: Value) -> Result<()> {
    let msg = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
    write_frame(stdin, &msg).await
}

async fn write_notification(stdin: &mut ChildStdin, method: &str, params: Value) -> Result<()> {
    let msg = json!({ "jsonrpc": "2.0", "method": method, "params": params });
    write_frame(stdin, &msg).await
}

async fn write_frame(stdin: &mut ChildStdin, msg: &Value) -> Result<()> {
    let body = serde_json::to_vec(msg)?;
    let header = format!("Content-Length: {}\r\n\r\n", body.len());
    stdin.write_all(header.as_bytes()).await?;
    stdin.write_all(&body).await?;
    stdin.flush().await?;
    Ok(())
}

/// Frame reader that pulls JSON-RPC messages off the server's stdout and
/// returns the response matching a given request id. Server-initiated
/// notifications (progress, diagnostics, etc.) are silently dropped so we
/// don't surface them to the model as noise.
struct LspReader {
    stdout: ChildStdout,
    /// Buffered raw bytes from the child's stderr. Drained whenever the
    /// child closes or we error out, so the user sees crash info instead
    /// of a generic "no response" message.
    stderr: ChildStderr,
}

impl LspReader {
    fn new(stdout: ChildStdout, stderr: ChildStderr) -> Self {
        Self { stdout, stderr }
    }

    async fn read_response_for(&mut self, id: i64) -> Result<Value> {
        loop {
            let msg = match self.read_one_frame().await {
                Ok(v) => v,
                Err(e) => {
                    // Try to give the user the server's stderr — usually
                    // explains why the connection died.
                    let mut err_buf = Vec::new();
                    let _ = self.stderr.read_to_end(&mut err_buf).await;
                    if err_buf.is_empty() {
                        return Err(e);
                    }
                    anyhow::bail!(
                        "{e}; LSP server stderr:\n{}",
                        String::from_utf8_lossy(&err_buf)
                    );
                }
            };
            if let Some(resp_id) = msg.get("id").and_then(|v| v.as_i64()) {
                if resp_id != id {
                    continue;
                }
                if let Some(err) = msg.get("error") {
                    anyhow::bail!("LSP server returned error: {}", err);
                }
                return Ok(msg.get("result").cloned().unwrap_or(Value::Null));
            }
            // Otherwise (server-initiated request/notification, or an
            // unrelated response) just keep reading.
        }
    }

    async fn read_one_frame(&mut self) -> Result<Value> {
        // Read the header lines until we see the blank-line terminator.
        let mut headers = Vec::new();
        loop {
            let mut byte = [0u8; 1];
            self.stdout
                .read_exact(&mut byte)
                .await
                .context("LSP stdout closed while reading header")?;
            headers.push(byte[0]);
            // Look for the terminating CRLF CRLF.
            if headers.len() >= 4 && &headers[headers.len() - 4..] == b"\r\n\r\n" {
                break;
            }
        }
        let header_str = std::str::from_utf8(&headers)?;
        let content_length = header_str
            .lines()
            .find_map(|l| {
                l.strip_prefix("Content-Length:")
                    .and_then(|v| v.trim().parse::<usize>().ok())
            })
            .context("LSP header missing valid Content-Length")?;
        if content_length > 16 * 1024 * 1024 {
            anyhow::bail!(
                "LSP server sent suspiciously large frame ({} bytes); refusing",
                content_length
            );
        }
        let mut body = vec![0u8; content_length];
        self.stdout
            .read_exact(&mut body)
            .await
            .context("LSP stdout closed while reading body")?;
        let value: Value = serde_json::from_slice(&body)
            .with_context(|| format!("LSP body is not valid JSON ({} bytes)", content_length))?;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lsp_action_parses_aliases() {
        assert_eq!(LspAction::from_str("hover"), Some(LspAction::Hover));
        assert_eq!(LspAction::from_str("def"), Some(LspAction::Definition));
        assert_eq!(
            LspAction::from_str("definition"),
            Some(LspAction::Definition)
        );
        assert_eq!(LspAction::from_str("refs"), Some(LspAction::References));
        assert_eq!(
            LspAction::from_str("references"),
            Some(LspAction::References)
        );
        assert_eq!(LspAction::from_str("nonsense"), None);
    }

    #[test]
    fn detect_language_id_covers_common_extensions() {
        assert_eq!(detect_language_id(Path::new("x.rs")), "rust");
        assert_eq!(detect_language_id(Path::new("x.py")), "python");
        assert_eq!(detect_language_id(Path::new("x.ts")), "typescript");
        assert_eq!(detect_language_id(Path::new("x.tsx")), "typescriptreact");
        assert_eq!(detect_language_id(Path::new("README")), "plaintext");
    }

    #[test]
    fn path_to_file_uri_emits_absolute_form() {
        let cwd = std::env::current_dir().unwrap();
        let uri = path_to_file_uri(&cwd);
        assert!(uri.starts_with("file:///"), "got: {uri}");
    }

    #[test]
    fn uri_to_path_strips_file_scheme() {
        assert_eq!(uri_to_path("file:///tmp/foo"), "/tmp/foo");
        // Non-URI input is returned unchanged.
        assert_eq!(uri_to_path("/tmp/foo"), "/tmp/foo");
    }

    #[test]
    fn uri_to_path_decodes_percent_escapes() {
        assert_eq!(
            uri_to_path("file:///home/user/My%20Project/main.rs"),
            "/home/user/My Project/main.rs"
        );
        // Malformed escapes are left as-is rather than panicking.
        assert_eq!(uri_to_path("file:///a%2"), "/a%2");
    }

    #[test]
    fn render_hover_contents_handles_string_object_and_array() {
        assert_eq!(render_hover_contents(&json!("just text")), "just text");
        assert_eq!(
            render_hover_contents(&json!({ "kind": "markdown", "value": "**bold**" })),
            "**bold**"
        );
        assert_eq!(
            render_hover_contents(&json!({ "language": "rust", "value": "fn main() {}" })),
            "fn main() {}"
        );
        let arr = render_hover_contents(&json!([
            "first",
            { "language": "rust", "value": "fn x() {}" }
        ]));
        assert!(arr.contains("first"));
        assert!(arr.contains("fn x()"));
        assert!(arr.contains("---"));
    }

    #[test]
    fn render_locations_handles_single_object() {
        let loc = json!({
            "uri": "file:///abs/path.rs",
            "range": { "start": { "line": 9, "character": 4 }, "end": { "line": 9, "character": 8 } }
        });
        // 0-based 9:4 -> 1-based 10:5
        assert_eq!(render_locations(&loc), "/abs/path.rs:10:5");
    }

    #[test]
    fn render_locations_handles_array() {
        let arr = json!([
            {
                "uri": "file:///a.rs",
                "range": { "start": { "line": 0, "character": 0 }, "end": { "line": 0, "character": 1 } }
            },
            {
                "uri": "file:///b.rs",
                "range": { "start": { "line": 5, "character": 2 }, "end": { "line": 5, "character": 3 } }
            }
        ]);
        let out = render_locations(&arr);
        assert!(out.contains("/a.rs:1:1"));
        assert!(out.contains("/b.rs:6:3"));
    }

    #[test]
    fn render_locations_handles_null() {
        assert!(render_locations(&Value::Null).contains("no location"));
    }
}
