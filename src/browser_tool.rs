//! Headless-browser tool family backed by `chromiumoxide`.
//!
//! Gated behind the `browser` cargo feature so the default binary
//! stays under ~8 MB; chromiumoxide pulls in a large CDP-generated
//! crate. Each open browser is keyed by a caller-supplied opaque
//! session id (model-generated UUID, mirroring the long-lived REPL
//! tools in `builtin_tools.rs`). The browser process is owned by the
//! session and dies when the session is closed (or when the manager
//! itself is dropped).

#![cfg(feature = "browser")]

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use chromiumoxide::browser::{Browser, BrowserConfig};
use chromiumoxide::cdp::browser_protocol::page::CaptureScreenshotFormat;
use chromiumoxide::page::{Page, ScreenshotParams};
use futures::StreamExt;
use tokio::sync::Mutex as AsyncMutex;
use tokio::time::sleep;

/// One live headless-browser session.
struct BrowserSession {
    browser: Browser,
    page: Page,
    /// Drives chromiumoxide's event loop. Aborted on close so the
    /// browser handle can be dropped cleanly (dropping the handle is
    /// what actually terminates the Chrome process).
    handler_task: tokio::task::JoinHandle<()>,
}

/// Manages a set of long-lived browser sessions, one per opaque
/// session id. Cheap to clone (`Arc` inside).
#[derive(Clone, Default)]
pub struct BrowserManager {
    sessions: Arc<AsyncMutex<HashMap<String, BrowserSession>>>,
}

impl BrowserManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Opens `url` in the session named `session_id`. If a session with
    /// that id already exists, navigates the existing page rather than
    /// launching a new browser. When `wait_for` is provided we poll for
    /// the selector to appear (up to a fixed timeout) before returning.
    pub async fn open(
        &self,
        session_id: &str,
        url: &str,
        wait_for: Option<&str>,
    ) -> Result<String> {
        let mut sessions = self.sessions.lock().await;
        let page = if let Some(existing) = sessions.get(session_id) {
            existing
                .page
                .goto(url)
                .await
                .with_context(|| format!("failed to navigate to {url}"))?;
            existing
                .page
                .wait_for_navigation()
                .await
                .with_context(|| format!("navigation to {url} did not settle"))?;
            existing.page.clone()
        } else {
            let config = BrowserConfig::builder()
                .build()
                .map_err(|e| anyhow!(launch_hint(&e)))?;
            let (browser, mut handler) = Browser::launch(config)
                .await
                .map_err(|e| anyhow!(launch_hint(&e.to_string())))?;
            let handler_task = tokio::spawn(async move {
                // Drive the CDP event loop until the channel closes
                // (which happens when the browser is dropped). We
                // deliberately swallow per-event errors — they're
                // transient by nature (page reloads, frame detaches)
                // and the per-call methods surface real failures.
                while let Some(_event) = handler.next().await {}
            });
            let page = browser
                .new_page(url)
                .await
                .with_context(|| format!("failed to open {url}"))?;
            page.wait_for_navigation()
                .await
                .with_context(|| format!("navigation to {url} did not settle"))?;
            let session = BrowserSession {
                browser,
                page: page.clone(),
                handler_task,
            };
            sessions.insert(session_id.to_string(), session);
            page
        };

        if let Some(selector) = wait_for {
            wait_for_selector(&page, selector, Duration::from_secs(15)).await?;
        }

        let url_now = page.url().await.ok().flatten().unwrap_or_default();
        Ok(format!(
            "opened '{session_id}' at {}",
            if url_now.is_empty() { url } else { &url_now }
        ))
    }

    /// Evaluates a JavaScript expression in the page and returns the
    /// result as JSON. Promises are awaited (chromiumoxide handles the
    /// expression-vs-function fallback internally).
    pub async fn eval(&self, session_id: &str, js: &str) -> Result<serde_json::Value> {
        let sessions = self.sessions.lock().await;
        let session = sessions
            .get(session_id)
            .ok_or_else(|| anyhow!("no browser session '{session_id}'"))?;
        let result = session
            .page
            .evaluate(js)
            .await
            .with_context(|| "evaluate failed".to_string())?;
        // `into_value::<Value>` lets us pass through any JSON-serializable
        // payload (numbers, strings, objects, arrays, null). Non-JSON
        // values (functions, DOM nodes, Symbols) fall back to JSON null
        // rather than failing the tool call.
        let value = result
            .into_value::<serde_json::Value>()
            .unwrap_or(serde_json::Value::Null);
        Ok(value)
    }

    /// Saves a full-page PNG screenshot to `path`. The caller is
    /// expected to have already gone through `Permissions::check_write`.
    pub async fn screenshot(&self, session_id: &str, path: &Path) -> Result<()> {
        let sessions = self.sessions.lock().await;
        let session = sessions
            .get(session_id)
            .ok_or_else(|| anyhow!("no browser session '{session_id}'"))?;
        let params = ScreenshotParams::builder()
            .format(CaptureScreenshotFormat::Png)
            .full_page(true)
            .build();
        session
            .page
            .save_screenshot(params, path)
            .await
            .with_context(|| format!("screenshot to {} failed", path.display()))?;
        Ok(())
    }

    /// Extracts visible text from the page (or from a matched element
    /// when a selector is provided).
    pub async fn text(&self, session_id: &str, selector: Option<&str>) -> Result<String> {
        let sessions = self.sessions.lock().await;
        let session = sessions
            .get(session_id)
            .ok_or_else(|| anyhow!("no browser session '{session_id}'"))?;
        let expr = match selector {
            Some(sel) => {
                let escaped = sel.replace('\\', "\\\\").replace('"', "\\\"");
                format!(
                    "(() => {{ const el = document.querySelector(\"{escaped}\"); \
                     return el ? (el.innerText || el.textContent || '') : null; }})()"
                )
            }
            None => {
                "document.body ? (document.body.innerText || document.body.textContent || '') : ''"
                    .to_string()
            }
        };
        let result = session
            .page
            .evaluate(expr.as_str())
            .await
            .with_context(|| "text extraction failed".to_string())?;
        let value = result
            .into_value::<serde_json::Value>()
            .unwrap_or(serde_json::Value::Null);
        match value {
            serde_json::Value::Null => {
                if selector.is_some() {
                    Err(anyhow!(
                        "selector '{}' did not match any element",
                        selector.unwrap_or("")
                    ))
                } else {
                    Ok(String::new())
                }
            }
            serde_json::Value::String(s) => Ok(s),
            other => Ok(other.to_string()),
        }
    }

    /// Closes a session and terminates the underlying Chrome process.
    /// Idempotent: closing an unknown session is an error so misuse is
    /// surfaced to the model, but the manager state is unaffected.
    pub async fn close(&self, session_id: &str) -> Result<()> {
        let mut sessions = self.sessions.lock().await;
        let Some(mut session) = sessions.remove(session_id) else {
            return Err(anyhow!("no browser session '{session_id}'"));
        };
        // Best-effort orderly close: ask Chrome to shut down, then
        // abort the handler task so the Browser handle can drop.
        let _ = session.browser.close().await;
        let _ = session.browser.wait().await;
        session.handler_task.abort();
        Ok(())
    }
}

/// Polls for `selector` to appear, returning when the first match is
/// found or when `timeout` elapses (whichever comes first).
async fn wait_for_selector(page: &Page, selector: &str, timeout: Duration) -> Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        if page.find_element(selector).await.is_ok() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(anyhow!(
                "selector '{selector}' did not appear within {}s",
                timeout.as_secs()
            ));
        }
        sleep(Duration::from_millis(100)).await;
    }
}

/// Wraps a chromiumoxide launch failure with an install hint so the
/// model can suggest a remedy rather than retrying blindly.
fn launch_hint(inner: &str) -> String {
    format!(
        "browser launch failed: {inner}. Install Chromium/Chrome and ensure it's on PATH, \
         or set the CHROME env var to its binary path."
    )
}

/// Lightweight self-check used by `cubi doctor`: launches a headless
/// browser, immediately closes it, and reports success. Bounded by a
/// short overall timeout so a hung Chrome can't wedge the doctor.
pub async fn doctor_probe(timeout_secs: u64) -> Result<()> {
    let fut = async {
        let config = BrowserConfig::builder()
            .build()
            .map_err(|e| anyhow!(launch_hint(&e)))?;
        let (mut browser, mut handler) = Browser::launch(config)
            .await
            .map_err(|e| anyhow!(launch_hint(&e.to_string())))?;
        let handle = tokio::spawn(async move { while let Some(_event) = handler.next().await {} });
        let _ = browser.close().await;
        let _ = browser.wait().await;
        handle.abort();
        Ok::<(), anyhow::Error>(())
    };
    match tokio::time::timeout(Duration::from_secs(timeout_secs), fut).await {
        Ok(res) => res,
        Err(_) => Err(anyhow!(
            "browser launch probe timed out after {timeout_secs}s"
        )),
    }
}
