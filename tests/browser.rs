//! Integration test for the headless-browser tool family.
//!
//! Skips gracefully when no Chromium binary is detected so CI hosts
//! without one stay green. When a binary is present the test exercises
//! the full surface against a `data:text/html,...` URL, which removes
//! the need to stand up an in-process HTTP server.

#![cfg(feature = "browser")]

use std::time::Duration;

// The cubi binary crate has no `lib.rs`, so include the module's
// source directly. The optional `chromiumoxide` and `futures` crates
// it depends on are pulled in by the `browser` feature, which gates
// this whole test file via `#![cfg(feature = "browser")]`.
#[path = "../src/browser_tool.rs"]
#[allow(dead_code)]
mod browser_tool;

fn chromium_available() -> bool {
    if std::env::var("CHROME").is_ok() {
        return true;
    }
    // Probe PATH ourselves to avoid pulling in another dev-dependency.
    let path = match std::env::var_os("PATH") {
        Some(p) => p,
        None => return false,
    };
    let bins: &[&str] = if cfg!(windows) {
        &[
            "chromium.exe",
            "chrome.exe",
            "google-chrome.exe",
            "msedge.exe",
        ]
    } else {
        &[
            "chromium",
            "chromium-browser",
            "chrome",
            "google-chrome",
            "google-chrome-stable",
        ]
    };
    for bin in bins {
        for dir in std::env::split_paths(&path) {
            let candidate = dir.join(bin);
            if candidate.is_file() {
                return true;
            }
        }
    }
    std::path::Path::new("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome").exists()
        || std::path::Path::new("/Applications/Chromium.app/Contents/MacOS/Chromium").exists()
}

#[tokio::test(flavor = "multi_thread")]
async fn browser_open_text_eval_close_roundtrip() {
    if !chromium_available() {
        eprintln!("skipping browser integration test: no Chromium/Chrome binary detected");
        return;
    }
    let mgr = browser_tool::BrowserManager::new();
    let url = "data:text/html,<html><head><title>cubi-test</title></head>\
               <body><p id=greet>hello cubi</p></body></html>";
    let session = "test-session";

    let open = tokio::time::timeout(
        Duration::from_secs(30),
        mgr.open(session, url, Some("#greet")),
    )
    .await
    .expect("browser_open timed out")
    .expect("browser_open failed");
    assert!(open.contains(session), "unexpected open output: {open}");

    let text = mgr
        .text(session, Some("#greet"))
        .await
        .expect("browser_text failed");
    assert!(
        text.contains("hello cubi"),
        "expected greeting in text, got: {text:?}"
    );

    let value = mgr
        .eval(session, "document.title")
        .await
        .expect("browser_eval failed");
    assert_eq!(value, serde_json::json!("cubi-test"));

    mgr.close(session).await.expect("browser_close failed");
}

#[tokio::test]
async fn browser_close_unknown_session_is_idempotent() {
    let mgr = browser_tool::BrowserManager::new();
    mgr.close("never-opened")
        .await
        .expect("close on unknown session should be a no-op");
    // Twice, for good measure.
    mgr.close("never-opened")
        .await
        .expect("close should remain idempotent");
}
