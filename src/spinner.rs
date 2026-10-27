//! Lightweight terminal spinner used to signal that the model is thinking
//! or that the agent loop is between tool steps.
//!
//! Writes braille frames to stderr with carriage-return overwrites so the
//! streaming token output on stdout stays clean. A shared `AtomicBool`
//! lets the caller stop the spinner synchronously from the streaming
//! callback (which is not allowed to `await`); the background task
//! observes the flag, clears its line, and exits.
//!
//! No-ops when stderr is not a TTY (CI, piped output) or no Tokio runtime
//! is available, so synchronous callers and logs stay quiet.

use crate::style::CubiStyle;
use std::io::{IsTerminal, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

const FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const FRAME_MS: u64 = 80;

/// Handle to a running spinner. Drop or call [`Spinner::stop`] to retire it.
pub struct Spinner {
    stop: Arc<AtomicBool>,
    cleared: Arc<AtomicBool>,
    handle: Option<tokio::task::JoinHandle<()>>,
    active: bool,
}

impl Spinner {
    /// Starts a spinner with `label` (e.g. "thinking"). If stderr is not a
    /// TTY the spinner is a no-op so CI / piped runs don't get garbled.
    /// Also a no-op when `CUBI_NO_SPINNER` is set (used by `--quiet`).
    /// Without a current Tokio runtime, returns an inactive handle.
    pub fn start(label: impl Into<String>) -> Self {
        let suppress_env = std::env::var("CUBI_NO_SPINNER").is_ok();
        let active = !suppress_env && std::io::stderr().is_terminal();
        Self::start_inner(label, active)
    }

    fn start_inner(label: impl Into<String>, active: bool) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let cleared = Arc::new(AtomicBool::new(false));
        let runtime = match tokio::runtime::Handle::try_current() {
            Ok(runtime) if active => runtime,
            _ => {
                return Self {
                    stop,
                    cleared,
                    handle: None,
                    active: false,
                };
            }
        };
        let label = label.into();
        let s = stop.clone();
        let c = cleared.clone();
        let handle = runtime.spawn(async move {
            let mut i = 0usize;
            while !s.load(Ordering::SeqCst) {
                if c.load(Ordering::SeqCst) {
                    // Caller has already taken over the line (e.g. began
                    // streaming output). Park until asked to stop so we
                    // don't draw any more frames over their text.
                    tokio::time::sleep(Duration::from_millis(FRAME_MS)).await;
                    continue;
                }
                let frame = FRAMES[i % FRAMES.len()];
                // `\r` returns to col 0; `\x1b[2K` clears the line so an
                // earlier longer label can't leave trailing characters.
                eprint!("\r\x1b[2K{} {}", frame.bright_cyan(), label.bright_black());
                let _ = std::io::stderr().flush();
                tokio::time::sleep(Duration::from_millis(FRAME_MS)).await;
                i = i.wrapping_add(1);
            }
            // Only do a final wipe if the caller hasn't already cleared
            // the line. Otherwise we'd race with their stdout writes and
            // potentially scrub the first line of real output.
            if !c.load(Ordering::SeqCst) {
                eprint!("\r\x1b[2K");
                let _ = std::io::stderr().flush();
            }
        });
        Self {
            stop,
            cleared,
            handle: Some(handle),
            active: true,
        }
    }

    /// Returns the shared flag the caller can flip from a non-async
    /// context (e.g. a streaming-token callback) to ask the spinner to
    /// stop. The background task will exit on its next tick without
    /// touching the terminal — pair this with [`Spinner::clear_line`]
    /// from the same callback to wipe the status line atomically.
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        self.stop.clone()
    }

    /// Synchronously wipes the spinner's line right now and marks the
    /// line as "owned" by the caller, so the background task will neither
    /// draw new frames nor perform a final wipe (which would otherwise
    /// race with the caller's stdout writes).
    pub fn clear_line(&self) {
        if self.active {
            eprint!("\r\x1b[2K");
            let _ = std::io::stderr().flush();
            self.cleared.store(true, Ordering::SeqCst);
        }
    }

    /// Signals the background task to stop and awaits its termination so
    /// the terminal is in a clean state before the caller prints more.
    pub async fn stop(mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.await;
        }
    }
}

impl Drop for Spinner {
    fn drop(&mut self) {
        // Best-effort: if the caller forgot to `.stop().await`, at least
        // signal the task to exit so we don't leave a spinner running.
        self.stop.store(true, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enabled_spinner_without_runtime_is_inert() {
        assert!(tokio::runtime::Handle::try_current().is_err());

        let spinner = Spinner::start_inner("thinking", true);

        assert!(!spinner.active);
        assert!(spinner.handle.is_none());
        spinner.clear_line();
        assert!(!spinner.cleared.load(Ordering::SeqCst));
        let stop = spinner.stop_flag();
        drop(spinner);
        assert!(stop.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn enabled_spinner_uses_current_runtime() {
        let mut spinner = Spinner::start_inner("thinking", true);
        assert!(spinner.active);
        let handle = spinner.handle.take().expect("spinner task");

        // The current-thread runtime cannot poll the painter before stop
        // is signalled, so joining it needs neither a frame delay nor a race.
        spinner.stop().await;

        tokio::time::timeout(Duration::from_secs(5), handle)
            .await
            .expect("spinner task stopped")
            .expect("spinner task completed without panicking");
    }

    #[tokio::test]
    async fn disabled_spinner_with_runtime_is_inert() {
        let spinner = Spinner::start_inner("thinking", false);

        assert!(!spinner.active);
        assert!(spinner.handle.is_none());
        spinner.stop().await;
    }
}
