//! Lightweight terminal spinner used to signal that the model is thinking
//! or that the agent loop is between tool steps.
//!
//! Writes braille frames to stderr with carriage-return overwrites so the
//! streaming token output on stdout stays clean. A shared `AtomicBool`
//! lets the caller stop the spinner synchronously from the streaming
//! callback (which is not allowed to `await`); the background task
//! observes the flag, clears its line, and exits.
//!
//! No-ops when stderr is not a TTY (CI, piped output) so logs stay quiet.

use colored::*;
use std::io::{IsTerminal, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

const FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const FRAME_MS: u64 = 80;

/// Handle to a running spinner. Drop or call [`Spinner::stop`] to retire it.
pub struct Spinner {
    stop: Arc<AtomicBool>,
    handle: Option<tokio::task::JoinHandle<()>>,
    active: bool,
}

impl Spinner {
    /// Starts a spinner with `label` (e.g. "thinking"). If stderr is not a
    /// TTY the spinner is a no-op so CI / piped runs don't get garbled.
    pub fn start(label: impl Into<String>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let active = std::io::stderr().is_terminal();
        if !active {
            return Self {
                stop,
                handle: None,
                active: false,
            };
        }
        let label = label.into();
        let s = stop.clone();
        let handle = tokio::spawn(async move {
            let mut i = 0usize;
            while !s.load(Ordering::SeqCst) {
                let frame = FRAMES[i % FRAMES.len()];
                // `\r` returns to col 0; `\x1b[2K` clears the line so an
                // earlier longer label can't leave trailing characters.
                eprint!("\r\x1b[2K{} {}", frame.bright_cyan(), label.bright_black());
                let _ = std::io::stderr().flush();
                tokio::time::sleep(Duration::from_millis(FRAME_MS)).await;
                i = i.wrapping_add(1);
            }
            // Final wipe so nothing is left on the status line.
            eprint!("\r\x1b[2K");
            let _ = std::io::stderr().flush();
        });
        Self {
            stop,
            handle: Some(handle),
            active: true,
        }
    }

    /// Returns the shared flag the caller can flip from a non-async
    /// context (e.g. a streaming-token callback) to ask the spinner to
    /// stop. The background task will clean up its line on its next tick.
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        self.stop.clone()
    }

    /// Synchronously wipes the spinner's line right now. Use this from
    /// the streaming callback so the first real output token isn't
    /// printed on the same line as a half-drawn frame.
    pub fn clear_line(&self) {
        if self.active {
            eprint!("\r\x1b[2K");
            let _ = std::io::stderr().flush();
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
