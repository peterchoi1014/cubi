//! Tool-call spinner. Surfaces a braille-frame spinner with elapsed-time
//! readout while a tool call is in flight, so the user can tell that the
//! agent is making progress on something slow (LSP, MCP server, shell
//! command) instead of staring at a frozen cursor.
//!
//! The spinner is suppressed entirely when:
//!
//! * stderr is not a TTY (CI logs, piped runs)
//! * `NO_COLOR` or `CUBI_NO_COLOR` is set
//! * `CUBI_NO_SPINNER` is set
//! * the caller is in JSON / headless mode (must be passed in explicitly
//!   via [`ToolSpinner::start_with_mode`]; the module deliberately doesn't
//!   read CLI state to stay decoupled)
//!
//! The frame formatter is a pure function ([`format_frame`]) so tests
//! don't need a TTY.

use std::io::{IsTerminal, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const FRAME_INTERVAL: Duration = Duration::from_millis(100);
const GRACE_PERIOD: Duration = Duration::from_millis(400);

/// Returns the line a single repaint should emit for `(idx, name, elapsed)`.
///
/// Shape: `\r⠋ {name} • {elapsed:.1}s\x1b[K` (trailing escape clears the
/// rest of the line so a shrinking elapsed value can't leave garbage).
pub fn format_frame(idx: usize, name: &str, elapsed: Duration) -> String {
    let frame = FRAMES[idx % FRAMES.len()];
    format!("\r{} {} • {:.1}s\x1b[K", frame, name, elapsed.as_secs_f64())
}

/// Returns true when a spinner should be suppressed because the
/// environment doesn't want decorative output. Pure: reads env once per
/// call so unit tests can `std::env::set_var` in a serialised section.
fn env_suppresses_spinner() -> bool {
    std::env::var("NO_COLOR").is_ok()
        || std::env::var("CUBI_NO_COLOR").is_ok()
        || std::env::var("CUBI_NO_SPINNER").is_ok()
}

/// Handle to a running tool spinner. Drop or call [`ToolSpinner::finish`]
/// to clear the line and join the painter task.
pub struct ToolSpinner {
    stop: Arc<AtomicBool>,
    handle: Option<tokio::task::JoinHandle<()>>,
    active: bool,
}

impl ToolSpinner {
    /// Convenience for the common REPL case (TTY check, env check).
    #[allow(dead_code)]
    pub fn start(tool_name: impl Into<String>) -> Self {
        Self::start_with_mode(tool_name, false)
    }

    /// Starts a spinner unless suppressed. `json_or_headless=true` forces
    /// a no-op even when the rest of the environment is permissive, so
    /// JSON event streams stay machine-parsable.
    pub fn start_with_mode(tool_name: impl Into<String>, json_or_headless: bool) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stderr_tty = std::io::stderr().is_terminal();
        let suppress = json_or_headless || !stderr_tty || env_suppresses_spinner();
        if suppress {
            return Self {
                stop,
                handle: None,
                active: false,
            };
        }

        let name = tool_name.into();
        let s = stop.clone();
        let handle = tokio::spawn(async move {
            // Grace period: don't draw anything for the first ~400ms so
            // fast tools (cheap built-ins, cached lookups) never flash
            // the spinner. The flag is checked after every sleep so an
            // early `finish()` exits without painting at all.
            tokio::time::sleep(GRACE_PERIOD).await;
            if s.load(Ordering::SeqCst) {
                return;
            }
            let start = Instant::now();
            let mut idx = 0usize;
            while !s.load(Ordering::SeqCst) {
                let line = format_frame(idx, &name, start.elapsed());
                eprint!("{}", line);
                let _ = std::io::stderr().flush();
                tokio::time::sleep(FRAME_INTERVAL).await;
                idx = idx.wrapping_add(1);
            }
        });

        Self {
            stop,
            handle: Some(handle),
            active: true,
        }
    }

    /// Stops the painter, clears the spinner line, and joins the task.
    pub async fn finish(mut self) {
        self.stop_inner();
        if let Some(h) = self.handle.take() {
            let _ = h.await;
        }
        if self.active {
            eprint!("\r\x1b[K");
            let _ = std::io::stderr().flush();
        }
    }

    fn stop_inner(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }
}

impl Drop for ToolSpinner {
    fn drop(&mut self) {
        // Best-effort: if the caller forgot to await `finish`, at least
        // tell the painter task to exit. We can't await here, so the
        // join is left to the runtime to reap. Also wipe the line so
        // a forgotten spinner doesn't bleed into subsequent stderr.
        self.stop_inner();
        if self.active {
            eprint!("\r\x1b[K");
            let _ = std::io::stderr().flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_frame_shape() {
        let s = format_frame(0, "shell.run", Duration::from_millis(1234));
        assert!(s.starts_with('\r'));
        assert!(s.contains("shell.run"));
        assert!(s.contains("1.2s"));
        assert!(s.ends_with("\x1b[K"));
        assert!(s.contains('⠋'));
    }

    #[test]
    fn format_frame_cycles_frames() {
        let a = format_frame(0, "x", Duration::ZERO);
        let b = format_frame(1, "x", Duration::ZERO);
        let wrap = format_frame(FRAMES.len(), "x", Duration::ZERO);
        assert_ne!(a, b);
        assert_eq!(a, wrap);
    }

    #[test]
    fn format_frame_elapsed_rounded_to_tenths() {
        let s = format_frame(0, "n", Duration::from_millis(2050));
        // 2.05 -> "2.0s" or "2.1s" depending on float repr; just assert
        // a one-decimal format like "{:.1}s".
        assert!(s.contains("2.0s") || s.contains("2.1s"));
    }

    #[tokio::test]
    async fn json_mode_disables_spinner() {
        let sp = ToolSpinner::start_with_mode("x", true);
        assert!(!sp.active);
        sp.finish().await;
    }
}
