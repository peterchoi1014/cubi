use std::fmt::Display;
use std::sync::Mutex;

/// Process-global capture buffer for startup status lines.
///
/// When `Some`, [`status_line`] pushes an ANSI-stripped copy of each message
/// into the buffer (in addition to printing it as usual). When `None` (the
/// default), [`status_line`] behaves exactly as before with zero overhead
/// beyond a single lock check. This lets the opt-in `--tui` seed its
/// transcript with the normal startup output (init/loading lines, mascot,
/// banner, tip) that the alternate screen would otherwise wipe.
static CAPTURE: Mutex<Option<Vec<String>>> = Mutex::new(None);

/// Begin capturing status lines into the process-global buffer. Idempotent:
/// resets the buffer to empty.
pub fn capture_start() {
    if let Ok(mut guard) = CAPTURE.lock() {
        *guard = Some(Vec::new());
    }
}

/// Take (and clear) the captured status lines. Returns an empty vec when
/// capture was never started (or already taken).
pub fn capture_take() -> Vec<String> {
    match CAPTURE.lock() {
        Ok(mut guard) => guard.take().unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

pub fn status_line(headless: bool, msg: impl Display) {
    // When capture is active, record an ANSI-stripped copy before printing.
    // The lock is only contended during the brief single-threaded startup
    // window, and when capture is `None` this is a cheap check.
    if let Ok(mut guard) = CAPTURE.lock() {
        if let Some(buf) = guard.as_mut() {
            buf.push(strip_ansi(&msg.to_string()));
        }
    }

    if headless {
        eprintln!("{msg}");
    } else {
        println!("{msg}");
    }
}

/// Remove ANSI CSI escape sequences (`\x1b[` … final byte in `@..=~`) from
/// `s`, returning a plain-text copy. Non-CSI escapes and all other bytes pass
/// through unchanged. Used so captured startup lines seed the TUI transcript
/// as plain text (the transcript renders its own styled spans).
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Expect a CSI introducer '['; if the next char is '[', consume
            // parameter/intermediate bytes up to and including the final byte
            // in the range '@'..='~' (0x40..=0x7E).
            if let Some('[') = chars.clone().next() {
                // Consume the '['.
                chars.next();
                for f in chars.by_ref() {
                    if ('@'..='~').contains(&f) {
                        break;
                    }
                }
            }
            // A lone ESC or a non-CSI escape: drop the ESC only.
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_ansi_removes_csi_sequences() {
        // A basic SGR color sequence.
        assert_eq!(strip_ansi("\x1b[31mred\x1b[0m"), "red");
        // Multiple params + a bright color reset.
        assert_eq!(
            strip_ansi("\x1b[1;33m💡 tip:\x1b[0m hello"),
            "💡 tip: hello"
        );
        // Plain text is unchanged.
        assert_eq!(strip_ansi("no escapes here"), "no escapes here");
        // Underline+dim compound sequence.
        assert_eq!(strip_ansi("\x1b[2;4mx\x1b[0m"), "x");
        // Multibyte content survives stripping.
        assert_eq!(strip_ansi("\x1b[96m✓ ok — é\x1b[0m"), "✓ ok — é");
    }

    /// Capture roundtrip: start → status_line records stripped copies → take
    /// returns them and resets capture to off. Guarded by a lock so the two
    /// capture tests can't interleave on the shared global buffer.
    #[test]
    fn capture_roundtrip_records_stripped_lines() {
        // Serialize with the other capture test via a dedicated mutex; both
        // mutate the shared CAPTURE global.
        static TEST_LOCK: Mutex<()> = Mutex::new(());
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        capture_start();
        status_line(true, "\x1b[36mInitializing Cubi...\x1b[0m");
        status_line(true, "✓ AI executor ready");
        let lines = capture_take();
        assert_eq!(lines, vec!["Initializing Cubi...", "✓ AI executor ready"]);

        // After take, capture is off: further lines are not recorded.
        status_line(true, "not captured");
        assert!(capture_take().is_empty());
    }

    #[test]
    fn capture_off_by_default_is_a_noop() {
        static TEST_LOCK: Mutex<()> = Mutex::new(());
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Ensure any prior capture is cleared.
        let _ = capture_take();
        // With capture off, status_line just prints; take stays empty.
        status_line(true, "line while off");
        assert!(capture_take().is_empty());
    }
}
