//! Terminal setup / teardown for the Phase 2 TUI.
//!
//! Restore is **idempotent** and driven directly against `stdout()` — never
//! through the `Terminal` object, which is owned by the render task. Both a
//! [`TerminalGuard`]'s `Drop` and the installed panic hook can fire, so the
//! actual escape-sequence emission is gated by a shared [`AtomicBool`] via
//! [`run_once`]. This is belt-and-suspenders: on a normal return `Drop`
//! restores; on a panic the hook restores *first* (so the backtrace lands on a
//! clean screen) and the subsequent unwinding `Drop` becomes a no-op.

use crossterm::cursor::Show;
use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use std::io::{self, Stdout, Write, stdout};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

// Mouse capture (crossterm `EnableMouseCapture`/`DisableMouseCapture`, i.e. the
// xterm DECSET modes ?1000/?1002/?1003 + SGR ?1006) is enabled while the
// alternate screen is active so the mouse WHEEL is delivered as distinct
// `Event::Mouse(MouseEventKind::ScrollUp/ScrollDown)` events. The event loop
// maps those to transcript scrolling, while the keyboard arrow keys stay as
// `KeyCode::Up/Down` and keep driving input-history recall. We deliberately do
// NOT enable alternate-scroll (?1007): with mouse capture on it is redundant
// and would translate the wheel into arrow keys again, re-triggering the
// history-recall bug. The tradeoff is that native click-drag text selection
// now requires holding Shift/Option.

/// Run `restore` at most once, guarded by `done`. Returns `true` iff this call
/// actually executed the closure (i.e. it had not run before). Kept generic
/// over the closure so the idempotence contract is unit-testable with a plain
/// counter, independent of any real terminal.
pub(super) fn run_once(done: &AtomicBool, restore: impl FnOnce()) -> bool {
    if done.swap(true, Ordering::SeqCst) {
        return false;
    }
    restore();
    true
}

/// Emit the real terminal-restore escape sequences against `stdout()`. Every
/// step is best-effort: teardown must never itself panic or short-circuit.
fn real_restore() {
    let _ = disable_raw_mode();
    let mut out = stdout();
    // Disable mouse capture BEFORE leaving the alternate screen so the user's
    // normal terminal is never left in mouse-capture mode.
    let _ = execute!(out, DisableMouseCapture, LeaveAlternateScreen, Show);
}

/// Owns the "we entered raw mode + alt screen" state for the lifetime of a
/// `run_tui` call. Restoring is idempotent and shared with the panic hook.
pub(super) struct TerminalGuard {
    done: Arc<AtomicBool>,
}

impl TerminalGuard {
    /// Enter raw mode + the alternate screen. On error the partial state is
    /// rolled back before returning so we never leave the terminal wedged.
    pub(super) fn new() -> io::Result<Self> {
        enable_raw_mode()?;
        if let Err(e) = execute!(stdout(), EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(e);
        }
        // Enable mouse capture so the wheel arrives as `Event::Mouse` scroll
        // events (mapped to transcript scrolling) while keyboard arrows keep
        // driving history recall. Best-effort: terminals that don't support it
        // simply ignore the request.
        let _ = execute!(stdout(), EnableMouseCapture);
        let _ = stdout().flush();
        Ok(Self {
            done: Arc::new(AtomicBool::new(false)),
        })
    }

    /// A clone of the shared "already restored" flag, handed to the panic hook
    /// so the hook and this guard's `Drop` cooperate through one gate.
    pub(super) fn done_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.done)
    }

    /// Restore the terminal now (idempotent). Safe to call explicitly before
    /// `Drop` on the normal exit path.
    pub(super) fn restore(&self) {
        run_once(&self.done, real_restore);
    }

    /// Momentarily leave the TUI (leave the alternate screen + disable raw
    /// mode) so a slash-command handler can run on the *real* terminal exactly
    /// as it does in the standard REPL. Routed through the shared `done` gate
    /// via [`run_once`] so a concurrent `Drop`/panic-hook restore stays a
    /// no-op. Paired with [`re_enter`](Self::re_enter), which resets the gate
    /// so the eventual final restore still fires.
    pub(super) fn suspend(&self) {
        run_once(&self.done, real_restore);
    }

    /// Re-enter the TUI after a suspended command: re-enable raw mode, re-enter
    /// the alternate screen, and re-enable mouse capture. Crucially this RESETS
    /// the shared `done` flag to `false` so a later `Drop`/panic-hook restore
    /// (or a subsequent `suspend`) still fires — preserving the single-guard /
    /// single-flag invariant across any number of commands.
    pub(super) fn re_enter(&self) -> io::Result<()> {
        enable_raw_mode()?;
        if let Err(e) = execute!(stdout(), EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(e);
        }
        // Re-arm mouse capture so the wheel keeps scrolling after resume.
        let _ = execute!(stdout(), EnableMouseCapture);
        let _ = stdout().flush();
        // Arm the gate again so restore fires on the real exit / next suspend.
        self.done.store(false, Ordering::SeqCst);
        Ok(())
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        self.restore();
    }
}

/// Install a panic hook that restores the terminal **before** chaining to the
/// previous hook, so a panic backtrace prints onto a clean (non-alt) screen.
/// Restore is idempotent via the shared `done` flag, so it is harmless if the
/// guard's `Drop` also fires during unwinding.
pub(super) fn install_panic_hook(done: Arc<AtomicBool>) {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        run_once(&done, real_restore);
        prev(info);
    }));
}

/// Build the `Terminal` the render task owns. Split out so `run_tui` stays
/// readable; the backend targets the process `stdout()`.
pub(super) fn new_terminal()
-> io::Result<ratatui::Terminal<ratatui::backend::CrosstermBackend<Stdout>>> {
    let backend = ratatui::backend::CrosstermBackend::new(stdout());
    ratatui::Terminal::new(backend)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_once_executes_exactly_once() {
        let done = AtomicBool::new(false);
        let mut count = 0u32;
        // First call runs the closure and reports it did.
        assert!(run_once(&done, || count += 1));
        assert_eq!(count, 1);
        // Subsequent calls are inert no-ops.
        assert!(!run_once(&done, || count += 1));
        assert!(!run_once(&done, || count += 1));
        assert_eq!(count, 1);
    }

    #[test]
    fn run_once_is_shared_across_clones() {
        // Model the guard + panic-hook sharing one flag: whichever fires first
        // wins, the other becomes a no-op.
        let done = Arc::new(AtomicBool::new(false));
        let mut restores = 0u32;
        let hook_done = Arc::clone(&done);
        // "Panic hook" restores first.
        assert!(run_once(&hook_done, || restores += 1));
        // "Drop" then runs but must not double-restore.
        assert!(!run_once(&done, || restores += 1));
        assert_eq!(restores, 1);
    }
}
