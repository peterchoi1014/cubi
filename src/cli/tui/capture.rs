//! Cross-platform capture of everything written to the process stdout/stderr
//! (including raw `println!`/`eprintln!` and any child process that inherits
//! the fds) during a synchronous closure.
//!
//! # Why a tempfile, not a pipe
//!
//! [`capture_fds`] redirects file descriptors 1 (stdout) and 2 (stderr) at the
//! OS level to an anonymous temp **file**. A regular file has effectively
//! unbounded capacity, so — unlike a pipe — there is no fixed kernel buffer to
//! fill and therefore no risk of a writer deadlocking against a reader that is
//! not draining. That means no background draining thread is needed and the
//! captured bytes are simply read back off disk once the closure returns.
//!
//! Descriptor 0 (stdin) is pointed at the null device for the duration so any
//! code that tries to read the terminal gets EOF / fails fast instead of
//! blocking on a console that the caller (e.g. the TUI) is no longer driving.
//!
//! # Safety / correctness properties
//!
//! * **Panic-safe.** The original fds are saved with `dup` and restored from an
//!   RAII guard's `Drop`, so even if the closure panics (the crate is not
//!   `panic = "abort"`) the real terminal fds are put back.
//! * **No lost/misordered bytes.** `stdout`/`stderr` are flushed before the
//!   redirect is installed and again before the tempfile is read, so no
//!   buffered bytes straddle the swap. Because fd 1 and fd 2 are `dup2`'d from
//!   the *same* open file description, interleaved stdout/stderr writes share a
//!   single file offset and keep their relative order.
//! * **Color preserved.** A tempfile is not a TTY, so `colored` (and
//!   `is_terminal()` probes) would strip ANSI. The guard forces
//!   `colored::control::set_override(true)` for the duration and restores the
//!   previous state on `Drop`, mirroring the prior art in `src/style.rs`.

// This primitive is wired into command dispatch by a subsequent slice; until
// then the bin build has no caller, so silence the unused-code lint here.
#![allow(dead_code)]

use libc::c_int;
use std::ffi::CString;
use std::io::{self, Write};
use std::path::Path;
use tempfile::NamedTempFile;

#[cfg(windows)]
const O_BINARY: c_int = libc::O_BINARY;
#[cfg(not(windows))]
const O_BINARY: c_int = 0;

#[cfg(windows)]
const NULL_DEV: &str = "NUL";
#[cfg(not(windows))]
const NULL_DEV: &str = "/dev/null";

/// Redirect the process stdout + stderr to a tempfile (and stdin from the null
/// device) for the duration of `f`, returning `f`'s value alongside everything
/// that was written to fd 1 / fd 2. Cross-platform (Unix + Windows CRT fds).
///
/// The fds — and the `colored` override — are restored even if `f` panics. If
/// the redirect cannot be installed (e.g. the tempfile cannot be created) `f`
/// still runs, and the returned capture string is empty.
pub(crate) fn capture_fds<R>(f: impl FnOnce() -> R) -> (R, String) {
    match Redirect::install() {
        Ok(redir) => {
            let r = f();
            let captured = redir.read_captured();
            // Explicitly restore fds/color before returning so the caller's
            // next write goes to the real terminal.
            drop(redir);
            (r, captured)
        }
        // Best-effort degradation: run uncaptured rather than losing the call.
        Err(_) => (f(), String::new()),
    }
}

/// RAII guard that owns the redirect. Installing it saves the original fd
/// 0/1/2 with `dup` and points them at the null device / tempfile; dropping it
/// flushes, restores the originals with `dup2`, closes every fd it opened, and
/// restores the previous `colored` override.
struct Redirect {
    /// `dup`'d copies of the original fd 0, 1, 2 (in that order). `-1` marks a
    /// slot that failed to save (and therefore must not be restored/closed).
    saved: [c_int; 3],
    /// fd into the tempfile that fd 1 / fd 2 are redirected to.
    write_fd: c_int,
    /// fd into the null device that fd 0 is redirected from.
    null_fd: c_int,
    /// Backing tempfile; dropped (and deleted) with the guard.
    tmp: NamedTempFile,
    /// `colored` override state to restore on drop.
    prev_color: bool,
}

impl Redirect {
    fn install() -> io::Result<Self> {
        let prev_color = colored::control::SHOULD_COLORIZE.should_colorize();

        let tmp = NamedTempFile::new()?;
        let write_fd = open_fd(tmp.path(), libc::O_RDWR | libc::O_CREAT | O_BINARY, 0o600)?;
        let null_fd = match open_fd(Path::new(NULL_DEV), libc::O_RDWR | O_BINARY, 0) {
            Ok(fd) => fd,
            Err(e) => {
                unsafe { libc::close(write_fd) };
                return Err(e);
            }
        };

        // Flush any buffered bytes to the *real* terminal before swapping.
        let _ = io::stdout().flush();
        let _ = io::stderr().flush();

        // Save the originals. If any save fails, the guard's Drop closes the
        // fds we opened and restores whatever it can.
        let saved = unsafe { [libc::dup(0), libc::dup(1), libc::dup(2)] };
        let guard = Redirect {
            saved,
            write_fd,
            null_fd,
            tmp,
            prev_color,
        };
        if saved.iter().any(|&fd| fd < 0) {
            return Err(io::Error::last_os_error());
        }

        // Install the redirect. On any failure the guard's Drop restores the
        // originals from `saved`.
        let ok = unsafe {
            libc::dup2(null_fd, 0) >= 0
                && libc::dup2(write_fd, 1) >= 0
                && libc::dup2(write_fd, 2) >= 0
        };
        if !ok {
            return Err(io::Error::last_os_error());
        }

        // Keep ANSI color in the captured bytes even though the target is not a
        // TTY. Restored to `prev_color` on Drop.
        colored::control::set_override(true);

        Ok(guard)
    }

    /// Flush the std streams and read everything written to the tempfile so
    /// far. Reads through `write_fd` itself (fd 1/2 share its file
    /// description), so no additional handle to the file is opened.
    fn read_captured(&self) -> String {
        let _ = io::stdout().flush();
        let _ = io::stderr().flush();
        let bytes = unsafe {
            libc::lseek(self.write_fd, 0, libc::SEEK_SET);
            read_all_from_fd(self.write_fd)
        };
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

impl Drop for Redirect {
    fn drop(&mut self) {
        // Flush anything buffered before we hand the fds back.
        let _ = io::stdout().flush();
        let _ = io::stderr().flush();
        unsafe {
            // Restore the originals, then close the saved copies.
            if self.saved[0] >= 0 {
                libc::dup2(self.saved[0], 0);
                libc::close(self.saved[0]);
            }
            if self.saved[1] >= 0 {
                libc::dup2(self.saved[1], 1);
                libc::close(self.saved[1]);
            }
            if self.saved[2] >= 0 {
                libc::dup2(self.saved[2], 2);
                libc::close(self.saved[2]);
            }
            libc::close(self.write_fd);
            libc::close(self.null_fd);
        }
        colored::control::set_override(self.prev_color);
        // `self.tmp` is dropped here, deleting the backing file.
    }
}

/// Open `path` with `flags`/`mode` and return the raw CRT fd. On Windows this
/// resolves to `_open`, which yields the same fd namespace that `println!`
/// (the CRT stdout) writes to.
fn open_fd(path: &Path, flags: c_int, mode: c_int) -> io::Result<c_int> {
    let cpath = path_to_cstring(path)?;
    let fd = unsafe { libc::open(cpath.as_ptr(), flags, mode) };
    if fd < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(fd)
    }
}

/// Read all remaining bytes from `fd` (which has been rewound to the start).
///
/// # Safety
/// `fd` must be a valid, readable file descriptor.
unsafe fn read_all_from_fd(fd: c_int) -> Vec<u8> {
    let mut out = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        #[cfg(unix)]
        let n = unsafe { libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
        #[cfg(windows)]
        let n = unsafe { libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len() as u32) };
        if n <= 0 {
            break;
        }
        out.extend_from_slice(&buf[..n as usize]);
    }
    out
}

#[cfg(unix)]
fn path_to_cstring(p: &Path) -> io::Result<CString> {
    use std::os::unix::ffi::OsStrExt;
    CString::new(p.as_os_str().as_bytes())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))
}

#[cfg(windows)]
fn path_to_cstring(p: &Path) -> io::Result<CString> {
    let s = p
        .to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "non-UTF-8 temp path"))?;
    CString::new(s).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // These tests mutate *process-global* file descriptors, so they must never
    // run concurrently with one another. Serialize them behind a mutex.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    // The libtest harness intercepts the `print!`/`eprint!` *macros* (on every
    // thread) via `std::io::set_output_capture`, so under `cargo test` those
    // macros never reach fd 1/2. Writing to `std::io::stdout()`/`stderr()`
    // directly bypasses that macro-level capture and exercises the *exact* fd
    // path `println!`/`eprintln!` take in production (where there is no libtest
    // capture) — which is precisely what `capture_fds` redirects. See also the
    // `captures_child_process_inherited_stdio` test, which proves capture of a
    // real external program's stdout through the inherited fd.
    fn emit_stdout(msg: &str) {
        let mut so = io::stdout();
        writeln!(so, "{msg}").unwrap();
        so.flush().unwrap();
    }

    fn emit_stderr(msg: &str) {
        let mut se = io::stderr();
        writeln!(se, "{msg}").unwrap();
        se.flush().unwrap();
    }

    #[test]
    fn captures_stdout_and_stderr_and_passes_return_value() {
        let _g = lock();
        let (ret, out) = capture_fds(|| {
            emit_stdout("hello-from-stdout");
            emit_stderr("hello-from-stderr");
            123
        });
        assert_eq!(ret, 123, "closure return value must pass through");
        assert!(
            out.contains("hello-from-stdout"),
            "stdout not captured: {out:?}"
        );
        assert!(
            out.contains("hello-from-stderr"),
            "stderr not captured: {out:?}"
        );
    }

    #[test]
    fn fds_restored_between_captures() {
        let _g = lock();
        let (_, first) = capture_fds(|| emit_stdout("first-message"));
        // After the first capture the fds are restored; the second capture must
        // therefore only see its own output, not the first's.
        let (_, second) = capture_fds(|| emit_stdout("second-message"));
        assert!(first.contains("first-message"), "first: {first:?}");
        assert!(second.contains("second-message"), "second: {second:?}");
        assert!(
            !second.contains("first-message"),
            "fds not restored — second capture leaked first: {second:?}"
        );
    }

    #[test]
    fn panic_in_closure_still_restores_fds() {
        let _g = lock();
        let result = std::panic::catch_unwind(|| {
            capture_fds(|| {
                emit_stdout("before-panic");
                panic!("boom");
            })
        });
        assert!(result.is_err(), "panic should propagate out of capture_fds");

        // fd 1/2 must be usable again: a following capture works normally.
        let (_, after) = capture_fds(|| emit_stdout("after-panic"));
        assert!(
            after.contains("after-panic"),
            "fds not restored after panic: {after:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn captures_child_process_inherited_stdio() {
        let _g = lock();
        let (status, out) = capture_fds(|| {
            std::process::Command::new("echo")
                .arg("child-inherited-out")
                .status()
                .expect("spawn echo")
        });
        assert!(status.success(), "child exited non-zero");
        assert!(
            out.contains("child-inherited-out"),
            "child stdout (inherited fd) not captured: {out:?}"
        );
    }

    #[cfg(windows)]
    #[test]
    fn captures_child_process_inherited_stdio() {
        let _g = lock();
        let (status, out) = capture_fds(|| {
            std::process::Command::new("cmd")
                .args(["/C", "echo", "child-inherited-out"])
                .status()
                .expect("spawn cmd echo")
        });
        assert!(status.success(), "child exited non-zero");
        assert!(
            out.contains("child-inherited-out"),
            "child stdout (inherited fd) not captured: {out:?}"
        );
    }

    #[test]
    fn color_override_is_active_inside_window() {
        let _g = lock();
        let (colorized, _) = capture_fds(|| colored::control::SHOULD_COLORIZE.should_colorize());
        assert!(
            colorized,
            "color override must be forced on inside the capture window"
        );
    }
}
