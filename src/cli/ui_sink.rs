//! The single seam through which all *dynamic, per-turn interactive output*
//! flows: streamed tokens, the buffered final reply, status/info lines, the
//! post-turn usage footer, and the "model is thinking" spinner.
//!
//! Phase 1 goal: introduce this seam **without changing any user-visible
//! behavior**. [`LineSink`] is the default implementation and reproduces
//! today's stdout/stderr output byte-for-byte in every mode (streaming,
//! buffered, headless, JSON, quiet, `NO_COLOR`, `CUBI_NO_SPINNER`, non-TTY).
//! A later phase can implement the same [`UiSink`] trait with a full-screen
//! TUI renderer and swap it in on [`ChatCLI`](super::ChatCLI) without touching
//! the agent loop.
//!
//! Static one-time output (welcome banner, slash-command help) deliberately
//! does NOT flow through here — only the dynamic per-turn surfaces above.

use crate::ollama::ChatStats;
use crate::style::CubiStyle;
use std::io::Write;
use std::sync::atomic::Ordering;

/// A single unit of dynamic interactive output. Retained as a
/// forward-looking, renderer-agnostic vocabulary: a Phase-2 TUI can consume
/// these instead of the imperative trait methods. The [`UiSink`] trait's
/// methods map 1:1 onto these variants.
///
/// The `ToolStarted` / `ToolFinished` variants carry per-tool feedback to a
/// renderer. In `--tui` mode the [`TuiSink`](super::tui::sink::TuiSink)
/// forwards them so the render task can draw a framed tool block (header,
/// output, ✓/✗ status). The behavior-preserving [`LineSink`] ignores them
/// (its trait defaults are no-ops), so the default (non-TUI) output paths are
/// unaffected — tool feedback there is still the `⚙ tool:` status line and the
/// dim result preview printed directly in the agent loop.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum RenderEvent {
    /// One streamed assistant token.
    AssistantToken(String),
    /// The buffered (non-streaming) final reply body.
    AssistantFinal(String),
    /// A status / info line (what `emit_status` prints today).
    Status(String),
    /// A snapshot of the pinned status row's typed state. Consumed by the
    /// Phase 2 TUI render task to repaint the one-line status row; not emitted
    /// by the behavior-preserving [`LineSink`].
    StatusSnapshot(crate::cli::status::StatusState),
    /// The post-turn usage footer.
    UsageFooter {
        stats: ChatStats,
        window: Option<usize>,
    },
    /// The model-thinking spinner started (label differs by step).
    BeginThinking { label: String, continuation: bool },
    /// The model-thinking spinner retired.
    EndThinking,
    /// A tool began executing. In `--tui` mode this opens a framed tool block
    /// (a `⚙ <name>` header) in the transcript.
    ToolStarted { name: String },
    /// A tool finished executing. In `--tui` mode this appends the (capped)
    /// tool `output` beneath the header and a trailing `✓`/`✗` status row with
    /// the elapsed time. `ok` is `false` when the tool reported an error.
    ToolFinished {
        name: String,
        ok: bool,
        output: String,
        elapsed_ms: u64,
    },
}

/// Output-mode flags the sink needs to reproduce current behavior. These can
/// change after [`ChatCLI`](super::ChatCLI) construction (e.g. `run_one_shot`
/// flips `headless` on; slash commands toggle markdown), so the agent loop
/// re-syncs them onto the sink via [`UiSink::reconfigure`] at the start of
/// each turn.
#[derive(Debug, Clone, Copy)]
pub struct SinkFlags {
    /// One-shot / piped mode: progress on stderr, only reply body on stdout.
    pub headless: bool,
    /// Effective JSON mode (`json_enabled && headless_mode`): emit
    /// line-delimited JSON events instead of human text.
    pub json: bool,
    /// Apply in-house markdown polish to the buffered final reply (TTY only).
    pub markdown: bool,
}

/// The seam every dynamic interactive surface flows through.
pub trait UiSink {
    /// Emit one streamed assistant token. On the first token of a step this
    /// also clears the thinking spinner and prints the `AI:` prefix once per
    /// user turn (non-headless).
    fn assistant_token(&mut self, tok: &str);

    /// Render the buffered (non-streaming) final reply body.
    fn assistant_final(&mut self, content: &str);

    /// Print a status / info line (stderr in headless, stdout otherwise).
    fn status(&mut self, msg: &str);

    /// Print the one-line dim usage footer for the just-completed turn.
    fn usage_footer(&mut self, stats: &ChatStats, window: Option<usize>);

    /// Start the "model is thinking" spinner for a step. `continuation` is
    /// `true` for any step after the first in a turn, so the `AI:` prefix is
    /// only printed on the first step (matching the historical
    /// `printed_prefix = step > 0` behavior). Resets the per-step
    /// first-token state.
    fn begin_thinking(&mut self, label: &str, continuation: bool);

    /// Retire the thinking spinner. Returns the underlying [`Spinner`] (if
    /// any) so the async agent loop can `.stop().await` it — keeping this
    /// trait method non-async and therefore dyn-compatible. A non-terminal
    /// renderer returns `None`.
    fn end_thinking(&mut self) -> Option<crate::spinner::Spinner>;

    /// Whether at least one token was streamed in the current step. The agent
    /// loop reads this after a streaming call to decide on trailing
    /// newlines / fallbacks.
    fn got_token(&self) -> bool;

    /// Re-sync the output-mode flags from the owning `ChatCLI`.
    fn reconfigure(&mut self, flags: SinkFlags);

    /// A tool began executing. Default no-op: the behavior-preserving
    /// [`LineSink`] conveys tool activity through `status` lines printed in the
    /// agent loop, so the default (non-TUI) output stays byte-identical. Only
    /// the TUI sink overrides this to open a framed tool block.
    fn tool_started(&mut self, _name: &str) {}

    /// A tool finished executing. Default no-op (see [`tool_started`](Self::tool_started)).
    /// The TUI sink overrides this to append the tool output and a ✓/✗ status
    /// row. `ok` is `false` when the tool reported an error; `output` is the
    /// (already capped) result text and `elapsed_ms` its wall-clock duration.
    fn tool_finished(&mut self, _name: &str, _ok: bool, _output: &str, _elapsed_ms: u64) {}
}

/// The default, behavior-preserving sink: writes exactly what Cubi wrote
/// before this seam existed.
pub struct LineSink {
    headless: bool,
    /// Effective JSON mode (`json_enabled && headless_mode`).
    json: bool,
    markdown: bool,
    /// Whether a token has been emitted in the current step. Reset by
    /// [`begin_thinking`](UiSink::begin_thinking).
    got_token: bool,
    /// Whether the `AI:` prefix has been printed for the current turn.
    printed_prefix: bool,
    /// The active thinking spinner, owned across the streaming call so the
    /// first-token handler can clear it.
    thinking: Option<crate::spinner::Spinner>,
}

impl LineSink {
    pub fn new(flags: SinkFlags) -> Self {
        Self {
            headless: flags.headless,
            json: flags.json,
            markdown: flags.markdown,
            got_token: false,
            printed_prefix: false,
            thinking: None,
        }
    }
}

impl UiSink for LineSink {
    fn assistant_token(&mut self, tok: &str) {
        // First token of the step: clear the spinner (it owns the status
        // line) and print the `AI:` prefix once per user turn. Store the stop
        // flag *before* clearing so the background task observes it and does
        // not race a final wipe over our first line of output.
        if !self.got_token {
            if let Some(sp) = &self.thinking {
                sp.stop_flag().store(true, Ordering::SeqCst);
                sp.clear_line();
            }
            if !self.printed_prefix && !self.headless {
                print!("{} ", "AI:".bright_blue().bold());
                self.printed_prefix = true;
            }
        }
        if self.json {
            crate::json_events::emit(true, &crate::json_events::token(tok));
        } else if self.headless {
            print!("{}", tok);
            let _ = std::io::stdout().flush();
        } else {
            print!("{}", tok.bright_white());
            let _ = std::io::stdout().flush();
        }
        self.got_token = true;
    }

    fn assistant_final(&mut self, content: &str) {
        if self.headless {
            println!("{content}");
            return;
        }
        print!("{} ", "AI:".bright_blue().bold());
        if self.markdown && std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            println!();
            let color = std::env::var("NO_COLOR").is_err();
            let rendered = super::render::polish_markdown(content, color);
            print!("{}", rendered);
        } else {
            println!("{}", content.bright_white());
        }
    }

    fn status(&mut self, msg: &str) {
        // Mirrors `crate::out::status_line`: headless progress goes to stderr
        // so stdout stays a clean, pipeable stream of reply bytes.
        if self.headless {
            eprintln!("{msg}");
        } else {
            println!("{msg}");
        }
    }

    fn usage_footer(&mut self, stats: &ChatStats, window: Option<usize>) {
        super::render::print_stats_footer(stats, window);
    }

    fn begin_thinking(&mut self, label: &str, continuation: bool) {
        self.got_token = false;
        self.printed_prefix = continuation;
        self.thinking = Some(crate::spinner::Spinner::start(label));
    }

    fn end_thinking(&mut self) -> Option<crate::spinner::Spinner> {
        self.thinking.take()
    }

    fn got_token(&self) -> bool {
        self.got_token
    }

    fn reconfigure(&mut self, flags: SinkFlags) {
        self.headless = flags.headless;
        self.json = flags.json;
        self.markdown = flags.markdown;
    }
}

/// A no-op sink used only as a transient placeholder while the real sink is
/// temporarily moved out of `ChatCLI` (so a streaming closure can own it while
/// `self.executor` is borrowed by `chat_stream`). It is never used to render.
pub struct NullSink;

impl UiSink for NullSink {
    fn assistant_token(&mut self, _tok: &str) {}
    fn assistant_final(&mut self, _content: &str) {}
    fn status(&mut self, _msg: &str) {}
    fn usage_footer(&mut self, _stats: &ChatStats, _window: Option<usize>) {}
    fn begin_thinking(&mut self, _label: &str, _continuation: bool) {}
    fn end_thinking(&mut self) -> Option<crate::spinner::Spinner> {
        None
    }
    fn got_token(&self) -> bool {
        false
    }
    fn reconfigure(&mut self, _flags: SinkFlags) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flags(headless: bool, json: bool, markdown: bool) -> SinkFlags {
        SinkFlags {
            headless,
            json,
            markdown,
        }
    }

    #[test]
    fn got_token_tracks_streaming_state() {
        let mut sink = LineSink::new(flags(true, false, false));
        assert!(!sink.got_token());
        sink.assistant_token("hi");
        assert!(sink.got_token());
    }

    #[test]
    fn begin_thinking_resets_token_state_and_sets_prefix() {
        let mut sink = LineSink::new(flags(true, false, false));
        sink.assistant_token("x");
        assert!(sink.got_token());
        // A fresh step (first step of a turn) resets the flag and leaves the
        // prefix un-printed so `assistant_token` will print `AI:`.
        sink.begin_thinking("thinking…", false);
        assert!(!sink.got_token());
        assert!(!sink.printed_prefix);
        // A continuation step marks the prefix as already printed.
        sink.begin_thinking("processing tool results…", true);
        assert!(sink.printed_prefix);
    }

    #[test]
    fn reconfigure_updates_flags() {
        let mut sink = LineSink::new(flags(false, false, false));
        assert!(!sink.headless);
        sink.reconfigure(flags(true, true, true));
        assert!(sink.headless);
        assert!(sink.json);
        assert!(sink.markdown);
    }

    #[test]
    fn null_sink_is_inert() {
        let mut sink = NullSink;
        sink.assistant_token("ignored");
        sink.status("ignored");
        assert!(!sink.got_token());
        assert!(sink.end_thinking().is_none());
    }
}
