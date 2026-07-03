//! The `Send`-by-construction UI sink for the Phase 2 terminal UI.
//!
//! [`TuiSink`] holds *only* an unbounded [`mpsc`](tokio::sync::mpsc) sender of
//! [`RenderEvent`]s plus small `Copy` fields, so it satisfies `ChatCLI`'s
//! `ui: Box<dyn UiSink + Send>` without owning any terminal state. Every
//! [`UiSink`] method translates its call into a [`RenderEvent`] sent to the
//! render task, which owns the `Terminal` and [`AppState`](super::app::AppState).
//!
//! Critically, [`begin_thinking`](TuiSink::begin_thinking) does **not** start a
//! real [`crate::spinner::Spinner`] — the spinner writes to stdout and would
//! corrupt the alternate screen. Instead it sends a
//! [`RenderEvent::BeginThinking`], and the render task shows a thinking
//! indicator itself.

use crate::cli::ui_sink::{RenderEvent, SinkFlags, UiSink};
use crate::ollama::ChatStats;
use tokio::sync::mpsc::UnboundedSender;

/// A UI sink that forwards every interactive surface to the render task as a
/// [`RenderEvent`]. Cheap to clone-free move; contains no terminal handles.
pub struct TuiSink {
    /// Channel to the render task. Send errors (receiver dropped) are ignored.
    tx: UnboundedSender<RenderEvent>,
    /// Whether at least one token was sent in the current step. Reset by
    /// [`begin_thinking`](UiSink::begin_thinking) to mirror `LineSink`.
    got_token: bool,
    /// Output-mode flags, re-synced by [`reconfigure`](UiSink::reconfigure).
    /// Retained for parity with `LineSink` and future TUI use (e.g. a JSON
    /// side-channel); the render task does not consult them today.
    #[allow(dead_code)]
    flags: SinkFlags,
}

impl TuiSink {
    /// Build a sink over `tx`. Flags default to the interactive (non-headless,
    /// non-JSON, markdown-off) profile; the agent loop re-syncs them each turn
    /// via [`reconfigure`](UiSink::reconfigure).
    pub fn new(tx: UnboundedSender<RenderEvent>) -> Self {
        Self {
            tx,
            got_token: false,
            flags: SinkFlags {
                headless: false,
                json: false,
                markdown: false,
            },
        }
    }

    /// Send an event, ignoring the error if the render task has gone away.
    fn send(&self, ev: RenderEvent) {
        let _ = self.tx.send(ev);
    }
}

impl UiSink for TuiSink {
    fn assistant_token(&mut self, tok: &str) {
        self.got_token = true;
        self.send(RenderEvent::AssistantToken(tok.to_string()));
    }

    fn assistant_final(&mut self, content: &str) {
        self.send(RenderEvent::AssistantFinal(content.to_string()));
    }

    fn status(&mut self, msg: &str) {
        self.send(RenderEvent::Status(msg.to_string()));
    }

    fn usage_footer(&mut self, stats: &ChatStats, window: Option<usize>) {
        self.send(RenderEvent::UsageFooter {
            stats: stats.clone(),
            window,
        });
    }

    fn begin_thinking(&mut self, label: &str, continuation: bool) {
        // Reset per-step token state, mirroring `LineSink`. Do NOT start a
        // real `Spinner` — that writes to stdout and would corrupt the alt
        // screen. The render task renders its own thinking indicator.
        self.got_token = false;
        self.send(RenderEvent::BeginThinking {
            label: label.to_string(),
            continuation,
        });
    }

    fn end_thinking(&mut self) -> Option<crate::spinner::Spinner> {
        self.send(RenderEvent::EndThinking);
        // A non-terminal renderer owns no `Spinner` for the agent loop to
        // stop — the render task retires its own indicator.
        None
    }

    fn got_token(&self) -> bool {
        self.got_token
    }

    fn reconfigure(&mut self, flags: SinkFlags) {
        self.flags = flags;
    }

    fn tool_started(&mut self, name: &str) {
        self.send(RenderEvent::ToolStarted {
            name: name.to_string(),
        });
    }

    fn tool_finished(&mut self, name: &str, ok: bool, output: &str, elapsed_ms: u64) {
        self.send(RenderEvent::ToolFinished {
            name: name.to_string(),
            ok,
            output: output.to_string(),
            elapsed_ms,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc::{self, UnboundedReceiver};

    fn channel() -> (TuiSink, UnboundedReceiver<RenderEvent>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (TuiSink::new(tx), rx)
    }

    #[test]
    fn assistant_token_sends_event_and_tracks_got_token() {
        let (mut sink, mut rx) = channel();
        assert!(!sink.got_token());
        sink.assistant_token("hello");
        assert!(sink.got_token());
        match rx.try_recv() {
            Ok(RenderEvent::AssistantToken(t)) => assert_eq!(t, "hello"),
            other => panic!("expected AssistantToken, got {other:?}"),
        }
    }

    #[test]
    fn assistant_final_sends_event() {
        let (mut sink, mut rx) = channel();
        sink.assistant_final("done");
        match rx.try_recv() {
            Ok(RenderEvent::AssistantFinal(t)) => assert_eq!(t, "done"),
            other => panic!("expected AssistantFinal, got {other:?}"),
        }
    }

    #[test]
    fn status_sends_event() {
        let (mut sink, mut rx) = channel();
        sink.status("connected");
        match rx.try_recv() {
            Ok(RenderEvent::Status(t)) => assert_eq!(t, "connected"),
            other => panic!("expected Status, got {other:?}"),
        }
    }

    #[test]
    fn usage_footer_sends_event() {
        let (mut sink, mut rx) = channel();
        let stats = ChatStats {
            prompt_tokens: 1,
            completion_tokens: 2,
            elapsed_ms: 3,
        };
        sink.usage_footer(&stats, Some(8000));
        match rx.try_recv() {
            Ok(RenderEvent::UsageFooter { stats, window }) => {
                assert_eq!(stats.prompt_tokens, 1);
                assert_eq!(stats.completion_tokens, 2);
                assert_eq!(window, Some(8000));
            }
            other => panic!("expected UsageFooter, got {other:?}"),
        }
    }

    #[test]
    fn begin_thinking_only_sends_event_and_resets_got_token() {
        let (mut sink, mut rx) = channel();
        // Simulate a prior token so we can observe the reset.
        sink.assistant_token("x");
        assert!(sink.got_token());
        let _ = rx.try_recv(); // drain the token event

        // Must not block or panic (no real Spinner started).
        sink.begin_thinking("thinking…", true);
        assert!(!sink.got_token());
        match rx.try_recv() {
            Ok(RenderEvent::BeginThinking {
                label,
                continuation,
            }) => {
                assert_eq!(label, "thinking…");
                assert!(continuation);
            }
            other => panic!("expected BeginThinking, got {other:?}"),
        }
    }

    #[test]
    fn end_thinking_sends_event_and_returns_none() {
        let (mut sink, mut rx) = channel();
        let spinner = sink.end_thinking();
        assert!(spinner.is_none());
        match rx.try_recv() {
            Ok(RenderEvent::EndThinking) => {}
            other => panic!("expected EndThinking, got {other:?}"),
        }
    }

    #[test]
    fn reconfigure_updates_flags() {
        let (mut sink, _rx) = channel();
        assert!(!sink.flags.headless);
        sink.reconfigure(SinkFlags {
            headless: true,
            json: true,
            markdown: true,
        });
        assert!(sink.flags.headless);
        assert!(sink.flags.json);
        assert!(sink.flags.markdown);
    }

    #[test]
    fn tool_started_and_finished_send_events() {
        let (mut sink, mut rx) = channel();
        sink.tool_started("fs");
        match rx.try_recv() {
            Ok(RenderEvent::ToolStarted { name }) => assert_eq!(name, "fs"),
            other => panic!("expected ToolStarted, got {other:?}"),
        }
        sink.tool_finished("fs", false, "boom", 1234);
        match rx.try_recv() {
            Ok(RenderEvent::ToolFinished {
                name,
                ok,
                output,
                elapsed_ms,
            }) => {
                assert_eq!(name, "fs");
                assert!(!ok);
                assert_eq!(output, "boom");
                assert_eq!(elapsed_ms, 1234);
            }
            other => panic!("expected ToolFinished, got {other:?}"),
        }
    }

    #[test]
    fn send_after_receiver_dropped_is_silent() {
        let (mut sink, rx) = channel();
        drop(rx);
        // Must not panic even though the receiver is gone.
        sink.assistant_token("orphaned");
        sink.status("orphaned");
        sink.end_thinking();
    }
}
