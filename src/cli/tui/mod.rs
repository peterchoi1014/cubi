//! Opt-in full-screen terminal UI (Phase 2).
//!
//! Architecture: an opt-in ratatui + crossterm renderer runs on a dedicated
//! **render task** that solely owns the `Terminal`, the [`AppState`], and the
//! crossterm event stream. The [`TuiSink`] handed to the agent loop holds
//! *only* an [`mpsc::UnboundedSender`](tokio::sync::mpsc::UnboundedSender) of
//! [`RenderEvent`](super::ui_sink::RenderEvent)s (plus small `Copy` flags), so
//! it is `Send` by construction and satisfies `ChatCLI`'s
//! `ui: Box<dyn UiSink + Send>`. The sink never touches a `Terminal` or the
//! `AppState`; it only sends events.
//!
//! Communication (all `Send`):
//!   * `render_tx: UnboundedSender<RenderEvent>` (main → render), held inside a
//!     [`TuiSink`]. `unbounded` so the sync token callback never blocks the
//!     model stream.
//!   * `action_tx: UnboundedSender<UserAction>` (render → main): how the idle
//!     main loop receives the user's submitted line — this replaces
//!     `rl.readline()` in TUI mode.
//!   * `cancel: Arc<AtomicBool>`: set by the render task on Ctrl-C, observed by
//!     `agent_turn`'s extra cancel branch (raw mode disables ISIG, so
//!     `tokio::signal::ctrl_c()` never fires under the TUI).

mod app;
mod diff;
mod event;
mod highlight;
mod markdown;
mod sink;
mod term;
mod theme;
mod widgets;

use crate::cli::ChatCLI;
use crate::cli::ui_sink::RenderEvent;
use crate::commands::{self, Cmd};
use anyhow::{Context, Result};
use app::AppState;
use crossterm::event::{Event, EventStream, KeyEventKind};
use futures_util::StreamExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use std::time::Instant;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

use app::UserAction;
use sink::TuiSink;

/// Redraw tick — ~12fps, enough to animate a spinner without burning CPU.
const TICK: Duration = Duration::from_millis(80);

/// The dedicated render task. Sole owner of the `Terminal`, the [`AppState`],
/// and the crossterm [`EventStream`]. Runs until the user quits (Ctrl-D) or
/// either channel end closes.
///
/// The crossterm event source is injected as `events` so this loop is unit-
/// testable with a synthetic stream (a real `EventStream` panics without a
/// terminal reader). Production callers pass `EventStream::new()`.
async fn render_task<B, S>(
    mut terminal: ratatui::Terminal<B>,
    mut state: AppState,
    mut render_rx: UnboundedReceiver<RenderEvent>,
    action_tx: UnboundedSender<UserAction>,
    cancel: Arc<AtomicBool>,
    mut events: S,
) -> Result<(), B::Error>
where
    B: ratatui::backend::Backend,
    S: futures_util::Stream<Item = std::io::Result<Event>> + Unpin,
{
    let mut tick = tokio::time::interval(TICK);
    // Thinking-indicator timing is owned here: the render task owns the clock
    // and the redraw tick. `thinking_since` starts on the first frame where the
    // state is thinking and clears when it stops; `spinner_frame` advances once
    // per tick. `redraw` feeds the elapsed/frame into the (pure) widgets layer.
    let mut thinking_since: Option<Instant> = None;
    let mut spinner_frame: usize = 0;
    // Paint the initial frame (status row + empty composer) immediately.
    redraw(
        &mut terminal,
        &mut state,
        &mut thinking_since,
        &mut spinner_frame,
        false,
    )?;

    loop {
        tokio::select! {
            // Agent-loop output → fold into the view model and repaint.
            maybe_ev = render_rx.recv() => {
                match maybe_ev {
                    Some(ev) => {
                        state.apply(ev);
                        redraw(&mut terminal, &mut state, &mut thinking_since, &mut spinner_frame, false)?;
                    }
                    // Sender dropped (main restored its sink on shutdown).
                    None => break,
                }
            }
            // Local key input → edit / submit / cancel / quit.
            maybe_key = events.next() => {
                match maybe_key {
                    Some(Ok(Event::Key(key))) => {
                        // Ignore key *release* / *repeat* synthetic events
                        // (Windows / kitty protocol) so a keypress isn't
                        // processed twice.
                        if key.kind != KeyEventKind::Press {
                            continue;
                        }
                        match event::map_key(key) {
                            event::Action::Edit(e) => {
                                state.edit(e);
                                redraw(&mut terminal, &mut state, &mut thinking_since, &mut spinner_frame, false)?;
                            }
                            event::Action::Submit => {
                                let line = state.take_composer();
                                if !line.trim().is_empty() {
                                    // Echo the submitted line as a `You` role
                                    // block directly into the transcript (the
                                    // render task owns `state`), so the live
                                    // turn matches resumed history.
                                    state.push_user_turn(line.trim());
                                    // Best-effort: if main is gone we exit next loop.
                                    let _ = action_tx.send(UserAction::SubmitLine(line));
                                }
                                redraw(&mut terminal, &mut state, &mut thinking_since, &mut spinner_frame, false)?;
                            }
                            event::Action::Cancel => {
                                cancel.store(true, Ordering::SeqCst);
                            }
                            event::Action::Quit => {
                                let _ = action_tx.send(UserAction::Quit);
                                break;
                            }
                            event::Action::ScrollUp => {
                                state.scroll_up(3);
                                redraw(&mut terminal, &mut state, &mut thinking_since, &mut spinner_frame, false)?;
                            }
                            event::Action::ScrollDown => {
                                state.scroll_down(3);
                                redraw(&mut terminal, &mut state, &mut thinking_since, &mut spinner_frame, false)?;
                            }
                            event::Action::None => {}
                        }
                    }
                    // Resize / focus / paste etc: repaint against current size.
                    Some(Ok(_)) => {
                        redraw(&mut terminal, &mut state, &mut thinking_since, &mut spinner_frame, false)?;
                    }
                    Some(Err(_)) => {}
                    // Event stream ended (stdin closed).
                    None => break,
                }
            }
            // Periodic repaint so the thinking indicator animates. The tick is
            // the only site that advances the spinner frame.
            _ = tick.tick() => {
                redraw(&mut terminal, &mut state, &mut thinking_since, &mut spinner_frame, true)?;
            }
        }
    }
    Ok(())
}

/// Sync the thinking-indicator animation state and repaint.
///
/// The render task owns the clock: on the first thinking frame this stamps
/// `thinking_since`, and clears it once thinking stops. `advance` is `true`
/// only on the redraw tick, so the spinner glyph steps at a steady rate while
/// event-driven redraws keep the *elapsed* readout fresh without skipping
/// frames. The computed `(frame, elapsed_ms)` are pushed onto `state` so
/// [`widgets::draw`] stays a pure function of `AppState`.
fn redraw<B>(
    terminal: &mut ratatui::Terminal<B>,
    state: &mut AppState,
    thinking_since: &mut Option<Instant>,
    spinner_frame: &mut usize,
    advance: bool,
) -> Result<(), B::Error>
where
    B: ratatui::backend::Backend,
{
    if state.thinking() {
        let since = thinking_since.get_or_insert_with(Instant::now);
        if advance {
            *spinner_frame = spinner_frame.wrapping_add(1);
        }
        let elapsed_ms = since.elapsed().as_millis() as u64;
        state.set_thinking_anim(*spinner_frame, elapsed_ms);
    } else {
        *thinking_since = None;
        state.set_thinking_anim(0, 0);
    }
    terminal.draw(|f| widgets::draw(f, state))?;
    Ok(())
}

/// Replay prior conversation turns into the transcript when seeding a resumed
/// (or otherwise preloaded) session. User and assistant messages are shown as
/// role-headed blocks (`You` / `Cubi` headers + prose), identical to live
/// turns; system, pinned, and tool messages are skipped, and assistant
/// `<think>` blocks are stripped. Pure over `(state, history)` so it is
/// unit-testable without a terminal.
fn seed_history(state: &mut AppState, history: &[crate::ollama::Message]) {
    for msg in history {
        match msg.role.as_str() {
            "user" => {
                let text = msg.content.trim();
                if !text.is_empty() {
                    state.push_user_turn(text);
                }
            }
            "assistant" => {
                let stripped = crate::thinking_filter::strip_thinking_blocks(&msg.content);
                let text = stripped.trim();
                if !text.is_empty() {
                    state.apply(RenderEvent::AssistantFinal(text.to_string()));
                }
            }
            _ => {}
        }
    }
}

impl ChatCLI {
    /// Interactive entry point for the opt-in full-screen TUI (`--tui`).
    ///
    /// Spawns the render task, swaps `self.ui` to a [`TuiSink`] for the
    /// session, and drives an idle main loop that receives submitted lines from
    /// the render task (replacing `rl.readline()`). Restores the sink and
    /// terminal on exit. Never entered for headless / one-shot / JSON runs;
    /// falls back to [`run`](ChatCLI::run) when stdout is not a TTY.
    pub async fn run_tui(&mut self) -> Result<()> {
        use std::io::IsTerminal;
        if !std::io::stdout().is_terminal() {
            // Don't drive raw mode on a pipe; degrade to the standard REPL.
            self.emit_status("cubi: --tui requires a TTY; using standard mode.");
            return self.run().await;
        }

        // Lifecycle hooks + receipt, mirroring `run()` (interactive mode).
        self.hooks.fire_session_start(self.executor.get_model());
        self.emit_receipt(
            crate::receipts::ReceiptEvent::SessionStart {
                model: self.executor.get_model().to_string(),
                cwd: std::env::current_dir().unwrap_or_default(),
            },
            &serde_json::json!({
                "model": self.executor.get_model(),
                "cwd": std::env::current_dir().ok().map(|p| p.display().to_string()),
                "mode": "tui",
            }),
        );

        // Enter raw mode + alt screen (belt) and install the panic-restore
        // hook (suspenders). Both restore paths share one idempotent gate.
        let guard = term::TerminalGuard::new().context("failed to initialize the TUI terminal")?;
        term::install_panic_hook(guard.done_flag());

        let terminal = match term::new_terminal() {
            Ok(t) => t,
            Err(e) => {
                guard.restore();
                return Err(anyhow::Error::from(e).context("failed to build the TUI terminal"));
            }
        };

        // Wire the channels + the shared cancel flag.
        let (render_tx, render_rx) = mpsc::unbounded_channel::<RenderEvent>();
        let (action_tx, mut action_rx) = mpsc::unbounded_channel::<UserAction>();
        let cancel = Arc::clone(&self.cancel);
        cancel.store(false, Ordering::SeqCst);
        // A second handle on the render channel so the main task can push a
        // fresh status snapshot after each turn without reaching into the sink.
        let status_tx = render_tx.clone();

        // Seed the initial view with the normal startup output (which the
        // alternate screen would otherwise wipe) BEFORE the live status
        // snapshot: first the captured init/loading lines + tip, then the
        // welcome content (mascot + tagline + help-line + banner). Plain text
        // (color=false) — the transcript renders its own styled spans.
        let mut state = AppState::new();
        state.set_theme(theme::Theme::from_name(
            self.app_config.theme.as_deref().unwrap_or("auto"),
        ));
        for line in &self.startup_transcript {
            state.apply(RenderEvent::Status(line.clone()));
        }
        for line in self.welcome_lines(false) {
            state.apply(RenderEvent::Status(line));
        }
        // On resume (or any preloaded history), replay the prior user/assistant
        // turns into the transcript so the TUI shows the conversation being
        // continued rather than opening on just the banner.
        seed_history(&mut state, &self.history);
        state.apply(RenderEvent::StatusSnapshot(self.status_snapshot()));

        let render_handle = tokio::spawn(render_task(
            terminal,
            state,
            render_rx,
            action_tx,
            Arc::clone(&cancel),
            EventStream::new(),
        ));

        // Swap the sink for the whole session and mark the TUI active so tool
        // approval auto-denies and raw stdout prints are suppressed.
        let prev_ui = std::mem::replace(&mut self.ui, Box::new(TuiSink::new(render_tx)));
        self.tui_active = true;

        // Idle main loop: wait for the user's submitted line. Any non-submit
        // outcome (explicit Quit, or the render channel closing) ends the loop.
        while let Some(UserAction::SubmitLine(line)) = action_rx.recv().await {
            if !self.tui_submit(&line).await {
                break;
            }
            // Refresh the status row after each turn (tokens/cost move).
            let _ = status_tx.send(RenderEvent::StatusSnapshot(self.status_snapshot()));
        }

        // Restore the sink, then reap the render task, then restore the
        // terminal. Aborting + joining the render task BEFORE `guard.restore()`
        // guarantees no tick-driven `terminal.draw()` can fire after
        // `LeaveAlternateScreen`, which would otherwise paint a stray frame
        // onto the normal screen.
        self.ui = prev_ui;
        self.tui_active = false;
        render_handle.abort();
        let _ = render_handle.await;
        guard.restore();
        drop(guard);

        self.hooks.fire_stop();
        self.emit_receipt(
            crate::receipts::ReceiptEvent::SessionEnd,
            &serde_json::json!({"mode": "tui"}),
        );

        // Now that the terminal is back on the normal screen, print the
        // resume hint (with the session id) so the user can copy it — the
        // same hint the standard REPL shows on exit.
        if let Some(hint) = self.resume_hint() {
            println!("\n{hint}");
        }
        Ok(())
    }

    /// Handle one submitted line from the TUI, replicating the REPL's
    /// submission path (`@file` expansion, history push, journal bucket, agent
    /// turn). Returns `false` when the session should quit. All user-visible
    /// output flows through `self.ui` (never raw stdout) so it lands in the
    /// transcript, not on the alt screen. Slash commands are gated to quit-only
    /// in this preview (see the `input.starts_with('/')` branch) because the
    /// REPL command handlers print via raw stdout.
    async fn tui_submit(&mut self, input: &str) -> bool {
        let input = input.trim();
        if input.is_empty() {
            return true;
        }

        if input.starts_with('/') {
            // PROTOTYPE LIMITATION: `handle_command`/`try_user_command` print
            // via raw `println!` by design (correct for the normal REPL), but
            // that scribbles directly on the alternate screen here. Until that
            // output is routed through the sink, the only slash command honored
            // in `--tui` is quit. `commands::parse` gives us the same quit
            // recognition as the REPL (`/quit`, `/exit`, and unambiguous
            // prefixes like `/q`). Every other slash command is acknowledged
            // via a status line so nothing reaches raw stdout.
            if matches!(commands::parse(input), Some((Cmd::Quit, _))) {
                return false; // clean TUI exit — same false/Quit path as run_tui
            }
            self.ui.status(
                "slash commands are not yet available in --tui preview (use /quit to exit)",
            );
            return true;
        }

        // The user's line is echoed into the transcript as a `You` role block
        // by the render task (see `render_task`'s Submit branch) before this
        // runs, so we do not re-echo it here.
        let expanded = crate::file_mentions::expand_file_mentions(input);
        let turn_start = self.history.len();
        self.history
            .push(crate::ollama::Message::text("user", &expanded));
        self.journal.start_turn();
        if let Err(e) = self.agent_turn(turn_start).await {
            self.ui.status(&format!("Error: {e}"));
        }
        true
    }
}

#[cfg(test)]
mod render_loop_tests {
    use super::*;
    use crate::cli::status::StatusState;
    use crate::ollama::ChatStats;
    use std::path::PathBuf;
    use tokio::sync::mpsc;

    fn sample_status() -> StatusState {
        StatusState {
            model: "qwen3:8b".to_string(),
            context_used: Some(1000),
            context_window: Some(8000),
            cwd: PathBuf::from("/tmp/project"),
            prompt_tokens: 10,
            completion_tokens: 20,
            cost: "$0.00 (local)".to_string(),
            session_details: "ollama · 0 msgs · sessions ok".to_string(),
        }
    }

    #[test]
    fn seed_history_replays_user_and_assistant_turns() {
        use crate::ollama::Message;
        use app::LineKind;
        let history = vec![
            Message::text("system", "you are cubi"),
            Message::text("user", "pineapple castle marker"),
            Message::text("assistant", "<think>ponder</think>sure thing"),
            Message::text("tool", "{\"result\":1}"),
            Message::text("user", "   "), // whitespace-only: skipped
        ];
        let mut state = AppState::new();
        seed_history(&mut state, &history);
        let kinds: Vec<LineKind> = state.transcript().iter().map(|l| l.kind).collect();
        // Resumed turns use the SAME role-headed block model as live turns:
        // `You` header + prose, a blank separator, then `Cubi` header + prose.
        // System, tool, and blank user messages are skipped.
        assert_eq!(
            kinds,
            vec![
                LineKind::UserHeader,
                LineKind::User,
                LineKind::Blank,
                LineKind::AssistantHeader,
                LineKind::Assistant,
            ]
        );
        let texts: Vec<&str> = state.transcript().iter().map(|l| l.text.as_str()).collect();
        assert_eq!(
            texts,
            vec!["You", "pineapple castle marker", "", "Cubi", "sure thing"]
        );
    }

    /// Drive the render-task *state-folding* path over the render channel with
    /// synthetic events, using a headless `TestBackend` terminal (no tty). This
    /// exercises the same `apply` + `draw` sequence the real loop runs, and
    /// asserts a UserAction round-trips through the action channel. We stop the
    /// task by dropping the render sender (recv → None → break).
    #[tokio::test]
    async fn render_task_folds_events_and_emits_actions() {
        let backend = ratatui::backend::TestBackend::new(60, 12);
        let terminal = ratatui::Terminal::new(backend).unwrap();
        let (render_tx, render_rx) = mpsc::unbounded_channel::<RenderEvent>();
        let (action_tx, mut action_rx) = mpsc::unbounded_channel::<UserAction>();
        let cancel = Arc::new(AtomicBool::new(false));

        let mut state = AppState::new();
        state.apply(RenderEvent::StatusSnapshot(sample_status()));

        let handle = tokio::spawn(render_task(
            terminal,
            state,
            render_rx,
            action_tx.clone(),
            Arc::clone(&cancel),
            futures_util::stream::pending::<std::io::Result<Event>>(),
        ));

        // Feed a couple of events the task must fold + redraw without error.
        render_tx
            .send(RenderEvent::Status("session started".into()))
            .unwrap();
        render_tx
            .send(RenderEvent::AssistantFinal("hello there".into()))
            .unwrap();
        render_tx
            .send(RenderEvent::UsageFooter {
                stats: ChatStats {
                    prompt_tokens: 1,
                    completion_tokens: 2,
                    elapsed_ms: 3,
                },
                window: Some(8000),
            })
            .unwrap();

        // Simulate the render task handing a submitted line to main. (In the
        // real loop this is produced by a key event; here we assert the channel
        // wiring the main loop depends on.)
        action_tx
            .send(UserAction::SubmitLine("say hi".into()))
            .unwrap();

        // Confirm the action reaches the main-side receiver.
        let got = action_rx.recv().await;
        assert_eq!(got, Some(UserAction::SubmitLine("say hi".into())));

        // Close the render channel → task's recv returns None → it breaks.
        drop(render_tx);
        let joined = handle.await.expect("render task join");
        assert!(joined.is_ok(), "render task should exit cleanly");
    }

    /// The main loop relies on the render task ending cleanly when its render
    /// channel closes. Assert that invariant (no panic, `Ok`).
    #[tokio::test]
    async fn render_task_exits_when_render_channel_closes() {
        let backend = ratatui::backend::TestBackend::new(40, 8);
        let terminal = ratatui::Terminal::new(backend).unwrap();
        let (render_tx, render_rx) = mpsc::unbounded_channel::<RenderEvent>();
        let (action_tx, _action_rx) = mpsc::unbounded_channel::<UserAction>();
        let cancel = Arc::new(AtomicBool::new(false));
        let state = AppState::new();

        let handle = tokio::spawn(render_task(
            terminal,
            state,
            render_rx,
            action_tx,
            cancel,
            futures_util::stream::pending::<std::io::Result<Event>>(),
        ));
        drop(render_tx);
        let joined = handle.await.expect("join");
        assert!(joined.is_ok());
    }
}
