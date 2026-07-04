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

mod ansi;
mod app;
mod capture;
mod diff;
mod event;
mod highlight;
mod markdown;
mod sink;
mod term;
mod theme;
mod widgets;

use crate::cli::AgentCommand;
use crate::cli::ChatCLI;
use crate::cli::ui_sink::RenderEvent;
use crate::commands::{self, Cmd};
use anyhow::{Context, Result};
use app::AppState;
use crossterm::event::{Event, EventStream, KeyEventKind, MouseEventKind};
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

/// Control messages from the main task to the render task, on a channel
/// *separate* from [`RenderEvent`] so a suspend request is never queued behind
/// pending render output. Used to momentarily leave the TUI so a slash-command
/// handler can run on the real terminal.
///
/// The handshake is two-phase. On `Suspend` the render task drops its
/// [`EventStream`] (crossterm's stream eagerly consumes stdin via a background
/// reader, so it must be *dropped* — not merely un-polled — or it steals the
/// command's confirmation keystrokes), acks via `ack` that it has parked and
/// released the terminal, then blocks on `resume`. Only after the ack does the
/// main task touch the real terminal; only after the main task re-enters the
/// TUI does it fire `resume`, at which point the render task rebuilds its
/// event source, clears (full repaint) and resumes its select loop. While
/// parked the render task touches neither stdin nor stdout.
enum RenderControl {
    Suspend {
        /// render → main: sent once the task has parked and released the
        /// terminal, so the main task may safely drive the real terminal.
        ack: tokio::sync::oneshot::Sender<()>,
        /// main → render: fires once the command has finished and the terminal
        /// has been re-entered, telling the task to rebuild its `EventStream`,
        /// force a full repaint, and resume.
        resume: tokio::sync::oneshot::Receiver<()>,
    },
}

/// The dedicated render task. Sole owner of the `Terminal`, the [`AppState`],
/// and the crossterm [`EventStream`]. Runs until the user quits (Ctrl-D) or
/// either channel end closes.
///
/// The crossterm event source is injected as a *factory* `make_events` so this
/// loop is unit-testable with a synthetic stream (a real `EventStream` panics
/// without a terminal reader). Production callers pass `EventStream::new`.
/// A factory (rather than a single stream) is required because a slash-command
/// suspend must DROP the current stream and build a fresh one on resume.
async fn render_task<B, S, F>(
    mut terminal: ratatui::Terminal<B>,
    mut state: AppState,
    mut render_rx: UnboundedReceiver<RenderEvent>,
    action_tx: UnboundedSender<UserAction>,
    cancel: Arc<AtomicBool>,
    mut control_rx: UnboundedReceiver<RenderControl>,
    mut make_events: F,
) -> Result<(), B::Error>
where
    B: ratatui::backend::Backend,
    S: futures_util::Stream<Item = std::io::Result<Event>> + Unpin,
    F: FnMut() -> S,
{
    // `events` is an `Option` so a suspend can genuinely drop the stream
    // (releasing crossterm's background stdin reader) and rebuild it on resume.
    // It is `Some` whenever the select loop runs; while parked the task is
    // blocked inside the `Suspend` branch, not re-entering the loop.
    let mut events: Option<S> = Some(make_events());
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
            // Control channel (main → render): suspend/resume around a
            // slash-command that runs on the real terminal.
            maybe_ctrl = control_rx.recv() => {
                match maybe_ctrl {
                    Some(RenderControl::Suspend { ack, resume }) => {
                        // Park in place. Drop the EventStream so crossterm's
                        // background stdin reader releases the terminal (else it
                        // steals the command's confirmation keystrokes), ack
                        // that we've parked, then block until the main task has
                        // run the command and re-entered the TUI.
                        drop(events.take());
                        let _ = ack.send(());
                        // While parked we touch neither stdin nor stdout. If the
                        // resume sender is DROPPED instead of fired, the command
                        // asked to quit and the main task did NOT re-enter the
                        // TUI (terminal is on the normal screen); exit WITHOUT
                        // repainting so we never scribble a stray frame onto it.
                        if resume.await.is_err() {
                            break;
                        }
                        // Rebuild the event source now that the command has
                        // released stdin.
                        events = Some(make_events());
                        // The interval accrued "missed" ticks while parked;
                        // reset it so resume doesn't fire a burst of redraws
                        // (default `MissedTickBehavior::Burst`) all at once.
                        tick.reset();
                        // Force a full repaint: the command scribbled the
                        // normal screen and we re-entered a fresh (blank) alt
                        // screen, so ratatui's diff cache is stale. We must NOT
                        // use `Terminal::clear` here: it first issues an ESC[6n
                        // cursor-position query and *blocks reading the reply
                        // from stdin* (2s timeout). That read races the freshly
                        // rebuilt `EventStream` reader — and some terminals
                        // (and PTYs) never answer — so it intermittently errors
                        // with "cursor position could not be read"; propagated
                        // via `?` it would kill the render task, ending the
                        // whole TUI after a single command. `Terminal::resize`
                        // resets the back buffer and clears via `ClearType::All`
                        // with NO stdin round-trip, forcing the next `draw` to
                        // repaint every cell. A transient repaint hiccup on
                        // resume must likewise never kill the loop, so both
                        // steps are best-effort (no `?`); the periodic tick
                        // redraw will recover on the next frame.
                        force_full_repaint(&mut terminal);
                        let _ = redraw(&mut terminal, &mut state, &mut thinking_since, &mut spinner_frame, false);
                    }
                    // Main dropped the control sender: it is shutting down (it
                    // aborts + drops this task), so stop selecting on a closed
                    // channel and let the loop exit.
                    None => break,
                }
            }
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
            // Local key input → edit / submit / cancel / quit. `events` is an
            // `Option`; while parked (None) this branch parks forever so the
            // select loop is driven only by control/render/tick.
            maybe_key = async {
                match events.as_mut() {
                    Some(s) => s.next().await,
                    None => std::future::pending::<Option<std::io::Result<Event>>>().await,
                }
            } => {
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
                                // With the picker open, Enter executes the
                                // highlighted command immediately (no prompt):
                                // clear the partial token, route the full
                                // command through the same submit path as a
                                // typed line, and close the picker.
                                if let Some(cmd) =
                                    state.picker_selected_command().map(str::to_string)
                                {
                                    let _ = state.take_composer();
                                    state.push_history(&cmd);
                                    // Slash commands are not echoed (they run
                                    // suspended on the real terminal).
                                    let _ = action_tx.send(UserAction::SubmitLine(cmd));
                                    redraw(&mut terminal, &mut state, &mut thinking_since, &mut spinner_frame, false)?;
                                    continue;
                                }
                                let line = state.take_composer();
                                state.push_history(&line);
                                let trimmed = line.trim();
                                if !trimmed.is_empty() {
                                    // Echo non-slash input as a `You` role block
                                    // directly into the transcript (the render
                                    // task owns `state`), so the live turn
                                    // matches resumed history. Slash commands are
                                    // NOT echoed: they run suspended on the real
                                    // terminal, so echoing them would pollute the
                                    // redrawn transcript with a stray `You` block.
                                    // Echo non-command input as a `You` role
                                    // block into the transcript. Slash and
                                    // `!`-shell commands are NOT echoed: they
                                    // run suspended on the real terminal, so
                                    // echoing would pollute the redrawn
                                    // transcript with a stray `You` block.
                                    if !trimmed.starts_with('/') && !trimmed.starts_with('!') {
                                        state.push_user_turn(trimmed);
                                    }
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
                            event::Action::Complete => {
                                // Tab accepts the highlighted picker candidate
                                // when the picker is open; otherwise it runs the
                                // usual slash/`@file` token completion.
                                if state.picker_open() {
                                    state.picker_accept();
                                } else {
                                    state.complete();
                                }
                                redraw(&mut terminal, &mut state, &mut thinking_since, &mut spinner_frame, false)?;
                            }
                            event::Action::HistoryPrev => {
                                // Up moves the picker selection when open;
                                // otherwise recall an older entry, or scroll up
                                // if there is nothing to recall.
                                if state.picker_open() {
                                    state.picker_prev();
                                } else if !state.history_prev() {
                                    state.scroll_up(3);
                                }
                                redraw(&mut terminal, &mut state, &mut thinking_since, &mut spinner_frame, false)?;
                            }
                            event::Action::HistoryNext => {
                                // Down moves the picker selection when open;
                                // otherwise step to a newer entry, or scroll
                                // down when not navigating history.
                                if state.picker_open() {
                                    state.picker_next();
                                } else if !state.history_next() {
                                    state.scroll_down(3);
                                }
                                redraw(&mut terminal, &mut state, &mut thinking_since, &mut spinner_frame, false)?;
                            }
                            event::Action::DismissPicker => {
                                // Esc closes the picker (staying closed until the
                                // next edit); a no-op when it is not open.
                                if state.picker_open() {
                                    state.picker_dismiss();
                                    redraw(&mut terminal, &mut state, &mut thinking_since, &mut spinner_frame, false)?;
                                }
                            }
                            event::Action::None => {}
                        }
                    }
                    // Mouse wheel → scroll the transcript. With mouse capture
                    // enabled the wheel arrives here (not as Up/Down keys), so
                    // keyboard arrows can keep driving history recall. Mirror
                    // the `Action::ScrollUp`/`ScrollDown` handlers; ignore all
                    // other mouse kinds (clicks/drags) so they don't interfere.
                    Some(Ok(Event::Mouse(m))) => match m.kind {
                        MouseEventKind::ScrollUp => {
                            state.scroll_up(3);
                            redraw(&mut terminal, &mut state, &mut thinking_since, &mut spinner_frame, false)?;
                        }
                        MouseEventKind::ScrollDown => {
                            state.scroll_down(3);
                            redraw(&mut terminal, &mut state, &mut thinking_since, &mut spinner_frame, false)?;
                        }
                        _ => {}
                    },
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

/// Force the next [`redraw`] to repaint every cell, without any stdin round-trip.
///
/// After a suspend/resume cycle the terminal has been left and re-entered, so
/// the real screen is blank but ratatui's diff cache still believes the last
/// TUI frame is present — a plain `draw` would diff against that stale buffer
/// and paint nothing. [`Terminal::clear`] would fix the cache but issues an
/// `ESC[6n` cursor-position query and blocks reading the reply from stdin,
/// which races the rebuilt `EventStream` reader and hangs/errors on terminals
/// (and PTYs) that don't answer. [`Terminal::resize`] to the current size
/// achieves the same back-buffer reset + full clear (`ClearType::All`) with no
/// stdin read. Best-effort: a failure here is swallowed so a transient hiccup
/// on resume can never kill the render task; the next frame recovers.
fn force_full_repaint<B>(terminal: &mut ratatui::Terminal<B>)
where
    B: ratatui::backend::Backend,
{
    if let Ok(size) = terminal.size() {
        let _ = terminal.resize(size.into());
    }
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
        // Control channel (main → render) for the slash-command suspend/resume
        // handshake, kept separate from `render_tx` so a suspend is never
        // queued behind pending render output.
        let (control_tx, control_rx) = mpsc::unbounded_channel::<RenderControl>();
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
        // Seed input-history recall from the persisted REPL history file so
        // Up/Down recall spans sessions. Best-effort: ignore a missing or
        // unreadable file, and read one entry per line.
        if let Some(path) = crate::cli::repl::repl_history_path() {
            if let Ok(contents) = std::fs::read_to_string(&path) {
                state.seed_input_history(contents.lines().map(str::to_string));
            }
        }
        // Seed the slash-command picker catalog: built-in command names plus
        // every enabled agent as `/<name>`. Built-ins win any name collision,
        // so an agent whose name duplicates a built-in is skipped. This is a
        // startup snapshot; agents enabled/disabled mid-session are not
        // reflected until the next launch (acceptable for v1).
        {
            let mut catalog: Vec<String> =
                crate::commands::command_names().map(String::from).collect();
            let builtins: std::collections::HashSet<String> =
                catalog.iter().map(|c| c.to_ascii_lowercase()).collect();
            for agent in crate::agents::load_agents() {
                if crate::agents::is_disabled(&agent.name) {
                    continue;
                }
                let cmd = format!("/{}", agent.name);
                if !builtins.contains(&cmd.to_ascii_lowercase()) {
                    catalog.push(cmd);
                }
            }
            state.set_command_catalog(catalog);
        }
        state.apply(RenderEvent::StatusSnapshot(self.status_snapshot()));

        let render_handle = tokio::spawn(render_task(
            terminal,
            state,
            render_rx,
            action_tx,
            Arc::clone(&cancel),
            control_rx,
            EventStream::new,
        ));

        // Swap the sink for the whole session and mark the TUI active so tool
        // approval auto-denies and raw stdout prints are suppressed.
        let prev_ui = std::mem::replace(&mut self.ui, Box::new(TuiSink::new(render_tx)));
        self.tui_active = true;

        // Idle main loop: wait for the user's submitted line. Any non-submit
        // outcome (explicit Quit, or the render channel closing) ends the loop.
        while let Some(UserAction::SubmitLine(line)) = action_rx.recv().await {
            if !self.tui_submit(&line, &control_tx, &guard).await {
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
    /// turn). Returns `false` when the session should quit.
    ///
    /// Non-slash input flows through `self.ui` (the [`TuiSink`], never raw
    /// stdout) so it lands in the transcript. Slash commands cannot: the REPL
    /// handlers print via raw `println!`, which would scribble on the alt
    /// screen. So every non-quit slash command is run via [`run_slash_command`]
    /// (`ChatCLI::run_slash_command`), which momentarily suspends the TUI and
    /// runs the command on the real terminal exactly as the REPL does.
    async fn tui_submit(
        &mut self,
        input: &str,
        control_tx: &UnboundedSender<RenderControl>,
        guard: &term::TerminalGuard,
    ) -> bool {
        let input = input.trim();
        if input.is_empty() {
            return true;
        }

        if input.starts_with('/') {
            // Fast exit: `/quit` / `/exit` never need a suspend cycle — return
            // false and let `run_tui` run the normal shutdown path. This is the
            // same quit recognition the REPL uses (`commands::parse`).
            if matches!(commands::parse(input), Some((Cmd::Quit, _))) {
                return false;
            }
            // The three "managed" commands render their textual output INSIDE
            // the transcript as a framed block (like `!` shell output) rather
            // than suspending the TUI. Only their read-only listing output is
            // captured; `/skills run <name>` still executes an agent turn and
            // stays on the suspend/turn path below.
            if let Some((cmd, cmd_args)) = commands::parse(input) {
                match cmd {
                    Cmd::Skills
                        if {
                            let a = cmd_args.trim();
                            let head = a
                                .split_once(char::is_whitespace)
                                .map(|(h, _)| h)
                                .unwrap_or(a);
                            a.is_empty()
                                || head.eq_ignore_ascii_case("list")
                                || head.eq_ignore_ascii_case("enable")
                                || head.eq_ignore_ascii_case("disable")
                        } =>
                    {
                        let output = self.skills_manage(cmd_args);
                        self.render_captured_command("/skills", &output);
                        return true;
                    }
                    Cmd::Agents => {
                        // All `/agents` subcommands (list/enable/disable) now
                        // yield text we capture in-transcript — there is no
                        // editor/suspend path anymore.
                        let output = self.agents_manage(cmd_args);
                        self.render_captured_command("/agents", &output);
                        return true;
                    }
                    Cmd::Mcp => {
                        // Run the management dispatch (list/enable/disable/add/
                        // remove/reload) and render its text in-transcript, then
                        // keep the statusline denominator fresh exactly as the
                        // classic `/mcp` handler does.
                        let output = self.mcp_manage(cmd_args).await;
                        self.refresh_mcp_counts();
                        self.render_captured_command("/mcp", &output);
                        return true;
                    }
                    Cmd::McpTools => {
                        // Read-only tool listing — render it in-transcript like
                        // `/mcp` rather than suspending to the real terminal.
                        let output = self.mcp_tools_output();
                        self.render_captured_command("/mcp-tools", &output);
                        return true;
                    }
                    _ => {}
                }
            }
            // Dynamic `/<name>` agent command: run it as a LIVE turn streamed
            // through the TuiSink (exactly like a submitted user message), NOT a
            // suspended real-terminal command. `resolve_agent_command` enforces
            // precedence (built-ins and user/plugin commands win).
            match self.resolve_agent_command(input) {
                AgentCommand::Run(agent, agent_args) => {
                    self.run_agent_command(&agent, &agent_args).await;
                    return true;
                }
                AgentCommand::Disabled(name) => {
                    let msg = ChatCLI::agent_disabled_message(&name);
                    self.render_captured_command(&format!("/{name}"), &msg);
                    return true;
                }
                AgentCommand::NotAgent => {}
            }
            // Every other slash command: suspend the TUI, run it on the real
            // terminal, resume. Returns `false` if the command asked to quit.
            return self.run_suspended_command(input, control_tx, guard).await;
        }

        // Shell escape: `!<cmd>` runs a shell command and shows its output as a
        // framed block in the transcript. We CAPTURE the output (rather than
        // suspend to the real terminal like slash commands) because a quick
        // command's output would otherwise flash on the normal screen and be
        // hidden the instant we re-enter the alt screen. Reuses the tool-block
        // rendering (header + framed output + ✓/✗ + duration, incl. diff
        // coloring). Never quits the session.
        if let Some(shell_cmd) = input.strip_prefix('!') {
            let shell_cmd = shell_cmd.trim();
            if shell_cmd.is_empty() {
                self.ui.status("usage: !<shell command>  (e.g. !ls -la)");
                return true;
            }
            self.ui.tool_started(shell_cmd);
            let (ok, output, elapsed_ms) = self.run_shell_command_captured(shell_cmd).await;
            self.ui.tool_finished(shell_cmd, ok, &output, elapsed_ms);
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

    /// Render a slash command's captured textual output as a framed block in
    /// the transcript, reusing the same seam the shell escape (`!cmd`) and the
    /// agent-loop tool blocks use (`ui.tool_started` + `ui.tool_finished`). The
    /// `header` (e.g. `/mcp`) becomes the block's `⚙` header line. Leading and
    /// trailing blank lines from the classic layout are trimmed so the block
    /// doesn't open or close with an empty framed row. There is no meaningful
    /// wall-clock duration for a synchronous listing, so `elapsed_ms` is 0.
    fn render_captured_command(&mut self, header: &str, output: &str) {
        self.ui.tool_started(header);
        let body = output.trim_matches('\n');
        self.ui.tool_finished(header, true, body, 0);
    }

    /// Run one slash command on the *real* terminal via the suspend/resume
    /// handshake, exactly mirroring the REPL's dispatch order
    /// (`try_user_command` first, then `handle_command`). Returns `false` when
    /// the command asked to quit (so `run_tui` breaks its idle loop and runs
    /// the normal shutdown path — leaving the terminal restored).
    ///
    /// Sequence: (1) ask the render task to park and wait for its ack (which
    /// guarantees it has dropped its `EventStream` and released the terminal);
    /// (2) leave the alt screen + raw mode and swap `self.ui` from the
    /// [`TuiSink`] to a real-terminal [`LineSink`] so command output prints to
    /// the user's screen; (3) run the command; (4) restore the sink; (5) on
    /// quit, stop here (do not re-enter); otherwise re-enter the TUI and signal
    /// the render task to resume (rebuild its stream + full repaint).
    /// Run a `/`-command on the *real* terminal by suspending the TUI (park the
    /// render task, leave the alt screen, swap in a line sink), running it, then
    /// re-entering and repainting. Returns `false` when the command asked to
    /// quit. (Shell `!` commands are handled separately in `tui_submit` by
    /// capturing their output into the transcript.)
    async fn run_suspended_command(
        &mut self,
        input: &str,
        control_tx: &UnboundedSender<RenderControl>,
        guard: &term::TerminalGuard,
    ) -> bool {
        use crate::cli::ui_sink::{LineSink, SinkFlags, UiSink};
        use tokio::sync::oneshot;

        // (1) Ask the render task to park; wait until it has released the
        // terminal before we touch it.
        let (ack_tx, ack_rx) = oneshot::channel::<()>();
        let (resume_tx, resume_rx) = oneshot::channel::<()>();
        if control_tx
            .send(RenderControl::Suspend {
                ack: ack_tx,
                resume: resume_rx,
            })
            .is_err()
        {
            // Render task is gone; nothing to drive. Keep the session alive.
            return true;
        }
        // If the ack channel closes without a value the render task has died;
        // don't touch the terminal, just keep the session alive.
        if ack_rx.await.is_err() {
            return true;
        }

        // (2) Leave the TUI and swap in a real-terminal sink. Reconstruct the
        // same `LineSink` profile the REPL uses in interactive mode:
        // headless=false, json=false, markdown per session config.
        guard.suspend();
        let line_sink: Box<dyn UiSink + Send> = Box::new(LineSink::new(SinkFlags {
            headless: false,
            json: false,
            markdown: self.markdown_enabled,
        }));
        let tui_sink = std::mem::replace(&mut self.ui, line_sink);
        self.tui_active = false;

        // (3) Dispatch exactly as the REPL: user-defined commands first, then
        // the built-in handler; honor the handler's quit (`Ok(false)`) return.
        let keep_running = if self.try_user_command(input) {
            true
        } else {
            match self.handle_command(input).await {
                Ok(cont) => cont,
                Err(e) => {
                    eprintln!("Error: {e}");
                    true
                }
            }
        };

        // (4) Restore the original TuiSink regardless of outcome.
        self.ui = tui_sink;
        self.tui_active = true;

        if !keep_running {
            // (5a) Quit: leave the terminal on the normal screen and let the
            // idle loop break; `run_tui`'s shutdown reaps the (still-parked)
            // render task and finalizes. Do NOT re-enter the TUI.
            return false;
        }

        // (5) The command printed to the real terminal during the suspend
        // window; pause until the user presses Enter so quick output isn't
        // wiped the instant we repaint the alt screen. Raw mode is off here
        // (see `guard.suspend()`), so a line-buffered stdin read is correct.
        {
            use std::io::Write;
            print!("\n\x1b[2m— press Enter to return to Cubi —\x1b[0m ");
            let _ = std::io::stdout().flush();
            let mut ack = String::new();
            let _ = std::io::stdin().read_line(&mut ack);
        }

        // (6) Re-enter the TUI, then release the render task to rebuild its
        // event stream and force a full repaint. Ordering matters: enter the
        // alt screen BEFORE the render task draws, or it would paint the normal
        // screen.
        if let Err(e) = guard.re_enter() {
            eprintln!("cubi: failed to re-enter the TUI: {e}");
            return false;
        }
        let _ = resume_tx.send(());
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
        let (_control_tx, control_rx) = mpsc::unbounded_channel::<RenderControl>();
        let cancel = Arc::new(AtomicBool::new(false));

        let mut state = AppState::new();
        state.apply(RenderEvent::StatusSnapshot(sample_status()));

        let handle = tokio::spawn(render_task(
            terminal,
            state,
            render_rx,
            action_tx.clone(),
            Arc::clone(&cancel),
            control_rx,
            futures_util::stream::pending::<std::io::Result<Event>>,
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
        let (_control_tx, control_rx) = mpsc::unbounded_channel::<RenderControl>();
        let cancel = Arc::new(AtomicBool::new(false));
        let state = AppState::new();

        let handle = tokio::spawn(render_task(
            terminal,
            state,
            render_rx,
            action_tx,
            cancel,
            control_rx,
            futures_util::stream::pending::<std::io::Result<Event>>,
        ));
        drop(render_tx);
        let joined = handle.await.expect("join");
        assert!(joined.is_ok());
    }

    /// A `Suspend` control message must PARK the render task: it drops its
    /// event stream, acks that it has released the terminal, and then stops
    /// running its select loop (so it no longer drains `render_rx`) until it is
    /// resumed. On resume it rebuilds the event stream (a fresh
    /// `EventStream::new()` in production) and forces a full repaint, then
    /// resumes draining events. We prove the rebuild via a factory that counts
    /// its invocations, and prove the resumed loop is live by closing
    /// `render_rx` and observing a clean exit.
    #[tokio::test]
    async fn suspend_parks_then_resume_rebuilds_and_repaints() {
        use std::sync::atomic::AtomicUsize;
        use tokio::sync::oneshot;

        let backend = ratatui::backend::TestBackend::new(40, 8);
        let terminal = ratatui::Terminal::new(backend).unwrap();
        let (render_tx, render_rx) = mpsc::unbounded_channel::<RenderEvent>();
        let (action_tx, _action_rx) = mpsc::unbounded_channel::<UserAction>();
        let (control_tx, control_rx) = mpsc::unbounded_channel::<RenderControl>();
        let cancel = Arc::new(AtomicBool::new(false));
        let mut state = AppState::new();
        state.apply(RenderEvent::StatusSnapshot(sample_status()));

        // The factory counts stream (re)creations: 1 at startup, +1 per resume.
        let builds = Arc::new(AtomicUsize::new(0));
        let builds_factory = Arc::clone(&builds);
        let handle = tokio::spawn(render_task(
            terminal,
            state,
            render_rx,
            action_tx,
            Arc::clone(&cancel),
            control_rx,
            move || {
                builds_factory.fetch_add(1, Ordering::SeqCst);
                futures_util::stream::pending::<std::io::Result<Event>>()
            },
        ));

        // Ask the task to park and wait for its ack. Awaiting the ack forces the
        // task to run up to the point where it has dropped its event stream and
        // released the terminal — the precondition for touching the terminal.
        let (ack_tx, ack_rx) = oneshot::channel::<()>();
        let (resume_tx, resume_rx) = oneshot::channel::<()>();
        control_tx
            .send(RenderControl::Suspend {
                ack: ack_tx,
                resume: resume_rx,
            })
            .unwrap();
        ack_rx.await.expect("render task should ack the park");
        // Only the startup stream has been built so far; the parked task has NOT
        // rebuilt one (that happens on resume).
        assert_eq!(builds.load(Ordering::SeqCst), 1);

        // While parked the task is blocked awaiting resume, NOT selecting, so it
        // cannot drain `render_rx`. Queue an event and yield repeatedly: a
        // correctly parked task rebuilds nothing.
        render_tx
            .send(RenderEvent::Status("queued while parked".into()))
            .unwrap();
        for _ in 0..16 {
            tokio::task::yield_now().await;
        }
        assert_eq!(
            builds.load(Ordering::SeqCst),
            1,
            "task must stay parked (no rebuild) until resumed"
        );

        // Resume: the task rebuilds its stream (proving the full-repaint path
        // ran) and resumes its select loop. Closing the render channel then
        // drives it to a clean exit, proving events flow again.
        resume_tx.send(()).unwrap();
        drop(render_tx);
        let joined = handle.await.expect("join");
        assert!(
            joined.is_ok(),
            "render task should exit cleanly after resume"
        );
        assert_eq!(
            builds.load(Ordering::SeqCst),
            2,
            "resume must rebuild the EventStream and force a full repaint"
        );
    }

    /// A [`Backend`](ratatui::backend::Backend) that always fails the
    /// cursor-position query and delegates everything else to a `TestBackend`.
    /// This reproduces the crossterm `cursor::position()` failure the resume
    /// path used to hit: [`Terminal::clear`](ratatui::Terminal::clear) issues an
    /// `ESC[6n` cursor-position query and blocks reading the reply, which on a
    /// PTY (or when racing the rebuilt event reader) fails with "cursor position
    /// could not be read". The old resume code did `terminal.clear()?`, so that
    /// failure `?`-propagated out of `render_task`, killing the whole TUI after
    /// a single slash command. The fix repaints via
    /// [`Terminal::resize`](ratatui::Terminal::resize), which never queries the
    /// cursor — so a task on this backend must SURVIVE a suspend/resume cycle.
    struct CursorQueryFailsBackend {
        inner: ratatui::backend::TestBackend,
    }

    impl ratatui::backend::Backend for CursorQueryFailsBackend {
        type Error = std::io::Error;

        fn draw<'a, I>(&mut self, content: I) -> Result<(), Self::Error>
        where
            I: Iterator<Item = (u16, u16, &'a ratatui::buffer::Cell)>,
        {
            self.inner.draw(content).unwrap();
            Ok(())
        }
        fn hide_cursor(&mut self) -> Result<(), Self::Error> {
            self.inner.hide_cursor().unwrap();
            Ok(())
        }
        fn show_cursor(&mut self) -> Result<(), Self::Error> {
            self.inner.show_cursor().unwrap();
            Ok(())
        }
        fn get_cursor_position(&mut self) -> Result<ratatui::layout::Position, Self::Error> {
            // The exact failure mode: a blocking DSR read that never resolves.
            Err(std::io::Error::other(
                "The cursor position could not be read within a normal duration",
            ))
        }
        fn set_cursor_position<P: Into<ratatui::layout::Position>>(
            &mut self,
            position: P,
        ) -> Result<(), Self::Error> {
            self.inner.set_cursor_position(position).unwrap();
            Ok(())
        }
        fn clear(&mut self) -> Result<(), Self::Error> {
            self.inner.clear().unwrap();
            Ok(())
        }
        fn clear_region(
            &mut self,
            clear_type: ratatui::backend::ClearType,
        ) -> Result<(), Self::Error> {
            self.inner.clear_region(clear_type).unwrap();
            Ok(())
        }
        fn size(&self) -> Result<ratatui::layout::Size, Self::Error> {
            Ok(self.inner.size().unwrap())
        }
        fn window_size(&mut self) -> Result<ratatui::backend::WindowSize, Self::Error> {
            Ok(self.inner.window_size().unwrap())
        }
        fn flush(&mut self) -> Result<(), Self::Error> {
            self.inner.flush().unwrap();
            Ok(())
        }
    }

    /// Regression guard for the "TUI exits after one slash command" bug: on
    /// resume the render task must NOT die if the terminal's cursor-position
    /// query fails. We suspend → resume with a backend whose cursor query always
    /// errors, then prove the loop is still live by folding a post-resume event
    /// and observing a clean exit only when the render channel is closed.
    #[tokio::test]
    async fn resume_survives_cursor_position_query_failure() {
        use tokio::sync::oneshot;

        let backend = CursorQueryFailsBackend {
            inner: ratatui::backend::TestBackend::new(40, 8),
        };
        let terminal = ratatui::Terminal::new(backend).unwrap();
        let (render_tx, render_rx) = mpsc::unbounded_channel::<RenderEvent>();
        let (action_tx, _action_rx) = mpsc::unbounded_channel::<UserAction>();
        let (control_tx, control_rx) = mpsc::unbounded_channel::<RenderControl>();
        let cancel = Arc::new(AtomicBool::new(false));
        let mut state = AppState::new();
        state.apply(RenderEvent::StatusSnapshot(sample_status()));

        let handle = tokio::spawn(render_task(
            terminal,
            state,
            render_rx,
            action_tx,
            Arc::clone(&cancel),
            control_rx,
            futures_util::stream::pending::<std::io::Result<Event>>,
        ));

        // Suspend → resume around a (simulated) slash command.
        let (ack_tx, ack_rx) = oneshot::channel::<()>();
        let (resume_tx, resume_rx) = oneshot::channel::<()>();
        control_tx
            .send(RenderControl::Suspend {
                ack: ack_tx,
                resume: resume_rx,
            })
            .unwrap();
        ack_rx.await.expect("render task should ack the park");
        resume_tx.send(()).unwrap();

        // If the resume repaint queried the cursor (the old `clear` path) the
        // task would already be dead. Prove it is still draining events.
        render_tx
            .send(RenderEvent::Status("after resume".into()))
            .unwrap();
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
        drop(render_tx);
        let joined = handle.await.expect("join");
        assert!(
            joined.is_ok(),
            "render task must survive resume even when the cursor-position query fails"
        );
    }
}
