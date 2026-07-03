use super::*;

/// Returns the rationale text for an explain-tools surface, in order:
///   1. the assistant message that accompanied the tool call (if non-empty)
///   2. the tool's manifest `description` from MCP
///   3. `(no description)` as a final fallback
pub(crate) fn resolve_tool_rationale(
    assistant_text: &str,
    tool_name: &str,
    mcp: Option<&McpManager>,
) -> String {
    let trimmed = assistant_text.trim();
    if !trimmed.is_empty() {
        return trimmed.to_string();
    }
    if let Some(mgr) = mcp {
        if let Some(tool) = mgr.list_tools().into_iter().find(|t| t.name == tool_name) {
            let desc = tool.description.trim();
            if !desc.is_empty() {
                return desc.to_string();
            }
        }
    }
    "(no description)".to_string()
}

/// Cooperative-cancel poll for the opt-in TUI. Resolves once `flag` becomes
/// `true`. Under raw mode the terminal's ISIG is disabled, so
/// `tokio::signal::ctrl_c()` never fires; the TUI render task sets this flag on
/// Ctrl-C and this future lets `agent_turn`'s `select!` observe it as an extra
/// cancel branch. In the non-TUI path the flag is never set, so the branch
/// simply never resolves and behavior is unchanged.
async fn wait_for_cancel(flag: &std::sync::atomic::AtomicBool) {
    while !flag.load(std::sync::atomic::Ordering::SeqCst) {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

fn subprocess_step_cap_events(
    message: &str,
    steps_used: usize,
    stats: &ChatStats,
    window: Option<usize>,
) -> Vec<serde_json::Value> {
    let mut error = crate::json_events::error(message);
    if let Some(obj) = error.as_object_mut() {
        obj.insert("steps_used".to_string(), serde_json::json!(steps_used));
    }
    let final_event = serde_json::json!({
        "type": "final",
        "value": message,
        "steps_used": steps_used,
        "error": true,
    });
    let done = subprocess_done_event(stats, window, steps_used);
    vec![final_event, error, done]
}

fn format_time_cap(duration: Duration) -> String {
    let millis = duration.as_millis();
    if millis >= 1000 && millis % 1000 == 0 {
        format!("{}s", millis / 1000)
    } else {
        format!("{millis}ms")
    }
}

fn subprocess_done_event(
    stats: &ChatStats,
    window: Option<usize>,
    steps_used: usize,
) -> serde_json::Value {
    let mut done = crate::json_events::done_with_window(stats, window);
    if let Some(obj) = done.as_object_mut() {
        obj.insert("steps_used".to_string(), serde_json::json!(steps_used));
    }
    done
}

impl ChatCLI {
    pub(crate) fn build_turn_tool_specs(&self) -> Option<Vec<crate::ollama::ToolSpec>> {
        let tools = self
            .mcp_manager
            .as_ref()
            .and_then(agent_loop::build_tool_specs);
        Self::filter_turn_tool_specs_for_subagent_mode(self.subprocess_subagent_mode, tools)
    }

    pub(crate) fn agent_step_cap(&self) -> usize {
        self.max_agent_steps_override
            .unwrap_or(MAX_AGENT_STEPS)
            .clamp(1, MAX_AGENT_STEPS)
    }

    fn subprocess_time_cap_error(
        &self,
        cap: Duration,
        steps_used: usize,
        stats: &ChatStats,
        json_enabled: bool,
    ) -> anyhow::Error {
        let message = format!("subagent time cap reached after {}", format_time_cap(cap));
        if self.subprocess_subagent_mode && json_enabled {
            let window = crate::llm::context_window_for_model(self.executor.get_model());
            for event in subprocess_step_cap_events(&message, steps_used, stats, window) {
                Self::emit_json_event(event);
            }
        }
        exit_code::err(ExitCode::Tool, format!("cubi: {message}"))
    }

    fn meta_tool_blocked_in_subagent_mode(&self, tool_name: &str) -> Option<String> {
        if !self.subprocess_subagent_mode {
            return None;
        }
        match tool_name {
            AGENT_TOOL_NAME => Some("[tool error] nested `agent_run` is not allowed".to_string()),
            crate::agent_loop::CONSENSUS_TOOL_NAME => {
                Some("[tool error] nested `consensus_run` is not allowed".to_string())
            }
            _ => None,
        }
    }

    pub(crate) fn filter_turn_tool_specs_for_subagent_mode(
        subprocess_subagent_mode: bool,
        tools: Option<Vec<crate::ollama::ToolSpec>>,
    ) -> Option<Vec<crate::ollama::ToolSpec>> {
        if subprocess_subagent_mode {
            agent_loop::without_meta_tools(tools)
        } else {
            tools
        }
    }

    fn prompt_tool_approval(&mut self, tool_name: &str) -> bool {
        let cwd = std::env::current_dir().ok();
        let trusted = cwd
            .as_deref()
            .map(|path| {
                self.permissions
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .contains(path)
            })
            .unwrap_or(false);
        if trusted || self.approved_tools.contains(tool_name) {
            return true;
        }
        if self.headless_mode {
            self.emit_status(format!(
                "⚠ Tool `{}` wants to run; denying in headless mode.",
                tool_name
            ));
            return false;
        }
        // In the full-screen TUI, an interactive stdin prompt would fight the
        // render task (a second reader on the terminal) and corrupt the alt
        // screen. Auto-deny using the same non-interactive path as headless;
        // the notice is surfaced through the sink so it lands in the transcript.
        if self.tui_active {
            self.emit_status(format!(
                "⚠ Tool `{}` wants to run; auto-denied in TUI mode (approval prompt unavailable).",
                tool_name
            ));
            return false;
        }

        use std::io::{self, Write};
        print!(
            "⚠ Tool `{}` wants to run. Allow? [y/N/a(lways)] ",
            tool_name
        );
        let _ = io::stdout().flush();
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            return false;
        }
        match input.trim().to_ascii_lowercase().as_str() {
            "y" => true,
            "a" => {
                self.approved_tools.insert(tool_name.to_string());
                true
            }
            _ => false,
        }
    }

    async fn execute_tool_call(&mut self, call: &ToolCall) -> String {
        // Fire PreToolUse hook — may deny the call.
        use crate::hooks::HookDecision;
        let hook_decision = self
            .hooks
            .fire_pre_tool_use(&call.function.name, &call.function.arguments);

        if let HookDecision::Deny(reason) = hook_decision {
            let msg = format!(
                "  {} {}",
                "✗".bright_red(),
                "denied by PreToolUse hook".bright_red()
            );
            self.emit_status(msg);
            format!("[tool denied] {reason}")
        } else if self.policy.is_denied(&call.function.name) {
            format!(
                "[tool denied] `{}` is blocked by admin policy",
                call.function.name
            )
        } else if !self
            .permissions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .check_tool_allowed(&call.function.name)
        {
            format!(
                "[tool denied] `{}` is blocked by your tool permissions",
                call.function.name
            )
        } else if !self.prompt_tool_approval(&call.function.name) {
            format!(
                "[tool denied] user declined approval for `{}`",
                call.function.name
            )
        } else if call.function.name == AGENT_TOOL_NAME && agent_loop::meta_tools_disabled_by_env()
        {
            "[tool error] nested `agent_run` is not allowed".to_string()
        } else if let Some(error) = self.meta_tool_blocked_in_subagent_mode(&call.function.name) {
            error
        } else if call.function.name == AGENT_TOOL_NAME {
            // Meta-tool: spawn a focused subagent with fresh context. Handled
            // here (not in `McpManager`) because it needs the executor to
            // drive its own inner loop.
            let goal = call.function.arguments["goal"]
                .as_str()
                .unwrap_or("")
                .to_string();
            let max_steps = call.function.arguments["max_steps"]
                .as_u64()
                .map(|n| n as usize)
                .unwrap_or(SUBAGENT_DEFAULT_STEPS);
            if goal.is_empty() {
                "[tool error] `agent_run` requires a non-empty `goal`".to_string()
            } else {
                let start_msg = format!(
                    "  {} subagent goal: {}",
                    "↳".bright_magenta(),
                    goal.chars().take(120).collect::<String>().bright_white()
                );
                self.emit_status(start_msg);
                match agent_loop::run_subagent(
                    &self.executor,
                    &mut self.mcp_manager,
                    &goal,
                    max_steps,
                )
                .await
                {
                    Ok(report) => {
                        let done_msg = format!("  {} subagent done", "↳".bright_magenta());
                        self.emit_status(done_msg);
                        report
                    }
                    Err(e) => format!("[tool error] subagent failed: {e}"),
                }
            }
        } else if call.function.name == crate::agent_loop::CONSENSUS_TOOL_NAME
            && agent_loop::meta_tools_disabled_by_env()
        {
            "[tool error] nested `consensus_run` is not allowed".to_string()
        } else if call.function.name == crate::agent_loop::CONSENSUS_TOOL_NAME {
            // Meta-tool: spawn N consensus subagents in parallel. The
            // subagent token usage is folded into `session_stats` here
            // so `/cost` reflects the full bill.
            match self.execute_consensus_tool_call(call).await {
                Ok(text) => text,
                Err(e) => format!("[tool error] consensus_run failed: {e}"),
            }
        } else {
            match self.mcp_manager.as_mut() {
                Some(mcp) => match mcp
                    .call_tool(&call.function.name, call.function.arguments.clone())
                    .await
                {
                    Ok(r) => agent_loop::render_tool_result(&r),
                    Err(e) => {
                        if let Some(timeout) =
                            e.downcast_ref::<crate::mcp_manager::ToolTimeoutError>()
                        {
                            Self::emit_json_event_if(
                                self.json_enabled && self.headless_mode,
                                crate::json_events::tool_timeout(&timeout.name, timeout.secs),
                            );
                            format!("[tool error] {timeout}")
                        } else {
                            format!("[tool error] {e}")
                        }
                    }
                },
                None => format!(
                    "[tool error] no MCP manager available to execute `{}`",
                    call.function.name
                ),
            }
        }
    }

    /// Drives one user turn through the native tool-calling agent loop.
    ///
    /// * Streams the assistant's tokens to stdout as they arrive.
    /// * If the model returns `tool_calls`, executes each one through the
    ///   [`McpManager`] (which routes built-in vs. external tools), appends
    ///   the result as a `role:"tool"` message, and loops.
    /// * Honors plan mode at the call site: write/exec tools already refuse
    ///   in plan mode (see `BuiltinToolRegistry`), so the loop simply
    ///   reflects those refusals back to the model and lets it adapt.
    /// * Caps iterations at [`MAX_AGENT_STEPS`] so a confused model can't
    ///   spin forever.
    ///
    /// On a successful exchange (any number of tool round-trips), the final
    /// assistant message is appended to history, single-turn system hints
    /// are stripped, and the session is checkpointed.
    pub(super) async fn agent_turn(&mut self, turn_start: usize) -> Result<()> {
        // Auto-compact: when the prompt token estimate crosses
        // `compact_threshold_pct` of the active model's context window,
        // summarize older turns before sending. Best-effort; failures
        // are logged but never abort the turn — we'd rather attempt the
        // request than refuse silently.
        self.maybe_auto_compact().await;

        // Reset the cooperative TUI cancel flag at the start of each turn so a
        // stale Ctrl-C from a prior turn can't abort this one. No-op for the
        // non-TUI path (the flag is only ever set by the TUI render task).
        self.cancel
            .store(false, std::sync::atomic::Ordering::SeqCst);
        let cancel = std::sync::Arc::clone(&self.cancel);

        // Hard stop: refuse to send when the estimated prompt would
        // still exceed the model's context window. Keeps the user's
        // last turn in history so they can /compact, /pin less, or
        // switch model and retry from the same state.
        if self.check_token_budget(turn_start)? {
            return Ok(());
        }

        // Build the tool list once per turn. Older / non-tool-capable models
        // ignore it silently, so this is safe to always send.
        let tools = self.build_turn_tool_specs();

        // Track whether we've printed visible content on this turn so we
        // can choose when to add separating blank lines.
        let mut any_output = false;
        // Per-turn provider-usage accumulator. Summed into `session_stats`
        // at the very end so cancelled or errored turns don't pollute totals.
        let mut turn_stats = ChatStats::default();
        let headless_mode = self.headless_mode;
        let json_enabled = self.json_enabled && headless_mode;

        // Re-sync the UI sink's output-mode flags for this turn. `headless`
        // is flipped on by `run_one_shot` *after* construction and markdown /
        // json can be toggled mid-session via slash commands, so the sink
        // must observe the current values before any per-turn output.
        self.ui.reconfigure(super::ui_sink::SinkFlags {
            headless: headless_mode,
            json: json_enabled,
            markdown: self.markdown_enabled,
        });

        // Consecutive tool-error counter for the headless safety valve.
        // Reset to 0 on any successful tool call; when it reaches
        // MAX_CONSECUTIVE_TOOL_ERRORS the model is treated as stuck.
        let mut consecutive_tool_errors: u32 = 0;

        if let Some(sink) = self.event_sink.as_ref() {
            sink.emit(
                "turn_start",
                serde_json::json!({
                    "turn": self.usage_history.len() + 1,
                    "model": self.executor.get_model(),
                }),
            );
        }

        // Receipts: capture the user message that opens this turn.
        if let Some(msg) = self.history.get(turn_start) {
            if msg.role == "user" {
                self.emit_receipt(
                    crate::receipts::ReceiptEvent::UserMessage,
                    &serde_json::json!({"text": msg.content}),
                );
            }
        }

        // Snapshot the journal start so /rewind still works after Ctrl-C
        // cancel: we leave file edits in place but pop the history.
        // (turn_start was captured before the user message was pushed.)

        let max_agent_steps = self.agent_step_cap();
        let subprocess_time_cap = if self.subprocess_subagent_mode {
            self.max_agent_time_cap_override
        } else {
            None
        };
        let subprocess_deadline = subprocess_time_cap.map(|cap| tokio::time::Instant::now() + cap);
        for step in 0..max_agent_steps {
            if let Some(deadline) = subprocess_deadline {
                if let Some(cap) = subprocess_time_cap {
                    if tokio::time::Instant::now() >= deadline {
                        return Err(self.subprocess_time_cap_error(
                            cap,
                            step,
                            &turn_stats,
                            json_enabled,
                        ));
                    }
                }
            }
            // Show a spinner while we wait for the model's first token.
            // Label differs by step so the user can tell "first response"
            // from "reacting to tool output".
            let spinner_label = if step == 0 {
                "thinking…"
            } else {
                "processing tool results…"
            };
            // Start the thinking spinner through the sink. `continuation`
            // (step > 0) means the `AI:` prefix was already printed earlier in
            // this multi-step turn, so it is not repeated. The sink owns the
            // spinner across the streaming call so its first-token handler can
            // clear the status line at exactly the right moment.
            self.ui.begin_thinking(spinner_label, step > 0);

            // Either stream tokens (default) or run buffered chat_with_tools
            // when the user has /stream off (markdown mode is buffered-only).
            // Cancellation: SIGINT (Ctrl-C) interrupts the in-flight model
            // call; tool dispatch below is also cancellable between awaits.
            let mut hit_time_cap = false;
            let outcome: Option<Result<(Message, ChatStats)>> = if self.stream_enabled {
                // Move the sink out of `self` for the duration of the stream
                // so the token closure can own `&mut` it while `self.executor`
                // is borrowed by `chat_stream`. Restored immediately after.
                let mut ui = std::mem::replace(&mut self.ui, Box::new(super::ui_sink::NullSink));
                let stream_fut =
                    self.executor
                        .chat_stream(self.history.clone(), tools.clone(), |tok| {
                            ui.assistant_token(tok);
                        });
                let res = if let Some(deadline) = subprocess_deadline {
                    tokio::select! {
                        biased;
                        _ = tokio::signal::ctrl_c() => None,
                        _ = wait_for_cancel(&cancel) => None,
                        _ = tokio::time::sleep_until(deadline) => {
                            hit_time_cap = true;
                            None
                        },
                        r = stream_fut => Some(r),
                    }
                } else {
                    tokio::select! {
                        biased;
                        _ = tokio::signal::ctrl_c() => None,
                        _ = wait_for_cancel(&cancel) => None,
                        r = stream_fut => Some(r),
                    }
                };
                self.ui = ui;
                res
            } else {
                let buf_fut = self
                    .executor
                    .chat_with_tools(self.history.clone(), tools.clone());
                if let Some(deadline) = subprocess_deadline {
                    tokio::select! {
                        biased;
                        _ = tokio::signal::ctrl_c() => None,
                        _ = wait_for_cancel(&cancel) => None,
                        _ = tokio::time::sleep_until(deadline) => {
                            hit_time_cap = true;
                            None
                        },
                        r = buf_fut => Some(r),
                    }
                } else {
                    tokio::select! {
                        biased;
                        _ = tokio::signal::ctrl_c() => None,
                        _ = wait_for_cancel(&cancel) => None,
                        r = buf_fut => Some(r),
                    }
                }
            };
            // Always retire the spinner before we print anything else.
            if let Some(sp) = self.ui.end_thinking() {
                sp.stop().await;
            }
            let got_token = self.ui.got_token();

            if hit_time_cap {
                if got_token && headless_mode && !json_enabled {
                    println!();
                }
                let cap = subprocess_time_cap.unwrap_or(Duration::from_millis(0));
                return Err(self.subprocess_time_cap_error(
                    cap,
                    step + 1,
                    &turn_stats,
                    json_enabled,
                ));
            }

            // None == user pressed Ctrl-C mid-call. Truncate the history
            // back to the pre-turn snapshot so the conversation has no
            // dangling user / orphaned tool turns. File edits from prior
            // steps in this turn (if any) survive: use `/rewind` to undo
            // them.
            let Some(stream_result) = outcome else {
                if got_token && !self.tui_active {
                    // Move past any partial output.
                    println!();
                }
                self.ui.status(&format!(
                    "{} {}",
                    "✗".bright_red(),
                    "cancelled (Ctrl-C)".bright_red()
                ));
                if step > 0 {
                    let msg = format!(
                        "  {} prior tool side-effects in this turn may have run; \
                         `/rewind` can undo file edits.",
                        "ℹ".bright_blue()
                    );
                    self.ui.status(&msg);
                }
                // Drop the journal bucket that `run()` opened for this
                // turn if no tools have actually snapshotted anything yet,
                // so `/rewind 1` still targets the previous completed
                // turn instead of an empty bucket.
                self.journal.discard_last_turn_if_empty();
                self.history.truncate(turn_start);
                if headless_mode {
                    return Err(exit_code::err(
                        ExitCode::Cancelled,
                        "cubi: cancelled (Ctrl-C)",
                    ));
                }
                return Ok(());
            };
            let (msg, stats) = stream_result?;
            turn_stats.add(&stats);

            if got_token {
                any_output = true;
            }

            // Some backends (older Ollama) don't supply an `id` on each
            // tool_call. Synthesize a stable, position-based id so the
            // assistant message and its tool-result messages reference
            // the same id — strict OpenAI-compatible validators require
            // this.
            let mut msg = msg;
            if let Some(tcs) = msg.tool_calls.as_mut() {
                for (i, c) in tcs.iter_mut().enumerate() {
                    if c.id.is_none() {
                        c.id = Some(format!("call_{}_{}", i, c.function.name));
                    }
                }
            }
            let calls = msg.tool_calls.clone().unwrap_or_default();
            // Capture the final-content text BEFORE moving the message into
            // history — used by the markdown re-render below.
            let msg_content = msg.content.clone();

            // Persist the assistant message verbatim — including any
            // tool_calls — so the next iteration's context matches what the
            // model sent us.
            self.history.push(msg);

            if calls.is_empty() {
                // Receipts: final assistant content for this turn.
                // Filtered through `<think>` stripping so we never
                // commit chain-of-thought to a long-lived audit log.
                {
                    let filtered = crate::thinking_filter::strip_thinking_blocks(&msg_content);
                    self.emit_receipt(
                        crate::receipts::ReceiptEvent::AssistantMessage,
                        &serde_json::json!({"text": filtered}),
                    );
                }
                // Plain text response: we're done with this turn. Some
                // providers put the completed message only in the final chunk;
                // print it here if the streaming callback saw no tokens.
                if self.stream_enabled && !got_token && !msg_content.is_empty() {
                    if json_enabled {
                        Self::emit_json_event(crate::json_events::token(&msg_content));
                    } else if headless_mode {
                        println!("{msg_content}");
                    } else if self.tui_active {
                        // Route through the sink so the fallback reply lands in
                        // the transcript instead of scribbling on the alt screen.
                        self.ui.assistant_final(&msg_content);
                    } else {
                        print!("{} ", "AI:".bright_blue().bold());
                        println!("{}", msg_content.bright_white());
                    }
                    any_output = true;
                }
                if !self.stream_enabled && !msg_content.is_empty() {
                    if json_enabled {
                        Self::emit_json_event(crate::json_events::token(&msg_content));
                    } else {
                        // Buffered mode: render the message now. Markdown if
                        // enabled, otherwise plain text.
                        self.ui.assistant_final(&msg_content);
                    }
                    any_output = true;
                }
                let window = crate::llm::context_window_for_model(self.executor.get_model());
                let done = if self.subprocess_subagent_mode {
                    subprocess_done_event(&turn_stats, window, step + 1)
                } else {
                    crate::json_events::done_with_window(&turn_stats, window)
                };
                Self::emit_json_event_if(json_enabled, done);
                if any_output && headless_mode && !json_enabled {
                    // Streaming prints tokens via `print!` with no
                    // trailing newline; emit exactly one for piping.
                    // The buffered and stream-fallback paths above
                    // already used `println!`, so don't double-up.
                    if got_token {
                        println!();
                    }
                } else if any_output && !json_enabled && self.tui_active {
                    // TUI owns its own inter-turn spacing; a raw newline here
                    // would land on the alternate screen.
                } else if any_output && !json_enabled {
                    println!("\n");
                } else if !any_output && !json_enabled && self.tui_active {
                    // Surface the "no response" hint through the sink so it
                    // reaches the transcript rather than raw stdout.
                    self.ui.status(
                        "(no response — try rephrasing, switching model, or running /usage to check the context budget)",
                    );
                } else if !any_output && !json_enabled && !headless_mode {
                    // The model returned no content and no tool calls.
                    // Without this, the REPL would silently re-prompt
                    // and the user would have no idea their last turn
                    // produced nothing. Surface a dim hint instead.
                    println!(
                        "{} {}",
                        "AI:".bright_blue().bold(),
                        "(no response — try rephrasing, switching model, or running /usage to check the context budget)"
                            .bright_black()
                    );
                } else if !any_output && !json_enabled && headless_mode {
                    // Headless one-shot also produced nothing. Route a
                    // short warning to stderr so a `cubi -p ... | tool`
                    // pipeline keeps stdout empty (truthful: zero bytes
                    // of model output) while still telling the user the
                    // turn completed without a reply.
                    eprintln!(
                        "cubi: model returned no content; try rephrasing or switching model."
                    );
                }
                break;
            }

            // The model asked us to run one or more tools. Visually break
            // the stream so the user can tell the tools apart from the
            // model's prose.
            if any_output && !self.tui_active {
                println!();
            }

            for (idx, call) in calls.iter().enumerate() {
                Self::emit_json_event_if(
                    json_enabled,
                    crate::json_events::tool_call(&call.function.name, &call.function.arguments),
                );
                // In TUI mode the framed tool block (opened by
                // `tool_started` below) conveys the tool name, so skip the
                // plain `⚙ tool:` status line to avoid a duplicate. Non-TUI
                // keeps it exactly.
                if !self.tui_active {
                    self.emit_status(format!(
                        "{} {} {}",
                        "⚙".bright_blue(),
                        "tool:".bright_blue(),
                        call.function.name.bright_cyan()
                    ));
                }

                // `--explain-tools` (or CUBI_EXPLAIN_TOOLS=1): surface the
                // rationale for invoking this tool. In headless+JSON mode
                // emit a `tool_rationale` event; otherwise print one dim
                // line on stderr so prose output stays clean.
                if self.explain_tools_enabled {
                    let rationale = resolve_tool_rationale(
                        &msg_content,
                        &call.function.name,
                        self.mcp_manager.as_ref(),
                    );
                    if json_enabled {
                        Self::emit_json_event(serde_json::json!({
                            "type": "tool_rationale",
                            "tool": call.function.name,
                            "rationale": rationale,
                        }));
                    } else if !headless_mode {
                        eprintln!(
                            "{} {}: {}",
                            "↳".bright_black(),
                            call.function.name.bright_black(),
                            rationale.bright_black()
                        );
                    }
                    if let Some(sink) = self.event_sink.as_ref() {
                        sink.emit(
                            "tool_rationale",
                            serde_json::json!({
                                "tool": call.function.name,
                                "rationale": rationale,
                            }),
                        );
                    }
                }

                if let Some(sink) = self.event_sink.as_ref() {
                    let mut args = call.function.arguments.clone();
                    crate::trace_tools::redact_secrets(&mut args);
                    sink.emit(
                        "tool_call_start",
                        serde_json::json!({
                            "tool": call.function.name,
                            "args": args,
                        }),
                    );
                }

                // Receipts: capture the tool call before dispatch. The
                // payload is the *raw* args (post-redaction is a
                // separate concern handled by the event sink); the
                // receipts side-channel is meant for cryptographic
                // provenance, so we preserve the original arguments.
                self.emit_receipt(
                    crate::receipts::ReceiptEvent::ToolCall {
                        name: call.function.name.clone(),
                    },
                    &serde_json::json!({
                        "name": call.function.name,
                        "args": call.function.arguments,
                        "call_id": call.id,
                    }),
                );

                // --trace-tools: record tool_start / tool_complete
                // pairs around the dispatch. Best-effort: tracer write
                // failures only log a warning.
                let trace_ctx = self.tool_tracer.as_ref().map(|t| {
                    let id = t.next_call_id();
                    t.log_start(&call.function.name, &id, &call.function.arguments);
                    (Arc::clone(t), id, std::time::Instant::now())
                });

                let mut hit_time_cap = false;
                // Open a framed tool block in the TUI (no-op for LineSink, so
                // non-TUI output is byte-identical) and start the wall-clock
                // used for the trailing ✓/✗ status row's elapsed time.
                self.ui.tool_started(&call.function.name);
                let tool_started_at = std::time::Instant::now();
                let result_text = {
                    // Tool spinner: gives users a visible heartbeat
                    // (with elapsed seconds) while a slow tool runs.
                    // Suppressed in JSON mode so machine-parsed event
                    // streams stay clean, and in `--tui` mode where the
                    // spinner would paint braille frames to stderr over
                    // the alternate screen. In TUI mode tool activity is
                    // still conveyed by the `⚙ tool:` line emitted above
                    // via `self.emit_status(...)`, which flows through the
                    // sink into the transcript. Passing `true` here yields
                    // a no-op spinner, so the later `finish()` remains
                    // valid. Non-TUI behavior is unchanged.
                    let sp = super::spinner::ToolSpinner::start_with_mode(
                        call.function.name.clone(),
                        json_enabled || self.tui_active,
                    );
                    let tool_fut = self.execute_tool_call(call);
                    tokio::pin!(tool_fut);
                    let r = if let Some(deadline) = subprocess_deadline {
                        tokio::select! {
                            biased;
                            _ = tokio::signal::ctrl_c() => None,
                            _ = wait_for_cancel(&cancel) => None,
                            _ = tokio::time::sleep_until(deadline) => {
                                hit_time_cap = true;
                                None
                            },
                            r = &mut tool_fut => Some(r),
                        }
                    } else {
                        tokio::select! {
                            biased;
                            _ = tokio::signal::ctrl_c() => None,
                            _ = wait_for_cancel(&cancel) => None,
                            r = &mut tool_fut => Some(r),
                        }
                    };
                    sp.finish().await;
                    r
                };
                let Some(result_text) = result_text else {
                    // Close the framed tool block opened by `tool_started`
                    // above with a failure status (no-op for LineSink/NullSink,
                    // so non-TUI output stays byte-identical). Without this the
                    // TUI transcript keeps a dangling `⚙` header that never
                    // gets a ✓/✗ status row when a tool is cancelled/times out.
                    self.ui.tool_finished(
                        &call.function.name,
                        false,
                        "cancelled (Ctrl-C)",
                        tool_started_at.elapsed().as_millis() as u64,
                    );
                    if let Some((tracer, id, started)) = trace_ctx {
                        tracer.log_complete(
                            &call.function.name,
                            &id,
                            false,
                            started.elapsed().as_millis(),
                            0,
                        );
                    }
                    self.cancel_tool_calls(turn_start, &calls, idx);
                    if hit_time_cap {
                        let cap = subprocess_time_cap.unwrap_or(Duration::from_millis(0));
                        return Err(self.subprocess_time_cap_error(
                            cap,
                            step + 1,
                            &turn_stats,
                            json_enabled,
                        ));
                    }
                    if headless_mode {
                        return Err(exit_code::err(
                            ExitCode::Cancelled,
                            "cubi: cancelled (Ctrl-C)",
                        ));
                    }
                    return Ok(());
                };

                if let Some((tracer, id, started)) = trace_ctx {
                    let is_err = result_text.starts_with("[tool error]")
                        || result_text.starts_with("[tool denied]");
                    tracer.log_complete(
                        &call.function.name,
                        &id,
                        !is_err,
                        started.elapsed().as_millis(),
                        result_text.chars().count(),
                    );
                }

                if let Some(sink) = self.event_sink.as_ref() {
                    let is_err = result_text.starts_with("[tool error]")
                        || result_text.starts_with("[tool denied]");
                    sink.emit(
                        "tool_call_complete",
                        serde_json::json!({
                            "tool": call.function.name,
                            "ok": !is_err,
                            "result_chars": result_text.chars().count(),
                        }),
                    );
                }

                // Receipts: capture the tool result. Stores the full
                // (potentially large) text in the payload sidecar so
                // the JSONL line stays a single line.
                {
                    let is_err = result_text.starts_with("[tool error]")
                        || result_text.starts_with("[tool denied]");
                    self.emit_receipt(
                        crate::receipts::ReceiptEvent::ToolResult {
                            name: call.function.name.clone(),
                            ok: !is_err,
                        },
                        &serde_json::json!({
                            "name": call.function.name,
                            "ok": !is_err,
                            "result": result_text,
                            "call_id": call.id,
                        }),
                    );
                }

                // Fire PostToolUse hook.
                let is_error = result_text.starts_with("[tool error]")
                    || result_text.starts_with("[tool denied]");
                self.hooks
                    .fire_post_tool_use(&call.function.name, &result_text, is_error);

                // Tool errors are a normal part of an agentic loop: the model
                // is expected to read the error and retry (fix a path, re-read
                // a file, correct `old_text`, ...). Feed the error back into
                // history (below) and continue rather than aborting the whole
                // run on the first miss. As a safety valve, headless runs still
                // bail out once the model emits MAX_CONSECUTIVE_TOOL_ERRORS
                // errors in a row with no successful call in between (genuinely
                // stuck), so scripts get a non-zero exit instead of burning the
                // full step budget.
                if result_text.starts_with("[tool error]") {
                    consecutive_tool_errors += 1;
                    if headless_mode && consecutive_tool_errors >= MAX_CONSECUTIVE_TOOL_ERRORS {
                        // Record the final error in history so an inspected or
                        // resumed session shows what happened, then abort.
                        self.history.push(Message::tool_result(
                            &call.function.name,
                            result_text.clone(),
                            call.id.clone(),
                        ));
                        return Err(exit_code::err(
                            ExitCode::Tool,
                            format!(
                                "cubi: aborting after {} consecutive tool errors; \
                                 last: tool '{}' failed: {}",
                                consecutive_tool_errors, call.function.name, result_text
                            ),
                        ));
                    }
                } else {
                    consecutive_tool_errors = 0;
                }

                // Print a short preview so the user can see what came back
                // without us dumping a 10 KB log into the terminal. In TUI
                // mode the framed tool block renders the (capped) output and
                // ✓/✗ status instead, so skip this dim preview line there.
                if !self.tui_active {
                    let preview: String = result_text.chars().take(400).collect();
                    let ellipsis = if result_text.len() > preview.len() {
                        " …"
                    } else {
                        ""
                    };
                    self.emit_status(format!(
                        "  {}{}",
                        preview.bright_black(),
                        ellipsis.bright_black()
                    ));
                }
                // Close the framed tool block in the TUI (no-op for LineSink).
                // `ok` mirrors the error detection used above; the output is
                // capped so a huge tool result doesn't bloat the event/channel.
                let tool_ok = !result_text.starts_with("[tool error]")
                    && !result_text.starts_with("[tool denied]");
                let tool_output: String = result_text.chars().take(2000).collect();
                self.ui.tool_finished(
                    &call.function.name,
                    tool_ok,
                    &tool_output,
                    tool_started_at.elapsed().as_millis() as u64,
                );
                Self::emit_json_event_if(
                    json_enabled,
                    crate::json_events::tool_result(&call.function.name, &result_text),
                );

                self.history.push(Message::tool_result(
                    &call.function.name,
                    result_text,
                    call.id.clone(),
                ));
            }

            // Loop back: feed the tool outputs into the next model call.
            any_output = false;

            // Diagnostic if we hit the cap mid-loop. The body of the loop
            // executed the tools for this step; if `step + 1 == MAX`, the
            // next call_stream is what we're skipping, so warn here.
            if step + 1 == max_agent_steps {
                let message = format!(
                    "agent loop hit step cap ({max_agent_steps}) with pending tool calls; \
                     stopping before a final assistant reply"
                );
                eprintln!("{} {}", "Warn:".bright_yellow(), message);
                if self.subprocess_subagent_mode && json_enabled {
                    let window = crate::llm::context_window_for_model(self.executor.get_model());
                    for event in subprocess_step_cap_events(
                        &format!("subagent {message}"),
                        max_agent_steps,
                        &turn_stats,
                        window,
                    ) {
                        Self::emit_json_event(event);
                    }
                }
            }
        }

        // Per-turn footer (opt-in via `/stats-footer on`). Skipped when the
        // provider returned nothing useful or when `--quiet` is set.
        if self.stats_footer_enabled && !self.quiet_mode && !turn_stats.is_empty() {
            let window = crate::llm::context_window_for_model(self.executor.get_model());
            self.ui.usage_footer(&turn_stats, window);
        }
        // Roll the per-turn usage into the run-total. Done last so an early
        // cancel return (above) doesn't poison the counter.
        self.session_stats.add(&turn_stats);
        // Retain a copy of the per-turn stats so `/usage` and the optional
        // usage footer can render without re-querying the provider.
        self.usage_history.push(turn_stats.clone());
        // Optional one-line usage footer (suppressed in headless/JSON and
        // when `--quiet` is set).
        if self.usage_footer_enabled
            && !self.headless_mode
            && !self.json_enabled
            && !self.quiet_mode
            && !self.tui_active
            && !turn_stats.is_empty()
        {
            let pricing = crate::pricing::lookup(self.executor.get_model());
            let cost = crate::pricing::format_cost(
                pricing,
                turn_stats.prompt_tokens,
                turn_stats.completion_tokens,
            );
            let line = super::format_usage_footer_line(&turn_stats, &self.session_stats, &cost);
            println!("{}", line.bright_black());
        }
        // Emit the turn-end event to the configured `--events` sink, if any.
        if let Some(sink) = self.event_sink.as_ref() {
            sink.emit(
                "turn_end",
                serde_json::json!({
                    "usage": {
                        "prompt_tokens": turn_stats.prompt_tokens,
                        "completion_tokens": turn_stats.completion_tokens,
                        "elapsed_ms": turn_stats.elapsed_ms,
                    },
                    "model": self.executor.get_model(),
                }),
            );
        }

        let pre_counts = self.mcp_counts;
        // Refresh the cached MCP health triple shown in the banner so
        // any Ok↔Failed transitions during tool dispatch are visible
        // on the next prompt redraw.
        self.refresh_mcp_counts();
        if self.mcp_counts != pre_counts {
            if let Some(sink) = self.event_sink.as_ref() {
                sink.emit(
                    "mcp_status_change",
                    serde_json::json!({
                        "before": {
                            "ok": pre_counts.0,
                            "failed": pre_counts.1,
                            "not_loaded": pre_counts.2,
                        },
                        "after": {
                            "ok": self.mcp_counts.0,
                            "failed": self.mcp_counts.1,
                            "not_loaded": self.mcp_counts.2,
                        },
                    }),
                );
            }
        }

        // Drop any system messages tagged as single-turn (e.g. from `/ask`)
        // so they don't keep nudging every subsequent turn.
        self.strip_single_turn_system_messages();

        // Auto-checkpoint the session after every successful turn so a
        // crash never loses the conversation.
        self.checkpoint_session();
        Ok(())
    }

    fn cancel_tool_calls(&mut self, turn_start: usize, _calls: &[ToolCall], current_idx: usize) {
        let msg = format!("{} {}", "✗".bright_red(), "cancelled (Ctrl-C)".bright_red());
        self.emit_status(msg);
        // History is truncated back to `turn_start` to discard the in-flight
        // turn, so we deliberately do not push "[tool cancelled]" markers —
        // they would be dropped immediately by the truncate below.
        self.history.truncate(turn_start);
        self.journal.discard_last_turn_if_empty();
        let caveat = format!(
            "  {} tool future was dropped; subprocesses started by shell-out tools may keep running.",
            "ℹ".bright_blue()
        );
        self.emit_status(caveat);
        // Mirror the model-cancel path: warn that earlier tools in this
        // same turn may have already mutated state, and point at `/rewind`.
        if current_idx > 0 {
            let rewind = format!(
                "  {} prior tool side-effects in this turn may have run; \
                 `/rewind` can undo file edits.",
                "ℹ".bright_blue()
            );
            self.emit_status(rewind);
        }
    }

    /// Backs the `consensus_run` meta-tool. Parses arguments out of
    /// the model's JSON, drives [`consensus::run`], folds subagent
    /// stats into `session_stats`, and returns a text report for the
    /// model to consume on the next turn.
    pub(super) async fn execute_consensus_tool_call(
        &mut self,
        call: &ToolCall,
    ) -> anyhow::Result<String> {
        let args = &call.function.arguments;
        let goal = args
            .get("goal")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if goal.is_empty() {
            anyhow::bail!("missing or empty `goal`");
        }
        let models: Vec<String> = args
            .get("models")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        if models.len() < 2 {
            anyhow::bail!("consensus_run requires at least 2 models");
        }
        let strategy_str = args
            .get("strategy")
            .and_then(|v| v.as_str())
            .unwrap_or("vote");
        let judge_model = args
            .get("judge_model")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        let strategy = match strategy_str {
            "vote" => crate::consensus::ConsensusStrategy::Vote,
            "best-of-n" => crate::consensus::ConsensusStrategy::BestOfN {
                judge_model: judge_model
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("best-of-n requires `judge_model`"))?,
            },
            "judge" => crate::consensus::ConsensusStrategy::Judge {
                judge_model: judge_model
                    .ok_or_else(|| anyhow::anyhow!("judge requires `judge_model`"))?,
            },
            other => anyhow::bail!("unknown strategy `{other}`"),
        };
        let max_steps = args
            .get("max_steps")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .map(crate::consensus::normalize_max_steps_per_subagent)
            .unwrap_or(crate::consensus::CONSENSUS_DEFAULT_MAX_STEPS);
        let concurrency = args
            .get("concurrency")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(0);
        let use_tools = args
            .get("use_tools")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let isolate = args
            .get("isolate")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let isolated_time_cap_secs = args
            .get("isolated_time_cap_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let req = crate::consensus::ConsensusRequest {
            goal: goal.clone(),
            models: models.clone(),
            strategy,
            max_steps_per_subagent: max_steps,
            concurrency,
            use_tools,
            isolate,
            isolated_time_cap_secs,
        };
        if let Some(reason) = self.isolated_tool_consensus_policy_error(&req) {
            anyhow::bail!(reason);
        }

        self.emit_status(format!(
            "  {} consensus over {} models{}: {}",
            "↳".bright_magenta(),
            models.len(),
            if isolate {
                " (tools, isolated)"
            } else if use_tools {
                " (tools, sequential)"
            } else {
                ""
            },
            goal.chars().take(120).collect::<String>().bright_white()
        ));

        let sink = super::CliConsensusSink {
            json_enabled: self.json_enabled && self.headless_mode,
            event_sink: self.event_sink.clone(),
        };
        let result = if use_tools {
            crate::consensus::run_with_tools(req, &self.executor, &sink, &mut self.mcp_manager)
                .await?
        } else {
            crate::consensus::run(req, &self.executor, &sink).await?
        };
        self.session_stats.add(&result.aggregate_stats());
        self.emit_status(format!(
            "  {} consensus winner: {} — {}",
            "↳".bright_magenta(),
            result.winner_model,
            result.decision_reason
        ));

        // Format the tool-result text the model will read next turn.
        let mut report = String::new();
        report.push_str(&format!(
            "Consensus result (strategy: {}; winner: {}):\n",
            super::CliConsensusSink::strategy_label(&result),
            result.winner_model
        ));
        report.push_str(&format!("Decision: {}\n", result.decision_reason));
        for sub in &result.subagent_outputs {
            if let Some(err) = &sub.error {
                report.push_str(&format!("- {} ✗ {}\n", sub.model, err));
            } else {
                report.push_str(&format!(
                    "- {} ✓ ({} tokens, {} tool calls, {} ms)\n",
                    sub.model,
                    sub.prompt_tokens + sub.completion_tokens,
                    sub.tool_calls,
                    sub.elapsed_ms
                ));
            }
        }
        report.push_str("\n--- winning output ---\n");
        report.push_str(&result.winner_output);
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subprocess_step_cap_events_include_final_error_done_and_steps() {
        let stats = ChatStats {
            prompt_tokens: 3,
            completion_tokens: 4,
            elapsed_ms: 5,
        };

        let events = subprocess_step_cap_events("subagent hit cap", 2, &stats, Some(100));

        assert_eq!(events.len(), 3);
        assert_eq!(events[0]["type"], "final");
        assert_eq!(events[0]["value"], "subagent hit cap");
        assert_eq!(events[0]["steps_used"], 2);
        assert_eq!(events[1]["type"], "error");
        assert_eq!(events[1]["message"], "subagent hit cap");
        assert_eq!(events[1]["steps_used"], 2);
        assert_eq!(events[2]["type"], "done");
        assert_eq!(events[2]["stats"]["prompt_tokens"], 3);
        assert_eq!(events[2]["stats"]["completion_tokens"], 4);
        assert_eq!(events[2]["steps_used"], 2);
    }

    #[test]
    fn subprocess_done_event_includes_exact_success_steps() {
        let stats = ChatStats {
            prompt_tokens: 6,
            completion_tokens: 7,
            elapsed_ms: 8,
        };

        let event = subprocess_done_event(&stats, Some(200), 1);

        assert_eq!(event["type"], "done");
        assert_eq!(event["stats"]["prompt_tokens"], 6);
        assert_eq!(event["stats"]["completion_tokens"], 7);
        assert_eq!(event["window"], 200);
        assert_eq!(event["steps_used"], 1);
    }
}
