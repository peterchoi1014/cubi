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

impl ChatCLI {
    fn prompt_tool_approval(&mut self, tool_name: &str) -> bool {
        let cwd = std::env::current_dir().ok();
        let trusted = cwd
            .as_deref()
            .map(|path| self.permissions.lock().unwrap().contains(path))
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
            .unwrap()
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
        use std::io::Write;
        use std::sync::atomic::Ordering;

        // Auto-compact: when the prompt token estimate crosses
        // `compact_threshold_pct` of the active model's context window,
        // summarize older turns before sending. Best-effort; failures
        // are logged but never abort the turn — we'd rather attempt the
        // request than refuse silently.
        self.maybe_auto_compact().await;

        // Hard stop: refuse to send when the estimated prompt would
        // still exceed the model's context window. Keeps the user's
        // last turn in history so they can /compact, /pin less, or
        // switch model and retry from the same state.
        if self.check_token_budget(turn_start)? {
            return Ok(());
        }

        // Build the tool list once per turn. Older / non-tool-capable models
        // ignore it silently, so this is safe to always send.
        let tools = self
            .mcp_manager
            .as_ref()
            .and_then(agent_loop::build_tool_specs);

        // Track whether we've printed visible content on this turn so we
        // can choose when to add separating blank lines.
        let mut any_output = false;
        // Per-turn provider-usage accumulator. Summed into `session_stats`
        // at the very end so cancelled or errored turns don't pollute totals.
        let mut turn_stats = ChatStats::default();
        let headless_mode = self.headless_mode;
        let json_enabled = self.json_enabled && headless_mode;

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

        for step in 0..MAX_AGENT_STEPS {
            // Show a spinner while we wait for the model's first token.
            // Label differs by step so the user can tell "first response"
            // from "reacting to tool output".
            let spinner_label = if step == 0 {
                "thinking…"
            } else {
                "processing tool results…"
            };
            let spinner = crate::spinner::Spinner::start(spinner_label);
            let stop_flag = spinner.stop_flag();

            // Hold the spinner across the streaming call so the callback
            // can clear it on the first token. We can't move `spinner`
            // into the closure (we need to `.stop().await` after), so we
            // share a `&Spinner` via Rc/Arc-like discipline: the closure
            // only needs `clear_line` + `stop_flag`.
            let spinner_ref = &spinner;

            // Only print the "AI:" prefix once per user turn — multi-step
            // turns continue inline below the previous tool block. We
            // defer the print until the first streamed token so the
            // spinner has full ownership of the status line.
            let mut printed_prefix = step > 0;
            let mut got_token = false;

            // Either stream tokens (default) or run buffered chat_with_tools
            // when the user has /stream off (markdown mode is buffered-only).
            // Cancellation: SIGINT (Ctrl-C) interrupts the in-flight model
            // call; tool dispatch below is also cancellable between awaits.
            let outcome: Option<Result<(Message, ChatStats)>> = if self.stream_enabled {
                let stream_fut =
                    self.executor
                        .chat_stream(self.history.clone(), tools.clone(), |tok| {
                            if !got_token {
                                stop_flag.store(true, Ordering::SeqCst);
                                spinner_ref.clear_line();
                                if !printed_prefix && !headless_mode {
                                    print!("{} ", "AI:".bright_blue().bold());
                                    printed_prefix = true;
                                }
                            }
                            if json_enabled {
                                Self::emit_json_event(crate::json_events::token(tok));
                            } else if headless_mode {
                                print!("{}", tok);
                                let _ = std::io::stdout().flush();
                            } else {
                                print!("{}", tok.bright_white());
                                let _ = std::io::stdout().flush();
                            }
                            got_token = true;
                        });
                tokio::select! {
                    biased;
                    _ = tokio::signal::ctrl_c() => None,
                    r = stream_fut => Some(r),
                }
            } else {
                let buf_fut = self
                    .executor
                    .chat_with_tools(self.history.clone(), tools.clone());
                tokio::select! {
                    biased;
                    _ = tokio::signal::ctrl_c() => None,
                    r = buf_fut => Some(r),
                }
            };
            // Always retire the spinner before we print anything else.
            stop_flag.store(true, Ordering::SeqCst);
            spinner.stop().await;

            // None == user pressed Ctrl-C mid-call. Truncate the history
            // back to the pre-turn snapshot so the conversation has no
            // dangling user / orphaned tool turns. File edits from prior
            // steps in this turn (if any) survive: use `/rewind` to undo
            // them.
            let Some(stream_result) = outcome else {
                if got_token {
                    // Move past any partial output.
                    println!();
                }
                crate::out::status_line(
                    headless_mode,
                    format!("{} {}", "✗".bright_red(), "cancelled (Ctrl-C)".bright_red()),
                );
                if step > 0 {
                    let msg = format!(
                        "  {} prior tool side-effects in this turn may have run; \
                         `/rewind` can undo file edits.",
                        "ℹ".bright_blue()
                    );
                    crate::out::status_line(headless_mode, msg);
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
                        self.render_final_reply(&msg_content);
                    }
                    any_output = true;
                }
                let window = crate::llm::context_window_for_model(self.executor.get_model());
                Self::emit_json_event_if(
                    json_enabled,
                    crate::json_events::done_with_window(&turn_stats, window),
                );
                if any_output && headless_mode && !json_enabled {
                    // Streaming prints tokens via `print!` with no
                    // trailing newline; emit exactly one for piping.
                    // The buffered and stream-fallback paths above
                    // already used `println!`, so don't double-up.
                    if got_token {
                        println!();
                    }
                } else if any_output && !json_enabled {
                    println!("\n");
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
            if any_output {
                println!();
            }

            for (idx, call) in calls.iter().enumerate() {
                Self::emit_json_event_if(
                    json_enabled,
                    crate::json_events::tool_call(&call.function.name, &call.function.arguments),
                );
                self.emit_status(format!(
                    "{} {} {}",
                    "⚙".bright_blue(),
                    "tool:".bright_blue(),
                    call.function.name.bright_cyan()
                ));

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

                let result_text = {
                    // Tool spinner: gives users a visible heartbeat
                    // (with elapsed seconds) while a slow tool runs.
                    // Suppressed in JSON mode so machine-parsed event
                    // streams stay clean.
                    let sp = super::spinner::ToolSpinner::start_with_mode(
                        call.function.name.clone(),
                        json_enabled,
                    );
                    let tool_fut = self.execute_tool_call(call);
                    tokio::pin!(tool_fut);
                    let r = tokio::select! {
                        biased;
                        _ = tokio::signal::ctrl_c() => None,
                        r = &mut tool_fut => Some(r),
                    };
                    sp.finish().await;
                    r
                };
                let Some(result_text) = result_text else {
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
                if headless_mode && result_text.starts_with("[tool error]") {
                    return Err(exit_code::err(
                        ExitCode::Tool,
                        format!(
                            "cubi: tool '{}' failed: {}",
                            call.function.name, result_text
                        ),
                    ));
                }

                // Print a short preview so the user can see what came back
                // without us dumping a 10 KB log into the terminal.
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
            if step + 1 == MAX_AGENT_STEPS {
                eprintln!(
                    "{} agent loop hit step cap ({}); stopping. Ask me to continue \
                     if you want me to keep going.",
                    "Warn:".bright_yellow(),
                    MAX_AGENT_STEPS
                );
            }
        }

        // Per-turn footer (opt-in via `/stats-footer on`). Skipped when the
        // provider returned nothing useful or when `--quiet` is set.
        if self.stats_footer_enabled && !self.quiet_mode && !turn_stats.is_empty() {
            let window = crate::llm::context_window_for_model(self.executor.get_model());
            super::render::print_stats_footer(&turn_stats, window);
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
    async fn execute_consensus_tool_call(&mut self, call: &ToolCall) -> anyhow::Result<String> {
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
            .unwrap_or(crate::consensus::CONSENSUS_DEFAULT_MAX_STEPS);
        let concurrency = args
            .get("concurrency")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(0);

        let req = crate::consensus::ConsensusRequest {
            goal: goal.clone(),
            models: models.clone(),
            strategy,
            max_steps_per_subagent: max_steps,
            concurrency,
        };

        self.emit_status(format!(
            "  {} consensus over {} models: {}",
            "↳".bright_magenta(),
            models.len(),
            goal.chars().take(120).collect::<String>().bright_white()
        ));

        let sink = super::CliConsensusSink {
            json_enabled: self.json_enabled && self.headless_mode,
            event_sink: self.event_sink.clone(),
        };
        let result = crate::consensus::run(req, &self.executor, &sink).await?;
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
                    "- {} ✓ ({} tokens, {} ms)\n",
                    sub.model,
                    sub.prompt_tokens + sub.completion_tokens,
                    sub.elapsed_ms
                ));
            }
        }
        report.push_str("\n--- winning output ---\n");
        report.push_str(&result.winner_output);
        Ok(report)
    }
}
