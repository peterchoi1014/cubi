use super::*;

pub(super) fn repl_history_path() -> Option<PathBuf> {
    crate::sessions::cubi_dir().map(|dir| dir.join("history"))
}

/// Drops cached session / plugin / MCP suggestion lists when the user
/// runs a slash command that could change them, so the next Tab press
/// re-reads from disk. Matches on the canonical `Cmd` so prefix-typed
/// commands (e.g. `/sav` → `/save`) still invalidate.
fn invalidate_completer_caches_if_mutating(rl: &Editor<SlashHelper, DefaultHistory>, input: &str) {
    let Some((cmd, _)) = commands::parse(input) else {
        return;
    };
    let mutates = matches!(
        cmd,
        Cmd::Save
            | Cmd::Load
            | Cmd::Resume
            | Cmd::Sessions
            | Cmd::Plugin
            | Cmd::ReloadPlugins
            | Cmd::Mcp
            | Cmd::McpReload
            | Cmd::McpInstall
            | Cmd::McpUninstall
            | Cmd::Fork
    );
    if !mutates {
        return;
    }
    if let Some(helper) = rl.helper() {
        helper.invalidate_caches();
    }
}

impl ChatCLI {
    pub async fn run(&mut self) -> Result<()> {
        if !self.no_banner {
            self.print_welcome();
        }

        // Fire SessionStart hooks.
        self.hooks.fire_session_start(self.executor.get_model());
        self.emit_receipt(
            crate::receipts::ReceiptEvent::SessionStart {
                model: self.executor.get_model().to_string(),
                cwd: std::env::current_dir().unwrap_or_default(),
            },
            &serde_json::json!({
                "model": self.executor.get_model(),
                "cwd": std::env::current_dir().ok().map(|p| p.display().to_string()),
                "mode": "interactive",
            }),
        );

        let mut rl: Editor<SlashHelper, DefaultHistory> = Editor::new()?;
        rl.set_helper(Some(SlashHelper::new()));
        let readline_history_path = repl_history_path();
        if let Some(path) = &readline_history_path {
            if let Some(parent) = path.parent() {
                if let Err(err) = fs::create_dir_all(parent) {
                    eprintln!(
                        "{} could not create REPL history directory '{}': {}",
                        "Warn:".bright_yellow(),
                        parent.display(),
                        err
                    );
                }
            }
            if let Err(err) = rl.load_history(path) {
                let is_not_found = matches!(
                    &err,
                    ReadlineError::Io(io_err)
                        if io_err.kind() == std::io::ErrorKind::NotFound
                );
                if !is_not_found {
                    eprintln!(
                        "{} could not load REPL history '{}': {}",
                        "Warn:".bright_yellow(),
                        path.display(),
                        err
                    );
                }
            }
        }

        loop {
            let prompt = if self.plan_mode.load(Ordering::SeqCst) {
                format!(
                    "{}{} ",
                    "[plan] ".bright_yellow().bold(),
                    "You:".bright_green().bold()
                )
            } else {
                format!("{} ", "You:".bright_green().bold())
            };

            // Optional between-turn status line. Suppressed in headless,
            // JSON, and quiet modes and when stdout is not a TTY, so the
            // headless/JSON output paths stay byte-identical. Scrolls like
            // normal output (pinning is a later phase).
            {
                use std::io::IsTerminal;
                if !self.headless_mode
                    && !self.json_enabled
                    && !self.quiet_mode
                    && std::io::stdout().is_terminal()
                {
                    println!("{}", self.render_status_line());
                }
            }

            match rl.readline(&prompt) {
                Ok(line) => {
                    // Multi-line fold: if this line is a `"""` fence opener
                    // or ends in an unescaped backslash, keep reading until
                    // the block closes. Ctrl-C inside a block aborts the
                    // buffer cleanly. The bare-`"""`-only edge case is
                    // covered by `opener_kind` returning `Fence`, so we
                    // never treat an isolated `"""` as a no-op.
                    use super::multiline::{
                        OpenerKind, fold_multiline, is_continuation, opener_kind,
                    };
                    let kind = opener_kind(&line);
                    let folded = match kind {
                        OpenerKind::None | OpenerKind::FenceClosedInline => {
                            // Single-line path; let fold_multiline collapse
                            // the inline-fence case into its body.
                            let buf = vec![line.clone()];
                            let (body, _) = fold_multiline(&buf);
                            Some(body)
                        }
                        OpenerKind::Fence | OpenerKind::Backslash => {
                            // Read continuation lines with a dim `… `
                            // prompt so users can see they're inside a
                            // multi-line block. We change rustyline's
                            // prompt rather than pre-emitting a hint line,
                            // because the prompt re-renders correctly on
                            // terminal resize and history navigation.
                            let cont_prompt = format!("{} ", "…".bright_black());
                            let mut buf: Vec<String> = vec![line.clone()];
                            let mut aborted = false;
                            loop {
                                let done = match kind {
                                    OpenerKind::Fence => buf
                                        .last()
                                        .map(|l| buf.len() > 1 && l == "\"\"\"")
                                        .unwrap_or(false),
                                    OpenerKind::Backslash => {
                                        buf.last().map(|l| !is_continuation(l)).unwrap_or(true)
                                    }
                                    _ => true,
                                };
                                if done {
                                    break;
                                }
                                match rl.readline(&cont_prompt) {
                                    Ok(l) => buf.push(l),
                                    Err(ReadlineError::Interrupted) => {
                                        println!("{}", "multi-line input cancelled".bright_black());
                                        aborted = true;
                                        break;
                                    }
                                    Err(ReadlineError::Eof) => {
                                        aborted = true;
                                        break;
                                    }
                                    Err(err) => {
                                        eprintln!("Error: {:?}", err);
                                        aborted = true;
                                        break;
                                    }
                                }
                            }
                            if aborted {
                                None
                            } else {
                                let (body, _) = fold_multiline(&buf);
                                Some(body)
                            }
                        }
                    };
                    let Some(folded) = folded else {
                        continue;
                    };
                    let input = folded.trim();

                    if input.is_empty() {
                        continue;
                    }

                    // Shell escape: `!<cmd>` runs a shell command on the
                    // terminal (like `!` in a shell / `:!` in vim).
                    if let Some(shell_cmd) = input.strip_prefix('!') {
                        rl.add_history_entry(input)?;
                        self.run_shell_command(shell_cmd);
                        continue;
                    }

                    // Handle commands
                    if input.starts_with('/') {
                        // Check user-defined commands first.
                        if self.try_user_command(input) {
                            invalidate_completer_caches_if_mutating(&rl, input);
                            continue;
                        }
                        // Then dynamic `/<name>` agent commands, before falling
                        // back to the built-in handler (which owns the
                        // unknown-command hint). Precedence is enforced inside
                        // `resolve_agent_command`.
                        match self.resolve_agent_command(input) {
                            AgentCommand::Run(agent, agent_args) => {
                                self.run_agent_command(&agent, &agent_args).await;
                                continue;
                            }
                            AgentCommand::Disabled(name) => {
                                eprintln!(
                                    "{} {}",
                                    "Error:".bright_red(),
                                    ChatCLI::agent_disabled_message(&name)
                                );
                                continue;
                            }
                            AgentCommand::NotAgent => {}
                        }
                        if !self.handle_command(input).await? {
                            break;
                        }
                        invalidate_completer_caches_if_mutating(&rl, input);
                        continue;
                    }

                    // Add line to readline history
                    rl.add_history_entry(input)?;

                    // Expand @file mentions in user input.
                    let expanded = file_mentions::expand_file_mentions(input);

                    // Snapshot history length BEFORE pushing the user
                    // message so a mid-turn Ctrl-C can truncate back to a
                    // clean state (no dangling user message, no orphaned
                    // tool turns).
                    let turn_start = self.history.len();

                    // Add user message to history
                    self.history.push(Message::text("user", &expanded));

                    // Open a fresh journal bucket so any file edits in
                    // this turn can be rolled back atomically by /rewind.
                    self.journal.start_turn();

                    // Run the agent: stream model output, execute any
                    // requested tools, loop until the model returns plain
                    // content (or we hit the safety cap).
                    if let Err(e) = self.agent_turn(turn_start).await {
                        eprintln!("{} {}\n", "Error:".bright_red().bold(), e);
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("{}", "Use /quit to exit".yellow());
                    continue;
                }
                Err(ReadlineError::Eof) => {
                    break;
                }
                Err(err) => {
                    eprintln!("Error: {:?}", err);
                    break;
                }
            }
        }

        if let Some(path) = &readline_history_path {
            if let Err(err) = rl.save_history(path) {
                eprintln!(
                    "{} could not save REPL history '{}': {}",
                    "Warn:".bright_yellow(),
                    path.display(),
                    err
                );
            }
        }

        // Fire Stop hooks.
        self.hooks.fire_stop();
        self.emit_receipt(
            crate::receipts::ReceiptEvent::SessionEnd,
            &serde_json::json!({"mode": "interactive"}),
        );

        // Leave the user with a clear hint on how to pick a chat back up.
        if let Some(hint) = self.resume_hint() {
            println!("\n{hint}");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repl_history_path_uses_cubi_home() {
        crate::compat::test_env::with_cubi_home(|cubi_home, other_home| {
            let path = repl_history_path().expect("history path");
            assert_eq!(path, cubi_home.join(".cubi").join("history"));
            assert!(!path.starts_with(other_home));
        });
    }
}
