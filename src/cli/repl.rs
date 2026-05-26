use super::*;

pub(super) fn repl_history_path() -> Option<PathBuf> {
    dirs::home_dir().map(|home| home.join(".cubi").join("history"))
}

impl ChatCLI {
    pub async fn run(&mut self) -> Result<()> {
        if !self.no_banner {
            self.print_welcome();
        }

        // Fire SessionStart hooks.
        self.hooks.fire_session_start(self.executor.get_model());

        let mut rl: Editor<SlashHelper, DefaultHistory> = Editor::new()?;
        rl.set_helper(Some(SlashHelper));
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

            match rl.readline(&prompt) {
                Ok(line) => {
                    let input = line.trim();

                    if input.is_empty() {
                        continue;
                    }

                    // Handle commands
                    if input.starts_with('/') {
                        // Check user-defined commands first.
                        if self.try_user_command(input) {
                            continue;
                        }
                        if !self.handle_command(input).await? {
                            break;
                        }
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

        // Leave the user with a clear hint on how to pick a chat back up.
        // Three cases:
        //   1. We have an on-disk checkpoint for *this* chat → point at it
        //      directly with /resume <id>.
        //   2. No current session, but other checkpoints exist in this
        //      cwd → mention /sessions so they can still find them.
        //   3. Nothing on disk at all → say nothing; a hint would just
        //      be noise.
        let resume_hint = self.session_store.as_ref().and_then(|store| {
            if let Some(session) = self.current_session.as_ref() {
                if !store.exists(&session.id) {
                    return None;
                }
                Some(format!(
                    "\n{} To pick this chat back up, run {}",
                    "↩".bright_cyan(),
                    format!("cubi --resume {}", session.id).bright_cyan()
                ))
            } else if store.list().map(|l| !l.is_empty()).unwrap_or(false) {
                Some(format!(
                    "\n{} Run {} to jump back into your most recent chat, or {} for a list.",
                    "↩".bright_cyan(),
                    "cubi --resume".bright_cyan(),
                    "cubi  →  /sessions".bright_cyan()
                ))
            } else {
                None
            }
        });
        if let Some(hint) = resume_hint {
            println!("{}", hint);
        }

        Ok(())
    }
}
