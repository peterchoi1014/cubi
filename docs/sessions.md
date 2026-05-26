# Sessions cookbook

cubi checkpoints conversations per project so interrupted work can be resumed.

- List sessions: `cubi --list-sessions`.
- Machine-readable list: `cubi --list-sessions --json`.
- Resume latest session for the current directory: `cubi --resume`.
- Resume by id or unique prefix: `cubi --resume 20250101-000000-abcd`.
- Delete by id or unique prefix: `cubi --delete-session 20250101`.
- Prune old sessions: `cubi --prune-sessions --older-than 30d`.
- Preview pruning: `cubi --prune-sessions --older-than 6m --dry-run`.

The global index lives at `~/.cubi/sessions/index.json`. Session files are stored under `~/.cubi/sessions/` and include the cwd, model, timestamps, and message history. Prefer the CLI commands over hand-editing the index so prefix lookups and pruning stay consistent.
