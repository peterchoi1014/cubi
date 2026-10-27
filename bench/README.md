# Cubi bench suite

A small, self-contained set of regression tasks the Cubi agent should be
able to solve end-to-end against a local model. Used by `cubi bench` and
the nightly CI workflow to track score drift over time.

> **Scope.** This is Cubi's *internal* regression suite — zero network,
> zero Docker. For runs against the real SWE-bench-Lite dataset (300 GitHub
> issues, official predictions + optional pytest scoring), see
> [`cubi swebench`](../docs/swebench.md), which lives alongside this suite
> rather than replacing it.

## Layout

```
bench/
├── README.md
├── tasks/
│   └── <task-id>/
│       ├── task.toml      # id, difficulty, prompt, caps
│       ├── verify.sh      # POSIX sh; exit 0 == agent's fix is correct
│       └── repo/          # initial repo state (starts in failing state)
└── results/
    └── <unix-ts>/         # written by `cubi bench`
        ├── <task>.events.jsonl
        └── summary.json
```

## task.toml schema

```toml
id = "rust-fizzbuzz"             # must match the parent directory name
difficulty = "easy"              # easy | medium | hard
description = "..."              # one-line human summary
prompt = "..."                   # exactly what gets sent to the agent
time_cap_seconds = 120           # hard timeout for the agent subprocess
step_cap = 15                    # advisory; not yet enforced (use time_cap)
```

`difficulty = "easy"` tasks make up the `quick` suite (the CI default).
Anything else falls through to `--suite all`.

## verify.sh

Run by the bench harness with `cwd = <copy of repo/>`. Must exit `0`
when the agent's edits made the project correct, non-zero otherwise.
Conventionally `cargo test --quiet` or `python3 -m pytest -q`.

The script is also a useful local sanity check: from `bench/tasks/<id>/`
run `cd repo && cargo test` (or `pytest -q`) to confirm the *initial*
state fails — that's what the agent is asked to fix.

## Running

Prerequisites: the Rust tasks need `cargo` (already required to build
Cubi). The Python tasks invoke `python3 -m pytest` from `verify.sh`,
so install `pytest` locally if you want to run them:

```sh
python3 -m pip install --user pytest
```

```sh
# Default (the quick suite of easy tasks, model from $CUBI_MODEL):
cubi bench

# Explicit model and JSON summary on stdout:
cubi bench --suite quick --model qwen3.5:9b --json

# Write run artifacts outside the repository:
cubi bench --suite quick --output /tmp/cubi-bench-results

# One task at a time, keeping the agent's working copy for inspection:
cubi bench --task rust-fizzbuzz --keep-workdir

# Triple every task's time cap — for CPU-only hosts, where the caps below
# are far too tight and every task would otherwise just time out:
cubi bench --time-cap-multiplier 3
```

### Time caps and slow hosts

Each task's `time_cap_seconds` assumes GPU-class inference. On a CPU-only
host an 8B-class model needs several times longer for the same work, and a
task that runs out of clock is recorded as `timeout` — indistinguishable at a
glance from a model that simply failed. `--time-cap-multiplier <n>` scales
every cap by `n`, preserving the relative per-task budgets rather than
flattening them to a single number. The nightly CI job uses `3`.

If a run comes back all-`timeout` with `elapsed_seconds` sitting exactly on
the cap, that is the signal to raise the multiplier (or use a smaller model),
not evidence about the model's ability.

To size the multiplier rather than guess it, read `steps_used` on the timed-out
tasks: it counts the tool calls the agent got through before the cap, parsed
from the partial event log. `0` means it never reached a first tool call (the
cap is far too small, or the model is not emitting tool calls at all); a
number close to what the task needs means a modest bump will do. `null` means
no event log was produced at all.

Results land in `bench/results/<unix-ts>/` by default. Pass `--output <dir>`
to write `summary.json` and per-task event logs directly into another
directory, without adding a timestamp subdirectory. Relative output paths
are resolved from the invocation directory, and missing directories are
created. The integration tests use temporary output directories so
`cargo test --test bench` leaves `bench/results/` untouched.

The `summary.json` schema is stable; CI consumes it as an artifact.

Each per-task result retains bounded diagnostics: the last 4 KiB of agent
stderr and the last 4 KiB each of verifier stdout/stderr. Verifiers have a
120-second timeout by default; `CUBI_BENCH_VERIFY_TIMEOUT_SECS` can lower or
raise it for controlled harness tests and unusually slow tasks. A verifier
timeout and agent startup/provider failures use status `error` (infrastructure
failure). A completed agent whose bounded verifier exits non-zero uses status
`fail` (model/task failure). Agent deadline expiry remains status `timeout`.

## Adding a task

1. Pick a short, descriptive `id` (`rust-typo-fix`, `py-leap-year`).
2. `mkdir -p bench/tasks/<id>/repo`.
3. Add a minimal Cargo / Python project in `repo/` whose tests **fail**.
4. Write `task.toml` (see schema above) and `verify.sh` (chmod +x).
5. From `bench/tasks/<id>/repo/`, confirm the initial state fails.
6. `cargo test` to make sure the harness still discovers and parses
   your task.

Keep tasks tiny and reproducible: no network, no large dependencies,
ideally < 30s to build and verify.

## How the harness sandboxes a run

Per task, `run_task` copies `repo/` into a throwaway workdir and gives the
spawned agent its own temporary `HOME`, so a run never touches the
developer's real dotfiles. Two consequences are easy to get wrong, and both
silently pin the score at 0% rather than erroring:

1. **The workdir must be pre-trusted.** Cubi refuses writes outside a trusted
   root, and headless `-p` mode *auto-denies* instead of prompting. Because
   the isolated `HOME` starts with an empty trust store, the harness seeds
   `$HOME/.cubi/trusted_dirs.json` with the workdir
   (`permissions::seed_trust_file`, shared with `cubi swebench`). Without it
   every `edit_file`/`write_file` call comes back `[tool denied]`.
2. **Paths handed across the cwd boundary must be absolute.** `verify.sh` and
   the `--events` log are resolved from the *harness* cwd but used from the
   *workdir*, so `tasks_root` and the output dir are canonicalized up front.
   A relative `verify.sh` path makes `sh` exit 2 without running the tests.
3. **`CUBI_HOME` must match the throwaway home.** Cubi gives `CUBI_HOME`
   precedence over `HOME`/`USERPROFILE`, so inheriting a parent value would
   leak sessions, policy, and trust into scored runs. The agent and verifier
   receive the isolated value explicitly, while `CUBI_EVENTS`,
   `CUBI_RECEIPTS`, `CUBI_TRACE_TOOLS`, and `CUBI_POLICY_FILE` are cleared.
4. **Rust toolchain roots stay deliberate.** Isolating `HOME` changes
   rustup/cargo defaults. The harness therefore preserves explicit
   `RUSTUP_HOME`, `CARGO_HOME`, and `RUSTUP_TOOLCHAIN` values, deriving the
   first two from the parent home when unset. This keeps a verifier under
   `cargo +1.88 test` on the installed toolchain instead of attempting a
   fresh sync inside the temporary home. `CARGO_TARGET_DIR` is always set to
   a `cargo-target` directory inside the task's isolated home, because every
   task is a copy of the same crate and a shared target directory would let
   one task reuse another task's build artifacts.

`tests/bench.rs` guards both with an end-to-end pass/fail pair that scripts
the `Fake` LLM backend (`CUBI_FAKE_LLM` + `CUBI_FAKE_LLM_TOOL_CALL`), so they
run in ordinary CI with **no local model** — only `cargo`.

> **Historical note.** Nightly `summary.json` artifacts produced before these
> fixes report 0% for every model and are not a usable baseline.

## CI integration

`.github/workflows/bench.yml` runs the quick suite nightly (and on
manual dispatch) using `qwen3.5:4b` via Ollama with
`--time-cap-multiplier 3`, prints the score to the job summary, and uploads
`summary.json` as a workflow artifact. It runs a smaller model than the
shipped default because the runner is CPU-only — see the header comment in
that workflow for the reasoning. The job does **not** fail the
build on score regression today; tightening that threshold comes later
once we have several runs of baseline data.

Regular CI (`.github/workflows/ci.yml`) does *not* run `cubi bench`
against a live model — it has no Ollama. The harness itself is covered by
unit tests (`src/bench.rs` `#[cfg(test)]`) and the model-free end-to-end
tests in `tests/bench.rs`.

## Interpreting a score

The quick suite is **6 binary tasks**, so its resolution is coarse: a
one-task swing is ±16.7 points, and at n=6 the 95% confidence interval on a
5/6 result spans roughly 42–100%. That is wide enough that a single run
cannot separate two similar models. To compare models (e.g. a default-model
bump), run each several times and compare distributions, or grow the suite —
don't read a single `score_pct` as a verdict.
