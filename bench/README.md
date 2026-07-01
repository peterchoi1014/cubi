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
cubi bench --suite quick --model qwen3:8b --json

# One task at a time, keeping the agent's working copy for inspection:
cubi bench --task rust-fizzbuzz --keep-workdir
```

Results land in `bench/results/<unix-ts>/`. The `summary.json` schema is
stable; CI consumes it as an artifact.

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

## CI integration

`.github/workflows/bench.yml` runs the quick suite nightly (and on
manual dispatch) using `qwen3:8b` via Ollama and uploads the
`summary.json` as a workflow artifact. The job does **not** fail the
build on score regression today; tightening that threshold comes later
once we have several runs of baseline data.

Regular CI (`.github/workflows/ci.yml`) does *not* run `cubi bench` —
it has no local model. The harness itself is covered by unit tests
(`src/bench.rs` `#[cfg(test)]`) and `tests/bench.rs`.
