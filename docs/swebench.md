# SWE-bench-Lite with Cubi

[`cubi swebench`](../src/swebench.rs) drives Cubi against the real
[SWE-bench-Lite](https://www.swebench.com/) dataset — 300 GitHub issues from
popular Python projects — and produces predictions in the **official schema**
plus an optional **local resolved-rate** score.

> **Scope.** This complements the curated, network-free
> [`cubi bench`](../bench/README.md) suite. `bench` is a fast internal
> regression metric; `swebench` measures Cubi against the same tasks the
> broader ecosystem reports on.

## How it works

For each selected instance, Cubi:

1. Checks out the instance repo at its `base_commit` in a clean working tree
   (from a pre-cloned cache via `--repos-root`, or by cloning from GitHub).
2. Seeds a throwaway trust root so the headless agent may edit the checkout.
3. Runs the Cubi binary headlessly with the issue `problem_statement`.
4. Captures the resulting `git diff` as the `model_patch`.
5. With `--score`: applies the instance's `test_patch` and runs the
   `FAIL_TO_PASS` + `PASS_TO_PASS` node ids with `pytest` to decide whether the
   issue is *resolved* (every `FAIL_TO_PASS` passes and every `PASS_TO_PASS`
   still passes).

Two files are written to the output directory:

| File | Purpose |
| --- | --- |
| `predictions.jsonl` | One record per instance in the official schema (`instance_id`, `model_name_or_path`, `model_patch`). |
| `summary.json` | Cubi-local aggregation: instances, patches produced, resolved count/percentage, per-instance detail. |

## The canonical number vs. the local score

The **canonical** SWE-bench-Lite resolved rate comes from the upstream
Dockerized harness, which pins each instance's Python environment. Generate
predictions with Cubi, then score them upstream:

```bash
python -m swebench.harness.run_evaluation \
  --predictions_path <output>/predictions.jsonl \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --run_id cubi-run
```

`--score` is a **fast, best-effort local approximation** that skips Docker. It
requires the target repo's Python environment (dependencies, correct
interpreter) to already be importable, so treat its number as a smoke signal,
not the published result. The scorer prefers `python3`, falls back to `python`,
and honors `CUBI_SWEBENCH_PYTHON`.

## Canonical scoring (CI)

Use `.github/workflows/swebench-score.yml` when you need the canonical
resolved-rate score for a Cubi-generated `predictions.jsonl`. The workflow runs
the upstream SWE-bench Docker harness on GitHub's `ubuntu-latest` x86_64
runners, where Docker is already available. This avoids the maintainer
Apple-Silicon path: SWE-bench's prebuilt images target Linux/x86_64, and its
ARM support requires local image builds and is still experimental.

First, generate predictions locally or on any machine that can run Cubi:

```bash
cubi swebench --dataset swe-bench-lite.jsonl \
  --model qwen3:8b \
  --output bench/swebench-results/cubi-run
```

The CI workflow uses a repository path for the input because
`workflow_dispatch` cannot attach a local file directly. Commit
`predictions.jsonl` on the branch you dispatch. Keep that file on a scratch
scoring branch if you do not want to merge generated predictions into `main`:

```bash
git checkout -b swebench-score/cubi-run
git add bench/swebench-results/cubi-run/predictions.jsonl
git commit -m "Add SWE-bench predictions for scoring"
git push origin swebench-score/cubi-run
```

Then run the manual workflow from that branch:

```bash
gh workflow run swebench-score.yml \
  --ref swebench-score/cubi-run \
  -f predictions_path=bench/swebench-results/cubi-run/predictions.jsonl \
  -f dataset_name=princeton-nlp/SWE-bench_Lite \
  -f run_id=cubi-run
```

The workflow installs `swebench==4.1.0` and runs:

```bash
python -m swebench.harness.run_evaluation \
  --predictions_path <predictions_path> \
  --dataset_name <dataset_name> \
  --run_id <run_id> \
  --max_workers 4
```

When the run finishes, download the `swebench-score-<github-run-id>` artifact.
It contains `run_evaluation.log`, Docker build/evaluation logs from `logs/`,
and the upstream report files from `evaluation_results/`. Use the report under
`evaluation_results/` as the canonical resolved-rate result; use the logs to
debug Docker image builds or individual failing instances.

## Getting the dataset

`cubi swebench` reads a JSONL file with one instance per line. Export the HF
dataset once:

```python
from datasets import load_dataset
ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
ds.to_json("swe-bench-lite.jsonl")
```

Each line needs at least `instance_id`, `repo`, `base_commit`,
`problem_statement`, and (for `--score`) `test_patch`, `FAIL_TO_PASS`,
`PASS_TO_PASS`. The `FAIL_TO_PASS`/`PASS_TO_PASS` fields may be either JSON
arrays or JSON-encoded strings (the dataset's on-disk form) — both are accepted.

## Usage

```bash
# Generate predictions for the whole dataset (clones repos from GitHub):
cubi swebench --dataset swe-bench-lite.jsonl --model qwen3:8b

# Offline: reuse a directory of pre-cloned repos (named owner__name or name):
cubi swebench --dataset swe-bench-lite.jsonl --repos-root ~/swe-repos

# Fast local iteration: one instance, keep its checkout, local score:
cubi swebench --dataset swe-bench-lite.jsonl \
  --instance astropy__astropy-12907 --score --keep-workdir

# First N instances, JSON summary on stdout:
cubi swebench --dataset swe-bench-lite.jsonl --limit 10 --json
```

### Flags

| Flag | Meaning |
| --- | --- |
| `--dataset <path.jsonl>` | Instances, one JSON per line. **Required.** |
| `--model <name>` | Model to drive (default `$CUBI_MODEL` or `qwen3:8b`). |
| `--instance <id>` | Run a single instance by id. |
| `--limit <n>` | Cap the number of instances. |
| `--repos-root <dir>` | Directory of pre-cloned repos (offline); otherwise clone from GitHub. |
| `--score` | Apply `test_patch` and run pytest for a local resolved rate. |
| `--time-cap <seconds>` | Per-instance wall-clock cap for the agent (default 900). |
| `--output <dir>` | Output dir (default `bench/swebench-results/<unix-ts>/`). |
| `--keep-workdir` | Keep each instance's checkout for inspection. |
| `--json` | Print the summary as JSON to stdout. |

## Notes and limitations

- Repo cloning and full environment setup are heavy; use `--repos-root` and
  `--limit` while iterating.
- The local `--score` does not reproduce the upstream per-instance dependency
  pins — a green local score should be confirmed with the Docker harness before
  it is reported.
- The harness is covered by unit tests (`src/swebench.rs` `#[cfg(test)]`) and
  CLI-plumbing tests (`tests/swebench.rs`); end-to-end runs need a live model.
