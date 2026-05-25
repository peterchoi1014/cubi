# GitHub Copilot Instructions

These instructions apply to GitHub Copilot when working in this repository, including code generation, commit message authoring, and code review.

## Commit Message Instructions

Commit messages must describe **what changed** and **why** (rationale). Keep them focused and free of process or tooling noise.

- Include only:
  - A clear description of the change.
  - The rationale or motivation for the change (when not obvious).
  - References to related issues/PRs when relevant.
- Do **not** include:
  - References to "Copilot", "Ralph Loop", agents, or any LLM/AI assistant.
  - "Agent-Logs-Url" or links to agent logs/traces.
  - References to LLM instructions, prompts, or system messages.
  - Descriptions of how the user requested the change (e.g., "as requested by user", "per user instruction").
  - Inference details, reasoning steps, or meta-commentary about the change process.
- Style:
  - Use the imperative mood in the subject line (e.g., "Add", "Fix", "Refactor").
  - Keep the subject line concise; put details in the body when needed.

## Code Review Instructions

When performing code review (including Copilot Code Review), follow these rules:

### 1. Prioritize functional and critical issues
Focus first on issues that affect correctness, behavior, or stability:
- Logic errors and incorrect behavior.
- Crashes, panics, unwraps on values that can fail, and unhandled errors.
- Concurrency issues (data races, deadlocks, incorrect async usage).
- Security issues (input validation, secret handling, unsafe code).
- Performance problems that meaningfully impact users.

Surface these before stylistic or minor suggestions.

### 2. Check for regressions
Actively look for regressions introduced by the change:
- Behavior changes in existing features or public APIs.
- Removed or altered error handling, validation, or edge-case coverage.
- Breaking changes to CLI flags, configuration, output formats, or persisted data.
- Tests that were removed, weakened, or no longer exercise the original behavior.

### 3. Follow standard coding conventions and best practices
Verify the change conforms to language and project conventions:
- Rust best practices (idiomatic error handling with `Result`, avoid unnecessary `unwrap`/`expect`, prefer `?`, proper ownership/borrowing).
- Passes `cargo fmt --all -- --check`, `cargo clippy --all-targets -- -D warnings`, and `cargo test`.
- Clear naming, appropriate module boundaries, and documentation for public items.
- Tests added or updated for new/changed behavior.

### 4. Complete the review in one pass
- Identify **all** issues during the first review of a given change.
- Do not surface new comments on subsequent iterations unless they pertain to code introduced or modified by those subsequent changes.
- Avoid generating additional comments simply because the review is being re-run; iterative re-reviews on unchanged code should produce no new comments.

### 5. No comment is a valid review outcome
- If no functional, regression, or convention issues are found, returning **no comments** is the correct and expected outcome.
- Do not invent comments, nitpicks, or speculative suggestions to fill space.
- Approve cleanly when the change is good.
