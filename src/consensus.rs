//! Multi-model consensus.
//!
//! `consensus_run` (and the `/consensus` slash command) spawns N
//! subagents against a caller-supplied list of models, then arbitrates
//! the candidate outputs via one of three strategies:
//!
//!   * [`ConsensusStrategy::Vote`] — plurality of normalized outputs.
//!   * [`ConsensusStrategy::BestOfN`] — judge model scores each output.
//!   * [`ConsensusStrategy::Judge`] — judge model picks one with a
//!     short reasoning string.
//!
//! No coding agent in the comparison runs parallel multi-model
//! arbitration natively. This module is the load-bearing implementation
//! of that differentiator.
//!
//! ## Scope
//!
//! Consensus subagents default to a **single LLM call** with the goal as
//! the user message and a short focusing system prompt. Tool access is
//! strictly opt-in via [`ConsensusRequest::use_tools`]. Tool-enabled
//! consensus runs subagents sequentially through the standard subagent
//! loop so the live `McpManager` is never shared concurrently.
//!
//! ## Anti-recursion
//!
//! `consensus_run` is stripped from every subagent's tool list (both
//! `agent_run` subagents and opt-in tool-enabled consensus subagents).
//! The same [`agent_loop::without_meta_tools`] helper strips both
//! `agent_run` and `consensus_run` so they can't drift apart.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use futures_util::stream::{FuturesUnordered, StreamExt};
use serde::Serialize;
use serde_json::json;
use tokio::sync::Semaphore;

use crate::executor::AIExecutor;
use crate::mcp_manager::McpManager;
use crate::ollama::{ChatStats, Message, ToolFunction, ToolSpec};

/// System prompt for every LLM-only consensus subagent. Mirrors the
/// focused tone of `agent_loop::SUBAGENT_SYSTEM_PROMPT` but emphasises
/// that the reply is a single self-contained answer (no tools, no
/// follow-up).
const CONSENSUS_SYSTEM_PROMPT: &str = "You are one of several focused worker subagents answering the SAME goal in \
     parallel under different models. Produce a single concise, self-contained \
     final answer. Do not ask clarifying questions — make a reasonable assumption \
     and note it. Do not chat — emit only the final answer.";

/// Default per-subagent step cap. Kept as a constant so the tool spec
/// and the dispatcher agree on the same default.
pub const CONSENSUS_DEFAULT_MAX_STEPS: usize = 8;

/// Arbitration strategy used by [`run`] to pick a winner.
#[derive(Debug, Clone)]
pub enum ConsensusStrategy {
    /// Plurality of normalized outputs. Ties escalate to a single
    /// [`Judge`](ConsensusStrategy::Judge) call using the first model
    /// in the request — recorded in `decision_reason`.
    Vote,
    /// LLM-graded best-of-N: the judge scores each candidate on a
    /// 1–10 rubric and the highest score wins.
    BestOfN { judge_model: String },
    /// LLM picks one candidate with a short reasoning string.
    Judge { judge_model: String },
}

/// Caller-supplied configuration for [`run`].
#[derive(Debug, Clone)]
pub struct ConsensusRequest {
    pub goal: String,
    pub models: Vec<String>,
    pub strategy: ConsensusStrategy,
    /// Per-subagent step cap. LLM-only consensus uses a single call;
    /// tool-enabled consensus clamps this into the subagent loop cap.
    pub max_steps_per_subagent: usize,
    /// Maximum number of subagents in flight at once. `0` means
    /// `models.len()` — i.e. fully parallel. `1` is sequential.
    pub concurrency: usize,
    /// When false (the default), each subagent is a single LLM-only call.
    /// When true, each subagent may use tools and runs sequentially.
    pub use_tools: bool,
}

/// One subagent's output. Recorded for every model in `models` —
/// successes and failures alike — so the caller can show a per-model
/// summary table.
#[derive(Debug, Clone, Serialize)]
pub struct SubagentOutput {
    pub model: String,
    pub output: String,
    pub steps_used: usize,
    pub elapsed_ms: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// `None` on success. On failure, the (truncated) error message.
    pub error: Option<String>,
}

impl SubagentOutput {
    pub fn ok(&self) -> bool {
        self.error.is_none()
    }
}

/// Final result of a consensus run.
#[derive(Debug, Clone, Serialize)]
pub struct ConsensusResult {
    pub winner_model: String,
    pub winner_output: String,
    pub subagent_outputs: Vec<SubagentOutput>,
    pub decision_reason: String,
    /// Strategy the caller asked for. Recorded so display and report
    /// code don't have to infer the strategy back out of
    /// `decision_reason` (which is fragile — a `Vote` run that
    /// tie-breaks via a judge mentions "judge" in the reason).
    pub requested_strategy: String,
}

impl ConsensusResult {
    /// Sum of token usage across every subagent (successful or not).
    /// Folded into the parent turn's `ChatStats` by the dispatcher so
    /// `/cost` and the usage footer include consensus traffic.
    pub fn aggregate_stats(&self) -> ChatStats {
        let mut s = ChatStats::default();
        for sub in &self.subagent_outputs {
            s.prompt_tokens += sub.prompt_tokens;
            s.completion_tokens += sub.completion_tokens;
            s.elapsed_ms += sub.elapsed_ms;
        }
        s
    }
}

/// `ToolSpec` for the `consensus_run` meta-tool. Schema enforces
/// `minItems: 2` on `models` so a one-model consensus is rejected at
/// the schema layer rather than after the parallel dispatch starts.
pub fn consensus_run_spec() -> ToolSpec {
    ToolSpec {
        tool_type: "function".to_string(),
        function: ToolFunction {
            name: crate::agent_loop::CONSENSUS_TOOL_NAME.to_string(),
            description: "Run the same goal under multiple models and arbitrate. \
                          Useful when you want a more reliable answer than any single model \
                          can provide. Cannot be called from within a subagent (anti-recursion). \
                          By default subagents are LLM-only and may run in parallel. Set \
                          use_tools=true to let subagents use tools; tool mode is sequential."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "Self-contained goal sent verbatim to each subagent."
                    },
                    "models": {
                        "type": "array",
                        "items": { "type": "string" },
                        "minItems": 2,
                        "description": "Models to consult. LLM-only mode may run them in parallel; tool mode runs sequentially."
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["vote", "best-of-n", "judge"],
                        "default": "vote"
                    },
                    "judge_model": {
                        "type": "string",
                        "description": "Required when strategy is `best-of-n` or `judge`."
                    },
                    "max_steps": {
                        "type": "integer",
                        "default": CONSENSUS_DEFAULT_MAX_STEPS
                    },
                    "concurrency": {
                        "type": "integer",
                        "default": 0,
                        "description": "0 = parallel (len(models)); 1 = sequential. Ignored when use_tools=true because tool mode is always sequential."
                    },
                    "use_tools": {
                        "type": "boolean",
                        "default": false,
                        "description": "Opt in to tool-enabled subagents. Defaults to false; when true, subagents run sequentially with the live tool manager."
                    }
                },
                "required": ["goal", "models"]
            }),
        },
    }
}

/// Hook the dispatcher uses to emit `consensus_*` JSON events without
/// `consensus::run` having to know about CLI plumbing. Implementations
/// route to whatever sink is active (stdout JSON line, `--events`
/// file, both).
pub trait ConsensusEventSink: Send + Sync {
    fn start(&self, goal: &str, models: &[String], strategy: &str, max_steps_per_subagent: usize);
    fn subagent_result(&self, sub: &SubagentOutput);
    fn decision(&self, winner_model: &str, decision_reason: &str);
}

/// No-op sink used by callers (and tests) that don't need event emission.
#[allow(dead_code)]
pub struct NullSink;
impl ConsensusEventSink for NullSink {
    fn start(&self, _: &str, _: &[String], _: &str, _: usize) {}
    fn subagent_result(&self, _: &SubagentOutput) {}
    fn decision(&self, _: &str, _: &str) {}
}

/// Drives a consensus run using the default LLM-only mode: launches N
/// subagents in parallel (limited by [`ConsensusRequest::concurrency`]),
/// arbitrates the outputs, and returns a [`ConsensusResult`].
///
/// If [`ConsensusRequest::use_tools`] is true, call [`run_with_tools`]
/// instead so the live MCP manager can be threaded through the sequential
/// tool loop.
///
/// Errors in individual subagents are recorded with `error: Some(...)`
/// and excluded from arbitration; only a wholesale failure (no model
/// produced an output) bubbles up as `Err`.
pub async fn run(
    req: ConsensusRequest,
    executor: &AIExecutor,
    sink: &dyn ConsensusEventSink,
) -> Result<ConsensusResult> {
    run_inner(req, executor, sink, None).await
}

/// Drives a consensus run that may opt in to tool-enabled subagents.
///
/// When [`ConsensusRequest::use_tools`] is false this delegates to the
/// unchanged LLM-only path. When it is true, subagents are run
/// sequentially regardless of [`ConsensusRequest::concurrency`] because
/// tool execution needs a single live mutable [`McpManager`].
pub async fn run_with_tools(
    req: ConsensusRequest,
    executor: &AIExecutor,
    sink: &dyn ConsensusEventSink,
    mcp: &mut Option<McpManager>,
) -> Result<ConsensusResult> {
    run_inner(req, executor, sink, Some(mcp)).await
}

async fn run_inner(
    req: ConsensusRequest,
    executor: &AIExecutor,
    sink: &dyn ConsensusEventSink,
    mcp: Option<&mut Option<McpManager>>,
) -> Result<ConsensusResult> {
    if req.models.len() < 2 {
        anyhow::bail!(
            "consensus_run requires at least 2 models, got {}",
            req.models.len()
        );
    }
    if req.use_tools {
        match mcp.as_ref() {
            None => {
                anyhow::bail!(
                    "tool-enabled consensus requires an MCP manager; call run_with_tools"
                );
            }
            Some(slot) if slot.is_none() => {
                anyhow::bail!("tool-enabled consensus requires an active MCP manager");
            }
            Some(_) => {}
        }
    }
    let strategy_label = match &req.strategy {
        ConsensusStrategy::Vote => "vote",
        ConsensusStrategy::BestOfN { .. } => "best-of-n",
        ConsensusStrategy::Judge { .. } => "judge",
    };
    sink.start(
        &req.goal,
        &req.models,
        strategy_label,
        req.max_steps_per_subagent,
    );

    let subagent_outputs = if req.use_tools {
        let mcp = mcp.expect("tool-enabled consensus requires MCP manager");
        run_tool_subagents(&req, executor, mcp).await
    } else {
        run_llm_subagents(&req, executor).await
    };

    for sub in &subagent_outputs {
        sink.subagent_result(sub);
    }

    let successes: Vec<&SubagentOutput> = subagent_outputs.iter().filter(|s| s.ok()).collect();
    if successes.is_empty() {
        anyhow::bail!(
            "consensus_run: every subagent failed ({} models)",
            subagent_outputs.len()
        );
    }
    if successes.len() == 1 {
        let only: SubagentOutput = (*successes[0]).clone();
        let reason = "only successful subagent".to_string();
        sink.decision(&only.model, &reason);
        return Ok(ConsensusResult {
            winner_model: only.model,
            winner_output: only.output,
            subagent_outputs,
            decision_reason: reason,
            requested_strategy: strategy_label.to_string(),
        });
    }

    let (winner_idx, decision_reason) =
        arbitrate(&req, executor, &subagent_outputs, &successes).await;
    let winner = &subagent_outputs[winner_idx];
    sink.decision(&winner.model, &decision_reason);
    Ok(ConsensusResult {
        winner_model: winner.model.clone(),
        winner_output: winner.output.clone(),
        subagent_outputs: subagent_outputs.clone(),
        decision_reason,
        requested_strategy: strategy_label.to_string(),
    })
}

async fn run_llm_subagents(req: &ConsensusRequest, executor: &AIExecutor) -> Vec<SubagentOutput> {
    let permits = if req.concurrency == 0 {
        req.models.len()
    } else {
        req.concurrency.min(req.models.len())
    };
    let semaphore = Arc::new(Semaphore::new(permits.max(1)));

    let mut in_flight: FuturesUnordered<_> = FuturesUnordered::new();
    for (idx, model) in req.models.iter().cloned().enumerate() {
        let sem = Arc::clone(&semaphore);
        let goal = req.goal.clone();
        let fut = async move {
            // Permit is dropped when the closure returns, releasing the
            // slot for the next queued subagent.
            let _permit = sem.acquire_owned().await.expect("semaphore closed");
            let started = Instant::now();
            let messages = vec![
                Message::text("system", CONSENSUS_SYSTEM_PROMPT),
                Message::text("user", goal.as_str()),
            ];
            let res = executor.chat_with_model(&model, messages).await;
            let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
            let sub = match res {
                Ok((msg, stats)) => SubagentOutput {
                    model: model.clone(),
                    output: msg.content,
                    steps_used: 1,
                    elapsed_ms: elapsed_ms.max(stats.elapsed_ms),
                    prompt_tokens: stats.prompt_tokens,
                    completion_tokens: stats.completion_tokens,
                    error: None,
                },
                Err(e) => SubagentOutput {
                    model: model.clone(),
                    output: String::new(),
                    steps_used: 0,
                    elapsed_ms,
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    error: Some(truncate(&e.to_string(), 400)),
                },
            };
            (idx, sub)
        };
        in_flight.push(fut);
    }

    let mut indexed: Vec<(usize, SubagentOutput)> = Vec::with_capacity(req.models.len());
    while let Some((idx, sub)) = in_flight.next().await {
        indexed.push((idx, sub));
    }
    indexed.sort_by_key(|(i, _)| *i);
    indexed.into_iter().map(|(_, o)| o).collect()
}

async fn run_tool_subagents(
    req: &ConsensusRequest,
    executor: &AIExecutor,
    mcp: &mut Option<McpManager>,
) -> Vec<SubagentOutput> {
    let mut subagent_outputs = Vec::with_capacity(req.models.len());
    for model in &req.models {
        let started = Instant::now();
        let res = crate::agent_loop::run_subagent_with_model(
            executor,
            mcp,
            Some(model.as_str()),
            &req.goal,
            req.max_steps_per_subagent,
        )
        .await;
        let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        let sub = match res {
            Ok(result) => SubagentOutput {
                model: model.clone(),
                output: result.output,
                steps_used: result.steps_used,
                elapsed_ms: elapsed_ms.max(result.stats.elapsed_ms),
                prompt_tokens: result.stats.prompt_tokens,
                completion_tokens: result.stats.completion_tokens,
                error: None,
            },
            Err(e) => SubagentOutput {
                model: model.clone(),
                output: String::new(),
                steps_used: 0,
                elapsed_ms,
                prompt_tokens: 0,
                completion_tokens: 0,
                error: Some(truncate(&e.to_string(), 400)),
            },
        };
        subagent_outputs.push(sub);
    }
    subagent_outputs
}

/// Returns `(winner_index_in_subagent_outputs, decision_reason)`.
async fn arbitrate(
    req: &ConsensusRequest,
    executor: &AIExecutor,
    all: &[SubagentOutput],
    successes: &[&SubagentOutput],
) -> (usize, String) {
    match &req.strategy {
        ConsensusStrategy::Vote => vote(all, successes, &req.models, executor, &req.goal).await,
        ConsensusStrategy::BestOfN { judge_model } => {
            best_of_n(all, successes, judge_model, executor, &req.goal).await
        }
        ConsensusStrategy::Judge { judge_model } => {
            judge(all, successes, judge_model, executor, &req.goal).await
        }
    }
}

async fn vote(
    all: &[SubagentOutput],
    successes: &[&SubagentOutput],
    models: &[String],
    executor: &AIExecutor,
    goal: &str,
) -> (usize, String) {
    let mut buckets: HashMap<String, Vec<usize>> = HashMap::new();
    for sub in successes {
        let key = hash_normalized(&sub.output);
        let global_idx = all
            .iter()
            .position(|s| std::ptr::eq::<SubagentOutput>(s, *sub))
            .expect("success is from same slice");
        buckets.entry(key).or_default().push(global_idx);
    }
    let max_count = buckets.values().map(|v| v.len()).max().unwrap_or(0);
    let top: Vec<&Vec<usize>> = buckets.values().filter(|v| v.len() == max_count).collect();

    if top.len() == 1 {
        let winner_idx = top[0][0];
        let total = successes.len();
        return (winner_idx, format!("majority vote {max_count}/{total}"));
    }
    // Tie: escalate to a Judge call using the first model in `models`.
    let judge_model = models
        .first()
        .cloned()
        .unwrap_or_else(|| executor.get_model().to_string());
    let (idx, reason) = judge(all, successes, &judge_model, executor, goal).await;
    (
        idx,
        format!("vote tie ({max_count}-way); escalated to judge `{judge_model}`: {reason}"),
    )
}

async fn best_of_n(
    all: &[SubagentOutput],
    successes: &[&SubagentOutput],
    judge_model: &str,
    executor: &AIExecutor,
    goal: &str,
) -> (usize, String) {
    let prompt = build_best_of_n_prompt(goal, successes);
    let messages = vec![
        Message::text(
            "system",
            "You are an impartial grader. Score each candidate on a 1-10 rubric for \
             correctness, completeness, and clarity. Reply with one line per candidate \
             in the form `i: <score>` where i is the 1-based index.",
        ),
        Message::text("user", prompt),
    ];
    let raw = match executor.chat_with_model(judge_model, messages).await {
        Ok((msg, _)) => msg.content,
        Err(e) => {
            // Judge failed → fall back to Vote with no escalation.
            let (idx, vote_reason) =
                vote(all, successes, &[judge_model.to_string()], executor, goal).await;
            return (
                idx,
                format!(
                    "best-of-n judge `{judge_model}` failed ({e}); fell back to vote: {vote_reason}"
                ),
            );
        }
    };
    let scores = parse_scores(&raw, successes.len());
    let (i, score) = scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, s)| (i, *s))
        .unwrap_or((0, 0.0));
    let winner_idx = all
        .iter()
        .position(|s| std::ptr::eq::<SubagentOutput>(s, successes[i]))
        .expect("success is from same slice");
    (
        winner_idx,
        format!(
            "best-of-n: `{}` scored {score} (judge `{judge_model}`)",
            successes[i].model
        ),
    )
}

async fn judge(
    all: &[SubagentOutput],
    successes: &[&SubagentOutput],
    judge_model: &str,
    executor: &AIExecutor,
    goal: &str,
) -> (usize, String) {
    let prompt = build_judge_prompt(goal, successes);
    let messages = vec![
        Message::text(
            "system",
            "You are an impartial judge. Pick the single best candidate. Reply with one \
             line `pick: i` (1-based) followed by 1-2 sentences of reasoning.",
        ),
        Message::text("user", prompt),
    ];
    let raw = match executor.chat_with_model(judge_model, messages).await {
        Ok((msg, _)) => msg.content,
        Err(e) => {
            // Judge unavailable → fall back to plain vote (no recursion
            // into vote() with the judge as model, to avoid endless
            // escalation when only the judge is broken).
            let max_count = successes.len();
            let winner = successes
                .first()
                .map(|s| {
                    all.iter()
                        .position(|x| std::ptr::eq::<SubagentOutput>(x, *s))
                        .expect("success is from same slice")
                })
                .unwrap_or(0);
            return (
                winner,
                format!(
                    "judge `{judge_model}` failed ({e}); fell back to first successful subagent of {max_count}"
                ),
            );
        }
    };
    let (pick, reason) = parse_pick(&raw, successes.len());
    let winner_idx = all
        .iter()
        .position(|s| std::ptr::eq::<SubagentOutput>(s, successes[pick]))
        .expect("success is from same slice");
    (
        winner_idx,
        format!(
            "judge `{judge_model}` picked `{}`: {reason}",
            successes[pick].model
        ),
    )
}

fn build_judge_prompt(goal: &str, successes: &[&SubagentOutput]) -> String {
    let mut s = format!("Goal:\n{goal}\n\nCandidates:\n");
    for (i, sub) in successes.iter().enumerate() {
        s.push_str(&format!(
            "\n--- Candidate {} (model: {}) ---\n{}\n",
            i + 1,
            sub.model,
            sub.output
        ));
    }
    s.push_str("\nReply with `pick: i` and 1-2 sentences of reasoning.");
    s
}

fn build_best_of_n_prompt(goal: &str, successes: &[&SubagentOutput]) -> String {
    let mut s = format!("Goal:\n{goal}\n\nCandidates:\n");
    for (i, sub) in successes.iter().enumerate() {
        s.push_str(&format!(
            "\n--- Candidate {} (model: {}) ---\n{}\n",
            i + 1,
            sub.model,
            sub.output
        ));
    }
    s.push_str("\nReply with one `i: <score>` line per candidate.");
    s
}

fn parse_pick(raw: &str, n: usize) -> (usize, String) {
    for line in raw.lines() {
        let lower = line.trim().to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("pick:") {
            if let Some(num) = rest.split_whitespace().next() {
                if let Ok(i) = num
                    .trim_end_matches(|c: char| !c.is_ascii_digit())
                    .parse::<usize>()
                {
                    if i >= 1 && i <= n {
                        let reason = raw
                            .lines()
                            .skip_while(|l| !l.to_ascii_lowercase().trim().starts_with("pick:"))
                            .skip(1)
                            .collect::<Vec<_>>()
                            .join(" ");
                        let reason = if reason.trim().is_empty() {
                            "(no reason given)".to_string()
                        } else {
                            truncate(reason.trim(), 240)
                        };
                        return (i - 1, reason);
                    }
                }
            }
        }
    }
    (
        0,
        "unparseable judge reply, defaulted to candidate 1".to_string(),
    )
}

fn parse_scores(raw: &str, n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n];
    for line in raw.lines() {
        let trimmed = line.trim();
        if let Some((idx_str, score_str)) = trimmed.split_once(':') {
            let idx_str = idx_str.trim();
            // Tolerate `1`, `1.`, `(1)`, etc.
            let cleaned: String = idx_str.chars().filter(|c| c.is_ascii_digit()).collect();
            if let Ok(i) = cleaned.parse::<usize>() {
                if i >= 1 && i <= n {
                    let score = score_str
                        .split_whitespace()
                        .next()
                        .and_then(|s| {
                            s.trim_end_matches(|c: char| !c.is_ascii_digit() && c != '.')
                                .parse::<f32>()
                                .ok()
                        })
                        .unwrap_or(0.0);
                    out[i - 1] = score;
                }
            }
        }
    }
    out
}

/// Hash an output for vote bucketing. Whitespace is collapsed and the
/// text is lowercased so trivial formatting differences don't split
/// the vote. The hash itself is just a stable bucketing key — content
/// is preserved unchanged in [`SubagentOutput::output`] — so the
/// non-cryptographic stdlib hasher is sufficient (and avoids a new
/// dependency). `DefaultHasher::new()` is fixed-seeded, so two
/// identical inputs in the same process always land in the same bucket.
fn hash_normalized(s: &str) -> String {
    use std::hash::Hasher;
    let normalized: String = s
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write(normalized.as_bytes());
    format!("{:016x}", h.finish())
}

/// Truncate a string to `max` characters, appending `…` when cut.
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max).collect();
        out.push('…');
        out
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────
//
// Tests live in-module rather than under `tests/` because cubi is a
// bin-only crate (no `lib.rs`); integration tests would have to drive
// the full binary via `assert_cmd`. Unit tests against `LlmBackend::Fake`
// here keep the consensus arbitration logic verifiable without spawning
// processes.

#[cfg(test)]
mod tests {
    use super::*;

    fn unsafe_unset_all() {
        // SAFETY: tests are serialized by the outer mutex `lock()`
        // before touching env vars.
        unsafe {
            std::env::remove_var("CUBI_FAKE_LLM_MODEL_RESPONSES");
            std::env::remove_var("CUBI_FAKE_LLM_RESPONSE");
            std::env::remove_var("CUBI_FAKE_LLM_TOOL_CALL");
            std::env::remove_var("CUBI_FAKE_LLM_TOOL_CALL_REPEAT");
            std::env::remove_var("CUBI_FAKE_LLM_FAIL_MODELS");
        }
    }

    fn unsafe_set(key: &str, value: &str) {
        unsafe {
            std::env::set_var(key, value);
        }
    }

    fn executor() -> AIExecutor {
        // Build the executor directly with the Fake backend so the
        // tests don't have to flip the global `CUBI_FAKE_LLM` env var
        // (which would persist across the binary's other tests).
        AIExecutor::with_backend(crate::llm::LlmBackend::Fake, "default-model".to_string())
    }

    // Env-var manipulation is process-global; serialize tests that
    // touch the fake-backend scripting knobs. tokio::sync::Mutex so
    // the guard is await-safe (clippy: await_holding_lock).
    async fn lock() -> tokio::sync::MutexGuard<'static, ()> {
        use std::sync::OnceLock;
        use tokio::sync::Mutex;
        static M: OnceLock<Mutex<()>> = OnceLock::new();
        M.get_or_init(|| Mutex::new(())).lock().await
    }

    #[tokio::test]
    async fn vote_picks_majority() {
        let _g = lock().await;
        unsafe_unset_all();
        unsafe_set(
            "CUBI_FAKE_LLM_MODEL_RESPONSES",
            r#"{"m1":"answer A","m2":"answer A","m3":"answer B"}"#,
        );
        let req = ConsensusRequest {
            goal: "pick a letter".into(),
            models: vec!["m1".into(), "m2".into(), "m3".into()],
            strategy: ConsensusStrategy::Vote,
            max_steps_per_subagent: 1,
            concurrency: 0,
            use_tools: false,
        };
        let r = run(req, &executor(), &NullSink).await.unwrap();
        assert!(
            r.winner_model == "m1" || r.winner_model == "m2",
            "winner was {}",
            r.winner_model
        );
        assert_eq!(r.winner_output, "answer A");
        assert!(
            r.decision_reason.contains("majority vote 2/3"),
            "got: {}",
            r.decision_reason
        );
        assert_eq!(r.subagent_outputs.len(), 3);
        assert!(r.subagent_outputs.iter().all(|s| s.ok()));
    }

    #[tokio::test]
    async fn vote_normalizes_whitespace() {
        let _g = lock().await;
        unsafe_unset_all();
        unsafe_set(
            "CUBI_FAKE_LLM_MODEL_RESPONSES",
            r#"{"m1":"Hello   world","m2":"hello world","m3":"different"}"#,
        );
        let req = ConsensusRequest {
            goal: "x".into(),
            models: vec!["m1".into(), "m2".into(), "m3".into()],
            strategy: ConsensusStrategy::Vote,
            max_steps_per_subagent: 1,
            concurrency: 0,
            use_tools: false,
        };
        let r = run(req, &executor(), &NullSink).await.unwrap();
        assert!(
            r.decision_reason.contains("2/3"),
            "got: {}",
            r.decision_reason
        );
    }

    #[tokio::test]
    async fn best_of_n_picks_highest_score() {
        let _g = lock().await;
        unsafe_unset_all();
        unsafe_set(
            "CUBI_FAKE_LLM_MODEL_RESPONSES",
            r#"{"m1":"A","m2":"B","judge":"1: 4\n2: 9"}"#,
        );
        let req = ConsensusRequest {
            goal: "x".into(),
            models: vec!["m1".into(), "m2".into()],
            strategy: ConsensusStrategy::BestOfN {
                judge_model: "judge".into(),
            },
            max_steps_per_subagent: 1,
            concurrency: 0,
            use_tools: false,
        };
        let r = run(req, &executor(), &NullSink).await.unwrap();
        assert_eq!(r.winner_model, "m2");
        assert_eq!(r.winner_output, "B");
        assert!(
            r.decision_reason.contains("best-of-n"),
            "got: {}",
            r.decision_reason
        );
    }

    #[tokio::test]
    async fn judge_parses_pick_line() {
        let _g = lock().await;
        unsafe_unset_all();
        unsafe_set(
            "CUBI_FAKE_LLM_MODEL_RESPONSES",
            r#"{"m1":"A","m2":"B","judge":"pick: 2\nclear and concise"}"#,
        );
        let req = ConsensusRequest {
            goal: "x".into(),
            models: vec!["m1".into(), "m2".into()],
            strategy: ConsensusStrategy::Judge {
                judge_model: "judge".into(),
            },
            max_steps_per_subagent: 1,
            concurrency: 0,
            use_tools: false,
        };
        let r = run(req, &executor(), &NullSink).await.unwrap();
        assert_eq!(r.winner_model, "m2");
        assert!(
            r.decision_reason.contains("clear"),
            "got: {}",
            r.decision_reason
        );
    }

    #[tokio::test]
    async fn failed_subagent_does_not_abort_others() {
        let _g = lock().await;
        unsafe_unset_all();
        unsafe_set(
            "CUBI_FAKE_LLM_MODEL_RESPONSES",
            r#"{"m1":"A","m2":"A","m3":"B"}"#,
        );
        unsafe_set("CUBI_FAKE_LLM_FAIL_MODELS", "m3");
        let req = ConsensusRequest {
            goal: "x".into(),
            models: vec!["m1".into(), "m2".into(), "m3".into()],
            strategy: ConsensusStrategy::Vote,
            max_steps_per_subagent: 1,
            concurrency: 0,
            use_tools: false,
        };
        let r = run(req, &executor(), &NullSink).await.unwrap();
        assert_eq!(r.winner_output, "A");
        let failed = r.subagent_outputs.iter().find(|s| s.model == "m3").unwrap();
        assert!(failed.error.is_some(), "m3 should have errored");
        assert!(
            r.decision_reason.contains("2/2") || r.decision_reason.contains("2/3"),
            "got: {}",
            r.decision_reason
        );
    }

    #[tokio::test]
    async fn sequential_concurrency_matches_parallel() {
        let _g = lock().await;
        unsafe_unset_all();
        unsafe_set(
            "CUBI_FAKE_LLM_MODEL_RESPONSES",
            r#"{"m1":"A","m2":"A","m3":"B"}"#,
        );
        let req_seq = ConsensusRequest {
            goal: "x".into(),
            models: vec!["m1".into(), "m2".into(), "m3".into()],
            strategy: ConsensusStrategy::Vote,
            max_steps_per_subagent: 1,
            concurrency: 1,
            use_tools: false,
        };
        let req_par = ConsensusRequest {
            concurrency: 0,
            ..req_seq.clone()
        };
        let a = run(req_seq, &executor(), &NullSink).await.unwrap();
        let b = run(req_par, &executor(), &NullSink).await.unwrap();
        assert_eq!(a.winner_output, b.winner_output);
        assert_eq!(a.subagent_outputs.len(), b.subagent_outputs.len());
    }

    #[tokio::test]
    async fn vote_tie_escalates_to_judge() {
        let _g = lock().await;
        unsafe_unset_all();
        unsafe_set(
            "CUBI_FAKE_LLM_MODEL_RESPONSES",
            // m1 (used as judge fallback) returns a pick line because
            // it's the first model in the request.
            r#"{"m1":"pick: 2\ngood","m2":"B"}"#,
        );
        let req = ConsensusRequest {
            goal: "x".into(),
            // Two outputs that disagree → tie → escalate.
            models: vec!["m1".into(), "m2".into()],
            strategy: ConsensusStrategy::Vote,
            max_steps_per_subagent: 1,
            concurrency: 0,
            use_tools: false,
        };
        let r = run(req, &executor(), &NullSink).await.unwrap();
        assert!(
            r.decision_reason.contains("tie"),
            "got: {}",
            r.decision_reason
        );
    }

    #[tokio::test]
    async fn aggregate_stats_sums_subagents() {
        let _g = lock().await;
        unsafe_unset_all();
        unsafe_set("CUBI_FAKE_LLM_MODEL_RESPONSES", r#"{"m1":"A","m2":"A"}"#);
        let req = ConsensusRequest {
            goal: "x".into(),
            models: vec!["m1".into(), "m2".into()],
            strategy: ConsensusStrategy::Vote,
            max_steps_per_subagent: 1,
            concurrency: 0,
            use_tools: false,
        };
        let r = run(req, &executor(), &NullSink).await.unwrap();
        let agg = r.aggregate_stats();
        // Fake stats are 1/1/1 each, two successful calls → 2/2.
        assert_eq!(agg.prompt_tokens, 2);
        assert_eq!(agg.completion_tokens, 2);
    }

    #[tokio::test]
    async fn use_tools_false_keeps_llm_only_path_with_tools_entrypoint() {
        let _g = lock().await;
        unsafe_unset_all();
        unsafe_set("CUBI_FAKE_LLM_MODEL_RESPONSES", r#"{"m1":"A","m2":"A"}"#);
        let req = ConsensusRequest {
            goal: "x".into(),
            models: vec!["m1".into(), "m2".into()],
            strategy: ConsensusStrategy::Vote,
            max_steps_per_subagent: 8,
            concurrency: 0,
            use_tools: false,
        };
        let direct = run(req.clone(), &executor(), &NullSink).await.unwrap();
        let mut no_mcp = None;
        let via_tools_entrypoint = run_with_tools(req, &executor(), &NullSink, &mut no_mcp)
            .await
            .unwrap();

        assert_eq!(via_tools_entrypoint.winner_model, direct.winner_model);
        assert_eq!(via_tools_entrypoint.winner_output, direct.winner_output);
        assert_eq!(via_tools_entrypoint.decision_reason, direct.decision_reason);
        assert_eq!(
            via_tools_entrypoint.requested_strategy,
            direct.requested_strategy
        );
        assert_eq!(
            via_tools_entrypoint
                .subagent_outputs
                .iter()
                .map(|s| (
                    &s.model,
                    &s.output,
                    s.steps_used,
                    s.prompt_tokens,
                    s.completion_tokens
                ))
                .collect::<Vec<_>>(),
            direct
                .subagent_outputs
                .iter()
                .map(|s| (
                    &s.model,
                    &s.output,
                    s.steps_used,
                    s.prompt_tokens,
                    s.completion_tokens
                ))
                .collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn tool_mode_requires_active_mcp_manager() {
        let _g = lock().await;
        unsafe_unset_all();
        let req = ConsensusRequest {
            goal: "x".into(),
            models: vec!["m1".into(), "m2".into()],
            strategy: ConsensusStrategy::Vote,
            max_steps_per_subagent: 8,
            concurrency: 0,
            use_tools: true,
        };
        let mut no_mcp = None;
        let err = run_with_tools(req, &executor(), &NullSink, &mut no_mcp)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("active MCP manager"), "got: {err}");
    }

    #[tokio::test]
    async fn tool_mode_runner_is_sequential_and_accounts_steps() {
        let _g = lock().await;
        unsafe_unset_all();
        unsafe_set(
            "CUBI_FAKE_LLM_MODEL_RESPONSES",
            r#"{"m1":"final A","m2":"final B"}"#,
        );
        unsafe_set(
            "CUBI_FAKE_LLM_TOOL_CALL",
            r#"{"id":"call_meta","type":"function","function":{"name":"agent_run","arguments":{"goal":"nested work"}}}"#,
        );
        let req = ConsensusRequest {
            goal: "x".into(),
            models: vec!["m1".into(), "m2".into()],
            strategy: ConsensusStrategy::Vote,
            max_steps_per_subagent: 8,
            concurrency: 99,
            use_tools: true,
        };
        let mut no_mcp = None;
        let outputs = run_tool_subagents(&req, &executor(), &mut no_mcp).await;

        assert_eq!(
            outputs.iter().map(|s| s.model.as_str()).collect::<Vec<_>>(),
            vec!["m1", "m2"]
        );
        assert_eq!(outputs[0].output, "final A");
        assert_eq!(outputs[1].output, "final B");
        assert!(outputs.iter().all(|s| s.error.is_none()));
        assert!(outputs.iter().all(|s| s.steps_used == 2));
        assert!(outputs.iter().all(|s| s.prompt_tokens == 2));
        assert!(outputs.iter().all(|s| s.completion_tokens == 2));
    }

    #[test]
    fn parse_pick_handles_trailing_reasoning() {
        let (i, r) = parse_pick("pick: 2\nthe second is correct because…", 3);
        assert_eq!(i, 1);
        assert!(r.contains("second"));
    }

    #[test]
    fn parse_pick_defaults_on_garbage() {
        let (i, r) = parse_pick("I cannot decide.", 2);
        assert_eq!(i, 0);
        assert!(r.contains("unparseable"));
    }

    #[test]
    fn parse_scores_extracts_per_line_scores() {
        let s = parse_scores("1: 7\n2: 9\n3: 4", 3);
        assert_eq!(s, vec![7.0, 9.0, 4.0]);
    }

    #[test]
    fn hash_normalized_collapses_whitespace_and_case() {
        assert_eq!(
            hash_normalized("Hello\nworld"),
            hash_normalized("  hello   world  ")
        );
        assert_ne!(hash_normalized("hello"), hash_normalized("world"));
    }

    #[test]
    fn consensus_run_spec_has_min_two_models() {
        let spec = consensus_run_spec();
        assert_eq!(spec.function.name, crate::agent_loop::CONSENSUS_TOOL_NAME);
        assert_eq!(
            spec.function.parameters["properties"]["models"]["minItems"],
            2
        );
        assert!(
            !spec.function.parameters["properties"]["use_tools"]["default"]
                .as_bool()
                .unwrap_or(true)
        );
        assert!(
            spec.function.parameters["required"]
                .as_array()
                .map(|a| a.iter().any(|v| v == "goal") && a.iter().any(|v| v == "models"))
                .unwrap_or(false)
        );
    }
}
