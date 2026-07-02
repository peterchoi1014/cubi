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
//! loop (sharing the live `McpManager` and working tree) *unless*
//! [`ConsensusRequest::isolate`] is also set — see the isolated mode below.
//!
//! ## Why in-process tool mode is sequential (do not "optimize" into parallelism)
//!
//! It is tempting to parallelize tool-enabled subagents (as the LLM-only
//! path does) by wrapping the single `McpManager` in an async mutex and
//! locking only around each tool call. **Do not do this.** A per-call
//! mutex serializes individual tool calls but does NOT isolate a
//! subagent's multi-step edit sequence: N subagents share ONE process
//! working tree, so their `edit_file`/`write_file`/`bash` effects
//! interleave (A reads a file, B rewrites it, A edits a stale view, B
//! runs tests against A's half-applied change …). That is a logical
//! filesystem race the mutex cannot prevent. Real parallelism requires
//! per-subagent workspace isolation, which the process-global current
//! directory makes impossible in-process.
//!
//! Note that in-process tool mode is therefore **side-effecting**: every
//! subagent's tool actions (including losing candidates') persist in the
//! working tree. Consensus selects one *text* answer; it is not isolated
//! best-of-N execution.
//!
//! ## Isolated tool mode (`isolate: true`)
//!
//! Setting [`ConsensusRequest::isolate`] (requires `use_tools: true`)
//! sidesteps the shared-working-tree constraint above entirely: each
//! subagent gets its own throwaway git worktree
//! ([`crate::worktree_session`]) and runs inside it via a headless `cubi`
//! subprocess with an isolated `HOME`
//! ([`crate::proc_subagent::run_isolated_subagent`]), like the
//! `bench`/`swebench` harnesses. Because no two subagents ever touch the
//! same working tree, `run_isolated_tool_subagents` dispatches them with
//! a low-default, capped `FuturesUnordered` + semaphore pattern respecting
//! [`ConsensusRequest::concurrency`]. Each worktree (and its
//! branch) is torn down once its subagent finishes, so only the winner's
//! *text* answer survives — losing candidates' file edits are discarded
//! along with their worktrees. This is real isolated best-of-N execution.
//!
//! ## Anti-recursion
//!
//! `consensus_run` is stripped from every subagent's tool list. In-process
//! subagents use [`agent_loop::without_meta_tools`]; isolated subprocess
//! subagents set [`agent_loop::DISABLE_META_TOOLS_ENV`] before starting the
//! child so the top-level tool builder omits `agent_run` and `consensus_run`
//! too. Both paths also reject explicit nested meta-tool calls defensively.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use anyhow::Result;
use futures_util::future::BoxFuture;
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

const CONSENSUS_ISOLATED_DEFAULT_CONCURRENCY: usize = 1;
const CONSENSUS_ISOLATED_MAX_CONCURRENCY: usize = 2;

/// Normalize caller-supplied per-subagent step caps before dispatch.
///
/// Shared in-process subagents already clamp to at least one step in
/// [`crate::agent_loop::run_subagent_with_model`]. Isolated subprocess
/// subagents must receive the same positive lower bound before the hidden CLI
/// argument is built; `--internal-max-steps 0` exits during flag parsing before
/// it can emit the structured JSON final/error/done events the parent expects.
pub(crate) fn normalize_max_steps_per_subagent(max_steps: usize) -> usize {
    max_steps.max(1)
}

fn isolated_permits(concurrency: usize, models: usize) -> usize {
    let requested = if concurrency == 0 {
        CONSENSUS_ISOLATED_DEFAULT_CONCURRENCY
    } else {
        concurrency.min(CONSENSUS_ISOLATED_MAX_CONCURRENCY)
    };
    requested.min(models.max(1)).max(1)
}

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
    /// Maximum number of subagents in flight at once. In LLM-only mode,
    /// `0` means `models.len()`; isolated tool mode uses its own safer
    /// default and cap.
    pub concurrency: usize,
    /// When false (the default), each subagent is a single LLM-only call.
    /// When true, each subagent may use tools and runs sequentially
    /// (unless `isolate` is also set — see below).
    pub use_tools: bool,
    /// Opt in to per-subagent workspace isolation for tool-enabled
    /// consensus. Ignored unless `use_tools` is also true (validated in
    /// [`run_inner`]). When true, each subagent runs in its own ephemeral
    /// git worktree ([`crate::worktree_session`]) driven by a headless
    /// `cubi` subprocess with an isolated `HOME`
    /// ([`crate::proc_subagent::run_isolated_subagent`]) instead of the
    /// shared in-process `McpManager`. Because each subagent's tool
    /// actions are confined to its own worktree, subagents can run in
    /// parallel (respecting `concurrency`) without the interleaved-edit
    /// race described in the module docs above. This is the isolation
    /// scoped as "future work" in that note.
    pub isolate: bool,
    /// Wall-clock cap per isolated subprocess subagent. Ignored unless
    /// `isolate` is true. `0` is clamped to
    /// [`CONSENSUS_ISOLATED_DEFAULT_TIME_CAP_SECS`].
    pub isolated_time_cap_secs: u64,
}

/// Default wall-clock budget for one isolated (worktree + subprocess)
/// tool-enabled subagent, used when
/// [`ConsensusRequest::isolated_time_cap_secs`] is `0`.
pub const CONSENSUS_ISOLATED_DEFAULT_TIME_CAP_SECS: u64 = 300;

/// One subagent's output. Recorded for every model in `models` —
/// successes and failures alike — so the caller can show a per-model
/// summary table.
#[derive(Debug, Clone, Serialize)]
pub struct SubagentOutput {
    pub model: String,
    pub output: String,
    pub steps_used: usize,
    /// Tool calls observed for this subagent. LLM-only subagents report `0`;
    /// tool-enabled subagents report the runner's observed tool-call count.
    pub tool_calls: usize,
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
    #[serde(skip_serializing_if = "ChatStats::is_empty")]
    pub arbitration_stats: ChatStats,
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
        let mut s = self.arbitration_stats.clone();
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
                          use_tools=true to let subagents use tools; add isolate=true for \
                          parallel tool use in throwaway git worktrees."
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
                        "description": "Models to consult. LLM-only mode may run them in parallel; shared-tree tool mode runs sequentially; isolated tool mode uses capped `concurrency`."
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
                        "minimum": 1,
                        "default": CONSENSUS_DEFAULT_MAX_STEPS,
                        "description": "Positive per-subagent agent-loop step cap (default 8)."
                    },
                    "concurrency": {
                        "type": "integer",
                        "default": 0,
                        "description": format!("0 = LLM-only parallel (len(models)); 1 = sequential. In shared-tree tool mode (use_tools=true, isolate=false), subagents are forced sequential regardless of this value. WARNING: isolated tool mode spawns a full model-driving `cubi` subprocess per subagent, is memory-heavy, defaults to sequential ({}), and caps requested concurrency at {}.", CONSENSUS_ISOLATED_DEFAULT_CONCURRENCY, CONSENSUS_ISOLATED_MAX_CONCURRENCY)
                    },
                    "use_tools": {
                        "type": "boolean",
                        "default": false,
                        "description": "Opt in to tool-enabled subagents. Defaults to false. When true and `isolate` is false, subagents run SEQUENTIALLY and their tool actions are side-effecting: every subagent (including losing candidates) mutates the shared working tree. Consensus still selects one text answer; it is not isolated best-of-N execution. Set `isolate=true` to confine each tool-enabled subagent to its own throwaway git worktree."
                    },
                    "isolate": {
                        "type": "boolean",
                        "default": false,
                        "description": format!("Requires use_tools=true. Runs each tool-enabled subagent in its own ephemeral git worktree via a headless `cubi` subprocess, so subagents can run without their tool actions interleaving on a shared working tree. WARNING: isolate mode spawns a full model-driving subprocess per subagent, is memory-heavy, defaults to sequential ({}), and caps requested concurrency at {}. Losing candidates' worktrees are discarded, so only the winner's reasoning (not its file edits) survives — this is isolated best-of-N execution, unlike the shared-tree sequential mode.", CONSENSUS_ISOLATED_DEFAULT_CONCURRENCY, CONSENSUS_ISOLATED_MAX_CONCURRENCY)
                    },
                    "isolated_time_cap_secs": {
                        "type": "integer",
                        "default": CONSENSUS_ISOLATED_DEFAULT_TIME_CAP_SECS,
                        "description": "Ignored unless `isolate` is true. Wall-clock cap per isolated subprocess subagent; the subagent is killed and recorded as a timeout error if exceeded."
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
/// If [`ConsensusRequest::use_tools`] is true without isolation, call
/// [`run_with_tools`] instead so the live MCP manager can be threaded
/// through the sequential shared-tree tool loop.
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
/// unchanged LLM-only path. When it is true and
/// [`ConsensusRequest::isolate`] is false, subagents are run sequentially
/// regardless of [`ConsensusRequest::concurrency`] because tool execution
/// uses one live mutable [`McpManager`] and one shared working tree. Isolated
/// tool mode ignores the MCP manager and runs subprocess subagents in
/// throwaway worktrees, bounded by `concurrency`.
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
    run_inner_with_isolated_runner(req, executor, sink, mcp, None).await
}

async fn run_inner_with_isolated_runner(
    mut req: ConsensusRequest,
    executor: &AIExecutor,
    sink: &dyn ConsensusEventSink,
    mcp: Option<&mut Option<McpManager>>,
    isolated_runner: Option<IsolatedSubagentRunner>,
) -> Result<ConsensusResult> {
    req.max_steps_per_subagent = normalize_max_steps_per_subagent(req.max_steps_per_subagent);
    if req.models.len() < 2 {
        anyhow::bail!(
            "consensus_run requires at least 2 models, got {}",
            req.models.len()
        );
    }
    if req.isolate && !req.use_tools {
        anyhow::bail!(
            "consensus_run: `isolate` requires `use_tools=true` (LLM-only mode has no \
             side effects to isolate)"
        );
    }
    if req.use_tools && !req.isolate {
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

    let subagent_outputs = if req.use_tools && req.isolate {
        if let Some(runner) = isolated_runner {
            run_isolated_tool_subagents_with_runner(&req, runner).await
        } else {
            run_isolated_tool_subagents(&req).await?
        }
    } else if req.use_tools {
        let Some(mcp) = mcp else {
            anyhow::bail!("tool-enabled consensus requires an MCP manager; call run_with_tools");
        };
        run_tool_subagents(&req, executor, mcp).await
    } else {
        run_llm_subagents(&req, executor).await
    };

    for sub in &subagent_outputs {
        sink.subagent_result(sub);
    }

    let successes: Vec<&SubagentOutput> = subagent_outputs.iter().filter(|s| s.ok()).collect();
    if successes.is_empty() {
        let details = subagent_outputs
            .iter()
            .filter_map(|sub| {
                sub.error
                    .as_ref()
                    .map(|err| format!("{}: {err}", sub.model))
            })
            .collect::<Vec<_>>()
            .join("; ");
        let suffix = if details.is_empty() {
            String::new()
        } else {
            format!(": {}", truncate(&details, 1000))
        };
        anyhow::bail!(
            "consensus_run: every subagent failed ({} models){}",
            subagent_outputs.len(),
            suffix
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
            arbitration_stats: ChatStats::default(),
            requested_strategy: strategy_label.to_string(),
        });
    }

    let decision = arbitrate(&req, executor, &subagent_outputs, &successes).await;
    let winner = &subagent_outputs[decision.winner_idx];
    sink.decision(&winner.model, &decision.reason);
    Ok(ConsensusResult {
        winner_model: winner.model.clone(),
        winner_output: winner.output.clone(),
        subagent_outputs: subagent_outputs.clone(),
        decision_reason: decision.reason,
        arbitration_stats: decision.stats,
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
            let started = Instant::now();
            let Ok(_permit) = sem.acquire_owned().await else {
                let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
                return (
                    idx,
                    SubagentOutput {
                        model,
                        output: String::new(),
                        steps_used: 0,
                        tool_calls: 0,
                        elapsed_ms,
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        error: Some("consensus dispatcher semaphore closed".to_string()),
                    },
                );
            };
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
                    tool_calls: 0,
                    elapsed_ms: elapsed_ms.max(stats.elapsed_ms),
                    prompt_tokens: stats.prompt_tokens,
                    completion_tokens: stats.completion_tokens,
                    error: None,
                },
                Err(e) => SubagentOutput {
                    model: model.clone(),
                    output: String::new(),
                    steps_used: 0,
                    tool_calls: 0,
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

/// Run tool-enabled consensus subagents **sequentially**.
///
/// This loop is intentionally sequential and must stay that way: see the
/// module-level "Why tool mode is sequential" note. A plain `for` over the
/// models (rather than the LLM-only path's `FuturesUnordered`) guarantees
/// one subagent's tool actions fully complete before the next begins, so
/// their edits to the shared working tree cannot interleave. `concurrency`
/// is deliberately not consulted here.
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
                tool_calls: result.tool_calls,
                elapsed_ms: elapsed_ms.max(result.stats.elapsed_ms),
                prompt_tokens: result.stats.prompt_tokens,
                completion_tokens: result.stats.completion_tokens,
                error: None,
            },
            Err(e) => SubagentOutput {
                model: model.clone(),
                output: String::new(),
                steps_used: 0,
                tool_calls: 0,
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

/// Run tool-enabled consensus subagents **in parallel**, each confined to
/// its own ephemeral git worktree and driven by a headless `cubi`
/// subprocess with an isolated `HOME` (see [`crate::worktree_session`] and
/// [`crate::proc_subagent`]). Unlike [`run_tool_subagents`], parallelism
/// here is safe because no two subagents ever touch the same working
/// tree — this is the isolation the module-level "Why tool mode is
/// sequential" note scopes as future work.
///
/// Per-model failures (worktree provisioning, subprocess spawn, timeout,
/// non-zero exit, empty output) are recorded as
/// `SubagentOutput { error: Some(_), .. }`
/// rather than aborting the whole run, matching the LLM-only and
/// sequential-tool paths.
type IsolatedSubagentRunner = Arc<
    dyn Fn(
            String,
            String,
            Duration,
            usize,
        ) -> BoxFuture<'static, Result<crate::proc_subagent::ProcSubagentResult>>
        + Send
        + Sync,
>;

fn real_isolated_subagent_runner(context: IsolatedRepoContext) -> IsolatedSubagentRunner {
    Arc::new(move |model, goal, time_cap, max_steps| {
        let context = context.clone();
        Box::pin(async move {
            run_one_isolated_subagent_in_repo_with_trust_root(
                &context.repo_top,
                &context.relative_cwd,
                &context.trusted_root_relative,
                "HEAD",
                &model,
                &goal,
                time_cap,
                max_steps,
            )
            .await
        })
    })
}

#[derive(Debug, Clone)]
struct IsolatedRepoContext {
    repo_top: PathBuf,
    relative_cwd: PathBuf,
    trusted_root_relative: PathBuf,
}

async fn run_isolated_tool_subagents(req: &ConsensusRequest) -> Result<Vec<SubagentOutput>> {
    let context = prepare_isolated_repo_context().await?;
    Ok(run_isolated_tool_subagents_with_runner(req, real_isolated_subagent_runner(context)).await)
}

async fn run_isolated_tool_subagents_with_runner(
    req: &ConsensusRequest,
    runner: IsolatedSubagentRunner,
) -> Vec<SubagentOutput> {
    let permits = isolated_permits(req.concurrency, req.models.len());
    let semaphore = Arc::new(Semaphore::new(permits));
    let time_cap_secs = if req.isolated_time_cap_secs == 0 {
        CONSENSUS_ISOLATED_DEFAULT_TIME_CAP_SECS
    } else {
        req.isolated_time_cap_secs
    };
    let time_cap = std::time::Duration::from_secs(time_cap_secs);
    let max_steps_per_subagent = normalize_max_steps_per_subagent(req.max_steps_per_subagent);

    let mut in_flight: FuturesUnordered<_> = FuturesUnordered::new();
    for (idx, model) in req.models.iter().cloned().enumerate() {
        let sem = Arc::clone(&semaphore);
        let runner = Arc::clone(&runner);
        let goal = req.goal.clone();
        let max_steps = max_steps_per_subagent;
        let fut = async move {
            // Permit is dropped when the closure returns, releasing the
            // slot for the next queued subagent.
            let started = Instant::now();
            let Ok(_permit) = sem.acquire_owned().await else {
                let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
                return (
                    idx,
                    SubagentOutput {
                        model,
                        output: String::new(),
                        steps_used: 0,
                        tool_calls: 0,
                        elapsed_ms,
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        error: Some("isolated consensus dispatcher semaphore closed".to_string()),
                    },
                );
            };
            let res = (runner)(model.clone(), goal, time_cap, max_steps).await;
            let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
            let sub = match res {
                Ok(proc) if proc.timed_out => SubagentOutput {
                    model: model.clone(),
                    output: String::new(),
                    steps_used: proc.steps_used,
                    tool_calls: proc.tool_calls,
                    elapsed_ms,
                    prompt_tokens: proc.prompt_tokens,
                    completion_tokens: proc.completion_tokens,
                    error: Some(proc_error_message(
                        format!("isolated subagent timed out after {time_cap_secs}s"),
                        &proc,
                    )),
                },
                Ok(proc) if proc.error.is_some() => SubagentOutput {
                    model: model.clone(),
                    output: String::new(),
                    steps_used: proc.steps_used,
                    tool_calls: proc.tool_calls,
                    elapsed_ms,
                    prompt_tokens: proc.prompt_tokens,
                    completion_tokens: proc.completion_tokens,
                    error: Some(proc_error_message(
                        "isolated subagent reported error".to_string(),
                        &proc,
                    )),
                },
                Ok(proc) if proc.exit_code != Some(0) => SubagentOutput {
                    model: model.clone(),
                    output: String::new(),
                    steps_used: proc.steps_used,
                    tool_calls: proc.tool_calls,
                    elapsed_ms,
                    prompt_tokens: proc.prompt_tokens,
                    completion_tokens: proc.completion_tokens,
                    error: Some(proc_error_message(
                        format!("isolated subagent exited with status {:?}", proc.exit_code),
                        &proc,
                    )),
                },
                Ok(proc) if proc.output.trim().is_empty() => SubagentOutput {
                    model: model.clone(),
                    output: String::new(),
                    steps_used: proc.steps_used,
                    tool_calls: proc.tool_calls,
                    elapsed_ms,
                    prompt_tokens: proc.prompt_tokens,
                    completion_tokens: proc.completion_tokens,
                    error: Some(proc_error_message(
                        format!(
                            "isolated subagent produced empty output (exit code {:?})",
                            proc.exit_code
                        ),
                        &proc,
                    )),
                },
                Ok(proc) => SubagentOutput {
                    model: model.clone(),
                    output: proc.output,
                    steps_used: proc.steps_used,
                    tool_calls: proc.tool_calls,
                    elapsed_ms,
                    prompt_tokens: proc.prompt_tokens,
                    completion_tokens: proc.completion_tokens,
                    error: None,
                },
                Err(e) => SubagentOutput {
                    model: model.clone(),
                    output: String::new(),
                    steps_used: 0,
                    tool_calls: 0,
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

fn proc_error_message(base: String, proc: &crate::proc_subagent::ProcSubagentResult) -> String {
    let diagnostics = proc.diagnostics();
    if diagnostics.trim().is_empty() {
        truncate(&base, 1000)
    } else {
        truncate(&format!("{base}: {diagnostics}"), 1000)
    }
}

async fn prepare_isolated_repo_context() -> Result<IsolatedRepoContext> {
    tokio::task::spawn_blocking(|| {
        let cwd =
            std::env::current_dir().context("read current directory for isolated subagent")?;
        let context = crate::worktree_session::resolve_repo_context(&cwd)?;
        let permissions = crate::permissions::Permissions::load();
        let trusted_root_relative =
            translated_isolated_trust_root_relative(&cwd, &context.top_level, &permissions)?;
        crate::worktree_session::ensure_clean_worktree(&context.top_level).with_context(|| {
            format!(
                "Refusing isolated tool consensus from `{}`",
                context.top_level.display()
            )
        })?;
        Ok::<_, anyhow::Error>(IsolatedRepoContext {
            repo_top: context.top_level,
            relative_cwd: context.relative_cwd,
            trusted_root_relative,
        })
    })
    .await
    .context("join isolated repository preflight task")?
}

fn translated_isolated_trust_root_relative(
    cwd: &Path,
    repo_top: &Path,
    permissions: &crate::permissions::Permissions,
) -> Result<PathBuf> {
    let cwd = std::fs::canonicalize(cwd)
        .with_context(|| format!("canonicalize cwd `{}`", cwd.display()))?;
    let repo_top = std::fs::canonicalize(repo_top)
        .with_context(|| format!("canonicalize git top-level `{}`", repo_top.display()))?;

    let mut best: Option<(usize, PathBuf)> = None;
    for trusted_root in permissions.trusted_roots() {
        let Ok(trusted_root) = std::fs::canonicalize(trusted_root) else {
            continue;
        };
        if !cwd.starts_with(&trusted_root) {
            continue;
        }
        let translated = if trusted_root.starts_with(&repo_top) {
            trusted_root
                .strip_prefix(&repo_top)
                .with_context(|| {
                    format!(
                        "translate trusted root `{}` under git top-level `{}`",
                        trusted_root.display(),
                        repo_top.display()
                    )
                })?
                .to_path_buf()
        } else if repo_top.starts_with(&trusted_root) {
            PathBuf::new()
        } else {
            continue;
        };
        let depth = trusted_root.components().count();
        match &mut best {
            Some((best_depth, best_path)) if depth < *best_depth => {
                *best_depth = depth;
                *best_path = translated;
            }
            None => best = Some((depth, translated)),
            _ => {}
        }
    }

    best.map(|(_, relative)| relative).ok_or_else(|| {
        anyhow::anyhow!(
            "Refusing isolated tool consensus: `{}` is not under a trusted root that can be \
             translated into git worktree `{}`. Run `/trust` in the project directory to \
             approve it.",
            cwd.display(),
            repo_top.display()
        )
    })
}

#[allow(dead_code)]
async fn run_one_isolated_subagent_in_repo(
    repo_dir: &Path,
    relative_cwd: &Path,
    base_ref: &str,
    model: &str,
    goal: &str,
    time_cap: Duration,
    max_steps: usize,
) -> Result<crate::proc_subagent::ProcSubagentResult> {
    run_one_isolated_subagent_in_repo_with_trust_root(
        repo_dir,
        relative_cwd,
        relative_cwd,
        base_ref,
        model,
        goal,
        time_cap,
        max_steps,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn run_one_isolated_subagent_in_repo_with_trust_root(
    repo_dir: &Path,
    relative_cwd: &Path,
    trusted_root_relative: &Path,
    base_ref: &str,
    model: &str,
    goal: &str,
    time_cap: Duration,
    max_steps: usize,
) -> Result<crate::proc_subagent::ProcSubagentResult> {
    let cubi_bin = crate::proc_subagent::resolve_cubi_binary();
    run_one_isolated_subagent_in_repo_with_binary_and_trust_root(
        repo_dir,
        relative_cwd,
        trusted_root_relative,
        base_ref,
        model,
        goal,
        time_cap,
        max_steps,
        &cubi_bin,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
async fn run_one_isolated_subagent_in_repo_with_binary(
    repo_dir: &Path,
    relative_cwd: &Path,
    base_ref: &str,
    model: &str,
    goal: &str,
    time_cap: Duration,
    max_steps: usize,
    cubi_bin: &Path,
) -> Result<crate::proc_subagent::ProcSubagentResult> {
    run_one_isolated_subagent_in_repo_with_binary_and_trust_root(
        repo_dir,
        relative_cwd,
        relative_cwd,
        base_ref,
        model,
        goal,
        time_cap,
        max_steps,
        cubi_bin,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn run_one_isolated_subagent_in_repo_with_binary_and_trust_root(
    repo_dir: &Path,
    relative_cwd: &Path,
    trusted_root_relative: &Path,
    base_ref: &str,
    model: &str,
    goal: &str,
    time_cap: Duration,
    max_steps: usize,
    cubi_bin: &Path,
) -> Result<crate::proc_subagent::ProcSubagentResult> {
    let label = model.to_string();
    let repo_dir = repo_dir.to_path_buf();
    let base_ref = base_ref.to_string();
    // `worktree_session::create` shells out to `git` synchronously; run it
    // on a blocking thread so it doesn't stall the async executor.
    let session = tokio::task::spawn_blocking(move || {
        crate::worktree_session::create_in(&repo_dir, &base_ref, &label)
    })
    .await
    .context("join worktree provisioning task")??;
    // Wrap in a cancel-safe guard: if this `.await` below is dropped before
    // completing (e.g. a caller races this future against Ctrl-C/timeout
    // and drops the loser), the guard's `Drop` schedules teardown onto a
    // blocking thread instead of running synchronous `git` commands inline
    // on whatever thread is driving cancellation. See `CancelSafeWorktreeSession`.
    let mut session = CancelSafeWorktreeSession::new(session);

    let child_workdir = session.path()?.join(relative_cwd);
    if !child_workdir.is_dir() {
        let missing = child_workdir.display().to_string();
        session.cleanup().await?;
        anyhow::bail!(
            "isolated worktree does not contain requested cwd `{}`; \
             ensure the current directory is committed before using isolated consensus",
            missing
        );
    }
    let child_trusted_root = session.path()?.join(trusted_root_relative);
    if !child_trusted_root.is_dir() {
        let missing = child_trusted_root.display().to_string();
        session.cleanup().await?;
        anyhow::bail!(
            "isolated worktree does not contain translated trusted root `{}`; \
             ensure the trusted project root is committed before using isolated consensus",
            missing
        );
    }

    let result = crate::proc_subagent::run_isolated_subagent_with_binary_and_trust_root(
        cubi_bin,
        model,
        goal,
        &child_workdir,
        &child_trusted_root,
        time_cap,
        max_steps,
    )
    .await;

    // Happy path (success, error, or timeout all return normally from the
    // call above): tear the worktree down on a blocking thread and await
    // completion so the worktree/branch are gone before we return.
    session.cleanup().await?;

    result
}

/// Guards a [`crate::worktree_session::WorktreeSession`] so that cancelling
/// the future holding it (e.g. dropping it out of a `select!`/`abort()`)
/// never runs the session's blocking `git worktree remove`/`git branch -D`
/// teardown inline on the thread driving the drop. That thread is often a
/// Tokio worker thread, and blocking it stalls every other task on the
/// runtime until the git commands finish.
///
/// Call [`CancelSafeWorktreeSession::cleanup`] on the normal/happy path to
/// await teardown on a blocking thread before returning. If the guard is
/// instead dropped without calling `cleanup` (cancellation), teardown is
/// scheduled best-effort via `spawn_blocking` and not waited on. If no Tokio
/// runtime is available at drop time (e.g. in a plain synchronous context),
/// this falls back to inline synchronous cleanup — there is no runtime to
/// offload the blocking work to.
struct CancelSafeWorktreeSession(Option<crate::worktree_session::WorktreeSession>);

impl CancelSafeWorktreeSession {
    fn new(session: crate::worktree_session::WorktreeSession) -> Self {
        Self(Some(session))
    }

    fn path(&self) -> Result<&Path> {
        self.0
            .as_ref()
            .map(crate::worktree_session::WorktreeSession::path)
            .context("CancelSafeWorktreeSession used after cleanup")
    }

    /// Explicit async cleanup for the happy path. Takes the session out (so
    /// `Drop` becomes a no-op afterwards, avoiding double cleanup), tears it
    /// down on a blocking thread, and awaits completion.
    async fn cleanup(&mut self) -> Result<()> {
        if let Some(session) = self.0.take() {
            tokio::task::spawn_blocking(move || drop(session))
                .await
                .context("join worktree teardown task")?;
        }
        Ok(())
    }
}

impl Drop for CancelSafeWorktreeSession {
    fn drop(&mut self) {
        let Some(session) = self.0.take() else {
            return;
        };
        match tokio::runtime::Handle::try_current() {
            // Best-effort: fire-and-forget the blocking teardown. We're in
            // `Drop`, so we cannot `.await` here even if we wanted to.
            Ok(handle) => {
                handle.spawn_blocking(move || drop(session));
            }
            // No runtime to offload to (e.g. this guard outlived the async
            // runtime). Fall back to inline synchronous cleanup — there is
            // no worker thread being blocked in this case.
            Err(_) => drop(session),
        }
    }
}

struct ArbitrationDecision {
    winner_idx: usize,
    reason: String,
    stats: ChatStats,
}

impl ArbitrationDecision {
    fn new(winner_idx: usize, reason: String) -> Self {
        Self {
            winner_idx,
            reason,
            stats: ChatStats::default(),
        }
    }

    fn with_stats(winner_idx: usize, reason: String, stats: ChatStats) -> Self {
        Self {
            winner_idx,
            reason,
            stats,
        }
    }
}

/// Returns the winning index, reason, and any LLM usage spent arbitrating.
async fn arbitrate(
    req: &ConsensusRequest,
    executor: &AIExecutor,
    all: &[SubagentOutput],
    successes: &[&SubagentOutput],
) -> ArbitrationDecision {
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
) -> ArbitrationDecision {
    let mut buckets: HashMap<String, Vec<usize>> = HashMap::new();
    for sub in successes {
        let key = hash_normalized(&sub.output);
        if let Some(global_idx) = all
            .iter()
            .position(|s| std::ptr::eq::<SubagentOutput>(s, *sub))
        {
            buckets.entry(key).or_default().push(global_idx);
        }
    }
    let max_count = buckets.values().map(|v| v.len()).max().unwrap_or(0);
    let top: Vec<&Vec<usize>> = buckets.values().filter(|v| v.len() == max_count).collect();

    if top.len() == 1 {
        let winner_idx = top[0][0];
        let total = successes.len();
        return ArbitrationDecision::new(winner_idx, format!("majority vote {max_count}/{total}"));
    }
    if top.is_empty() {
        return ArbitrationDecision::new(
            0,
            "vote had no indexable successful candidates".to_string(),
        );
    }
    // Tie: escalate to a Judge call using the first model in `models`.
    let judge_model = models
        .first()
        .cloned()
        .unwrap_or_else(|| executor.get_model().to_string());
    let decision = judge(all, successes, &judge_model, executor, goal).await;
    ArbitrationDecision::with_stats(
        decision.winner_idx,
        format!(
            "vote tie ({max_count}-way); escalated to judge `{judge_model}`: {}",
            decision.reason
        ),
        decision.stats,
    )
}

async fn best_of_n(
    all: &[SubagentOutput],
    successes: &[&SubagentOutput],
    judge_model: &str,
    executor: &AIExecutor,
    goal: &str,
) -> ArbitrationDecision {
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
    let (raw, stats) = match executor.chat_with_model(judge_model, messages).await {
        Ok((msg, stats)) => (msg.content, stats),
        Err(e) => {
            // Judge failed → fall back to Vote; if Vote needs its own
            // tie-breaker, preserve that arbitration usage too.
            let vote_decision =
                vote(all, successes, &[judge_model.to_string()], executor, goal).await;
            return ArbitrationDecision::with_stats(
                vote_decision.winner_idx,
                format!(
                    "best-of-n judge `{judge_model}` failed ({e}); fell back to vote: {}",
                    vote_decision.reason
                ),
                vote_decision.stats,
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
        .unwrap_or(0);
    ArbitrationDecision::with_stats(
        winner_idx,
        format!(
            "best-of-n: `{}` scored {score} (judge `{judge_model}`)",
            successes[i].model
        ),
        stats,
    )
}

async fn judge(
    all: &[SubagentOutput],
    successes: &[&SubagentOutput],
    judge_model: &str,
    executor: &AIExecutor,
    goal: &str,
) -> ArbitrationDecision {
    let prompt = build_judge_prompt(goal, successes);
    let messages = vec![
        Message::text(
            "system",
            "You are an impartial judge. Pick the single best candidate. Reply with one \
             line `pick: i` (1-based) followed by 1-2 sentences of reasoning.",
        ),
        Message::text("user", prompt),
    ];
    let (raw, stats) = match executor.chat_with_model(judge_model, messages).await {
        Ok((msg, stats)) => (msg.content, stats),
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
                        .unwrap_or(0)
                })
                .unwrap_or(0);
            return ArbitrationDecision::new(
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
        .unwrap_or(0);
    ArbitrationDecision::with_stats(
        winner_idx,
        format!(
            "judge `{judge_model}` picked `{}`: {reason}",
            successes[pick].model
        ),
        stats,
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
    use std::path::PathBuf;

    fn unsafe_unset_all() {
        // SAFETY: tests are serialized by the outer mutex `lock()`
        // before touching env vars.
        unsafe {
            std::env::remove_var("CUBI_FAKE_LLM");
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

    static TEST_ISOLATED_ACTIVE_SUBAGENTS: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);
    static TEST_ISOLATED_MAX_SUBAGENTS: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    fn fake_isolated_runner() -> IsolatedSubagentRunner {
        Arc::new(|model, goal, _time_cap, max_steps| {
            Box::pin(async move {
                if model == "isolated-error" {
                    anyhow::bail!("test isolated subprocess failure");
                }
                if model.starts_with("isolated-slow") {
                    let active = TEST_ISOLATED_ACTIVE_SUBAGENTS
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                        + 1;
                    update_test_isolated_max(active);
                    tokio::time::sleep(Duration::from_millis(40)).await;
                    TEST_ISOLATED_ACTIVE_SUBAGENTS
                        .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                }
                let exit_code = if model == "isolated-exit" {
                    Some(2)
                } else {
                    Some(0)
                };
                Ok(crate::proc_subagent::ProcSubagentResult {
                    output: if model == "isolated-empty" {
                        String::new()
                    } else if model == "isolated-max-steps" {
                        format!("max_steps={max_steps}")
                    } else if model == "isolated-json-error" {
                        "partial structured failure".to_string()
                    } else {
                        goal
                    },
                    exit_code,
                    timed_out: model == "isolated-timeout",
                    tool_calls: 1,
                    prompt_tokens: 11,
                    completion_tokens: 13,
                    steps_used: if model == "isolated-max-steps" {
                        max_steps
                    } else {
                        2
                    },
                    stderr: if model == "isolated-json-error" || model == "isolated-exit" {
                        "child stderr diagnostic".to_string()
                    } else {
                        String::new()
                    },
                    error: if model == "isolated-json-error" {
                        Some("error: structured child failure".to_string())
                    } else {
                        None
                    },
                })
            })
        })
    }

    fn update_test_isolated_max(active: usize) {
        let mut current = TEST_ISOLATED_MAX_SUBAGENTS.load(std::sync::atomic::Ordering::SeqCst);
        while active > current {
            match TEST_ISOLATED_MAX_SUBAGENTS.compare_exchange(
                current,
                active,
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
    }

    #[test]
    fn isolated_permits_defaults_and_caps() {
        assert_eq!(isolated_permits(0, 5), 1);
        assert_eq!(isolated_permits(99, 5), 2);
        assert_eq!(isolated_permits(2, 1), 1);
        assert_eq!(isolated_permits(1, 5), 1);
    }

    async fn run_with_fake_isolated_runner(req: ConsensusRequest) -> Result<ConsensusResult> {
        let mut no_mcp = None;
        run_inner_with_isolated_runner(
            req,
            &executor(),
            &NullSink,
            Some(&mut no_mcp),
            Some(fake_isolated_runner()),
        )
        .await
    }

    fn init_temp_git_repo(repo: &Path) {
        run_test_git(repo, &["init"]);
        run_test_git(repo, &["config", "user.email", "cubi@example.invalid"]);
        run_test_git(repo, &["config", "user.name", "Cubi Test"]);
        std::fs::write(repo.join("tracked.txt"), "tracked\n").unwrap();
        run_test_git(repo, &["add", "tracked.txt"]);
        run_test_git(repo, &["commit", "-m", "init"]);
    }

    fn run_test_git(repo: &Path, args: &[&str]) -> String {
        let output = std::process::Command::new("git")
            .current_dir(repo)
            .args(args)
            .output()
            .unwrap_or_else(|err| panic!("failed to run git {args:?}: {err}"));
        assert!(
            output.status.success(),
            "git {args:?} failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).into_owned()
    }

    #[test]
    fn isolated_trust_translation_maps_repo_root_trust_to_worktree_root_for_nested_cwd() {
        let repo = tempfile::tempdir().unwrap();
        let nested = repo.path().join("nested/deeper");
        std::fs::create_dir_all(&nested).unwrap();
        let mut permissions = crate::permissions::Permissions::default();
        permissions.trust_dir(repo.path()).unwrap();

        let relative =
            translated_isolated_trust_root_relative(&nested, repo.path(), &permissions).unwrap();

        assert_eq!(relative, PathBuf::new());
    }

    #[test]
    fn isolated_trust_translation_preserves_trusted_subdir_scope() {
        let repo = tempfile::tempdir().unwrap();
        let trusted = repo.path().join("nested");
        let cwd = trusted.join("deeper");
        std::fs::create_dir_all(&cwd).unwrap();
        let mut permissions = crate::permissions::Permissions::default();
        permissions.trust_dir(&trusted).unwrap();

        let relative =
            translated_isolated_trust_root_relative(&cwd, repo.path(), &permissions).unwrap();

        assert_eq!(relative, PathBuf::from("nested"));
    }

    fn compile_offline_subprocess_helper(dir: &Path) -> PathBuf {
        let source = dir.join("offline_subagent_helper.rs");
        let exe = dir.join(if cfg!(windows) {
            "offline_subagent_helper.exe"
        } else {
            "offline_subagent_helper"
        });
        std::fs::write(
            &source,
            r##"
use std::{env, fs, path::PathBuf, process};

fn find_worktree_root(mut dir: PathBuf) -> Option<PathBuf> {
    loop {
        if dir.join(".git").exists() {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if !args.iter().any(|arg| arg == "--internal-subagent") {
        eprintln!("missing internal subagent flag: {args:?}");
        process::exit(10);
    }
    let max_steps = args
        .windows(2)
        .find(|window| window[0] == "--internal-max-steps")
        .and_then(|window| window[1].parse::<usize>().ok())
        .unwrap_or(0);
    if max_steps == 0 {
        eprintln!("missing internal max-step cap: {args:?}");
        process::exit(11);
    }

    let cubi_home = env::var("CUBI_HOME").expect("CUBI_HOME set by parent");
    if env::var("HOME").as_deref() != Ok(cubi_home.as_str()) {
        eprintln!("HOME did not match CUBI_HOME");
        process::exit(12);
    }
    if env::var("USERPROFILE").as_deref() != Ok(cubi_home.as_str()) {
        eprintln!("USERPROFILE did not match CUBI_HOME");
        process::exit(13);
    }
    if env::var("CUBI_DISABLE_META_TOOLS").as_deref() != Ok("1") {
        eprintln!("missing CUBI_DISABLE_META_TOOLS=1");
        process::exit(14);
    }

    let cwd = env::current_dir().expect("current dir");
    let cwd_marker = if cwd.join("subdir_marker.txt").is_file() {
        "subdir"
    } else if cwd.join("tracked.txt").is_file() {
        "root"
    } else {
        eprintln!("helper did not start in isolated worktree: {}", cwd.display());
        process::exit(15);
    };
    let trust_raw = fs::read_to_string(
        PathBuf::from(&cubi_home).join(".cubi").join("trusted_dirs.json")
    )
    .unwrap_or_default();
    let canonical_cwd = fs::canonicalize(&cwd).expect("canonical cwd");
    let worktree_root =
        find_worktree_root(canonical_cwd.clone()).unwrap_or_else(|| canonical_cwd.clone());
    let canonical_root = fs::canonicalize(&worktree_root).unwrap_or(worktree_root);
    // trusted_dirs.json is JSON, so backslashes in Windows paths are escaped
    // (`\\`). Escape the needles the same way so the substring match works on
    // Windows; on Unix paths have no backslashes, so this is a no-op.
    let root_needle = format!("\"{}\"", canonical_root.to_string_lossy().replace('\\', "\\\\"));
    let cwd_needle = format!("\"{}\"", canonical_cwd.to_string_lossy().replace('\\', "\\\\"));
    let trust_marker = if trust_raw.contains(&root_needle) {
        "root"
    } else if trust_raw.contains(&cwd_needle) {
        "cwd"
    } else {
        "missing"
    };

    for _ in 0..max_steps {
        println!("{}", r#"{"type":"tool_call","name":"read_file","arguments":{}}"#);
    }
    let model = env::var("CUBI_MODEL").unwrap_or_else(|_| "missing-model".to_string());
    println!(
        "{}",
        format!(
            r#"{{"type":"token","value":"helper output from {model} with max_steps={max_steps} cwd_marker={cwd_marker} trust_marker={trust_marker}"}}"#
        )
    );
    println!("{}", r#"{"type":"done","stats":{"prompt_tokens":21,"completion_tokens":34}}"#);
}
"##,
        )
        .unwrap();
        let output = std::process::Command::new("rustc")
            .arg("--edition=2021")
            .arg(&source)
            .arg("-o")
            .arg(&exe)
            .output()
            .unwrap_or_else(|err| panic!("failed to invoke rustc for helper: {err}"));
        assert!(
            output.status.success(),
            "rustc helper failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        exe
    }

    fn assert_no_consensus_worktree_artifacts(repo: &Path) {
        let worktrees = run_test_git(repo, &["worktree", "list", "--porcelain"]);
        assert!(
            !worktrees.contains("cubi-consensus-"),
            "temporary consensus worktree was not cleaned up:\n{worktrees}"
        );
        let branches = run_test_git(repo, &["branch", "--list", "cubi-consensus/*"]);
        assert!(
            branches.trim().is_empty(),
            "temporary consensus branch was not cleaned up:\n{branches}"
        );
    }

    #[cfg(unix)]
    fn has_consensus_worktree_artifacts(repo: &Path) -> bool {
        let worktrees = run_test_git(repo, &["worktree", "list", "--porcelain"]);
        if worktrees.contains("cubi-consensus-") {
            return true;
        }
        let branches = run_test_git(repo, &["branch", "--list", "cubi-consensus/*"]);
        !branches.trim().is_empty()
    }

    /// Writes a tiny shell script that ignores its arguments and sleeps
    /// well past the test's own cancellation window, standing in for the
    /// headless `cubi` subprocess. Unix-only: relies on a `#!/bin/sh`
    /// shebang and executable bit rather than a compiled binary, so the
    /// cancellation test below doesn't pay `rustc`'s cost just to prove a
    /// child process was in flight when cancelled.
    #[cfg(unix)]
    fn write_sleepy_subprocess_helper(dir: &Path) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let script = dir.join("sleepy_subagent_helper.sh");
        std::fs::write(&script, "#!/bin/sh\nsleep 300\n").unwrap();
        let mut perms = std::fs::metadata(&script).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script, perms).unwrap();
        script
    }

    /// Polls (without blocking the async runtime) until `predicate` returns
    /// true or `timeout` elapses. Returns whether the predicate succeeded.
    #[cfg(unix)]
    async fn wait_until(timeout: Duration, mut predicate: impl FnMut() -> bool) -> bool {
        let start = Instant::now();
        loop {
            if predicate() {
                return true;
            }
            if start.elapsed() >= timeout {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    // Regression test for the worktree-isolated subprocess consensus
    // cancellation-cleanup defect: dropping/aborting the future driving
    // `run_one_isolated_subagent_in_repo_with_binary` used to run
    // `WorktreeSession::drop`'s blocking `git worktree remove`/`git branch
    // -D` inline on whatever thread was driving the cancellation (e.g. a
    // Tokio worker thread servicing a `select!` that lost a race). This
    // proves cancellation (a) returns promptly instead of blocking on git,
    // and (b) still cleans the worktree/branch up, just asynchronously.
    #[cfg(unix)]
    #[tokio::test]
    async fn cancelling_in_flight_isolated_subagent_cleans_up_worktree_without_blocking() {
        let _g = lock().await;
        unsafe_unset_all();
        let repo = tempfile::tempdir().unwrap();
        init_temp_git_repo(repo.path());
        let helper_dir = tempfile::tempdir().unwrap();
        let helper = write_sleepy_subprocess_helper(helper_dir.path());
        let repo_path = repo.path().to_path_buf();

        let task = tokio::spawn(async move {
            run_one_isolated_subagent_in_repo_with_binary(
                &repo_path,
                Path::new(""),
                "HEAD",
                "helper-model",
                "cancel me mid-flight",
                Duration::from_secs(300),
                4,
                &helper,
            )
            .await
        });

        // Don't cancel until the subagent is genuinely in flight: wait for
        // its throwaway worktree to actually show up in `git worktree
        // list`. This proves we're cancelling a live worktree + running
        // child process, not a future that never got past provisioning.
        let provisioned = wait_until(Duration::from_secs(10), || {
            has_consensus_worktree_artifacts(repo.path())
        })
        .await;
        assert!(
            provisioned,
            "isolated subagent's worktree never appeared within the wait window"
        );

        // Cancel the in-flight future. Aborting must return promptly: it
        // must NOT block on the guard's synchronous git teardown running
        // inline on this (or any runtime worker) thread.
        let abort_started = Instant::now();
        task.abort();
        let joined = task.await;
        let abort_elapsed = abort_started.elapsed();
        assert!(
            joined.is_err(),
            "expected the aborted task's join to observe cancellation, got: {joined:?}"
        );
        assert!(
            abort_elapsed < Duration::from_secs(5),
            "aborting the in-flight isolated subagent took too long \
             (likely blocked on synchronous git cleanup): {abort_elapsed:?}"
        );

        // Cleanup after cancellation is scheduled best-effort on a
        // blocking thread (see `CancelSafeWorktreeSession::drop`), so it
        // may finish slightly after `task.abort()` returns. Give it a
        // bounded window to complete rather than asserting it's instant.
        let cleaned_up = wait_until(Duration::from_secs(10), || {
            !has_consensus_worktree_artifacts(repo.path())
        })
        .await;
        assert!(
            cleaned_up,
            "temporary consensus worktree/branch was not cleaned up after cancellation"
        );
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
            isolate: false,
            isolated_time_cap_secs: 0,
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
        assert!(
            r.subagent_outputs.iter().all(|s| s.tool_calls == 0),
            "LLM-only subagents must report zero tool calls: {:?}",
            r.subagent_outputs
        );
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
            isolate: false,
            isolated_time_cap_secs: 0,
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
            isolate: false,
            isolated_time_cap_secs: 0,
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
            isolate: false,
            isolated_time_cap_secs: 0,
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
            isolate: false,
            isolated_time_cap_secs: 0,
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
            isolate: false,
            isolated_time_cap_secs: 0,
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
            isolate: false,
            isolated_time_cap_secs: 0,
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
            isolate: false,
            isolated_time_cap_secs: 0,
        };
        let r = run(req, &executor(), &NullSink).await.unwrap();
        let agg = r.aggregate_stats();
        // Fake stats are 1/1/1 each, two successful calls → 2/2.
        assert_eq!(agg.prompt_tokens, 2);
        assert_eq!(agg.completion_tokens, 2);
    }

    #[tokio::test]
    async fn aggregate_stats_includes_best_of_n_judge_usage() {
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
            isolate: false,
            isolated_time_cap_secs: 0,
        };

        let r = run(req, &executor(), &NullSink).await.unwrap();
        let agg = r.aggregate_stats();

        assert_eq!(r.arbitration_stats.prompt_tokens, 1);
        assert_eq!(r.arbitration_stats.completion_tokens, 1);
        assert_eq!(agg.prompt_tokens, 3);
        assert_eq!(agg.completion_tokens, 3);
    }

    #[tokio::test]
    async fn aggregate_stats_includes_judge_strategy_usage() {
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
            isolate: false,
            isolated_time_cap_secs: 0,
        };

        let r = run(req, &executor(), &NullSink).await.unwrap();
        let agg = r.aggregate_stats();

        assert_eq!(r.arbitration_stats.prompt_tokens, 1);
        assert_eq!(r.arbitration_stats.completion_tokens, 1);
        assert_eq!(agg.prompt_tokens, 3);
        assert_eq!(agg.completion_tokens, 3);
    }

    #[tokio::test]
    async fn aggregate_stats_includes_vote_tie_break_judge_usage() {
        let _g = lock().await;
        unsafe_unset_all();
        unsafe_set(
            "CUBI_FAKE_LLM_MODEL_RESPONSES",
            r#"{"m1":"pick: 2\ngood","m2":"B"}"#,
        );
        let req = ConsensusRequest {
            goal: "x".into(),
            models: vec!["m1".into(), "m2".into()],
            strategy: ConsensusStrategy::Vote,
            max_steps_per_subagent: 1,
            concurrency: 0,
            use_tools: false,
            isolate: false,
            isolated_time_cap_secs: 0,
        };

        let r = run(req, &executor(), &NullSink).await.unwrap();
        let agg = r.aggregate_stats();

        assert!(r.decision_reason.contains("tie"));
        assert_eq!(r.arbitration_stats.prompt_tokens, 1);
        assert_eq!(r.arbitration_stats.completion_tokens, 1);
        assert_eq!(agg.prompt_tokens, 3);
        assert_eq!(agg.completion_tokens, 3);
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
            isolate: false,
            isolated_time_cap_secs: 0,
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
            isolate: false,
            isolated_time_cap_secs: 0,
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
            isolate: false,
            isolated_time_cap_secs: 0,
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
        assert!(outputs.iter().all(|s| s.tool_calls == 1));
        assert!(outputs.iter().all(|s| s.prompt_tokens == 2));
        assert!(outputs.iter().all(|s| s.completion_tokens == 2));
    }

    #[tokio::test]
    async fn isolate_requires_use_tools() {
        let _g = lock().await;
        unsafe_unset_all();
        let req = ConsensusRequest {
            goal: "x".into(),
            models: vec!["m1".into(), "m2".into()],
            strategy: ConsensusStrategy::Vote,
            max_steps_per_subagent: 8,
            concurrency: 0,
            use_tools: false,
            isolate: true,
            isolated_time_cap_secs: 0,
        };

        let err = run(req, &executor(), &NullSink)
            .await
            .unwrap_err()
            .to_string();

        assert!(
            err.contains("isolate` requires `use_tools=true"),
            "got: {err}"
        );
    }

    #[tokio::test]
    async fn isolated_tool_mode_does_not_require_mcp_and_accounts_proc_metadata() {
        let _g = lock().await;
        unsafe_unset_all();
        let req = ConsensusRequest {
            goal: "same isolated answer".into(),
            models: vec!["m1".into(), "m2".into()],
            strategy: ConsensusStrategy::Vote,
            max_steps_per_subagent: 8,
            concurrency: 2,
            use_tools: true,
            isolate: true,
            isolated_time_cap_secs: 7,
        };
        let result = run_with_fake_isolated_runner(req).await.unwrap();

        assert_eq!(result.winner_output, "same isolated answer");
        assert_eq!(result.subagent_outputs.len(), 2);
        assert!(result.subagent_outputs.iter().all(|s| s.error.is_none()));
        assert!(result.subagent_outputs.iter().all(|s| s.steps_used == 2));
        assert!(
            result
                .subagent_outputs
                .iter()
                .all(|s| s.prompt_tokens == 11)
        );
        assert!(
            result
                .subagent_outputs
                .iter()
                .all(|s| s.completion_tokens == 13)
        );
        assert!(
            result.subagent_outputs.iter().all(|s| s.tool_calls == 1),
            "isolated subprocess tool calls were not propagated: {:?}",
            result.subagent_outputs
        );
        let serialized = serde_json::to_value(&result).unwrap();
        assert_eq!(serialized["subagent_outputs"][0]["tool_calls"], 1);
    }

    #[tokio::test]
    async fn isolated_tool_mode_forwards_requested_step_cap_to_runner() {
        let _g = lock().await;
        unsafe_unset_all();
        let req = ConsensusRequest {
            goal: "check cap".into(),
            models: vec!["isolated-max-steps".into(), "m2".into()],
            strategy: ConsensusStrategy::Vote,
            max_steps_per_subagent: 5,
            concurrency: 1,
            use_tools: true,
            isolate: true,
            isolated_time_cap_secs: 7,
        };

        let result = run_with_fake_isolated_runner(req).await.unwrap();
        let capped = result
            .subagent_outputs
            .iter()
            .find(|s| s.model == "isolated-max-steps")
            .unwrap();

        assert_eq!(capped.output, "max_steps=5");
        assert_eq!(capped.steps_used, 5);
    }

    #[tokio::test]
    async fn isolated_tool_mode_clamps_zero_step_cap_before_dispatch() {
        let _g = lock().await;
        unsafe_unset_all();
        let req = ConsensusRequest {
            goal: "check cap".into(),
            models: vec!["isolated-max-steps".into(), "m2".into()],
            strategy: ConsensusStrategy::Vote,
            max_steps_per_subagent: 0,
            concurrency: 1,
            use_tools: true,
            isolate: true,
            isolated_time_cap_secs: 7,
        };

        let result = run_with_fake_isolated_runner(req).await.unwrap();
        let capped = result
            .subagent_outputs
            .iter()
            .find(|s| s.model == "isolated-max-steps")
            .unwrap();

        assert_eq!(capped.output, "max_steps=1");
        assert_eq!(capped.steps_used, 1);
    }

    #[tokio::test]
    async fn isolated_dispatch_clamps_zero_step_cap_before_invoking_runner() {
        let req = ConsensusRequest {
            goal: "check direct dispatch cap".into(),
            models: vec!["isolated-max-steps".into(), "m2".into()],
            strategy: ConsensusStrategy::Vote,
            max_steps_per_subagent: 0,
            concurrency: 1,
            use_tools: true,
            isolate: true,
            isolated_time_cap_secs: 7,
        };

        let outputs = run_isolated_tool_subagents_with_runner(&req, fake_isolated_runner()).await;
        let capped = outputs
            .iter()
            .find(|s| s.model == "isolated-max-steps")
            .unwrap();

        assert_eq!(capped.output, "max_steps=1");
        assert_eq!(capped.steps_used, 1);
    }

    #[tokio::test]
    async fn isolated_tool_mode_records_timeout_empty_output_and_spawn_errors() {
        let _g = lock().await;
        unsafe_unset_all();
        let req = ConsensusRequest {
            goal: "survivor".into(),
            models: vec![
                "m1".into(),
                "isolated-timeout".into(),
                "isolated-empty".into(),
                "isolated-exit".into(),
                "isolated-json-error".into(),
                "isolated-error".into(),
            ],
            strategy: ConsensusStrategy::Vote,
            max_steps_per_subagent: 8,
            concurrency: 0,
            use_tools: true,
            isolate: true,
            isolated_time_cap_secs: 9,
        };
        let result = run_with_fake_isolated_runner(req).await.unwrap();

        assert_eq!(result.winner_model, "m1");
        assert_eq!(result.decision_reason, "only successful subagent");
        let timeout = result
            .subagent_outputs
            .iter()
            .find(|s| s.model == "isolated-timeout")
            .unwrap();
        assert!(
            timeout
                .error
                .as_deref()
                .is_some_and(|e| e.contains("timed out after 9s")),
            "timeout error: {:?}",
            timeout.error
        );
        let empty = result
            .subagent_outputs
            .iter()
            .find(|s| s.model == "isolated-empty")
            .unwrap();
        assert!(
            empty
                .error
                .as_deref()
                .is_some_and(|e| e.contains("empty output")),
            "empty error: {:?}",
            empty.error
        );
        let failed = result
            .subagent_outputs
            .iter()
            .find(|s| s.model == "isolated-exit")
            .unwrap();
        assert!(
            failed
                .error
                .as_deref()
                .is_some_and(|e| e.contains("exited with status Some(2)")
                    && e.contains("child stderr diagnostic")),
            "exit error: {:?}",
            failed.error
        );
        let failed = result
            .subagent_outputs
            .iter()
            .find(|s| s.model == "isolated-json-error")
            .unwrap();
        assert_eq!(failed.steps_used, 2);
        assert!(
            failed.error.as_deref().is_some_and(|e| {
                e.contains("reported error")
                    && e.contains("structured child failure")
                    && e.contains("child stderr diagnostic")
            }),
            "json error: {:?}",
            failed.error
        );
        let failed = result
            .subagent_outputs
            .iter()
            .find(|s| s.model == "isolated-error")
            .unwrap();
        assert!(
            failed
                .error
                .as_deref()
                .is_some_and(|e| e.contains("test isolated subprocess failure")),
            "failure error: {:?}",
            failed.error
        );
    }

    #[tokio::test]
    async fn isolated_tool_mode_respects_concurrency_limit() {
        let _g = lock().await;
        unsafe_unset_all();
        TEST_ISOLATED_ACTIVE_SUBAGENTS.store(0, std::sync::atomic::Ordering::SeqCst);
        TEST_ISOLATED_MAX_SUBAGENTS.store(0, std::sync::atomic::Ordering::SeqCst);
        let req = ConsensusRequest {
            goal: "same answer".into(),
            models: vec![
                "isolated-slow-a".into(),
                "isolated-slow-b".into(),
                "isolated-slow-c".into(),
            ],
            strategy: ConsensusStrategy::Vote,
            max_steps_per_subagent: 8,
            concurrency: 2,
            use_tools: true,
            isolate: true,
            isolated_time_cap_secs: 3,
        };
        let result = run_with_fake_isolated_runner(req).await.unwrap();

        assert_eq!(result.subagent_outputs.len(), 3);
        assert!(result.subagent_outputs.iter().all(|s| s.error.is_none()));
        assert_eq!(
            TEST_ISOLATED_MAX_SUBAGENTS.load(std::sync::atomic::Ordering::SeqCst),
            2
        );
    }

    #[tokio::test]
    async fn isolated_subagent_real_worktree_subprocess_forwards_cap_and_cleans_up() {
        let _g = lock().await;
        unsafe_unset_all();
        let repo = tempfile::tempdir().unwrap();
        init_temp_git_repo(repo.path());
        let helper_dir = tempfile::tempdir().unwrap();
        let helper = compile_offline_subprocess_helper(helper_dir.path());

        let result = run_one_isolated_subagent_in_repo_with_binary(
            repo.path(),
            Path::new(""),
            "HEAD",
            "helper-model",
            "offline isolated subprocess",
            Duration::from_secs(20),
            4,
            &helper,
        )
        .await
        .unwrap();

        assert_eq!(result.exit_code, Some(0));
        assert!(!result.timed_out);
        assert_eq!(result.tool_calls, 4);
        assert_eq!(result.steps_used, 5);
        assert_eq!(result.prompt_tokens, 21);
        assert_eq!(result.completion_tokens, 34);
        assert!(
            result.output.contains("helper output from helper-model")
                && result.output.contains("max_steps=4"),
            "unexpected helper output: {}",
            result.output
        );
        assert_no_consensus_worktree_artifacts(repo.path());
    }

    #[tokio::test]
    async fn isolated_subagent_translates_repo_root_trust_while_running_from_subdir() {
        let _g = lock().await;
        unsafe_unset_all();
        let repo = tempfile::tempdir().unwrap();
        init_temp_git_repo(repo.path());
        std::fs::create_dir_all(repo.path().join("nested/deeper")).unwrap();
        std::fs::write(
            repo.path().join("nested/deeper/subdir_marker.txt"),
            "subdir\n",
        )
        .unwrap();
        run_test_git(repo.path(), &["add", "nested/deeper/subdir_marker.txt"]);
        run_test_git(repo.path(), &["commit", "-m", "nested"]);
        let helper_dir = tempfile::tempdir().unwrap();
        let helper = compile_offline_subprocess_helper(helper_dir.path());

        let result = run_one_isolated_subagent_in_repo_with_binary_and_trust_root(
            repo.path(),
            Path::new("nested/deeper"),
            Path::new(""),
            "HEAD",
            "helper-model",
            "offline isolated subprocess from subdir",
            Duration::from_secs(20),
            2,
            &helper,
        )
        .await
        .unwrap();

        assert_eq!(result.exit_code, Some(0));
        assert!(
            result.output.contains("cwd_marker=subdir"),
            "subprocess did not run in matching worktree subdir: {}",
            result.output
        );
        assert!(
            result.output.contains("trust_marker=root"),
            "isolated trust root was not translated to the worktree root: {}",
            result.output
        );
        assert_eq!(result.tool_calls, 2);
        assert_eq!(result.steps_used, 3);
        assert_no_consensus_worktree_artifacts(repo.path());
    }

    #[tokio::test]
    async fn isolated_consensus_real_worktrees_subprocesses_forward_caps_and_clean_up() {
        let _g = lock().await;
        unsafe_unset_all();
        let repo = tempfile::tempdir().unwrap();
        init_temp_git_repo(repo.path());
        let helper_dir = tempfile::tempdir().unwrap();
        let helper = compile_offline_subprocess_helper(helper_dir.path());
        let repo_path = repo.path().to_path_buf();
        let runner: IsolatedSubagentRunner = Arc::new(move |model, goal, time_cap, max_steps| {
            let repo_path = repo_path.clone();
            let helper = helper.clone();
            Box::pin(async move {
                run_one_isolated_subagent_in_repo_with_binary(
                    &repo_path,
                    Path::new(""),
                    "HEAD",
                    &model,
                    &goal,
                    time_cap,
                    max_steps,
                    &helper,
                )
                .await
            })
        });
        let req = ConsensusRequest {
            goal: "offline isolated subprocess".into(),
            models: vec!["helper-model".into(), "helper-model".into()],
            strategy: ConsensusStrategy::Vote,
            max_steps_per_subagent: 3,
            concurrency: 2,
            use_tools: true,
            isolate: true,
            isolated_time_cap_secs: 20,
        };
        let mut no_mcp = None;

        let result = run_inner_with_isolated_runner(
            req,
            &executor(),
            &NullSink,
            Some(&mut no_mcp),
            Some(runner),
        )
        .await
        .unwrap();

        assert_eq!(result.subagent_outputs.len(), 2);
        assert!(result.subagent_outputs.iter().all(SubagentOutput::ok));
        assert!(result.subagent_outputs.iter().all(|sub| {
            sub.steps_used == 4
                && sub.tool_calls == 3
                && sub.prompt_tokens == 21
                && sub.completion_tokens == 34
        }));
        assert!(
            result
                .subagent_outputs
                .iter()
                .all(|sub| sub.output.contains("helper output from helper-model")
                    && sub.output.contains("max_steps=3")),
            "outputs: {:?}",
            result.subagent_outputs
        );
        assert_no_consensus_worktree_artifacts(repo.path());
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
        assert_eq!(
            spec.function.parameters["properties"]["max_steps"]["minimum"],
            1
        );
        assert_eq!(
            spec.function.parameters["properties"]["max_steps"]["default"],
            CONSENSUS_DEFAULT_MAX_STEPS
        );
        assert!(
            !spec.function.parameters["properties"]["use_tools"]["default"]
                .as_bool()
                .unwrap_or(true)
        );
        assert!(
            !spec.function.parameters["properties"]["isolate"]["default"]
                .as_bool()
                .unwrap_or(true)
        );
        assert_eq!(
            spec.function.parameters["properties"]["isolated_time_cap_secs"]["default"],
            CONSENSUS_ISOLATED_DEFAULT_TIME_CAP_SECS
        );
        assert!(
            spec.function.parameters["properties"]
                .get("time_cap_seconds")
                .is_none(),
            "tool schema should expose isolated_time_cap_secs, not time_cap_seconds"
        );
        assert!(
            spec.function.parameters["required"]
                .as_array()
                .map(|a| a.iter().any(|v| v == "goal") && a.iter().any(|v| v == "models"))
                .unwrap_or(false)
        );
    }
}
