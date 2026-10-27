//! Native tool-calling agent loop.
//!
//! Ollama's `/api/chat` accepts a `tools` array that tool-capable models
//! (llama3.1+, qwen2.5, mistral 0.3, etc.) consult when deciding whether to
//! emit `tool_calls` instead of (or in addition to) free-form text. This
//! module turns that into an honest agent loop:
//!
//! 1. Forward the conversation + tool list to the model.
//! 2. If the model returns plain `content`, we're done — print and persist.
//! 3. If the model returns `tool_calls`, execute each one through the
//!    [`McpManager`] (which routes between built-in and external MCP tools),
//!    append a `role:"tool"` result message, and loop.
//!
//! The loop has a fixed step cap so a misbehaving model can't burn forever.
//! Tool execution failures become tool messages (so the model can react and
//! recover) rather than aborting the whole turn.
//!
//! Older models simply ignore the `tools` field and the loop collapses to a
//! single iteration of streaming chat — no behavioral regression for users
//! who haven't pulled a tool-capable model yet.

use crate::config::ToolProfile;
use anyhow::Result;
use serde_json::json;
use std::future::Future;
use std::pin::Pin;

use crate::cli::tool_dispatch::{
    ApprovalHost, ToolAccess, ToolCallHost, ToolOutcome, dispatch_tool,
};
use crate::executor::AIExecutor;
use crate::mcp_client::Tool;
use crate::mcp_manager::McpManager;
use crate::ollama::{ChatStats, Message, ToolCall, ToolFunction, ToolSpec};

/// Maximum number of "model → tool → model" iterations per user turn before
/// the loop bails out with a diagnostic. Generous enough for realistic
/// multi-step plans (write file → run tests → grep output → edit again),
/// small enough that a runaway model gives up well before the user does.
pub const MAX_AGENT_STEPS: usize = 12;

/// In headless mode, tool errors are fed back to the model so it can
/// recover (fix a path, re-read a file, correct `old_text`, ...) instead
/// of aborting the whole run on the first miss. If the model emits this
/// many *consecutive* tool errors with no successful call in between it is
/// considered genuinely stuck and the run bails out with `ExitCode::Tool`.
pub const MAX_CONSECUTIVE_TOOL_ERRORS: u32 = 6;

/// Default ceiling on a subagent's own loop. The subagent always uses the
/// minimum of (caller-supplied `max_steps`, this constant) so a misbehaving
/// `agent_run` call can never grow the parent's budget by surprise.
pub const SUBAGENT_DEFAULT_STEPS: usize = 8;
pub const SUBAGENT_MAX_STEPS: usize = MAX_AGENT_STEPS;

/// Name of the meta-tool the model uses to spawn a subagent. Kept as a
/// constant so the agent-loop dispatcher and the tool-spec builder can't
/// drift out of sync.
pub const AGENT_TOOL_NAME: &str = "agent_run";

/// Name of the multi-model arbitration meta-tool. Kept alongside
/// [`AGENT_TOOL_NAME`] so the same anti-recursion machinery (the
/// [`without_meta_tools`] helper and the matching reject branches in
/// `run_subagent` / `cli::agent::execute_tool_call`) can strip both
/// in one place.
pub const CONSENSUS_TOOL_NAME: &str = "consensus_run";

/// Environment flag used by isolated subprocess subagents to suppress
/// top-level meta-tools. The subprocess is already a subagent, so exposing
/// `agent_run` or `consensus_run` inside it would re-enable unbounded nested
/// subagent/consensus spawning.
pub const DISABLE_META_TOOLS_ENV: &str = "CUBI_DISABLE_META_TOOLS";
pub const TOOL_PROFILE_ENV: &str = "CUBI_TOOL_PROFILE";

const CODING_TOOL_NAMES: [&str; 8] = [
    "bash",
    "edit_file",
    "grep",
    "list_files",
    "read_file",
    "repo_map",
    "search_glob",
    "write_file",
];

/// System prompt prepended to every subagent's context. Kept terse — the
/// subagent inherits the model's general capabilities, but we want to keep
/// it focused on the single goal and discourage it from chatting back.
const SUBAGENT_SYSTEM_PROMPT: &str = "You are a focused worker subagent spawned by a larger \
assistant. You will receive ONE goal. Accomplish it using the tools you have, then return a \
single concise final report describing what you did and what you found. Do not ask clarifying \
questions — make a reasonable assumption and note it. Do not chat — emit only the final report \
when finished.";

/// Builds the Ollama-shaped `tools` list from every tool the MCP manager
/// knows about (built-in + each connected external server), plus the
/// `agent_run` meta-tool. Returns `None` when there are no tools at all
/// (and `mcp` is empty), so the caller can skip the field and let older
/// Ollama versions handle the payload identically to before.
pub fn build_tool_specs(mcp: &McpManager) -> Option<Vec<ToolSpec>> {
    let mcp_tools = mcp.list_tools();
    if mcp_tools.is_empty() {
        // Even with no MCP tools, the agent_run meta-tool is meaningless
        // without other tools for the subagent to use, so we skip it too.
        return None;
    }
    let specs = build_tool_specs_for_profile(
        mcp_tools,
        crate::config::with(|config| config.tool_profile),
        !meta_tools_disabled_by_env(),
    );
    (!specs.is_empty()).then_some(specs)
}

fn build_tool_specs_for_profile(
    tools: Vec<&Tool>,
    profile: ToolProfile,
    include_meta_tools: bool,
) -> Vec<ToolSpec> {
    let mut selected: Vec<&Tool> = tools
        .into_iter()
        .filter(|tool| {
            profile == ToolProfile::Full || CODING_TOOL_NAMES.contains(&tool.name.as_str())
        })
        .collect();
    if profile == ToolProfile::Coding {
        selected.sort_by(|a, b| a.name.cmp(&b.name));
    }

    let mut specs: Vec<ToolSpec> = selected
        .into_iter()
        .map(|tool| ToolSpec {
            tool_type: "function".to_string(),
            function: ToolFunction {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.input_schema.clone(),
            },
        })
        .collect();
    if include_meta_tools && !specs.is_empty() {
        specs.push(agent_run_spec());
        specs.push(crate::consensus::consensus_run_spec());
    }
    specs
}

pub fn meta_tools_disabled_by_env() -> bool {
    crate::config::with(|config| config.disable_meta_tools)
}

/// `ToolSpec` for the [`AGENT_TOOL_NAME`] meta-tool. Kept in this module so
/// the schema and the dispatch logic live next to each other.
pub fn agent_run_spec() -> ToolSpec {
    ToolSpec {
        tool_type: "function".to_string(),
        function: ToolFunction {
            name: AGENT_TOOL_NAME.to_string(),
            description: "Spawn a focused worker subagent with its own fresh context and the \
                          same toolset (minus this meta-tool) to accomplish ONE specific goal \
                          independently. Use this for chunks of work that don't need the full \
                          conversation history — investigations, batch edits, focused research \
                          — to keep the main context lean."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "Self-contained description of what the subagent should accomplish. Include all context it will need; the subagent does NOT see the parent conversation."
                    },
                    "max_steps": {
                        "type": "integer",
                        "description": "Maximum model→tool round-trips the subagent may take (default 8, capped at 12)",
                        "default": SUBAGENT_DEFAULT_STEPS
                    }
                },
                "required": ["goal"]
            }),
        },
    }
}

/// Strip every meta-tool (`agent_run`, `consensus_run`, …) from a tool
/// list. Used when building a subagent's tool list so neither
/// `agent_run` nor `consensus_run` can be invoked from inside another
/// subagent. The single helper guarantees both meta-tools are stripped
/// in lockstep — adding a new meta-tool only requires updating this
/// list and the matching reject branch below.
pub fn without_meta_tools(tools: Option<Vec<ToolSpec>>) -> Option<Vec<ToolSpec>> {
    tools.map(|mut v| {
        v.retain(|t| t.function.name != AGENT_TOOL_NAME && t.function.name != CONSENSUS_TOOL_NAME);
        v
    })
}

/// Renders a tool's `ToolCallResult` content into a single plain-text blob
/// suitable for feeding back to the model as a `role:"tool"` message.
/// Non-text content blocks are summarized rather than dropped so the model
/// at least learns they existed.
pub fn render_tool_result(result: &crate::mcp_client::ToolCallResult) -> String {
    let mut buf = String::new();
    for c in &result.content {
        if c.content_type == "text" {
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(&c.text);
        } else {
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(&format!("[non-text content: {}]", c.content_type));
        }
    }
    if let Some(true) = result.is_error {
        // Tag errors so the model can tell apart "tool ran and returned an
        // empty string" from "tool failed" — important when deciding whether
        // to retry.
        if buf.is_empty() {
            buf.push_str("[tool reported error with no message]");
        } else {
            buf.insert_str(0, "[tool error] ");
        }
    }
    buf
}

/// Final report and accounting from a subagent loop.
#[derive(Debug, Clone, Default)]
pub struct SubagentRunResult {
    pub output: String,
    pub stats: ChatStats,
    pub steps_used: usize,
    pub tool_calls: usize,
}

pub struct SubagentApproval<'a> {
    pub host: &'a mut dyn ApprovalHost,
    pub origin: &'a str,
}

struct SubagentToolHost<'a> {
    mcp: &'a mut Option<McpManager>,
    approval: &'a mut dyn ApprovalHost,
    origin: &'a str,
}

impl ToolCallHost for SubagentToolHost<'_> {
    fn approve_tool<'a>(
        &'a mut self,
        call: &'a ToolCall,
    ) -> Pin<Box<dyn Future<Output = crate::cli::tool_dispatch::Approval> + 'a>> {
        self.approval.request_approval(call, self.origin)
    }

    fn execute_authorized<'a>(
        &'a mut self,
        call: &'a ToolCall,
    ) -> Pin<Box<dyn Future<Output = ToolOutcome> + 'a>> {
        Box::pin(async move {
            if call.function.name == AGENT_TOOL_NAME {
                return ToolOutcome::error("[tool error] nested `agent_run` is not allowed");
            }
            if call.function.name == CONSENSUS_TOOL_NAME {
                return ToolOutcome::error("[tool error] nested `consensus_run` is not allowed");
            }
            let Some(mcp) = self.mcp.as_mut() else {
                return ToolOutcome::error(format!(
                    "[tool error] no MCP manager available to execute `{}`",
                    call.function.name
                ));
            };
            match mcp
                .call_tool(&call.function.name, call.function.arguments.clone())
                .await
            {
                Ok(result) => {
                    let is_error = result.is_error == Some(true);
                    let text = render_tool_result(&result);
                    if is_error {
                        ToolOutcome::error(text)
                    } else {
                        ToolOutcome::success(text)
                    }
                }
                Err(error) => ToolOutcome::error(format!("[tool error] {error}")),
            }
        })
    }
}

/// Runs a subagent loop with a fresh context. Returns the subagent's final
/// assistant message text. Used by the top-level agent loop in response to
/// an `agent_run` tool call from the parent model.
///
/// The subagent has access to the same toolset as the parent, *minus* the
/// `agent_run` meta-tool itself — recursion is intentionally disallowed
/// (cheap to add later, but the failure modes are nasty without explicit
/// depth/cost accounting).
pub async fn run_subagent(
    executor: &AIExecutor,
    mcp: &mut Option<McpManager>,
    access: ToolAccess,
    approval: &mut dyn ApprovalHost,
    origin: &str,
    goal: &str,
    requested_max_steps: usize,
) -> Result<SubagentRunResult> {
    let approval = SubagentApproval {
        host: approval,
        origin,
    };
    run_subagent_with_model(
        executor,
        mcp,
        access,
        approval,
        None,
        goal,
        requested_max_steps,
    )
    .await
}

/// Runs the same subagent loop as [`run_subagent`] but optionally overrides
/// the model for each chat call and returns token/step accounting.
pub async fn run_subagent_with_model(
    executor: &AIExecutor,
    mcp: &mut Option<McpManager>,
    access: ToolAccess,
    approval: SubagentApproval<'_>,
    model: Option<&str>,
    goal: &str,
    requested_max_steps: usize,
) -> Result<SubagentRunResult> {
    // Hard-cap the caller's budget so the parent can't be tricked into
    // burning unbounded budget via a large `max_steps`.
    let max_steps = requested_max_steps.clamp(1, SUBAGENT_MAX_STEPS);

    let tools = without_meta_tools(mcp.as_ref().and_then(build_tool_specs));

    let mut history = vec![
        Message::text("system", SUBAGENT_SYSTEM_PROMPT),
        Message::text("user", goal),
    ];
    let mut total_stats = ChatStats::default();
    let mut tool_calls = 0usize;
    let mut tool_host = SubagentToolHost {
        mcp,
        approval: approval.host,
        origin: approval.origin,
    };

    for step in 0..max_steps {
        let (msg, stats) = match model {
            Some(model) => {
                executor
                    .chat_with_model_and_tools(model, history.clone(), tools.clone())
                    .await?
            }
            None => {
                executor
                    .chat_with_tools(history.clone(), tools.clone())
                    .await?
            }
        };
        total_stats.add(&stats);
        let steps_used = step + 1;
        // Some backends (older Ollama) don't supply an `id` on each
        // tool_call. Synthesize a stable, position-based id so the
        // assistant message and its tool-result messages reference the
        // same id — strict OpenAI-compatible validators require this.
        let mut msg = msg;
        if let Some(calls) = msg.tool_calls.as_mut() {
            for (i, c) in calls.iter_mut().enumerate() {
                if c.id.is_none() {
                    c.id = Some(format!("call_{}_{}", i, c.function.name));
                }
            }
        }
        let calls = msg.tool_calls.clone().unwrap_or_default();
        tool_calls += calls.len();
        let content = msg.content.clone();
        history.push(msg);

        if calls.is_empty() {
            // No more tools to run — this is the subagent's final report.
            let output = if content.is_empty() {
                "[subagent returned empty report]".to_string()
            } else {
                content
            };
            return Ok(SubagentRunResult {
                output,
                stats: total_stats,
                steps_used,
                tool_calls,
            });
        }

        for call in calls {
            // Recursion guard: even though we strip `agent_run` and
            // `consensus_run` from the tool list, a confused model might
            // emit the name anyway. Reject explicitly so it shows up as
            // a tool error and the model gives up rather than us
            // silently doing nothing.
            let result_text = dispatch_tool(&access, &mut tool_host, &call).await.text;
            history.push(Message::tool_result(
                &call.function.name,
                result_text,
                call.id.clone(),
            ));
        }

        // If we're about to leave the loop without ever getting a clean
        // final-content turn, return the last non-empty assistant content
        // as a best-effort report.
        if step + 1 == max_steps {
            let last_text = history
                .iter()
                .rev()
                .find(|m| m.role == "assistant" && !m.content.is_empty())
                .map(|m| m.content.clone())
                .unwrap_or_default();
            return Ok(SubagentRunResult {
                output: format!(
                    "[subagent hit step cap of {max_steps}; partial result:]\n{last_text}"
                ),
                stats: total_stats,
                steps_used,
                tool_calls,
            });
        }
    }
    // Unreachable: the `step + 1 == max_steps` branch above always returns
    // on the final iteration. Kept as a safety net.
    Ok(SubagentRunResult {
        output: String::new(),
        stats: total_stats,
        steps_used: 0,
        tool_calls,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::tool_dispatch::Approval;
    use crate::mcp_client::{Content, Tool, ToolCallResult};
    use std::collections::BTreeSet;
    use std::sync::{Arc, Mutex};

    struct AlwaysApprove;

    impl ApprovalHost for AlwaysApprove {
        fn request_approval<'a>(
            &'a mut self,
            _call: &'a ToolCall,
            _origin: &'a str,
        ) -> Pin<Box<dyn Future<Output = Approval> + 'a>> {
            Box::pin(async { Approval::AllowOnce })
        }
    }

    #[derive(Default)]
    struct RecordingApproval {
        origins: Vec<String>,
    }

    impl ApprovalHost for RecordingApproval {
        fn request_approval<'a>(
            &'a mut self,
            _call: &'a ToolCall,
            origin: &'a str,
        ) -> Pin<Box<dyn Future<Output = Approval> + 'a>> {
            Box::pin(async move {
                self.origins.push(origin.to_string());
                Approval::Deny
            })
        }
    }

    fn fake_tool(name: &str) -> Tool {
        Tool {
            name: name.to_string(),
            description: format!("description for {}", name),
            input_schema: json!({}),
        }
    }

    #[test]
    fn render_tool_result_joins_text_blocks() {
        let r = ToolCallResult {
            content: vec![
                Content {
                    content_type: "text".into(),
                    text: "line 1".into(),
                },
                Content {
                    content_type: "text".into(),
                    text: "line 2".into(),
                },
            ],
            is_error: None,
        };
        assert_eq!(render_tool_result(&r), "line 1\nline 2");
    }

    #[test]
    fn render_tool_result_tags_errors() {
        let r = ToolCallResult {
            content: vec![Content {
                content_type: "text".into(),
                text: "boom".into(),
            }],
            is_error: Some(true),
        };
        assert!(render_tool_result(&r).starts_with("[tool error]"));
    }

    #[test]
    fn render_tool_result_empty_error_still_signals() {
        let r = ToolCallResult {
            content: vec![],
            is_error: Some(true),
        };
        assert!(render_tool_result(&r).contains("[tool reported error"));
    }

    #[tokio::test]
    async fn subagent_dispatch_inherits_parent_policy_denial() {
        let access = ToolAccess {
            hooks: crate::hooks::HookRegistry::empty(),
            policy: crate::policy::Policy {
                denied_tools: BTreeSet::from(["read_file".to_string()]),
                note: None,
            },
            permissions: Arc::new(Mutex::new(crate::permissions::Permissions::default())),
        };
        let mut no_mcp = None;
        let mut approval = AlwaysApprove;
        let mut host = SubagentToolHost {
            mcp: &mut no_mcp,
            approval: &mut approval,
            origin: "agent_run",
        };
        let call = ToolCall {
            id: Some("call-read".to_string()),
            call_type: None,
            function: crate::ollama::ToolCallFunction {
                name: "read_file".to_string(),
                arguments: serde_json::json!({"path": "secret.txt"}),
            },
        };

        let outcome = dispatch_tool(&access, &mut host, &call).await;

        assert_eq!(
            outcome.kind,
            crate::cli::tool_dispatch::ToolOutcomeKind::Denied
        );
        assert!(outcome.text.contains("admin policy"));
        assert!(!outcome.text.contains("no MCP manager"));
    }

    #[tokio::test]
    async fn subagent_approval_reaches_parent_with_origin() {
        let access = ToolAccess {
            hooks: crate::hooks::HookRegistry::empty(),
            policy: crate::policy::Policy::default(),
            permissions: Arc::new(Mutex::new(crate::permissions::Permissions::default())),
        };
        let mut no_mcp = None;
        let mut approval = RecordingApproval::default();
        let mut host = SubagentToolHost {
            mcp: &mut no_mcp,
            approval: &mut approval,
            origin: "consensus qwen3:8b",
        };
        let call = ToolCall {
            id: Some("call-read".to_string()),
            call_type: None,
            function: crate::ollama::ToolCallFunction {
                name: "read_file".to_string(),
                arguments: serde_json::json!({"path": "fixture.txt"}),
            },
        };

        let outcome = dispatch_tool(&access, &mut host, &call).await;

        assert_eq!(
            outcome.kind,
            crate::cli::tool_dispatch::ToolOutcomeKind::Denied
        );
        assert_eq!(approval.origins, vec!["consensus qwen3:8b"]);
    }

    #[test]
    fn render_tool_result_describes_non_text_content() {
        let r = ToolCallResult {
            content: vec![Content {
                content_type: "image".into(),
                text: "<binary>".into(),
            }],
            is_error: None,
        };
        let out = render_tool_result(&r);
        assert!(out.contains("[non-text content: image]"), "got: {out}");
    }

    #[test]
    fn agent_run_spec_has_required_goal_parameter() {
        let spec = agent_run_spec();
        assert_eq!(spec.function.name, AGENT_TOOL_NAME);
        assert_eq!(spec.function.parameters["required"][0], "goal");
        assert!(
            spec.function.parameters["properties"]["goal"].is_object(),
            "goal parameter must be an object"
        );
    }

    #[test]
    fn coding_profile_advertises_exact_core_and_meta_tool_set() {
        let tools = [
            "write_file",
            "think",
            "read_file",
            "external_mcp_tool",
            "repo_map",
            "bash",
            "search_glob",
            "grep",
            "edit_file",
            "list_files",
        ]
        .into_iter()
        .map(fake_tool)
        .collect::<Vec<_>>();

        let specs = build_tool_specs_for_profile(tools.iter().collect(), ToolProfile::Coding, true);
        let names: Vec<&str> = specs
            .iter()
            .map(|spec| spec.function.name.as_str())
            .collect();

        assert_eq!(
            names,
            vec![
                "bash",
                "edit_file",
                "grep",
                "list_files",
                "read_file",
                "repo_map",
                "search_glob",
                "write_file",
                AGENT_TOOL_NAME,
                "consensus_run",
            ]
        );
    }

    #[test]
    fn without_meta_tools_strips_only_the_meta_tools() {
        let specs = Some(vec![
            agent_run_spec(),
            ToolSpec {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "bash".into(),
                    description: "shell".into(),
                    parameters: json!({}),
                },
            },
        ]);
        let stripped = without_meta_tools(specs).unwrap();
        assert_eq!(stripped.len(), 1);
        assert_eq!(stripped[0].function.name, "bash");
    }

    #[test]
    fn without_meta_tools_handles_none() {
        assert!(without_meta_tools(None).is_none());
    }

    #[test]
    fn without_meta_tools_strips_both_meta_tools() {
        let specs = Some(vec![
            agent_run_spec(),
            crate::consensus::consensus_run_spec(),
            ToolSpec {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "bash".into(),
                    description: "shell".into(),
                    parameters: json!({}),
                },
            },
        ]);
        let stripped = without_meta_tools(specs).unwrap();
        assert_eq!(stripped.len(), 1);
        assert_eq!(stripped[0].function.name, "bash");
    }

    #[test]
    fn subagent_blocks_oversized_post_tool_request_before_backend() {
        let _env_guard = crate::compat::test_env::env_guard();
        let _llm_guard = crate::llm::ENV_LOCK.blocking_lock();
        let _restore = crate::compat::test_env::EnvRestore::capture(&[
            "CUBI_CONTEXT_WINDOW",
            "CUBI_FAKE_LLM",
            "CUBI_MAX_PROMPT_TOKENS_OVERRIDE",
            "CUBI_TOOL_PROFILE",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
        ]);
        let temp = tempfile::tempdir().unwrap();
        let large_file = temp.path().join("large.txt");
        std::fs::write(&large_file, "x".repeat(40_000)).unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        runtime.block_on(async {
            let fixture = crate::test_support::OpenAiFixture::spawn(vec![
                serde_json::json!({
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": null,
                            "tool_calls": [{
                                "id": "call-read",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": serde_json::json!({"path": large_file}).to_string()
                                }
                            }]
                        }
                    }]
                }),
                serde_json::json!({
                    "choices": [{
                        "message": {"role": "assistant", "content": "final"}
                    }]
                }),
            ])
            .await;
            // SAFETY: environment mutation is serialized by the shared test locks.
            unsafe {
                std::env::set_var("OPENAI_BASE_URL", &fixture.base_url);
                std::env::set_var("OPENAI_API_KEY", "test-key");
                std::env::set_var("CUBI_CONTEXT_WINDOW", "4096");
                std::env::set_var("CUBI_TOOL_PROFILE", "coding");
                std::env::remove_var("CUBI_FAKE_LLM");
                std::env::remove_var("CUBI_MAX_PROMPT_TOKENS_OVERRIDE");
            }
            let executor = AIExecutor::new_from_env("budget-fixture-model".to_string());
            let permissions = Arc::new(Mutex::new(crate::permissions::Permissions::default()));
            let plan_mode = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let mut mcp = Some(
                McpManager::new(Arc::clone(&permissions), plan_mode)
                    .await
                    .unwrap(),
            );
            let access = ToolAccess {
                hooks: crate::hooks::HookRegistry::empty(),
                policy: crate::policy::Policy::default(),
                permissions,
            };

            let mut approval = AlwaysApprove;
            let result = run_subagent(
                &executor,
                &mut mcp,
                access,
                &mut approval,
                "agent_run",
                "Read the requested file and report its contents.",
                2,
            )
            .await;

            let requests = fixture.requests.lock().unwrap().clone();
            fixture.shutdown().await;
            let error = result.expect_err("the post-tool request must be rejected locally");
            assert!(
                error.to_string().contains("context window"),
                "unexpected error: {error}"
            );
            assert_eq!(
                requests.len(),
                1,
                "the oversized follow-up must never reach the backend"
            );
        });
    }
}
