use crate::hooks::{HookDecision, HookRegistry};
use crate::ollama::ToolCall;
use crate::permissions::Permissions;
use crate::policy::Policy;
use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

#[derive(Debug, Clone)]
pub(crate) struct ToolAccess {
    pub hooks: HookRegistry,
    pub policy: Policy,
    pub permissions: Arc<Mutex<Permissions>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolOutcomeKind {
    Success,
    Denied,
    Error,
}

#[derive(Debug, Clone)]
pub(crate) struct ToolOutcome {
    pub text: String,
    pub kind: ToolOutcomeKind,
}

impl ToolOutcome {
    pub fn success(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            kind: ToolOutcomeKind::Success,
        }
    }

    pub fn denied(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            kind: ToolOutcomeKind::Denied,
        }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            kind: ToolOutcomeKind::Error,
        }
    }

    pub fn is_failure(&self) -> bool {
        self.kind != ToolOutcomeKind::Success
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Approval {
    AllowOnce,
    AllowToolForSession,
    Deny,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ApprovalMode {
    Headless,
    Tui,
    Classic,
}

struct TuiApprovalClient {
    render_tx: UnboundedSender<super::ui_sink::RenderEvent>,
    reply_rx: UnboundedReceiver<super::ui_sink::ApprovalReply>,
}

pub(crate) struct ApprovalSession {
    approved_tools: HashSet<String>,
    next_id: u64,
    tui: Option<TuiApprovalClient>,
}

pub(crate) trait ApprovalHost {
    fn request_approval<'a>(
        &'a mut self,
        call: &'a ToolCall,
        origin: &'a str,
    ) -> Pin<Box<dyn Future<Output = Approval> + 'a>>;
}

pub(crate) struct ParentApproval<'a> {
    session: &'a mut ApprovalSession,
    ui: &'a mut (dyn super::ui_sink::UiSink + Send),
    permissions: &'a Arc<Mutex<Permissions>>,
    cancel: &'a AtomicBool,
    mode: ApprovalMode,
}

impl<'a> ParentApproval<'a> {
    pub fn new(
        session: &'a mut ApprovalSession,
        ui: &'a mut (dyn super::ui_sink::UiSink + Send),
        permissions: &'a Arc<Mutex<Permissions>>,
        cancel: &'a AtomicBool,
        mode: ApprovalMode,
    ) -> Self {
        Self {
            session,
            ui,
            permissions,
            cancel,
            mode,
        }
    }
}

impl ApprovalHost for ParentApproval<'_> {
    fn request_approval<'a>(
        &'a mut self,
        call: &'a ToolCall,
        origin: &'a str,
    ) -> Pin<Box<dyn Future<Output = Approval> + 'a>> {
        Box::pin(async move {
            let trusted = std::env::current_dir().ok().is_some_and(|cwd| {
                self.permissions
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .contains(&cwd)
            });
            if trusted {
                Approval::AllowOnce
            } else {
                self.session
                    .request(call, origin, self.mode, self.ui, self.cancel)
                    .await
            }
        })
    }
}

impl ApprovalSession {
    pub fn new() -> Self {
        Self {
            approved_tools: HashSet::new(),
            next_id: 0,
            tui: None,
        }
    }

    pub fn attach_tui(
        &mut self,
        render_tx: UnboundedSender<super::ui_sink::RenderEvent>,
        reply_rx: UnboundedReceiver<super::ui_sink::ApprovalReply>,
    ) {
        self.tui = Some(TuiApprovalClient {
            render_tx,
            reply_rx,
        });
    }

    pub fn detach_tui(&mut self) {
        self.tui = None;
    }

    pub fn is_approved(&self, tool_name: &str) -> bool {
        self.approved_tools.contains(tool_name)
    }

    pub fn allow_tool_for_session(&mut self, tool_name: &str) {
        self.approved_tools.insert(tool_name.to_string());
    }

    pub async fn request(
        &mut self,
        call: &ToolCall,
        origin: &str,
        mode: ApprovalMode,
        ui: &mut (dyn super::ui_sink::UiSink + Send),
        cancel: &AtomicBool,
    ) -> Approval {
        if self.is_approved(&call.function.name) {
            return Approval::AllowOnce;
        }
        let decision = match mode {
            ApprovalMode::Headless => {
                ui.status(&format!(
                    "⚠ Tool `{}` wants to run; denying in headless mode.",
                    call.function.name
                ));
                Approval::Deny
            }
            ApprovalMode::Tui => self.request_tui(call, origin, cancel).await,
            ApprovalMode::Classic => request_classic(call, origin).await,
        };
        if decision == Approval::AllowToolForSession {
            self.allow_tool_for_session(&call.function.name);
        }
        decision
    }

    async fn request_tui(
        &mut self,
        call: &ToolCall,
        origin: &str,
        cancel: &AtomicBool,
    ) -> Approval {
        let Some(client) = self.tui.as_mut() else {
            return Approval::Deny;
        };
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        if client
            .render_tx
            .send(super::ui_sink::RenderEvent::ApprovalRequest {
                id,
                tool: call.function.name.clone(),
                summary: summarize_tool_call(call),
                origin: origin.to_string(),
            })
            .is_err()
        {
            return Approval::Deny;
        }
        loop {
            tokio::select! {
                biased;
                () = super::agent::wait_for_cancel(cancel) => {
                    let _ = client
                        .render_tx
                        .send(super::ui_sink::RenderEvent::ApprovalCancelled { id });
                    return Approval::Deny;
                },
                reply = client.reply_rx.recv() => match reply {
                    Some(super::ui_sink::ApprovalReply::Decision {
                        id: reply_id,
                        decision,
                    }) if reply_id == id => return decision,
                    Some(super::ui_sink::ApprovalReply::Shown { .. })
                    | Some(super::ui_sink::ApprovalReply::Decision { .. }) => continue,
                    None => return Approval::Deny,
                },
            }
        }
    }
}

async fn request_classic(call: &ToolCall, origin: &str) -> Approval {
    let tool = call.function.name.clone();
    let summary = summarize_tool_call(call);
    let origin = origin.to_string();
    tokio::task::spawn_blocking(move || {
        use std::io::{self, Write};
        print!(
            "⚠ Tool `{tool}` wants to run ({summary}; origin: {origin}). Allow? [y/N/a(lways)] "
        );
        let _ = io::stdout().flush();
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            return Approval::Deny;
        }
        match input.trim().to_ascii_lowercase().as_str() {
            "y" => Approval::AllowOnce,
            "a" => Approval::AllowToolForSession,
            _ => Approval::Deny,
        }
    })
    .await
    .unwrap_or(Approval::Deny)
}

pub(crate) trait ToolCallHost {
    fn approve_tool<'a>(
        &'a mut self,
        call: &'a ToolCall,
    ) -> Pin<Box<dyn Future<Output = Approval> + 'a>>;

    fn execute_authorized<'a>(
        &'a mut self,
        call: &'a ToolCall,
    ) -> Pin<Box<dyn Future<Output = ToolOutcome> + 'a>>;
}

impl ToolCallHost for super::ChatCLI {
    fn approve_tool<'a>(
        &'a mut self,
        call: &'a ToolCall,
    ) -> Pin<Box<dyn Future<Output = Approval> + 'a>> {
        Box::pin(self.prompt_tool_approval(call, "main"))
    }

    fn execute_authorized<'a>(
        &'a mut self,
        call: &'a ToolCall,
    ) -> Pin<Box<dyn Future<Output = ToolOutcome> + 'a>> {
        Box::pin(self.execute_authorized_tool_call(call))
    }
}

pub(crate) async fn dispatch_tool(
    access: &ToolAccess,
    host: &mut impl ToolCallHost,
    call: &ToolCall,
) -> ToolOutcome {
    let outcome = match access
        .hooks
        .fire_pre_tool_use(&call.function.name, &call.function.arguments)
        .await
    {
        HookDecision::Deny(reason) => ToolOutcome::denied(format!("[tool denied] {reason}")),
        HookDecision::Allow if access.policy.is_denied(&call.function.name) => {
            ToolOutcome::denied(format!(
                "[tool denied] `{}` is blocked by admin policy",
                call.function.name
            ))
        }
        HookDecision::Allow
            if !access
                .permissions
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .check_tool_allowed(&call.function.name) =>
        {
            ToolOutcome::denied(format!(
                "[tool denied] `{}` is blocked by your tool permissions",
                call.function.name
            ))
        }
        HookDecision::Allow => match host.approve_tool(call).await {
            Approval::AllowOnce | Approval::AllowToolForSession => {
                host.execute_authorized(call).await
            }
            Approval::Deny => ToolOutcome::denied(format!(
                "[tool denied] user declined approval for `{}`",
                call.function.name
            )),
        },
    };

    access
        .hooks
        .fire_post_tool_use(&call.function.name, &outcome.text, outcome.is_failure())
        .await;
    outcome
}

pub(crate) fn summarize_tool_call(call: &ToolCall) -> String {
    let (label, value) = match call.function.name.as_str() {
        "bash" => ("command", call.function.arguments.get("command")),
        "edit_file" | "read_file" | "write_file" => ("path", call.function.arguments.get("path")),
        "grep" | "search_glob" => ("pattern", call.function.arguments.get("pattern")),
        "web_fetch" | "browser_open" => ("url", call.function.arguments.get("url")),
        "web_search" => ("query", call.function.arguments.get("query")),
        _ => return "arguments hidden".to_string(),
    };
    let Some(value) = value.and_then(serde_json::Value::as_str) else {
        return "arguments hidden".to_string();
    };
    let normalized = value.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = normalized.chars();
    let visible: String = chars.by_ref().take(120).collect();
    let suffix = if chars.next().is_some() { "…" } else { "" };
    format!("{label}: {visible}{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::{HookDef, HookEvent, HookRegistry};
    use crate::ollama::{ToolCall, ToolCallFunction};
    use crate::permissions::Permissions;
    use crate::policy::Policy;
    use std::collections::BTreeSet;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};

    struct FakeHost {
        approval: Approval,
        approvals: usize,
        executions: usize,
    }

    impl ToolCallHost for FakeHost {
        fn approve_tool<'a>(
            &'a mut self,
            _: &'a ToolCall,
        ) -> Pin<Box<dyn Future<Output = Approval> + 'a>> {
            Box::pin(async move {
                self.approvals += 1;
                self.approval
            })
        }

        fn execute_authorized<'a>(
            &'a mut self,
            _: &'a ToolCall,
        ) -> Pin<Box<dyn Future<Output = ToolOutcome> + 'a>> {
            Box::pin(async move {
                self.executions += 1;
                ToolOutcome::success("ok")
            })
        }
    }

    fn call(name: &str) -> ToolCall {
        ToolCall {
            id: Some("call-1".to_string()),
            call_type: None,
            function: ToolCallFunction {
                name: name.to_string(),
                arguments: serde_json::json!({}),
            },
        }
    }

    #[test]
    fn tool_summary_uses_command_without_exposing_other_arguments() {
        let call = ToolCall {
            id: Some("call-bash".to_string()),
            call_type: None,
            function: ToolCallFunction {
                name: "bash".to_string(),
                arguments: serde_json::json!({
                    "command": "cargo test --all-targets",
                    "secret": "must-not-appear"
                }),
            },
        };

        let summary = summarize_tool_call(&call);

        assert_eq!(summary, "command: cargo test --all-targets");
        assert!(!summary.contains("secret"));
    }

    #[test]
    fn tool_summary_uses_path_and_truncates_long_values() {
        let call = ToolCall {
            id: Some("call-write".to_string()),
            call_type: None,
            function: ToolCallFunction {
                name: "write_file".to_string(),
                arguments: serde_json::json!({"path": "a".repeat(200)}),
            },
        };

        let summary = summarize_tool_call(&call);

        assert!(summary.starts_with("path: "));
        assert!(summary.ends_with('…'));
        assert!(summary.chars().count() <= 128);
    }

    #[tokio::test]
    async fn policy_denial_never_reaches_host() {
        let access = ToolAccess {
            hooks: HookRegistry::empty(),
            policy: Policy {
                denied_tools: BTreeSet::from(["read_file".to_string()]),
                note: None,
            },
            permissions: Arc::new(Mutex::new(Permissions::default())),
        };
        let mut host = FakeHost {
            approval: Approval::AllowOnce,
            approvals: 0,
            executions: 0,
        };

        let outcome = dispatch_tool(&access, &mut host, &call("read_file")).await;

        assert!(outcome.is_failure());
        assert_eq!(host.executions, 0);
    }

    #[tokio::test]
    async fn allowed_call_fires_pre_and_post_hooks_once() {
        let dir = tempfile::tempdir().unwrap();
        let audit = dir.path().join("hooks.log");
        let quoted = format!("'{}'", audit.display());
        let access = ToolAccess {
            hooks: HookRegistry::with_hooks(vec![
                HookDef {
                    event: HookEvent::PreToolUse,
                    match_tool: Some("read_file".to_string()),
                    command: format!("printf 'pre\\n' >> {quoted}"),
                },
                HookDef {
                    event: HookEvent::PostToolUse,
                    match_tool: Some("read_file".to_string()),
                    command: format!("printf 'post\\n' >> {quoted}"),
                },
            ]),
            policy: Policy::default(),
            permissions: Arc::new(Mutex::new(Permissions::default())),
        };
        let mut host = FakeHost {
            approval: Approval::AllowOnce,
            approvals: 0,
            executions: 0,
        };

        let outcome = dispatch_tool(&access, &mut host, &call("read_file")).await;

        assert!(!outcome.is_failure());
        assert_eq!(host.executions, 1);
        assert_eq!(std::fs::read_to_string(audit).unwrap(), "pre\npost\n");
    }

    #[tokio::test]
    async fn denied_approval_never_executes_tool() {
        let access = ToolAccess {
            hooks: HookRegistry::empty(),
            policy: Policy::default(),
            permissions: Arc::new(Mutex::new(Permissions::default())),
        };
        let mut host = FakeHost {
            approval: Approval::Deny,
            approvals: 0,
            executions: 0,
        };

        let outcome = dispatch_tool(&access, &mut host, &call("read_file")).await;

        assert_eq!(outcome.kind, ToolOutcomeKind::Denied);
        assert_eq!(host.approvals, 1);
        assert_eq!(host.executions, 0);
    }

    #[tokio::test]
    async fn allowed_approval_executes_tool_exactly_once() {
        let access = ToolAccess {
            hooks: HookRegistry::empty(),
            policy: Policy::default(),
            permissions: Arc::new(Mutex::new(Permissions::default())),
        };
        let mut host = FakeHost {
            approval: Approval::AllowOnce,
            approvals: 0,
            executions: 0,
        };

        let outcome = dispatch_tool(&access, &mut host, &call("read_file")).await;

        assert_eq!(outcome.kind, ToolOutcomeKind::Success);
        assert_eq!(host.approvals, 1);
        assert_eq!(host.executions, 1);
    }

    #[tokio::test]
    async fn session_grant_skips_the_next_tui_prompt() {
        let (render_tx, mut render_rx) = tokio::sync::mpsc::unbounded_channel();
        let (reply_tx, reply_rx) = tokio::sync::mpsc::unbounded_channel();
        reply_tx
            .send(crate::cli::ui_sink::ApprovalReply::Decision {
                id: 0,
                decision: Approval::AllowToolForSession,
            })
            .unwrap();
        let mut approvals = ApprovalSession::new();
        approvals.attach_tui(render_tx, reply_rx);
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let mut sink = crate::cli::ui_sink::NullSink;
        let tool_call = call("read_file");

        let first = approvals
            .request(&tool_call, "main", ApprovalMode::Tui, &mut sink, &cancel)
            .await;
        let second = approvals
            .request(
                &tool_call,
                "agent_run",
                ApprovalMode::Tui,
                &mut sink,
                &cancel,
            )
            .await;

        assert_eq!(first, Approval::AllowToolForSession);
        assert_eq!(second, Approval::AllowOnce);
        assert!(matches!(
            render_rx.try_recv(),
            Ok(crate::cli::ui_sink::RenderEvent::ApprovalRequest { .. })
        ));
        assert!(render_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn cancelled_pending_tui_approval_resolves_to_deny() {
        let (render_tx, mut render_rx) = tokio::sync::mpsc::unbounded_channel();
        let (_reply_tx, reply_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut approvals = ApprovalSession::new();
        approvals.attach_tui(render_tx, reply_rx);
        let cancel = std::sync::atomic::AtomicBool::new(true);
        let mut sink = crate::cli::ui_sink::NullSink;

        let decision = approvals
            .request(&call("bash"), "main", ApprovalMode::Tui, &mut sink, &cancel)
            .await;

        assert_eq!(decision, Approval::Deny);
        assert!(matches!(
            render_rx.try_recv(),
            Ok(crate::cli::ui_sink::RenderEvent::ApprovalRequest { id: 0, .. })
        ));
        assert!(matches!(
            render_rx.try_recv(),
            Ok(crate::cli::ui_sink::RenderEvent::ApprovalCancelled { id: 0 })
        ));
    }
}
