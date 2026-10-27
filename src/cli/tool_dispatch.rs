use crate::hooks::{HookDecision, HookRegistry};
use crate::ollama::ToolCall;
use crate::permissions::Permissions;
use crate::policy::Policy;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

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

pub(crate) trait ToolCallHost {
    fn approve_tool(&mut self, tool_name: &str) -> bool;

    fn execute_authorized<'a>(
        &'a mut self,
        call: &'a ToolCall,
    ) -> Pin<Box<dyn Future<Output = ToolOutcome> + 'a>>;
}

impl ToolCallHost for super::ChatCLI {
    fn approve_tool(&mut self, tool_name: &str) -> bool {
        self.prompt_tool_approval(tool_name)
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
        HookDecision::Allow if !host.approve_tool(&call.function.name) => {
            ToolOutcome::denied(format!(
                "[tool denied] user declined approval for `{}`",
                call.function.name
            ))
        }
        HookDecision::Allow => host.execute_authorized(call).await,
    };

    access
        .hooks
        .fire_post_tool_use(&call.function.name, &outcome.text, outcome.is_failure())
        .await;
    outcome
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
        approved: bool,
        executions: usize,
    }

    impl ToolCallHost for FakeHost {
        fn approve_tool(&mut self, _: &str) -> bool {
            self.approved
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
            approved: true,
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
            approved: true,
            executions: 0,
        };

        let outcome = dispatch_tool(&access, &mut host, &call("read_file")).await;

        assert!(!outcome.is_failure());
        assert_eq!(host.executions, 1);
        assert_eq!(std::fs::read_to_string(audit).unwrap(), "pre\npost\n");
    }
}
