use std::fmt;

use crate::ollama::ChatStats;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
#[allow(dead_code)]
pub enum ExitCode {
    Ok = 0,
    Usage = 2,
    Model = 10,
    Tool = 11,
    /// Refused: the prompt would exceed the model's context window even
    /// after any auto-compaction attempt. Surfaced as a separate code so
    /// scripts can branch on it (e.g. retry with a smaller system prompt
    /// or `/pin` fewer items).
    Budget = 12,
    /// Network-class failure (connect refused, DNS, TLS). Distinct from
    /// `Model = 10` so headless scripts can branch on connectivity
    /// problems separately from provider/model errors.
    Network = 13,
    /// The agent stopped before producing a final assistant reply (for
    /// example after exhausting its step budget).
    Incomplete = 14,
    Cancelled = 130,
}

impl ExitCode {
    pub fn as_i32(self) -> i32 {
        self as i32
    }
}

#[derive(Debug)]
pub struct AppExit {
    pub code: ExitCode,
    pub message: String,
    pub terminal_usage: Option<TerminalUsage>,
}

#[derive(Debug, Clone)]
pub struct TerminalUsage {
    pub stats: ChatStats,
    pub steps_used: usize,
}

impl AppExit {
    pub fn new(code: ExitCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            terminal_usage: None,
        }
    }

    pub fn with_terminal_usage(mut self, stats: ChatStats, steps_used: usize) -> Self {
        self.terminal_usage = Some(TerminalUsage { stats, steps_used });
        self
    }
}

impl fmt::Display for AppExit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for AppExit {}

pub fn err(code: ExitCode, message: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(AppExit::new(code, message))
}

pub fn err_with_usage(
    code: ExitCode,
    message: impl Into<String>,
    stats: ChatStats,
    steps_used: usize,
) -> anyhow::Error {
    anyhow::Error::new(AppExit::new(code, message).with_terminal_usage(stats, steps_used))
}

pub fn exit(code: ExitCode) -> ! {
    std::process::exit(code.as_i32())
}
