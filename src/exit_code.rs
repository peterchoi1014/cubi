use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
#[allow(dead_code)]
pub enum ExitCode {
    Ok = 0,
    Usage = 2,
    Model = 10,
    Tool = 11,
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
}

impl AppExit {
    pub fn new(code: ExitCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
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

pub fn exit(code: ExitCode) -> ! {
    std::process::exit(code.as_i32())
}
