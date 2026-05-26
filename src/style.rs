use colored::{ColoredString, Colorize};
use std::fmt::Display;
use std::io::IsTerminal;
use std::sync::atomic::{AtomicU8, Ordering};

const COLOR_OVERRIDE_AUTO: u8 = 0;
const COLOR_OVERRIDE_OFF: u8 = 1;
const COLOR_OVERRIDE_ON: u8 = 2;
static COLOR_OVERRIDE: AtomicU8 = AtomicU8::new(COLOR_OVERRIDE_AUTO);

pub fn should_color() -> bool {
    if let Some(enabled) = color_override() {
        return enabled;
    }
    should_color_with(
        |key| std::env::var_os(key).and_then(|v| v.into_string().ok()),
        std::io::stdout().is_terminal(),
    )
}

pub fn init_color_control() {
    colored::control::set_override(should_color());
}

pub fn set_color_override(enabled: bool) {
    COLOR_OVERRIDE.store(
        if enabled {
            COLOR_OVERRIDE_ON
        } else {
            COLOR_OVERRIDE_OFF
        },
        Ordering::SeqCst,
    );
    colored::control::set_override(enabled);
}

fn should_color_with<F>(env: F, stdout_is_tty: bool) -> bool
where
    F: Fn(&str) -> Option<String>,
{
    if env_flag_enabled(env("CLICOLOR_FORCE")) {
        return true;
    }
    if env("NO_COLOR").is_some() {
        return false;
    }
    match env("CUBI_COLOR").as_deref() {
        Some("on") => return true,
        Some("off") => return false,
        _ => {}
    }
    if matches!(env("CLICOLOR").as_deref(), Some("0")) {
        return false;
    }
    stdout_is_tty
}

fn color_override() -> Option<bool> {
    match COLOR_OVERRIDE.load(Ordering::SeqCst) {
        COLOR_OVERRIDE_OFF => Some(false),
        COLOR_OVERRIDE_ON => Some(true),
        _ => None,
    }
}

fn env_flag_enabled(value: Option<String>) -> bool {
    value.is_some_and(|v| !v.is_empty() && v != "0")
}

pub trait CubiStyle: Display {
    fn bright_cyan(&self) -> ColoredString {
        Colorize::bright_cyan(self.to_string().as_str())
    }

    fn bright_yellow(&self) -> ColoredString {
        Colorize::bright_yellow(self.to_string().as_str())
    }

    fn bright_green(&self) -> ColoredString {
        Colorize::bright_green(self.to_string().as_str())
    }

    fn bright_red(&self) -> ColoredString {
        Colorize::bright_red(self.to_string().as_str())
    }

    fn bright_blue(&self) -> ColoredString {
        Colorize::bright_blue(self.to_string().as_str())
    }

    fn bright_black(&self) -> ColoredString {
        Colorize::bright_black(self.to_string().as_str())
    }

    fn bright_white(&self) -> ColoredString {
        Colorize::bright_white(self.to_string().as_str())
    }

    fn bright_magenta(&self) -> ColoredString {
        Colorize::bright_magenta(self.to_string().as_str())
    }

    fn yellow(&self) -> ColoredString {
        Colorize::yellow(self.to_string().as_str())
    }

    fn bold(&self) -> ColoredString {
        Colorize::bold(self.to_string().as_str())
    }
}

impl<T> CubiStyle for T where T: Display {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env<'a>(vars: &'a [(&str, &str)]) -> impl Fn(&str) -> Option<String> + 'a {
        let map: HashMap<&'a str, &'a str> = vars.iter().copied().collect();
        move |key| map.get(key).map(|v| (*v).to_string())
    }

    #[test]
    fn color_defaults_to_tty_detection() {
        assert!(should_color_with(env(&[]), true));
        assert!(!should_color_with(env(&[]), false));
    }

    #[test]
    fn no_color_disables_color() {
        assert!(!should_color_with(env(&[("NO_COLOR", "1")]), true));
    }

    #[test]
    fn no_color_empty_still_disables_color() {
        assert!(!should_color_with(env(&[("NO_COLOR", "")]), true));
    }

    #[test]
    fn clicolor_force_overrides_no_color_and_tty() {
        assert!(should_color_with(
            env(&[("NO_COLOR", "1"), ("CLICOLOR_FORCE", "1")]),
            false
        ));
    }

    #[test]
    fn clicolor_zero_disables_auto_color() {
        assert!(!should_color_with(env(&[("CLICOLOR", "0")]), true));
    }
}
