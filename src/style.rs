use crate::config::UiConfig;
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
    crate::config::with(|config| should_color_with(&config.ui, std::io::stdout().is_terminal()))
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

fn should_color_with(ui: &UiConfig, stdout_is_tty: bool) -> bool {
    if ui.force_color {
        return true;
    }
    if ui.no_color {
        return false;
    }
    match ui.color.as_deref() {
        Some("on") => return true,
        Some("off") => return false,
        _ => {}
    }
    if ui.clicolor_off {
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
    use crate::config::{Environment, RuntimeConfig, StartupOverrides};
    use crate::onboarding::AppConfig;

    fn env(vars: &[(&str, &str)]) -> UiConfig {
        let env = Environment::from_values(
            vars.iter()
                .map(|(key, value)| ((*key).into(), (*value).into())),
        );
        RuntimeConfig::resolve(&env, AppConfig::default(), &StartupOverrides::default()).ui
    }

    #[test]
    fn color_defaults_to_tty_detection() {
        assert!(should_color_with(&env(&[]), true));
        assert!(!should_color_with(&env(&[]), false));
    }

    #[test]
    fn no_color_disables_color() {
        assert!(!should_color_with(&env(&[("NO_COLOR", "1")]), true));
    }

    #[test]
    fn no_color_empty_still_disables_color() {
        assert!(!should_color_with(&env(&[("NO_COLOR", "")]), true));
    }

    #[test]
    fn clicolor_force_overrides_no_color_and_tty() {
        assert!(should_color_with(
            &env(&[("NO_COLOR", "1"), ("CLICOLOR_FORCE", "1")]),
            false
        ));
    }

    #[test]
    fn clicolor_zero_disables_auto_color() {
        assert!(!should_color_with(&env(&[("CLICOLOR", "0")]), true));
    }

    #[test]
    fn cubi_color_overrides_tty_but_not_no_color() {
        for (vars, tty, expected) in [
            (vec![("CUBI_COLOR", "on"), ("CLICOLOR", "0")], false, true),
            (vec![("CUBI_COLOR", "off")], true, false),
            (vec![("CUBI_COLOR", "on"), ("NO_COLOR", "")], true, false),
        ] {
            assert_eq!(should_color_with(&env(&vars), tty), expected);
        }
    }
}
