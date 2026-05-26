use std::io::IsTerminal;

pub fn should_color() -> bool {
    should_color_with(
        |key| std::env::var_os(key).and_then(|v| v.into_string().ok()),
        std::io::stdout().is_terminal(),
    )
}

pub fn init_color_control() {
    colored::control::set_override(should_color());
}

pub fn set_color_override(enabled: bool) {
    // SAFETY: this is called from the single-threaded REPL command path.
    unsafe { std::env::set_var("CUBI_COLOR", if enabled { "on" } else { "off" }) };
    colored::control::set_override(enabled);
}

fn should_color_with<F>(env: F, stdout_is_tty: bool) -> bool
where
    F: Fn(&str) -> Option<String>,
{
    if env_flag_enabled(env("CLICOLOR_FORCE")) {
        return true;
    }
    if env_nonempty(env("NO_COLOR")) {
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

fn env_nonempty(value: Option<String>) -> bool {
    value.is_some_and(|v| !v.is_empty())
}

fn env_flag_enabled(value: Option<String>) -> bool {
    value.is_some_and(|v| !v.is_empty() && v != "0")
}

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
