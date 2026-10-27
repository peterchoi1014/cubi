use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::str::FromStr;

/// Transient input to startup resolution, not a runtime configuration store.
#[derive(Debug, Clone, Default)]
pub(crate) struct Environment {
    values: BTreeMap<OsString, OsString>,
    case_insensitive: bool,
}

impl Environment {
    pub(crate) fn capture() -> Self {
        Self::from_values(std::env::vars_os())
    }

    pub(crate) fn from_values(values: impl IntoIterator<Item = (OsString, OsString)>) -> Self {
        let mut values = values.into_iter().collect();
        crate::compat::promote_legacy_env(&mut values);
        Self {
            values,
            case_insensitive: cfg!(windows),
        }
    }

    #[cfg(test)]
    pub(crate) fn from_values_with_case(
        values: impl IntoIterator<Item = (OsString, OsString)>,
        case_insensitive: bool,
    ) -> Self {
        let mut values = values.into_iter().collect();
        crate::compat::promote_legacy_env_with_case(&mut values, case_insensitive);
        Self {
            values,
            case_insensitive,
        }
    }

    pub(crate) fn os_value(&self, name: &str) -> Option<&OsStr> {
        self.values
            .iter()
            .find(|(key, _)| {
                super::env_key::matches_with_case(key, OsStr::new(name), self.case_insensitive)
            })
            .map(|(_, value)| value.as_os_str())
    }

    pub(crate) fn value(&self, name: &str) -> Option<&str> {
        self.os_value(name)?.to_str()
    }

    pub(crate) fn string(&self, name: &str) -> Option<String> {
        self.value(name).map(str::to_string)
    }

    pub(crate) fn flag(&self, name: &str) -> bool {
        self.value(name).is_some_and(|v| !v.is_empty() && v != "0")
    }

    pub(crate) fn one_or_true(&self, name: &str) -> bool {
        self.value(name)
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    }

    pub(crate) fn positive<T>(&self, name: &str) -> Option<T>
    where
        T: FromStr + PartialOrd + Default,
    {
        self.value(name)?
            .trim()
            .parse::<T>()
            .ok()
            .filter(|value| *value > T::default())
    }

    pub(crate) fn provider_keys(&self) -> BTreeMap<String, OsString> {
        self.values
            .iter()
            .filter_map(|(name, value)| {
                let normalized = super::env_key::normalize_with_case(name, self.case_insensitive);
                let normalized = normalized.to_str()?;
                let provider = normalized.strip_prefix("CUBI_")?.strip_suffix("_API_KEY")?;
                (!provider.is_empty()).then(|| (name.to_string_lossy().into_owned(), value.clone()))
            })
            .collect()
    }

    pub(crate) fn home_dir(&self) -> Option<PathBuf> {
        let platform_home = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
        for name in ["CUBI_HOME", platform_home] {
            if let Some(value) = self.os_value(name).filter(|v| !v.is_empty()) {
                let path = PathBuf::from(value);
                if path.is_absolute() {
                    return Some(path);
                }
            }
        }
        dirs::home_dir()
    }
}
