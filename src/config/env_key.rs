use std::ffi::{OsStr, OsString};

pub(crate) fn normalize_with_case(name: &OsStr, case_insensitive: bool) -> OsString {
    if case_insensitive {
        name.to_ascii_uppercase()
    } else {
        name.to_owned()
    }
}

pub(crate) fn matches_with_case(actual: &OsStr, expected: &OsStr, case_insensitive: bool) -> bool {
    normalize_with_case(actual, case_insensitive) == normalize_with_case(expected, case_insensitive)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizer_preserves_case_in_sensitive_mode() {
        assert_eq!(
            normalize_with_case(OsStr::new("Cubi_Model"), false),
            OsString::from("Cubi_Model")
        );
    }

    #[test]
    fn normalizer_folds_ascii_case_in_windows_mode() {
        assert_eq!(
            normalize_with_case(OsStr::new("Cubi_Model"), true),
            OsString::from("CUBI_MODEL")
        );
    }
}
