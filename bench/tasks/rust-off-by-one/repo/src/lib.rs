pub fn in_range(value: i32, lo: i32, hi: i32) -> bool {
    value >= lo && value <= hi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_range_includes_lower() {
        assert!(in_range(0, 0, 10));
    }

    #[test]
    fn in_range_middle() {
        assert!(in_range(5, 0, 10));
    }

    #[test]
    fn in_range_excludes_upper() {
        assert!(!in_range(10, 0, 10));
    }

    #[test]
    fn in_range_excludes_below() {
        assert!(!in_range(-1, 0, 10));
    }
}
