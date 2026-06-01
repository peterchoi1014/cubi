pub fn greet(name: &str) -> String {
    let msg = format!("hello, {name}!");
    prnitln!("{msg}");
    msg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greets() {
        assert_eq!(greet("world"), "hello, world!");
    }
}
