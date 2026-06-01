pub fn fizzbuzz(_n: u32) -> String {
    todo!("implement fizzbuzz")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_is_one() {
        assert_eq!(fizzbuzz(1), "1");
    }

    #[test]
    fn three_is_fizz() {
        assert_eq!(fizzbuzz(3), "Fizz");
    }

    #[test]
    fn five_is_buzz() {
        assert_eq!(fizzbuzz(5), "Buzz");
    }

    #[test]
    fn fifteen_is_fizzbuzz() {
        assert_eq!(fizzbuzz(15), "FizzBuzz");
    }

    #[test]
    fn seven_is_seven() {
        assert_eq!(fizzbuzz(7), "7");
    }
}
