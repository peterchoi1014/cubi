use std::fmt::Display;

pub fn status_line(headless: bool, msg: impl Display) {
    if headless {
        eprintln!("{msg}");
    } else {
        println!("{msg}");
    }
}
