use assert_cmd::Command;
use std::path::Path;
use tempfile::tempdir;

fn cubi(home: &Path) -> Command {
    let mut cmd = Command::cargo_bin("cubi").expect("cubi binary is built by cargo test");
    cmd.env("HOME", home)
        .env("USERPROFILE", home)
        .env("CUBI_NO_ONBOARD", "1")
        .env("CUBI_COLOR", "off");
    cmd
}

fn completion_script(shell: &str) -> String {
    let home = tempdir().unwrap();
    let output = cubi(home.path())
        .args(["completions", shell])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    String::from_utf8(output).unwrap()
}

#[test]
fn bash_completion_snapshot() {
    // Update intentionally with `cargo insta review` after running with
    // `INSTA_UPDATE=always cargo test --test completions_snapshots`.
    insta::assert_snapshot!("bash_completion", completion_script("bash"));
}

#[test]
fn zsh_completion_snapshot() {
    insta::assert_snapshot!("zsh_completion", completion_script("zsh"));
}

#[test]
fn fish_completion_snapshot() {
    insta::assert_snapshot!("fish_completion", completion_script("fish"));
}
