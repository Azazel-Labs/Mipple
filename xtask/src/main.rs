//! Repo chores, run as `cargo xtask <task>`:
//!
//!   fmt [--check]   rustfmt every crate in the workspace (`--check`: fail on a diff, as CI does)
//!   clippy          clippy on all targets with warnings as errors, then the library without default
//!                   features, as CI runs it
//!   ci              fmt --check, then clippy, then the release tests

use std::process::{Command, ExitCode};

fn cargo(args: &[&str]) -> bool {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    eprintln!("> cargo {}", args.join(" "));
    Command::new(cargo).args(args).status().is_ok_and(|s| s.success())
}

fn fmt(check: bool) -> bool {
    cargo(if check {
        &["fmt", "--all", "--check"]
    } else {
        &["fmt", "--all"]
    })
}

fn clippy() -> bool {
    let all = cargo(&[
        "clippy",
        "--workspace",
        "--all-targets",
        "--release",
        "--",
        "-D",
        "warnings",
    ]);
    // The library alone, without the `cli` and `gpu` features (as `web/` and library users build it).
    all && cargo(&[
        "clippy",
        "-p",
        "mipple",
        "--lib",
        "--no-default-features",
        "--release",
        "--",
        "-D",
        "warnings",
    ])
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let ok = match args.as_slice() {
        ["fmt"] => fmt(false),
        ["fmt", "--check"] => fmt(true),
        ["clippy"] => clippy(),
        ["ci"] => fmt(true) && clippy() && cargo(&["test", "--release"]),
        _ => {
            eprintln!("usage: cargo xtask <fmt [--check] | clippy | ci>");
            return ExitCode::from(2);
        }
    };
    if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}
