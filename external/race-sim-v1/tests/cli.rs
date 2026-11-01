use std::io::Write;
use std::process::{Command, Stdio};
fn run(mode: &str, bytes: &[u8]) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_trinite-race-sim-v1"))
        .arg(mode)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(bytes).unwrap();
    child.wait_with_output().unwrap()
}
#[test]
fn each_explicit_operation_has_one_json_response() {
    for mode in ["observe-before", "observe-after", "simulate-privileged"] {
        let out = run(mode, include_bytes!("../example-request.json"));
        assert!(out.status.success());
        assert!(out.stderr.is_empty());
        assert!(trinite_race_sim_v1::json::parse(&out.stdout).is_ok());
        assert_eq!(out.stdout.iter().filter(|&&c| c == b'\n').count(), 1);
    }
}
#[test]
fn rejected_request_never_emits_partial_success() {
    for bytes in [b"{}".as_slice(), b"[]", b"{\"schema\":\"wrong\"}"] {
        let out = run("observe-before", bytes);
        assert_eq!(out.status.code(), Some(2));
        assert!(out.stdout.is_empty());
        assert!(!out.stderr.is_empty());
    }
}
#[test]
fn default_operation_is_not_privileged() {
    let out = run("", include_bytes!("../example-request.json"));
    assert_eq!(out.status.code(), Some(2));
    assert!(out.stdout.is_empty());
}
