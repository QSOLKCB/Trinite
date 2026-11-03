// SPDX-License-Identifier: MPL-2.0
use std::io::{ErrorKind, Write};
use std::process::{Command, Output, Stdio};

fn run(args: &[&str], bytes: &[u8]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_trinite-qutrit-conformance-v1"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    if let Err(error) = stdin.write_all(bytes) {
        assert_eq!(
            error.kind(),
            ErrorKind::BrokenPipe,
            "unexpected stdin error: {error}"
        );
    }
    drop(stdin);
    child.wait_with_output().unwrap()
}
const REQUEST: &str = r#"{"schema":"trinite.qutrit-conformance-request.v1","word":"1001","circuit":"S1","task":"apply","candidate":""}"#;
#[test]
fn explicit_operations_and_exact_public_response() {
    let public = run(&["observe"], REQUEST.as_bytes());
    assert!(public.status.success());
    assert!(public.stderr.is_empty());
    assert_eq!(
        String::from_utf8(public.stdout).unwrap(),
        format!("{}\n", REQUEST.replace("request.v1", "observation.v1"))
    );
    let privileged = run(&["evaluate-privileged"], REQUEST.as_bytes());
    assert!(privileged.status.success());
    assert!(privileged.stderr.is_empty());
    assert_eq!(
        privileged.stdout,
        b"{\"schema\":\"trinite.qutrit-conformance-outcome.v1\",\"answer\":\"1121\"}\n"
    );
}
#[test]
fn early_stdin_close_and_malformed_requests_have_no_semantic_answer() {
    for args in [vec![], vec!["unknown"], vec!["observe", "extra"]] {
        let result = run(&args, &vec![b' '; 65_536]);
        assert_eq!(result.status.code(), Some(2));
        assert!(result.stdout.is_empty());
        assert!(!result.stderr.is_empty());
    }
    for bytes in [
        b"null".to_vec(),
        vec![b' '; 16_385],
        REQUEST.replace("\"1001\"", "true").into_bytes(),
        REQUEST.replace("\"1001\"", "\"1001\\u0000\"").into_bytes(),
        REQUEST
            .replace("\"candidate\":\"\"", "\"candidate\":null")
            .into_bytes(),
        REQUEST
            .replace(
                "\"circuit\":\"S1\"",
                "\"circuit\":\"S1\",\"circuit\":\"S2\"",
            )
            .into_bytes(),
        format!("{REQUEST} false").into_bytes(),
    ] {
        for operation in ["observe", "evaluate-privileged"] {
            let result = run(&[operation], &bytes);
            assert_eq!(result.status.code(), Some(2));
            assert!(result.stdout.is_empty());
            assert!(!result.stderr.is_empty());
        }
    }
}
