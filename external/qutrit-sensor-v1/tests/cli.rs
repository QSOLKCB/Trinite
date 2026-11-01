use std::io::{self, ErrorKind, Write};
use std::process::{Command, Stdio};
// An invalid operation can exit before reading its request. Only BrokenPipe is
// expected in that race; all other write failures must still fail the test.
fn write_request(writer: &mut impl Write, bytes: &[u8]) -> io::Result<()> {
    match writer.write_all(bytes) {
        Err(error) if error.kind() == ErrorKind::BrokenPipe => Ok(()),
        result => result,
    }
}
fn run(mode: &str, bytes: &[u8]) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_trinite-qutrit-sensor-v1"))
        .arg(mode)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    write_request(&mut child.stdin.take().unwrap(), bytes).unwrap();
    child.wait_with_output().unwrap()
}
#[test]
fn each_explicit_operation_has_one_json_response() {
    {
        let mode = "observe";
        let out = run(mode, include_bytes!("../example-request.json"));
        assert!(out.status.success());
        assert!(out.stderr.is_empty());
        assert!(trinite_qutrit_sensor_v1::json::parse(&out.stdout).is_ok());
        assert_eq!(out.stdout.iter().filter(|&&c| c == b'\n').count(), 1);
    }
}
#[test]
fn rejected_request_never_emits_partial_success() {
    for bytes in [b"{}".as_slice(), b"[]", b"{\"schema\":\"wrong\"}"] {
        let out = run("observe", bytes);
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

#[test]
fn already_exited_child_still_yields_its_rejection_response() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_trinite-qutrit-sensor-v1"))
        .arg("invalid-operation")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // Keep the writer while ensuring the child has closed its read end. This
    // forces BrokenPipe deterministically, rather than hoping for a race.
    let mut input = child.stdin.take().unwrap();
    assert_eq!(child.wait().unwrap().code(), Some(2));
    write_request(&mut input, include_bytes!("../example-request.json")).unwrap();
    drop(input);
    let out = child.wait_with_output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(out.stdout.is_empty());
    assert!(!out.stderr.is_empty());
}

#[test]
fn only_broken_pipe_is_accepted() {
    struct FailedWriter(ErrorKind);
    impl Write for FailedWriter {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::from(self.0))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    assert!(write_request(&mut FailedWriter(ErrorKind::BrokenPipe), b"request").is_ok());
    for kind in [
        ErrorKind::PermissionDenied,
        ErrorKind::Other,
        ErrorKind::WriteZero,
    ] {
        assert_eq!(
            write_request(&mut FailedWriter(kind), b"request")
                .unwrap_err()
                .kind(),
            kind
        );
    }
    let mut success = Vec::new();
    write_request(&mut success, b"request").unwrap();
    assert_eq!(success, b"request");
}
