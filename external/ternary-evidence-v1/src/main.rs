use std::io::{self, Read, Write};
use trinite_ternary_evidence_v1::{json::MAX_BYTES, score, Request};
fn write_output(mut writer: impl Write, bytes: &[u8]) -> io::Result<()> {
    match writer.write_all(bytes) {
        Err(error) if error.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        result => result,
    }
}
fn run() -> Result<(), String> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if args.len() != 1 || args[0] != "score" {
        return Err("usage: trinite-ternary-evidence-v1 score < public-request.json".into());
    }
    let mut bytes = Vec::new();
    io::stdin()
        .lock()
        .take((MAX_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|_| "cannot read request")?;
    let request = Request::parse(&bytes)?;
    write_output(io::stdout().lock(), score(&request).as_bytes())
        .map_err(|_| "cannot write response".into())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("ternary-evidence-v1: {error}");
        std::process::exit(2);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    struct Failing(io::ErrorKind);
    impl Write for Failing {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::from(self.0))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    #[test]
    fn early_broken_pipe_only_is_success() {
        assert!(write_output(Failing(io::ErrorKind::BrokenPipe), b"x").is_ok());
        assert!(write_output(Failing(io::ErrorKind::PermissionDenied), b"x").is_err());
    }
}
