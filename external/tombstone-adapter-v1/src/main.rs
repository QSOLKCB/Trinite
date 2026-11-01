use std::io::{self, Read, Write};
use trinite_tombstone_adapter_v1::{json::MAX_BYTES, Request};

fn run() -> Result<String, u8> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 1 || !(args[0] == "observe" || args[0] == "evaluate-privileged") {
        return Err(2);
    }
    let mut input = Vec::new();
    io::stdin()
        .take((MAX_BYTES + 1) as u64)
        .read_to_end(&mut input)
        .map_err(|_| 3)?;
    let request = Request::parse(&input).map_err(|_| 2)?;
    if args[0] == "observe" {
        Ok(request.observation())
    } else {
        request.evaluate_privileged().map_err(|_| 3)
    }
}
fn main() -> std::process::ExitCode {
    match run() {
        Ok(output) if output.len() < 4096 => {
            if writeln!(io::stdout(), "{output}").is_ok() {
                0.into()
            } else {
                3.into()
            }
        }
        result => {
            let code = result.err().unwrap_or(3);
            // Sanitized diagnostics never include input bytes or native labels.
            let _ = writeln!(
                io::stderr(),
                "{}",
                if code == 2 {
                    "invalid request"
                } else {
                    "infrastructure failure"
                }
            );
            code.into()
        }
    }
}
