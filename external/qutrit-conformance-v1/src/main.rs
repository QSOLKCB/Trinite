use std::io::{self, Read, Write};
use trinite_qutrit_conformance_v1::{json::MAX_BYTES, observation, privileged_json, Request};
fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 2 || !(args[1] == "observe" || args[1] == "evaluate-privileged") {
        return Err(
            "usage: trinite-qutrit-conformance-v1 observe|evaluate-privileged < request.json"
                .into(),
        );
    }
    let mut bytes = Vec::new();
    io::stdin()
        .lock()
        .take((MAX_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|_| "cannot read request")?;
    let request = Request::parse(&bytes)?;
    let output = if args[1] == "observe" {
        observation(&request)?
    } else {
        privileged_json(&request)?
    };
    io::stdout()
        .lock()
        .write_all(output.as_bytes())
        .map_err(|_| "cannot write response".to_string())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("qutrit-conformance-v1: {error}");
        std::process::exit(2);
    }
}
