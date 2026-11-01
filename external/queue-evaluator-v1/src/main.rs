//! Explicit observation or privileged target projection; queue dynamics stay authoritative.
use std::io::{self, Read, Write};
use trinite_queue_sim_v1::{json::MAX_BYTES, observation, simulate, Request};
fn run() -> Result<(), String> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if args.len() != 1 || (args[0] != "observe" && args[0] != "target") {
        return Err("expected explicit observe or target operation".into());
    }
    let mut bytes = Vec::new();
    io::stdin()
        .lock()
        .take((MAX_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|_| "cannot read request")?;
    let request = Request::parse(&bytes)?;
    let output = if args[0] == "observe" {
        observation(&request)?
    } else {
        format!(
            "{{\"schema\":\"trinite.queue-target.v1\",\"dropped\":{}}}\n",
            simulate(&request)?.forecast.dropped
        )
    };
    io::stdout()
        .lock()
        .write_all(output.as_bytes())
        .map_err(|_| "cannot write response".into())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("queue-evaluator-v1: {error}");
        std::process::exit(2);
    }
}
