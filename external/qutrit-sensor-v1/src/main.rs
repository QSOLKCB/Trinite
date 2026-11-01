use std::io::{self, Read, Write};
use trinite_qutrit_sensor_v1::{json::MAX_BYTES, process};
fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 || args[1] != "observe" {
        return Err("usage: trinite-qutrit-sensor-v1 observe < request.json".into());
    }
    let mut bytes = Vec::new();
    io::stdin()
        .lock()
        .take((MAX_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|_| "cannot read request")?;
    let output = process(&bytes)?;
    io::stdout()
        .lock()
        .write_all(output.as_bytes())
        .map_err(|_| "cannot write response".into())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("qutrit-sensor-v1: {error}");
        std::process::exit(2);
    }
}
