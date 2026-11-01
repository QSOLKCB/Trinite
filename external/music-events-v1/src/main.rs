use std::io::{self, Read, Write};
use trinite_music_events_v1::{json::MAX_BYTES, observation, privileged_json, Request};
fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 2
        || !["observe", "simulate-privileged"].contains(&args[1].to_str().unwrap_or(""))
    {
        return Err(
            "usage: trinite-music-events-v1 observe|simulate-privileged < request.json".into(),
        );
    }
    let mut bytes = Vec::new();
    io::stdin()
        .lock()
        .take((MAX_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|_| "cannot read request")?;
    let request = Request::parse(&bytes)?;
    let output = match args[1].to_str().unwrap_or("") {
        "observe" => observation(&request)?,
        _ => privileged_json(&request)?,
    };
    io::stdout()
        .lock()
        .write_all(output.as_bytes())
        .map_err(|_| "cannot write response".to_string())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("music-events-v1: {error}");
        std::process::exit(2);
    }
}
