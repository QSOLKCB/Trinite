// SPDX-License-Identifier: Apache-2.0
// ARCH OMI reference: Brian Thorne. Adapted to Rust with strict public/privileged
// operations from QSOLKCB/VANILLAJS 9d228aed9791dccac6e62c9dbc55738cc26ac679.
// Encoding and attribution limits: docs/OMI-MODULE-V1.md. Parser is repository MPL-2.0.
pub mod json;
use json::Value;

pub const SCHEMA: &str = "trinite.omi-request.v1";
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    pub word: u16,
    pub task: String,
    pub horizon: u8,
}
impl Request {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let Value::Object(mut f) = json::parse(bytes)? else {
            return Err("expected object".into());
        };
        if f.remove("schema") != Some(Value::Str(SCHEMA.into())) {
            return Err("unknown schema".into());
        }
        let word = match f.remove("word") {
            Some(Value::Int(n)) if (0..=65535).contains(&n) => n as u16,
            _ => return Err("invalid word".into()),
        };
        let task = match f.remove("task") {
            Some(Value::Str(s)) => s,
            _ => return Err("invalid task".into()),
        };
        let horizon = match f.remove("horizon") {
            Some(Value::Int(n)) if (1..=7).contains(&n) => n as u8,
            _ => return Err("invalid horizon".into()),
        };
        if !f.is_empty() {
            return Err("unknown field".into());
        }
        let r = Self {
            word,
            task,
            horizon,
        };
        r.validate()?;
        Ok(r)
    }
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=7).contains(&self.horizon)
            || !matches!(
                self.task.as_str(),
                "next" | "inverse" | "forward" | "backward" | "period"
            )
            || (matches!(self.task.as_str(), "next" | "inverse" | "period") && self.horizon != 1)
        {
            return Err("invalid task/horizon".into());
        }
        Ok(())
    }
}
pub fn delta(word: u16) -> u16 {
    word.rotate_left(1) ^ word.rotate_left(3) ^ word.rotate_right(2) ^ 0x5a3c
}
pub fn advance(mut word: u16, steps: u8) -> u16 {
    for _ in 0..steps {
        word = delta(word);
    }
    word
}
pub fn period(word: u16) -> u8 {
    (1..=8)
        .find(|&n| advance(word, n) == word)
        .expect("exhaustively verified delta closure")
}
pub fn observation(r: &Request) -> Result<String, String> {
    r.validate()?;
    Ok(format!("{{\"schema\":\"trinite.omi-observation.v1\",\"word\":{},\"task\":\"{}\",\"horizon\":{}}}\n",r.word,r.task,r.horizon))
}
pub fn answer(r: &Request) -> Result<String, String> {
    r.validate()?;
    if r.task == "period" {
        return Ok(period(r.word).to_string());
    }
    let n = if matches!(r.task.as_str(), "inverse" | "backward") {
        8 - r.horizon
    } else {
        r.horizon
    };
    Ok(format!("{:04X}", advance(r.word, n)))
}
pub fn privileged_json(r: &Request) -> Result<String, String> {
    Ok(format!(
        "{{\"schema\":\"trinite.omi-outcome.v1\",\"answer\":\"{}\"}}\n",
        answer(r)?
    ))
}
#[cfg(test)]
mod tests {
    use super::*;
    // Arithmetic-only bit extraction and rotation. No XOR, shift or production delta.
    fn oracle(x: u16) -> u16 {
        let x = u32::from(x);
        let rot = |n: u32| (x * 2u32.pow(n)) % 65536 + x / 2u32.pow(16 - n);
        let inputs = [rot(1), rot(3), rot(14), 0x5a3c];
        (0..16)
            .map(|b| ((inputs.iter().map(|v| v / 2u32.pow(b) % 2).sum::<u32>()) % 2) * 2u32.pow(b))
            .sum::<u32>() as u16
    }
    #[test]
    fn exhaustive_oracle_permutation_and_periods() {
        let mut seen = vec![false; 65536];
        let mut counts = [0; 9];
        for word in 0..=u16::MAX {
            let next = delta(word);
            assert_eq!(next, oracle(word));
            assert!(!seen[usize::from(next)]);
            seen[usize::from(next)] = true;
            assert_eq!(advance(word, 8), word);
            assert_eq!(delta(advance(word, 7)), word);
            counts[usize::from(period(word))] += 1;
        }
        assert_eq!(counts, [0, 8, 56, 0, 4032, 0, 0, 0, 61440]);
    }
    #[test]
    fn invalid_direct_request_and_no_answer_in_observation() {
        let mut r = Request {
            word: 42,
            task: "forward".into(),
            horizon: 7,
        };
        assert!(!observation(&r).unwrap().contains("answer"));
        r.horizon = 8;
        assert!(answer(&r).is_err());
        assert!(observation(&r).is_err());
        r.horizon = 1;
        r.task = "hidden".into();
        assert!(answer(&r).is_err());
    }
}
