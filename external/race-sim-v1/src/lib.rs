//! Synthetic integer race dynamics; not a model, trainer or racing predictor.
pub mod json;
use json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub const SCHEMA: &str = "trinite.race-request.v1";
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Runner {
    pub id: u16,
    pub observed: [bool; 5],
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub seed: u32,
    pub steps: u16,
    pub track: u8,
    pub runners: Vec<Runner>,
    pub scratches: BTreeSet<u16>,
}
fn object(value: Value) -> Result<BTreeMap<String, Value>, String> {
    match value {
        Value::Object(v) => Ok(v),
        _ => Err("expected object".into()),
    }
}
fn take(fields: &mut BTreeMap<String, Value>, key: &str) -> Result<Value, String> {
    fields.remove(key).ok_or_else(|| format!("missing {key}"))
}
fn integer(value: Value, min: i64, max: i64) -> Result<i64, String> {
    match value {
        Value::Int(n) if (min..=max).contains(&n) => Ok(n),
        _ => Err("integer outside admitted bounds".into()),
    }
}
fn array(value: Value) -> Result<Vec<Value>, String> {
    match value {
        Value::Array(v) => Ok(v),
        _ => Err("expected array".into()),
    }
}
fn exhausted(fields: &BTreeMap<String, Value>) -> Result<(), String> {
    if fields.is_empty() {
        Ok(())
    } else {
        Err("unknown field".into())
    }
}
impl Request {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let mut f = object(json::parse(bytes)?)?;
        if take(&mut f, "schema")? != Value::Str(SCHEMA.into()) {
            return Err("unknown request schema".into());
        }
        let seed = integer(take(&mut f, "seed")?, 0, u32::MAX as i64)? as u32;
        let steps = integer(take(&mut f, "steps")?, 1, 256)? as u16;
        let track = integer(take(&mut f, "track")?, 0, 2)? as u8;
        let mut runners = Vec::new();
        for value in array(take(&mut f, "runners")?)? {
            let mut r = object(value)?;
            let id = integer(take(&mut r, "id")?, 1, u16::MAX as i64)? as u16;
            let mask = array(take(&mut r, "observed")?)?;
            if mask.len() != 5 {
                return Err("observed must have five booleans".into());
            }
            let mut observed = [false; 5];
            for (i, value) in mask.into_iter().enumerate() {
                observed[i] = match value {
                    Value::Bool(b) => b,
                    _ => return Err("observed must be boolean".into()),
                };
            }
            exhausted(&r)?;
            runners.push(Runner { id, observed });
        }
        let mut scratches = BTreeSet::new();
        for value in array(take(&mut f, "scratches")?)? {
            let id = integer(value, 1, u16::MAX as i64)? as u16;
            if !scratches.insert(id) {
                return Err("duplicate scratching".into());
            }
        }
        exhausted(&f)?;
        runners.sort_by_key(|r| r.id);
        let request = Self {
            seed,
            steps,
            track,
            runners,
            scratches,
        };
        request.validate()?;
        Ok(request)
    }
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=256).contains(&self.steps)
            || self.track > 2
            || !(2..=32).contains(&self.runners.len())
        {
            return Err("invalid race bounds".into());
        }
        let ids: BTreeSet<_> = self.runners.iter().map(|r| r.id).collect();
        if ids.len() != self.runners.len() || ids.contains(&0) || !self.scratches.is_subset(&ids) {
            return Err("invalid runner IDs or scratching".into());
        }
        if ids.len() - self.scratches.len() < 2 {
            return Err("at least two active runners required".into());
        }
        Ok(())
    }
    fn sorted(&self) -> Vec<&Runner> {
        let mut runners: Vec<_> = self.runners.iter().collect();
        runners.sort_by_key(|r| r.id);
        runners
    }
}

/// SplitMix64 finalizer with explicit wrapping arithmetic. Each counter is
/// domain-separated by stream, step and runner; no mutable shared RNG state.
pub fn random_word(seed: u32, id: u16, stream: u8, step: u16) -> u64 {
    let counter = (u64::from(id) << 32) | (u64::from(stream) << 16) | u64::from(step);
    let mut z = u64::from(seed).wrapping_add(counter.wrapping_mul(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Latent {
    pub speed: i32,
    pub stamina: i32,
    pub fatigue: i32,
    pub track_fit: [i32; 3],
}
fn latent(seed: u32, id: u16) -> Latent {
    let draw = |stream, size| (random_word(seed, id, stream, 0) % size) as i32;
    Latent {
        speed: 40 + draw(0, 41),
        stamina: 32 + draw(1, 65),
        fatigue: draw(2, 17),
        track_fit: [draw(3, 17) - 8, draw(4, 17) - 8, draw(5, 17) - 8],
    }
}
fn ternary(value: i32, low: i32, high: i32) -> i32 {
    if value < low {
        -1
    } else if value > high {
        1
    } else {
        0
    }
}

/// Only this projection is suitable for a future learner. It contains no seed,
/// latent values, distances, outcomes or future scratching list.
pub fn observation(request: &Request, after_scratches: bool) -> Result<String, String> {
    request.validate()?;
    let ordered = request.sorted();
    let active: Vec<_> = ordered
        .iter()
        .filter(|r| !after_scratches || !request.scratches.contains(&r.id))
        .collect();
    let mut rows = Vec::new();
    for runner in &ordered {
        let lane = active.iter().position(|r| r.id == runner.id);
        let state = latent(request.seed, runner.id);
        let raw = [
            ternary(state.speed, 52, 68),
            ternary(state.stamina, 48, 80),
            -ternary(state.fatigue, 4, 12),
            ternary(state.track_fit[request.track as usize], -2, 2),
            lane.map_or(0, |n| -ternary(n as i32, 2, 8)),
        ];
        let mut observed = runner.observed;
        if lane.is_none() {
            observed[4] = false;
        }
        let signal: Vec<_> = raw
            .iter()
            .zip(observed)
            .map(|(v, present)| if present { v.to_string() } else { "0".into() })
            .collect();
        rows.push(format!(
            "{{\"id\":{},\"active\":{},\"values\":[{}],\"observed\":[{}]}}",
            runner.id,
            lane.is_some(),
            signal.join(","),
            observed.map(|b| b.to_string()).join(",")
        ));
    }
    Ok(format!("{{\"schema\":\"trinite.race-observation.v1\",\"phase\":\"{}\",\"steps\":{},\"track\":{},\"runners\":[{}]}}\n", if after_scratches { "after-scratches" } else { "before-scratches" }, request.steps, request.track, rows.join(",")))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finish {
    pub id: u16,
    pub distance: i32,
    pub latent: Latent,
}
/// Rankings over a fixed duration, not physical race times. Higher distance
/// wins, lower runner ID breaks exact ties. Output is privileged ground truth.
pub fn simulate(request: &Request) -> Result<Vec<Finish>, String> {
    request.validate()?;
    let mut field: Vec<_> = request
        .sorted()
        .into_iter()
        .filter(|r| !request.scratches.contains(&r.id))
        .map(|r| Finish {
            id: r.id,
            distance: 0,
            latent: latent(request.seed, r.id),
        })
        .collect();
    let pace = field.iter().map(|r| r.latent.speed).sum::<i32>() / field.len() as i32;
    let crowd = field.len() as i32 / 4;
    for step in 0..request.steps {
        for (lane, runner) in field.iter_mut().enumerate() {
            let l = &runner.latent;
            let fatigue = (l.fatigue + i32::from(step) * (100 - l.stamina) / 256).min(60);
            let pressure = (pace - l.stamina).max(0) / 8;
            let noise = (random_word(request.seed, runner.id, 6, step) % 7) as i32 - 3;
            let start = if step == 0 { lane as i32 } else { 0 };
            let advance = (l.speed + l.track_fit[request.track as usize]
                - fatigue
                - pressure
                - crowd
                - start
                + noise)
                .clamp(1, 100);
            runner.distance += advance;
        }
    }
    field.sort_by_key(|r| (-r.distance, r.id));
    Ok(field)
}
pub fn privileged_json(request: &Request) -> Result<String, String> {
    let result = simulate(request)?;
    let rows: Vec<_> = result.iter().map(|r| format!("{{\"id\":{},\"distance\":{},\"latent\":{{\"speed\":{},\"stamina\":{},\"fatigue\":{},\"track_fit\":[{},{},{}]}}}}", r.id, r.distance, r.latent.speed, r.latent.stamina, r.latent.fatigue, r.latent.track_fit[0], r.latent.track_fit[1], r.latent.track_fit[2])).collect();
    Ok(format!("{{\"schema\":\"trinite.race-ground-truth.v1\",\"privileged\":true,\"seed\":{},\"ranking\":[{}]}}\n", request.seed, rows.join(",")))
}
