//! Exact synthetic note grids with separate privileged events and corruptions.
pub mod json;
use json::Value;
use std::collections::BTreeMap;
pub const SCHEMA: &str = "trinite.music-events-request.v1";
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    pub channel: usize,
    pub start: usize,
    pub end: usize,
    pub pitch: i32,
    pub velocity: i32,
    pub delivered: bool,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Channel {
    pub delay: usize,
    pub transpose: i32,
    pub muted: Vec<bool>,
    pub observed: Vec<bool>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub steps: usize,
    pub observe_ticks: usize,
    pub channels: Vec<Channel>,
    pub events: Vec<Event>,
}
fn object(v: Value) -> Result<BTreeMap<String, Value>, String> {
    if let Value::Object(x) = v {
        Ok(x)
    } else {
        Err("expected object".into())
    }
}
fn take(f: &mut BTreeMap<String, Value>, k: &str) -> Result<Value, String> {
    f.remove(k).ok_or_else(|| format!("missing {k}"))
}
fn integer(v: Value, lo: i64, hi: i64) -> Result<i32, String> {
    match v {
        Value::Int(x) if (lo..=hi).contains(&x) => Ok(x as i32),
        _ => Err("integer outside bounds".into()),
    }
}
fn boolean(v: Value) -> Result<bool, String> {
    if let Value::Bool(x) = v {
        Ok(x)
    } else {
        Err("expected boolean".into())
    }
}
fn array(v: Value) -> Result<Vec<Value>, String> {
    if let Value::Array(x) = v {
        Ok(x)
    } else {
        Err("expected array".into())
    }
}
fn flags(v: Value) -> Result<Vec<bool>, String> {
    array(v)?.into_iter().map(boolean).collect()
}
fn no_extra(f: &BTreeMap<String, Value>) -> Result<(), String> {
    if f.is_empty() {
        Ok(())
    } else {
        Err("unknown field".into())
    }
}
impl Request {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let mut f = object(json::parse(bytes)?)?;
        if take(&mut f, "schema")? != Value::Str(SCHEMA.into()) {
            return Err("unknown schema".into());
        }
        let steps = integer(take(&mut f, "steps")?, 2, 128)? as usize;
        let observe_ticks = integer(take(&mut f, "observe_ticks")?, 1, 127)? as usize;
        let mut channels = Vec::new();
        for value in array(take(&mut f, "channels")?)? {
            let mut c = object(value)?;
            channels.push(Channel {
                delay: integer(take(&mut c, "delay")?, 0, 8)? as usize,
                transpose: integer(take(&mut c, "transpose")?, -12, 12)?,
                muted: flags(take(&mut c, "muted")?)?,
                observed: flags(take(&mut c, "observed")?)?,
            });
            no_extra(&c)?;
        }
        let mut events = Vec::new();
        for value in array(take(&mut f, "events")?)? {
            let mut e = object(value)?;
            events.push(Event {
                channel: integer(take(&mut e, "channel")?, 0, 3)? as usize,
                start: integer(take(&mut e, "start")?, 0, 127)? as usize,
                end: integer(take(&mut e, "end")?, 1, 128)? as usize,
                pitch: integer(take(&mut e, "pitch")?, 0, 127)?,
                velocity: integer(take(&mut e, "velocity")?, 1, 127)?,
                delivered: boolean(take(&mut e, "delivered")?)?,
            });
            no_extra(&e)?;
        }
        no_extra(&f)?;
        let r = Self {
            steps,
            observe_ticks,
            channels,
            events,
        };
        r.validate()?;
        Ok(r)
    }
    pub fn validate(&self) -> Result<(), String> {
        if !(2..=128).contains(&self.steps)
            || self.observe_ticks == 0
            || self.observe_ticks >= self.steps
            || !(1..=4).contains(&self.channels.len())
            || self.events.len() > 128
        {
            return Err("invalid dimensions".into());
        }
        for c in &self.channels {
            if c.delay > 8
                || !(-12..=12).contains(&c.transpose)
                || c.muted.len() != self.steps
                || c.observed.len() != self.steps
            {
                return Err("invalid channel schedule".into());
            }
        }
        for e in &self.events {
            if e.channel >= self.channels.len()
                || e.start >= e.end
                || e.end > self.steps
                || !(0..=127).contains(&e.pitch)
                || !(1..=127).contains(&e.velocity)
            {
                return Err("invalid event".into());
            }
            if !(0..=127).contains(&(e.pitch + self.channels[e.channel].transpose)) {
                return Err("transposed pitch outside bounds".into());
            }
        }
        let events = self.sorted();
        for pair in events.windows(2) {
            if pair[0].channel == pair[1].channel && pair[0].end > pair[1].start {
                return Err("overlapping channel events".into());
            }
        }
        Ok(())
    }
    fn sorted(&self) -> Vec<&Event> {
        let mut e: Vec<_> = self.events.iter().collect();
        e.sort_by_key(|e| (e.channel, e.start));
        e
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Note {
    pub sounding: bool,
    pub pitch: i32,
    pub velocity: i32,
    pub onset: bool,
}
fn note_at(r: &Request, channel: usize, tick: usize, transported: bool) -> Note {
    let c = &r.channels[channel];
    if transported && c.muted[tick] {
        return Note::default();
    }
    for e in &r.events {
        if e.channel != channel || (transported && !e.delivered) {
            continue;
        }
        let delay = if transported { c.delay } else { 0 };
        if e.start + delay <= tick && tick < e.end + delay {
            return Note {
                sounding: true,
                pitch: e.pitch + if transported { c.transpose } else { 0 },
                velocity: e.velocity,
                onset: tick == e.start + delay,
            };
        }
    }
    Note::default()
}
pub fn simulate(r: &Request) -> Result<Vec<Vec<(Note, Note)>>, String> {
    r.validate()?;
    Ok((0..r.steps)
        .map(|t| {
            (0..r.channels.len())
                .map(|c| (note_at(r, c, t, false), note_at(r, c, t, true)))
                .collect()
        })
        .collect())
}
fn note_json(n: Note) -> String {
    format!(
        "{{\"sounding\":{},\"pitch\":{},\"velocity\":{},\"onset\":{}}}",
        n.sounding, n.pitch, n.velocity, n.onset
    )
}
pub fn observation(r: &Request) -> Result<String, String> {
    r.validate()?;
    let rows = (0..r.observe_ticks)
        .map(|t| {
            let cells = (0..r.channels.len())
                .map(|c| {
                    let observed = r.channels[c].observed[t];
                    let n = if observed {
                        note_at(r, c, t, true)
                    } else {
                        Note::default()
                    };
                    format!(
                        "{{\"channel\":{c},\"observed\":{observed},\"note\":{}}}",
                        note_json(n)
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{\"tick\":{t},\"channels\":[{cells}]}}")
        })
        .collect::<Vec<_>>()
        .join(",");
    Ok(format!("{{\"schema\":\"trinite.music-events-observation.v1\",\"channel_count\":{},\"observe_ticks\":{},\"horizon\":{},\"rows\":[{rows}]}}\n",r.channels.len(),r.observe_ticks,r.steps-r.observe_ticks))
}
pub fn privileged_json(r: &Request) -> Result<String, String> {
    let grid = simulate(r)?;
    let events=r.sorted().iter().map(|e|format!("{{\"channel\":{},\"start\":{},\"end\":{},\"pitch\":{},\"velocity\":{},\"delivered\":{}}}",e.channel,e.start,e.end,e.pitch,e.velocity,e.delivered)).collect::<Vec<_>>().join(",");
    let channels = r
        .channels
        .iter()
        .enumerate()
        .map(|(id, c)| {
            format!(
                "{{\"channel\":{id},\"delay\":{},\"transpose\":{}}}",
                c.delay, c.transpose
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let rows = grid
        .iter()
        .enumerate()
        .map(|(t, row)| {
            let cells = row
                .iter()
                .enumerate()
                .map(|(c, (ideal, transported))| {
                    format!(
                        "{{\"channel\":{c},\"ideal\":{},\"transported\":{},\"muted\":{}}}",
                        note_json(*ideal),
                        note_json(*transported),
                        r.channels[c].muted[t]
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{\"tick\":{t},\"channels\":[{cells}]}}")
        })
        .collect::<Vec<_>>()
        .join(",");
    let censored: usize = r
        .events
        .iter()
        .filter(|e| e.delivered)
        .map(|e| {
            let d = r.channels[e.channel].delay;
            (e.end + d).saturating_sub(r.steps) - (e.start + d).saturating_sub(r.steps)
        })
        .sum();
    Ok(format!("{{\"schema\":\"trinite.music-events-ground-truth.v1\",\"privileged\":true,\"steps\":{},\"observe_ticks\":{},\"channels\":[{channels}],\"events\":[{events}],\"censored_transport_ticks\":{censored},\"rows\":[{rows}]}}\n",r.steps,r.observe_ticks))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn example() -> Request {
        Request::parse(include_bytes!("../example-request.json")).unwrap()
    }
    #[test]
    fn half_open_and_delay() {
        let r = example();
        let g = simulate(&r).unwrap();
        assert!(g[0][0].0.onset);
        assert!(!g[0][0].1.sounding);
        assert_eq!(g[1][0].1.pitch, 62);
        assert!(g[1][0].1.onset);
        assert!(!g[2][0].0.sounding);
        assert!(g[2][0].1.sounding);
        assert!(!g[3][0].1.sounding);
    }
    #[test]
    fn missing_is_not_rest() {
        let mut r = example();
        r.channels[0].observed.fill(false);
        let s = observation(&r).unwrap();
        assert!(s.contains("\"observed\":false,\"note\":{\"sounding\":false,\"pitch\":0"));
        assert_ne!(s, observation(&example()).unwrap());
    }
    #[test]
    fn mask_does_not_change_truth() {
        let mut r = example();
        let a = privileged_json(&r).unwrap();
        for c in &mut r.channels {
            c.observed.fill(false)
        }
        assert_eq!(a, privileged_json(&r).unwrap());
    }
    #[test]
    fn event_order_canonical() {
        let mut r = example();
        let a = observation(&r).unwrap();
        let b = privileged_json(&r).unwrap();
        r.events.reverse();
        assert_eq!(a, observation(&r).unwrap());
        assert_eq!(b, privileged_json(&r).unwrap());
    }
    #[test]
    fn loss_and_silence_are_not_unknown() {
        let mut r = example();
        r.events[0].delivered = false;
        assert!(!simulate(&r).unwrap()[1][0].1.sounding);
        r.events[0].delivered = true;
        r.channels[0].muted[1] = true;
        assert!(!simulate(&r).unwrap()[1][0].1.sounding);
        r.channels[0].muted[1] = false;
        r.events.remove(0);
        assert!(!simulate(&r).unwrap()[1][0].1.sounding);
    }
    #[test]
    fn adjacent_same_notes_have_distinct_onsets() {
        let mut r = example();
        r.channels[0].delay = 0;
        r.events = vec![
            Event {
                channel: 0,
                start: 0,
                end: 1,
                pitch: 0,
                velocity: 1,
                delivered: true,
            },
            Event {
                channel: 0,
                start: 1,
                end: 2,
                pitch: 0,
                velocity: 1,
                delivered: true,
            },
        ];
        let g = simulate(&r).unwrap();
        assert!(g[0][0].1.onset && g[1][0].1.onset);
        assert!(g[1][0].1.sounding);
        r.channels[0].transpose = 0;
        assert!(simulate(&r).unwrap()[0][0].1.sounding);
    }
    #[test]
    fn future_does_not_leak() {
        let mut r = example();
        let a = observation(&r).unwrap();
        r.events.push(Event {
            channel: 0,
            start: 4,
            end: 6,
            pitch: 80,
            velocity: 70,
            delivered: true,
        });
        r.channels[0].muted[4] = true;
        r.channels[1].observed[5] = false;
        assert_eq!(a, observation(&r).unwrap());
    }
    #[test]
    fn invalid_direct_api_rejected() {
        for which in 0..9 {
            let mut r = example();
            match which {
                0 => r.steps = 0,
                1 => r.observe_ticks = r.steps,
                2 => r.channels.clear(),
                3 => r.events[0].end = 0,
                4 => r.events[0].pitch = 128,
                5 => r.channels[0].transpose = 13,
                6 => r.channels[0].delay = 9,
                7 => {
                    r.channels[0].observed.pop();
                }
                _ => r.events.push(r.events[0].clone()),
            }
            assert!(observation(&r).is_err());
            assert!(simulate(&r).is_err());
        }
    }
    #[test]
    fn no_clipping_and_censored_tail() {
        let mut r = example();
        r.events[0].pitch = 127;
        assert!(r.validate().is_err());
        r.events[0].pitch = 0;
        r.channels[0].transpose = 0;
        r.channels[0].delay = 8;
        assert!(privileged_json(&r)
            .unwrap()
            .contains("\"censored_transport_ticks\":2"));
    }
}
