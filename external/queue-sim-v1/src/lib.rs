//! Bounded integer FIFO environment, separate from native Trinite and learning.
pub mod json;
use json::Value;
use std::collections::{BTreeMap, VecDeque};

pub const SCHEMA: &str = "trinite.queue-request.v1";
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub capacity: u16,
    pub initial_queue: u16,
    pub arrivals: Vec<u16>,
    pub service: Vec<u16>,
    pub outages: Vec<bool>,
    pub observe_ticks: u16,
    pub observed: [bool; 6],
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
fn integer(value: Value, min: i64, max: i64) -> Result<u16, String> {
    match value {
        Value::Int(n) if (min..=max).contains(&n) => Ok(n as u16),
        _ => Err("integer outside admitted bounds".into()),
    }
}
fn array(value: Value) -> Result<Vec<Value>, String> {
    match value {
        Value::Array(v) => Ok(v),
        _ => Err("expected array".into()),
    }
}
fn numbers(value: Value) -> Result<Vec<u16>, String> {
    array(value)?
        .into_iter()
        .map(|v| integer(v, 0, 32))
        .collect()
}
fn booleans(value: Value) -> Result<Vec<bool>, String> {
    array(value)?
        .into_iter()
        .map(|v| match v {
            Value::Bool(b) => Ok(b),
            _ => Err("expected boolean".into()),
        })
        .collect()
}
impl Request {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let mut f = object(json::parse(bytes)?)?;
        if take(&mut f, "schema")? != Value::Str(SCHEMA.into()) {
            return Err("unknown request schema".into());
        }
        let request = Self {
            capacity: integer(take(&mut f, "capacity")?, 1, 128)?,
            initial_queue: integer(take(&mut f, "initial_queue")?, 0, 128)?,
            arrivals: numbers(take(&mut f, "arrivals")?)?,
            service: numbers(take(&mut f, "service")?)?,
            outages: booleans(take(&mut f, "outages")?)?,
            observe_ticks: integer(take(&mut f, "observe_ticks")?, 1, 255)?,
            observed: booleans(take(&mut f, "observed")?)?
                .try_into()
                .map_err(|_| "observed must have six booleans")?,
        };
        if !f.is_empty() {
            return Err("unknown field".into());
        }
        request.validate()?;
        Ok(request)
    }
    pub fn validate(&self) -> Result<(), String> {
        let n = self.arrivals.len();
        if !(1..=128).contains(&self.capacity)
            || self.initial_queue > self.capacity
            || !(2..=256).contains(&n)
            || self.service.len() != n
            || self.outages.len() != n
            || self.observe_ticks == 0
            || usize::from(self.observe_ticks) >= n
            || self.arrivals.iter().chain(&self.service).any(|&x| x > 32)
        {
            return Err("invalid queue bounds or schedule lengths".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tick {
    pub tick: u16,
    pub offered: u16,
    pub admitted: u16,
    pub dropped: u16,
    pub nominal_service: u16,
    pub effective_service: u16,
    pub outage: bool,
    pub served: u16,
    pub queue: u16,
    pub wait_sum: u32,
    pub wait_max: u16,
    pub oldest_pending_age: u16,
}
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Summary {
    pub offered: u32,
    pub admitted: u32,
    pub dropped: u32,
    pub served: u32,
    pub wait_sum: u32,
    pub wait_max: u16,
}
impl Summary {
    fn add(&mut self, row: &Tick) {
        self.offered += u32::from(row.offered);
        self.admitted += u32::from(row.admitted);
        self.dropped += u32::from(row.dropped);
        self.served += u32::from(row.served);
        self.wait_sum += row.wait_sum;
        self.wait_max = self.wait_max.max(row.wait_max);
    }
    fn json(&self) -> String {
        format!("{{\"offered\":{},\"admitted\":{},\"dropped\":{},\"served\":{},\"wait_sum\":{},\"wait_max\":{}}}",
            self.offered, self.admitted, self.dropped, self.served, self.wait_sum, self.wait_max)
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Simulation {
    pub rows: Vec<Tick>,
    pub summary: Summary,
    pub forecast: Summary,
    pub pending_ages: Vec<u16>,
}

// A single transition implementation owns both prefix observations and outcomes.
// Observe executes only the prefix, never computes or renders future labels.
fn run(request: &Request, steps: usize) -> Simulation {
    let mut queue = VecDeque::from(vec![0u16; usize::from(request.initial_queue)]);
    let mut rows = Vec::new();
    let mut summary = Summary::default();
    let mut forecast = Summary::default();
    for t in 0..steps {
        let tick = t as u16;
        let offered = request.arrivals[t];
        let admitted = offered.min(request.capacity - queue.len() as u16);
        queue.extend(std::iter::repeat_n(tick, usize::from(admitted)));
        let effective_service = if request.outages[t] {
            0
        } else {
            request.service[t]
        };
        let served = effective_service.min(queue.len() as u16);
        let mut wait_sum = 0;
        let mut wait_max = 0;
        for _ in 0..served {
            // served <= queue length, so the pop is guaranteed by construction.
            let wait = tick - queue.pop_front().expect("bounded FIFO service");
            wait_sum += u32::from(wait);
            wait_max = wait_max.max(wait);
        }
        let row = Tick {
            tick,
            offered,
            admitted,
            dropped: offered - admitted,
            nominal_service: request.service[t],
            effective_service,
            outage: request.outages[t],
            served,
            queue: queue.len() as u16,
            wait_sum,
            wait_max,
            oldest_pending_age: queue.front().map_or(0, |born| tick + 1 - born),
        };
        summary.add(&row);
        if tick >= request.observe_ticks {
            forecast.add(&row);
        }
        rows.push(row);
    }
    Simulation {
        rows,
        summary,
        forecast,
        pending_ages: queue.into_iter().map(|born| steps as u16 - born).collect(),
    }
}
pub fn simulate(request: &Request) -> Result<Simulation, String> {
    request.validate()?;
    Ok(run(request, request.arrivals.len()))
}
pub fn observation(request: &Request) -> Result<String, String> {
    request.validate()?;
    let simulation = run(request, usize::from(request.observe_ticks));
    let mask = request
        .observed
        .iter()
        .map(bool::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let rows = simulation
        .rows
        .iter()
        .map(|r| {
            let raw = [
                r.queue,
                r.offered,
                r.effective_service,
                r.served,
                r.dropped,
                r.oldest_pending_age,
            ];
            let values = raw
                .iter()
                .zip(request.observed)
                .map(|(v, known)| if known { v.to_string() } else { "0".into() })
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "{{\"tick\":{},\"outage\":{},\"values\":[{}],\"observed\":[{}]}}",
                r.tick, r.outage, values, mask
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    Ok(format!("{{\"schema\":\"trinite.queue-observation.v1\",\"capacity\":{},\"observe_ticks\":{},\"horizon\":{},\"rows\":[{}]}}\n",
        request.capacity, request.observe_ticks, request.arrivals.len()-usize::from(request.observe_ticks), rows))
}
pub fn privileged_json(request: &Request) -> Result<String, String> {
    let simulation = simulate(request)?;
    let rows = simulation.rows.iter().map(|r| format!(
        "{{\"tick\":{},\"offered\":{},\"admitted\":{},\"dropped\":{},\"nominal_service\":{},\"effective_service\":{},\"outage\":{},\"served\":{},\"queue\":{},\"wait_sum\":{},\"wait_max\":{},\"oldest_pending_age\":{}}}",
        r.tick,r.offered,r.admitted,r.dropped,r.nominal_service,r.effective_service,r.outage,r.served,r.queue,r.wait_sum,r.wait_max,r.oldest_pending_age
    )).collect::<Vec<_>>().join(",");
    let ages = simulation
        .pending_ages
        .iter()
        .map(u16::to_string)
        .collect::<Vec<_>>()
        .join(",");
    Ok(format!("{{\"schema\":\"trinite.queue-ground-truth.v1\",\"privileged\":true,\"capacity\":{},\"observe_ticks\":{},\"steps\":{},\"initial_queue\":{},\"rows\":[{}],\"summary\":{},\"forecast\":{},\"pending_ages\":[{}]}}\n",
        request.capacity, request.observe_ticks, request.arrivals.len(), request.initial_queue, rows,
        simulation.summary.json(), simulation.forecast.json(), ages))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn example() -> Request {
        Request::parse(include_bytes!("../example-request.json")).unwrap()
    }
    #[test]
    fn admission_before_service_and_fifo_wait() {
        let mut r = example();
        r.capacity = 2;
        r.initial_queue = 2;
        r.arrivals = vec![2, 1, 0];
        r.service = vec![1, 0, 2];
        r.outages = vec![false; 3];
        r.observe_ticks = 1;
        let s = simulate(&r).unwrap();
        assert_eq!(
            (s.rows[0].admitted, s.rows[0].dropped, s.rows[0].served),
            (0, 2, 1)
        );
        assert_eq!((s.rows[2].wait_sum, s.rows[2].wait_max), (3, 2));
        assert_eq!(
            s.summary,
            Summary {
                offered: 3,
                admitted: 1,
                dropped: 2,
                served: 3,
                wait_sum: 3,
                wait_max: 2
            }
        );
        assert_eq!(s.forecast.served, 2);
        assert!(s.pending_ages.is_empty());
    }
    #[test]
    fn outages_are_not_missing_measurements() {
        let mut r = example();
        r.observed = [false; 6];
        let s = simulate(&r).unwrap();
        assert_eq!(s.rows[2].served, 0);
        assert_eq!(s.rows[2].effective_service, 0);
        assert_eq!(s.rows[2].nominal_service, 3);
        let text = observation(&r).unwrap();
        assert!(text.contains("\"outage\":true"));
        assert!(text.contains("\"values\":[0,0,0,0,0,0]"));
    }
    #[test]
    fn masks_do_not_change_dynamics() {
        let mut r = example();
        let expected = simulate(&r).unwrap();
        r.observed = [false; 6];
        assert_eq!(simulate(&r).unwrap(), expected);
    }
    #[test]
    fn future_suffix_cannot_change_observation() {
        let mut r = example();
        let expected = observation(&r).unwrap();
        for t in usize::from(r.observe_ticks)..r.arrivals.len() {
            r.arrivals[t] = 32;
            r.service[t] = 0;
            r.outages[t] = true;
        }
        assert_eq!(observation(&r).unwrap(), expected);
    }
    #[test]
    fn no_served_jobs_have_explicit_zero_count_and_censored_ages() {
        let mut r = example();
        r.service.fill(0);
        let s = simulate(&r).unwrap();
        assert_eq!(s.summary.served, 0);
        assert_eq!(s.summary.wait_sum, 0);
        assert_eq!(s.summary.wait_max, 0);
        assert_eq!(s.pending_ages.len(), 8);
        assert_eq!(s.pending_ages[0], 6);
    }
    #[test]
    fn maximum_and_conservation() {
        let r = Request {
            capacity: 128,
            initial_queue: 128,
            arrivals: vec![32; 256],
            service: vec![32; 256],
            outages: vec![false; 256],
            observe_ticks: 255,
            observed: [true; 6],
        };
        let s = simulate(&r).unwrap();
        assert_eq!(s.summary.offered, s.summary.admitted + s.summary.dropped);
        assert_eq!(
            u32::from(r.initial_queue) + s.summary.admitted,
            s.summary.served + s.pending_ages.len() as u32
        );
        assert_eq!(s.rows.len(), 256);
        assert!(s.pending_ages.iter().all(|&age| age <= 256));
    }
    #[test]
    fn direct_api_bounds_are_checked() {
        let mut r = example();
        r.capacity = 0;
        assert!(simulate(&r).is_err());
        assert!(observation(&r).is_err());
        r = example();
        r.service.pop();
        assert!(simulate(&r).is_err());
        r = example();
        r.arrivals[0] = 33;
        assert!(simulate(&r).is_err());
        r = example();
        r.observe_ticks = 6;
        assert!(observation(&r).is_err());
    }
    #[test]
    fn parse_rejects_unknown_missing_and_wrong_types() {
        let source = String::from_utf8(include_bytes!("../example-request.json").to_vec()).unwrap();
        for text in [
            source.replace("\"capacity\":8", "\"capacity\":true"),
            source.replace("\"capacity\":8", "\"capacity\":8,\"other\":0"),
            source.replace("\"capacity\":8,", ""),
            source.replace("\"capacity\":8", "\"capacity\":1"),
            source.replace("\"observe_ticks\":3", "\"observe_ticks\":0"),
        ] {
            assert!(Request::parse(text.as_bytes()).is_err());
        }
    }
}
