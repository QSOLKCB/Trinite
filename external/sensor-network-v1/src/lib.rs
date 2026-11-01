//! Bounded synthetic sensing, with privileged truth separate from observations.
pub mod json;
use json::Value;
use std::collections::{BTreeMap, BTreeSet};
pub const SCHEMA: &str = "trinite.sensor-network-request.v1";
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sensor {
    pub id: u16,
    pub signal: usize,
    pub noise: Vec<i32>,
    pub fault: Vec<i32>,
    pub fault_value: Vec<i32>,
    pub observed: Vec<bool>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub latent: Vec<Vec<i32>>,
    pub observe_ticks: usize,
    pub sensors: Vec<Sensor>,
}
fn object(v: Value) -> Result<BTreeMap<String, Value>, String> {
    match v {
        Value::Object(x) => Ok(x),
        _ => Err("expected object".into()),
    }
}
fn take(f: &mut BTreeMap<String, Value>, k: &str) -> Result<Value, String> {
    f.remove(k).ok_or_else(|| format!("missing {k}"))
}
fn integer(v: Value, low: i64, high: i64) -> Result<i32, String> {
    match v {
        Value::Int(x) if (low..=high).contains(&x) => Ok(x as i32),
        _ => Err("integer outside bounds".into()),
    }
}
fn array(v: Value) -> Result<Vec<Value>, String> {
    match v {
        Value::Array(x) => Ok(x),
        _ => Err("expected array".into()),
    }
}
fn numbers(v: Value, low: i64, high: i64) -> Result<Vec<i32>, String> {
    array(v)?
        .into_iter()
        .map(|x| integer(x, low, high))
        .collect()
}
fn booleans(v: Value) -> Result<Vec<bool>, String> {
    array(v)?
        .into_iter()
        .map(|x| match x {
            Value::Bool(b) => Ok(b),
            _ => Err("expected boolean".into()),
        })
        .collect()
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
        let latent = array(take(&mut f, "latent")?)?
            .into_iter()
            .map(|v| numbers(v, -1000, 1000))
            .collect::<Result<Vec<_>, _>>()?;
        let observe_ticks = integer(take(&mut f, "observe_ticks")?, 1, 63)? as usize;
        let mut sensors = Vec::new();
        for v in array(take(&mut f, "sensors")?)? {
            let mut s = object(v)?;
            sensors.push(Sensor {
                id: integer(take(&mut s, "id")?, 1, 65535)? as u16,
                signal: integer(take(&mut s, "signal")?, 0, 3)? as usize,
                noise: numbers(take(&mut s, "noise")?, -100, 100)?,
                fault: numbers(take(&mut s, "fault")?, 0, 2)?,
                fault_value: numbers(take(&mut s, "fault_value")?, -1000, 1000)?,
                observed: booleans(take(&mut s, "observed")?)?,
            });
            no_extra(&s)?;
        }
        no_extra(&f)?;
        let request = Self {
            latent,
            observe_ticks,
            sensors,
        };
        request.validate()?;
        Ok(request)
    }
    pub fn validate(&self) -> Result<(), String> {
        let steps = self.latent.len();
        let signals = self.latent.first().map_or(0, Vec::len);
        if !(2..=64).contains(&steps)
            || !(1..=4).contains(&signals)
            || self.observe_ticks == 0
            || self.observe_ticks >= steps
            || self
                .latent
                .iter()
                .any(|row| row.len() != signals || row.iter().any(|x| !(-1000..=1000).contains(x)))
            || !(1..=8).contains(&self.sensors.len())
        {
            return Err("invalid dimensions or latent bounds".into());
        }
        let mut ids = BTreeSet::new();
        for s in &self.sensors {
            if s.id == 0
                || !ids.insert(s.id)
                || s.signal >= signals
                || s.noise.len() != steps
                || s.fault.len() != steps
                || s.fault_value.len() != steps
                || s.observed.len() != steps
                || s.noise.iter().any(|x| !(-100..=100).contains(x))
                || s.fault.iter().any(|x| !(0..=2).contains(x))
                || s.fault_value.iter().any(|x| !(-1000..=1000).contains(x))
                || s.fault
                    .iter()
                    .zip(&s.fault_value)
                    .any(|(&mode, &v)| mode == 0 && v != 0)
            {
                return Err("invalid sensor bounds, schedule, ID or healthy value".into());
            }
        }
        Ok(())
    }
    fn sorted(&self) -> Vec<&Sensor> {
        let mut sensors: Vec<_> = self.sensors.iter().collect();
        sensors.sort_by_key(|s| s.id);
        sensors
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reading {
    pub id: u16,
    pub ideal: i32,
    pub response: i32,
    pub measured: i32,
}
fn reading(r: &Request, s: &Sensor, t: usize) -> Reading {
    let ideal = r.latent[t][s.signal];
    let response = match s.fault[t] {
        0 => ideal,
        1 => ideal + s.fault_value[t],
        2 => s.fault_value[t],
        _ => unreachable!("validated fault"),
    };
    Reading {
        id: s.id,
        ideal,
        response,
        measured: response + s.noise[t],
    }
}
pub fn simulate(r: &Request) -> Result<Vec<Vec<Reading>>, String> {
    r.validate()?;
    let sensors = r.sorted();
    Ok((0..r.latent.len())
        .map(|t| sensors.iter().map(|s| reading(r, s, t)).collect())
        .collect())
}
fn integers(values: &[i32]) -> String {
    values
        .iter()
        .map(i32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}
fn topology(r: &Request) -> String {
    r.sorted()
        .iter()
        .map(|s| format!("{{\"id\":{},\"signal\":{}}}", s.id, s.signal))
        .collect::<Vec<_>>()
        .join(",")
}
pub fn observation(r: &Request) -> Result<String, String> {
    r.validate()?;
    let sensors = r.sorted();
    // No future measurements or labels are computed here.
    let rows = (0..r.observe_ticks)
        .map(|t| {
            let values = sensors
                .iter()
                .map(|s| {
                    if s.observed[t] {
                        reading(r, s, t).measured.to_string()
                    } else {
                        "0".into()
                    }
                })
                .collect::<Vec<_>>()
                .join(",");
            let observed = sensors
                .iter()
                .map(|s| s.observed[t].to_string())
                .collect::<Vec<_>>()
                .join(",");
            format!("{{\"tick\":{t},\"values\":[{values}],\"observed\":[{observed}]}}")
        })
        .collect::<Vec<_>>()
        .join(",");
    Ok(format!("{{\"schema\":\"trinite.sensor-network-observation.v1\",\"signal_count\":{},\"sensors\":[{}],\"observe_ticks\":{},\"horizon\":{},\"rows\":[{}]}}\n",r.latent[0].len(),topology(r),r.observe_ticks,r.latent.len()-r.observe_ticks,rows))
}
pub fn privileged_json(r: &Request) -> Result<String, String> {
    let rows = simulate(r)?;
    let sensors = r.sorted();
    let rows = rows.iter().enumerate().map(|(t,readings)| {
        let readings = readings.iter().zip(&sensors).map(|(v,s)| format!("{{\"id\":{},\"ideal\":{},\"fault\":{},\"fault_value\":{},\"response\":{},\"noise\":{},\"measured\":{}}}",v.id,v.ideal,s.fault[t],s.fault_value[t],v.response,s.noise[t],v.measured)).collect::<Vec<_>>().join(",");
        format!("{{\"tick\":{t},\"latent\":[{}],\"readings\":[{readings}]}}",integers(&r.latent[t]))
    }).collect::<Vec<_>>().join(",");
    Ok(format!("{{\"schema\":\"trinite.sensor-network-ground-truth.v1\",\"privileged\":true,\"signal_count\":{},\"sensors\":[{}],\"observe_ticks\":{},\"steps\":{},\"rows\":[{}]}}\n",r.latent[0].len(),topology(r),r.observe_ticks,r.latent.len(),rows))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn example() -> Request {
        Request::parse(include_bytes!("../example-request.json")).unwrap()
    }
    #[test]
    fn fault_before_noise_and_topology() {
        let r = example();
        let rows = simulate(&r).unwrap();
        assert_eq!(
            rows[1][0],
            Reading {
                id: 1,
                ideal: 1,
                response: 6,
                measured: 7
            }
        );
        assert_eq!(
            rows[2][0],
            Reading {
                id: 1,
                ideal: 2,
                response: -3,
                measured: -4
            }
        );
        assert_eq!(rows[2][1].measured, 13);
    }
    #[test]
    fn masks_do_not_change_truth() {
        let mut r = example();
        let before = privileged_json(&r).unwrap();
        for s in &mut r.sensors {
            s.observed.fill(false);
        }
        assert_eq!(before, privileged_json(&r).unwrap());
        let public = observation(&r).unwrap();
        assert!(public.contains("\"values\":[0,0],\"observed\":[false,false]"));
    }
    #[test]
    fn future_is_not_observed() {
        let mut r = example();
        let before = observation(&r).unwrap();
        for t in r.observe_ticks..r.latent.len() {
            r.latent[t].fill(-1000);
            for s in &mut r.sensors {
                s.noise[t] = 100;
                s.fault[t] = 2;
                s.fault_value[t] = 1000;
                s.observed[t] = false;
            }
        }
        assert_eq!(before, observation(&r).unwrap());
    }
    #[test]
    fn sensor_order_is_canonical() {
        let mut r = example();
        let a = observation(&r).unwrap();
        let b = privileged_json(&r).unwrap();
        r.sensors.reverse();
        assert_eq!(a, observation(&r).unwrap());
        assert_eq!(b, privileged_json(&r).unwrap());
    }
    #[test]
    fn indistinguishable_faults_do_not_leak_labels() {
        let mut r = example();
        r.sensors.truncate(1);
        r.sensors[0].fault.fill(0);
        r.sensors[0].fault_value.fill(0);
        let before = observation(&r).unwrap();
        for row in &mut r.latent {
            row[0] -= 10;
        }
        r.sensors[0].fault.fill(1);
        r.sensors[0].fault_value.fill(10);
        assert_eq!(before, observation(&r).unwrap());
    }
    #[test]
    fn direct_api_rejects_invalid_dimensions_and_values() {
        for which in 0..9 {
            let mut r = example();
            match which {
                0 => r.latent.clear(),
                1 => r.latent[0].clear(),
                2 => r.observe_ticks = 4,
                3 => r.sensors[0].signal = 4,
                4 => r.sensors[0].fault[0] = 3,
                5 => r.sensors[0].fault_value[0] = 1,
                6 => r.sensors[0].noise[0] = 101,
                7 => r.sensors[1].id = 1,
                _ => {
                    r.sensors[0].observed.pop();
                }
            }
            assert!(simulate(&r).is_err());
            assert!(observation(&r).is_err());
        }
    }
    #[test]
    fn extrema_do_not_clip() {
        let mut r = example();
        r.latent.iter_mut().for_each(|v| v.fill(1000));
        let s = &mut r.sensors[0];
        s.fault.fill(1);
        s.fault_value.fill(1000);
        s.noise.fill(100);
        assert_eq!(simulate(&r).unwrap()[0][0].measured, 2100);
        r.latent.iter_mut().for_each(|v| v.fill(-1000));
        let s = &mut r.sensors[0];
        s.fault_value.fill(-1000);
        s.noise.fill(-100);
        assert_eq!(simulate(&r).unwrap()[0][0].measured, -2100);
    }
    #[test]
    fn strict_fields() {
        let source = String::from_utf8(include_bytes!("../example-request.json").to_vec()).unwrap();
        for bad in [
            source.replace("\"id\":1", "\"id\":true"),
            source.replace("\"id\":1", "\"id\":1,\"hidden\":0"),
            source.replace("\"observe_ticks\":2", "\"observe_ticks\":0"),
            source.replace("\"signal\":0", "\"signal\":3"),
        ] {
            assert!(Request::parse(bad.as_bytes()).is_err());
        }
    }
}
