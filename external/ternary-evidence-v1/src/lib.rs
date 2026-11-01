//! Exact descriptive aggregation of explicitly public masked trits, not prediction.
pub mod json;
use json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub const MAX_FACTORS: usize = 64;
pub const MAX_ENTRIES: usize = 64;
pub const MAX_WEIGHT: i64 = 1_000_000_000;

fn object(v: Value, keys: &[&str]) -> Result<BTreeMap<String, Value>, String> {
    let Value::Object(m) = v else {
        return Err("expected object".into());
    };
    if m.len() != keys.len() || keys.iter().any(|k| !m.contains_key(*k)) {
        return Err("missing or unknown field".into());
    }
    Ok(m)
}
fn array(v: Value) -> Result<Vec<Value>, String> {
    if let Value::Array(a) = v {
        Ok(a)
    } else {
        Err("expected array".into())
    }
}
fn integer(v: Value) -> Result<i64, String> {
    if let Value::Int(n) = v {
        Ok(n)
    } else {
        Err("expected integer".into())
    }
}
fn boolean(v: Value) -> Result<bool, String> {
    if let Value::Bool(b) = v {
        Ok(b)
    } else {
        Err("expected boolean".into())
    }
}
fn literal(v: Value, expected: &str) -> Result<(), String> {
    if v == Value::Str(expected.into()) {
        Ok(())
    } else {
        Err("unsupported schema or input kind".into())
    }
}

/// Fields are private so every scoring request must pass complete admission.
#[derive(Debug)]
pub struct Request {
    weights: Vec<i64>,
    entries: Vec<Entry>,
}
#[derive(Debug)]
struct Entry {
    id: String,
    active: bool,
    trits: Vec<i64>,
    mask: Vec<bool>,
}
impl Request {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let mut root = object(
            json::parse(bytes)?,
            &["schema", "input_kind", "weights", "entries"],
        )?;
        literal(
            root.remove("schema").unwrap(),
            "trinite.ternary-evidence-request.v1",
        )?;
        literal(root.remove("input_kind").unwrap(), "generic-public-trits")?;
        let weights = array(root.remove("weights").unwrap())?
            .into_iter()
            .map(integer)
            .collect::<Result<Vec<_>, _>>()?;
        if weights.is_empty()
            || weights.len() > MAX_FACTORS
            || weights.iter().any(|w| !(0..=MAX_WEIGHT).contains(w))
            || weights.iter().all(|w| *w == 0)
        {
            return Err(
                "weights require 1..64 nonnegative bounded integers with positive total".into(),
            );
        }
        let rows = array(root.remove("entries").unwrap())?;
        if rows.is_empty() || rows.len() > MAX_ENTRIES {
            return Err("entries require 1..64 rows".into());
        }
        let mut ids = BTreeSet::new();
        let mut entries = Vec::new();
        for row in rows {
            let mut row = object(row, &["id", "active", "trits", "mask"])?;
            let Value::Str(id) = row.remove("id").unwrap() else {
                return Err("expected ID string".into());
            };
            if id.is_empty()
                || id.len() > 64
                || !id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
                || !ids.insert(id.clone())
            {
                return Err("invalid or duplicate ID".into());
            }
            let active = boolean(row.remove("active").unwrap())?;
            let trits = array(row.remove("trits").unwrap())?
                .into_iter()
                .map(integer)
                .collect::<Result<Vec<_>, _>>()?;
            let mask = array(row.remove("mask").unwrap())?
                .into_iter()
                .map(boolean)
                .collect::<Result<Vec<_>, _>>()?;
            if trits.len() != weights.len()
                || mask.len() != weights.len()
                || trits.iter().any(|t| !(-1..=1).contains(t))
                || trits.iter().zip(&mask).any(|(t, m)| !m && *t != 0)
            {
                return Err("invalid trits, dimensions or noncanonical missing value".into());
            }
            entries.push(Entry {
                id,
                active,
                trits,
                mask,
            });
        }
        Ok(Self { weights, entries })
    }
}

/// Bounds prove |S| <= D <= W <= 64*10^9 < i64::MAX; no products of sums.
/// No ranking, cross-products, probabilities, rewards or completion estimates.
pub fn score(request: &Request) -> String {
    let total: i64 = request.weights.iter().sum();
    let mut rows = Vec::new();
    for entry in &request.entries {
        let mut sum = 0i64;
        let mut observed_weight = 0i64;
        let mut observed_count = 0;
        for ((weight, trit), seen) in request.weights.iter().zip(&entry.trits).zip(&entry.mask) {
            if *seen {
                sum += weight * trit;
                observed_weight += weight;
                observed_count += 1;
            }
        }
        let status = if !entry.active {
            "inactive"
        } else if observed_weight == 0 {
            "abstain"
        } else {
            "scored"
        };
        let value = if status == "scored" {
            format!("{{\"numerator\":{sum},\"denominator\":{observed_weight}}}")
        } else {
            "null".into()
        };
        rows.push(format!("{{\"id\":\"{}\",\"active\":{},\"status\":\"{status}\",\"score\":{value},\"coverage\":{{\"numerator\":{observed_weight},\"denominator\":{total}}},\"observed_count\":{observed_count},\"factor_count\":{}}}", entry.id, entry.active, request.weights.len()));
    }
    format!(
        "{{\"schema\":\"trinite.ternary-evidence-result.v1\",\"entries\":[{}]}}\n",
        rows.join(",")
    )
}
