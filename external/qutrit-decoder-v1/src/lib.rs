pub mod json;
use json::Value;
use std::collections::BTreeMap;
pub const CODE: &str = "cyclic-5-qutrit-v1";
fn object(bytes: &[u8], fields: &[&str], schema: &str) -> Result<BTreeMap<String, Value>, String> {
    let Value::Object(mut m) = json::parse(bytes)? else {
        return Err("expected object".into());
    };
    if m.len() != fields.len() || fields.iter().any(|k| !m.contains_key(*k)) {
        return Err("exact request fields required".into());
    }
    if m.remove("schema") != Some(Value::Str(schema.into()))
        || m.remove("code") != Some(Value::Str(CODE.into()))
    {
        return Err("unsupported schema or code".into());
    }
    Ok(m)
}
fn trits(value: Value) -> Result<[i8; 4], String> {
    let Value::Array(v) = value else {
        return Err("expected four trits".into());
    };
    if v.len() != 4 {
        return Err("expected four trits".into());
    }
    let mut out = [0; 4];
    for (i, x) in v.into_iter().enumerate() {
        let Value::Int(n) = x else {
            return Err("trit must be integer".into());
        };
        if !(-1..=1).contains(&n) {
            return Err("trit out of range".into());
        }
        out[i] = n as i8;
    }
    Ok(out)
}
fn mask(value: Value) -> Result<[bool; 4], String> {
    let Value::Array(v) = value else {
        return Err("expected four booleans".into());
    };
    if v.len() != 4 {
        return Err("expected four booleans".into());
    }
    let mut out = [false; 4];
    for (i, x) in v.into_iter().enumerate() {
        let Value::Bool(b) = x else {
            return Err("mask must be boolean".into());
        };
        out[i] = b;
    }
    Ok(out)
}
fn list<T: std::fmt::Display>(v: &[T]) -> String {
    format!(
        "[{}]",
        v.iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}
fn balanced(n: i8) -> i8 {
    let r = n.rem_euclid(3);
    if r == 2 {
        -1
    } else {
        r
    }
}
// Finite algebra reference: QSOLKCB/QEC 4ad55333f888ca33200f0e6f062cb7afd20f1eec.
// Independently implemented Rust; see docs/QUTRIT-MODULES-V1.md for attribution.
const CHECKS: [[i8; 10]; 4] = [
    [1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
];
fn syndrome(e: &[i8; 10]) -> [i8; 4] {
    let mut s = [0; 4];
    for (i, g) in CHECKS.iter().enumerate() {
        s[i] = balanced((0..5).map(|j| g[j] * e[j + 5] - g[j + 5] * e[j]).sum());
    }
    s
}
fn candidate(s: [i8; 4]) -> Option<[i8; 10]> {
    if s == [0; 4] {
        return Some([0; 10]);
    }
    for site in 0..5 {
        for x in -1..=1 {
            for z in -1..=1 {
                if x == 0 && z == 0 {
                    continue;
                }
                let mut e = [0; 10];
                e[site] = x;
                e[site + 5] = z;
                if syndrome(&e) == s {
                    return Some(e.map(|n| balanced(-n)));
                }
            }
        }
    }
    None
}
pub fn process(bytes: &[u8]) -> Result<String, String> {
    let mut m = object(
        bytes,
        &["schema", "code", "syndrome", "observed"],
        "trinite.qutrit-syndrome.v1",
    )?;
    let s = trits(m.remove("syndrome").unwrap())?;
    let known = mask(m.remove("observed").unwrap())?;
    if s.iter().zip(known).any(|(&v, k)| !k && v != 0) {
        return Err("missing trit must have zero placeholder".into());
    }
    let correction = if known.iter().all(|&b| b) {
        candidate(s)
    } else {
        None
    };
    let status = if !known.iter().all(|&b| b) {
        "incomplete_observation"
    } else if correction.is_some() {
        "candidate"
    } else {
        "outside_bounded_table"
    };
    let (x, z) = match correction {
        Some(c) => (list(&c[..5]), list(&c[5..])),
        None => ("[]".into(), "[]".into()),
    };
    Ok(format!("{{\"schema\":\"trinite.qutrit-decoding.v1\",\"code\":\"{CODE}\",\"syndrome\":{},\"observed\":{},\"status\":\"{status}\",\"search_assumption\":\"at_most_one_data_qudit_pauli\",\"correction_x\":{x},\"correction_z\":{z}}}\n",list(&s),list(&known)))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn req(s: [i8; 4], m: [bool; 4]) -> Vec<u8> {
        format!("{{\"schema\":\"trinite.qutrit-syndrome.v1\",\"code\":\"{CODE}\",\"syndrome\":{},\"observed\":{}}}",list(&s),list(&m)).into_bytes()
    }
    #[test]
    fn all_single_errors_get_inverse() {
        let mut seen = std::collections::BTreeSet::new();
        seen.insert([0; 4]);
        for site in 0..5 {
            for x in -1..=1 {
                for z in -1..=1 {
                    if x == 0 && z == 0 {
                        continue;
                    }
                    let mut e = [0; 10];
                    e[site] = x;
                    e[site + 5] = z;
                    let s = syndrome(&e);
                    assert!(seen.insert(s));
                    let c = candidate(s).unwrap();
                    assert!(e.iter().zip(c).all(|(a, b)| balanced(a + b) == 0));
                }
            }
        }
        assert_eq!(seen.len(), 41);
    }
    #[test]
    fn all_syndromes_count() {
        let mut count = 0;
        for a in -1..=1 {
            for b in -1..=1 {
                for c in -1..=1 {
                    for d in -1..=1 {
                        count += usize::from(candidate([a, b, c, d]).is_some());
                    }
                }
            }
        }
        assert_eq!(count, 41);
    }
    #[test]
    fn all_masks_fail_closed_except_full() {
        for bits in 0..16 {
            let m = std::array::from_fn(|i| bits & (1 << i) != 0);
            let out = process(&req([0; 4], m)).unwrap();
            assert!(out.contains(if bits == 15 {
                "\"status\":\"candidate\""
            } else {
                "\"status\":\"incomplete_observation\""
            }));
        }
    }
    #[test]
    fn missing_is_not_nonzero() {
        assert!(process(&req([1, 0, 0, 0], [false; 4])).is_err());
    }
    #[test]
    fn hidden_state_is_not_a_decode_input() {
        let s = String::from_utf8(req([0; 4], [true; 4]))
            .unwrap()
            .replace("}", ",\"error\":[0]}");
        assert!(process(s.as_bytes()).is_err());
    }
}
