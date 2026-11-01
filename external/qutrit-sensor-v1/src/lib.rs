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
pub fn process(bytes: &[u8]) -> Result<String, String> {
    let mut m = object(
        bytes,
        &[
            "schema",
            "code",
            "true_syndrome",
            "additive_noise",
            "observed",
        ],
        "trinite.qutrit-sensor-request.v1",
    )?;
    let truth = trits(m.remove("true_syndrome").unwrap())?;
    let noise = trits(m.remove("additive_noise").unwrap())?;
    let known = mask(m.remove("observed").unwrap())?;
    let measured: [i8; 4] = std::array::from_fn(|i| {
        if known[i] {
            balanced(truth[i] + noise[i])
        } else {
            0
        }
    });
    Ok(format!("{{\"schema\":\"trinite.qutrit-syndrome.v1\",\"code\":\"{CODE}\",\"syndrome\":{},\"observed\":{}}}\n",list(&measured),list(&known)))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn req(t: [i8; 4], n: [i8; 4], m: [bool; 4]) -> Vec<u8> {
        format!("{{\"schema\":\"trinite.qutrit-sensor-request.v1\",\"code\":\"{CODE}\",\"true_syndrome\":{},\"additive_noise\":{},\"observed\":{}}}",list(&t),list(&n),list(&m)).into_bytes()
    }
    #[test]
    fn modular_noise() {
        for a in -1..=1 {
            for b in -1..=1 {
                let out = process(&req([a; 4], [b; 4], [true; 4])).unwrap();
                let Value::Object(m) = json::parse(out.as_bytes()).unwrap() else {
                    panic!()
                };
                assert_eq!(
                    m["syndrome"],
                    Value::Array(
                        (0..4)
                            .map(|_| Value::Int(i64::from(balanced(a + b))))
                            .collect()
                    )
                );
            }
        }
    }
    #[test]
    fn all_masks_canonical() {
        for bits in 0..16 {
            let known = std::array::from_fn(|i| bits & (1 << i) != 0);
            let out = process(&req([1; 4], [0; 4], known)).unwrap();
            let Value::Object(m) = json::parse(out.as_bytes()).unwrap() else {
                panic!()
            };
            assert_eq!(m.len(), 4);
            assert_eq!(
                m["syndrome"],
                Value::Array(known.map(|k| Value::Int(i64::from(k))).into())
            );
        }
    }
    #[test]
    fn hidden_components_not_emitted() {
        assert_eq!(
            process(&req([1; 4], [-1; 4], [true; 4])),
            process(&req([0; 4], [0; 4], [true; 4]))
        );
    }
}
