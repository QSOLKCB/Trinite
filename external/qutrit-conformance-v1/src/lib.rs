// SPDX-License-Identifier: MPL-2.0
//! Original bounded GF(3) Pauli-conjugation implementation; no donor code.
//! Word order is x0,x1,z0,z1, global Pauli phase omitted. Circuits act left to right.
pub mod json;
use json::Value;

pub const CIRCUITS: [&str; 7] = ["S1", "S2", "W", "S1W", "WS1", "S2W", "WS2"];
pub const SCHEMA: &str = "trinite.qutrit-conformance-request.v1";
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    pub word: String,
    pub circuit: String,
    pub task: String,
    pub candidate: String,
}
fn valid_word(word: &str) -> bool {
    word.len() == 4 && word.bytes().all(|b| (b'0'..=b'2').contains(&b))
}
impl Request {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let Value::Object(mut fields) = json::parse(bytes)? else {
            return Err("expected object".into());
        };
        if fields.remove("schema") != Some(Value::Str(SCHEMA.into())) {
            return Err("unknown schema".into());
        }
        let mut string = |key| match fields.remove(key) {
            Some(Value::Str(s)) => Ok(s),
            _ => Err(format!("invalid {key}")),
        };
        let request = Self {
            word: string("word")?,
            circuit: string("circuit")?,
            task: string("task")?,
            candidate: string("candidate")?,
        };
        if !fields.is_empty() {
            return Err("unknown field".into());
        }
        request.validate()?;
        Ok(request)
    }
    pub fn validate(&self) -> Result<(), String> {
        if !valid_word(&self.word)
            || !CIRCUITS.contains(&self.circuit.as_str())
            || !matches!(self.task.as_str(), "apply" | "inverse" | "check")
            || (self.task == "check" && !valid_word(&self.candidate))
            || (self.task != "check" && !self.candidate.is_empty())
        {
            return Err("invalid word/circuit/task/candidate".into());
        }
        Ok(())
    }
}
fn gates(circuit: &str) -> Vec<u8> {
    match circuit {
        "S1" => vec![1],
        "S2" => vec![2],
        "W" => vec![0],
        "S1W" => vec![1, 0],
        "WS1" => vec![0, 1],
        "S2W" => vec![2, 0],
        "WS2" => vec![0, 2],
        _ => unreachable!("validated circuit"),
    }
}
fn transform(word: &str, circuit: &str, inverse: bool) -> String {
    let mut p: Vec<u8> = word.bytes().map(|b| b - b'0').collect();
    let mut operations = gates(circuit);
    if inverse {
        operations.reverse();
    }
    for mut weight in operations {
        if weight == 0 {
            p.swap(0, 1);
            p.swap(2, 3);
        } else {
            if inverse {
                weight = 3 - weight;
            }
            p[1] = (p[1] + weight * p[0]) % 3;
            p[2] = (p[2] + 6 - weight * p[3]) % 3;
        }
    }
    p.into_iter().map(|b| char::from(b + b'0')).collect()
}
pub fn answer(request: &Request) -> Result<String, String> {
    request.validate()?;
    let output = transform(&request.word, &request.circuit, request.task == "inverse");
    Ok(if request.task == "check" {
        if output == request.candidate {
            "1"
        } else {
            "0"
        }
        .into()
    } else {
        output
    })
}
pub fn observation(r: &Request) -> Result<String, String> {
    r.validate()?;
    Ok(format!("{{\"schema\":\"trinite.qutrit-conformance-observation.v1\",\"word\":\"{}\",\"circuit\":\"{}\",\"task\":\"{}\",\"candidate\":\"{}\"}}\n",r.word,r.circuit,r.task,r.candidate))
}
pub fn privileged_json(r: &Request) -> Result<String, String> {
    Ok(format!(
        "{{\"schema\":\"trinite.qutrit-conformance-outcome.v1\",\"answer\":\"{}\"}}\n",
        answer(r)?
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn word(n: usize) -> String {
        (0..4)
            .map(|i| char::from(b'0' + ((n / 3usize.pow(i)) % 3) as u8))
            .collect()
    }
    fn request(word: &str, circuit: &str, task: &str) -> Request {
        Request {
            word: word.into(),
            circuit: circuit.into(),
            task: task.into(),
            candidate: String::new(),
        }
    }
    // Independent computational-basis permutation oracle. Recover U P U^-1 from
    // all nine kets, including exact phase exponents; no production gate helper.
    fn permutation(circuit: &str) -> [usize; 9] {
        let names: Vec<&str> = match circuit {
            "S1" => vec!["S1"],
            "S2" => vec!["S2"],
            "W" => vec!["W"],
            "S1W" => vec!["S1", "W"],
            "S2W" => vec!["S2", "W"],
            "WS1" => vec!["W", "S1"],
            "WS2" => vec!["W", "S2"],
            _ => panic!(),
        };
        std::array::from_fn(|n| {
            let (mut a, mut b) = (n / 3, n % 3);
            for name in &names {
                match *name {
                    "W" => std::mem::swap(&mut a, &mut b),
                    "S1" => b = (b + a) % 3,
                    "S2" => b = (b + 2 * a) % 3,
                    _ => panic!(),
                }
            }
            3 * a + b
        })
    }
    fn oracle(word: &str, circuit: &str, inverse: bool) -> String {
        let p: Vec<usize> = word.bytes().map(|b| usize::from(b - b'0')).collect();
        let forward = permutation(circuit);
        let mut backward = [0; 9];
        for (i, &j) in forward.iter().enumerate() {
            backward[j] = i;
        }
        let (u, ui) = if inverse {
            (backward, forward)
        } else {
            (forward, backward)
        };
        let mut shifts = Vec::new();
        let mut phases = Vec::new();
        for (ket, &before) in ui.iter().enumerate() {
            let (a, b) = (before / 3, before % 3);
            let after = u[3 * ((a + p[0]) % 3) + (b + p[1]) % 3];
            shifts.push([(after / 3 + 3 - ket / 3) % 3, (after % 3 + 3 - ket % 3) % 3]);
            phases.push((p[2] * a + p[3] * b) % 3);
        }
        assert!(shifts.iter().all(|s| *s == shifts[0]));
        let output = [shifts[0][0], shifts[0][1], phases[3], phases[1]];
        for (ket, &phase) in phases.iter().enumerate() {
            assert_eq!(phase, (output[2] * (ket / 3) + output[3] * (ket % 3)) % 3);
        }
        output.iter().map(|n| char::from(b'0' + *n as u8)).collect()
    }
    fn symplectic(a: &str, b: &str) -> i32 {
        let a: Vec<i32> = a.bytes().map(|x| i32::from(x - b'0')).collect();
        let b: Vec<i32> = b.bytes().map(|x| i32::from(x - b'0')).collect();
        (a[0] * b[2] + a[1] * b[3] - a[2] * b[0] - a[3] * b[1]).rem_euclid(3)
    }
    #[test]
    fn exhaustive_basis_conjugation_inverse_and_symplectic() {
        for circuit in CIRCUITS {
            let mut outputs = Vec::new();
            for n in 0..81 {
                let w = word(n);
                let actual = answer(&request(&w, circuit, "apply")).unwrap();
                assert_eq!(actual, oracle(&w, circuit, false));
                assert_eq!(
                    answer(&request(&w, circuit, "inverse")).unwrap(),
                    oracle(&w, circuit, true)
                );
                assert_eq!(answer(&request(&actual, circuit, "inverse")).unwrap(), w);
                outputs.push(actual);
            }
            for a in 0..81 {
                for b in 0..81 {
                    assert_eq!(
                        symplectic(&word(a), &word(b)),
                        symplectic(&outputs[a], &outputs[b])
                    );
                }
            }
        }
    }
    #[test]
    fn negative_sign_order_weight_and_check_candidates() {
        let r = request("1001", "S1", "apply");
        assert_eq!(answer(&r).unwrap(), "1121");
        assert_ne!(answer(&r).unwrap(), "1111"); // wrong Z sign
        assert_ne!(
            answer(&r).unwrap(),
            answer(&request("1001", "S2", "apply")).unwrap()
        );
        assert_ne!(
            answer(&request("1001", "S1W", "apply")).unwrap(),
            answer(&request("1001", "WS1", "apply")).unwrap()
        );
        for n in 0..81 {
            for circuit in CIRCUITS {
                let w = word(n);
                let correct = oracle(&w, circuit, false);
                for c in 0..81 {
                    let mut r = request(&w, circuit, "check");
                    r.candidate = word(c);
                    assert_eq!(
                        answer(&r).unwrap(),
                        if r.candidate == correct { "1" } else { "0" }
                    );
                }
            }
        }
    }
    #[test]
    fn strict_request_and_public_boundary() {
        let valid = r#"{"schema":"trinite.qutrit-conformance-request.v1","word":"1001","circuit":"S1","task":"apply","candidate":""}"#;
        let r = Request::parse(valid.as_bytes()).unwrap();
        assert!(!observation(&r).unwrap().contains("answer"));
        for invalid in [
            valid.replace("1001", "1003"),
            valid.replace("\"S1\"", "true"),
            valid.replace("\"candidate\":\"\"", "\"candidate\":\"0000\""),
            valid.replace("apply", "check"),
            valid.replace("apply", "unknown"),
            valid.replace("\"S1\"", "\"S3\""),
            valid.replace("\"word\":", "\"extra\":0,\"word\":"),
            valid.replace("\"word\":", "\"word\":\"0000\",\"word\":"),
        ] {
            assert!(Request::parse(invalid.as_bytes()).is_err(), "{invalid}");
        }
        let mut bad = r.clone();
        bad.word = "1234".into();
        assert!(answer(&bad).is_err());
        assert!(observation(&bad).is_err());
    }
}
