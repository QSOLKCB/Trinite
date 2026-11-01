//! Finite contextual relation evaluator. No learner, latent answer or stochastic state.
pub mod json;
use json::Value;
use std::collections::BTreeMap;

pub const SCHEMA: &str = "trinite.context-symbol-request.v1";
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    pub contexts: u8,
    pub symbols: u8,
    pub meanings: u8,
    pub table: Vec<Vec<u8>>,
    pub context_mask: Vec<bool>,
    pub symbol_mask: Vec<bool>,
    pub query: u8,
}
fn take(f: &mut BTreeMap<String, Value>, key: &str) -> Result<Value, String> {
    f.remove(key).ok_or_else(|| format!("missing {key}"))
}
fn integer(v: Value, lo: i64, hi: i64) -> Result<u8, String> {
    match v {
        Value::Int(n) if (lo..=hi).contains(&n) => Ok(n as u8),
        _ => Err("integer outside bounds".into()),
    }
}
fn array(v: Value) -> Result<Vec<Value>, String> {
    match v {
        Value::Array(a) => Ok(a),
        _ => Err("expected array".into()),
    }
}
fn mask(v: Value) -> Result<Vec<bool>, String> {
    array(v)?
        .into_iter()
        .map(|v| match v {
            Value::Bool(b) => Ok(b),
            _ => Err("expected boolean mask".into()),
        })
        .collect()
}
impl Request {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let mut f = match json::parse(bytes)? {
            Value::Object(f) => f,
            _ => return Err("expected object".into()),
        };
        if take(&mut f, "schema")? != Value::Str(SCHEMA.into()) {
            return Err("unknown schema".into());
        }
        let r = Self {
            contexts: integer(take(&mut f, "contexts")?, 1, 8)?,
            symbols: integer(take(&mut f, "symbols")?, 1, 8)?,
            meanings: integer(take(&mut f, "meanings")?, 1, 8)?,
            table: array(take(&mut f, "table")?)?
                .into_iter()
                .map(|row| array(row)?.into_iter().map(|v| integer(v, 0, 7)).collect())
                .collect::<Result<_, String>>()?,
            context_mask: mask(take(&mut f, "context_mask")?)?,
            symbol_mask: mask(take(&mut f, "symbol_mask")?)?,
            query: integer(take(&mut f, "query")?, 0, 7)?,
        };
        if !f.is_empty() {
            return Err("unknown field".into());
        }
        r.validate()?;
        Ok(r)
    }
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=8).contains(&self.contexts)
            || !(1..=8).contains(&self.symbols)
            || !(1..=8).contains(&self.meanings)
            || self.query >= self.meanings
            || self.table.len() != usize::from(self.contexts)
            || self.table.iter().any(|r| {
                r.len() != usize::from(self.symbols) || r.iter().any(|&m| m >= self.meanings)
            })
            || self.context_mask.len() != usize::from(self.contexts)
            || self.symbol_mask.len() != usize::from(self.symbols)
        {
            return Err("invalid finite table, dimensions or query".into());
        }
        if !self.context_mask.contains(&true) || !self.symbol_mask.contains(&true) {
            return Err("empty compatible-world set".into());
        }
        Ok(())
    }
}
// The public carrier has only supplied rules and evidence, never computed answers.
pub fn observation(r: &Request) -> Result<String, String> {
    r.validate()?;
    Ok(format!("{{\"schema\":\"trinite.context-symbol-observation.v1\",\"contexts\":{},\"symbols\":{},\"meanings\":{},\"table\":{:?},\"context_mask\":{:?},\"symbol_mask\":{:?},\"query\":{}}}\n",
        r.contexts, r.symbols, r.meanings, r.table, r.context_mask, r.symbol_mask, r.query))
}
pub fn relation(r: &Request) -> Result<i8, String> {
    r.validate()?;
    let mut support = false;
    let mut contradiction = false;
    for (c, row) in r.table.iter().enumerate() {
        for (s, &meaning) in row.iter().enumerate() {
            if r.context_mask[c] && r.symbol_mask[s] {
                support |= meaning == r.query;
                contradiction |= meaning != r.query;
            }
        }
    }
    Ok(match (support, contradiction) {
        (true, false) => 1,
        (false, true) => -1,
        (true, true) => 0,
        _ => return Err("empty compatible-world set".into()),
    })
}
pub fn privileged_json(r: &Request) -> Result<String, String> {
    Ok(format!(
        "{{\"schema\":\"trinite.context-symbol-outcome.v1\",\"relation\":{}}}\n",
        relation(r)?
    ))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn request() -> Request {
        Request {
            contexts: 2,
            symbols: 2,
            meanings: 2,
            table: vec![vec![0, 1], vec![1, 1]],
            context_mask: vec![true, false],
            symbol_mask: vec![true, false],
            query: 0,
        }
    }
    #[test]
    fn counterfactual_and_missing_context() {
        let mut r = request();
        assert_eq!(relation(&r).unwrap(), 1);
        r.context_mask = vec![false, true];
        assert_eq!(relation(&r).unwrap(), -1);
        r.context_mask = vec![true, true];
        assert_eq!(relation(&r).unwrap(), 0);
        r.symbol_mask = vec![false, true];
        assert_eq!(relation(&r).unwrap(), -1);
        r.query = 1;
        assert_eq!(relation(&r).unwrap(), 1);
    }
    #[test]
    fn direct_api_revalidates() {
        let mut r = request();
        r.context_mask = vec![false, false];
        assert!(relation(&r).is_err());
        assert!(observation(&r).is_err());
        r = request();
        r.table[0].clear();
        assert!(privileged_json(&r).is_err());
        r = request();
        r.query = 2;
        assert!(relation(&r).is_err());
    }
}
