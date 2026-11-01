//! Public synthetic byte observations and separately privileged C99 witnesses.
pub mod json;
use json::Value;
use tombstone_ffi::{self as native, Error};

#[derive(Debug, PartialEq)]
pub struct Request {
    before: Vec<u8>,
    removed: Vec<u8>,
    after: Vec<u8>,
    offset: usize,
}
impl Request {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let Value::Object(mut fields) = json::parse(bytes)? else {
            return Err("expected object".into());
        };
        if fields.remove("schema") != Some(Value::Str("trinite.tombstone-request.v1".into())) {
            return Err("schema mismatch".into());
        }
        let mut take = |name: &str| -> Result<Vec<u8>, String> {
            let Some(Value::Array(values)) = fields.remove(name) else {
                return Err("required byte array".into());
            };
            if values.len() > 16 {
                return Err("byte array exceeds 16".into());
            }
            values
                .into_iter()
                .map(|v| match v {
                    Value::Int(n) if (0..=255).contains(&n) => Ok(n as u8),
                    _ => Err("byte must be integer 0..255".into()),
                })
                .collect()
        };
        let before = take("before")?;
        let removed = take("removed")?;
        let after = take("after")?;
        let Some(Value::Int(offset)) = fields.remove("offset") else {
            return Err("required integer offset".into());
        };
        if !(0..=16).contains(&offset) || !fields.is_empty() {
            return Err("unknown field or offset outside 0..16".into());
        }
        Ok(Self {
            before,
            removed,
            after,
            offset: offset as usize,
        })
    }

    /// No native call, labels, hashes, return codes, or receipts in model input.
    pub fn observation(&self) -> String {
        format!("{{\"schema\":\"trinite.tombstone-observation.v1\",\"before\":{:?},\"removed\":{:?},\"after\":{:?},\"offset\":{}}}", self.before, self.removed, self.after, self.offset)
    }

    /// Sole semantic authority: the pinned safe Rust FFI delegates to actual C99.
    pub fn evaluate_privileged(&self) -> Result<String, Error> {
        native::check_abi()?;
        match native::witness_delete(&self.before, &self.removed, &self.after, self.offset) {
            Ok(evidence) => {
                let receipt = native::create_receipt(&evidence, None)?;
                let canonical = receipt.json()?;
                // Self-hash check only; this is NOT a second witness or history proof.
                let parsed = native::parse_receipt(canonical.as_bytes())?;
                native::verify_chain(&[parsed], Some(&receipt.digest()))?;
                Ok(format!("{{\"schema\":\"trinite.tombstone-outcome.v1\",\"label\":\"valid-splice\",\"native_code\":0,\"receipt\":{canonical}}}"))
            }
            Err(Error::Native(code @ (1 | 2 | 6))) => {
                let label = match code {
                    1 => "argument",
                    2 => "length",
                    _ => "mismatch",
                };
                Ok(format!("{{\"schema\":\"trinite.tombstone-outcome.v1\",\"label\":\"{label}\",\"native_code\":{code},\"receipt\":null}}"))
            }
            Err(error) => Err(error),
        }
    }
}
