//! Deliberately restricted JSON wire grammar: ASCII unescaped strings, i64
//! decimal integers, booleans, arrays and objects. No null/floats/escapes.
//! Fixed protocol keys need no general text decoder; reject rather than repair.
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum Value {
    Int(i64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
    Object(BTreeMap<String, Value>),
}

pub const MAX_BYTES: usize = 16_384;
pub fn parse(input: &[u8]) -> Result<Value, String> {
    if input.len() > MAX_BYTES {
        return Err("request exceeds 16384 bytes".into());
    }
    let mut parser = Parser {
        bytes: input,
        pos: 0,
    };
    let value = parser.value(0)?;
    parser.space();
    if parser.pos != input.len() {
        return Err("trailing input".into());
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}
impl Parser<'_> {
    fn space(&mut self) {
        while matches!(self.bytes.get(self.pos), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            self.pos += 1;
        }
    }
    fn eat(&mut self, byte: u8) -> bool {
        self.space();
        if self.bytes.get(self.pos) == Some(&byte) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn string(&mut self) -> Result<String, String> {
        if !self.eat(b'"') {
            return Err("expected string".into());
        }
        let start = self.pos;
        while let Some(&byte) = self.bytes.get(self.pos) {
            if byte == b'"' {
                let text = String::from_utf8(self.bytes[start..self.pos].to_vec())
                    .map_err(|_| "invalid string")?;
                self.pos += 1;
                return Ok(text);
            }
            if !(0x20..=0x7e).contains(&byte) || byte == b'\\' {
                return Err("only unescaped ASCII strings are admitted".into());
            }
            self.pos += 1;
        }
        Err("unterminated string".into())
    }
    fn value(&mut self, depth: usize) -> Result<Value, String> {
        if depth > 8 {
            return Err("JSON nesting exceeds 8".into());
        }
        self.space();
        match self.bytes.get(self.pos).copied() {
            Some(b'"') => Ok(Value::Str(self.string()?)),
            Some(b'[') => {
                self.pos += 1;
                let mut values = Vec::new();
                if !self.eat(b']') {
                    loop {
                        values.push(self.value(depth + 1)?);
                        if self.eat(b']') {
                            break;
                        }
                        if !self.eat(b',') {
                            return Err("expected array comma".into());
                        }
                    }
                }
                Ok(Value::Array(values))
            }
            Some(b'{') => {
                self.pos += 1;
                let mut values = BTreeMap::new();
                if !self.eat(b'}') {
                    loop {
                        let key = self.string()?;
                        if !self.eat(b':') {
                            return Err("expected colon".into());
                        }
                        if values.insert(key, self.value(depth + 1)?).is_some() {
                            return Err("duplicate object key".into());
                        }
                        if self.eat(b'}') {
                            break;
                        }
                        if !self.eat(b',') {
                            return Err("expected object comma".into());
                        }
                    }
                }
                Ok(Value::Object(values))
            }
            Some(b't') if self.bytes[self.pos..].starts_with(b"true") => {
                self.pos += 4;
                Ok(Value::Bool(true))
            }
            Some(b'f') if self.bytes[self.pos..].starts_with(b"false") => {
                self.pos += 5;
                Ok(Value::Bool(false))
            }
            Some(b'-' | b'0'..=b'9') => {
                let start = self.pos;
                if self.bytes[self.pos] == b'-' {
                    self.pos += 1;
                }
                let digits = self.pos;
                while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
                if digits == self.pos || (self.pos - digits > 1 && self.bytes[digits] == b'0') {
                    return Err("invalid integer".into());
                }
                let text = std::str::from_utf8(&self.bytes[start..self.pos])
                    .map_err(|_| "integer encoding")?;
                if text == "-0" {
                    return Err("negative zero is not canonical".into());
                }
                Ok(Value::Int(
                    text.parse().map_err(|_| "integer out of range")?,
                ))
            }
            _ => Err("unsupported or invalid JSON value".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grammar_rejections() {
        for input in [
            "",
            "null",
            "1.0",
            "01",
            "-0",
            "-",
            "true false",
            "[1,]",
            "{\"x\":1,}",
            "{\"x\":1,\"x\":2}",
            "\"\\u0061\"",
            "\"é\"",
            "9223372036854775808",
            "[[[[[[[[[[0]]]]]]]]]]",
        ] {
            assert!(parse(input.as_bytes()).is_err(), "{input}");
        }
    }
    #[test]
    fn bounds_and_whitespace() {
        assert_eq!(
            parse(b" \r\n[true, false, -1] \t").unwrap(),
            Value::Array(vec![Value::Bool(true), Value::Bool(false), Value::Int(-1)])
        );
        assert!(parse(&vec![b' '; MAX_BYTES + 1]).is_err());
    }
}
