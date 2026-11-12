//! Minimal JSON request parser for the official STDIO protocol
//! (`STDIO_PROTOCOL.md`, commit f4d4e2665).
//!
//! The judge sends one JSON object per stdin line:
//!
//! ```json
//! {"id":1,"op":"execute","sql":"SELECT ...;","search_mode":"exact"}
//! {"id":2,"op":"shutdown"}
//! ```
//!
//! The response must echo `id` verbatim including its JSON type (a number id
//! stays a number, a string id stays a string), so the parser keeps the raw
//! source token for `id` instead of a decoded value. Only the four protocol
//! keys are interpreted; any other members are parsed over and ignored. The
//! judge never sends malformed lines; if one arrives anyway we report Bad and
//! the caller logs to stderr and reads on (stdout must stay silent).

pub struct Request {
    /// Raw JSON token of the request id, spliced verbatim into responses.
    pub id_raw: String,
    pub op: String,
    pub sql: Option<String>,
    pub search_mode: Option<String>,
}

pub enum Parsed {
    Ok(Request),
    /// Reason; the line is skipped without any stdout output.
    Bad(&'static str),
}

pub fn parse(line: &str) -> Parsed {
    let b = line.as_bytes();
    let mut i = 0usize;
    skip_ws(b, &mut i);
    if i >= b.len() || b[i] != b'{' {
        return Parsed::Bad("not a JSON object");
    }
    i += 1;
    let mut id_raw: Option<String> = None;
    let mut op: Option<String> = None;
    let mut sql: Option<String> = None;
    let mut search_mode: Option<String> = None;
    skip_ws(b, &mut i);
    if i < b.len() && b[i] == b'}' {
        i += 1;
    } else {
        loop {
            skip_ws(b, &mut i);
            let key = match parse_string(b, &mut i) {
                Some(k) => k,
                None => return Parsed::Bad("object key is not a string"),
            };
            skip_ws(b, &mut i);
            if i >= b.len() || b[i] != b':' {
                return Parsed::Bad("missing ':' after key");
            }
            i += 1;
            skip_ws(b, &mut i);
            match key.as_str() {
                "id" => {
                    let start = i;
                    if skip_value(b, &mut i).is_err() {
                        return Parsed::Bad("invalid id value");
                    }
                    // Raw token, whitespace-free span of the value itself.
                    id_raw = Some(line[start..i].to_string());
                }
                "op" => match parse_string(b, &mut i) {
                    Some(v) => op = Some(v),
                    None => return Parsed::Bad("op is not a string"),
                },
                "sql" => match parse_string(b, &mut i) {
                    Some(v) => sql = Some(v),
                    None => return Parsed::Bad("sql is not a string"),
                },
                "search_mode" => match parse_string(b, &mut i) {
                    Some(v) => search_mode = Some(v),
                    None => return Parsed::Bad("search_mode is not a string"),
                },
                _ => {
                    if skip_value(b, &mut i).is_err() {
                        return Parsed::Bad("invalid member value");
                    }
                }
            }
            skip_ws(b, &mut i);
            if i < b.len() && b[i] == b',' {
                i += 1;
                continue;
            }
            if i < b.len() && b[i] == b'}' {
                i += 1;
                break;
            }
            return Parsed::Bad("expected ',' or '}' in object");
        }
    }
    skip_ws(b, &mut i);
    if i != b.len() {
        return Parsed::Bad("trailing bytes after object");
    }
    let id_raw = match id_raw {
        Some(v) => v,
        None => return Parsed::Bad("missing id"),
    };
    let op = match op {
        Some(v) => v,
        None => return Parsed::Bad("missing op"),
    };
    Parsed::Ok(Request { id_raw, op, sql, search_mode })
}

fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && matches!(b[*i], b' ' | b'\t' | b'\r' | b'\n') {
        *i += 1;
    }
}

/// Skip one JSON value, leaving `i` just past it. Err on malformed input.
fn skip_value(b: &[u8], i: &mut usize) -> Result<(), ()> {
    if *i >= b.len() {
        return Err(());
    }
    match b[*i] {
        b'"' => {
            let mut j = *i + 1;
            while j < b.len() {
                match b[j] {
                    b'\\' => j += 2,
                    b'"' => {
                        *i = j + 1;
                        return Ok(());
                    }
                    _ => j += 1,
                }
            }
            Err(())
        }
        b'{' | b'[' => {
            let open = b[*i];
            let close = if open == b'{' { b'}' } else { b']' };
            let mut depth = 0usize;
            while *i < b.len() {
                match b[*i] {
                    b'"' => {
                        skip_value(b, i)?;
                        continue;
                    }
                    c if c == open => depth += 1,
                    c if c == close => {
                        depth -= 1;
                        if depth == 0 {
                            *i += 1;
                            return Ok(());
                        }
                    }
                    _ => {}
                }
                *i += 1;
            }
            Err(())
        }
        b't' | b'f' | b'n' => {
            for token in ["true", "false", "null"] {
                if b[*i..].starts_with(token.as_bytes()) {
                    *i += token.len();
                    return Ok(());
                }
            }
            Err(())
        }
        c if c == b'-' || c.is_ascii_digit() => {
            let mut j = *i;
            if j < b.len() && b[j] == b'-' {
                j += 1;
            }
            while j < b.len() && (b[j].is_ascii_digit() || matches!(b[j], b'.' | b'e' | b'E' | b'+' | b'-')) {
                j += 1;
            }
            if j == *i {
                return Err(());
            }
            *i = j;
            Ok(())
        }
        _ => Err(()),
    }
}

/// Parse a JSON string (with full escape handling incl. surrogate pairs),
/// leaving `i` just past the closing quote.
fn parse_string(b: &[u8], i: &mut usize) -> Option<String> {
    if *i >= b.len() || b[*i] != b'"' {
        return None;
    }
    *i += 1;
    let mut out = String::new();
    let mut span = *i;
    loop {
        if *i >= b.len() {
            return None;
        }
        match b[*i] {
            b'"' => {
                out.push_str(std::str::from_utf8(&b[span..*i]).ok()?);
                *i += 1;
                return Some(out);
            }
            b'\\' => {
                out.push_str(std::str::from_utf8(&b[span..*i]).ok()?);
                *i += 1;
                if *i >= b.len() {
                    return None;
                }
                match b[*i] {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{08}'),
                    b'f' => out.push('\u{0c}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        *i += 1;
                        let hi = parse_hex4(b, i)?;
                        let cp = if (0xD800..0xDC00).contains(&hi) {
                            if *i + 1 < b.len() && b[*i] == b'\\' && b[*i + 1] == b'u' {
                                *i += 2;
                                let lo = parse_hex4(b, i)?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return None;
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                return None;
                            }
                        } else if (0xDC00..0xE000).contains(&hi) {
                            return None;
                        } else {
                            hi
                        };
                        out.push(char::from_u32(cp)?);
                        span = *i;
                        continue;
                    }
                    _ => return None,
                }
                *i += 1;
                span = *i;
            }
            // Quote and backslash are ASCII and cannot occur inside a UTF-8
            // continuation byte, so byte scanning can defer the UTF-8 check
            // and copy to one complete span instead of one char at a time.
            _ => *i += 1,
        }
    }
}

fn parse_hex4(b: &[u8], i: &mut usize) -> Option<u32> {
    if *i + 4 > b.len() {
        return None;
    }
    let mut v = 0u32;
    for k in 0..4 {
        let c = b[*i + k];
        let d = match c {
            b'0'..=b'9' => (c - b'0') as u32,
            b'a'..=b'f' => (c - b'a' + 10) as u32,
            b'A'..=b'F' => (c - b'A' + 10) as u32,
            _ => return None,
        };
        v = (v << 4) | d;
    }
    *i += 4;
    Some(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_execute_request() {
        match parse(
            r#"{"id":1,"op":"execute","sql":"INSERT INTO t VALUES (1,'你好')","search_mode":"exact"}"#,
        ) {
            Parsed::Ok(req) => {
                assert_eq!(req.id_raw, "1");
                assert_eq!(req.op, "execute");
                assert_eq!(req.sql.unwrap(), "INSERT INTO t VALUES (1,'你好')");
                assert_eq!(req.search_mode.unwrap(), "exact");
            }
            _ => panic!("expected Ok"),
        }
    }

    #[test]
    fn id_type_is_preserved_verbatim() {
        match parse(r#"{"id":"abc","op":"shutdown"}"#) {
            Parsed::Ok(req) => {
                assert_eq!(req.id_raw, "\"abc\"");
                assert_eq!(req.op, "shutdown");
                assert!(req.sql.is_none());
            }
            _ => panic!("expected Ok"),
        }
        match parse(r#"{"op":"shutdown","id":1.50}"#) {
            Parsed::Ok(req) => assert_eq!(req.id_raw, "1.50"),
            _ => panic!("expected Ok"),
        }
    }

    #[test]
    fn string_escapes_and_surrogates() {
        match parse(r#"{"id":7,"op":"execute","sql":"a\n\t\"\\A🎉b"}"#) {
            Parsed::Ok(req) => assert_eq!(req.sql.unwrap(), "a\n\t\"\\A🎉b"),
            _ => panic!("expected Ok"),
        }
        // 🎉 = U+1F389 = D83C DF89 surrogate pair in JSON.
    }

    #[test]
    fn unknown_members_are_ignored() {
        match parse(r#"{"extra":{"nested":[1,2,{"x":null}]},"id":3,"op":"execute","sql":"SELECT 1","search_mode":"ann"}"#) {
            Parsed::Ok(req) => {
                assert_eq!(req.search_mode.unwrap(), "ann");
                assert_eq!(req.sql.unwrap(), "SELECT 1");
            }
            _ => panic!("expected Ok"),
        }
    }

    #[test]
    fn rejects_malformed() {
        assert!(matches!(parse("SELECT 1;"), Parsed::Bad(_)));
        assert!(matches!(parse("{\"op\":\"execute\"}"), Parsed::Bad(_))); // no id
        assert!(matches!(parse("{\"id\":1}"), Parsed::Bad(_))); // no op
        assert!(matches!(parse("{\"id\":1,\"op\":\"x\"} trailing"), Parsed::Bad(_)));
        assert!(matches!(parse(""), Parsed::Bad(_)));
    }

    #[test]
    fn string_escape_boundaries() {
        let valid = [
            (r#"{"id":1,"op":"execute","sql":""}"#, ""),
            (r#"{"id":1,"op":"execute","sql":"a\/b"}"#, "a/b"),
            (r#"{"id":1,"op":"execute","sql":"a\b\f"}"#, "a\u{08}\u{0c}"),
            (r#"{"id":1,"op":"execute","sql":"a\u0000b"}"#, "a\0b"),
            (r#"{"id":1,"op":"execute","sql":"\n你好🎉"}"#, "\n你好🎉"),
        ];
        for (line, expected) in valid {
            match parse(line) {
                Parsed::Ok(req) => assert_eq!(req.sql.unwrap(), expected),
                _ => panic!("expected valid JSON: {line:?}"),
            }
        }
        for sql in [r#"\uD800"#, r#"\uDC00"#, r#"\uD800\uD800"#] {
            let line = format!(r#"{{"id":1,"op":"execute","sql":"{sql}"}}"#);
            assert!(matches!(parse(&line), Parsed::Bad(_)), "accepted {sql}");
        }
    }
}
