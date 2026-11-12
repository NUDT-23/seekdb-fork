//! SQL tokenizer for the MiniVectorDB subset.

use crate::error::{DbError, DbResult};

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    /// Identifier or keyword (original text; keywords matched case-insensitively).
    Word(String),
    /// Integer literal text (validated as i64 later, so overflow is an error
    /// at the value level, not the lexer level).
    Int(String),
    /// Floating-point-looking literal (only valid inside vector literals,
    /// which arrive as quoted strings; kept for defensive parsing).
    Float(String),
    /// Single-quoted string, `''` unescaped to `'`.
    Str(String),
    // punctuation
    LParen,
    RParen,
    Comma,
    Semi,
    Star,
    // operators
    Eq,
    Ne,   // != or <>
    Lt,
    Le,
    Gt,
    Ge,
    Minus,
}

pub struct Lexer {
    toks: Vec<Tok>,
    pos: usize,
}

impl Lexer {
    pub fn tokenize(sql: &str) -> DbResult<Lexer> {
        if sql.is_ascii() {
            Self::tokenize_ascii(sql)
        } else {
            Self::tokenize_chars(sql)
        }
    }

    /// ASCII INSERT/SELECT statements avoid materializing a four-byte char per
    /// input byte. Any non-ASCII input uses the original Unicode char path.
    fn tokenize_ascii(sql: &str) -> DbResult<Lexer> {
        let b = sql.as_bytes();
        let mut toks = Vec::new();
        let mut i = 0usize;
        let n = b.len();
        while i < n {
            let c = b[i];
            if matches!(c, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r') {
                i += 1;
                continue;
            }
            match c {
                b'(' => { toks.push(Tok::LParen); i += 1; }
                b')' => { toks.push(Tok::RParen); i += 1; }
                b',' => { toks.push(Tok::Comma); i += 1; }
                b';' => { toks.push(Tok::Semi); i += 1; }
                b'*' => { toks.push(Tok::Star); i += 1; }
                b'=' => { toks.push(Tok::Eq); i += 1; }
                b'<' => {
                    if i + 1 < n && b[i + 1] == b'=' {
                        toks.push(Tok::Le); i += 2;
                    } else if i + 1 < n && b[i + 1] == b'>' {
                        toks.push(Tok::Ne); i += 2;
                    } else {
                        toks.push(Tok::Lt); i += 1;
                    }
                }
                b'>' => {
                    if i + 1 < n && b[i + 1] == b'=' {
                        toks.push(Tok::Ge); i += 2;
                    } else {
                        toks.push(Tok::Gt); i += 1;
                    }
                }
                b'!' => {
                    if i + 1 < n && b[i + 1] == b'=' {
                        toks.push(Tok::Ne); i += 2;
                    } else {
                        return Err(DbError::SyntaxError);
                    }
                }
                b'-' => {
                    if i + 1 < n && b[i + 1] == b'-' { break; }
                    toks.push(Tok::Minus); i += 1;
                }
                b'\'' => {
                    let mut s = String::new();
                    i += 1;
                    let mut span = i;
                    let mut closed = false;
                    while i < n {
                        if b[i] == b'\'' {
                            s.push_str(&sql[span..i]);
                            if i + 1 < n && b[i + 1] == b'\'' {
                                s.push('\'');
                                i += 2;
                                span = i;
                            } else {
                                i += 1;
                                closed = true;
                                break;
                            }
                        } else {
                            i += 1;
                        }
                    }
                    if !closed { return Err(DbError::SyntaxError); }
                    toks.push(Tok::Str(s));
                }
                b'0'..=b'9' => {
                    let start = i;
                    while i < n && b[i].is_ascii_digit() { i += 1; }
                    if i < n && b[i] == b'.' {
                        i += 1;
                        while i < n && b[i].is_ascii_digit() { i += 1; }
                        toks.push(Tok::Float(sql[start..i].to_owned()));
                    } else {
                        toks.push(Tok::Int(sql[start..i].to_owned()));
                    }
                }
                _ => {
                    if c.is_ascii_alphabetic() || c == b'_' {
                        let start = i;
                        i += 1;
                        while i < n && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'$') {
                            i += 1;
                        }
                        toks.push(Tok::Word(sql[start..i].to_owned()));
                    } else {
                        return Err(DbError::SyntaxError);
                    }
                }
            }
        }
        Ok(Lexer { toks, pos: 0 })
    }

    fn tokenize_chars(sql: &str) -> DbResult<Lexer> {
        let b: Vec<char> = sql.chars().collect();
        let mut toks = Vec::new();
        let mut i = 0usize;
        let n = b.len();
        while i < n {
            let c = b[i];
            if c.is_whitespace() {
                i += 1;
                continue;
            }
            match c {
                '(' => {
                    toks.push(Tok::LParen);
                    i += 1;
                }
                ')' => {
                    toks.push(Tok::RParen);
                    i += 1;
                }
                ',' => {
                    toks.push(Tok::Comma);
                    i += 1;
                }
                ';' => {
                    toks.push(Tok::Semi);
                    i += 1;
                }
                '*' => {
                    toks.push(Tok::Star);
                    i += 1;
                }
                '=' => {
                    toks.push(Tok::Eq);
                    i += 1;
                }
                '<' => {
                    if i + 1 < n && b[i + 1] == '=' {
                        toks.push(Tok::Le);
                        i += 2;
                    } else if i + 1 < n && b[i + 1] == '>' {
                        toks.push(Tok::Ne);
                        i += 2;
                    } else {
                        toks.push(Tok::Lt);
                        i += 1;
                    }
                }
                '>' => {
                    if i + 1 < n && b[i + 1] == '=' {
                        toks.push(Tok::Ge);
                        i += 2;
                    } else {
                        toks.push(Tok::Gt);
                        i += 1;
                    }
                }
                '!' => {
                    if i + 1 < n && b[i + 1] == '=' {
                        toks.push(Tok::Ne);
                        i += 2;
                    } else {
                        return Err(DbError::SyntaxError);
                    }
                }
                '-' => {
                    // `--` line comment (defensive; the driver strips these)
                    if i + 1 < n && b[i + 1] == '-' {
                        break;
                    }
                    toks.push(Tok::Minus);
                    i += 1;
                }
                '\'' => {
                    // string literal with '' escaping
                    let mut s = String::new();
                    i += 1;
                    let mut closed = false;
                    while i < n {
                        if b[i] == '\'' {
                            if i + 1 < n && b[i + 1] == '\'' {
                                s.push('\'');
                                i += 2;
                            } else {
                                i += 1;
                                closed = true;
                                break;
                            }
                        } else {
                            s.push(b[i]);
                            i += 1;
                        }
                    }
                    if !closed {
                        return Err(DbError::SyntaxError);
                    }
                    toks.push(Tok::Str(s));
                }
                '0'..='9' => {
                    let start = i;
                    while i < n && b[i].is_ascii_digit() {
                        i += 1;
                    }
                    // decimals are not valid scalar literals in this subset;
                    // keep them lexed so the parser can reject precisely
                    if i < n && b[i] == '.' {
                        i += 1;
                        while i < n && b[i].is_ascii_digit() {
                            i += 1;
                        }
                        let text: String = b[start..i].iter().collect();
                        toks.push(Tok::Float(text));
                    } else {
                        let text: String = b[start..i].iter().collect();
                        toks.push(Tok::Int(text));
                    }
                }
                _ => {
                    if c.is_alphabetic() || c == '_' {
                        let start = i;
                        while i < n && (b[i].is_alphanumeric() || b[i] == '_' || b[i] == '$') {
                            i += 1;
                        }
                        let text: String = b[start..i].iter().collect();
                        toks.push(Tok::Word(text));
                    } else {
                        return Err(DbError::SyntaxError);
                    }
                }
            }
        }
        Ok(Lexer { toks, pos: 0 })
    }

    pub fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    pub fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    pub fn is_kw(&self, kw: &str) -> bool {
        matches!(self.peek(), Some(Tok::Word(w)) if w.eq_ignore_ascii_case(kw))
    }

    pub fn eat_kw(&mut self, kw: &str) -> bool {
        if self.is_kw(kw) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    pub fn expect_kw(&mut self, kw: &str) -> DbResult<()> {
        if self.eat_kw(kw) {
            Ok(())
        } else {
            Err(DbError::SyntaxError)
        }
    }

    pub fn expect(&mut self, t: Tok) -> DbResult<()> {
        if self.peek() == Some(&t) {
            self.pos += 1;
            Ok(())
        } else {
            Err(DbError::SyntaxError)
        }
    }

    pub fn at_end(&self) -> bool {
        self.pos >= self.toks.len()
    }

    /// Consume trailing semicolons; error if anything else remains.
    pub fn expect_done(&mut self) -> DbResult<()> {
        while self.peek() == Some(&Tok::Semi) {
            self.pos += 1;
        }
        if self.at_end() {
            Ok(())
        } else {
            Err(DbError::SyntaxError)
        }
    }

    pub fn ident(&mut self) -> DbResult<String> {
        match self.next() {
            Some(Tok::Word(w)) => Ok(w),
            _ => Err(DbError::SyntaxError),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_fast_path_matches_char_lexer() {
        let cases = [
            "INSERT INTO t VALUES (1,'even','[0.1,-2.3,4e-5]'),(2,'it''s','[1,2]');",
            "SELECT * FROM t WHERE a!=1 AND b<=2 OR c<>3 AND d>=4 -- comment",
            " \t\n\x0b\x0c\r CREATE TABLE _x (id BIGINT);",
            "SELECT 12., 34.56 FROM t;",
            "!",
            "'unterminated",
            "$invalid",
        ];
        for sql in cases {
            let fast = Lexer::tokenize_ascii(sql).map(|lexer| lexer.toks);
            let original = Lexer::tokenize_chars(sql).map(|lexer| lexer.toks);
            assert_eq!(fast, original, "token mismatch for {sql:?}");
        }
    }

    #[test]
    fn non_ascii_uses_original_unicode_path() {
        let sql = "SELECT * FROM 表 WHERE 名称='你好'";
        let public = Lexer::tokenize(sql).map(|lexer| lexer.toks);
        let original = Lexer::tokenize_chars(sql).map(|lexer| lexer.toks);
        assert_eq!(public, original);
    }

    #[test]
    fn ascii_fast_path_random_differential() {
        let alphabet = b"abcXYZ019_'$(),;*<>=!- .\t\n\x0b\x0c\r";
        let mut state = 0x6000_81u64;
        for case in 0..5_000 {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let len = (state as usize >> 32) % 25;
            let mut sql = String::with_capacity(len);
            for _ in 0..len {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                sql.push(alphabet[(state as usize >> 32) % alphabet.len()] as char);
            }
            let fast = Lexer::tokenize_ascii(&sql).map(|lexer| lexer.toks);
            let original = Lexer::tokenize_chars(&sql).map(|lexer| lexer.toks);
            assert_eq!(fast, original, "token mismatch in random case {case}: {sql:?}");
        }
    }
}
