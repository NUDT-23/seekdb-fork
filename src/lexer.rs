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
