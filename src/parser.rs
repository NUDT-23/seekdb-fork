//! Parser for the MiniVectorDB SQL subset -> AST.

use crate::error::{DbError, DbResult};
use crate::lexer::{Lexer, Tok};
use crate::value::{parse_bigint, parse_vector};

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(i64),
    /// Single-quoted string (vectors are bound from this form).
    Str(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Cmp { col: String, op: CmpOp, lit: Literal },
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProjItem {
    /// Plain column reference.
    Col(String),
    /// L2_DISTANCE(<col>, '<vec>') AS <alias>  (alias defaults to "distance"
    /// only when explicitly written; otherwise the output name is the
    /// canonical expression text).
    L2 { col: String, qvec: Vec<f32>, alias: Option<String> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderDir {
    Asc,
    Desc,
}

#[derive(Debug, Clone)]
pub enum OrderKey {
    Col { name: String, dir: OrderDir },
    /// `ORDER BY L2_DISTANCE(col, 'vec')` - kept even when the projection
    /// does not include the distance, so it can be computed hidden.
    L2 { col: String, qvec: Vec<f32>, dir: OrderDir },
}

#[derive(Debug, Clone)]
pub struct SelectStmt {
    /// None = `SELECT *`
    pub items: Option<Vec<ProjItem>>,
    pub table: String,
    pub filter: Option<Expr>,
    pub order: Vec<OrderKey>,
    pub limit: Option<u64>,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    CreateTable {
        name: String,
        columns: Vec<(String, crate::value::ColKind, bool)>,
    },
    CreateVectorIndex {
        index: String,
        table: String,
        column: String,
    },
    Insert {
        table: String,
        cols: Option<Vec<String>>,
        rows: Vec<Vec<Literal>>,
    },
    Select(SelectStmt),
    Update {
        table: String,
        sets: Vec<(String, Literal)>,
        filter: Option<Expr>,
    },
    Delete {
        table: String,
        filter: Option<Expr>,
    },
    Begin,
    Commit,
    Rollback,
}

/// Statement head words that must be rejected as `unsupported_sql`
/// (recognized-but-out-of-subset), not as syntax errors.
const UNSUPPORTED_HEADS: &[&str] = &[
    "ALTER", "DROP", "JOIN", "TRUNCATE", "RENAME", "GRANT", "REVOKE", "SET",
    "EXPLAIN", "DESCRIBE", "SHOW", "USE", "ANALYZE", "OPTIMIZE", "REPLACE",
    "MERGE", "UNION", "VALUES", "WITH", "PRAGMA",
];

pub fn parse(sql: &str) -> DbResult<Stmt> {
    let mut lx = Lexer::tokenize(sql)?;
    if lx.at_end() {
        return Err(DbError::SyntaxError);
    }
    // Detect unsupported statement heads for the dedicated error code.
    if let Some(Tok::Word(w)) = lx.peek() {
        let upper = w.to_ascii_uppercase();
        if UNSUPPORTED_HEADS.contains(&upper.as_str()) {
            return Err(DbError::UnsupportedSql);
        }
    }
    let stmt = parse_stmt(&mut lx)?;
    lx.expect_done()?;
    Ok(stmt)
}

fn parse_stmt(lx: &mut Lexer) -> DbResult<Stmt> {
    if lx.eat_kw("CREATE") {
        if lx.eat_kw("TABLE") {
            return parse_create_table(lx);
        }
        if lx.eat_kw("VECTOR") {
            lx.expect_kw("INDEX")?;
            return parse_create_vector_index(lx);
        }
        if lx.eat_kw("INDEX") {
            // CREATE INDEX ... ON t(col) treated same as vector index
            return parse_create_vector_index(lx);
        }
        if lx.eat_kw("UNIQUE") || lx.eat_kw("SPATIAL") || lx.eat_kw("FULLTEXT") {
            return Err(DbError::UnsupportedSql);
        }
        return Err(DbError::UnsupportedSql);
    }
    if lx.eat_kw("INSERT") {
        return parse_insert(lx);
    }
    if lx.eat_kw("SELECT") {
        return parse_select(lx);
    }
    if lx.eat_kw("UPDATE") {
        return parse_update(lx);
    }
    if lx.eat_kw("DELETE") {
        lx.expect_kw("FROM")?;
        let table = lx.ident()?;
        let filter = if lx.eat_kw("WHERE") { Some(parse_expr(lx)?) } else { None };
        return Ok(Stmt::Delete { table, filter });
    }
    if lx.eat_kw("BEGIN") {
        // accept BEGIN / BEGIN WORK / BEGIN TRANSACTION
        lx.eat_kw("WORK");
        lx.eat_kw("TRANSACTION");
        return Ok(Stmt::Begin);
    }
    if lx.eat_kw("START") {
        lx.expect_kw("TRANSACTION")?;
        return Ok(Stmt::Begin);
    }
    if lx.eat_kw("COMMIT") {
        lx.eat_kw("WORK");
        return Ok(Stmt::Commit);
    }
    if lx.eat_kw("ROLLBACK") {
        lx.eat_kw("WORK");
        return Ok(Stmt::Rollback);
    }
    if lx.eat_kw("ABORT") {
        return Ok(Stmt::Rollback);
    }
    Err(DbError::SyntaxError)
}

fn parse_create_table(lx: &mut Lexer) -> DbResult<Stmt> {
    // optional IF NOT EXISTS -> unsupported (keeps subset strict)
    if lx.is_kw("IF") {
        return Err(DbError::UnsupportedSql);
    }
    let name = lx.ident()?;
    lx.expect(Tok::LParen)?;
    let mut columns: Vec<(String, crate::value::ColKind, bool)> = Vec::new();
    loop {
        let col_name = lx.ident()?;
        let kind = parse_col_kind(lx)?;
        // PRIMARY KEY may appear inline after the type, or as a separate
        // `PRIMARY KEY (col)` table constraint.
        let mut is_pk = false;
        if lx.is_kw("PRIMARY") {
            lx.expect_kw("PRIMARY")?;
            lx.expect_kw("KEY")?;
            is_pk = true;
        }
        columns.push((col_name, kind, is_pk));
        if lx.eat_kw("NOT") {
            lx.expect_kw("NULL")?;
        } else if lx.eat_kw("NULL") {
            // NULLs not supported; declaring NULL is a no-op default
        }
        if lx.eat_kw("AUTO_INCREMENT") || lx.eat_kw("AUTOINCREMENT") {
            return Err(DbError::UnsupportedSql);
        }
        if lx.peek() == Some(&Tok::Comma) {
            lx.next();
            // possible table-level constraint follows
            if lx.is_kw("PRIMARY") {
                lx.expect_kw("PRIMARY")?;
                lx.expect_kw("KEY")?;
                lx.expect(Tok::LParen)?;
                let pkcol = lx.ident()?;
                lx.expect(Tok::RParen)?;
                let mut found = false;
                for c in columns.iter_mut() {
                    if c.0.eq_ignore_ascii_case(&pkcol) {
                        if c.2 {
                            return Err(DbError::SyntaxError);
                        }
                        c.2 = true;
                        found = true;
                    }
                }
                if !found {
                    return Err(DbError::SyntaxError);
                }
                if lx.peek() == Some(&Tok::Comma) {
                    lx.next();
                    continue;
                }
                break;
            }
            if lx.is_kw("FOREIGN") || lx.is_kw("UNIQUE") || lx.is_kw("KEY") || lx.is_kw("CONSTRAINT") || lx.is_kw("CHECK") || lx.is_kw("INDEX") {
                return Err(DbError::UnsupportedSql);
            }
            continue;
        }
        break;
    }
    lx.expect(Tok::RParen)?;
    Ok(Stmt::CreateTable { name, columns })
}

fn parse_col_kind(lx: &mut Lexer) -> DbResult<crate::value::ColKind> {
    if lx.eat_kw("BIGINT") {
        return Ok(crate::value::ColKind::BigInt);
    }
    if lx.eat_kw("INT") || lx.eat_kw("INTEGER") {
        return Err(DbError::UnsupportedSql);
    }
    if lx.eat_kw("VARCHAR") {
        lx.expect(Tok::LParen)?;
        let n = expect_uint(lx)?;
        lx.expect(Tok::RParen)?;
        return Ok(crate::value::ColKind::Varchar(n as u32));
    }
    if lx.eat_kw("VECTOR") {
        lx.expect(Tok::LParen)?;
        let n = expect_uint(lx)?;
        lx.expect(Tok::RParen)?;
        return Ok(crate::value::ColKind::Vector(n as u32));
    }
    if lx.eat_kw("CHAR") || lx.eat_kw("TEXT") || lx.eat_kw("FLOAT") || lx.eat_kw("DOUBLE")
        || lx.eat_kw("REAL") || lx.eat_kw("BOOLEAN") || lx.eat_kw("BOOL") || lx.eat_kw("DECIMAL") || lx.eat_kw("BLOB")
    {
        return Err(DbError::UnsupportedSql);
    }
    Err(DbError::SyntaxError)
}

fn expect_uint(lx: &mut Lexer) -> DbResult<u64> {
    match lx.next() {
        Some(Tok::Int(t)) => t.parse::<u64>().map_err(|_| DbError::ValueOverflow),
        _ => Err(DbError::SyntaxError),
    }
}

fn parse_create_vector_index(lx: &mut Lexer) -> DbResult<Stmt> {
    if lx.is_kw("IF") {
        return Err(DbError::UnsupportedSql);
    }
    let index = lx.ident()?;
    lx.expect_kw("ON")?;
    let table = lx.ident()?;
    lx.expect(Tok::LParen)?;
    let column = lx.ident()?;
    lx.expect(Tok::RParen)?;
    Ok(Stmt::CreateVectorIndex { index, table, column })
}

fn parse_insert(lx: &mut Lexer) -> DbResult<Stmt> {
    lx.expect_kw("INTO")?;
    let table = lx.ident()?;
    let mut cols: Option<Vec<String>> = None;
    if lx.peek() == Some(&Tok::LParen) {
        lx.next();
        let mut cs = Vec::new();
        loop {
            cs.push(lx.ident()?);
            if lx.peek() == Some(&Tok::Comma) {
                lx.next();
            } else {
                break;
            }
        }
        lx.expect(Tok::RParen)?;
        cols = Some(cs);
    }
    lx.expect_kw("VALUES")?;
    let mut rows = Vec::new();
    loop {
        lx.expect(Tok::LParen)?;
        let mut vals = Vec::new();
        loop {
            vals.push(parse_value_literal(lx)?);
            if lx.peek() == Some(&Tok::Comma) {
                lx.next();
            } else {
                break;
            }
        }
        lx.expect(Tok::RParen)?;
        rows.push(vals);
        if lx.peek() == Some(&Tok::Comma) {
            lx.next();
        } else {
            break;
        }
    }
    Ok(Stmt::Insert { table, cols, rows })
}

/// Parse a scalar literal: integer or string (vectors arrive as strings and
/// are converted at bind time when the target column is known).
fn parse_value_literal(lx: &mut Lexer) -> DbResult<Literal> {
    match lx.next() {
        Some(Tok::Int(t)) => Ok(Literal::Int(parse_bigint(&t)?)),
        Some(Tok::Minus) => match lx.next() {
            Some(Tok::Int(t)) => {
                // Parse the sign and digits together so i64::MIN is accepted.
                let signed = format!("-{}", t);
                let value = signed.parse::<i64>().map_err(|_| DbError::ValueOverflow)?;
                Ok(Literal::Int(value))
            }
            Some(Tok::Float(t)) => {
                // only reachable inside vector text in practice; reject here
                let _ = t;
                Err(DbError::InvalidVector)
            }
            _ => Err(DbError::SyntaxError),
        },
        Some(Tok::Str(s)) => Ok(Literal::Str(s)),
        Some(Tok::Float(_)) => Err(DbError::TypeMismatch), // floats invalid as scalars
        _ => Err(DbError::SyntaxError),
    }
}

fn parse_update(lx: &mut Lexer) -> DbResult<Stmt> {
    let table = lx.ident()?;
    lx.expect_kw("SET")?;
    let mut sets = Vec::new();
    loop {
        let column = lx.ident()?;
        lx.expect(Tok::Eq)?;
        let value = parse_value_literal(lx)?;
        sets.push((column, value));
        if lx.peek() == Some(&Tok::Comma) {
            lx.next();
        } else {
            break;
        }
    }
    let filter = if lx.eat_kw("WHERE") { Some(parse_expr(lx)?) } else { None };
    Ok(Stmt::Update { table, sets, filter })
}

fn parse_select(lx: &mut Lexer) -> DbResult<Stmt> {
    // No DISTINCT / aggregates support
    if lx.is_kw("DISTINCT") {
        return Err(DbError::UnsupportedSql);
    }
    let mut items: Option<Vec<ProjItem>> = Some(Vec::new());
    if lx.peek() == Some(&Tok::Star) {
        lx.next();
        items = None;
    } else {
        loop {
            let item = parse_proj_item(lx)?;
            items.as_mut().unwrap().push(item);
            if lx.peek() == Some(&Tok::Comma) {
                lx.next();
            } else {
                break;
            }
        }
    }
    lx.expect_kw("FROM")?;
    let table = lx.ident()?;
    let filter = if lx.eat_kw("WHERE") { Some(parse_expr(lx)?) } else { None };
    let mut order = Vec::new();
    if lx.eat_kw("ORDER") {
        lx.expect_kw("BY")?;
        loop {
            let key = parse_order_key(lx)?;
            order.push(key);
            if lx.peek() == Some(&Tok::Comma) {
                lx.next();
            } else {
                break;
            }
        }
    }
    let mut limit = None;
    if lx.eat_kw("LIMIT") {
        limit = Some(expect_uint(lx)?);
        // LIMIT n OFFSET m not in the subset
        if lx.is_kw("OFFSET") {
            return Err(DbError::UnsupportedSql);
        }
    }
    Ok(Stmt::Select(SelectStmt { items, table, filter, order, limit }))
}

fn parse_proj_item(lx: &mut Lexer) -> DbResult<ProjItem> {
    // L2_DISTANCE ( col , '<vector>' ) [AS alias]
    if lx.is_kw("L2_DISTANCE") {
        lx.next();
        lx.expect(Tok::LParen)?;
        let col = lx.ident()?;
        lx.expect(Tok::Comma)?;
        let qtext = match lx.next() {
            Some(Tok::Str(s)) => s,
            _ => return Err(DbError::SyntaxError),
        };
        // strip optional brackets
        let inner = qtext.trim();
        let inner = inner.strip_prefix('[').and_then(|s| s.strip_suffix(']')).unwrap_or(inner);
        let qvec = parse_vector(inner, None)?;
        lx.expect(Tok::RParen)?;
        let alias = if lx.eat_kw("AS") {
            Some(lx.ident()?)
        } else {
            None
        };
        return Ok(ProjItem::L2 { col, qvec, alias });
    }
    let name = lx.ident()?;
    Ok(ProjItem::Col(name))
}

fn parse_order_key(lx: &mut Lexer) -> DbResult<OrderKey> {
    if lx.is_kw("L2_DISTANCE") {
        lx.next();
        lx.expect(Tok::LParen)?;
        let col = lx.ident()?;
        lx.expect(Tok::Comma)?;
        let qtext = match lx.next() {
            Some(Tok::Str(s)) => s,
            _ => return Err(DbError::SyntaxError),
        };
        let inner = qtext.trim();
        let inner = inner
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .ok_or(DbError::InvalidVector)?;
        let qvec = parse_vector(inner, None)?;
        lx.expect(Tok::RParen)?;
        let dir = if lx.eat_kw("DESC") {
            OrderDir::Desc
        } else {
            lx.eat_kw("ASC");
            OrderDir::Asc
        };
        return Ok(OrderKey::L2 { col, qvec, dir });
    }
    let name = lx.ident()?;
    let dir = if lx.eat_kw("DESC") {
        OrderDir::Desc
    } else {
        lx.eat_kw("ASC");
        OrderDir::Asc
    };
    // resolution of the name (column vs L2 alias) happens at bind time
    Ok(OrderKey::Col { name, dir })
}

fn parse_expr(lx: &mut Lexer) -> DbResult<Expr> {
    parse_and_expr(lx)
}

fn parse_and_expr(lx: &mut Lexer) -> DbResult<Expr> {
    let mut left = parse_or_expr(lx)?;
    while lx.eat_kw("AND") {
        let right = parse_or_expr(lx)?;
        left = Expr::And(Box::new(left), Box::new(right));
    }
    Ok(left)
}

fn parse_or_expr(lx: &mut Lexer) -> DbResult<Expr> {
    let mut left = parse_predicate(lx)?;
    while lx.eat_kw("OR") {
        let right = parse_predicate(lx)?;
        left = Expr::Or(Box::new(left), Box::new(right));
    }
    Ok(left)
}

fn parse_predicate(lx: &mut Lexer) -> DbResult<Expr> {
    if lx.peek() == Some(&Tok::LParen) {
        lx.next();
        let e = parse_expr(lx)?;
        lx.expect(Tok::RParen)?;
        return Ok(e);
    }
    let col = lx.ident()?;
    let op = match lx.next() {
        Some(Tok::Eq) => CmpOp::Eq,
        Some(Tok::Ne) => CmpOp::Ne,
        Some(Tok::Lt) => CmpOp::Lt,
        Some(Tok::Le) => CmpOp::Le,
        Some(Tok::Gt) => CmpOp::Gt,
        Some(Tok::Ge) => CmpOp::Ge,
        _ => return Err(DbError::SyntaxError),
    };
    let lit = parse_value_literal(lx)?;
    Ok(Expr::Cmp { col, op, lit })
}
