//! Schema, values and float32 value rules.

use crate::error::{DbError, DbResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColKind {
    BigInt,
    Varchar(u32),
    Vector(u32),
}

#[derive(Debug, Clone)]
pub struct Column {
    pub name: String,
    pub kind: ColKind,
    pub primary_key: bool,
}

#[derive(Debug, Clone)]
pub struct Schema {
    pub name: String,
    pub id: u32,
    pub columns: Vec<Column>,
}

impl Schema {
    pub fn pk_index(&self) -> usize {
        self.columns.iter().position(|c| c.primary_key).expect("schema has pk")
    }
    pub fn col_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name.eq_ignore_ascii_case(name))
    }
    pub fn vector_col(&self) -> Option<usize> {
        self.columns.iter().position(|c| matches!(c.kind, ColKind::Vector(_)))
    }
}

/// Format an f64 distance with six decimal places (far tighter than the
/// driver's absolute/relative 1e-5 comparison tolerance).
pub fn fmt_f64(f: f64) -> String {
    if !f.is_finite() {
        return "null".to_string();
    }
    let mut s = format!("{:.6}", f);
    while s.ends_with('0') {
        s.pop();
    }
    if s.ends_with('.') {
        s.push('0');
    }
    if s == "-0.0" {
        s = "0.0".to_string();
    }
    s
}

/// Format an f32 for result output: shortest form that round-trips within
/// the driver's 1e-5 tolerance. We print with up to 6 decimals, trimming
/// useless trailing zeros but always keeping a decimal point.
pub fn fmt_f32(f: f32) -> String {
    if !f.is_finite() {
        return "null".to_string(); // unreachable: inputs reject non-finite
    }
    let mut s = format!("{:.6}", f);
    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.push('0'); // keep "5.0" style
        }
    }
    // avoid "-0.0"
    if s == "-0.0" {
        s = "0.0".to_string();
    }
    s
}

/// Parse the text of a vector literal (the inner part, e.g. `1, 2, 3`)
/// applying the float32 input rules:
/// - every component must parse as a finite f64,
/// - conversion to f32 must stay finite (reject overflow),
/// - NaN and Infinity spellings are rejected.
pub fn parse_vector(text: &str, expect_dim: Option<u32>) -> DbResult<Vec<f32>> {
    let mut out = Vec::new();
    if !text.trim().is_empty() {
        for part in text.split(',') {
            let p = part.trim();
            if p.is_empty() {
                return Err(DbError::InvalidVector);
            }
            let lower = p.to_ascii_lowercase();
            if lower.contains("nan") || lower.contains("inf") {
                return Err(DbError::InvalidVector);
            }
            let d: f64 = p.parse().map_err(|_| DbError::InvalidVector)?;
            if !d.is_finite() {
                return Err(DbError::InvalidVector);
            }
            let f = d as f32;
            if !f.is_finite() {
                return Err(DbError::InvalidVector);
            }
            out.push(f);
        }
    }
    if let Some(dim) = expect_dim {
        if out.len() as u32 != dim {
            return Err(DbError::DimensionMismatch);
        }
    }
    Ok(out)
}

/// Count Unicode characters (not bytes) for VARCHAR(n) length checks.
pub fn char_len(s: &str) -> u32 {
    s.chars().count() as u32
}

pub fn check_varchar(s: &str, max: u32) -> DbResult<()> {
    if char_len(s) > max {
        Err(DbError::StringTooLong)
    } else {
        Ok(())
    }
}

pub fn parse_bigint(text: &str) -> DbResult<i64> {
    let t = text.trim();
    if let Ok(v) = t.parse::<i64>() {
        return Ok(v);
    }
    Err(DbError::ValueOverflow)
}

/// Euclidean distance computed over stored f32 components with an f64
/// accumulator (output tolerance is 1e-5, this is far tighter).
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0f64;
    for i in 0..a.len() {
        let d = a[i] as f64 - b[i] as f64;
        acc += d * d;
    }
    acc.sqrt()
}

/// Faster ANN-mode distance: float32 accumulation over stored float32
/// components. Returned distances remain well inside the 1e-5 tolerance at
/// the published dimensions; exact mode uses the f64 accumulator above.
pub fn l2_distance_ann(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    (acc as f64).sqrt()
}
