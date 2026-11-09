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
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { dist_sq_avx2_f64(a, b).sqrt() };
        }
    }
    let mut acc = 0f64;
    for i in 0..a.len() {
        let d = a[i] as f64 - b[i] as f64;
        acc += d * d;
    }
    acc.sqrt()
}

/// f64-accumulated squared distance via 4-lane AVX2. Numerically each lane
/// accumulates a disjoint subset of components in program order, so results
/// match the scalar loop exactly for the 8-aligned body.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dist_sq_avx2_f64(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::{
        __m256d, _mm256_add_pd, _mm256_castps256_ps128, _mm256_cvtps_pd, _mm256_cvtsd_f64,
        _mm256_extractf128_ps, _mm256_fmadd_pd, _mm256_loadu_ps, _mm256_permute2f128_pd,
        _mm256_setzero_pd, _mm256_sub_pd, _mm256_unpackhi_pd,
    };
    let mut acc: __m256d = _mm256_setzero_pd();
    let mut i = 0usize;
    while i + 8 <= a.len() {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let lo = _mm256_sub_pd(
            _mm256_cvtps_pd(_mm256_castps256_ps128(va)),
            _mm256_cvtps_pd(_mm256_castps256_ps128(vb)),
        );
        let hi = _mm256_sub_pd(
            _mm256_cvtps_pd(_mm256_extractf128_ps(va, 1)),
            _mm256_cvtps_pd(_mm256_extractf128_ps(vb, 1)),
        );
        acc = _mm256_fmadd_pd(lo, lo, acc);
        acc = _mm256_fmadd_pd(hi, hi, acc);
        i += 8;
    }
    // Horizontal sum: pairwise then linear, all f64.
    let mut hi = _mm256_unpackhi_pd(acc, acc);
    let mut sum_v = _mm256_add_pd(acc, hi);
    hi = _mm256_permute2f128_pd(sum_v, sum_v, 1);
    sum_v = _mm256_add_pd(sum_v, hi);
    let mut sum = _mm256_cvtsd_f64(sum_v);
    while i < a.len() {
        let d = a[i] as f64 - b[i] as f64;
        sum += d * d;
        i += 1;
    }
    sum
}

/// Scalar fallback with identical semantics.
#[cfg(not(target_arch = "x86_64"))]
fn dist_sq_avx2_f64(a: &[f32], b: &[f32]) -> f64 {
    l2_distance_sq(a, b)
}

/// Squared distance with early abandon: returns None as soon as the partial
/// sum is strictly greater than `limit` (the row then cannot enter a TopK
/// whose current worst squared distance is `limit`; exact ties are kept
/// because only strict excess aborts). AVX2 path checks every 16 components.
pub fn dist_sq_f64_abandon(a: &[f32], b: &[f32], limit: f64) -> Option<f64> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { dist_sq_abandon_avx2(a, b, limit) };
        }
    }
    let mut acc = 0f64;
    for i in 0..a.len() {
        let d = a[i] as f64 - b[i] as f64;
        acc += d * d;
        if acc > limit {
            return None;
        }
    }
    Some(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dist_sq_abandon_avx2(a: &[f32], b: &[f32], limit: f64) -> Option<f64> {
    use std::arch::x86_64::{
        __m256d, _mm256_add_pd, _mm256_castps256_ps128, _mm256_cvtps_pd, _mm256_cvtsd_f64,
        _mm256_extractf128_ps, _mm256_fmadd_pd, _mm256_loadu_ps, _mm256_permute2f128_pd,
        _mm256_setzero_pd, _mm256_sub_pd, _mm256_unpackhi_pd,
    };
    let hsum4 = |v0: __m256d, v1: __m256d, v2: __m256d, v3: __m256d| -> f64 {
        let s = _mm256_add_pd(_mm256_add_pd(v0, v1), _mm256_add_pd(v2, v3));
        let hi = _mm256_unpackhi_pd(s, s);
        let t = _mm256_add_pd(s, hi);
        _mm256_cvtsd_f64(_mm256_add_pd(t, _mm256_permute2f128_pd(t, t, 1)))
    };
    // Four independent accumulators break the FMA latency chain; the
    // threshold check runs once per 32 components. Accumulation grouping is
    // irrelevant to value: f64 adds of a fixed disjoint partition are
    // independent per accumulator.
    let (mut acc0, mut acc1, mut acc2, mut acc3) = (
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
    );
    let mut i = 0usize;
    while i + 32 <= a.len() {
        for block in 0..4 {
            let va = _mm256_loadu_ps(a.as_ptr().add(i + block * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i + block * 8));
            let lo = _mm256_sub_pd(
                _mm256_cvtps_pd(_mm256_castps256_ps128(va)),
                _mm256_cvtps_pd(_mm256_castps256_ps128(vb)),
            );
            let hi = _mm256_sub_pd(
                _mm256_cvtps_pd(_mm256_extractf128_ps(va, 1)),
                _mm256_cvtps_pd(_mm256_extractf128_ps(vb, 1)),
            );
            // products into per-block temporaries, then distributed adds:
            // each accumulator sees one fma-free add per block, keeping the
            // four chains disjoint.
            let p_lo = _mm256_fmadd_pd(lo, lo, _mm256_setzero_pd());
            let p_hi = _mm256_fmadd_pd(hi, hi, _mm256_setzero_pd());
            match block {
                0 => {
                    acc0 = _mm256_add_pd(acc0, p_lo);
                    acc1 = _mm256_add_pd(acc1, p_hi);
                }
                1 => {
                    acc2 = _mm256_add_pd(acc2, p_lo);
                    acc3 = _mm256_add_pd(acc3, p_hi);
                }
                2 => {
                    acc0 = _mm256_add_pd(acc0, p_hi);
                    acc1 = _mm256_add_pd(acc1, p_lo);
                }
                _ => {
                    acc2 = _mm256_add_pd(acc2, p_hi);
                    acc3 = _mm256_add_pd(acc3, p_lo);
                }
            }
        }
        if hsum4(acc0, acc1, acc2, acc3) > limit {
            return None;
        }
        i += 32;
    }
    let mut sum = hsum4(acc0, acc1, acc2, acc3);
    while i < a.len() {
        let d = a[i] as f64 - b[i] as f64;
        sum += d * d;
        if sum > limit {
            return None;
        }
        i += 1;
    }
    Some(sum)
}

/// Squared distance with f64 accumulation (exact ranking / k-means).
pub fn l2_distance_sq(a: &[f32], b: &[f32]) -> f64 {
    let mut acc = 0f64;
    for i in 0..a.len() {
        let d = a[i] as f64 - b[i] as f64;
        acc += d * d;
    }
    acc
}

/// f32-accumulated squared distance (scalar fallback).
pub fn l2_distance_sq_f32(a: &[f32], b: &[f32]) -> f64 {
    let mut acc = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc as f64
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dist_sq_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::{
        __m256, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps,
        _mm256_sub_ps,
    };
    let mut acc: __m256 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= a.len() {
        unsafe {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let d = _mm256_sub_ps(va, vb);
            acc = _mm256_fmadd_ps(d, d, acc);
        }
        i += 8;
    }
    let mut tmp = [0f32; 8];
    unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), acc) };
    let mut sum: f32 = tmp.iter().sum();
    while i < a.len() {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }
    sum
}

/// Fast squared distance for the ANN path: AVX2 when available, scalar f32
/// otherwise. f32 accumulation stays far inside the 1e-5 output tolerance.
pub fn l2_distance_sq_fast(a: &[f32], b: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { dist_sq_avx2(a, b) as f64 };
        }
    }
    l2_distance_sq_f32(a, b)
}
