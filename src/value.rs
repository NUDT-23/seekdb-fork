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
    // A component needs at least one input byte. Bounding the reservation by
    // the text size preserves the fast path without trusting an unbounded DDL
    // dimension (a hostile VECTOR(u32::MAX) must not request a 16 GiB Vec).
    let capacity = expect_dim.map_or(16, |dim| dim as usize).min(text.len().max(1));
    let mut out = Vec::with_capacity(capacity);
    if !text.trim().is_empty() {
        for part in text.split(',') {
            let p = part.trim();
            if p.is_empty() {
                return Err(DbError::InvalidVector);
            }
            // from_str accepts every supported NaN/Infinity spelling as a
            // non-finite value; malformed mixed spellings still fail parsing.
            // Checking the parsed value avoids one lowercase String per component.
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

/// TopM-only 128-dimensional f32 squared distance. Four independent FMA
/// chains shorten the dependency chain; the certified query bound permits
/// this different nonnegative accumulation grouping. ANN keeps using the
/// unchanged `l2_distance_sq_fast` implementation above.
pub fn l2_distance_sq_fast_topm_x4(a: &[f32], b: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        // Keep the same per-row feature dispatch as the accepted TopM kernel.
        if a.len() == 128
            && b.len() == 128
            && std::arch::is_x86_feature_detected!("avx2")
        {
            return unsafe { dist_sq_avx2_topm_x4(a, b) as f64 };
        }
    }
    l2_distance_sq_f32(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dist_sq_avx2_topm_x4(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps,
        _mm256_sub_ps,
    };
    debug_assert_eq!(a.len(), 128);
    debug_assert_eq!(b.len(), 128);
    let (mut acc0, mut acc1, mut acc2, mut acc3) = (
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    );
    let mut i = 0usize;
    while i < 128 {
        for block in 0..4 {
            let va = _mm256_loadu_ps(a.as_ptr().add(i + block * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i + block * 8));
            let d = _mm256_sub_ps(va, vb);
            match block {
                0 => acc0 = _mm256_fmadd_ps(d, d, acc0),
                1 => acc1 = _mm256_fmadd_ps(d, d, acc1),
                2 => acc2 = _mm256_fmadd_ps(d, d, acc2),
                _ => acc3 = _mm256_fmadd_ps(d, d, acc3),
            }
        }
        i += 32;
    }
    // Combine the four chains, then use the same single horizontal sum as the
    // accepted TopM kernel; each chain executed exactly four FMA operations.
    let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    tmp.iter().sum()
}

pub fn certified_query_i8_enabled() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// First pass of int8 cache construction: choose one scale for the whole
/// repaired snapshot so every excluded row has the same quantization bound.
pub fn i8_scale_topm(flat: &[f32]) -> Option<f64> {
    if flat.len() % 128 != 0 || !certified_query_i8_enabled() {
        return None;
    }
    let mut max_abs = 0f64;
    for &value in flat {
        let value = value as f64;
        if !value.is_finite() {
            return None;
        }
        max_abs = max_abs.max(value.abs());
    }
    if max_abs <= 0.0 {
        return None;
    }
    let scale = max_abs / 127.0;
    if scale.is_finite() && scale > 0.0 {
        Some(scale)
    } else {
        None
    }
}

/// Second pass: encode with the already fixed table scale. f64 division and
/// round keep the actual error within scale*(1/2+tiny); the certificate's
/// 1e-12 slack covers the division/scale rounding and tie behavior.
pub fn i8_encode_topm(vector: &[f32], scale: f64, out: &mut Vec<i8>) -> bool {
    if vector.len() != 128 || !scale.is_finite() || scale <= 0.0 {
        return false;
    }
    for &value in vector {
        let value = value as f64;
        if !value.is_finite() {
            return false;
        }
        let quantized = (value / scale).round();
        if !quantized.is_finite() || quantized.abs() > 127.0 {
            return false;
        }
        out.push(quantized as i8);
    }
    true
}

/// Exact integer squared distance on the quantized grid, scaled once in f64.
pub fn l2_distance_sq_topm_i8(a: &[i8], b: &[i8], scale_sq: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if a.len() == 128 && b.len() == 128 && std::arch::is_x86_feature_detected!("avx2") {
            let sum = unsafe { dist_sq_avx2_topm_i8(a, b) };
            return (sum as f64) * scale_sq;
        }
    }
    f64::NAN
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dist_sq_avx2_topm_i8(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::{
        __m128i, _mm256_add_epi32, _mm256_cvtepi8_epi16, _mm256_madd_epi16,
        _mm256_setzero_si256, _mm256_storeu_si256, _mm256_sub_epi16, _mm_loadu_si128,
    };
    debug_assert_eq!(a.len(), 128);
    debug_assert_eq!(b.len(), 128);
    let (mut acc0, mut acc1, mut acc2, mut acc3) = (
        _mm256_setzero_si256(),
        _mm256_setzero_si256(),
        _mm256_setzero_si256(),
        _mm256_setzero_si256(),
    );
    let mut i = 0usize;
    while i < 128 {
        for block in 0..4 {
            let offset = i + block * 16;
            let va = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                a.as_ptr().add(offset) as *const __m128i,
            ));
            let vb = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                b.as_ptr().add(offset) as *const __m128i,
            ));
            let d = _mm256_sub_epi16(va, vb);
            // Inputs are [-127,127], so d is [-255,255]. Each madd pair and
            // the final 128-dimensional sum (<=8,323,200) fit signed i32.
            let squared = _mm256_madd_epi16(d, d);
            match block {
                0 => acc0 = _mm256_add_epi32(acc0, squared),
                1 => acc1 = _mm256_add_epi32(acc1, squared),
                2 => acc2 = _mm256_add_epi32(acc2, squared),
                _ => acc3 = _mm256_add_epi32(acc3, squared),
            }
        }
        i += 64;
    }
    let acc = _mm256_add_epi32(_mm256_add_epi32(acc0, acc1), _mm256_add_epi32(acc2, acc3));
    let mut lanes = [0i32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr().cast(), acc);
    lanes.iter().sum()
}

/// The same table-scale error proof as int8, with a finer [-8191,8191] grid.
pub fn i14_encode_topm(vector: &[f32], scale: f64, out: &mut Vec<i16>) -> bool {
    if vector.len() != 128 || !scale.is_finite() || scale <= 0.0 { return false; }
    for &value in vector {
        let quantized = ((value as f64) / scale).round();
        if !quantized.is_finite() || quantized.abs() > 8191.0 { return false; }
        out.push(quantized as i16);
    }
    true
}

/// Exact squared grid distance. Both inputs must be encoded in [-8191,8191].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dist_sq_avx2_topm_i14(a: &[i16], b: &[i16]) -> u64 {
    use std::arch::x86_64::{
        _mm256_add_epi32, _mm256_loadu_si256, _mm256_madd_epi16,
        _mm256_setzero_si256, _mm256_storeu_si256, _mm256_sub_epi16,
    };
    debug_assert_eq!(a.len(), 128);
    debug_assert_eq!(b.len(), 128);
    debug_assert!(a.iter().chain(b).all(|v| (-8191..=8191).contains(v)));
    let (mut acc0, mut acc1, mut acc2, mut acc3) = (
        _mm256_setzero_si256(), _mm256_setzero_si256(),
        _mm256_setzero_si256(), _mm256_setzero_si256(),
    );
    for i in (0..128).step_by(64) {
        for block in 0..4 {
            let offset = i + block * 16;
            let va = _mm256_loadu_si256(a.as_ptr().add(offset).cast());
            let vb = _mm256_loadu_si256(b.as_ptr().add(offset).cast());
            let d = _mm256_sub_epi16(va, vb);
            // |d|<=16382; each pair<=536739848<i32::MAX. Each final
            // lane sums 16 squares<=4293918784<u32::MAX. add_epi32 is
            // modular, so reinterpret its final lanes as unsigned, not i32.
            let squared = _mm256_madd_epi16(d, d);
            match block {
                0 => acc0 = _mm256_add_epi32(acc0, squared),
                1 => acc1 = _mm256_add_epi32(acc1, squared),
                2 => acc2 = _mm256_add_epi32(acc2, squared),
                _ => acc3 = _mm256_add_epi32(acc3, squared),
            }
        }
    }
    let acc = _mm256_add_epi32(_mm256_add_epi32(acc0, acc1), _mm256_add_epi32(acc2, acc3));
    let mut lanes = [0u32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr().cast(), acc);
    lanes.iter().map(|&v| v as u64).sum()
}

/// At8191, f64 division rounding is below2^-40 grid units per operand.
/// Budget more than twice that in the Minkowski error, independently of
/// the separate f32/f64-kernel slack. Only the error scale is inflated;
/// the caller's squared grid distance retains its actual table scale.
pub fn certified_query_i14_lower_bound(s: f64, scale: f64) -> f64 {
    certified_query_i8_lower_bound(s, scale * (1.0 + 4e-12))
}

/// Conservative original-distance lower bound from the exact integer grid.
pub fn certified_query_i8_lower_bound(s: f64, scale: f64) -> f64 {
    if !s.is_finite() || s < 0.0 || !scale.is_finite() || scale <= 0.0 {
        return f64::NAN;
    }
    // Cover scale^2 and the final f64 multiplication. This is intentionally
    // looser than two-ulp error and does not assume any f32 accumulation.
    let grid_lower = next_down_f64(s * (1.0 - 1.0 / 1_125_899_906_842_624.0));
    // Both the row and query have per-component error <=scale/2+tiny.
    // Minkowski therefore gives a single global L2 error sqrt(128)*scale.
    let error_upper = next_up_f64(128f64.sqrt() * scale * (1.0 + 1e-12));
    if !grid_lower.is_finite() || grid_lower <= 0.0 || !error_upper.is_finite() {
        return f64::NAN;
    }
    let grid_root = next_down_f64(grid_lower.sqrt());
    let original_root = next_down_f64(grid_root - error_upper);
    if !original_root.is_finite() || original_root <= 0.0 {
        return f64::NAN;
    }
    let original_lower = next_down_f64(original_root * original_root);
    certified_query_lower_bound(original_lower)
}

/// F16C compression is optional; the original f32 flat cache remains the
/// source for exact rescoring and for every fallback path.
pub fn certified_query_f16_enabled() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        certified_query_f32_enabled() && std::arch::is_x86_feature_detected!("f16c")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Append the RNE f16 encoding and return a conservative upper bound for the
/// original f32 vector's Euclidean norm. Rejecting any component above the
/// largest finite f16 avoids per-query overflow surprises.
pub fn f16_encode_topm(vector: &[f32], out: &mut Vec<u16>) -> Option<f64> {
    if vector.len() != 128 || !certified_query_f16_enabled() {
        return None;
    }
    let mut norm_sq = 0f64;
    for &value in vector {
        let value = value as f64;
        if !value.is_finite() || value.abs() > 65504.0 {
            return None;
        }
        norm_sq += value * value;
    }
    // The 128-term f64 sum and sqrt have many orders of magnitude more
    // precision than this slack; nextUp makes the upper-bound intent explicit.
    let norm_upper = next_up_f64(norm_sq.sqrt() * (1.0 + 1e-12));
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{
            __m128i, _mm256_cvtps_ph, _mm256_loadu_ps, _mm_storeu_si128,
            _MM_FROUND_TO_NEAREST_INT,
        };
        let mut encoded = [0u16; 8];
        for chunk in vector.chunks_exact(8) {
            let values = _mm256_loadu_ps(chunk.as_ptr());
            let half = _mm256_cvtps_ph(values, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128(encoded.as_mut_ptr() as *mut __m128i, half);
            out.extend_from_slice(&encoded);
        }
    }
    Some(norm_upper)
}

/// Decode a query's f16 grid back to f32 once per query. Row decoding stays in
/// the distance kernel; the query side can then use ordinary aligned-width
/// f32 loads without repeating sixteen conversion instructions per row.
pub fn f16_decode_topm(encoded: &[u16], out: &mut Vec<f32>) -> bool {
    if encoded.len() != 128 || !certified_query_f16_enabled() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{
            __m128i, _mm256_cvtph_ps, _mm256_storeu_ps, _mm_loadu_si128,
        };
        for chunk in encoded.chunks_exact(8) {
            let half = _mm_loadu_si128(chunk.as_ptr() as *const __m128i);
            let values = _mm256_cvtph_ps(half);
            let start = out.len();
            out.resize(start + 8, 0.0);
            _mm256_storeu_ps(out.as_mut_ptr().add(start), values);
        }
    }
    true
}

/// Complete-f32 distance over a compressed row and an already decoded query.
/// NaN requests the f32 TopM fallback; it is never an exact query result.
pub fn l2_distance_sq_fast_topm_f16_x4(a: &[u16], b: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if a.len() == 128
            && b.len() == 128
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("f16c")
        {
            return unsafe { dist_sq_avx2_topm_f16_x4(a, b) as f64 };
        }
    }
    f64::NAN
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dist_sq_avx2_topm_f16_x4(a: &[u16], b: &[f32]) -> f32 {
    use std::arch::x86_64::{
        __m128i, _mm256_add_ps, _mm256_cvtph_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_setzero_ps, _mm256_storeu_ps, _mm256_sub_ps, _mm_loadu_si128,
    };
    debug_assert_eq!(a.len(), 128);
    debug_assert_eq!(b.len(), 128);
    let (mut acc0, mut acc1, mut acc2, mut acc3) = (
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    );
    let mut i = 0usize;
    while i < 128 {
        for block in 0..4 {
            let half = _mm_loadu_si128(a.as_ptr().add(i + block * 8) as *const __m128i);
            let va = _mm256_cvtph_ps(half);
            let vb = _mm256_loadu_ps(b.as_ptr().add(i + block * 8));
            let d = _mm256_sub_ps(va, vb);
            match block {
                0 => acc0 = _mm256_fmadd_ps(d, d, acc0),
                1 => acc1 = _mm256_fmadd_ps(d, d, acc1),
                2 => acc2 = _mm256_fmadd_ps(d, d, acc2),
                _ => acc3 = _mm256_fmadd_ps(d, d, acc3),
            }
        }
        i += 32;
    }
    let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    tmp.iter().sum()
}

/// Lower-bound the original f64 distance from a compressed candidate pass.
/// L0 first covers f32 accumulation on the f16 grid. RNE normal-component
/// error is at most 2^-11 |x_i|; f16 subnormals add at most 2^-25 per
/// component. Minkowski and the reverse triangle inequality then bound every
/// original row, including rows excluded from the candidate heap.
pub fn certified_query_f16_lower_bound(
    s: f64,
    max_row_norm_upper: f64,
    query_norm_upper: f64,
) -> f64 {
    let grid_lower = certified_query_lower_bound(s);
    if !grid_lower.is_finite()
        || grid_lower < 0.0
        || !max_row_norm_upper.is_finite()
        || !query_norm_upper.is_finite()
        || max_row_norm_upper < 0.0
        || query_norm_upper < 0.0
    {
        return f64::NAN;
    }
    let quantization = (1.0f64 / 2048.0).mul_add(
        max_row_norm_upper + query_norm_upper,
        2.0 * 128f64.sqrt() * (1.0 / 33_554_432.0),
    );
    let error_upper = next_up_f64(quantization * (1.0 + 1e-12));
    let grid_root = next_down_f64(grid_lower.sqrt());
    let original_root = next_down_f64(grid_root - error_upper);
    if !original_root.is_finite() || original_root <= 0.0 {
        return f64::NAN;
    }
    let original_lower = next_down_f64(original_root * original_root);
    // Reuse the much looser f64 factor from the f32 certificate to cover the
    // original kernel's accumulation error and primary-key equality margin.
    certified_query_lower_bound(original_lower)
}

/// Whether the query-level f32 candidate pass has the arithmetic environment
/// required by its certified error bound.
pub fn certified_query_f32_enabled() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        return std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && unsafe { std::arch::x86_64::_mm_getcsr() } & 0xe040 == 0;
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Rust 1.85-compatible IEEE-754 nextDown.
#[inline]
fn next_down_f64(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return x;
    }
    if x == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = x.to_bits();
    if x > 0.0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

/// Rust 1.85-compatible IEEE-754 nextUp.
#[inline]
fn next_up_f64(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(1);
    }
    let bits = x.to_bits();
    if x > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

/// Conservative lower bound for both the real 128-dimensional squared
/// distance and the existing f64 kernel, from its complete f32 approximation.
#[inline]
pub fn certified_query_lower_bound(s: f64) -> f64 {
    next_down_f64(s.mul_add(0.99996, -3e-43))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i14_simd_matches_u64_at_unsigned_lane_boundary() {
        if !certified_query_i8_enabled() { return; }
        #[cfg(target_arch = "x86_64")]
        {
            let mut seed = 0x140014008191u64;
            for case in 0..1028 {
                let mut a = [0i16; 128];
                let mut b = [0i16; 128];
                for i in 0..128 {
                    seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                    a[i] = match case { 0 => -8191, 1 => 8191, 2 => if i%2==0 {8191} else {-8191}, _ => (seed % 16383) as i16 - 8191 };
                    seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                    b[i] = match case { 0 => 8191, 1 => -8191, 2 => -a[i], _ => (seed % 16383) as i16 - 8191 };
                }
                let expected: u64 = a.iter().zip(b).map(|(&x,y)| {
                    let d = x as i64 - y as i64; (d*d) as u64
                }).sum();
                assert_eq!(unsafe { dist_sq_avx2_topm_i14(&a, &b) }, expected);
                if case < 3 { assert_eq!(expected, 34351350272); }
            }
        }
    }

    #[test]
    fn i14_encoding_and_certificate_cover_f32_truth() {
        if !certified_query_i8_enabled() || !certified_query_f32_enabled() { return; }
        #[cfg(target_arch = "x86_64")]
        for maximum in [f32::from_bits(1), 1e-30, 1e-5, 1.0, 1000.0, f32::MAX] {
            let scale = maximum as f64 / 8191.0;
            for step in 0..32 {
                let a: Vec<f32> = (0..128).map(|i| maximum * (((i*37+step*71)%257) as f32 / 128.0 - 1.0)).collect();
                let b: Vec<f32> = (0..128).map(|i| maximum * (((i*19+step*47)%257) as f32 / 128.0 - 1.0)).collect();
                let mut ea = Vec::new(); let mut eb = Vec::new();
                assert!(i14_encode_topm(&a, scale, &mut ea));
                assert!(i14_encode_topm(&b, scale, &mut eb));
                for (raw, encoded) in a.iter().zip(&ea).chain(b.iter().zip(&eb)) {
                    assert!(((*raw as f64) - (*encoded as f64)*scale).abs() <= scale*(0.5+2e-12));
                }
                let s = unsafe { dist_sq_avx2_topm_i14(&ea, &eb) } as f64 * scale * scale;
                let lower = certified_query_i14_lower_bound(s, scale);
                let truth: f64 = a.iter().zip(&b).map(|(&x,&y)| { let d=x as f64-y as f64; d*d }).sum();
                assert!(lower.is_nan() || lower <= truth, "lower={lower} truth={truth}");
            }
        }
        for query in [[f32::NAN; 128], [f32::INFINITY; 128], [8192.0; 128]] {
            assert!(!i14_encode_topm(&query, 1.0, &mut Vec::new()));
        }
    }

    #[test]
    fn vector_reservation_is_bounded_by_input_size() {
        assert!(matches!(
            parse_vector("1,2", Some(u32::MAX)),
            Err(DbError::DimensionMismatch)
        ));
        assert!(matches!(
            parse_vector("", Some(u32::MAX)),
            Err(DbError::DimensionMismatch)
        ));
        assert_eq!(parse_vector("1,2", Some(2)).unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn malformed_nan_and_infinity_spellings_are_rejected() {
        for text in ["1nan,0", "nan1,0", "1inf,0", "infinityx,0", "0x1p3,0"] {
            assert!(matches!(
                parse_vector(text, Some(2)),
                Err(DbError::InvalidVector)
            ));
        }
    }
}
