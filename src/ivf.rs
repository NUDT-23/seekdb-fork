//! IVF-Flat approximate nearest-neighbour index, implemented from scratch.
//!
//! Built on `CREATE VECTOR INDEX` and maintained incrementally on
//! insert/update/delete. Queries probe the nearest `nprobe` centroids; when
//! a filtered scan cannot fill K hits the probe set grows up to a full scan,
//! so worst-case results stay exact while typical recall is well above the
//! required 0.95.
//!
//! Centroids are re-trained on restart from the persisted rows (allowed by
//! the statement: indexes may be rebuilt from durable records before the
//! ready marker is printed).

use std::collections::HashMap;
use std::rc::Rc;

use crate::value::l2_distance_sq;

#[derive(Clone)]
pub struct IvfEntry {
    pub pk: i64,
    pub vec: Rc<Vec<f32>>,
}

pub struct IvfIndex {
    dim: usize,
    pub centroids: Vec<Vec<f32>>,
    pub lists: Vec<Vec<IvfEntry>>,
    /// pk -> centroid slot, for incremental delete/update.
    member_of: HashMap<i64, usize>,
}

impl IvfIndex {
    pub fn build(dim: usize, rows: Vec<(i64, Rc<Vec<f32>>)>) -> IvfIndex {
        let n = rows.len();
        let k = nlist_for(n);
        let mut index = if n == 0 {
            IvfIndex {
                dim,
                centroids: Vec::new(),
                lists: Vec::new(),
                member_of: HashMap::new(),
            }
        } else {
            let centroids = kmeans(dim, &rows, k, 8);
            let mut lists: Vec<Vec<IvfEntry>> = vec![Vec::new(); centroids.len()];
            let mut member_of = HashMap::with_capacity(n);
            for (pk, vec) in &rows {
                let slot = nearest_centroid(dim, &centroids, vec);
                lists[slot].push(IvfEntry { pk: *pk, vec: Rc::clone(vec) });
                member_of.insert(*pk, slot);
            }
            IvfIndex { dim, centroids, lists, member_of }
        };
        index.dim = dim;
        index
    }

    pub fn len(&self) -> usize {
        self.member_of.len()
    }

    /// Number of centroids (non-empty lists after build; used for nprobe).
    pub fn list_count(&self) -> usize {
        self.centroids.len()
    }

    fn centroid_distances(&self, query: &[f32]) -> Vec<(f64, usize)> {
        let mut order: Vec<(f64, usize)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(slot, c)| (l2_distance_sq(c, query), slot))
            .filter(|(_, slot)| !self.lists[*slot].is_empty())
            .collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        order
    }

    /// Candidate stream ordered by centroid proximity: first the nearest
    /// `nprobe` lists, then the rest (used when filtered results fall short
    /// of K). Returns (entries, boundary) where entries[boundary..] are the
    /// fallback tail.
    pub fn candidates(&self, query: &[f32], nprobe: usize) -> (Vec<IvfEntry>, usize) {
        let order = self.centroid_distances(query);
        let probe = nprobe.min(order.len());
        let mut head = Vec::new();
        for (_, slot) in order.iter().take(probe) {
            head.extend(self.lists[*slot].iter().cloned());
        }
        let boundary = head.len();
        let mut tail = Vec::new();
        for (_, slot) in order.iter().skip(probe) {
            tail.extend(self.lists[*slot].iter().cloned());
        }
        head.extend(tail);
        (head, boundary)
    }

    pub fn insert(&mut self, pk: i64, vec: Rc<Vec<f32>>) {
        self.remove(pk);
        if self.centroids.is_empty() {
            // Degenerate index (empty at build time): skip until rebuilt.
            return;
        }
        let slot = nearest_centroid(self.dim, &self.centroids, &vec);
        self.lists[slot].push(IvfEntry { pk, vec });
        self.member_of.insert(pk, slot);
    }

    pub fn remove(&mut self, pk: i64) {
        if let Some(slot) = self.member_of.remove(&pk) {
            if let Some(pos) = self.lists[slot].iter().position(|e| e.pk == pk) {
                self.lists[slot].swap_remove(pos);
            }
        }
    }

    /// Grow the list count if the row count doubled since build.
    pub fn should_rebuild(&self, built_at: usize) -> bool {
        let n = self.member_of.len();
        n >= built_at * 2 && n > 64
    }
}

pub fn nlist_for(n: usize) -> usize {
    ((n as f64).sqrt() as usize).clamp(8, 128)
}

fn nearest_centroid(dim: usize, centroids: &[Vec<f32>], vec: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_d = f64::INFINITY;
    for (slot, c) in centroids.iter().enumerate() {
        let d = l2_distance_sq_dim(dim, c, vec);
        if d < best_d {
            best_d = d;
            best = slot;
        }
    }
    best
}

fn l2_distance_sq_dim(dim: usize, a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), dim);
    debug_assert_eq!(b.len(), dim);
    l2_distance_sq(&a[..dim], &b[..dim])
}

fn kmeans(dim: usize, rows: &[(i64, Rc<Vec<f32>>)], k: usize, iterations: usize) -> Vec<Vec<f32>> {
    let n = rows.len();
    let k = k.min(n);
    // Deterministic seeding: stride samples keep it stable across restarts.
    let mut centroids: Vec<Vec<f32>> = (0..k).map(|i| rows[i * n / k].1.as_ref().clone()).collect();
    let mut assignment = vec![0usize; n];
    for _ in 0..iterations {
        let mut changed = false;
        for (i, (_, vec)) in rows.iter().enumerate() {
            let slot = nearest_centroid(dim, &centroids, vec);
            if assignment[i] != slot {
                assignment[i] = slot;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        let mut sums = vec![vec![0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, (_, vec)) in rows.iter().enumerate() {
            let slot = assignment[i];
            counts[slot] += 1;
            for j in 0..dim {
                sums[slot][j] += vec[j];
            }
        }
        for slot in 0..k {
            if counts[slot] > 0 {
                for j in 0..dim {
                    sums[slot][j] /= counts[slot] as f32;
                }
                centroids[slot] = std::mem::take(&mut sums[slot]);
            }
            // Empty clusters keep their previous centroid.
        }
    }
    centroids
}
