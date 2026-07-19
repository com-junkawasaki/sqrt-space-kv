//! Apply a residency keep-set to K/V tensors.
//!
//! Pure port of the residency-application functions in cloud-murakumo's
//! `kv_runtime.cljc`. The canonical `.cljc` walks arbitrarily-nested vectors
//! at a caller-chosen sequence axis (Clojure's dynamic typing makes that
//! natural); Rust's static typing makes an arbitrary-depth generic walker
//! more machinery than this small library needs. Instead: a generic 1-D
//! primitive ([`apply_residency_seq`]) plus a concretely-typed convenience
//! for the common `[H][S][D]` K/V layout ([`apply_residency_kv_hsd`],
//! sequence axis = the middle dimension). For other layouts, call
//! `apply_residency_seq` directly on your sequence-axis slices.

/// Drop non-keep positions along a 1-D sequence (token axis).
pub fn apply_residency_seq<T: Clone>(xs: &[T], keep: &[u64]) -> Vec<T> {
    let n = xs.len() as u64;
    keep.iter()
        .filter(|&&i| i < n)
        .map(|&i| xs[i as usize].clone())
        .collect()
}

/// Apply residency on K/V shaped as `[H][S][D]` (sequence axis = 1, the
/// middle dimension) -- the common transformer K/V cache layout after
/// collapsing the batch dimension. Returns `(k2, v2)` with sequence length
/// `keep.len()`.
pub fn apply_residency_kv_hsd<T: Clone>(
    k: &[Vec<T>],
    v: &[Vec<T>],
    keep: &[u64],
) -> (Vec<Vec<T>>, Vec<Vec<T>>) {
    let k2 = k
        .iter()
        .map(|head| apply_residency_seq(head, keep))
        .collect();
    let v2 = v
        .iter()
        .map(|head| apply_residency_seq(head, keep))
        .collect();
    (k2, v2)
}

#[derive(Debug, Clone)]
pub struct ShapeNbytes {
    pub shape: Vec<u64>,
    pub nbytes: u64,
}

/// Estimate kept bytes given full nbytes and |keep|/S on seq-shaped tensors.
/// Non-seq tensors (no dim == s, e.g. linear-attn O(1) state) counted fully.
pub fn residency_bytes_estimate(shapes_nbytes: &[ShapeNbytes], keep: &[u64], s: u64) -> u64 {
    let n_keep = keep.len().max(1) as f64;
    let s = s.max(1);
    let mut total = 0u64;
    for entry in shapes_nbytes {
        let nb = entry.nbytes;
        if s > 1 && entry.shape.contains(&s) {
            total += (nb as f64 * (n_keep / s as f64)) as u64;
        } else {
            total += nb;
        }
    }
    total
}

/// Convenience wrapper: drop non-keep K/V slots on a `[H][S][D]` layout
/// unless the plan is a no-op (`device_mode == "all-tokens"` or
/// `strategy == "full-kv"` both mean "keep everything").
pub fn apply_plan_to_kv_hsd<T: Clone>(
    k: &[Vec<T>],
    v: &[Vec<T>],
    keep: &[u64],
    strategy: &str,
    device_mode: Option<&str>,
) -> (Vec<Vec<T>>, Vec<Vec<T>>) {
    if device_mode == Some("all-tokens") || strategy == "full-kv" {
        return (k.to_vec(), v.to_vec());
    }
    apply_residency_kv_hsd(k, v, keep)
}
