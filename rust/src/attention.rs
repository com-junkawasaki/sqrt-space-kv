//! Exact reference attention over a residency-reduced K/V set.
//!
//! Pure port of `online-softmax-attn` / `block-stream-attn` from
//! cloud-murakumo's `kv_runtime.cljc`. Single-query, plain f64 math -- a
//! reference implementation for exactness testing and small-scale use, not a
//! performance kernel (no batching, no SIMD/vectorized backend).

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Exact single-query attention. q `[D]`, k `[S][D]`, v `[S][D]` -> out `[D]`.
pub fn online_softmax_attn(q: &[f64], k: &[Vec<f64>], v: &[Vec<f64>], scale: f64) -> Vec<f64> {
    let s = k.len();
    let scores: Vec<f64> = (0..s).map(|i| scale * dot(q, &k[i])).collect();
    let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ex: Vec<f64> = scores.iter().map(|&sc| (sc - m).exp()).collect();
    let den: f64 = ex.iter().sum();
    let w: Vec<f64> = ex.iter().map(|&e| e / den).collect();
    let d = q.len();
    (0..d)
        .map(|j| (0..s).map(|i| w[i] * v[i][j]).sum())
        .collect()
}

/// Exact attention via online-softmax over blocks of size `block`. Matches
/// [`online_softmax_attn`] within floating-point noise regardless of block
/// size -- this is what makes the residency strategy "exact with recompute"
/// rather than approximate.
pub fn block_stream_attn(
    q: &[f64],
    k: &[Vec<f64>],
    v: &[Vec<f64>],
    scale: f64,
    block: u64,
) -> Vec<f64> {
    let s = k.len();
    let d = q.len();
    let block = block.max(1) as usize;
    let mut start = 0usize;
    let mut m = f64::NEG_INFINITY;
    let mut l = 0.0f64;
    let mut acc = vec![0.0f64; d];
    while start < s {
        let end = (start + block).min(s);
        let scores: Vec<f64> = (start..end).map(|i| scale * dot(q, &k[i])).collect();
        let m_blk = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let m_new = m.max(m_blk);
        let alpha = (m - m_new).exp();
        let ex: Vec<f64> = scores.iter().map(|&sc| (sc - m_new).exp()).collect();
        l = l * alpha + ex.iter().sum::<f64>();
        acc = (0..d)
            .map(|j| {
                acc[j] * alpha
                    + (0..ex.len())
                        .map(|ii| ex[ii] * v[start + ii][j])
                        .sum::<f64>()
            })
            .collect();
        // carry m_new into the next iteration's `m` -- forgetting this line
        // silently discards all prior blocks (caught by this crate's own
        // golden-fixture test suite during the initial Python/Rust port).
        m = m_new;
        start = end;
    }
    acc.iter().map(|&a| a / l.max(1e-20)).collect()
}
