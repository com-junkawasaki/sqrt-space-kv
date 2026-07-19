//! Cost models for KV / activation memory residency strategies.
//!
//! Pure port of cloud-murakumo's `sqrt_space.cljc`. Zero dependencies (std
//! f64 math only). `cook_mertz_space`, `block_respecting_cells` and
//! `williams_space` are research/comparison tooling for the complexity-theory
//! analogy, not residency policies to call directly -- for actual KV
//! residency use [`keep_indices`] / [`strategy_cost`] with `"sqrt-checkpoint"`.

use std::collections::{BTreeSet, HashMap};

/// b ~= ceil(sqrt(t * ln(t))). `t` is sequence length S for LLM decode.
pub fn optimal_block(t: u64) -> u64 {
    optimal_block_min(t, 1)
}

/// [`optimal_block`] with a floor on the returned block size.
pub fn optimal_block_min(t: u64, min_b: u64) -> u64 {
    let t = t.max(1) as f64;
    let raw = (t * t.ln().max(1.0)).sqrt();
    min_b.max(1).max(raw.ceil() as u64)
}

/// Number of blocks: h = ceil(t / b).
pub fn tree_height(t: u64, b: u64) -> u64 {
    let t = t.max(1) as f64;
    let b = b.max(1) as f64;
    (t / b).ceil() as u64
}

/// Cook-Mertz Tree Evaluation space proxy: O(d*b + h*log(d*b)).
pub fn cook_mertz_space(b: u64, h: u64, d: u64) -> f64 {
    let b = b.max(1) as f64;
    let h = h.max(1) as f64;
    let d = d.max(2) as f64;
    let db = d * b;
    let log_term = db.ln().max(1.0);
    db + h * log_term
}

#[derive(Debug, Clone, Copy)]
pub struct WilliamsSpace {
    pub t: u64,
    pub b: u64,
    pub h: u64,
    pub d: u64,
    pub space: f64,
    pub space_over_t: f64,
}

/// End-to-end Williams-style space proxy for time-t simulation.
pub fn williams_space(t: u64, d: u64) -> WilliamsSpace {
    let b = optimal_block(t);
    let h = tree_height(t, b);
    let space = cook_mertz_space(b, h, d);
    WilliamsSpace {
        t,
        b,
        h,
        d,
        space,
        space_over_t: space / (t.max(1) as f64),
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StrategyInfo {
    pub id: &'static str,
    pub title: &'static str,
    pub fidelity: &'static str,
    pub note: &'static str,
}

pub const STRATEGY_NAMES: [&str; 6] = [
    "full-kv",
    "sliding-window",
    "sqrt-checkpoint",
    "block-respecting",
    "expert-page",
    "sqrt-plus-page",
];

pub fn strategy_info(strategy: &str) -> Option<StrategyInfo> {
    Some(match strategy {
        "full-kv" => StrategyInfo {
            id: "full-kv",
            title: "Full KV cache (baseline)",
            fidelity: "exact",
            note: "Standard autoregressive decode: store K,V for every past token. \
                   Space Theta(S). Recompute per new token Theta(1) relative to cached attention.",
        },
        "sliding-window" => StrategyInfo {
            id: "sliding-window",
            title: "Sliding-window KV (local attention)",
            fidelity: "approximate",
            note: "Keep only last W tokens. Space Theta(W). Loses long-range exact attention -- \
                   not bit-identical to full context.",
        },
        "sqrt-checkpoint" => StrategyInfo {
            id: "sqrt-checkpoint",
            title: "sqrt(S) checkpoint recompute (Williams-inspired)",
            fidelity: "exact-with-recompute",
            note: "Partition the sequence into blocks of length b ~= sqrt(S log S). Persist \
                   only block-boundary checkpoints and recompute in-block KVs when needed. \
                   Host-paging the evicted blocks measurably slows decode in practice -- see \
                   the README's Honest tradeoffs section.",
        },
        "block-respecting" => StrategyInfo {
            id: "block-respecting",
            title: "Block-respecting tape + recomputation graph",
            fidelity: "exact-with-recompute",
            note: "Closer to Williams' block-respecting TM. Space proxy uses the Cook-Mertz \
                   formula rather than plain sqrt(S) checkpoints.",
        },
        "expert-page" => StrategyInfo {
            id: "expert-page",
            title: "MoE expert paging (mlx-moe style)",
            fidelity: "exact-for-moe",
            note: "Orthogonal axis: page router-selected experts from SSD. Cuts weight memory, \
                   not KV. Composes with KV strategies.",
        },
        "sqrt-plus-page" => StrategyInfo {
            id: "sqrt-plus-page",
            title: "sqrt(S) checkpoint KV + MoE expert paging",
            fidelity: "exact-with-recompute",
            note: "Compose sqrt-checkpoint KV residency with MoE expert paging -- useful for \
                   long-context MoE on unified/shared memory.",
        },
        _ => return None,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct ModelShape {
    pub layers: u64,
    pub heads: u64,
    pub head_dim: u64,
    pub layers_kv: u64,
    pub kv_bytes: u64,
    pub bytes_per_token: u64,
}

/// Default model shape for byte-cost estimates. Override for real models.
pub fn model_shape(
    layers: u64,
    heads: u64,
    head_dim: u64,
    kv_bytes: u64,
    layers_kv: Option<u64>,
) -> ModelShape {
    let layers_kv = layers_kv.unwrap_or(layers);
    ModelShape {
        layers,
        heads,
        head_dim,
        layers_kv,
        kv_bytes,
        bytes_per_token: layers_kv * heads * head_dim * kv_bytes * 2,
    }
}

pub fn full_kv_cells(s: u64) -> u64 {
    s
}

pub fn sliding_cells(s: u64, w: u64) -> u64 {
    s.min(w.max(1))
}

/// Working-set size for the sqrt-checkpoint strategy: h checkpoints + one active block.
pub fn sqrt_checkpoint_cells(s: u64) -> u64 {
    let b = optimal_block(s);
    let h = tree_height(s, b);
    h + b
}

/// Token indices retained on-device: boundary checkpoints union active tail block.
pub fn keep_indices(s: u64) -> Vec<u64> {
    keep_indices_with_block(s, optimal_block(s))
}

pub fn keep_indices_with_block(s: u64, b: u64) -> Vec<u64> {
    let s = s.max(1);
    let b = b.max(1);
    let mut set: BTreeSet<u64> = BTreeSet::new();
    let mut i = 0u64;
    while i < s {
        set.insert(i);
        i += b;
    }
    let start = s.saturating_sub(b);
    for i in start..s {
        set.insert(i);
    }
    set.into_iter().collect()
}

pub fn block_respecting_cells(s: u64, d: u64) -> u64 {
    williams_space(s, d).space.ceil() as u64
}

#[derive(Debug, Clone)]
pub struct StrategyCost {
    pub strategy: String,
    pub s: u64,
    pub kv_cells: u64,
    pub weight_frac: f64,
    pub recompute_per_token: f64,
    pub fidelity: &'static str,
    pub title: &'static str,
    pub bytes: u64,
    pub params: HashMap<String, f64>,
    pub model: Option<ModelShape>,
}

#[derive(Debug, Clone)]
pub struct StrategyCostOpts {
    pub window: u64,
    pub expert_active_frac: f64,
    pub d: u64,
    pub model: Option<ModelShape>,
}

impl Default for StrategyCostOpts {
    fn default() -> Self {
        StrategyCostOpts {
            window: 4096,
            expert_active_frac: 0.125,
            d: 5,
            model: None,
        }
    }
}

/// Pure cost model for one strategy at sequence length `s`.
pub fn strategy_cost(strategy: &str, s: u64) -> Result<StrategyCost, String> {
    strategy_cost_opts(strategy, s, &StrategyCostOpts::default())
}

pub fn strategy_cost_opts(
    strategy: &str,
    s: u64,
    opts: &StrategyCostOpts,
) -> Result<StrategyCost, String> {
    let s = s.max(1);
    let info = strategy_info(strategy).ok_or_else(|| format!("unknown strategy: {strategy}"))?;
    let bpt = opts.model.map(|m| m.bytes_per_token).unwrap_or(1);

    let mut params: HashMap<String, f64> = HashMap::new();
    let (kv_cells, weight_frac, recompute_per_token) = match strategy {
        "full-kv" => {
            params.insert("b".into(), s as f64);
            (full_kv_cells(s), 1.0, 0.0)
        }
        "sliding-window" => {
            params.insert("window".into(), opts.window as f64);
            (sliding_cells(s, opts.window), 1.0, 0.0)
        }
        "sqrt-checkpoint" => {
            let b = optimal_block(s);
            let h = tree_height(s, b);
            params.insert("b".into(), b as f64);
            params.insert("h".into(), h as f64);
            (sqrt_checkpoint_cells(s), 1.0, b as f64 / 2.0)
        }
        "block-respecting" => {
            let ws = williams_space(s, opts.d);
            params.insert("b".into(), ws.b as f64);
            params.insert("h".into(), ws.h as f64);
            params.insert("d".into(), ws.d as f64);
            params.insert("space".into(), ws.space);
            (ws.space.ceil() as u64, 1.0, ws.b as f64 / 2.0)
        }
        "expert-page" => {
            params.insert("expert_active_frac".into(), opts.expert_active_frac);
            (full_kv_cells(s), opts.expert_active_frac, 0.0)
        }
        "sqrt-plus-page" => {
            let b = optimal_block(s);
            let h = tree_height(s, b);
            params.insert("b".into(), b as f64);
            params.insert("h".into(), h as f64);
            params.insert("expert_active_frac".into(), opts.expert_active_frac);
            (
                sqrt_checkpoint_cells(s),
                opts.expert_active_frac,
                b as f64 / 2.0,
            )
        }
        _ => unreachable!("validated by strategy_info above"),
    };

    Ok(StrategyCost {
        strategy: strategy.to_string(),
        s,
        kv_cells,
        weight_frac,
        recompute_per_token,
        fidelity: info.fidelity,
        title: info.title,
        bytes: kv_cells * bpt,
        params,
        model: opts.model,
    })
}

pub fn memory_ratio(cost: &StrategyCost) -> f64 {
    cost.kv_cells as f64 / (cost.s.max(1) as f64)
}

pub fn work_cost(cost: &StrategyCost) -> f64 {
    work_cost_alpha(cost, 0.05)
}

pub fn work_cost_alpha(cost: &StrategyCost, alpha: f64) -> f64 {
    let beta = 0.1;
    cost.kv_cells as f64
        + alpha * cost.recompute_per_token
        + beta * (cost.s as f64) * cost.weight_frac
}

/// Cost table for all strategies (or a subset) across sequence lengths.
pub fn sweep(lengths: &[u64], strategies: Option<&[&str]>) -> Vec<StrategyCost> {
    let strats: Vec<&str> = strategies
        .map(|s| s.to_vec())
        .unwrap_or_else(|| STRATEGY_NAMES.to_vec());
    let mut out = Vec::new();
    for &s in lengths {
        for &strat in &strats {
            if let Ok(c) = strategy_cost(strat, s) {
                out.push(c);
            }
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct ComparedCost {
    pub cost: StrategyCost,
    pub vs_full_kv_ratio: f64,
    pub cells_saved: i64,
    pub work: f64,
    pub work_vs_full: f64,
}

/// For each strategy at S, report savings vs. full-kv.
pub fn compare_to_baseline(s: u64) -> Vec<ComparedCost> {
    let base = strategy_cost("full-kv", s).expect("full-kv is always known");
    STRATEGY_NAMES
        .iter()
        .filter_map(|&strat| {
            let c = strategy_cost(strat, s).ok()?;
            let vs_full_kv_ratio = memory_ratio(&c);
            let cells_saved = base.kv_cells as i64 - c.kv_cells as i64;
            let work = work_cost(&c);
            let work_vs_full = work / work_cost(&base).max(1.0);
            Some(ComparedCost {
                cost: c,
                vs_full_kv_ratio,
                cells_saved,
                work,
                work_vs_full,
            })
        })
        .collect()
}
