//! Sqrt-Checkpoint KV-cache residency -- pure port of cloud-murakumo's
//! `sqrt_space.cljc` / `kv_runtime.cljc` (canonical reference).
//!
//! See the [repository README](https://github.com/com-junkawasaki/sqrt-space-kv)
//! for the honest tradeoffs of this strategy (host-paging the evicted blocks
//! measurably slows decode -- this is capacity-for-latency, not a free
//! lunch) before using it in a serving path.

pub mod attention;
pub mod cost_model;
pub mod residency;

pub use attention::{block_stream_attn, online_softmax_attn};
pub use cost_model::{
    compare_to_baseline, keep_indices, model_shape, optimal_block, strategy_cost, sweep,
    tree_height, STRATEGY_NAMES,
};
pub use residency::{apply_plan_to_kv_hsd, apply_residency_kv_hsd, apply_residency_seq};
