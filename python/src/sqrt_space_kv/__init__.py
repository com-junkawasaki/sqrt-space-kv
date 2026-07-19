"""sqrt-space-kv: Sqrt-Checkpoint KV-cache residency.

Pure-Python (stdlib only, zero runtime dependencies) port of the canonical
implementation in cloud-murakumo (Clojure ``.cljc``). See the repository
README for the honest tradeoffs of this strategy (host-paging the evicted
blocks measurably slows decode -- this is capacity-for-latency, not a free
lunch) before using it in a serving path.
"""

from .attention import block_stream_attn, online_softmax_attn
from .cost_model import (
    STRATEGY_CATALOG,
    block_respecting_cells,
    compare_to_baseline,
    cook_mertz_space,
    full_kv_cells,
    keep_indices,
    memory_ratio,
    model_shape,
    optimal_block,
    sliding_cells,
    sqrt_checkpoint_cells,
    strategy_cost,
    sweep,
    tree_height,
    williams_space,
    work_cost,
)
from .residency import (
    apply_plan_to_kv,
    apply_residency_kv,
    apply_residency_seq,
    residency_bytes_estimate,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "optimal_block",
    "tree_height",
    "cook_mertz_space",
    "williams_space",
    "STRATEGY_CATALOG",
    "model_shape",
    "full_kv_cells",
    "sliding_cells",
    "sqrt_checkpoint_cells",
    "keep_indices",
    "block_respecting_cells",
    "strategy_cost",
    "memory_ratio",
    "work_cost",
    "sweep",
    "compare_to_baseline",
    "apply_residency_seq",
    "apply_residency_kv",
    "residency_bytes_estimate",
    "apply_plan_to_kv",
    "online_softmax_attn",
    "block_stream_attn",
]
