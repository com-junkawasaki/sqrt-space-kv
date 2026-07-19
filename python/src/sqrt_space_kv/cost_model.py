"""Cost models for KV / activation memory residency strategies.

Pure port of cloud-murakumo's ``sqrt_space.cljc``. All functions are
dependency-free (stdlib ``math`` only) and operate on plain ints/floats/dicts
so results are trivially JSON-serializable and directly comparable against
this package's golden conformance fixture.

Honest framing (matches the upstream .cljc docstring): the "sqrt-checkpoint"
strategy transfers the same parameter-balancing idea used in gradient
checkpointing and KV recompute -- store O(b) content at checkpoints, recompute
blocks of size b on demand, with b chosen ~ sqrt(S * log S) for sequence
length S. This is NOT a claim that Transformers are multitape Turing
machines; ``cook_mertz_space``, ``block_respecting_cells`` and
``williams_space`` are research/comparison tooling for the complexity-theory
analogy, not residency policies to call directly -- for actual KV residency
use ``keep_indices`` / ``strategy_cost("sqrt-checkpoint", ...)``.
"""

from __future__ import annotations

import math

__all__ = [
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
]


def optimal_block(t: int, log_fn=math.log, min_b: int = 1) -> int:
    """b ~= ceil(sqrt(t * log(t))). `t` is sequence length S for LLM decode."""
    t = max(1, int(t))
    raw = math.sqrt(t * max(1.0, log_fn(t)))
    return max(int(min_b), math.ceil(raw))


def tree_height(t: int, b: int) -> int:
    """Number of blocks: h = ceil(t / b)."""
    t = max(1, int(t))
    b = max(1, int(b))
    return math.ceil(t / b)


def cook_mertz_space(b: int, h: int, d: int = 5) -> float:
    """Cook-Mertz Tree Evaluation space proxy: O(d*b + h*log(d*b))."""
    b = max(1, int(b))
    h = max(1, int(h))
    d = max(2, int(d))
    db = d * b
    log_term = max(1.0, math.log(db))
    return db + h * log_term


def williams_space(t: int, d: int = 5) -> dict:
    """End-to-end Williams-style space proxy for time-t simulation."""
    b = optimal_block(t)
    h = tree_height(t, b)
    space = cook_mertz_space(b, h, d)
    return {
        "t": t,
        "b": b,
        "h": h,
        "d": d,
        "space": space,
        "space_over_t": space / max(1, t),
        "regime": "sqrt-t",
    }


STRATEGY_CATALOG: dict[str, dict] = {
    "full-kv": {
        "id": "full-kv",
        "title": "Full KV cache (baseline)",
        "fidelity": "exact",
        "note": (
            "Standard autoregressive decode: store K,V for every past token. "
            "Space Theta(S). Recompute per new token Theta(1) relative to cached attention."
        ),
        "genes": {"kv": "store-all", "recompute": "none", "window": "none"},
    },
    "sliding-window": {
        "id": "sliding-window",
        "title": "Sliding-window KV (local attention)",
        "fidelity": "approximate",
        "note": (
            "Keep only last W tokens. Space Theta(W). Loses long-range exact attention -- "
            "not bit-identical to full context. Good latency, fails hard when exact "
            "long-context is required."
        ),
        "genes": {"kv": "window", "recompute": "none", "window": "fixed"},
    },
    "sqrt-checkpoint": {
        "id": "sqrt-checkpoint",
        "title": "sqrt(S) checkpoint recompute (Williams-inspired)",
        "fidelity": "exact-with-recompute",
        "note": (
            "Partition the sequence into blocks of length b ~= sqrt(S log S). Persist only "
            "block-boundary checkpoints (O(h)=O(sqrt(S)) block summaries / restart states) "
            "and recompute in-block KVs when needed. Bit-identical to full-KV if recompute "
            "is exact (same kernels). Space Theta(sqrt(S)) for checkpoints + Theta(b) working "
            "set; worst-case recompute Theta(b) per miss. Host-paging the evicted blocks "
            "measurably slows decode in practice -- see the README's Honest tradeoffs section."
        ),
        "genes": {"kv": "checkpoint", "recompute": "block", "window": "none"},
    },
    "block-respecting": {
        "id": "block-respecting",
        "title": "Block-respecting tape + recomputation graph",
        "fidelity": "exact-with-recompute",
        "note": (
            "Closer to Williams' block-respecting TM: time blocks of length b, tape blocks "
            "of length b, fan-in bounded recomputation DAG. Space proxy uses the Cook-Mertz "
            "formula rather than plain sqrt(S) checkpoints. More recompute sharing than naive "
            "checkpointing when multiple heads share tape blocks."
        ),
        "genes": {"kv": "tape-blocks", "recompute": "tree-eval", "window": "none"},
    },
    "expert-page": {
        "id": "expert-page",
        "title": "MoE expert paging (mlx-moe style)",
        "fidelity": "exact-for-moe",
        "note": (
            "Orthogonal axis: don't load all expert weights; page router-selected experts "
            "from SSD. Cuts *weight* memory, not KV. Composes with KV strategies. Space for "
            "weights ~= active-experts/total-experts * W_full."
        ),
        "genes": {"kv": "store-all", "recompute": "none", "experts": "paged"},
    },
    "sqrt-plus-page": {
        "id": "sqrt-plus-page",
        "title": "sqrt(S) checkpoint KV + MoE expert paging",
        "fidelity": "exact-with-recompute",
        "note": (
            "Compose sqrt-checkpoint KV residency with MoE expert paging -- useful for "
            "long-context MoE on unified/shared memory."
        ),
        "genes": {"kv": "checkpoint", "recompute": "block", "experts": "paged"},
    },
}


def model_shape(
    *,
    layers: int = 48,
    heads: int = 8,
    head_dim: int = 128,
    kv_bytes: int = 2,
    layers_kv: int | None = None,
) -> dict:
    """Default model shape for byte-cost estimates. Override for real models."""
    layers_kv = layers_kv or layers
    return {
        "layers": layers,
        "heads": heads,
        "head_dim": head_dim,
        "layers_kv": layers_kv,
        "kv_bytes": kv_bytes,
        "bytes_per_token": layers_kv * heads * head_dim * kv_bytes * 2,
    }


def full_kv_cells(s: int) -> int:
    return max(0, int(s))


def sliding_cells(s: int, w: int) -> int:
    return min(max(0, int(s)), max(1, int(w)))


def sqrt_checkpoint_cells(s: int) -> int:
    """Working-set size for the sqrt-checkpoint strategy: h checkpoints + one active block."""
    b = optimal_block(s)
    h = tree_height(s, b)
    return h + b


def keep_indices(s: int, b: int | None = None) -> list[int]:
    """Token indices retained on-device: boundary checkpoints union active tail block."""
    s = max(1, int(s))
    if b is None:
        b = optimal_block(s)
    b = max(1, int(b))
    ckpt = range(0, s, b)
    active = range(max(0, s - b), s)
    return sorted(set(ckpt) | set(active))


def block_respecting_cells(s: int, d: int = 5) -> int:
    return math.ceil(williams_space(s, d)["space"])


def strategy_cost(
    strategy: str,
    s: int,
    *,
    window: int = 4096,
    expert_active_frac: float = 0.125,
    model: dict | None = None,
    d: int = 5,
) -> dict:
    """Pure cost model for one strategy at sequence length `s`."""
    s = max(1, int(s))
    cat = STRATEGY_CATALOG.get(strategy)
    if cat is None:
        raise ValueError(f"unknown strategy: {strategy!r}")
    shape = model_shape(**model) if model else None
    bpt = shape["bytes_per_token"] if shape else 1

    if strategy == "full-kv":
        base = {
            "kv_cells": full_kv_cells(s),
            "weight_frac": 1.0,
            "recompute_per_token": 0.0,
            "params": {"b": s},
        }
    elif strategy == "sliding-window":
        base = {
            "kv_cells": sliding_cells(s, window),
            "weight_frac": 1.0,
            "recompute_per_token": 0.0,
            "params": {"window": window},
        }
    elif strategy == "sqrt-checkpoint":
        b = optimal_block(s)
        h = tree_height(s, b)
        base = {
            "kv_cells": sqrt_checkpoint_cells(s),
            "weight_frac": 1.0,
            "recompute_per_token": b / 2.0,
            "params": {"b": b, "h": h},
        }
    elif strategy == "block-respecting":
        ws = williams_space(s, d)
        base = {
            "kv_cells": math.ceil(ws["space"]),
            "weight_frac": 1.0,
            "recompute_per_token": ws["b"] / 2.0,
            "params": {"b": ws["b"], "h": ws["h"], "d": ws["d"], "space": ws["space"]},
        }
    elif strategy == "expert-page":
        base = {
            "kv_cells": full_kv_cells(s),
            "weight_frac": float(expert_active_frac),
            "recompute_per_token": 0.0,
            "params": {"expert_active_frac": expert_active_frac},
        }
    elif strategy == "sqrt-plus-page":
        b = optimal_block(s)
        h = tree_height(s, b)
        base = {
            "kv_cells": sqrt_checkpoint_cells(s),
            "weight_frac": float(expert_active_frac),
            "recompute_per_token": b / 2.0,
            "params": {"b": b, "h": h, "expert_active_frac": expert_active_frac},
        }
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")

    result = {
        "strategy": strategy,
        "s": s,
        "fidelity": cat["fidelity"],
        "title": cat["title"],
        "bytes": base["kv_cells"] * bpt,
        **base,
    }
    if shape:
        result["model"] = shape
    return result


def memory_ratio(cost: dict) -> float:
    """kv_cells(strategy) / s -- lower is better for memory."""
    return cost["kv_cells"] / max(1, cost["s"])


def work_cost(cost: dict, alpha: float = 0.05) -> float:
    """Scalar 'total work' proxy used for ranking strategies (lower is better)."""
    wfrac = cost.get("weight_frac") or 1.0
    beta = 0.1
    return cost["kv_cells"] + alpha * cost["recompute_per_token"] + beta * cost["s"] * wfrac


def sweep(lengths, strategies: list[str] | None = None, **opts) -> list[dict]:
    """Cost table for all strategies (or a subset) across sequence lengths."""
    strategies = strategies or list(STRATEGY_CATALOG.keys())
    return [strategy_cost(strat, s, **opts) for s in lengths for strat in strategies]


def compare_to_baseline(s: int, **opts) -> list[dict]:
    """For each strategy at S, report savings vs. full-kv."""
    base = strategy_cost("full-kv", s, **opts)
    results = []
    for strat in STRATEGY_CATALOG:
        c = dict(strategy_cost(strat, s, **opts))
        c["vs_full_kv_ratio"] = memory_ratio(c)
        c["cells_saved"] = base["kv_cells"] - c["kv_cells"]
        c["work"] = work_cost(c)
        c["work_vs_full"] = work_cost(c) / max(1.0, work_cost(base))
        results.append(c)
    return results
