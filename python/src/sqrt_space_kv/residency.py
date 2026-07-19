"""Apply a residency keep-set to K/V tensors represented as nested lists.

Pure port of the residency-application functions in cloud-murakumo's
``kv_runtime.cljc``. Tensor backends (PyTorch/MLX/etc.) stay outside this
package: it operates on plain nested Python lists so it has zero framework
dependency; callers convert to/from their own tensor type at the boundary.
"""

from __future__ import annotations

__all__ = [
    "apply_residency_seq",
    "apply_residency_kv",
    "residency_bytes_estimate",
    "apply_plan_to_kv",
]


def apply_residency_seq(xs: list, keep: list[int]) -> list:
    """Drop non-keep positions along a 1-D sequence (token axis)."""
    n = len(xs)
    return [xs[i] for i in keep if 0 <= i < n]


def apply_residency_kv(k, v, keep: list[int], seq_axis: int = 2):
    """Apply on classic K/V as nested lists [B H S D] (or [S D] / [H S D]).

    `seq_axis` is the 0-based depth of the sequence dimension (default: axis
    2 for a 4-D [B H S D] layout). Returns (k2, v2) with sequence length
    ``len(keep)`` at that axis.
    """
    keep = list(keep)

    def walk(x, depth):
        if not isinstance(x, (list, tuple)):
            return x
        if depth == seq_axis:
            return apply_residency_seq(x, keep)
        return [walk(el, depth + 1) for el in x]

    return walk(k, 0), walk(v, 0)


def residency_bytes_estimate(shapes_nbytes: list[dict], keep: list[int], s: int) -> int:
    """Estimate kept bytes given full nbytes and |keep|/S on seq-shaped tensors.

    `shapes_nbytes` is a list of {"shape": [...], "nbytes": n} dicts.
    Non-seq tensors (no dim == s, e.g. linear-attn O(1) state) counted fully.
    """
    n_keep = max(1, len(keep))
    s = max(1, int(s))
    total = 0
    for entry in shapes_nbytes:
        shape = entry.get("shape") or []
        nb = int(entry.get("nbytes") or 0)
        if s > 1 and any(int(dim) == s for dim in shape):
            total += int(nb * (n_keep / s))
        else:
            total += nb
    return total


def apply_plan_to_kv(
    k,
    v,
    keep: list[int],
    *,
    strategy: str = "sqrt-checkpoint",
    device_mode: str | None = None,
    seq_axis: int = 2,
):
    """Convenience wrapper: drop non-keep K/V slots unless the plan is a no-op.

    A no-op plan is ``device_mode == "all-tokens"`` or ``strategy == "full-kv"``
    -- both mean "keep everything," matching cloud-murakumo's ``apply-plan-to-kv``.
    """
    if device_mode == "all-tokens" or strategy == "full-kv":
        return k, v
    return apply_residency_kv(k, v, keep, seq_axis)
