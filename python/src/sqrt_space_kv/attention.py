"""Exact reference attention over a residency-reduced K/V set.

Pure port of ``online-softmax-attn`` / ``block-stream-attn`` from
cloud-murakumo's ``kv_runtime.cljc``. Single-query, pure Python-list math --
a reference implementation for exactness testing and small-scale use, not a
performance kernel (no batching, no vectorized backend).
"""

from __future__ import annotations

import math

__all__ = ["online_softmax_attn", "block_stream_attn"]


def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def online_softmax_attn(q, k, v, scale: float) -> list[float]:
    """Exact single-query attention. q [D], k [S D], v [S D] -> out [D]."""
    s = len(k)
    scores = [scale * _dot(q, k[i]) for i in range(s)]
    m = max(scores)
    ex = [math.exp(sc - m) for sc in scores]
    den = sum(ex)
    w = [e / den for e in ex]
    d = len(q)
    return [sum(w[i] * v[i][j] for i in range(s)) for j in range(d)]


def block_stream_attn(q, k, v, scale: float, block: int) -> list[float]:
    """Exact attention via online-softmax over blocks of size `block`.

    Matches ``online_softmax_attn`` within floating-point noise regardless of
    block size -- this is what makes the residency strategy "exact with
    recompute" rather than approximate.
    """
    s = len(k)
    d = len(q)
    block = max(1, int(block))
    start = 0
    m = float("-inf")
    l_sum = 0.0
    acc = [0.0] * d
    while start < s:
        end = min(s, start + block)
        scores = [scale * _dot(q, k[i]) for i in range(start, end)]
        m_blk = max(scores)
        m_new = max(m, m_blk)
        alpha = math.exp(m - m_new)
        ex = [math.exp(sc - m_new) for sc in scores]
        l_sum = l_sum * alpha + sum(ex)
        acc = [
            acc[j] * alpha + sum(ex[ii] * v[start + ii][j] for ii in range(len(ex)))
            for j in range(d)
        ]
        m = m_new
        start = end
    return [a / max(l_sum, 1e-20) for a in acc]
