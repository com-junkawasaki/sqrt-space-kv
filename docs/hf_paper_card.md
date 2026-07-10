---
title: "Simulating Autoregressive Memory with Square-Root Space"
authors:
  - Jun Kawasaki
  - cloud-murakumo / com-junkawasaki
tags:
  - llm
  - kv-cache
  - memory-efficiency
  - complexity
  - williams
  - serving
  - modal
  - moe
  - qwen3.6
---

# Square-Root Space KV Residency for LLM Decode

**TL;DR:** Williams (STOC 2025) √(t log t) space simulation → set KV checkpoint block \(b\approx\sqrt{S\log S}\). On real models (135M→14B dense, plus **Qwen3.6-35B-A3B MoE**), **~50–97% less cache storage** depending on architecture and S. Not a TM proof for Transformers—an engineering transfer with measurements.

## Headline numbers

### Dense (Modal A100, Qwen2.5 bf16)

| Model | S=16 384 full → √S | Save |
|-------|-------------------:|-----:|
| 7B | 940 MB → 25 MB | **97.3%** |
| 14B | 3.22 GB → 87 MB | **97.3%** |

### Hybrid MoE (Modal A100-80GB, Qwen3.6-35B-A3B bf16)

| S | full cache | √S resident | Save |
|--:|----------:|------------:|-----:|
| 4 096 | 117 MB | 38 MB | 67.9% |
| 8 192 | 201 MB | 38 MB | **81.3%** |
| 16 384 | 369 MB | 42 MB | **88.5%** |

Hybrid linear-attn state is O(1) (does not shrink); full-attn K/V shrinks with √S → **save grows with S**.

### Attention fidelity

Block-stream online-softmax vs full: max\|Δ\| within **bf16 noise** (~0.016 @ S=2048). Fidelity maintained under compression.

## What fails

- **Delete cache + full re-forward every token:** peak GPU/Metal memory goes *up*.
- **Process RSS alone:** often weight-dominated.

## Links

- Results (public): https://github.com/com-junkawasaki/sqrt-space-kv
- Code: https://github.com/gftdcojp/cloud-murakumo (`scripts/modal_sqrt_kv_bench.py`)
- Theory: Williams arXiv:2502.17779 (STOC 2025 Best Paper)
- arXiv draft: 7807366 (cs.CL, submitted 2026-07-10; public id pending)

## Cite

```bibtex
@misc{sqrt-space-kv-2026,
  title  = {Simulating Autoregressive Memory with Square-Root Space},
  author = {Kawasaki, Jun},
  year   = {2026},
  note   = {Code: github.com/gftdcojp/cloud-murakumo; Results: github.com/com-junkawasaki/sqrt-space-kv},
  howpublished = {\url{https://github.com/com-junkawasaki/sqrt-space-kv}}
}
```
