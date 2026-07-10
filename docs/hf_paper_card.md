---
title: "Simulating Autoregressive Memory with Square-Root Space"
authors:
  - cloud-murakumo / com-junkawasaki
tags:
  - llm
  - kv-cache
  - memory-efficiency
  - complexity
  - williams
  - serving
  - modal
---

# Square-Root Space KV Residency for LLM Decode

**TL;DR:** Williams (STOC 2025) √(t log t) space simulation → set KV checkpoint block \(b\approx\sqrt{S\log S}\). On real models (135M→7B, Mac + Modal A100), **90–97% less KV storage** at 1k–16k context. Not a TM proof for Transformers—an engineering transfer with measurements.

## Headline numbers (Modal NVIDIA)

**Qwen2.5-7B bf16 · A100-40GB**

| Context S | Full KV | √S resident | Saved |
|----------:|--------:|------------:|------:|
| 4 096 | 235 MB | 12 MB | 94.9% |
| 8 192 | 470 MB | 17 MB | 96.3% |
| 16 384 | 940 MB | 25 MB | **97.3%** |

**Qwen2.5-14B bf16 · A100-80GB** — same save %; at S=16 384 full KV **3.22 GB → 87 MB** (~3.1 GB freed per sequence).

Same **save %** on SmolLM-135M / Qwen0.5B (Mac MLX)—ratio is universal; absolute MB scales with KV width.
## What fails

- **Delete cache + full re-forward every token:** peak GPU/Metal memory goes *up*.
- **Process RSS alone:** often weight-dominated (gemma4-E4B SIZE ~flat to 32k).

## Links

- Draft: [`sqrt_space_llm_kv.md`](sqrt_space_llm_kv.md)
- Code: `scripts/modal_sqrt_kv_bench.py`, `scripts/bench_sqrt_kv_mac.py`
- Theory: Williams arXiv:2502.17779 (STOC 2025 Best Paper)
- Product surface: `cloud-murakumo` `:serve :kv-policy`

## Cite (temporary)

```bibtex
@misc{cloudmurakumo-sqrt-kv-2026,
  title  = {Simulating Autoregressive Memory with Square-Root Space},
  author = {cloud-murakumo},
  year   = {2026},
  note   = {Draft; code github.com/gftdcojp/cloud-murakumo},
  howpublished = {\url{https://github.com/gftdcojp/cloud-murakumo}}
}
```
