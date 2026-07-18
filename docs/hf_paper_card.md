---
title: "sqrt(S) KV-Cache Residency for Long-Context LLM Decode"
former_title: "Simulating Autoregressive Memory with Square-Root Space"
authors:
  - Jun Kawasaki
  - cloud-murakumo / com-junkawasaki
tags:
  - llm
  - kv-cache
  - memory-efficiency
  - long-context
  - apple-silicon
  - on-device
  - complexity
  - williams
  - serving
  - modal
  - moe
  - mla
  - qwen2.5
  - qwen3.6
  - glm-4
  - deepseek-v2
---

# Square-Root Space KV Residency: a real memory-capability unlock, validated end-to-end

**TL;DR:** This is a *memory-capability* technique, not a speed technique — the
goal is running long-context generation on hardware that cannot otherwise fit
it, at an accepted decode-speed cost. Applying Williams' (STOC 2025)
√(t log t) checkpoint-stride idea to LLM KV caches, we validated the full
chain end-to-end on real hardware: **exact correctness** (byte-identical
generated tokens vs. full-KV decode), **honest cost** (2.2–3.7× slower
decode, measured two independent ways), a **real competitive edge** (exact
at compression ratios where lossy competitors score 0% on retrieval), and a
**real on-device capability unlock** on Apple Silicon — turning a model that
can use only 6% of its own designed context window on a 16GB Mac into one
that can use effectively all of it.

## The headline result: unlocking a model's own designed capability

**GLM-4-9B-Chat-1M** (native 1,048,576-token context, 7.7GB weights at 6-bit)
on an Apple M4:

| RAM | reachable context **without** this technique | with √S residency + `mmap` paging |
|---|---:|---:|
| 16GB | 64,374 tokens (**6.1%** of the model's own 1M design) | unlocked — resident set stays O(√S), tens of MB |
| 32GB | 259,687 tokens (**24.8%**) | unlocked |

Real weights, real prefill, real measured bytes-per-token (80.0 KB/token) —
not a spec-sheet estimate. Weights are cheap; the KV cache alone caps a
16GB Mac at a sixteenth of what the model was built to do.

## Why it costs what it costs, on two different architectures

- **Discrete GPU (Modal A100):** paging the non-resident set crosses a real
  PCIe bus. Measured (two independent methods — isolated transfer, and a
  real `Cache` subclass fused into a real multi-token `generate()` loop):
  **2.18×–3.66× slower decode**, worsening as context grows. Generated
  tokens were **byte-identical** to standard full-KV decode.
- **Apple Silicon (unified memory):** no PCIe bus exists — so does
  "offloading" even help here? We tested it directly: moving KV to a plain
  host buffer does **not** free real memory (RSS goes *up*); a naive
  `write()`+`delete()` disk round trip mostly doesn't either. **Only
  `np.memmap`-based paging does** — confirmed at **0.0MB RSS cost** up to
  6.3GB of non-resident data, with genuine on-demand partial residency.
  This is a concrete, previously-undocumented implementation requirement,
  not a theoretical aside.

## The competitive edge that makes the cost worth paying (sometimes)

At sqrt-space-kv's own ~90% compression regime, NVIDIA `kvpress` presses
(StreamingLLM, Knorm, SnapKV) scored **0% accuracy** on a needle-in-haystack
retrieval task on the same model. sqrt-space-kv is exact by construction —
it relocates KV, it doesn't delete it — so it scores 100% at any
compression ratio, at the cost of the paging tax above. Neither approach is
a free lunch; which cost you can afford is a deployment decision, not
something either project's numbers alone can answer for you.

## Headline storage numbers (the original claim, still holds)

| Model | S=16,384 full → √S resident | Save |
|-------|-------------------:|-----:|
| Qwen2.5-7B | 940 MB → 25 MB | 97.3% |
| Qwen2.5-14B | 3.22 GB → 87 MB | 97.3% |
| Qwen3.6-35B-A3B (hybrid MoE) | 369 MB → 42 MB | 88.5% (save grows with S) |
| DeepSeek-V2-Lite (MLA-architected) | 2.26 GB → 83 MB | 96.3% |

## What we found doesn't work (negative results, reported as such)

- **Whole-prefix recompute every token:** peak GPU/Metal memory goes *up*,
  not down.
- **MLA "for free":** DeepSeek-V2's reference cache — in *both* the PyTorch
  and MLX ecosystems — stores the decompressed per-head form, not the true
  compressed latent. The absorbed-cache optimization that would make this
  compose with MLA exists only in vLLM/SGLang (CUDA-only); the real
  composability question is still open.
- **CPU-side "offload" on unified memory:** measured, not assumed, to not
  free real memory (see above) — the intuition carried over from discrete
  GPUs doesn't transfer.

## Links

- Results (public): https://github.com/com-junkawasaki/sqrt-space-kv
- Maturity ADR (full evidence trail, M0–M6): `com-junkawasaki/root`
  `90-docs/adr/2607182800-sqrt-space-kv-mla-composability-maturity-review.edn`
- Code: https://github.com/gftdcojp/cloud-murakumo (`scripts/modal_sqrt_kv_*.py`,
  `scripts/bench_sqrt_kv_mac_*.py`)
- Theory: Williams arXiv:2502.17779 (STOC 2025 Best Paper) — read as
  inspiration for the checkpoint-stride formula, not a technical dependency;
  the underlying checkpoint+recompute pattern predates it (Griewank;
  Chen et al. 2016, "Training Deep Nets with Sublinear Memory Cost").
- arXiv draft: 7807366 (cs.CL, submitted 2026-07-10; **on hold** in
  moderation as of 2026-07-18, public id pending; paper revised 2026-07-18
  to de-emphasize the Williams framing while on hold — not yet resubmitted)

## Cite

```bibtex
@misc{sqrt-space-kv-2026,
  title  = {sqrt(S) KV-Cache Residency for Long-Context LLM Decode: Real
            Costs, Real Correctness, and a Real Capability Unlock on
            Memory-Constrained Hardware},
  author = {Kawasaki, Jun},
  year   = {2026},
  note   = {Code: github.com/gftdcojp/cloud-murakumo; Results: github.com/com-junkawasaki/sqrt-space-kv},
  howpublished = {\url{https://github.com/com-junkawasaki/sqrt-space-kv}}
}
```
