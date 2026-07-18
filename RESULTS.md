# Results summary — √S KV residency (2026-07-09)

## What we measured

1. **Full KV bytes** after real-model prefill of length S (all layers, K+V).
2. **√S resident set** = boundary checkpoints every  
   \(b=\lceil\sqrt{S\ln S}\rceil\) **plus** one active block of length b.
3. **Save %** = 1 − resident/full.
4. **Attention exactness** via host-paged online-softmax vs full scores
   (\(\max|\Delta|\)).
5. **Negative control**: no-cache whole-prefix re-forward each decode step
   (peak device memory).

We do **not** claim Transformers are multitape TMs. We transfer the
**block-size balancing** idea from Williams' TIME\[t\] ⊆ SPACE\[√(t log t)\]
result to KV residency policy.

## Mac (Apple M4, 32 GB)

### SmolLM-135M-Instruct-4bit (MLX)

| S | full KV | √S res. | save | notes |
|---:|---:|---:|---:|---|
| 8192 | 188.7 MB | 7.0 MB | **96.3%** | |
| 2048 (attn) | — | — | — | block attn max\|Δ\|=7e-5 |

### Qwen2.5-0.5B-Instruct-4bit (MLX)

| S | full KV | √S res. | save |
|---:|---:|---:|---:|
| 2048 | 25.2 MB | 1.73 MB | 93.1% |
| 4096 | 50.3 MB | 2.54 MB | 94.9% |
| 8192 | 100.7 MB | 3.71 MB | 96.3% |
| 16384 | 201.3 MB | 5.41 MB | **97.3%** |

### Negative: recompute-no-cache peak Metal

| S | full-KV peak | recompute peak | ratio |
|---:|---:|---:|---:|
| 512 | 819 MB | 974 MB | 1.19 |
| 4096 | 2059 MB | 3207 MB | 1.56 |
| 8192 | 3270 MB | 5557 MB | 1.70 |

Whole-prefix re-forward is **not** a peak-memory win.

### Ollama gemma4:e4b process SIZE

SIZE stays ~3.3 GB from 4k–32k context (weights dominate). KV savings
exist but are masked at process RSS until long multi-batch context.

## Modal NVIDIA A100

### Qwen2.5-7B-Instruct (A100-40GB, bf16)

| S | full KV | √S res. | save | peak alloc |
|---:|---:|---:|---:|---:|
| 1024 | 58.7 MB | 5.6 MB | 90.5% | 15.7 GB |
| 4096 | 234.9 MB | 11.9 MB | 94.9% | 17.4 GB |
| 8192 | 469.8 MB | 17.3 MB | 96.3% | 18.9 GB |
| 16384 | 939.5 MB | 25.2 MB | **97.3%** | 21.9 GB |

### Qwen2.5-14B-Instruct (A100-80GB, bf16)

| S | full KV | √S res. | save |
|---:|---:|---:|---:|
| 16384 | **3.22 GB** | **86.5 MB** | **97.3%** (~3.1 GB freed / sequence) |

Save percentages match Mac / 7B to three digits — ratio is |K|/S, not width.

## Interpretation

| Question | Answer |
|---|---|
| Does √S cut **KV storage**? | **Yes**, ~90–97% across 135M–14B |
| Does attention stay exact? | **Yes** (online-softmax; tiny fp error) |
| Does process RSS always shrink? | **Not when weights dominate** |
| Does full recompute save peak mem? | **No — worse** |
| Product surface | `cloud-murakumo` `:kv-policy` `:sqrt-checkpoint` |

## Raw data

See `benchmarks/` for JSON + markdown notes. Paper: `paper/sqrt_space_kv.tex`.

## Addendum — maturity review (2026-07-18)

An external literature review against the 2026 KV-cache research landscape
(eviction: H2O/StreamingLLM/SnapKV/PyramidKV; quantization: KIVI/GEAR/
TurboQuant; depth-share: MiniCache; architecture-native: **MLA**, adopted by
DeepSeek-V2/V3, Kimi K2, GLM-5; offload/tiering: NEO/LMCache/InfLLM/
FlexiCache/KVPR — the class this repo's method actually belongs to) found
the current 90–97% numbers to be **device-resident-byte snapshots only**,
not validated efficiency claims. Full analysis and a maturity ladder
(M0–M6) are recorded in the superproject ADR:

`com-junkawasaki/root` → `90-docs/adr/2607182800-sqrt-space-kv-mla-composability-maturity-review.edn`

### M2 landed (2026-07-18) — real decode-throughput cost of host paging

Item 1 below is no longer open. `gftdcojp/cloud-murakumo`
`scripts/modal_sqrt_kv_throughput_bench.py` measured a real
`.to('cpu')`/`.to('cuda')` round trip on the actual non-resident K/V
tensors on a Modal A100, against a real baseline decode loop
(Qwen2.5-7B-Instruct bf16):

| S | baseline (full-KV) | non-resident KV | measured bandwidth | **re-page every step** | amortized-once (optimistic) |
|--:|---:|---:|---:|---:|---:|
| 8192  | 34.7 tok/s | 452 MB | 13.3 GB/s | **15.9 tok/s (2.18x slower)** | 32.3 tok/s (1.07x slower) |
| 16384 | 43.8 tok/s | 914 MB | 15.0 GB/s | **12.0 tok/s (3.66x slower)** | 37.5 tok/s (1.17x slower) |

The "re-page every step" column is the operationally honest one: exact
online-softmax attention needs the full history for every new query, and
the current `:sqrt-checkpoint` design has no cross-step page cache. Under
that reading, **the slowdown gets worse as S grows** — exactly the
long-context regime this project's own §6.1 says the technique "matters
most" in. The 90–97% storage-save headline, taken alone, is not a free
lunch. Full write-up: `gftdcojp/cloud-murakumo`
`docs/benchmarks/sqrt-kv-throughput-modal-summary.md`.

Still open before this method can be called "validated" (not "proved
wrong", just **not yet measured**):

1. ~~Real decode throughput/latency with host paging~~ — **landed above.**
2. **Head-to-head vs. NEO/InfLLM/LMCache/FlexiCache and vs. H2O/SnapKV**
   using NVIDIA's `kvpress` harness (30+ methods, LongBench-based) — the
   only baseline tested so far is "full KV" and a straw-man "recompute
   everything" policy.
3. **Downstream task quality** (LongBench / RULER / Needle-in-a-Haystack) —
   only a single-point `max|Δ|` numerical check exists today.
4. **MLA composability** — MLA compresses *bytes per token* (the width
   axis); √S residency compresses *how many token slots stay resident*
   (the sequence axis). The save-ratio law (`≈|K|/S`, width-independent)
   suggests these compose multiplicatively in theory, but **no MLA model
   (DeepSeek-V2-Lite etc.) has been benchmarked** — the "what does this add
   on top of MLA" question is an untested hypothesis, not a result. See the
   ADR for the recommended DeepSeek-V2-Lite experiment design.

The Williams STOC 2025 framing should also be read as inspiration for the
checkpoint-stride formula, not a technical dependency — the underlying
checkpoint+recompute pattern predates Williams (Griewank's checkpointing
theory, Chen et al. 2016 "Training Deep Nets with Sublinear Memory Cost",
already cited in `paper/references.bib`).

## Addendum — Qwen3.6-35B-A3B (Modal A100-80GB, 2026-07-10)

User request: "Qwen 3.6 26B A3B". Public open model is **35B-A3B** (total/activated).

| S | full | √S res. | save |
|---:|---:|---:|---:|
| 2048 | 75.4 MB | 36.3 MB | 51.8% |
| 4096 | 117.3 MB | 37.7 MB | 67.9% |
| 8192 | 201.2 MB | 37.7 MB | 81.3% |
| 16384 | 369.0 MB | 42.4 MB | **88.5%** |

- Hybrid linear+full attention: O(1) state floors absolute residency ~36–42 MB.
- Block-stream vs full online-softmax: max|Δ|≈0.016 (bf16 noise) → fidelity held.
- See `benchmarks/sqrt-kv-modal-Qwen3.6-35B-A3B.md`.
