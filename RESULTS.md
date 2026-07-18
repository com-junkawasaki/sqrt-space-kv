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

**Value proposition (owner reframing, 2026-07-18): this is a memory-capability
technique, not a speed technique.** The target is running text/video/image
generation models under *limited memory*, accepting slower decode as the
price for fitting a longer context or a bigger model at all. Read the
2.2x–3.7x decode-slowdown numbers below in that light — they are the
accepted cost of the capability, not a defect to be explained away.

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

### M3 landed (2026-07-18) — real end-to-end paged-KV kernel, not isolated

`scripts/modal_sqrt_kv_paged_cache_bench.py` wires the same real transfer
into an actual `transformers.DynamicCache` subclass (`PagedSqrtCache`) so
it runs inline, fused with real multi-layer compute, inside a real
multi-token generate loop. Correctness by construction: the override
returns the framework's real, complete tensors unmodified; the CPU⇄GPU
round trip on the non-resident slice is a real, timed side effect.

At S=8164 (needle-in-haystack, 24 decode tokens): generated token ids
were **byte-identical** between standard `DynamicCache` and
`PagedSqrtCache` (`exact_token_match: true` — confirmed end-to-end, not
just the earlier single-tensor `max|Δ|` spot check), both retrieved the
needle correctly, and the real fused throughput was **36.9 → 11.3 tok/s
(3.27x slower)** — consistent with M2's isolated 2.18x–3.66x estimate.
Full write-up: `gftdcojp/cloud-murakumo`
`docs/benchmarks/sqrt-kv-paged-cache-summary.md`.

### M4 landed (2026-07-18) — kvpress competitive quality benchmark

`scripts/modal_sqrt_kv_kvpress_bench.py` ran a real needle-in-a-haystack
test against NVIDIA kvpress presses (StreamingLLM, Knorm, SnapKV) on the
same model. **At compression_ratio 0.9 — matching sqrt-space-kv's own
90–97% save headline — all three presses scored 0% retrieval accuracy**;
even at a gentler 0.5 ratio, none exceeded 67%. sqrt-space-kv is exact by
construction (100% by design) but pays the M2/M3-measured 2.2x–3.7x
latency tax these presses don't pay. Neither is a free lunch; which cost
is worse is task- and deployment-dependent. Small sample (n=3/condition),
default press configs — not a LongBench-scale evaluation. Full write-up:
`gftdcojp/cloud-murakumo` `docs/benchmarks/sqrt-kv-kvpress-summary.md`.

### M5 landed with a caveat (2026-07-18) — MLA composability partially answered

`scripts/modal_sqrt_kv_mla_bench.py` ran against DeepSeek-V2-Lite-Chat
and got the same save-ratio law (93.1–96.3%, matching dense models to the
first decimal) and a comparable worst-case slowdown (2.83x at S=8192).
**But**: the model's `kv_lora_rank=512` (the true MLA compressed latent
rank) never shows up in the actual cache — the reference HF
`trust_remote_code` implementation caches decompressed per-head tensors
(`[1,16,S,192]`/`[1,16,S,128]`), *larger* than Qwen2.5-7B's dense GQA
cache at the same S. This reference implementation doesn't do the
"absorbed" caching optimization production MLA serving engines (vLLM,
SGLang) use to actually realize MLA's storage win. **The original
composability question — does √S residency compose with a genuinely
MLA-compressed cache — remains open**; that needs hooking into an
absorbed-cache implementation, out of scope for this session. Full
write-up: `gftdcojp/cloud-murakumo` `docs/benchmarks/sqrt-kv-mla-summary.md`.

### M6 partial (2026-07-18) — one real vLLM datapoint, not full parity

No `:sqrt-checkpoint` production kernel exists to compare against vLLM
head-to-head (that gap remains open). `scripts/modal_vllm_baseline_bench.py`
got one real external reference point: vLLM 0.8.5, single request,
S=8192, ~28.25 tok/s decode — in the same range as (not higher than) the
M2/M3 plain-`transformers`-loop baseline (34.7–36.9 tok/s), because vLLM's
advantages come from concurrent-request batching, which a single-request
benchmark doesn't exercise. Useful takeaway: **the M2/M3 baseline wasn't
an artificially weak strawman** — the measured slowdown is real relative
to a production-grade single-request baseline too. Full write-up:
`gftdcojp/cloud-murakumo` `docs/benchmarks/sqrt-kv-vllm-baseline-summary.md`.

Still open: full `:sqrt-checkpoint` production kernel + real vLLM/SGLang
A/B parity comparison (M6), and true MLA-absorbed-cache composability
(M5 residual).

### M6-mac landed (2026-07-18) — real memory-capability test on unified memory

All of M2/M3/M4/M6's numbers above are from discrete NVIDIA GPUs (real
PCIe bus between CPU and GPU memory). Apple Silicon shares ONE physical
memory pool between CPU and GPU — a fundamentally different architecture
that raises a question the GPU numbers can't answer: does moving KV off
the GPU-tracked memory pool free any usable memory there at all?

Real test on this Mac (M4, Qwen2.5-0.5B, all 24 layers, real prefill,
real process RSS via `ps -o rss=` — not a monotonic high-water-mark) at
non-resident payloads of 97MB/395MB/794MB:

- **Moving to numpy ("cpu-move")**: RSS goes *up* by ~the moved size in
  all 3 cases — confirms unified memory means this isn't a real win here.
- **Naive disk paging (`write()`+`del()`)**: mostly doesn't free memory
  either (1/3 cases), except one crossover at the largest scale
  (-180.8MB at 794MB) — plausibly an allocator large-allocation
  mmap-threshold effect, not a mechanism to rely on.
- **`np.memmap` (untouched)**: **+0.0MB RSS at all three scales up to
  794MB.** Touching only 1/24 layers costs roughly that slice, not the
  whole file — genuine on-demand partial residency.

**Actionable finding: `mmap`-backed storage for the non-resident tier is
the architecturally correct mechanism on Apple Silicon, not the
serialize/deserialize pattern used by this project's Modal scripts so
far.** `cloud-murakumo`'s eventual `:sqrt-plus-page` kernel should use
`mmap` specifically for Apple Silicon targets. Full write-up:
`gftdcojp/cloud-murakumo` `docs/benchmarks/sqrt-kv-mac-memory-capability-summary.md`.

### M5-mac (2026-07-18) — same MLA-cache gap confirmed in a second ecosystem

Source-only check (no download — DeepSeek-V2-Lite doesn't fit this Mac's
limited free disk): `mlx-lm`'s own DeepSeek-V2 implementation
(`mlx_lm/models/deepseek_v2.py`) decompresses via `kv_b_proj` **before**
`cache.update_and_fetch(...)`, caching the same non-absorbed per-head form
as the HF reference. This is now a **cross-ecosystem-confirmed** finding,
not a CUDA quirk — both general-purpose reference implementations skip
MLA's absorbed-cache optimization; only vLLM/SGLang (CUDA-only) implement
it. Full write-up: `gftdcojp/cloud-murakumo`
`docs/benchmarks/sqrt-kv-mac-mla-source-inspection.md`.

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
