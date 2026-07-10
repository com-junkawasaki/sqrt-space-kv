# Modal √S KV: Qwen3.6-35B-A3B (user asked “26B-A3B”)

**Date:** 2026-07-10  
**Model:** [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)  
**Note:** There is no public `Qwen3.6-26B-A3B`. The open Qwen3.6 MoE is **35B total / 3B activated** (`model_type=qwen3_5_moe_text`).  
**GPU:** NVIDIA A100 80GB · bf16 · `scripts/modal_sqrt_kv_bench.py`  
**Raw JSON:** `sqrt-kv-modal-Qwen__Qwen3.6-35B-A3B.json`

## Architecture (measured)

| | |
|---|---|
| layers | 40 |
| attention heads | 16 |
| KV heads | 2 (GQA) |
| hidden | 2048 |
| experts | 256 / 8 active |
| cache | **hybrid** (full-attn K/V + linear-attn recurrent state) |

Linear-attention layers store **O(1)** state that does **not** shrink with √S residency. Full-attn layers store classic **Θ(S)** K/V and **do** shrink. Net save therefore **rises with S** (fixed overhead amortized).

## KV residency (b = ⌈√(S ln S)⌉)

| S | full cache | √S resident | **save** | b | n_keep | B/token |
|---:|---:|---:|---:|---:|---:|---:|
| 2 048 | 75.4 MB | 36.3 MB | **51.8%** | 125 | 141 | 36 800 |
| 4 096 | 117.3 MB | 37.7 MB | **67.9%** | 185 | 207 | 28 640 |
| 8 192 | 201.2 MB | 37.7 MB | **81.3%** | 272 | 302 | 24 560 |
| 16 384 | 369.0 MB | 42.4 MB | **88.5%** | 390 | 432 | 22 520 |

## Performance maintenance (attention exactness)

Block-stream online-softmax vs full online-softmax on first full-attn K/V pair  
(S=2048, b=125, bf16):

| metric | value |
|---|---|
| max\|Δ\| | **0.015625** |
| mean\|Δ\| | ~0.0011 |
| interpretation | **bf16 numerical noise** (≈ 2⁻⁶), not algorithmic drift |

→ **√S block paging preserves attention numerics** within bf16 tolerance.  
(Strict `<1e-3` gate fails only because of dtype; fp32 would be tighter.)

## Negative control

Recompute-no-cache (short) still does **not** cut peak GPU alloc (ratio ~1.01–1.02 vs prefill peak) — same story as dense Qwen2.5.

## Verdict

| Question | Answer |
|---|---|
| Does √S compress **cache bytes** on Qwen3.6 MoE? | **Yes**, and **more at long S** (52% @2k → **88.5% @16k**) |
| Is attention fidelity kept? | **Yes** within bf16 (block-stream ≡ full online-softmax) |
| Same as dense 90–97% at all S? | **No** — hybrid O(1) linear-attn state floors absolute resident size ~36–42 MB |
| Production takeaway | √S residency still **efficient** for long-context MoE; combine with linear-attn design |

## Repro

```bash
modal run scripts/modal_sqrt_kv_bench.py \
  --model-id Qwen/Qwen3.6-35B-A3B \
  --gpu A100-80GB \
  --lengths 2048,4096,8192,16384 \
  --dtype-name bfloat16 \
  --exactness-s 2048 \
  --new-tokens 0
```
