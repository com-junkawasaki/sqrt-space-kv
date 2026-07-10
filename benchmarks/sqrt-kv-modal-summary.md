# Modal √S KV validation summary (2026-07-09)

**App:** `cloud-murakumo-sqrt-kv-bench` · **Script:** `scripts/modal_sqrt_kv_bench.py`  
**Profile:** Modal user `jun784`

## Models

| Model | GPU | bpt (B/tok) | L | H_kv | d |
|-------|-----|------------:|--:|-----:|--:|
| Qwen2.5-7B-Instruct bf16 | A100-40GB | 57 344 | 28 | 4 | 128 |
| Qwen2.5-14B-Instruct bf16 | A100-80GB | 196 608 | 48 | 8 | 128 |

## Save % (identical across scales)

| S | save % |
|--:|-------:|
| 1024 | 90.5 |
| 2048 | 93.1 |
| 4096 | 94.9 |
| 8192 | 96.3 |
| 16384 | **97.3** |

Matches Mac SmolLM-135M / Qwen0.5B — residency ratio depends only on \(b(S)=\lceil\sqrt{S\ln S}\rceil\).

## Absolute KV at S=16384

| Model | full | √S | freed |
|-------|-----:|---:|------:|
| 7B | 0.94 GB | 25 MB | **0.91 GB** |
| 14B | 3.22 GB | 87 MB | **3.13 GB** |

## Recompute-no-cache (7B, S≤2048)

Peak CUDA alloc ratio vs full prefill: **1.04–1.06** (worse).

## Links

- https://modal.com/apps/jun784/main/ap-1jwHgwQyxUMMmcho8mv7Ir (7B)
- https://modal.com/apps/jun784/main/ap-Wkx4qMCP5NKUoaBLOQbKQI (14B)
