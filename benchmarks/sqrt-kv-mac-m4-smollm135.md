# √S KV on smaller model: SmolLM-135M (Apple M4)

**Model:** `mlx-community/SmolLM-135M-Instruct-4bit`  
**Arch:** 30 layers, 9 heads, 3 KV heads, head_dim 64, hidden 576  
**Host:** Apple M4 / 32 GB · 2026-07-09  
**JSON:** `sqrt-kv-mac-m4-smollm135.json`

Compared to prior Qwen2.5-0.5B run: ~3.7× fewer params, still a full Transformer decode path.

## KV linear growth (~23 KB/token all layers)

| \(S\) | full KV | √theory (h+b) | theory save |
|------:|--------:|--------------:|------------:|
| 256 | 5.9 MB | 1.0 MB | 82% |
| 1024 | 23.6 MB | 2.3 MB | 90% |
| 4096 | 94.4 MB | 4.8 MB | 95% |
| 8192 | 188.7 MB | 7.0 MB | 96% |

Bytes/token = 23040 (30 × 2 × 3 × 64 × 2 B).

## √S residency (measured)

| \(S\) | full | √resident | **SAVE** |
|------:|-----:|----------:|---------:|
| 1024 | 23.6 MB | 2.2 MB | **90.5%** |
| 2048 | 47.2 MB | 3.3 MB | **93.1%** |
| 4096 | 94.4 MB | 4.8 MB | **94.9%** |
| 8192 | 188.7 MB | 7.0 MB | **96.3%** |

Same order as Qwen0.5B (93–97%). Ratio is architecture-agnostic in \(S\); absolute MB is smaller because KV width is smaller.

## Peak Metal: recompute-no-cache still not a win

| \(S\) | full-KV peak | recompute peak | ratio |
|------:|-------------:|---------------:|------:|
| 512 | 369 MB | 384 MB | 1.04 |
| 1024 | 625 MB | 657 MB | 1.05 |
| 2048 | 1098 MB | 1148 MB | 1.05 |

## Exactness

Host-paged online-softmax block attention (layer0, last query), \(S=2048\), \(b=125\):  
\(\max|\Delta| = 7\times10^{-5}\) → **exact**.

## Repro

```bash
source /tmp/mlx-sqrt-bench312/bin/activate   # py3.12 + mlx-lm
python scripts/bench_sqrt_kv_mac.py \
  --model mlx-community/SmolLM-135M-Instruct-4bit \
  --lengths 256,512,1024,2048,4096 \
  --new-tokens 2 \
  --out docs/benchmarks/sqrt-kv-mac-m4-smollm135-gen.json
```
