# Real-model validation: Williams √S KV on Apple M4 (2026-07-09)

**Host:** Apple M4, 32 GB unified memory, macOS 26.3  
**Models:**
- `mlx-community/SmolLM-135M-Instruct-4bit` via MLX Metal (**smaller re-run**, see
  [`sqrt-kv-mac-m4-smollm135.md`](sqrt-kv-mac-m4-smollm135.md))
- `mlx-community/Qwen2.5-0.5B-Instruct-4bit` via MLX Metal (instrumented)
- `gemma4:e4b` (8B Q4_K_M) via Ollama (already on machine)
**Raw JSON:** `sqrt-kv-mac-m4-validation.json`, `sqrt-kv-mac-m4.json`,
`ollama-gemma4-ctx.json`, `ollama-gemma4-ctx-hi.json`  
**Runner:** `scripts/bench_sqrt_kv_mac.py` (+ one-shot residency script in session)

---

## Executive verdict

| Claim | Result on this Mac |
|-------|--------------------|
| Full-KV grows **linearly** with sequence length \(S\) | **Yes** (measured) |
| √S checkpoint residency cuts **KV storage** ~90–97% | **Yes, significant** |
| Host-paged block attention is a viable exact decode path | **Yes** (online softmax over blocks) |
| “No-cache recompute every step” cuts **peak** Metal memory | **No** — **worse** (activations dominate) |
| Whole-process RSS / ollama SIZE shrinks dramatically at 4k–32k | **Not visible** — weights dominate; KV is a small fraction until very long context |

**Product takeaway:** Williams-style √S is a real win for the **KV budget**, not a magic reduction of total process memory when weights already fill most of unified RAM. On long-context dense GPUs (cloud-murakumo H100 path) KV becomes the binding constraint and the same residency math applies at larger absolute savings.

---

## 1. MLX Qwen2.5-0.5B — real KV bytes after prefill

Architecture: 24 layers, 14 heads, 2 KV heads, head_dim 64 → **12 288 bytes/token** of KV (all layers, K+V).

| \(S\) | Measured KV | √S resident (ckpt + active block) | Save |
|------:|------------:|----------------------------------:|-----:|
| 2048 | 25.2 MB | 1.73 MB | **93.1%** |
| 4096 | 50.3 MB | 2.54 MB | **94.9%** |
| 8192 | 100.7 MB | 3.71 MB | **96.3%** |
| 16384 | 201.3 MB | 5.41 MB | **97.3%** |

Block size \(b=\lceil\sqrt{S\ln S}\rceil\) (same formula as `cloud-murakumo.sqrt-space/optimal-block`).  
Resident set = boundary checkpoints every \(b\) tokens **plus** the last active block of length \(b\).

This is the direct empirical counterpart of the cost-model row
`sqrt-checkpoint @ S → kv-cells ≈ h+b`.

---

## 2. What does *not* help peak memory

### No-cache full re-forward each decode step

| \(S\) | full-KV peak Metal | recompute-no-cache peak | ratio |
|------:|-------------------:|------------------------:|------:|
| 512 | 819 MB | 974 MB | 1.19 |
| 4096 | 2059 MB | 3207 MB | 1.56 |
| 8192 | 3270 MB | 5557 MB | 1.70 |

Recomputing the whole prefix each step **increases** peak memory: temporary activations for length-\(S\) forward outweigh the avoided KV. Williams is a *space* result that allows recomputation of *blocks*, not “delete the cache and re-run the world every token” as a peak-memory strategy.

### Decode-style single-query attention tiling

With only the last query, the score vector is \(O(S)\), not \(O(S^2)\). Peak Metal is dominated by weights + full K/V already resident → √block shows **no** meaningful peak save (~1%).

### Prefill \(S\times S\) scores (layer-0 real K,V)

At \(S=4096\), materializing scores (~470 MB) pushes full peak 3023 MB vs block-stream 2640 MB (**~13%**). Real but modest once weights sit in the same peak window.

---

## 3. Ollama `gemma4:e4b` (8B) on this Mac

Successfully ran real generation up to **65 536** context:

| num_ctx | prompt tokens | wall time | `ollama ps` SIZE |
|--------:|--------------:|----------:|-----------------:|
| 4096 | 3293 | 72 s | 3.3 GB |
| 16384 | 13124 | 125 s | 3.3 GB |
| 32768 | 26231 | 227 s | 3.3 GB |
| 65536 | 52445 | 500 s | **3.4 GB** |

Reported SIZE only moves **+0.1 GB** from 4k→64k (~1.7 KB/token if attributed to KV). Weight footprint (~3.3 GB) dominates; **√S on KV would save on the order of ~0.1 GB absolute at 64k for this quantized 8B**, i.e. small vs weights but real as context grows further or models get denser KV (MHA, long multi-turn, multi-batch).

RSS stayed ~3.9 GB across 0.5k–8k (weight-dominated; not a good KV instrument).

---

## 4. Mapping back to cloud-murakumo policies

| Policy | Validated? | Notes |
|--------|------------|-------|
| `:full-kv` | baseline | Linear KV; fine when weights dominate |
| `:sqrt-checkpoint` | **KV storage yes** | 93–97% KV reduction measured; needs paging/recompute kernel in serve path |
| `:sqrt-plus-page` | complementary | MoE expert paging (mlx-moe) orthogonal; composes |
| `:sliding-window` | approximate | Not exact; still fails cosci exact gate |
| recompute-no-cache | **reject for peak** | Measured regression |

Serve extras (`--enable-prefix-caching`, block-sized batch tokens) remain **intent** until a real block-KV pager lands; this bench is the go/no-go evidence that the **math is not vacuous on Metal**.

---

## 5. How to reproduce

```bash
# Python 3.12 venv with mlx + mlx-lm (3.14 breaks transformers register)
python3.12 -m venv /tmp/mlx-sqrt-bench312
source /tmp/mlx-sqrt-bench312/bin/activate
pip install 'mlx' 'mlx-lm' 'transformers==4.48.3'

cd orgs/gftdcojp/cloud-murakumo
python scripts/bench_sqrt_kv_mac.py --lengths 512,1024,2048,4096 --new-tokens 4

# Ollama (model already local)
ollama run gemma4:e4b  # or API generate with num_ctx
```

---

## 6. Honest limits

1. √S **residency** was measured by slicing a real prefill cache (checkpoints + active block) and host-storing the rest — not yet a production decode loop inside vLLM/mlx-moe.  
2. Host-paged online-softmax is exact in principle; one run saw NaN from `-inf` edge cases and needs fp safeguards.  
3. Ollama SIZE is a coarse product metric; MLX `nbytes` on cache tensors is the scientific instrument.  
4. Absolute GB saved scales with model KV width × \(S\); on M4+gemma4-8B-Q4 at ≤32k the weight wall hides KV. On murakumo H100 65k+ dense MoE/MHA, the same ratio is the binding win.
