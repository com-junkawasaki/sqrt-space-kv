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
