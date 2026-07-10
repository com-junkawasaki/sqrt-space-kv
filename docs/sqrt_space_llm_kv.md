# Simulating Autoregressive Memory with Square-Root Space:  
# From Williams' Time–Space Theorem to Practical KV-Cache Residency

**Status:** Draft for arXiv (cs.LG / cs.CC / cs.CL) and Hugging Face Papers  
**Date:** 2026-07-09  
**Code:** `gftdcojp/cloud-murakumo` — `scripts/bench_sqrt_kv_mac.py`, `scripts/modal_sqrt_kv_bench.py`  
**Related theory:** R. Ryan Williams, *Simulating Time With Square-Root Space*, STOC 2025 (Best Paper); arXiv:2502.17779

---

## Abstract

Williams (STOC 2025) proved that every multitape Turing machine running in time \(t\) can be simulated in space \(O(\sqrt{t\log t})\), via a reduction to Tree Evaluation and the Cook–Mertz space-efficient algorithm. We study a *transfer* of this parameter-balancing idea to the dominant sequential memory structure in modern LLM serving: the key–value (KV) cache of length \(S\).

We do **not** claim that Transformers are multitape TMs. Instead we isolate the same design pattern that already appears in gradient checkpointing: partition the sequence into blocks of length \(b \approx \sqrt{S\log S}\), retain only \(O(S/b + b)\) boundary and active state, and recompute or page the remainder. We implement this *√S residency* policy over real model caches and measure bytes retained.

**Findings.** Across Apple Silicon (SmolLM-135M, Qwen2.5-0.5B, Ollama gemma4-E4B) and Modal NVIDIA GPUs (7B-class models; see §5), full KV storage grows linearly in \(S\) at a constant bytes-per-token set by architecture. Retaining only Williams-optimal checkpoints plus one active block reduces **KV storage by 90–97%** at \(S\in[1\text{k},16\text{k}]\) while preserving an exact host-paged attention path (online softmax). Naive “delete the cache and re-forward the whole prefix every token” *increases* peak device memory (activations dominate). Whole-process RSS often remains weight-dominated until long context or large multi-batch KV.

We release a co-scientist-shaped closed tournament of memory policies (`cloud-murakumo.cosci`), pure cost models, Mac and Modal runners, and this draft for arXiv / Hugging Face.

---

## 1. Introduction

Autoregressive Transformers store key/value tensors for every past token so that each new token can attend in \(O(1)\) relative cache reads. For sequence length \(S\), layers \(L\), KV heads \(H_{\mathrm{kv}}\), and head dimension \(d\), the cache occupies

\[
\mathrm{KVBytes}(S) \;=\; 2 \cdot L \cdot H_{\mathrm{kv}} \cdot d \cdot S \cdot \mathrm{sizeof}(\mathrm{dtype}),
\]

i.e. \(\Theta(S)\) for fixed model shape. Long-context and multi-tenant serving make this term the primary memory wall after weights.

Independently, complexity theory recently improved the generic simulation of time by space from Hopcroft–Paul–Valiant’s \(O(t/\log t)\) (1975/77) to Williams’ \(O(\sqrt{t\log t})\) (STOC 2025). The proof reduces a length-\(t\) computation to Tree Evaluation of height \(h=\Theta(t/b)\) with node values of size \(O(b)\), then applies Cook–Mertz (STOC 2024) space \(O(d\cdot b + h\log(d\cdot b))\). Optimizing \(b\sim\sqrt{t\log t}\) yields square-root space.

**Question.** Does the *same block-size optimization* yield a measurable, architecture-faithful reduction in LLM decode memory when applied to KV caches?

**Contributions.**

1. An honest transfer map from Williams’ parameters \((t,b,h)\) to LLM parameters \((S,b,h)\) with fidelity labels (exact / exact-with-recompute / approximate).
2. A closed co-scientist tournament over memory policies (hard gates + Elo on work-cost), selecting `:sqrt-checkpoint` and `:sqrt-plus-page` for product wiring in `cloud-murakumo`.
3. Empirical validation on real models: Mac Metal (135M–8B) and Modal CUDA (7B+), measuring KV *nbytes* and residency under Williams-optimal \(b\).
4. Negative results: whole-prefix recompute is not a peak-memory win; process-level RSS can mask KV savings when weights dominate.

---

## 2. Background

### 2.1 Williams’ square-root space simulation

**Theorem (Williams 2025, informal).** For all \(t(n)\ge n\),

\[
\mathsf{TIME}[t] \;\subseteq\; \mathsf{SPACE}\!\big[\sqrt{t\log t}\big]
\]

on multitape Turing machines. The simulation (i) makes the machine block-respecting with block length \(b\), (ii) builds a computation graph of time blocks, (iii) unfolds it into an implicit Tree Evaluation instance of height \(O(t/b)\), and (iv) evaluates it with Cook–Mertz in space \(O(d\cdot b + h\log(d\cdot b))\). Setting \(b=\Theta(\sqrt{t\log t})\) balances the two terms.

### 2.2 LLM KV caches

Standard decode keeps \(\mathrm{past\_key\_values}\) of length \(S\). Practical systems already trade memory for recompute or approximation: gradient checkpointing (training), PagedAttention (paging), multi-query/GQA (shrinking \(H_{\mathrm{kv}}\)), sliding-window attention (approximation), and offloading to host/SSD.

### 2.3 What we transfer—and what we do not

| Theory object | Serving analogue |
|---------------|------------------|
| Time bound \(t\) | Sequence length \(S\) |
| Block length \(b\) | Checkpoint stride / active block |
| Tree height \(h\sim t/b\) | Number of checkpoints |
| Cook–Mertz multipoint evaluation | Optional deeper recomputation schedule |
| Multitape TM | **Not claimed** for Transformers |

The transfer is *parameter balancing + block recompute/paging*, not a complexity-theoretic embedding of attention into multitape TMs.

---

## 3. Method: √S KV residency

### 3.1 Optimal block size

Mirror Williams’ warm-up optimum (constants absorbed by \(\ln\)):

\[
b(S) \;=\; \Big\lceil \sqrt{S \ln S} \Big\rceil, \qquad
h(S) \;=\; \Big\lceil S / b(S) \Big\rceil.
\]

### 3.2 Resident set

After a real model prefill producing full K/V of length \(S\), retain only indices

\[
\mathcal{K}
\;=\;
\underbrace{\{0,b,2b,\ldots\}}_{\text{boundary checkpoints}}
\;\cup\;
\underbrace{\{S-b,\ldots,S-1\}}_{\text{active block}}.
\]

Report \(\mathrm{nbytes}(\mathcal{K})/\mathrm{nbytes}([0..S))\). This measures **storage residency** under a Williams-sized working set. Decode may page non-resident blocks from host and combine them with **online softmax** (exact attention without materializing \(S\times S\) scores for a single query).

### 3.3 Baselines

| Policy | Fidelity | Storage |
|--------|----------|---------|
| Full KV | exact | \(\Theta(S)\) |
| √S residency | exact-with-page/recompute | \(\Theta(b+h)=\tilde\Theta(\sqrt{S})\) |
| Sliding window \(W\) | approximate | \(\Theta(W)\) |
| No-cache re-forward | exact | no KV, but large activations |

### 3.4 Co-scientist tournament (software)

A deterministic Generation → Reflection → Elo Ranking → Evolution loop over a closed gene pool of policies (`cloud-murakumo.cosci`) selects memory-first strategies under hard exactness gates. Sliding-window is disqualified when exact long-context is required. Winner on the closed pool: `:sqrt-plus-page` (√S KV + MoE expert paging).

---

## 4. Experimental setup

### 4.1 Apple Silicon (local)

| Model | Stack | Notes |
|-------|-------|-------|
| SmolLM-135M-Instruct-4bit | MLX | Smallest real Transformer |
| Qwen2.5-0.5B-Instruct-4bit | MLX | Instrumented `past`/`state` nbytes |
| gemma4:e4b (8B Q4) | Ollama | Process-level SIZE to 65k ctx |

### 4.2 Modal (NVIDIA)

| Model | GPU | Stack |
|-------|-----|-------|
| Qwen2.5-7B-Instruct | A100 40GB | HF Transformers bf16, `use_cache=True` |
| *(optional)* Qwen2.5-14B-Instruct | A100-80GB | same |
| *(optional)* Llama-3.1-8B-Instruct | A100 | same |

Runner: `scripts/modal_sqrt_kv_bench.py` (Modal app `cloud-murakumo-sqrt-kv-bench`).

### 4.3 Metrics

- **full_kv_mb / keep_kv_mb / save_pct** — primary
- **bytes per token (bpt)** — architecture constant
- **CUDA/Metal peak allocation** — secondary (weights + activations)
- **Exactness** — \(\max|\Delta|\) between full and block-stream attention on real K/V

---

## 5. Results

### 5.1 Mac — SmolLM-135M

Architecture: \(L=30\), \(H_{\mathrm{kv}}=3\), \(d=64\) → \(\mathrm{bpt}=23040\).

| \(S\) | full KV | √S resident | save |
|------:|--------:|------------:|-----:|
| 1024 | 23.6 MB | 2.2 MB | **90.5%** |
| 2048 | 47.2 MB | 3.3 MB | **93.1%** |
| 4096 | 94.4 MB | 4.8 MB | **94.9%** |
| 8192 | 188.7 MB | 7.0 MB | **96.3%** |

Host-paged block attention at \(S=2048\): \(\max|\Delta|=7\cdot10^{-5}\).

### 5.2 Mac — Qwen2.5-0.5B

Architecture: \(L=24\), \(H_{\mathrm{kv}}=2\), \(d=64\) → \(\mathrm{bpt}=12288\).

| \(S\) | full KV | √S resident | save |
|------:|--------:|------------:|-----:|
| 2048 | 25.2 MB | 1.7 MB | **93.1%** |
| 4096 | 50.3 MB | 2.5 MB | **94.9%** |
| 8192 | 100.7 MB | 3.7 MB | **96.3%** |
| 16384 | 201.3 MB | 5.4 MB | **97.3%** |

### 5.3 Mac — gemma4-E4B (process view)

Real generation to \(S=65536\). `ollama ps` SIZE moves only 3.3→3.4 GB (weight-dominated). KV savings exist but are **masked** at process level for this quantized 8B footprint.

### 5.4 Modal — Qwen2.5-7B-Instruct (A100-40GB, bf16)

**Run:** `modal run scripts/modal_sqrt_kv_bench.py --model-id Qwen/Qwen2.5-7B-Instruct --gpu A100`  
**GPU:** NVIDIA A100-SXM4-40GB · load 96 s · arch \(L=28\), \(H=28\), \(H_{\mathrm{kv}}=4\), \(d=128\) → \(\mathrm{bpt}=57344\).

| \(S\) | full KV | √S resident | **save** | CUDA peak alloc |
|------:|--------:|------------:|---------:|----------------:|
| 1024 | 58.7 MB | 5.6 MB | **90.5%** | 15.7 GB |
| 2048 | 117.4 MB | 8.1 MB | **93.1%** | 16.4 GB |
| 4096 | 234.9 MB | 11.9 MB | **94.9%** | 17.4 GB |
| 8192 | 469.8 MB | 17.3 MB | **96.3%** | 18.9 GB |
| 16384 | **939.5 MB** | **25.2 MB** | **97.3%** | 21.9 GB |

Save percentages match Mac SmolLM/Qwen0.5B to three digits—the ratio is \((|\mathcal{K}|)/S\), independent of width. Absolute KV at \(S=16\text{k}\) is already ~0.94 GB full vs 25 MB √S (**~0.91 GB saved** on the KV budget alone).

No-cache recompute at \(S\le2048\): peak alloc ratio 1.04–1.06 vs full prefill (still worse).

JSON: `docs/benchmarks/sqrt-kv-modal-Qwen__Qwen2.5-7B-Instruct.json`.

### 5.5 Modal — Qwen2.5-14B-Instruct (A100-80GB, bf16)

**Run:** `--gpu A100-80GB` · load 169 s · arch \(L=48\), \(H=40\), \(H_{\mathrm{kv}}=8\), \(d=128\) → \(\mathrm{bpt}=196608\).

| \(S\) | full KV | √S resident | **save** | CUDA peak alloc |
|------:|--------:|------------:|---------:|----------------:|
| 1024 | 201 MB | 19.1 MB | **90.5%** | 30.2 GB |
| 4096 | 805 MB | 40.7 MB | **94.9%** | 31.8 GB |
| 8192 | 1.61 GB | 59.4 MB | **96.3%** | 33.9 GB |
| 16384 | **3.22 GB** | **86.5 MB** | **97.3%** | 38.1 GB |

At \(S=16\text{k}\), √S residency frees **~3.1 GB** of KV on a single sequence—material for multi-tenant packing on one 80 GB GPU. Save % identical to 7B/Mac (universal in \(S\)).

JSON: `docs/benchmarks/sqrt-kv-modal-Qwen__Qwen2.5-14B-Instruct.json`.
### 5.6 Negative result: no-cache recompute

On Mac and Modal (short \(S\)), peak device memory for whole-prefix re-forward each step is **higher** than full-KV prefill peak (activation tensors of length \(S\)). Williams-style recomputation must be **blocked**, not whole-trace.
---

## 6. Discussion

### 6.1 When √S matters operationally

- **Long context** (\(S\gtrsim 32\text{k}\)–\(128\text{k}\)) and multi-request batching.
- **High \(\mathrm{bpt}\)** (many layers, MHA not GQA, fp16 KV).
- **Unified memory** edge devices where KV and weights share a pool (Mac), *if* KV is a large fraction of the pool.

### 6.2 Relation to PagedAttention / FlashAttention

PagedAttention manages fragmentation of a still-\(\Theta(S)\) cache. FlashAttention tiles for IO without reducing asymptotic stored KV. √S residency reduces **how many token slots are stored on-device**, at the cost of paging/recompute bandwidth—the classical Williams tradeoff.

### 6.3 Complexity caveat

A proof that Transformer decode lies in \(\mathsf{SPACE}[\tilde O(\sqrt{S})]\) in the multitape sense would require a full TM accounting of arithmetic and random-access issues Williams flags for non-oblivious RAM. We explicitly **do not** claim that theorem. We claim an engineering reduction of KV *storage* with Williams-optimal block size and measured ratios.

### 6.4 Product integration

`cloud-murakumo` declares `:serve :kv-policy` (`:sqrt-checkpoint` for dense vLLM, `:sqrt-plus-page` for mlx-moe). Serve extras currently express intent; production kernels remain follow-up.

---

## 7. Related work

- Williams (STOC 2025); Cook–Mertz Tree Evaluation (STOC 2024); Hopcroft–Paul–Valiant (1975/77).
- Gradient checkpointing (Chen et al.); FlashAttention (Dao et al.); PagedAttention / vLLM; MQA/GQA; streaming / sink attention; MoE expert offload (mlx-moe).

---

## 8. Conclusion

Williams’ square-root space simulation supplies a *principled* block size for LLM KV residency. On real models from 135M to multi-billion parameters, retaining only \(\tilde O(\sqrt{S})\) token slots cuts KV storage by roughly an order of magnitude, while naive full recompute fails as a peak-memory strategy. Absolute impact grows with context and KV width—the regime of cloud GPU inference that `cloud-murakumo` targets.

---

## Reproducibility

```bash
# Mac (MLX)
python scripts/bench_sqrt_kv_mac.py \
  --model mlx-community/SmolLM-135M-Instruct-4bit

# Modal (CUDA)
modal run scripts/modal_sqrt_kv_bench.py \
  --model-id Qwen/Qwen2.5-7B-Instruct --gpu A100

# Cost model / co-scientist (no GPU)
clj -M:cosci 3
clj -M:sqrt-space 65536
```

Artifacts: `docs/benchmarks/sqrt-kv-*.json`, `docs/benchmarks/sqrt-kv-modal-*.json`.

---

## References

1. R. Ryan Williams. *Simulating Time With Square-Root Space*. STOC 2025. arXiv:2502.17779.  
2. J. Cook, I. Mertz. Tree Evaluation papers, culminating STOC 2024.  
3. J. Hopcroft, W. Paul, L. Valiant. On time versus space. JACM 1977.  
4. T. Dao et al. FlashAttention.  
5. W. Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM).  
6. T. Chen et al. Training Deep Nets with Sublinear Memory Cost (gradient checkpointing).  

---

## Appendix A: Fidelity and ethics

This draft is an empirical systems note grounded in a published complexity theorem. It does not claim a new complexity lower/upper bound for neural sequence models. Benchmarks use public model weights under their respective licenses. Modal GPU spend is attributed to the operator’s Modal workspace.

## Appendix B: Hugging Face Papers blurb (≤2000 chars)

We connect Williams’ STOC 2025 result—time-\(t\) multitape TMs simulate in \(O(\sqrt{t\log t})\) space—to LLM serving memory. Treating decode length \(S\) as “time,” we set KV checkpoint stride \(b\approx\sqrt{S\log S}\) and measure *residency*: keep only boundary checkpoints plus one active block. On Mac (SmolLM-135M, Qwen2.5-0.5B) and Modal GPUs (7B-class), full KV is linear in \(S\); √S residency saves **90–97%** of KV bytes at 1k–16k context, with exact host-paged attention. Deleting the cache and re-forwarding the whole prefix every step *worsens* peak GPU/Metal memory. Code and JSON benchmarks: `cloud-murakumo` (`scripts/modal_sqrt_kv_bench.py`, `scripts/bench_sqrt_kv_mac.py`). Not a TM simulation of Transformers—an engineering transfer of Williams’ block-size optimization to KV caches.
