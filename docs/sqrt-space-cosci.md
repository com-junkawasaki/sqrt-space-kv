# Williams √t-space → LLM inference memory (co-scientist validation)

**Paper:** R. Ryan Williams, *Simulating Time With Square-Root Space* (STOC 2025 Best Paper),
MIT CSAIL — [arXiv:2502.17779](https://arxiv.org/abs/2502.17779),
[PDF](https://people.csail.mit.edu/rrw/time-vs-space.pdf).

**Product surface:** `gftdcojp/cloud-murakumo` LLM serving (`:vllm` / `:mlx-moe`).

**Method:** Google co-scientist / AlphaEvolve-shaped tournament (same shape as
`sha256d-clj` / ADR-2607012300): Generation → Reflection → Ranking → Proximity →
Evolution → Meta-review under a Supervisor. Deterministic, no LLM in the loop.

---

## 1. What the theorem actually says

\[
\mathsf{TIME}[t] \subseteq \mathsf{SPACE}\!\left[\sqrt{t\log t}\right]
\]

for multitape Turing machines. Proof sketch:

1. Make the machine **block-respecting** (Hopcroft–Paul–Valiant): time/tape blocks of length \(b\).
2. Build a **computation graph** of time blocks; unfold it into an implicit **Tree Evaluation**
   instance of height \(h=\Theta(t/b)\), fan-in \(d=O(1)\), node values of \(O(b)\) bits.
3. Evaluate with **Cook–Mertz** (STOC 2024) in space \(O(d\cdot b + h\log(d\cdot b))\).
4. Optimize \(b \approx \sqrt{t\log t}\) → overall space \(O(\sqrt{t\log t})\).

Williams is explicit that this is surprising: the community long expected that \(t\) time
could not be simulated in \(t^{1-\varepsilon}\) space.

## 2. Honest transfer to LLM decode (what we claim / do not claim)

| Claim | Status |
|-------|--------|
| Transformers *are* multitape TMs in the formal sense of the theorem | **No** |
| Autoregressive decode of length \(S\) is a long sequential computation whose *working set* can be traded against recompute | **Yes (engineering)** |
| Optimal checkpoint block \(b\sim\sqrt{S\log S}\) balances store vs recompute the same way Williams balances Tree Evaluation parameters | **Yes (analogy + measured cost model)** |
| Bit-identical full-context attention if recompute uses the same kernels | **Yes (for exact strategies)** |
| Sliding window is a Williams consequence | **No** — it is approximate attention |

The usable design pattern is the same as **gradient checkpointing** and **KV recompute**:

```text
store O(b) content at checkpoints,
recompute a block of size b on miss,
choose b ~ √(S log S) to minimize peak memory under a recompute budget.
```

## 3. Strategy catalog (closed gene pool)

| Strategy | KV peak | Weights | Fidelity | Role |
|----------|---------|---------|----------|------|
| `:full-kv` | \(\Theta(S)\) | full | exact | baseline |
| `:sliding-window` | \(\Theta(W)\) | full | **approximate** | hard-fail under exact gate |
| `:sqrt-checkpoint` | \(\Theta(\sqrt{S\log S})\) | full | exact+recompute | Williams transfer (dense) |
| `:block-respecting` | Cook–Mertz proxy | full | exact+recompute | closer theory, heavier |
| `:expert-page` | \(\Theta(S)\) | active experts | exact-for-MoE | mlx-moe axis (weights) |
| `:sqrt-plus-page` | \(\Theta(\sqrt{S\log S})\) | active experts | exact+recompute | **compose both axes** |

## 4. Co-scientist loop

```text
Generation   enumerate strategy × α × window × expert-frac
Reflection   hard gates (disqualify, never soft-score):
               - approximate fidelity rejected when require-exact?
               - kv-cells ≤ full-KV on all probe lengths
               - √S claimers must be o(S) (≤ S^0.75 check)
               - :expert-page must not pretend to cut KV
Ranking      Elo on work-cost = kv-cells + α·recompute + β·S·weight-frac
Proximity    cluster within 2% work
Evolution    elitism + crossover + mutation (reintroduce dropped genes)
Meta-review  product recommendation + theory note
```

Run:

```sh
clj -M:cosci 3
clj -M:sqrt-space 65536
clj -M:kv-policy minimax-m27 65536
```

## 5. Measured tournament result (2026-07-09)

Deterministic run, 3 generations, probe lengths up to 65 536:

| Metric | Value |
|--------|-------|
| **Winner** | `:sqrt-plus-page` |
| Elo (final) | ~1635 |
| Fidelity | `exact-with-recompute` |
| Williams transfer? | yes |
| Sliding-window | **disqualified** (approximate) |

At \(S=65536\):

| Strategy | kv-cells | vs full-KV | recompute/tok |
|----------|----------|------------|---------------|
| full-kv | 65 536 | 1.00 | 0 |
| sqrt-checkpoint | **930** | **0.014** | 426.5 |
| block-respecting | 4 909 | 0.075 | 426.5 |
| sqrt-plus-page | **930** | **0.014** + weight paging | 426.5 |

Williams parameters: \(b=853\), \(h=77\), Cook–Mertz space proxy ≈ 4909 cells (~7.5% of \(S\)).

**Product recommendation (meta-review):** compose mlx-moe expert paging with √S KV
checkpoints on Apple-unified fleet; for dense vLLM paths use `:sqrt-checkpoint`.

## 6. Wiring in cloud-murakumo

`resources/murakumo.edn`:

- `minimax-m27` / `kimi-k27` → `:kv-policy {:strategy :sqrt-checkpoint ...}`
- `qwen3-coder-next-moe` → `:kv-policy {:strategy :sqrt-plus-page ...}`

`vllm.cljc` / `mlx_moe.cljc` only inject flags when `:kv-policy` is present (legacy serve
maps stay bit-identical).

Emitted intent flags (kernel recompute is a follow-up):

- vLLM: `--max-num-batched-tokens <b>` + `--enable-prefix-caching`
- mlx-moe: `--profile sqrt-kv` (unless operator sets `:profile`)

## 7. What is verified vs follow-up

**Verified now (cost model + tournament + CLI + tests):**

- Sublinear space scaling of Williams-optimal \(b\)
- Exact strategies beat full-KV on peak kv-cells by ~70× at 64k
- Co-scientist hard gates kill approximate long-context
- Serve command builders honor opt-in policies

**Verified on real models (Apple M4, 2026-07-09):**

See [`benchmarks/sqrt-kv-mac-m4-validation.md`](benchmarks/sqrt-kv-mac-m4-validation.md).

- MLX **Qwen2.5-0.5B**: real prefill KV is linear (~12 KB/tok); √S residency saves
  **93–97%** of KV bytes (e.g. 201 MB→5.4 MB at \(S=16384\)).
- **No-cache recompute** makes Metal peak **worse** (activations).
- Ollama **gemma4:e4b** runs to 65k ctx; process SIZE weight-dominated (3.3→3.4 GB).

**Follow-up (serve kernel + fleet A/B):**

1. Implement true block-KV pager for vLLM / mlx path and measure tokens/s vs VRAM on
   H100 and asher (absolute GB matter more than on 0.5B/8B-Q4).
2. Replace `--max-num-batched-tokens` heuristic with engine-native checkpoint API.
3. Feed live run-ledger numbers into `cost.cljc` BMC gate for √S vs full-KV fleet ¥/tok.
## 8. References

- Williams 2025, STOC Best Paper — arXiv:2502.17779
- Cook & Mertz 2024, Tree Evaluation in nearly linear space
- Hopcroft, Paul, Valiant 1975/77 — classic \(O(t/\log t)\) space simulation
- Chen et al. gradient checkpointing (training analogue of recompute-for-space)
- ADR-2607012300 sha256d co-scientist tournament shape
- ADR-2607092800 this decision
