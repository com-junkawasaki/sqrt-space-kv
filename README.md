# sqrt-space-kv

**A memory-capability technique for LLM decode, not a speed technique —
validated end-to-end on real hardware (Modal A100 + Apple Silicon).**

## Impact, in one table

| Question | Real answer |
|---|---|
| Does it actually unlock capability? | **Yes, measured.** GLM-4-9B-Chat-1M (1,048,576-token native context) can only use **6.1%** of that on a 16GB Mac without this — with √S residency + `mmap` paging, the resident set stays tens of MB regardless of S. |
| Is it actually correct? | **Yes, proven end-to-end.** A real `Cache` subclass fused into a real multi-token `generate()` loop produced **byte-identical** tokens vs. standard full-KV decode — not just a single-tensor numerical spot check. |
| What does it cost? | **2.2×–3.7× slower decode**, measured two independent ways (isolated transfer + fused generation), worsening with longer context. This is the accepted price of the capability, not a defect. |
| Does "offloading" even work on unified memory? | **We tested it — mostly no.** On Apple Silicon, moving KV to a plain host buffer does *not* free real memory (RSS goes up). Only `np.memmap`-based paging does (0.0MB RSS cost confirmed up to 6.3GB). A concrete implementation requirement, not a footnote. |
| How does it compare to the competition? | At sqrt-space-kv's own ~90% compression regime, NVIDIA `kvpress`'s StreamingLLM/Knorm/SnapKV score **0% accuracy** on needle retrieval. sqrt-space-kv is exact at any ratio — the tradeoff is exactness vs. the latency tax above. |
| What's still open? | True MLA-absorbed-cache composability (needs vLLM/SGLang, CUDA-only) and a production `:sqrt-checkpoint` kernel. Reported honestly, not glossed over. |
| Does this work on video/image models, not just text? | **Depends on the architecture — tested, not guessed.** Autoregressive image tokens (LlamaGen-B real specs): mechanism transfers with zero code changes, exact match confirmed. Autoregressive video (MAGI-1 real specs): a 104GB-KV grounded projection, not locally run. Bidirectional diffusion (WAN/HunyuanVideo-class): activation checkpointing gives **zero** benefit for generation (self-correcting an earlier casual claim), real benefit only for training. |
| How would this actually land in `murakumo.cloud`? | **Neither "model surgery" nor "just a CLI flag" — tested against the real engine.** `mlx-moe` (the Apple-Silicon serving engine) reuses `mlx_lm.models.cache.KVCache` directly, no cache of its own. A real `PagedSqrtKVCache(KVCache)` subclass, run against a real cached model in a real `generate()` loop, produced **byte-identical tokens** (S=2048/8192, all 24 layers). It's an engine-layer cache-class swap, the same kind of change `--kv-bits` already ships. |

Full evidence trail (M0–M6, real Modal A100 + real Apple Silicon
experiments, every number reproducible from committed scripts):
`RESULTS.md` and the superproject ADR
`com-junkawasaki/root` → `90-docs/adr/2607182800-sqrt-space-kv-mla-composability-maturity-review.edn`.

## Metadata

| | |
|---|---|
| **Primary claim** | √S checkpoint residency cuts measured KV storage **90–97%** for S∈[1k,16k] |
| **Models tested** | 135M (Mac MLX) → 35B-A3B MoE / 14B dense (Modal A100 bf16) → GLM-4-9B-Chat-1M / Qwen2.5-14B (Mac, real weights) |
| **arXiv** | **submitted** 2026-07-10 · draft `7807366` · primary **cs.CL** · account `junkawasaki-n24y` |
| **Public id** | *pending announce* (update when `arXiv:YYMM.NNNNN` appears) |
| **Production serving integration** | [`gftdcojp/cloud-murakumo`](https://github.com/gftdcojp/cloud-murakumo) (`:serve :kv-policy`, private -- vLLM/mlx engine wiring) |
| **Portable library (Python + Rust)** | this repo, `python/` and `rust/` -- see [Library](#library-python--rust) below |
| **Submit actor** | [`kotoba-lang/arxiv`](https://github.com/kotoba-lang/arxiv) |
| **HF paper card** | [`com-junkawasaki/sqrt-space-kv-paper`](https://huggingface.co/datasets/com-junkawasaki/sqrt-space-kv-paper) |

## Library (Python + Rust)

The core residency math -- block sizing, keep-set computation, exact
reference attention -- is also available as small, dependency-free
Python and Rust packages, so the technique is usable outside this
monorepo's Clojure serving stack. This is the same "capability, not speed"
tradeoff described above: read the table before reaching for it in a
serving path.

```bash
pip install sqrt-space-kv        # python/
cargo add sqrt-space-kv          # rust/
```

```python
from sqrt_space_kv import keep_indices, strategy_cost

keep = keep_indices(4096)                       # token indices to keep resident
cost = strategy_cost("sqrt-checkpoint", 4096)    # {"kv_cells": ..., "fidelity": "exact-with-recompute", ...}
```

Both ports are pure, hand-written translations of the canonical
implementation in `cloud-murakumo`'s `sqrt_space.cljc` / `kv_runtime.cljc`
(no FFI, no shared binary -- this workspace deliberately avoids Rust-core +
generated-binding setups, see `CONTRIBUTING.md`). Conformance to the
canonical implementation is enforced by `tests/fixtures/golden.json`, a
fixture dumped from the `.cljc` and checked by both `python/tests/` and
`rust/tests/` -- see `CONTRIBUTING.md` for how it's regenerated.

## Layout

```text
sqrt-space-kv/
├── README.md                 # this file
├── RESULTS.md                # executive summary of measured results
├── LICENSE                   # MIT (library + repo code)
├── CONTRIBUTING.md           # golden-fixture sync process, running the test suites
├── RELEASE.md                # manual PyPI + crates.io release checklist
├── CHANGELOG.md
├── SECURITY.md
├── package.edn               # research package identity
├── arxiv-package.edn         # snapshot of arXiv package.edn
├── arxiv-status.edn          # snapshot of submission status
├── paper/                    # LaTeX source + abstract + PDF
│   ├── sqrt_space_kv.tex
│   ├── references.bib
│   ├── abstract.txt          # ASCII-only (arXiv metadata safe)
│   ├── sqrt_space_kv.pdf
│   └── Makefile
├── benchmarks/                # raw JSON + validation notes
│   ├── sqrt-kv-mac-m4-*.json|md
│   ├── sqrt-kv-modal-*.json|md
│   └── ollama-gemma4-*.json
├── docs/                      # long-form paper draft + co-scientist notes
│   ├── sqrt_space_llm_kv.md
│   └── sqrt-space-cosci.md
├── 90-docs/adr/2607190900-*.md # repo-local decision record for the library port
├── tests/fixtures/golden.json # cross-language conformance fixture (python/ + rust/)
├── python/                    # pip install sqrt-space-kv
└── rust/                      # cargo add sqrt-space-kv
```

## Key numbers (snapshot 2026-07-09)

| Platform | Model | S | Full KV | √S resident | Save |
|---|---|---:|---:|---:|---:|
| Mac M4 | SmolLM-135M 4bit | 8192 | 188.7 MB | 7.0 MB | 96.3% |
| Mac M4 | Qwen2.5-0.5B 4bit | 16384 | 201.3 MB | 5.4 MB | 97.3% |
| Modal A100 | Qwen2.5-7B bf16 | 16384 | 939.5 MB | 25.2 MB | 97.3% |
| Modal A100 | Qwen2.5-14B bf16 | 16384 | **3.22 GB** | **86.5 MB** | **97.3%** |

Save ratio ≈ \|K\|/S (checkpoints + active block); width-independent.

## Related paths in the monorepo

| Role | Path |
|---|---|
| This repo (paper, results, and the portable library) | `orgs/com-junkawasaki/sqrt-space-kv` |
| Canonical implementation (`.cljc`) + production serving glue | `orgs/gftdcojp/cloud-murakumo` |
| arXiv organism actor | `orgs/kotoba-lang/arxiv` |
| Live arXiv package | `orgs/kotoba-lang/arxiv/submissions/sqrt-space-kv` |

## Build paper PDF

```bash
cd paper && make   # pdflatex + bibtex
```

## License / citation

MIT license for the repo code (`python/`, `rust/`) -- see `LICENSE`.
Paper text and benchmark JSON are research artifacts for the
`com-junkawasaki` public research surface. The canonical implementation
remains in `gftdcojp/cloud-murakumo` (private; production serving
integration). Cite Williams STOC 2025 for the complexity result; this
package is an empirical systems transfer, not a TM simulation of
Transformers.
