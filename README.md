# sqrt-space-kv

**Williams STOC 2025 √t-space → LLM KV-cache residency**

Research results home under `com-junkawasaki`: paper package, measured
benchmarks (Mac Apple Silicon + Modal NVIDIA A100), and pointers to the
implementation / arXiv submission workflow.

| | |
|---|---|
| **Primary claim** | √S checkpoint residency cuts measured KV storage **90–97%** for S∈[1k,16k] |
| **Models** | 135M (Mac MLX) → 14B (Modal A100 bf16) |
| **Exactness** | Host-paged online-softmax block attention is numerically exact (max\|Δ\|≈7e-5) |
| **Negative** | Whole-prefix recompute each token *increases* peak device memory |
| **arXiv** | **submitted** 2026-07-10 · draft `7807366` · primary **cs.CL** · account `junkawasaki-n24y` |
| **Public id** | *pending announce* (update when `arXiv:YYMM.NNNNN` appears) |
| **Code** | [`gftdcojp/cloud-murakumo`](https://github.com/gftdcojp/cloud-murakumo) (`:serve :kv-policy`) |
| **Submit actor** | [`kotoba-lang/arxiv`](https://github.com/kotoba-lang/arxiv) |
| **HF paper dataset** | [`com-junkawasaki/sqrt-space-kv-paper`](https://huggingface.co/datasets/com-junkawasaki/sqrt-space-kv-paper) |
| **Qwen3.6 MoE** | 35B-A3B Modal: **88.5%** KV save @ S=16k (hybrid; see benchmarks) |
| **Maturity review** | 2026-07-18, M0–M4 **landed**, M5–M6 landed-with-caveats. Real Modal A100 evidence: exact end-to-end token match confirmed (M3), but re-paging every decode step is **2.2x–3.7x slower** than full-KV decode (M2/M3). At sqrt-space-kv's own ~90–97% compression regime, competing lossy presses (H2O/SnapKV/StreamingLLM) score **0% needle-retrieval accuracy** (M4) — sqrt-space-kv is exact but pays the latency tax they don't. MLA composability (M5) got the same save-ratio law on DeepSeek-V2-Lite, but its reference cache doesn't actually store the MLA-compressed latent, so the real composability question stays open. One vLLM datapoint (M6) shows the HF-loop baseline wasn't a weak strawman. Full production kernel + vLLM/SGLang A/B (M6) and true MLA-absorbed-cache test (M5) remain open — see `RESULTS.md` and superproject ADR `2607182800-sqrt-space-kv-mla-composability-maturity-review.edn` |

## Layout

```text
sqrt-space-kv/
├── README.md                 # this file
├── RESULTS.md                # executive summary of measured results
├── package.edn               # research package identity
├── arxiv-package.edn         # snapshot of arXiv package.edn
├── arxiv-status.edn          # snapshot of submission status
├── paper/                    # LaTeX source + abstract + PDF
│   ├── sqrt_space_kv.tex
│   ├── references.bib
│   ├── abstract.txt          # ASCII-only (arXiv metadata safe)
│   ├── sqrt_space_kv.pdf
│   └── Makefile
├── benchmarks/               # raw JSON + validation notes
│   ├── sqrt-kv-mac-m4-*.json|md
│   ├── sqrt-kv-modal-*.json|md
│   └── ollama-gemma4-*.json
└── docs/                     # long-form paper draft + co-scientist notes
    ├── sqrt_space_llm_kv.md
    └── sqrt-space-cosci.md
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
| This results repo | `orgs/com-junkawasaki/sqrt-space-kv` |
| Serving / KV policy code | `orgs/gftdcojp/cloud-murakumo` |
| arXiv organism actor | `orgs/kotoba-lang/arxiv` |
| Live arXiv package | `orgs/kotoba-lang/arxiv/submissions/sqrt-space-kv` |

## Build paper PDF

```bash
cd paper && make   # pdflatex + bibtex
```

## License / citation

Paper text and benchmark JSON are research artifacts for the
`com-junkawasaki` public research surface. Upstream code remains in
`gftdcojp/cloud-murakumo`. Cite Williams STOC 2025 for the complexity
result; this package is an empirical systems transfer, not a TM simulation
of Transformers.
