# Paper pack — √S KV from Williams (arXiv + Hugging Face)

| File | Purpose |
|------|---------|
| [`sqrt_space_llm_kv.md`](sqrt_space_llm_kv.md) | Full draft (Markdown). Primary source for arXiv HTML / HF Papers. |
| [`hf_paper_card.md`](hf_paper_card.md) | Short card + abstract for Hugging Face Papers / model-card style. |
| [`arxiv_abstract.txt`](arxiv_abstract.txt) | Plain-text abstract (≤1920 chars) for arXiv submit form. |

## Submission checklist

### arXiv (cs.LG primary; cross-list cs.CL, cs.CC)

1. Convert `sqrt_space_llm_kv.md` → PDF (pandoc + preferred template) or upload HTML.
2. Paste `arxiv_abstract.txt`.
3. Comments: “Code: github.com/gftdcojp/cloud-murakumo (scripts/modal_sqrt_kv_bench.py, scripts/bench_sqrt_kv_mac.py). Empirical systems note; not a new TM lower bound.”
4. Attach or link benchmark JSON under `docs/benchmarks/`.

### Hugging Face Papers

1. Open [hf.co/papers/submit](https://huggingface.co/papers) (or daily papers flow).
2. Use arXiv id once assigned, **or** upload the Markdown/PDF with `hf_paper_card.md` blurb.
3. Link GitHub repo + Modal/Mac JSON artifacts.

## Repro one-liners

```bash
# Mac
python scripts/bench_sqrt_kv_mac.py --model mlx-community/SmolLM-135M-Instruct-4bit

# Modal A100
modal run scripts/modal_sqrt_kv_bench.py \
  --model-id Qwen/Qwen2.5-7B-Instruct --gpu A100

# Policy tournament (CPU)
clj -M:cosci 3
```
