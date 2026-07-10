# ADR-2607101000: com-junkawasaki/sqrt-space-kv research results home

## Status

Accepted (2026-07-10)

## Context

Williams STOC 2025 √t-space transfer to LLM KV residency was validated on
Mac (MLX/Ollama) and Modal (7B/14B A100). Paper packaging and arXiv draft
live under `kotoba-lang/arxiv`; implementation under `gftdcojp/cloud-murakumo`.
Owner asked to place the **research results** under **com-junkawasaki** orgs.

## Decision

Create public GitHub repo `com-junkawasaki/sqrt-space-kv` as the results
home:

- `paper/` — LaTeX + ASCII abstract + PDF
- `benchmarks/` — measured JSON/md from Mac + Modal
- `docs/` — long-form draft + co-scientist notes
- `RESULTS.md` / `package.edn` — executive summary + machine-readable claims

Code stays in `cloud-murakumo`. arXiv submit workflow stays in
`kotoba-lang/arxiv`. This repo is the **SSoT for results artifacts** under
the personal research org (`com-junkawasaki`; this results repo is public).

## Consequences

- west-registered under `orgs/com-junkawasaki/sqrt-space-kv`
- Cloneers of the monorepo get results via `west update` without pulling
  murakumo GPU product surface
- arXiv draft 7807366 continues under N24 account (primary cs.CL)
