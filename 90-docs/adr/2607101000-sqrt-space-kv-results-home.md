# ADR-2607101000: com-junkawasaki/sqrt-space-kv research results home

## Status

Accepted (2026-07-10).  
**Addendum 1 (2026-07-10):** repo is **public**.  
**Addendum 2 (2026-07-10):** arXiv **submitted** (draft `7807366`, primary `cs.CL`,
account `junkawasaki-n24y`). Public arXiv id pending announcement.

## Context

Williams STOC 2025 √t-space transfer to LLM KV residency was validated on
Mac (MLX/Ollama) and Modal (7B/14B A100). Paper packaging and arXiv workflow
live under `kotoba-lang/arxiv`; implementation under `gftdcojp/cloud-murakumo`.
Owner asked to place the **research results** under **com-junkawasaki** orgs,
then to make the results repo **public**, then confirmed **arXiv submit** succeeded.

## Decision

Create public GitHub repo `com-junkawasaki/sqrt-space-kv` as the results
home:

- `paper/` — LaTeX + ASCII abstract + PDF
- `benchmarks/` — measured JSON/md from Mac + Modal
- `docs/` — long-form draft + co-scientist notes
- `RESULTS.md` / `package.edn` — executive summary + machine-readable claims
- `arxiv-status.edn` — submission state (SSoT for results-side tracking)

Code stays in `cloud-murakumo`. arXiv submit workflow stays in
`kotoba-lang/arxiv`. This repo is the **SSoT for results artifacts** under
the personal research org (`com-junkawasaki`; this results repo is **public**).

### arXiv submission (addendum 2)

| Field | Value |
|---|---|
| Draft id | `7807366` |
| Account | `junkawasaki-n24y` |
| Primary | `cs.CL` (N24 endorsed for CL; cs.LG blocked without new endorsement) |
| Cross-list intent | `cs.LG`, `cs.CC` (when endorsed / moderator path allows) |
| License | CC BY 4.0 |
| State | **submitted** (owner-confirmed 2026-07-10) |
| Public id | *pending* (not yet in export.arxiv.org API) |

ASCII abstract only (no Unicode `≈` / `√` in metadata fields).

## Consequences

- west-registered under `orgs/com-junkawasaki/sqrt-space-kv` (public GitHub)
- Cloneers of the monorepo get results via `west update` without pulling
  murakumo GPU product surface
- After announce, update `package.edn` / `arxiv-status.edn` / README with
  `arXiv:YYMM.NNNNN` and abs/pdf URLs
