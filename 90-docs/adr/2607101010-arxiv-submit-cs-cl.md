# ADR-2607101010: arXiv submit of √S KV paper (cs.CL, draft 7807366)

## Status

Accepted (2026-07-10) — owner confirmed final Submit succeeded.

## Context

Browser automation (`kotoba-lang/arxiv` Playwright runner) prepared draft
`7807366` under account `junkawasaki-n24y`:

- Authorship fix: select `is_author` **value=1** (hidden sentinel value=0 rejected)
- Category probe: account endorsed only for **cs.CL** among tested subjects
  (cs.LG / cs.DB / physics.* / math.* blocked despite prior cs.DB papers)
- Primary retargeted **cs.LG → cs.CL**; cross-list intent remains cs.LG, cs.CC
- Source tar uploaded (tex / bib / bbl); Process + Metadata filled by human
- Metadata Abstract rejected Unicode `≈` — replaced with ASCII `~`
- Final public Submit completed by owner (APPROVE_FINAL human gate)

## Decision

1. Record submission state as **`:submitted`** for draft `7807366`.
2. Keep primary **cs.CL** in package SSoT (endorsement reality).
3. Public results remain in `com-junkawasaki/sqrt-space-kv` (public repo).
4. When arXiv announces an id, patch:
   - `package.edn` `:research/arxiv-id`
   - `arxiv-status.edn` / `submissions/sqrt-space-kv/status.edn`
   - README badge / abs URL
   - this ADR addendum

## Consequences

- No further final-submit automation required for this draft
- Replacement / withdraw follow normal arXiv policy via same account
- Endorsement for cs.LG still available as follow-up if reclassification wanted
