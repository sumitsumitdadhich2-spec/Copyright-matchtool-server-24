# LOCKED: VLM full-movie scan plan (user-confirmed, planning only — not yet implemented)

## Decision (locked by user)
- **7 fps, DEFAULT (medium) resolution** for both videos per request.
- Per request: 1-min short video (fixed) + 1-min movie chunk = 2 × 420 frames × 258 tok ≈ **217K tokens** (fits 250K TPM with margin for prompt/output).
- Movie scanned as sliding 1-min chunks: request N = short video + movie minute N.

## Parallel model strategy (locked)
- Each Gemini free-tier model has its OWN quota pool (RPM/TPM/RPD independent).
- Run ALL usable models in parallel, 1 request/min each (TPM-bound: 217K/250K):
  - `gemini-flash-lite-latest` — 15 RPM / 500 RPD
  - `gemini-3.1-flash-lite` — 15 RPM / 500 RPD
  - Flash models (3.6 / 2.5 / 3.5 Flash) — 5 RPM / **20 RPD each**
- When a 20-RPD model exhausts its daily quota, drop it from rotation; keep collecting results from remaining models. Lite models carry the tail.

## Timing (2-hour movie = 120 chunks)
- Phase 1: 5 models × 1 req/min = 5 chunks/min → 20 min = 100 chunks (flash models hit 20 RPD each and stop).
- Phase 2: 2 lite models × 1 req/min = 2 chunks/min → remaining 20 chunks = 10 min.
- **Total ≈ 30 min per movie** (vs ~2 h single-model sequential).
- Same-day second movie: flash RPD spent → 2 lite models only ≈ 60 min.

## Constraints to remember
- TPM counts video + prompt + output; never push past ~230K video tokens.
- Resolution is a fixed enum (LOW ~66 / MEDIUM ~258 tok/frame); no custom values.
- 8 fps default = ~248K → too tight, rejected. LOW-res options rejected for detail loss.
