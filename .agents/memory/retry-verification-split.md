# Retry verification system split

`server/retry-verification/` is a dedicated 1:1 copy of `server/verification/`
(verify.ts, store.ts, clip.ts, prompt.ts, timeline.ts, index.ts), created so the
manual Retry flow can be changed independently.

- The manual Retry endpoints in server.ts (segment Retry `POST /segment/:i/retry`
  and gap Retry `POST /gap/retry`) call `retryRecheckSegment` imported from
  `./server/retry-verification`.
- The initial verify pass (`verifyMatchedSegments`) and all record read/write
  helpers used elsewhere in server.ts still come from `server/verification/` —
  the protected main candidate system. Never edit it for Retry behavior.
- Both copies share the SAME on-disk candidate record files (identical store
  path scheme), so records stay compatible across systems.
- Rule: Retry-only behavior changes go in `server/retry-verification/` only.
