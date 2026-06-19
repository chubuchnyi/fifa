# ADR-0004 — Offline job queue + content-addressable cache

- **Status:** Accepted
- **Date:** 2026-06-19
- **Deciders:** architecture
- **Related:** ADR-0001 (hexagonal core), ADR-0006 (swappable providers); C5, NFR-4, UX-8, R-10

## Context

Almost every stage is slow or expensive: HMR, 3DGS/NeRF environment reconstruction, per-subject
avatars, ViewSynthesizer passes, photoreal render. Three forces follow:

- **The UI must stay responsive** while these run (UX-8, R-10) — they cannot block the editor.
- **Re-running with unchanged inputs must not recompute** — especially the paid generative passes
  (NFR-4). Recomputing a cloud avatar because an unrelated frame changed is wasted money.
- **Cost and reproducibility must be visible** — which model/version/params produced each artifact.

## Decision

1. **Every stage is a job** submitted to a `JobQueue` port (`DETECT`, `TRACK`, `CALIBRATE`,
   `POSE`, `BALL`, `ENV`, `AVATAR`, `RENDER`, `OBSERVE`, `EXPORT`, and the optional `AMPLIFY`).
   The fake `InProcessJobQueue` runs thunks synchronously for tests and the dry-run; a real worker
   (subprocess / Celery-class) is an adapter swap.
2. **Outputs are content-addressed.** `content_key(stage, input_hash, params, model_version)`
   derives a canonical, order-independent key; `run_cached(queue, cache, stage, thunk, ...)`
   returns the cached value on a hit and only executes the thunk on a miss.
3. **`model_version` and `params` are part of the key.** Upgrading a model or changing a parameter
   invalidates exactly the affected artifacts — no manual cache busting, no stale results.
4. **Provenance rides along** (ADR-0006): each executed stage appends a `RunRecord` (model, cache
   key, hit/miss, cost) to the scene's `RunLog`, so total spend and reproducibility are queryable.

## Consequences

**Positive**
- The editor never blocks on a heavy stage (UX-8); generative passes are never recomputed for
  unchanged inputs (NFR-4).
- Deterministic, order-independent keys make runs reproducible and debuggable.
- Cost is observable per scene via the `RunLog`.

**Negative / costs**
- Cache storage grows; a real deployment needs an eviction/retention policy (out of scope here).
- The key is only as honest as `input_hash`/`params`/`model_version` — an adapter that forgets to
  bump `model_version` will serve stale results.
- Synchronous fake queue hides concurrency bugs that only a real async worker would surface.

## Alternatives considered

- **Synchronous in-line pipeline** — rejected: blocks the UI (R-10) and recomputes everything.
- **Timestamp/path-based caching** — rejected: not content-addressable; misses parameter and
  model-version changes, and produces false hits when a file is touched but unchanged.
- **Memoise in-process only** — rejected: loses results across runs, which is exactly when the
  expensive generative passes most need to be reused.
