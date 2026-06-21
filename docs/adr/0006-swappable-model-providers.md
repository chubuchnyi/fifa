# ADR-0006 — Swappable model providers behind one `ModelProvider` family, with provenance

- **Status:** Accepted
- **Date:** 2026-06-19
- **Deciders:** architecture
- **Related:** ADR-0001 (hexagonal core), ADR-0004 (queue + cache), ADR-0007 (ViewSynthesizer); C4, NFR-6, NFR-7, NFR-8, UX-7

## Context

Every CV/ML stage has several viable backends — detection, tracking, HMR, ball tracking, env
reconstruction, avatars, view-synthesis — and each can run either **self-hosted on a GPU** or via
an **external cloud API**. These choices will change over the project's life, they carry different
**licenses** (some non-commercial, e.g. SMPL-X) and different **monetary costs** (API calls), and
NFR-7 requires us to record **what produced each artifact** for reproducibility.

## Decision

1. **Every model-backed port derives from `ModelProvider`** (`core/ports/base.py`) with a single
   `info() -> ModelInfo` method. `ModelInfo` carries `name`, `version`, `backend`, `license`,
   `est_cost_usd`, and the frozen `params` dict.
2. **`Backend` is an enum:** `local` (self-hosted GPU), `api` (external/cloud), `fake`
   (deterministic test double), `builtin` (pure-core math, e.g. the ball lift).
3. **Provenance is recorded per stage.** A `RunRecord` (model, cache key, hit/miss, duration, cost)
   is appended to the scene's append-only `RunLog`; `total_cost_usd()` sums the spend (UX-7).
4. **`params` feed the cache key** (ADR-0004), so swapping a backend or changing a parameter
   invalidates exactly the affected artifacts.
5. **Adding a model is adding an adapter** (NFR-6): the fakes report `backend=fake`; the real stubs
   in `adapters/models` name their intended backend + license (e.g. RF-DETR Apache-2.0/local,
   Rodin api) and raise `NotImplementedError` until wired. Swap one fake for its real stub at a
   time; each must pass the same port test the fake passes.

## Consequences

**Positive**
- Vendor-neutral: self-hosted ↔ API is an adapter swap, no core change (C4, NFR-6).
- The swap is reachable **without editing the wiring**: a heavy backend (a vendored GVHMR/
  TrackNet/keypoint network) is injected into its real adapter by **dotted path** from the
  composition root and the CLI (`--pose-backend`/`--ball-backend`/`--calibrator-backend
  pkg.module:Factory`, resolved by `app.wiring._resolve_backend`). The factory lives in the
  on-box engineer's own code, so the research network stays out of the core tree (ADR-0001);
  the path is protocol-guarded at startup and requires its real adapter to be selected.
- License and cost are visible in the UI and the run log (UX-7, NFR-8) — important given SMPL-X /
  AGPL constraints and per-call API spend.
- Every artifact is reproducible: the run log says exactly which model/version/params made it (NFR-7).

**Negative / costs**
- Every adapter must report honest `ModelInfo`; a wrong `version`/`params` corrupts the cache key
  and provenance.
- `est_cost_usd` is an estimate, not a billed figure — useful for budgeting, not accounting.
- Provenance plumbing touches every artifact-producing path (asset refs, run records).

## Alternatives considered

- **Hardcode one backend per stage** — rejected: violates C4/NFR-6 and bakes a license/cost choice
  into the core.
- **Config-only backend switching without provenance** — rejected: loses reproducibility (NFR-7)
  and the license/cost transparency UX-7 requires.
