# ADR-0009 — Inference device is a runtime knob; CPU is the local concept-validation profile

- **Status:** Accepted
- **Date:** 2026-06-20
- **Deciders:** architecture
- **Related:** ADR-0001 (hexagonal core), ADR-0006 (swappable model providers), ADR-0008 (LLM-in-the-loop); NFR-6, NFR-8; roadmap M1

## Context

The real perception adapters (RF-DETR, ByteTrack, pitch keypoints, GVHMR, TrackNet) target a
self-hosted GPU in production. But we need to **validate the concept end-to-end on this machine**,
which has a display but no GPU we want to depend on. The adapters already carried a `device: str`
field (defaulting to `"cuda"`) and the detector a `weights` path, but the composition root
(`wiring.default_ports`) constructed every real adapter with **no arguments** — so neither the CLI
nor the MCP agent could pick the device or point at local weights. The knobs existed but were
unreachable.

A second wrinkle: the request to "do it on GPU" and the caveat "validate on CPU" are not a
contradiction once device is a *runtime* choice rather than a code path — the same wiring runs on
either by flipping one string.

## Decision

1. **Inference device is a runtime knob, never a code fork.** Every real perception adapter keeps a
   `device: str` field; the composition root forwards it to all of them. There is exactly one code
   path; CPU vs GPU is data.
2. **The adapter dataclass default stays `"cuda"`** (it documents the *production* target), but the
   **composition root and both entry points default to `"cpu"`** — the local concept-validation
   profile. `wiring.default_ports(device="cpu", ...)`; CLI `--device {cpu,cuda}` defaults to `cpu`.
   The composition root is the seam that picks the deployment profile (ADR-0001), not the adapter
   and not the core.
3. **Weights are sourced per adapter, and we expose only the flags that map to a real field.**
   RF-DETR exposes a user-settable `weights` path (forwarded to the backend as `pretrain_weights`),
   surfaced as `default_ports(detector_weights=...)` / CLI `--detector-weights`. The other adapters
   source weights inside their backend today, so **no `--*-weights` flag is wired for them yet** — a
   dead flag would violate "no half-finished implementations". Those flags arrive when ball/pose/
   calibration weights are wired (roadmap P2.3/P2.4).
4. **The pure/heavy split is unchanged (ADR-0001/0006).** Construction stays lazy — threading
   `device`/`weights` imports no torch/cv2 — so the wiring is unit-tested with no GPU (`test_wiring`).
5. **CPU is for the concept proof; GPU is production, reached by `--device cuda` with no code
   change.** GVHMR pose is flagged as the one stage that may not be CPU-viable even for a single
   clip; it may require a GPU regardless (roadmap P2.4).

## Consequences

**Positive**
- The identical golden path runs CPU↔GPU by configuration (NFR-6 spirit); the concept can be
  validated locally with no GPU.
- The LLM and the human drive the same device/weights choice (ADR-0008): one knob in one seam.
- No dead CLI flags — every flag maps to a field that exists today.

**Negative / costs**
- CPU inference is slow (especially HMR); the CPU profile is for validation, not throughput.
- A small asymmetry: only the detector exposes a user-settable weights path today. Revisit when the
  other models' weights are wired.

## Alternatives considered

- **Hardcode `cuda`** — rejected: makes local concept validation impossible.
- **Separate CPU and GPU code paths / a build flag** — rejected: divergence risk; a `device` string
  threaded through one composition root is enough.
- **Expose `--ball-weights`/`--pose-weights`/`--calibrator-weights` now** — rejected for this step:
  those adapters carry no user-settable weights field yet, so the flags would be no-ops. Defer to
  when those models are wired.
