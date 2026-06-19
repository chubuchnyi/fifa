# ADR-0001 — Hexagonal (ports & adapters) core, pure-`numpy`

- **Status:** Accepted
- **Date:** 2026-06-19
- **Deciders:** architecture
- **Related:** ADR-0002 (dual representation), ADR-0004 (queue + cache), ADR-0006 (swappable providers), ADR-0008 (LLM-in-the-loop); C1, NFR-6, AC-7

## Context

The pipeline depends on heavy, platform-bound machinery: CUDA models (detection, tracking,
HMR), `cv2`, Blender's `bpy`, video-diffusion backends (some cloud APIs), photoreal renderers,
and — later — LLM SDKs. All of it is slow, GPU- or network-bound, and costly to run in CI.

Yet the parts that hold the product's value — the canonical scene model, the non-destructive
correction math, the ball 3D lift, the cache-key derivation, the viewpoint math for visual
feedback — are *pure algorithms*. If they are entangled with `torch`/`bpy`/an API client, none
of it is testable without a GPU, and swapping a model means editing the core (violating NFR-6).

## Decision

1. **`core/` imports only `numpy`.** It never imports `torch`, `cv2`, `bpy`, a video-diffusion
   client, or an LLM SDK. A `compileall` + import of `core` succeeds on a laptop with no GPU.
2. **Everything infrastructural sits behind an ABC port** in `core/ports/` (`Detector`,
   `Tracker`, `FieldCalibrator`, `PoseEstimator`, `BallTracker`, `EnvReconstructor`,
   `AvatarBuilder`, `ViewSynthesizer`, `RenderPass`, `SceneObserver`, `Exporter`, `Cache`,
   `JobQueue`). The core depends on the ABC, never on a concrete adapter.
3. **Dependencies point inward.** `adapters/` and the driving adapters (`app/cli`, `adapters/mcp`)
   may import `core`; `core` imports nothing outward. The composition happens in one place,
   `app/wiring.py` (the composition root).
4. **Every port has a deterministic fake** in `adapters/fakes/`. The fakes satisfy the same ABCs
   the real adapters do, so the whole tool — including the dry-run and the LLM-feedback loop —
   runs green with no GPU, no Blender, no models, no network.

## Consequences

**Positive**
- The core is unit-testable in milliseconds; CI needs no GPU (AC-7).
- A new model — or a whole new ViewSynthesizer backend — is a new adapter, not a core change (NFR-6).
- The CLI and the MCP server are both *driving* adapters over the same wiring, so an LLM and a
  human run identical use-cases (ADR-0008).

**Negative / costs**
- Indirection: a port + a fake + a real stub per capability is more files than a direct call.
- Port signatures are a contract; changing one ripples to every adapter and fake.
- Fakes must stay behaviourally faithful or green tests can mask a broken real adapter.

## Alternatives considered

- **Service classes that import models directly** — rejected: nothing is testable without the
  GPU/model present, and model choice leaks into the core.
- **A runtime flag to "skip the heavy parts" in tests** — rejected: that is a fake without the
  discipline of a port; it rots and diverges from the real path.
