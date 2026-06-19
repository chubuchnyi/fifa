# pitch3d

Offline desktop tool that reconstructs an **editable, photorealistic 3D football
episode** from a **single broadcast camera**, with the emphasis on pose estimation
and easy manual correction that **propagates across frames**.

> **Status: architecture scaffold.** This repository contains the *architecture* —
> typed scene model, port contracts, the correction engine, fake adapters, and one
> end-to-end dry-run. **No heavy ML/CV/render/generative model is implemented**;
> every real adapter is an honest stub (`NotImplementedError`). See
> [`docs/architecture.md`](docs/architecture.md).

Source of truth for requirements: [`TZ_3D_football_reconstruction.md`](TZ_3D_football_reconstruction.md) (v0.3).
Source of truth for the *shape* of this deliverable: [`CLAUDE_CODE_architecture_task.md`](CLAUDE_CODE_architecture_task.md).

## Core ideas

| Principle | Where it lives |
|---|---|
| **Hexagonal core** — pure Python, imports neither `bpy` nor any ML/render lib | `src/pitch3d/core/` |
| **Dual representation** — edit on SMPL-X + curves (proxy); render is derived | `core/scene`, `adapters/render` |
| **Models behind adapters** — self-hosted + external API, swappable | `core/ports`, `adapters/*` |
| **Offline job queue + content-addressable cache** | `core/orchestration`, `adapters/fakes` |
| **ViewSynthesizer (video-diffusion) at two seams** — render-adapter (A) + data amplifier (B) | `core/ports/view_synthesizer.py`, `adapters/viewsynth` |

Dependencies point **inward**: `app → adapters → core`. The core never imports an adapter.

## Layout

```
src/pitch3d/
  core/            # pure core (numpy only): scene model, correction math, orchestration contracts, ports
  adapters/        # everything infrastructural: models, viewsynth, blender, render, export, fakes
  app/             # composition root + CLI dry-run
tests/             # core tests on fakes — green without GPU/Blender
docs/              # architecture.md, scene-schema.md, adr/*, risk-map.md, roadmap.md
```

## Quickstart (no GPU, no Blender)

```bash
# tests
python3 -m pytest                       # or: just test

# end-to-end dry-run on fake adapters
python3 -m pitch3d.app.cli --workdir out/dryrun                      # or: just dryrun
python3 -m pitch3d.app.cli --workdir out/vs --amplify --render viewsynth   # exercises VS seams A & B
```

`numpy` is the only runtime dependency and is enough to run the tests and the dry-run.
Heavy extras (`torch`, `gsplat`, `bpy`, …) are **declared but not installed**.

## Reproducibility & licenses

- The core pins `numpy` to a compatible range so the scaffold runs anywhere; **exact**
  pins for a reproducible build belong in a lockfile (`uv.lock` / `requirements.txt`).
- Heavy adapter deps in `pyproject.toml` are **exact-pinned** with inline license notes.
  Per the TZ this is **internal / research** use, so GPL (Blender), AGPL (some detectors)
  and the SMPL/SMPL-X model licenses are acceptable (NFR-8).

## Where to read next

1. [`docs/architecture.md`](docs/architecture.md) — overview + data/control-flow diagrams (incl. both ViewSynthesizer seams).
2. [`docs/scene-schema.md`](docs/scene-schema.md) — the canonical scene model.
3. [`docs/adr/`](docs/adr/) — the decisions and why (0001–0007).
4. [`docs/roadmap.md`](docs/roadmap.md) — M0→M3 plan and the M1 vertical slice.
