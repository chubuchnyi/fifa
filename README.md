# pitch3d

Offline desktop tool that reconstructs an **editable, photorealistic 3D football
episode** from a **single broadcast camera**, with the emphasis on pose estimation
and easy manual correction that **propagates across frames**.

> **Status: architecture scaffold + M1 adapters wired.** The *architecture* — typed scene
> model, port contracts, the correction engine, fakes, and an end-to-end dry-run — is complete,
> and the M1 perception/render/export ports now carry real adapters. Each real adapter is split
> the same way: a **pure half** (canonical-type mapping + maths) unit-tested with **no GPU** via
> an injected stub backend, and a **heavy half** (inference/serialization) lazy-imported and gated
> behind its optional extra with an actionable install error. The dependency-free reals — the
> reprojection-overlay `RenderPass` (with confidence highlighting), a camera-free top-down **radar**
> VIEW, and the SMPL-X-`.npz`/JSON exporter — run the whole golden path on a real clip today with no
> GPU/Blender. Editing closes the same loop: a drag in a *live* Blender session (a radar dot or the
> root Empty) becomes a `ROOT_TRANSLATION` `Correction` through the **same** `apply_offset` use-case
> the LLM drives — the host owns the scene, Blender just reports world positions (ADR-0010). The
> host side (the `radar_to_world` inverse, the `subject_<id>` id↔name contract, and the socket edit
> loop) is unit-tested headlessly; only the GUI session needs a Blender binary + a display. The
> **calibrator** is the first backend benchmarked on *independent* real data — on SoccerNet
> `calibration-2023` the injected PnLCalib backend registers the pitch to a **0.236 m median** on the
> ¾ of broadcast frames where it locks on (completeness, not planar accuracy, is the limiter; B1). See
> [`docs/roadmap.md`](docs/roadmap.md) for per-step state and [`docs/architecture.md`](docs/architecture.md).

Source of truth for requirements: [`TZ_3D_football_reconstruction.md`](TZ_3D_football_reconstruction.md) (v0.3).
Documentation map: [`docs/README.md`](docs/README.md). The original architecture-kickoff brief
(now executed) is archived under [`docs/archive/`](docs/archive/).

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
# tests (pytest puts src/ on the path automatically — see pyproject `pythonpath`)
python3 -m pytest                                       # or: just test

# end-to-end dry-run on fake adapters — no install, straight from the source tree:
PYTHONPATH=src python3 -m pitch3d --out-dir out/dryrun   # or: just dryrun

# …or install once into a venv (same as `just setup`), then use the module / console script:
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
python -m pitch3d --out-dir out/dryrun                   # or: pitch3d-dryrun --out-dir out/dryrun

# flags: --out-dir DIR  --frames N  --subjects N  --format json|smplx_npz|gltf|glb
#        --clip path/to.mp4                       # real ingest via ffprobe (else a synthetic clip)
#        per-port adapter swap (fake default | real): --detector rfdetr  --tracker bytetrack
#        --calibrator keypoints  --pose gvhmr  --ball tracknet  --render overlay  --export gltf
#        --observer blender                      # real proxy SCENE_3D via blender --background (CPU)
#        --device cpu|cuda  --detector-classes coco|sports  --detector-weights PATH  # CPU validation vs GPU prod (ADR-0009)
#   e.g. fully-real-but-no-GPU render + export path:
#        PYTHONPATH=src python3 -m pitch3d --clip clip.mp4 --render overlay --export gltf --format smplx_npz
#   e.g. real RF-DETR + ByteTrack on CPU on a real clip (needs `pip install '.[cv]'`; the COCO base
#   weights auto-download — yields players + teams A/B, no GPU; `--ball tracknet` is still a stub):
#        PYTHONPATH=src python3 -m pitch3d --clip clip.mp4 --frames 6 --detector rfdetr --tracker bytetrack --device cpu --render overlay
#   e.g. real Blender proxy SCENE_3D feedback (needs a Blender binary, no GPU):
#        PITCH3D_BLENDER=/path/to/blender PYTHONPATH=src python3 -m pitch3d --observer blender
```

`numpy` is the only runtime dependency and is enough to run the tests and the dry-run.
Heavy extras (`torch`, `gsplat`, `bpy`, …) are **declared but not installed**.

**Render the actual SMPL-X body mesh in Blender** (not the box proxies): forward a `smplx_npz`
export through SMPL-X and render it with Blender Cycles — see [`docs/blender-demo.md`](docs/blender-demo.md)
([`scripts/smplx_export_meshes.py`](scripts/smplx_export_meshes.py) → [`scripts/blender_render_meshes.py`](scripts/blender_render_meshes.py)).
Bring up the local CPU env this demo needs (CPU `torch` + `smplx` + `matplotlib`) in one command:
`just setup-local` (or [`scripts/local_setup.sh`](scripts/local_setup.sh)) — `just setup` installs the
`[dev]`-only env, which is enough for the tests and the dry-run but not the mesh demos.

Moving development to a rented GPU box? [`scripts/cloud_setup.sh`](scripts/cloud_setup.sh) installs
a CUDA `torch` + the wired real adapters and verifies the GPU; [`docs/cloud-dev.md`](docs/cloud-dev.md)
is the spin-up → verify → `--device cuda` golden-path checklist (ADR-0009).

## Reproducibility & licenses

- `pyproject.toml` is the dependency **source of truth** (core declares `numpy` as a
  compatible range). **Exact** transitive pins for a reproducible core live in the
  compiled lockfiles — [`requirements.txt`](requirements.txt) (runtime) and
  [`requirements-dev.txt`](requirements-dev.txt) (core + dev) — regenerated with
  `pip-compile [--extra dev] --strip-extras pyproject.toml`. Install a frozen env with
  `pip install -r requirements-dev.txt`. Heavy GPU/Blender extras lock **per-milestone**
  (they can't resolve without CUDA/Blender), matching adapter isolation (ADR-0001).
- Heavy adapter deps in `pyproject.toml` are **exact-pinned** with inline license notes.
  Per the TZ this is **internal / research** use, so GPL (Blender), AGPL (some detectors)
  and the SMPL/SMPL-X model licenses are acceptable (NFR-8).

## Where to read next

Start at the [`docs/README.md`](docs/README.md) map, or jump straight in:

1. [`docs/architecture.md`](docs/architecture.md) — overview + data/control-flow diagrams (incl. both ViewSynthesizer seams).
2. [`docs/scene-schema.md`](docs/scene-schema.md) — the canonical scene model.
3. [`docs/adr/`](docs/adr/) — the decisions and why (0001–0010).
4. [`docs/roadmap.md`](docs/roadmap.md) — M0→M4 plan and the M1 vertical slice.
5. [`docs/m1-status-and-plan.md`](docs/m1-status-and-plan.md) — where M1 actually stands right now (blockers, next steps).
6. [`docs/cloud-dev.md`](docs/cloud-dev.md) — provisioning a GPU box for `--device cuda` development.
7. [`docs/runpod-runbook.md`](docs/runpod-runbook.md) — concrete RunPod spin-up runbook (MCP / `runpodctl`, SSH, cost control).
