# One-command demo

`scripts/demo.sh` runs the whole product end to end and narrates each step, so you can show
"single broadcast clip → editable 3D pitch" without remembering any of the wiring. It is the
automated version of the manual pod recipe in [`cloud-dev.md`](cloud-dev.md) §5 +
[`blender-demo.md`](blender-demo.md), with honest banners about what ran real vs. proxy.

```bash
scripts/demo.sh            # FULL  — GPU pod does the real perception, then stops itself
scripts/demo.sh --local    # LOCAL — re-render bodies from the last export, no pod, no cost
```

## What it does, step by step

1. **Preflight** the local render toolchain (venv with `torch`+`smplx`, the gated SMPL-X model,
   and Blender — falls back to a matplotlib render if Blender is absent).
2. **(FULL only) Bring up the GPU pod** — reuses a live pod or resumes a stopped one, with
   host failover (see [`pod.sh`](../scripts/pod.sh)).
3. **(FULL only) Sync** `src/` + `scripts/` to the pod and **check the staged GPU assets**
   (clip, SMPLest-X repo, WASB weight, optional PnLCalib).
4. **(FULL only) Run the real reconstruction on CUDA** —
   `RF-DETR detect → ByteTrack → SMPLest-X-H pose (0.69B) → WASB ball`, overlay + `smplx_npz`
   export (this is [`pod_real_e2e.sh`](../scripts/pod_real_e2e.sh)).
5. **(FULL only) Pull** the overlay renders, tactical radar, and the SMPL-X export down, then
   **stop the pod** (billing returns to volume-only).
6. **Forward the export through SMPL-X** to posed meshes for the scene mid-frame.
7. **Render the recovered bodies** — Blender Cycles (a wide pitch shot + a hero close-up), or
   matplotlib if Blender is unavailable.
8. **Print a summary** naming exactly which adapters ran real vs. proxy this pass.

The pod is **always** stopped on exit — even on error — unless you pass `--keep-pod`. GPU time
is billed (~$0.74/hr); the network volume persists, so stopping loses nothing.

## Prerequisites

Machine-specific paths and keys live in the repo-root **`.env`** (gitignored). Copy the template
and fill it in once:

```bash
cp .env.example .env        # then edit: local Blender + SMPL-X paths, pod access, on-pod paths
```

| Need | `.env` var | Notes |
|---|---|---|
| Repo venv (`torch`+`smplx`+`matplotlib`) | `PITCH3D_PY_LOCAL` | default `.venv/bin/python`; build with [`local_setup.sh`](../scripts/local_setup.sh) |
| SMPL-X body model (MPI-gated) | `PITCH3D_SMPLX_MODELS` | dir with `smplx/SMPLX_NEUTRAL.npz` |
| Blender 5.x binary (optional) | `PITCH3D_BLENDER` | photoreal Cycles render; without it the demo uses matplotlib |
| RunPod access (FULL only) | `RUNPODCTL`, `POD_SSH_KEY`, `POD_NAME_GLOB` | see [`runpod-runbook.md`](runpod-runbook.md) |
| On-pod clip + backends (FULL only) | `PITCH3D_CLIP`, `PITCH3D_SMPLESTX_REPO`, `PITCH3D_WASB_CKPT`, … | staged on the persistent volume; see [`runpod-agent-setup.md`](runpod-agent-setup.md) |
| Real field calibration (optional) | `PNLCALIB_REPO` (+ weights) | blank → proxy calibration; set to a staged checkout to run real PnLCalib |

`scripts/demo.sh --help` reprints all flags and env overrides.

## Flags & overrides

| Flag / env | Effect |
|---|---|
| `--local` | Skip the pod; render bodies from an existing export under `$OUT_LOCAL`. |
| `--keep-pod` | Do **not** stop the pod at the end (debugging). |
| `--frames N` | Frames to reconstruct (default `8`). |
| `FRAMES`, `MESH_FRAME`, `OUT_LOCAL` | Env overrides; `MESH_FRAME` defaults to the scene mid-frame. |

## Artifacts (under `$OUT_LOCAL`, default `out/demo/`)

| Artifact | Path | What it is |
|---|---|---|
| **3D bodies** (the headline) | `mesh/blender_scene.png` (+ `_hero.png`) | recovered SMPL-X bodies on a virtual pitch (Cycles), or `mesh/scene_meshes.png` (matplotlib) |
| Reprojection overlay | `render/scene-1_preview/frame_*.png` | resolved player/ball markers — honest reprojection, **not** a broadcast composite |
| Tactical radar | `observations/scene-1_radar_*.png` | top-down positions on the pitch |
| SMPL-X export | `export/scene.smplx_npz/subject_*.npz` | per-player body params (betas, pose, world transl) |

## What is real vs. proxy

The summary banner names the actual per-port lineup for the pass. On a fully-staged pod the real
adapters are **detect (RF-DETR) · track (ByteTrack) · pose (SMPLest-X-H) · ball (WASB)**;
calibration is **real PnLCalib** only if `PNLCALIB_REPO` is staged, otherwise a **proxy**.

Measured accuracy comes from **separate** benchmarks, not this clip:

- Field calibration ≈ **0.236 m** median reprojection (SoccerNet).
- Pose ≈ **0.51 m** Local MPJPE (3DPW, condition A).

The overlay shows reprojection markers, not a textured broadcast composite — that honesty is
deliberate (see the project's R-6 / no-overclaim rule).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `venv … missing deps` | run [`local_setup.sh`](../scripts/local_setup.sh), or point `PITCH3D_PY_LOCAL` at the right interpreter |
| `SMPL-X model missing` | the model is MPI-gated; stage it and set `PITCH3D_SMPLX_MODELS` ([`blender-demo.md`](blender-demo.md)) |
| `Blender not found` | demo auto-falls back to matplotlib; set `PITCH3D_BLENDER` for the photoreal render |
| `clip missing on pod` | stage a broadcast clip on the volume and set `PITCH3D_CLIP` |
| Pod left running | the trap stops it on exit; if `--keep-pod` was used, run `scripts/pod.sh down` |
