# Multi-angle video demo (on the pod)

`scripts/demo_video.sh` turns **one broadcast clip into a multi-angle 3D animation** — the
recovered SMPL-X bodies **and the ball**, rendered from several fixed cameras — and pulls the
finished `.mp4`s back to your machine. Unlike [`demo.sh`](demo.md) (which renders a *still* of the
bodies locally), the whole heavy half here runs **on the GPU pod**: your laptop needs no Blender and
no SMPL-X.

```bash
scripts/demo_video.sh                                   # default clip, 60 frames, 4 angles
scripts/demo_video.sh --clip samples/video/your.mp4     # another clip
scripts/demo_video.sh --reuse-scene --frames 30         # cheap re-render: skip reconstruction
```

The pod is **always** stopped on exit — even on error — unless you pass `--keep-pod`. GPU time is
billed (~$0.74/hr); the network volume persists, so stopping loses nothing.

## What it does, step by step

The local driver ([`demo_video.sh`](../scripts/demo_video.sh)) is thin — it orchestrates the pod:

1. **Preflight** the chosen clip exists locally.
2. **Bring up the GPU pod** — reuses a live pod or resumes a stopped one, with host failover
   (see [`pod.sh`](../scripts/pod.sh)).
3. **Sync** `src/` + `scripts/` to the pod (`/workspace/fifa`).
4. **Stage the clip** on the pod (`/workspace/<clip>`), skipped if a same-size copy is already there.
5. **Run the render on the pod** — [`pod_make_video.sh`](../scripts/pod_make_video.sh) does the work
   (the four-step chain below).
6. **Pull** `out/anim/video/*.mp4` back to the local machine (`$OUT_LOCAL/video`, default `out/anim`).
7. **Stop the pod** (billing returns to volume-only).

### The pod-side chain ([`pod_make_video.sh`](../scripts/pod_make_video.sh))

1. **Real reconstruction → canonical JSON** — reuses [`pod_real_e2e.sh`](../scripts/pod_real_e2e.sh)
   (`RF-DETR detect → ByteTrack → SMPLest-X-H pose (0.69B) → WASB ball`). Exported as **JSON
   because JSON is the only format that carries the ball**. `--reuse-scene` skips this ~3-min step
   when `out/anim/export/scene.json` already exists on the volume.
2. **Anim export** — [`anim_export.py`](../scripts/anim_export.py) forwards **every** frame of each
   subject through SMPL-X and resolves the ball, writing `anim_subject_*.npz` (z-up world verts +
   per-subject frame indices + team colour) and `ball.npz`. Each subject keeps its **own** frame
   range — real tracks come and go.
3. **Ensure Blender + render** — [`pod_ensure_blender.sh`](../scripts/pod_ensure_blender.sh) makes a
   usable Blender available (see below), then [`blender_animate.py`](../scripts/blender_animate.py)
   renders **N fixed cameras × all frames** with Cycles. Cameras are static; players and ball move
   within frame, and each body is hidden on frames where its track is absent.
4. **Stitch** — `ffmpeg` turns each camera's PNG sequence into one `out/anim/video/<camera>.mp4`.

## Blender on the pod (the non-obvious bit)

The pod venv is **Python 3.12**, and the `bpy` PyPI module ships wheels only for specific CPython
versions (e.g. 3.11) — **there is no `bpy` wheel for 3.12**. So
[`pod_ensure_blender.sh`](../scripts/pod_ensure_blender.sh) falls back to a **standalone Blender
binary**, downloaded **once** to the persistent volume (`/workspace/blender`, default Blender 4.2
LTS) and reused thereafter. It also `apt-get`s `ffmpeg` plus the X/GL shared libs Blender links even
under `--background` (`libxrender1`, `libxxf86vm1`, `libxfixes3`, `libxi6`, `libxkbcommon0`,
`libsm6`, `libgl1`). The script emits exactly one line — `BLENDER_MODE=module` or
`BLENDER_MODE=binary:/path/blender` — which the orchestrator turns into the right invocation. On a
pod whose Python *does* have a `bpy` wheel, the module path is taken automatically and nothing is
downloaded.

## Flags & overrides

| Flag | Effect | Default |
|---|---|---|
| `--clip PATH` | Broadcast clip to reconstruct | `samples/video/Colombia-1-0-Congo-DR1080p.mp4` |
| `--frames N` | Frames to reconstruct & animate | `60` |
| `--cameras LIST` | Comma list of `broadcast,sideline,top,goal` | all four |
| `--res WxH` | Render resolution | `1280x720` |
| `--samples N` | Cycles samples/px | `32` |
| `--device gpu\|cpu` | Pod render device (GPU falls back to CPU automatically) | `gpu` |
| `--reuse-scene` | Reuse the pod-side `scene.json` (skip reconstruction — cheap re-render) | off |
| `--keep-pod` | Do **not** stop the pod at the end (debugging) | off |

`OUT_LOCAL` (env) sets the local output dir. `scripts/demo_video.sh --help` reprints all flags.

## Prerequisites

Same machine `.env` as the rest of the GPU flow (see [`demo.md`](demo.md) and
[`runpod-runbook.md`](runpod-runbook.md)): RunPod access (`RUNPODCTL`, `POD_SSH_KEY`,
`POD_NAME_GLOB`) and the on-pod backend paths (`PITCH3D_SMPLESTX_REPO`, `PITCH3D_SMPLX_MODEL_PATH`,
`PITCH3D_WASB_*`, optional `PNLCALIB_REPO`). Everything Blender-related is installed **on the pod**
on first use — there is **nothing local to set up** for this demo. Optional:
`PITCH3D_BLENDER_TARBALL_URL` overrides which Blender build the pod downloads.

> **`.env` leak gotcha.** `anim_export.py`'s SMPL-X models dir comes from `PITCH3D_SMPLX_MODELS`,
> which in a local `.env` points at *this machine's* SMPL-X dir. The driver deliberately re-derives
> it from the **pod** path (`PITCH3D_SMPLX_MODEL_PATH`) before sending the ssh command, because
> `${VAR:-default}` in a command string expands **locally** — leaving the local default in would
> ship a non-existent path to the pod (`Unknown model type`).

## Artifacts (under `$OUT_LOCAL`, default `out/anim/`)

| Artifact | Path | What it is |
|---|---|---|
| **Multi-angle videos** (the headline) | `video/<camera>.mp4` | one `.mp4` per requested camera — bodies + ball, same instant from each angle |
| Per-camera PNG sequences | `mesh/frames/<camera>/frame_*.png` | the rendered frames `ffmpeg` stitched (pod-side; not pulled by default) |
| Posed meshes | `mesh/anim_subject_*.npz`, `mesh/ball.npz` | z-up world verts per frame + the resolved ball track (pod-side) |
| Canonical scene | `export/scene.json` | the reconstruction JSON `--reuse-scene` re-renders from (pod-side) |

## What is real vs. proxy

The reconstruction is the same real lineup as [`demo.md`](demo.md): **detect (RF-DETR) · track
(ByteTrack) · pose (SMPLest-X-H) · ball (WASB)**; calibration is **real PnLCalib** only if
`PNLCALIB_REPO` is staged, else a proxy. The bodies are recovered SMPL-X meshes; the ball is the
resolved 3D track. Measured accuracy comes from **separate** benchmarks, not this clip — field
calibration ≈ **0.236 m** (SoccerNet), pose ≈ **0.51 m** Local MPJPE (3DPW). The render is a clean
synthetic pitch, **not** a textured broadcast composite — that honesty is deliberate (R-6 /
no-overclaim).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `BLENDER_ENSURE_FAILED … binary not runnable` | the headless GL libs failed to install, or the tarball URL is unreachable — check `PITCH3D_BLENDER_TARBALL_URL`, retry (the download caches on the volume) |
| `No matching distribution found for bpy` | expected on Python 3.12 — the script then falls back to the binary; only a problem if the binary path *also* fails |
| `missing out/anim/export/scene.json` | a `--reuse-scene` run with no prior scene — drop the flag once to reconstruct |
| `Unknown model type` in anim export | `PITCH3D_SMPLX_MODELS` leaked a local path — see the leak gotcha above |
| Pod left running | the trap stops it on exit; if `--keep-pod` was used, run `scripts/pod.sh down` |
