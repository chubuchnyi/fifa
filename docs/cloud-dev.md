# Cloud development on a GPU box

A reproducible path for moving MVP development onto a rented GPU machine. The local
profile stays CPU-only (ADR-0009: `device` is a runtime knob; `--device cpu` is the
validation default, `--device cuda` the production target — **zero code change** to
switch). This page is the "spin up → clone → verify GPU → golden path on `--device cuda`"
checklist; [`scripts/cloud_setup.sh`](../scripts/cloud_setup.sh) does the install half.

## What a GPU box buys you (and what it doesn't)

**Unblocks**
- Running the **real** detection + tracking path on the GPU: RF-DETR (`--detector rfdetr`)
  + ByteTrack (`--tracker bytetrack`), the two adapters that are already wired (ADR-0009).
- The *ability to wire* the GPU-bound backends — GVHMR (pose), TrackNet (ball) — against a
  real CUDA torch, which the local CPU box can't host comfortably.

**Does NOT, on its own, make these work** — they are still **unwired stubs**, because their
upstreams aren't pip-installable; the extras ship only the substrate (`torch`/`smplx`/`chumpy`):
- `--pose gvhmr` · `--ball tracknet` · `--calibrator keypoints` still raise until wired.

So the *runnable* real pipeline on a fresh GPU box today is **real detection/tracking +
the dependency-free reals** (overlay render, top-down radar, SMPL-X/glTF export) — none of
which need a GPU, but all of which run end-to-end on the GPU box on a real clip.

## Box spec

| Resource | Recommended | Why |
|---|---|---|
| GPU | ≥ 24 GB VRAM (RTX 4090 / A5000 / L4 / A10) | headroom to wire HMR (SMPL-X) later; RF-DETR alone fits in far less |
| Driver / CUDA | driver supporting **CUDA 12.x** | matches the `cu124` torch wheel (override `PITCH3D_CUDA` for cu118/cu126) |
| Disk | ~30 GB | CUDA torch wheel (~2.5 GB) + venv + auto-downloaded weights (RF-DETR base ≈ 370 MB) |
| System pkg | `ffmpeg` (gives `ffprobe`) | the real video ingestor (`--clip`) shells out to `ffprobe` |

Provider notes (any will do — Claude Code runs over SSH exactly like local): **RunPod** /
**Vast.ai** (cheapest spot 4090s), **Lambda Cloud** (A10/A100), **AWS** `g5.xlarge` (A10G).
Pick an image with the NVIDIA driver pre-installed (a `cuda:12.x` or PyTorch base image).

## Checklist

```bash
# 1. Spin up the box (driver + ffmpeg present), then SSH in.
nvidia-smi            # confirm the GPU + note the max CUDA version (top-right)

# 2. Clone your work. Push from the local box first — see "Ephemeral disk" below.
git clone <your-remote> pitch3d && cd pitch3d

# 3. Install: CUDA torch + reals + dev tooling, then verify torch sees the GPU.
./scripts/cloud_setup.sh
#   override the CUDA build if nvidia-smi shows e.g. CUDA 11.8 or 12.6:
#   PITCH3D_CUDA=cu118 ./scripts/cloud_setup.sh

# 4. Core suite — GPU-free, must stay green everywhere.
PYTHONPATH=src python -m pytest

# 5. Golden path on the GPU: real detection + tracking, overlay + export.
#    (copy a clip up with scp, or omit --clip for a synthetic one — samples/ isn't in the repo)
PYTHONPATH=src python -m pitch3d \
  --clip clip.mp4 --frames 6 \
  --detector rfdetr --tracker bytetrack --device cuda \
  --render overlay --export gltf --format smplx_npz \
  --out-dir out/cuda
```

`scripts/cloud_setup.sh` is idempotent-ish (re-runnable); tune what it installs with
`PITCH3D_EXTRAS` (default `cv,hmr,ball,export,mcp,dev`).

## Gotchas

- **Ephemeral disk.** Spot/community boxes can vanish with their disk. Treat the box as
  cattle: `git push` your work at every checkpoint, keep nothing precious only on the box.
  If the provider offers a **persistent volume**, clone into it and point the venv there.
- **CUDA tag must match the driver.** `PITCH3D_CUDA` (default `cu124`) must be ≤ the CUDA
  version `nvidia-smi` reports. If `torch.cuda.is_available()` prints `False` after setup,
  this is almost always the mismatch — re-run with the right `cuXXX`.
- **Weights download on first run, and are git-ignored.** RF-DETR base auto-downloads
  (~370 MB) into the working dir; `*.pth/*.pt/*.onnx/*.safetensors` are in `.gitignore`,
  so they never bloat a commit and re-download on a fresh box.
- **`samples/` is not in the repo** (git-ignored, large media). `scp` a short clip up, or
  run without `--clip` for a synthetic one.
- **`bpy` is never installed by the script.** Blender provides its own Python; for live/proxy
  Blender work, install Blender on the box and point `PITCH3D_BLENDER` at the binary (ADR-0003).
