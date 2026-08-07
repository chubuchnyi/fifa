# Local GPU box — `demorig-pc`

A Windows 11 workstation with an **RTX 4080 (16 GB, sm_89)** that runs the full reconstruction
chain in Docker. It replaces the $0.74/hr RunPod pod for everything up to the generative tail.

Set up 2026-08-07. The pod runbook is [`runpod-runbook.md`](runpod-runbook.md); this is its
local counterpart, and the two boxes are **not** interchangeable — see "What it cannot do".

---

## 1. Connect

```bash
ssh demorig            # alias in the operator's ~/.ssh/config -> 172.16.10.203, key auth
```

Key auth is installed, so no password is needed. The credential is **not** in this repo — it is
public. If the alias is missing, the host is `172.16.10.203`, user `user`; ask the operator.

Two things about the far side that will waste your time otherwise:

- **The SSH shell is `cmd.exe`, not bash.** `head`, `grep`, `tail`, `|`, `>` are all either
  missing or interpreted by cmd. Everything real happens one level in, inside WSL.
- **PowerShell redirects (`>`) write UTF-16.** Piping such a log through `grep` yields nothing at
  all — grep sees a binary file. `scp` it back and decode, or write with `Out-File -Encoding utf8`.

**Do not compose commands through `ssh → cmd → wsl → bash`.** Quoting breaks at every hop and the
failure looks like a syntax error inside your script. Copy a `.sh` and run it by path:

```bash
scp myjob.sh demorig:C:/Users/user/
ssh demorig 'wsl.exe -d Ubuntu-24.04 --user root -- bash /mnt/c/Users/user/myjob.sh'
```

## 2. What is on it

| | |
|---|---|
| OS | Windows 11 Pro 26200 → WSL2 Ubuntu 24.04 (`wsl.exe -d Ubuntu-24.04 --user root`) |
| Docker | `docker-ce` 29.7.2 + `nvidia-container-toolkit` 1.19.1, inside WSL (not Docker Desktop) |
| Image | `pitch3d:cu124`, built from [`../docker/Dockerfile`](../docker/Dockerfile) |
| Volume | `/vol` in WSL — **always mount it at `/workspace`** (see §4) |
| Weights | `/vol/weights/{smplest-x,pnlcalib,wasb}`, repos in `/vol/repos`, checkout in `/vol/fifa` |
| SMPL-X | `/models/smplx` (pushed by hand; MPI login, no token download) |

## 3. Run

```bash
# reconstruction, all five real backends
docker run --rm --gpus all -v /vol:/workspace -v /models:/models \
  -e PITCH3D_VOL=/workspace -e PITCH3D_PY=python \
  -e PITCH3D_CLIP=/workspace/fifa/samples/video/Colombia-1-0-Congo-DR1080p.mp4 \
  -e FRAMES=48 -e OUT=out/run \
  -w /workspace/fifa pitch3d:cu124 bash scripts/pod_real_e2e.sh

# suite (6.5 s here vs 71 s on the laptop)
docker run --rm -v /vol/fifa:/app -v /models:/models \
  -e PITCH3D_SMPLX_MODELS=/models/smplx -w /app pitch3d:cu124 python -m pytest

# re-stage weights (idempotent)
docker run -d --name stage -v /vol:/workspace -v /models:/models \
  -w /workspace/fifa pitch3d:cu124 bash docker/stage_weights.sh
```

Code syncs from GitHub — the repo is public and the tracked tree is 5.5 MB, so
`git -C /vol/fifa pull` is seconds. **`samples/` is not tracked**; the clip is pushed by hand.

## 4. Gotchas that each cost a run

- **Mount the volume at `/workspace`.** The repo's scripts and the three model backends carry
  baked-in `/workspace` defaults. `scripts/stage_wasb_weight.sh` ignores `PITCH3D_VOL` outright and
  writes to `/workspace/weights` — under `--rm` its checkpoint vanished with the container, leaving
  an empty dir and no error.
- **WSL kills background jobs when the launching `wsl.exe` exits.** `setsid nohup ... &` does not
  survive; the job died before writing its first log line. Long work must run as a **detached
  container** (`docker run -d`), which dockerd owns.
- **WSL shut the whole VM down when idle**, taking dockerd and every container with it. Fixed by
  `vmIdleTimeout=-1` in `C:\Users\user\.wslconfig` (which also sets memory=48GB, processors=20 —
  the defaults were half the host). Diagnose a recurrence with `uptime` inside WSL: `up 0 min`
  after hours of work means the VM bounced.
- **Docker auto-creates a missing `-v` source directory.** An empty `/vol/fifa` is not a broken
  clone, it is a mount that ran before the clone did.
- **The link resets about every 250 MB.** Any large fetch must resume:
  `huggingface_hub.snapshot_download` restarts from zero and looped between 0 and 335 MB for an
  hour on the 8.2 GB checkpoint; `curl -C - --speed-limit 20000 --speed-time 45` finished it across
  34 resets. `docker/stage_weights.sh` does it that way for this reason.
- **SMPLest-X needs all three SMPL-X genders**, plus `SMPLX_to_J14.pkl`,
  `MANO_SMPLX_vertex_ids.pkl` and `SMPL-X__FLAME_vertex_ids.npy` in `smplx/` — **not** `aux/`.
  `human_models.py:20-32` builds neutral+male+female eagerly, so a neutral-only stage fails at
  POSE, four stages in.
- **`chumpy==0.70` cannot build under modern pip.** Its 2019 `setup.py` imports `pip`, which the
  isolated build env does not provide. The Dockerfile installs it first with
  `--no-build-isolation`; the pod never hit this because its pip is older.

## 5. Measured, 2026-08-07

| | |
|---|---|
| `pod_real_e2e.sh`, 48 frames | **75 s, exit 0** — RF-DETR · ByteTrack · PnLCalib · SMPLest-X-H · WASB → 16 gates → `smplx_npz` |
| **Peak VRAM, full chain** | **3930 MiB = 24 % of 16 GB** (69 samples @1 s) — ~12 GB spare |
| Suite in-container | 1123 passed / 46 skipped in 6.5 s |
| Transfer | laptop→box **0.11 MB/s**; box→internet **3.74 MB/s** (34×). Never push weights from the laptop — have the box pull them |

## 6. The generative tail — measured 2026-08-07, and it does not have room

Built as a separate image, `docker/Dockerfile.gen` (the pod keeps these stages in their own venvs
for the same reason: conflicting torch pins).

**SeedVR2 3B fp16 @720p, `batch_size=33` runs — with under 5% of the card to spare.**

| Run | Result |
|-----|--------|
| Full card | **completes**, 41 frames 1248×720 in 167 s, peak **15972 MiB = 97.5%** |
| `set_per_process_memory_fraction(0.95)` | **OOM** in Phase 3 (VAE decode), needed another 878 MiB |
| 0.75 / 0.60 | OOM in Phase 3 / Phase 1 |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | **does not help** — still OOM at 0.75 |

Two traps in reading those numbers:

- **`nvidia-smi` measures what the caching allocator RESERVED, not what the run needs.** Peak read
  an identical 15972 MiB at `batch_size` 33, 17 **and** 9 — that constancy is the tell. The real
  figure comes from `torch.cuda.max_memory_allocated()` inside the process, or from capping with
  `set_per_process_memory_fraction` and seeing whether it still finishes.
- **`batch_size` is not the lever it looks like.** The OOM is in the VAE encode/decode phase, which
  works over the whole sequence; dropping 33→9 changed runtime (167→66 s) and nothing else.

**So the generative tail stays on the pod.** It "works" here only on an idle, headless box: the
margin is ~400 MiB, and this is a workstation — a login session with a browser open would take
that. The obvious lever if you want it local is the **fp8 variant**
(`seedvr2_ema_3b_fp8_e4m3fn.safetensors`, 3.39 GB against fp16's 6.78 GB), untested, and a quality
change to a deliverable that is judged by eye — so it is the operator's call, not an optimisation
to apply quietly.

Wan2.1-VACE is the safer half and is **not yet measured here**: 1.3B with
`enable_model_cpu_offload()` at the script's default 832×480 (not 720p — that was the OOM case on a
32 GB card). Its download is 19 GB, of which 11.4 GB is the T5 text encoder.

## 7. Also not available here

- **Blender is not installed** in the image, so `anim_export` → Cycles render is pod-only for now.
- The box mirrors `origin/main`. Unpushed local commits are invisible to it.
