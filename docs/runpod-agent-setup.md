# RunPod from scratch — agent runbook for the full heavy-model stack

**Audience:** an autonomous coding agent (Claude Code or equivalent) running **on the RunPod GPU
box**, with a shell, network, and an injected Hugging Face token. The human launches you on the box
and hands you this file.

**Goal:** take a freshly-rented `cuda12.4.1` RTX 4090 pod and turn it into a working pitch3d
heavy-dev box — base package installed and GPU-verified, **plus every model repo, weight, and dataset
we picked for the three perception backends installed onto a big persistent disk.** Quality is the
priority, not speed/cost; download the real assets, do not stub.

**This doc only covers what the existing docs don't.** The pod-creation table, SSH wiring, and the
Community-vs-Secure / image-CUDA lessons live in [`runpod-runbook.md`](runpod-runbook.md); what a
GPU box buys and the on-box installer internals live in [`cloud-dev.md`](cloud-dev.md) +
[`../scripts/cloud_setup.sh`](../scripts/cloud_setup.sh). This file adds the three missing pieces:
**(a) sizing/growing the disk for the full stack, (b) handling the gated HF token securely after the
2026-06-21 leak, (c) installing the model repos + weights + datasets onto the persistent volume.**
Why these backends (SMPLest-X+SMART / WASB / PnLCalib) and how they plug in: ADR-0006 (the
dotted-path backend seam) and the WorldPose evidence in the project notes.

---

## Prereqs — disk sizing (CONTROL PLANE, done by the human before launching the agent)

> You (the in-box agent) **cannot resize your own pod's volume** — a volume grow only takes effect on
> a `stop-pod` + `start-pod`, which kills your session. So disk sizing is a human/control-plane step.
> If the box is already big enough (check with `df -h /workspace`), skip to Step 0.

The full stack is ~30 GB of weights/code/annotations **before** any WorldPose footage or fine-tune
checkpoints. Size the disk for what comes next, not just today.

| asset | approx size | disk |
|---|---|---|
| CUDA torch + venv | ~4 GB | persistent |
| SMPLest-X "Huge" (ViT-H) weights | ~8.2 GB | persistent |
| SAM 3D Body (`model.ckpt` + `mhr_model.pt`) | ~2.5 GB | persistent |
| SAM3 (`model.safetensors`) | ~3.3 GB | persistent |
| SMPL-X model (zip + extracted) | ~2 GB | persistent |
| PnLCalib + WASB weights | ~2 GB | persistent |
| WorldPose **Light** annotations (no video) | ~0.07 GB | persistent |
| WorldPose footage (if/when pulled for fine-tune) | tens of GB | persistent |
| fine-tune checkpoints + RAFT + scratch | 10–20 GB | persistent |

**Recommendation:** container disk **40–60 GB** (ephemeral, for the OS/venv build); **persistent
`/workspace` volume ≥ 100 GB** — bump to **≥ 200 GB** if you will pull WorldPose video for the SMART
fine-tune. Everything heavy goes on `/workspace` so it survives stop/start.

**Set it at create-time** (cheapest, no downtime): in the `create-pod` params from
[`runpod-runbook.md` §2](runpod-runbook.md), set `containerDiskInGb: 60` and a `volumeInGb: 100`
(or 200) at `volumeMountPath: "/workspace"`.

**Grow an existing pod** (volumes are **grow-only**): `update-pod` (MCP) or `runpodctl pod update`
to raise `volumeInGb`, then `stop-pod` + `start-pod` — the new size appears on the next start. The
GPU is released and re-acquired across the cycle (small availability risk on Secure), and the mapped
SSH port changes, so re-read `get-pod` after start (see runbook §3).

---

## Step 0 — Orient, then pin every path to the persistent disk

```bash
nvidia-smi                                  # confirm GPU + note max CUDA (must be >= 12.4)
df -h /workspace || df -h /                 # confirm the big disk; if no /workspace, use / and warn
export WS=/workspace                        # the persistent disk — ALL heavy assets live here
mkdir -p "$WS"/{repos,weights,datasets,.hf,.pip,.torch}
```

Redirect **all** caches to the big disk so nothing fills the 40–60 GB container disk:

```bash
export HF_HOME="$WS/.hf"                     # Hugging Face cache + token
export PIP_CACHE_DIR="$WS/.pip"
export TORCH_HOME="$WS/.torch"               # torch.hub / RF-DETR base download target
# Persist for future shells/sessions:
printf '\nexport WS=/workspace\nexport HF_HOME=$WS/.hf\nexport PIP_CACHE_DIR=$WS/.pip\nexport TORCH_HOME=$WS/.torch\n' >> ~/.bashrc
```

---

## Step 1 — Hugging Face token (FIRST, and handle it securely)

> **Security — read this.** On 2026-06-21 an HF token leaked because it was embedded in plaintext in
> a git remote URL (`https://user:hf_...@huggingface.co/...`) inside a cloned repo's `.git/config`.
> That token was revoked; this is the replacement. **Never** put the token in a git remote URL, never
> `git add` a file that contains it, never echo it into a committed doc. Authenticate via the HF CLI
> (it stores the token under `$HF_HOME`, outside any repo) and let `git`/`hf` read it from there.

The human injects the token as a **pod environment variable / RunPod secret** named `HF_TOKEN` (set at
create-time or via `update-pod env`). Do **not** hardcode it. Then:

```bash
test -n "$HF_TOKEN" || { echo "HF_TOKEN not set — ask the human to inject it as a pod secret"; }
pip install -U "huggingface_hub[cli]"
hf auth login --token "$HF_TOKEN"            # older alias: huggingface-cli login --token ...
hf auth whoami                               # expect: chubuchnyi
```

The two **gated** asset families (SAM weights `facebook/sam-3d-body-dinov3` + `facebook/sam3`, and the
WorldPose dataset `tijiang13/FIFA-Skeletal-Tracking-Light-2026`) also need their **terms accepted once
in a browser** under this account — if a download 401/403s, that acceptance is missing, not the token.

---

## Step 2 — Base package: clone, install, verify GPU

The repo `chubuchnyi/fifa` is **public** → clone over HTTPS with **no auth** (no key on the box):

```bash
cd "$WS/repos"
git clone https://github.com/chubuchnyi/fifa.git && cd fifa     # venv lands on /workspace
./scripts/cloud_setup.sh                                        # pinned cu124 torch + reals + verify
# expect the verify block to print: cuda available: True  /  device: NVIDIA GeForce RTX 4090
PYTHONPATH=src python -m pytest                                 # core suite — must be green, GPU-free
```

`cloud_setup.sh` installs `torch==2.6.0`+`torchvision==0.21.0` (cu124) **before** the extras, then
`-e ".[cv,hmr,ball,export,mcp,dev]"`. Keep this `torch` build fixed — see the conflict note in Step 3.

> **Blackwell (sm_120) box?** This runbook assumes a cu124 / RTX 4090 pod. On a Blackwell pod (cu128
> image, torch 2.8.0) **do not** run the line above as-is — run it in reuse mode so the script keeps
> the image's torch: `PITCH3D_VENV=/workspace/.venv PITCH3D_REUSE_SYSTEM_TORCH=1 ./scripts/cloud_setup.sh`
> (or `just cloud-setup-blackwell`). The why + the gpuTypeIds/creation traps are in
> [`runpod-runbook.md` §2](runpod-runbook.md).

---

## Step 3 — Model code repos (the ADR-0006 backends live in *your* tree, not core)

Clone under `$WS/repos`. These are research repos with their own dependency sets; install per their
READMEs. **Conflict rule:** the box's torch is **2.6.0+cu124** and must stay that way. If a repo's
`requirements.txt` would pull a different torch/torchvision, install it with `pip install -r ... --no-deps`
and add only the genuinely-missing deps, **or** give that repo its own venv and bridge it through the
dotted-path seam (a backend only needs to be importable on `PYTHONPATH`, not share pitch3d's venv).

```bash
cd "$WS/repos"
git clone https://github.com/MotrixLab/SMPLest-X.git            # PRIMARY pose backbone (SMPL/SMPL-X native)
git clone https://github.com/mguti97/PnLCalib.git               # calibration (GPL-2.0) — highest leverage
git clone https://github.com/nttcom/WASB-SBDT.git               # ball tracking (WASB, MIT)
git clone https://github.com/facebookresearch/sam-3d-body.git   # FALLBACK per-crop pose (outputs MHR)
git clone https://github.com/facebookresearch/MHR.git           # MHR rig (needed to convert SAM-3DB output)
git clone https://github.com/FIFA-Skeletal-Light-Tracking-Challenge/FIFA-Skeletal-Tracking-Starter-Kit-2026.git
# (SAM3 video-seg code, if you wire the SAM-Body4D fallback, is github.com/facebookresearch/sam3)
```

Follow each repo's README/INSTALL for its package install. Do **not** invent download URLs — use the
exact commands the repo documents.

---

## Step 4 — Open weights (no token needed)

- **SMPLest-X** — open weights. Follow the SMPLest-X repo's *Pretrained Models / Preparation* section;
  pull the **Huge (ViT-H, ~8.2 GB)** checkpoint into `$WS/weights/smplest-x/`. This is the primary
  pose net.
- **PnLCalib** — weights ship in the repo's **v1.0.0 GitHub release** (and the world-template pitch
  keypoints are in `utils/utils_keypoints.py`). Download the release assets into `$WS/weights/pnlcalib/`.
- **WASB** — soccer weights are in the repo's Google-Drive **model zoo** (linked from its README).
  Pull the soccer checkpoint into `$WS/weights/wasb/`.
- **RF-DETR base** (~370 MB) auto-downloads into `$TORCH_HOME` on the first real detection run — no
  manual step.

---

## Step 5 — Gated weights + the WorldPose dataset (token from Step 1)

```bash
# SAM 3D Body weights (gated; needs accepted terms + token) — primary FALLBACK pose net
hf download facebook/sam-3d-body-dinov3 --local-dir "$WS/weights/sam-3d-body-dinov3"
#   yields: model.ckpt (~2.0 GB), assets/mhr_model.pt, model_config.yaml (DINOv3 ViT-H/16+, MHR head)

# SAM3 (gated) — only if wiring the SAM-Body4D occlusion fallback (video seg)
# hf download facebook/sam3 --local-dir "$WS/weights/sam3"

# WorldPose / FIFA Skeletal-Tracking "Light" — our domain's GT benchmark (gated DATASET)
hf download tijiang13/FIFA-Skeletal-Tracking-Light-2026 --repo-type dataset \
  --local-dir "$WS/datasets/worldpose-light"
#   "Light" = ~65 MB annotations only (boxes / cameras K_t / 3D skel / pitch landmarks) — NO video.
#   The actual broadcast footage for the SMART fine-tune is a separate, much larger pull; only fetch
#   it once you reach the fine-tune step (and after sizing the disk to >= 200 GB).
```

If any of these 401/403: the account hasn't accepted that repo's terms in a browser yet (Step 1 note).

---

## Step 6 — SMPL-X body model (MANUAL transfer — the agent cannot download it)

The SMPL-X model is behind an MPI login form (non-commercial license) — there is **no token-based
download**. The human already has the verified archive locally at `AVATAR/SMPL-X/models_smplx_v1_1.zip`
(870 MB). **From the local machine** (not from inside the box), copy it up — build the SSH endpoint
from `get-pod` (`publicIp` + mapped `22` port), as in runbook §3:

```bash
# --- run this on the LOCAL machine ---
scp -P <mappedPort> -i ~/.ssh/id_ed25519_runpod \
  ~/AVATAR/SMPL-X/models_smplx_v1_1.zip  root@<publicIp>:/workspace/weights/
```

Then, **on the box**, verify + extract the NEUTRAL model (what we use for soccer):

```bash
cd "$WS/weights"
test "$(stat -c%s models_smplx_v1_1.zip)" = "870108517" && echo "size OK"
mkdir -p smplx && unzip -o models_smplx_v1_1.zip 'models/smplx/SMPLX_NEUTRAL.npz' -d smplx/
ls -la smplx/models/smplx/SMPLX_NEUTRAL.npz       # ~108 MB
```

> This file is MPI-licensed and **gitignored** (`/SMPL-X/`, `*.npz`, `*.pkl`) — it must never enter
> the public repo. Keep it only under `$WS/weights`.

---

## Step 7 — Verify the stack

```bash
cd "$WS/repos/fifa"
python - <<'PY'
import torch, os
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
import smplx; print("smplx OK")
for p in ["smplest-x", "pnlcalib", "wasb", "sam-3d-body-dinov3"]:
    d = f"{os.environ['WS']}/weights/{p}"
    print(p, "->", "present" if os.path.isdir(d) and os.listdir(d) else "MISSING", )
print("worldpose-light:", os.listdir(f"{os.environ['WS']}/datasets/worldpose-light")[:5])
PY
# Golden path on the real GPU pipeline (detection+tracking are already wired):
PYTHONPATH=src python -m pitch3d --clip clip.mp4 --frames 6 \
  --detector rfdetr --tracker bytetrack --device cuda \
  --render overlay --export gltf --format smplx_npz --out-dir out/cuda
```

(`samples/` isn't in the repo — `scp` a short broadcast clip up, or omit `--clip` for a synthetic one.)
Wiring the backends into the stubs is task #73, in leverage order: **PnLCalib + foot-anchoring →
SMPLest-X/SMART pose → WASB**, then eval on WorldPose in metres. This runbook only gets the assets
onto the box; it does not wire them.

---

## Step 8 — Disk hygiene + cost

```bash
df -h "$WS"                                   # watch headroom as downloads land
runpodctl pod stop <id>                       # or MCP stop-pod — stops GPU billing; /workspace persists
```

- Delete duplicate downloads (e.g. a stray second copy of a 3.3 GB SAM3 checkpoint) rather than
  letting the disk fill.
- Billing runs the whole time the pod is `RUNNING` (~$0.69/hr Secure 4090). **Stop it when idle**;
  `/workspace` survives so the next start resumes with everything installed.
- Never `git add` anything under `$WS/weights` or `$WS/datasets`, and never a file containing the
  token — the repo is public.

---

### The launch prompt (what the human pastes into the agent on the box)

> You are setting up a RunPod GPU box for the pitch3d project. Read
> `docs/runpod-agent-setup.md` in `https://github.com/chubuchnyi/fifa` and execute it end to end:
> verify the GPU, pin all caches to `/workspace`, authenticate to Hugging Face using the `HF_TOKEN`
> pod secret (never put it in a git URL or a committed file), clone + `cloud_setup.sh` the repo, then
> install every model repo, weight, and the WorldPose-Light dataset onto `/workspace`. Report disk
> usage and any 401/403 (means terms not accepted) or torch-version conflicts. Do not wire the
> backends yet — only install the assets. Stop and ask before resizing the pod volume.
