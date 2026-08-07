# `models/` — every downloaded weight, in one place

`models/` is gitignored and holds **~48 GB** (re-measured 2026-08-07; half of it is `worldpose/`). Nothing in it is redistributable: SMPL-X is
MPI non-commercial, the SAM families are behind Meta's gated repos. It is the *only*
place weights should land — if you find a checkpoint elsewhere on the box, move it here.
**Unenforced, and currently violated:** `rf-detr-base.pth` (355 M, auto-downloaded into the working
dir on first detection run) and `yolo11x-pose.pt` (113 M) sit at the repo root, and `sam3/` holds
*two* copies of the same 3.3 G checkpoint (`sam3.pt` and `model.safetensors`).
Research-repo *code* checkouts are the matching gitignored `backends/`.

Consolidated 2026-08-05 from four scattered locations (`~/sam3`, `~/sam3dbody`,
`AVATAR/SMPL-X`, `backends/PromptHMR/data`) plus the two framework caches. No compatibility
symlinks were left at the old paths, so a stale absolute path fails loudly instead of
silently reading a second copy.

| Dir | Size | What | Reached by |
|-----|------|------|-----------|
| `smplx/` | 1.2 G | SMPL-X neutral body model + the MPI download zip | `$PITCH3D_SMPLX_MODELS`, else the repo-local default `models/smplx` (`smplx_lbs.locate_smplx_model`) |
| `worldpose/` | 24 G | WorldPose GT — `WorldPose Dataset/{raw,compressed}` (22 G) + `FIFA Challenge 2026 Video Data/Videos/` (1.8 G). The frames `pose-bakeoff-runbook.md` used to call blocking | eval scripts; see [`pose-bakeoff-runbook.md`](pose-bakeoff-runbook.md) |
| `prompthmr/` | 3.4 G | PromptHMR image + video checkpoints, configs, body models | `scripts/check_prompthmr_weights.py`, `scripts/prompthmr_mask_ab.py` |
| `sam3/` | 6.5 G | SAM 3 (`facebook/sam3`) | not wired yet — needs its own env (transformers 5.x vs our 4.57.6 pin) |
| `sam3d-body/` | 2.7 G | SAM 3D Body + MHR rig (`facebook/sam-3d-body-dinov3`) | `adapters/models/sam3dbody_backend.py` (pod only) |
| `smplest-x/` | 11 G | SMPLest-X Huge (ViT-H) — our per-crop pose primary (`waanqii/SMPLest-X`, ungated). **Two** checkpoints: shipped `smplest_x_h.pth.tar` (8.2 G) and `smplest_x_h_slim.pth.tar` (2.75 G) | `adapters/models/smplestx_backend.py` via `$PITCH3D_SMPLESTX_REPO/pretrained_models/<ckpt_name>/`, linked by `scripts/stage_smplestx_weights.sh` |
| `cutie/` | 173 M | Cutie video object segmentation — the temporal mask cue for #133 (`hkchengrex/Cutie` v1.0, MIT) | `adapters/models/mask_propagation.py` via `$PITCH3D_CUTIE_REPO/weights/`, symlinked from here |
| `hf/` | 443 M | Hugging Face cache — SAM ViT-B, DINOv2-small | `$HF_HOME` |
| `torch/` | 1.3 G | torch.hub cache — MaskPose-b, RTMDet-ins-l-mask | `$TORCH_HOME` |

## The two things that will bite you

**`prompthmr/data/` is named after upstream, not after us.** PromptHMR hardcodes its asset
paths relative to the process cwd (`data/pretrain/…`, `data/body_models/…` — see
`prompt_hmr/models/phmr.py:11-13`). So our scripts `chdir` into `models/prompthmr` and put
the *code* checkout on `sys.path` instead. Renaming that `data/` level breaks the backend.

Its `data/body_models/smplx/SMPLX_NEUTRAL.npz` is a relative symlink into `models/smplx/`
— the same licensed 104 MB file, not a second copy.

**`HF_HOME` / `TORCH_HOME` only apply if the shell exports them.** They live in `.env`, and
`.env` is read by the pod shell scripts and by `pitch3d.env.load_env()` — *not* by a bare
`.venv/bin/python -m pitch3d.app.cli`. That run will re-download into `~/.cache`. To be sure:

```bash
set -a; . ./.env; set +a
```
