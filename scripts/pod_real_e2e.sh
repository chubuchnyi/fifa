#!/usr/bin/env bash
# scripts/pod_real_e2e.sh — one real-model E2E pass on a GPU pod.
#
# Runs the SAME golden path as docs/cloud-dev.md §5 (real video -> RF-DETR detect ->
# ByteTrack -> assemble -> overlay render + smplx_npz/glTF export) but with the wired
# **real SMPLest-X** pose + **real WASB** ball backends injected by dotted path (ADR-0006)
# — i.e. the genuine single-cam -> world-SMPL-X path, end to end, on CUDA.
#
# Env (all have sensible pod defaults — override to point elsewhere). $VOL below is the network
# volume, found by looking for `fifa` under /workspace then /runpod (see the resolver):
#   PITCH3D_VOL=<auto>                      network-volume mount; set it to skip the search
#   PITCH3D_REPO=$VOL/fifa                  repo root (PYTHONPATH=src is set from here)
#   PITCH3D_PY=$VOL/.venv/bin/python        interpreter with cuda torch + rfdetr + smplx
#   PITCH3D_CLIP=<required>                 input broadcast clip — no default, see below
#   FRAMES=8                                --frames
#   OUT=out/run                             --out-dir (relative to repo)
#   FORMAT=smplx_npz                        --format (json carries the ball; smplx_npz = bodies)
#   STITCH=1                                continuity stitching (ON by default; set 0 to disable)
#   COHERENCE=1                             --coherence (gap-fill + smoothing; off unless =1)
# The SMPLest-X backend reads its own env (PITCH3D_SMPLESTX_REPO / _CKPT / _DEVICE;
# defaults /workspace/repos/SMPLest-X + smplest_x_h on cuda) — see the backend factory.
# The WASB ball backend likewise reads PITCH3D_WASB_REPO / _CKPT / _DATASET / _DEVICE
# (defaults /workspace/repos/WASB-SBDT + wasb_soccer_best.pth.tar on cuda).
set -euo pipefail

# Where the persistent network volume is mounted is a PER-POD setting: most of our pods put it on
# /workspace, some on /runpod — and a pod that mounts it elsewhere still has an empty /workspace
# from the image, so the path existing proves nothing. Resolve it by content (who has `fifa`)
# instead of by name, because guessing wrong here is not loud: PNLCALIB_REPO below would miss and
# the run would silently fall back to the proxy calibrator, i.e. the #203 depth collapse.
VOL="${PITCH3D_VOL:-}"
if [ -z "$VOL" ]; then
  for c in /workspace /runpod; do [ -d "$c/fifa" ] && { VOL="$c"; break; }; done
  VOL="${VOL:-/workspace}"
fi

REPO="${PITCH3D_REPO:-$VOL/fifa}"
PY="${PITCH3D_PY:-$VOL/.venv/bin/python}"
# No default clip, deliberately. This used to be `${PITCH3D_CLIP:-$VOL/clip.mp4}`, and twice
# (2026-07-03, 2026-08-01) that default quietly reconstructed a stale stock video that happens to
# sit at that path instead of the target match. Nothing downstream can catch it: PnLCalib finds no
# pitch in the wrong footage, so every homography is the identity carry — "one pixel is one metre" —
# and the scene still exports, with phantom subjects and hundred-metre teleports. The operator has
# to name the footage, because only the operator knows which footage was meant.
CLIP="${PITCH3D_CLIP:-}"
if [ -z "$CLIP" ]; then
  echo "pod_real_e2e.sh: set PITCH3D_CLIP to the clip you mean to reconstruct." >&2
  ls -la "$VOL"/*.mp4 2>/dev/null >&2 || true
  exit 2
fi
[ -f "$CLIP" ] || { echo "pod_real_e2e.sh: clip not found: $CLIP" >&2; exit 2; }
FRAMES="${FRAMES:-8}"
OUT="${OUT:-out/run}"
FORMAT="${FORMAT:-smplx_npz}"

# REAL field calibration is the DEFAULT now (#203). The proxy "fake" calibrator maps the WHOLE
# frame into a 30 m top-down box with NO perspective, so an oblique broadcast collapses every
# player onto a thin world band — the v0 depth/scale bug. Default PNLCALIB_REPO to the pod's staged
# checkout; the `-d` guard falls back to the proxy ONLY when the repo is genuinely absent (e.g. a
# local CPU box). Force the proxy explicitly with PNLCALIB_REPO= (empty). The backend reads its own
# PNLCALIB_WEIGHTS_* paths (pod defaults are baked into pnlcalib_backend.make()).
PNLCALIB_REPO="${PNLCALIB_REPO-$VOL/repos/PnLCalib}"
export PNLCALIB_REPO
# SMPL-X model dir for the T6a v2 foot-plant provider (foot_plant_gate uses SMPL-X FK
# to measure per-subject pelvis-above-foot; foot-plane target is the median of that).
# The anim_export stage (pod_make_video.sh) already ships this — the reconstruction
# stage needs it too now that foot_plant runs via controller.run_reconstruction.
export PITCH3D_SMPLX_MODELS="${PITCH3D_SMPLX_MODELS:-$VOL/repos/SMPLest-X/human_models/human_model_files}"
# The three model backends each carry their own `/workspace/...` default baked into Python, which
# is right for most pods and wrong for a box that mounts the volume elsewhere. Re-point them from
# $VOL here rather than in the adapters: the driver is the layer that knows what a pod looks like.
export PITCH3D_SMPLESTX_REPO="${PITCH3D_SMPLESTX_REPO:-$VOL/repos/SMPLest-X}"
export PITCH3D_WASB_REPO="${PITCH3D_WASB_REPO:-$VOL/repos/WASB-SBDT}"
export PITCH3D_WASB_CKPT="${PITCH3D_WASB_CKPT:-$VOL/weights/wasb/wasb_soccer_best.pth.tar}"
export PNLCALIB_WEIGHTS_KP="${PNLCALIB_WEIGHTS_KP:-$VOL/weights/pnlcalib/SV_kp}"
export PNLCALIB_WEIGHTS_LINES="${PNLCALIB_WEIGHTS_LINES:-$VOL/weights/pnlcalib/SV_lines}"
CALIB_ARGS=()
if [ -n "${PNLCALIB_REPO}" ] && [ -d "${PNLCALIB_REPO}" ]; then
  CALIB_ARGS=(--calibrator keypoints
              --calibrator-backend pitch3d.adapters.models.pnlcalib_backend:make)
  echo "== calibration: REAL PnLCalib (${PNLCALIB_REPO})"
else
  echo "== calibration: PROXY fake — #203 depth COLLAPSE expected (PnLCalib repo absent: '${PNLCALIB_REPO:-unset}')"
fi
# R2 camera propagation (#104): each frame's homography is re-estimated from its +-N neighbours,
# carried on Lucas-Kanade inter-frame motion. CPU, no weights. This is a TRADE, not a free win —
# it removes 92% of the scene swim (0.119 -> 0.011 m) for ~0.004 m of paint accuracy. Set 0 for
# the A/B control run that scores every frame independently, as today's pipeline does.
if [ -n "${CAMERA_CARRY:-}" ] && [ ${#CALIB_ARGS[@]} -gt 0 ]; then
  CALIB_ARGS+=(--camera-carry "$CAMERA_CARRY")
  echo "== camera carry: +-${CAMERA_CARRY} frames (0 = per-frame, no propagation)"
fi

# Continuity stitching is ON by default (part of correct reconstruction): re-link fragmented
# tracklets before POSE so an occluded player keeps ONE identity instead of re-entering as a new
# track id and spawning a phantom body (#202). Set STITCH=0 to disable. Temporal coherence
# (gap-fill via slerp/lerp + auto temporal-smoothing) stays opt-in via COHERENCE=1.
COH_ARGS=()
if [ "${STITCH:-1}" = "1" ]; then echo "== continuity: stitch ON (default)";
else COH_ARGS+=(--no-stitch); echo "== continuity: stitch OFF (STITCH=${STITCH})"; fi
if [ "${COHERENCE:-0}" = "1" ]; then COH_ARGS+=(--coherence); echo "== coherence: --coherence ON"; fi
# HANDOVER=1 → merge two ids that are one human and drop the duplicate mannequin (#135 П3+П2).
# Off by default: it is the only pass that deletes a subject, so a wrong merge erases a player.
if [ "${HANDOVER:-0}" = "1" ]; then COH_ARGS+=(--handover); echo "== handover: --handover ON"; fi
# MAX_EXTEND → cap how far past its last measurement a subject may be extrapolated, in frames.
# Unset = unbounded, the behaviour since #102: a subject measured on 5 frames of 236 is still
# emitted on all 236. Measured 2026-08-09: 47.9 % of f236_res896 and 41.4 % of fan_auto sit
# further from ANY measurement than the 12 frames --coherence refuses to bridge *between two*.
if [ -n "${MAX_EXTEND:-}" ]; then
  COH_ARGS+=(--max-extend-frames "$MAX_EXTEND")
  echo "== edge reach: capped at ${MAX_EXTEND} frame(s) past the last measurement"
fi
# RIGID_CAMERA=1 → after the scene is written, put the ONE fitted camera (#119/#129) into it.
# Without this the scene keeps whatever camera_from_calibration could make of PnLCalib's free
# per-frame homographies — which on the broadcast clip is nothing at all: the closest realizable
# pinhole is ~471 px away, the fit refuses, and controller.py substitutes a synthetic 772 px
# stand-in. Measured 2026-08-08: with the stand-in the overlay residual is 240 px median and only
# 124 of 1088 projected subjects land within 250 px of any detection; with the fitted camera it is
# 8.0 px and 1088. This script never applied it — pod_make_video.sh, pod_physics_ab.sh and
# pod_129_ab.sh do — which is why every scene built here drew players through the wrong camera
# (#61/#140).
# DET_RES → RF-DETR input square. Unset means the per-clip lookup in
# config/detector_resolution.yaml decides (896 for the two clips measured so far). Set it to
# compare resolutions on the same code: 896 cut mid-pitch identity events 89->61 on the
# broadcast clip and 90->58 on the fan clip against 560, for 1.5x the detector.
if [ -n "${DET_RES:-}" ]; then
  COH_ARGS+=(--detector-resolution "$DET_RES"); echo "== detector: input square ${DET_RES} px"
else
  echo "== detector: input square from config/detector_resolution.yaml"
fi
# PHYSICS=1 → M3-9 kinematic gate: clamp impossible root speed/accel, mark teleports (#207).
if [ "${PHYSICS:-0}" = "1" ]; then COH_ARGS+=(--physics); echo "== physics: --physics ON (M3-9 kinematic gate)"; fi
# PHYSICS_PROFILE selects the config/physics.yaml named profile (default / conservative /
# strict / no_smoothing / future_full / safe_new / humanize_teleports). safe_new turns on
# foot_floor + joint + orientation (T1a/b/c) without collision (which introduces accel
# spikes without a compose-order fix). humanize_teleports interpolates ID-swap regions.
# This script is also run standalone, so it cannot rely on a caller having sourced the
# defaults; pull them in here (idempotent) before resolving the profile.
[ -f "$REPO/scripts/video_defaults.sh" ] && . "$REPO/scripts/video_defaults.sh"
PHYSICS_PROFILE="${PHYSICS_PROFILE:-${VIDEO_PHYSICS_PROFILE_DEFAULT:-safe_new}}"
if [ -n "${PHYSICS_PROFILE:-}" ] && [ "${PHYSICS:-0}" = "1" ]; then
  COH_ARGS+=(--physics-profile "$PHYSICS_PROFILE")
  echo "== physics profile: $PHYSICS_PROFILE"
fi
# PLAYER_PROFILES_DIR + AUTO_TUNE = T4 per-player + per-ball online tuning (7-layer filter).
# Persists <PLAYER_PROFILES_DIR>/players/<team>/<jersey>.json across runs.
if [ -n "${PLAYER_PROFILES_DIR:-}" ] && [ "${PHYSICS:-0}" = "1" ]; then
  COH_ARGS+=(--player-profiles-dir "$PLAYER_PROFILES_DIR")
  echo "== player profiles dir: $PLAYER_PROFILES_DIR"
  if [ "${AUTO_TUNE:-0}" = "1" ]; then
    COH_ARGS+=(--auto-tune)
    [ -n "${BALL_ID:-}" ] && COH_ARGS+=(--ball-id "$BALL_ID")
    echo "== auto-tune: ON (ball-id=${BALL_ID:-match_ball_1})"
  fi
fi
# IDENTITY=1 → identity_gate (GTA split + cross-track merge, HSV features).
# Kills per-player kit colour flicker by cleaning ID-swapped tracks before POSE.
if [ "${IDENTITY:-0}" = "1" ] && [ "${PHYSICS:-0}" = "1" ]; then
  COH_ARGS+=(--identity)
  echo "== identity: --identity ON (GTA split + merge)"
fi
# DEMO_EDITS=0 skips the dry-run edit walkthrough so no demo offset/refit correction lands in the
# exported scene (deliverable runs); default 1 keeps the full golden-path seam coverage.
if [ "${DEMO_EDITS:-1}" = "0" ]; then COH_ARGS+=(--no-demo-edits); echo "== demo edits: OFF (DEMO_EDITS=0)"; fi

# POSE backend is swappable for the A/B bake-off (both satisfy the HMRBackend port,
# both return SMPL-X, so they drop into the SAME gvhmr estimator + downstream FK):
#   A (default)  pitch3d.adapters.models.smplestx_backend:make   — SMPLest-X
#   B            pitch3d.adapters.models.sam3dbody_backend:make   — SAM 3D Body (MHR→SMPL-X)
# The SAM3DBody backend reads its own env (PITCH3D_SAM3D_REPO / _MHR_REPO / _CKPT /
# _MHR_ASSET / _SMPLX_MODELS / _DEVICE) — see the backend factory. Its checkpoint is
# HF-GATED (facebook/sam-3d-body-dinov3): accept the licence + `hf download` it first.
POSE_BACKEND="${POSE_BACKEND:-pitch3d.adapters.models.smplestx_backend:make}"

cd "$REPO"
echo "== pod real E2E :: frames=${FRAMES} out=${OUT} format=${FORMAT} clip=${CLIP} pose=${POSE_BACKEND} =="
t0=$(date +%s)
PYTHONPATH=src "$PY" -m pitch3d \
  --clip "$CLIP" --frames "$FRAMES" \
  --detector rfdetr --tracker bytetrack --device cuda \
  "${CALIB_ARGS[@]}" "${COH_ARGS[@]}" \
  --pose gvhmr --pose-backend "$POSE_BACKEND" \
  --ball tracknet --ball-backend pitch3d.adapters.models.wasb_backend:make \
  --render overlay --export gltf --format "$FORMAT" --out-dir "$OUT"
if [ "${RIGID_CAMERA:-1}" = "1" ] && [ -n "${CAMERA_NPZ:-}" ] && [ -f "$CAMERA_NPZ" ]; then
  echo "== rigid camera: applying $CAMERA_NPZ to ${OUT}/export/scene.json (#119/#129)"
  PYTHONPATH=src "$PY" scripts/apply_rigid_camera.py "$CAMERA_NPZ" \
    --scene "$REPO/$OUT/export/scene.json" --out "$REPO/$OUT/export/scene.json"
elif [ "${RIGID_CAMERA:-1}" = "1" ]; then
  echo "== rigid camera: SKIPPED — set CAMERA_NPZ=<calib/*.npz> to apply it. Without it the scene"
  echo "   keeps a synthetic camera and nothing comparing it to the source pixels is meaningful."
fi
echo "== done in $(( $(date +%s) - t0 ))s -> ${OUT} =="
ls -la "${OUT}/export/scene.${FORMAT}" 2>/dev/null | head
