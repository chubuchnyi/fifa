#!/usr/bin/env bash
# scripts/pod_real_e2e.sh — one real-model E2E pass on a GPU pod.
#
# Runs the SAME golden path as docs/cloud-dev.md §5 (real video -> RF-DETR detect ->
# ByteTrack -> assemble -> overlay render + smplx_npz/glTF export) but with the wired
# **real SMPLest-X** pose + **real WASB** ball backends injected by dotted path (ADR-0006)
# — i.e. the genuine single-cam -> world-SMPL-X path, end to end, on CUDA.
#
# Env (all have sensible pod defaults — override to point elsewhere):
#   PITCH3D_REPO=/workspace/fifa            repo root (PYTHONPATH=src is set from here)
#   PITCH3D_PY=/workspace/.venv/bin/python  interpreter with cuda torch + rfdetr + smplx
#   PITCH3D_CLIP=/workspace/clip.mp4        input broadcast clip
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

REPO="${PITCH3D_REPO:-/workspace/fifa}"
PY="${PITCH3D_PY:-/workspace/.venv/bin/python}"
CLIP="${PITCH3D_CLIP:-/workspace/clip.mp4}"
FRAMES="${FRAMES:-8}"
OUT="${OUT:-out/run}"
FORMAT="${FORMAT:-smplx_npz}"

# REAL field calibration is the DEFAULT now (#203). The proxy "fake" calibrator maps the WHOLE
# frame into a 30 m top-down box with NO perspective, so an oblique broadcast collapses every
# player onto a thin world band — the v0 depth/scale bug. Default PNLCALIB_REPO to the pod's staged
# checkout; the `-d` guard falls back to the proxy ONLY when the repo is genuinely absent (e.g. a
# local CPU box). Force the proxy explicitly with PNLCALIB_REPO= (empty). The backend reads its own
# PNLCALIB_WEIGHTS_* paths (pod defaults are baked into pnlcalib_backend.make()).
PNLCALIB_REPO="${PNLCALIB_REPO-/workspace/repos/PnLCalib}"
export PNLCALIB_REPO
CALIB_ARGS=()
if [ -n "${PNLCALIB_REPO}" ] && [ -d "${PNLCALIB_REPO}" ]; then
  CALIB_ARGS=(--calibrator keypoints
              --calibrator-backend pitch3d.adapters.models.pnlcalib_backend:make)
  echo "== calibration: REAL PnLCalib (${PNLCALIB_REPO})"
else
  echo "== calibration: PROXY fake — #203 depth COLLAPSE expected (PnLCalib repo absent: '${PNLCALIB_REPO:-unset}')"
fi

# Continuity stitching is ON by default (part of correct reconstruction): re-link fragmented
# tracklets before POSE so an occluded player keeps ONE identity instead of re-entering as a new
# track id and spawning a phantom body (#202). Set STITCH=0 to disable. Temporal coherence
# (gap-fill via slerp/lerp + auto temporal-smoothing) stays opt-in via COHERENCE=1.
COH_ARGS=()
if [ "${STITCH:-1}" = "1" ]; then echo "== continuity: stitch ON (default)";
else COH_ARGS+=(--no-stitch); echo "== continuity: stitch OFF (STITCH=${STITCH})"; fi
if [ "${COHERENCE:-0}" = "1" ]; then COH_ARGS+=(--coherence); echo "== coherence: --coherence ON"; fi
# PHYSICS=1 → M3-9 kinematic gate: clamp impossible root speed/accel, mark teleports (#207).
if [ "${PHYSICS:-0}" = "1" ]; then COH_ARGS+=(--physics); echo "== physics: --physics ON (M3-9 kinematic gate)"; fi
# PHYSICS_PROFILE selects the config/physics.yaml named profile (default / conservative /
# strict / no_smoothing / future_full / safe_new / humanize_teleports). safe_new turns on
# foot_floor + joint + orientation (T1a/b/c) without collision (which introduces accel
# spikes without a compose-order fix). humanize_teleports interpolates ID-swap regions.
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

cd "$REPO"
echo "== pod real E2E :: frames=${FRAMES} out=${OUT} format=${FORMAT} clip=${CLIP} =="
t0=$(date +%s)
PYTHONPATH=src "$PY" -m pitch3d \
  --clip "$CLIP" --frames "$FRAMES" \
  --detector rfdetr --tracker bytetrack --device cuda \
  "${CALIB_ARGS[@]}" "${COH_ARGS[@]}" \
  --pose gvhmr --pose-backend pitch3d.adapters.models.smplestx_backend:make \
  --ball tracknet --ball-backend pitch3d.adapters.models.wasb_backend:make \
  --render overlay --export gltf --format "$FORMAT" --out-dir "$OUT"
echo "== done in $(( $(date +%s) - t0 ))s -> ${OUT} =="
ls -la "${OUT}/export/scene.${FORMAT}" 2>/dev/null | head
