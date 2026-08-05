#!/usr/bin/env bash
# scripts/demo.sh — one-command, narrated, end-to-end product demo.
#
# What it shows: a single broadcast-soccer clip becomes a real, editable 3D scene —
# real perception on a GPU (RF-DETR detect -> ByteTrack -> SMPLest-X-H pose -> WASB ball,
# optional PnLCalib field calibration), an attention list (honest uncertainty), an edit
# that propagates across frames, reprojection overlay + tactical radar, an SMPL-X export,
# and a photoreal Blender render of the recovered bodies on a virtual pitch.
#
# Two modes:
#   scripts/demo.sh            FULL  — brings up the GPU pod, runs the real pipeline, pulls
#                                      the artifacts, renders the bodies locally, stops the pod.
#   scripts/demo.sh --local    LOCAL — skips the pod; renders bodies from an EXISTING export
#                                      under $OUT_LOCAL (run the full demo once first).
#
# Flags / env:
#   --local            no pod; render from an existing export
#   --keep-pod         do NOT stop the pod at the end (debugging)
#   --frames N         frames to reconstruct (default 8)
#   FRAMES, MESH_FRAME, OUT_LOCAL   env overrides (mesh frame defaults to the scene mid-frame)
#
# Machine paths/keys come from the repo-root .env (see .env.example). The pod is ALWAYS
# stopped on exit (even on error) unless --keep-pod — GPU time is billed.
set -uo pipefail

cd "$(dirname "$0")/.."            # repo root, wherever invoked from
ROOT="$(pwd)"
set -a; [ -f .env ] && . ./.env; set +a

# ── narration helpers ────────────────────────────────────────────────────────
if [ -t 1 ]; then C0=$'\033[0m'; CH=$'\033[1;36m'; CG=$'\033[1;32m'; CY=$'\033[1;33m'; CR=$'\033[1;31m'; CD=$'\033[2m'
else C0=""; CH=""; CG=""; CY=""; CR=""; CD=""; fi
STEP=0
say(){ STEP=$((STEP+1)); printf '\n%s━━ [%d] %s%s\n' "$CH" "$STEP" "$*" "$C0"; }
info(){ printf '   %s%s%s\n' "$CD" "$*" "$C0"; }
ok(){   printf '   %s✓ %s%s\n' "$CG" "$*" "$C0"; }
warn(){ printf '   %s! %s%s\n' "$CY" "$*" "$C0"; }
die(){  printf '\n%sDEMO FAILED: %s%s\n' "$CR" "$*" "$C0" >&2; exit 1; }

# ── args ─────────────────────────────────────────────────────────────────────
MODE="full"; KEEP_POD=0
FRAMES="${FRAMES:-8}"; OUT_LOCAL="${OUT_LOCAL:-out/demo}"; MESH_FRAME="${MESH_FRAME:-}"
while [ $# -gt 0 ]; do case "$1" in
  --local)    MODE="local";;
  --keep-pod) KEEP_POD=1;;
  --frames)   FRAMES="$2"; shift;;
  -h|--help)  sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
  *)          die "unknown arg: $1 (try --local | --keep-pod | --frames N | --help)";;
esac; shift; done
[ -z "$MESH_FRAME" ] && MESH_FRAME=$(( FRAMES / 2 ))   # pose the scene mid-frame by default

# local tool resolution (from .env, with sane fallbacks)
PY_LOCAL="${PITCH3D_PY_LOCAL:-.venv/bin/python}"
BLENDER="${PITCH3D_BLENDER:-$(command -v blender 2>/dev/null || echo blender)}"
SMPLX_MODELS="${PITCH3D_SMPLX_MODELS:-models/smplx}"
EXPORT_DIR="$OUT_LOCAL/export/scene.smplx_npz"
MESH_DIR="$OUT_LOCAL/mesh"

# ── always stop the pod on exit (cost discipline) ────────────────────────────
POD_UP=0
cleanup(){
  if [ "$MODE" = full ] && [ "$KEEP_POD" = 0 ] && [ "$POD_UP" = 1 ]; then
    printf '\n%s━━ cleanup: stopping GPU pod%s\n' "$CH" "$C0"
    ./scripts/pod.sh down || warn "pod down failed — CHECK scripts/pod.sh status and stop it manually (\$)"
  elif [ "$MODE" = full ] && [ "$KEEP_POD" = 1 ] && [ "$POD_UP" = 1 ]; then
    warn "pod left RUNNING (--keep-pod) — stop it when done: scripts/pod.sh down"
  fi
}
trap cleanup EXIT

printf '%s┌──────────────────────────────────────────────────────────────┐%s\n' "$CH" "$C0"
printf '%s│  pitch3d — single broadcast clip → editable 3D pitch (DEMO)   │%s\n' "$CH" "$C0"
printf '%s└──────────────────────────────────────────────────────────────┘%s\n' "$CH" "$C0"
info "mode=$MODE  frames=$FRAMES  mesh_frame=$MESH_FRAME  out=$OUT_LOCAL"

# ── preflight: local render toolchain (needed in BOTH modes) ─────────────────
say "Preflight — local render toolchain"
"$PY_LOCAL" -c 'import numpy, torch, smplx, matplotlib' 2>/dev/null \
  && ok "python venv: $PY_LOCAL (numpy+torch+smplx+matplotlib)" \
  || die "venv '$PY_LOCAL' missing deps — run scripts/local_setup.sh (or set PITCH3D_PY_LOCAL in .env)"
[ -f "$SMPLX_MODELS/smplx/SMPLX_NEUTRAL.npz" ] \
  && ok "SMPL-X model: $SMPLX_MODELS/smplx/SMPLX_NEUTRAL.npz" \
  || die "SMPL-X model missing under '$SMPLX_MODELS' (MPI-gated; set PITCH3D_SMPLX_MODELS in .env — docs/blender-demo.md)"
USE_BLENDER=1
if [ -x "$BLENDER" ] || command -v "$BLENDER" >/dev/null 2>&1; then
  ok "Blender: $BLENDER (photoreal Cycles render)"
else
  USE_BLENDER=0
  warn "Blender not found at '$BLENDER' — falling back to the matplotlib mesh render (set PITCH3D_BLENDER in .env for the photoreal version)"
fi

# ════════════════════════════════════════════════════════════════════════════
# FULL mode: real GPU reconstruction on the pod
# ════════════════════════════════════════════════════════════════════════════
if [ "$MODE" = full ]; then
  REPO_POD="${PITCH3D_REPO:-/workspace/fifa}"
  CLIP_POD="${PITCH3D_CLIP:-/workspace/clip.mp4}"

  say "Bring up the GPU pod"
  info "reusing a live pod or resuming an EXITED one (auto host-failover) ..."
  ./scripts/pod.sh up || die "could not place a GPU on any pod (see scripts/pod.sh output)"
  POD_UP=1
  ok "pod is RUNNING"

  say "Sync source → pod ($REPO_POD)"
  tar czf - src scripts | ./scripts/pod.sh ssh "mkdir -p $REPO_POD && cd $REPO_POD && tar xzf - --no-same-owner" \
    || die "source sync failed"
  ok "src/ + scripts/ synced"

  say "Check staged GPU assets on the pod"
  STAGE="$(./scripts/pod.sh ssh "
    test -f '$CLIP_POD' && echo clip_ok
    test -d '${PITCH3D_SMPLESTX_REPO:-/workspace/repos/SMPLest-X}' && echo pose_ok
    test -f '${PITCH3D_WASB_CKPT:-/workspace/weights/wasb/wasb_soccer_best.pth.tar}' && echo ball_ok
    test -n '${PNLCALIB_REPO:-}' && test -d '${PNLCALIB_REPO:-/nonexistent}' && echo calib_ok || true
  " 2>/dev/null)"
  case "$STAGE" in *clip_ok*) ok "clip: $CLIP_POD";; *) die "clip missing on pod at $CLIP_POD (stage a broadcast clip there, or set PITCH3D_CLIP in .env)";; esac
  case "$STAGE" in *pose_ok*) ok "pose backend (SMPLest-X) staged";;  *) warn "SMPLest-X repo not found — the pose step will error";; esac
  case "$STAGE" in *ball_ok*) ok "ball backend (WASB) weight staged";; *) warn "WASB weight not found — the ball step will error";; esac
  case "$STAGE" in *calib_ok*) ok "calibration: REAL PnLCalib staged";; *) info "calibration: proxy (PnLCalib not staged — set PNLCALIB_REPO in .env once it is)";; esac

  say "Run the REAL reconstruction on CUDA (this is the product)"
  info "RF-DETR detect → ByteTrack → SMPLest-X-H pose (0.69B) → WASB ball → overlay + export"
  info "first run loads ~0.7B of weights; expect ~2-3 min ..."
  ./scripts/pod.sh ssh "
    cd $REPO_POD
    export PITCH3D_REPO='$REPO_POD' PITCH3D_PY='${PITCH3D_PY:-/workspace/.venv/bin/python}' PITCH3D_CLIP='$CLIP_POD'
    export PITCH3D_SMPLESTX_REPO='${PITCH3D_SMPLESTX_REPO:-/workspace/repos/SMPLest-X}'
    export PITCH3D_SMPLX_MODEL_PATH='${PITCH3D_SMPLX_MODEL_PATH:-/workspace/repos/SMPLest-X/human_models/human_model_files}'
    export PITCH3D_WASB_REPO='${PITCH3D_WASB_REPO:-/workspace/repos/WASB-SBDT}' PITCH3D_WASB_CKPT='${PITCH3D_WASB_CKPT:-/workspace/weights/wasb/wasb_soccer_best.pth.tar}' PITCH3D_WASB_DATASET='${PITCH3D_WASB_DATASET:-soccer}'
    export PNLCALIB_REPO='${PNLCALIB_REPO:-}' PNLCALIB_WEIGHTS_KP='${PNLCALIB_WEIGHTS_KP:-}' PNLCALIB_WEIGHTS_LINES='${PNLCALIB_WEIGHTS_LINES:-}'
    FRAMES='$FRAMES' OUT='$OUT_LOCAL' bash scripts/pod_real_e2e.sh
  " || die "real reconstruction failed on the pod (see output above)"
  ok "reconstruction complete on the pod"

  say "Pull artifacts → local ($OUT_LOCAL)"
  rm -rf "$OUT_LOCAL"; mkdir -p "$OUT_LOCAL"
  ./scripts/pod.sh ssh "cd $REPO_POD/$OUT_LOCAL && tar czf - observations render export" \
    | tar xzf - -C "$OUT_LOCAL" || die "artifact pull failed"
  ok "pulled overlay renders, radar, and the SMPL-X export"

  # pod no longer needed — stop it now (the trap is a belt-and-braces backup)
  if [ "$KEEP_POD" = 0 ]; then
    say "Stop the GPU pod (done with it)"
    ./scripts/pod.sh down && POD_UP=0 && ok "pod stopped — billing back to volume-only"
  fi
else
  say "LOCAL mode — using the existing export under $OUT_LOCAL"
  [ -d "$EXPORT_DIR" ] && ls "$EXPORT_DIR"/subject_*.npz >/dev/null 2>&1 \
    && ok "found export: $EXPORT_DIR" \
    || die "no export at $EXPORT_DIR — run the full demo once first: scripts/demo.sh"
fi

# ════════════════════════════════════════════════════════════════════════════
# Shared: turn the real SMPL-X export into rendered bodies
# ════════════════════════════════════════════════════════════════════════════
say "Forward the SMPL-X export → posed meshes (frame $MESH_FRAME)"
PITCH3D_SMPLX_MODELS="$SMPLX_MODELS" PITCH3D_NPZ_DIR="$EXPORT_DIR" \
PITCH3D_MESH_OUT="$MESH_DIR" PITCH3D_MESH_FRAME="$MESH_FRAME" \
  "$PY_LOCAL" scripts/smplx_export_meshes.py || die "SMPL-X mesh export failed"
ok "meshes written → $MESH_DIR"

if [ "$USE_BLENDER" = 1 ]; then
  say "Render the recovered bodies in Blender (Cycles, CPU)"
  info "wide pitch shot + a hero close-up of one body ..."
  "$BLENDER" --background --python scripts/blender_render_meshes.py -- \
    --in "$MESH_DIR" --out "$MESH_DIR/blender_scene.png" >/dev/null 2>&1 \
    && ok "render → $MESH_DIR/blender_scene.png (+ _hero.png)" \
    || die "Blender render failed (run with PITCH3D_BLENDER set, or use the matplotlib path)"
  HERO="$MESH_DIR/blender_scene.png (wide) + $MESH_DIR/blender_scene_hero.png (hero)"
else
  say "Render the recovered bodies (matplotlib, no Blender)"
  PITCH3D_SMPLX_MODELS="$SMPLX_MODELS" PITCH3D_NPZ_DIR="$EXPORT_DIR" \
  PITCH3D_MESH_OUT="$MESH_DIR" PITCH3D_MESH_FRAME="$MESH_FRAME" \
    "$PY_LOCAL" scripts/render_smplx_mesh.py || die "matplotlib mesh render failed"
  HERO="$MESH_DIR/scene_meshes.png + $MESH_DIR/mesh_subject1.png"
fi

# ── summary ──────────────────────────────────────────────────────────────────
# what calibration actually ran (full mode only knows; local can't, so stays blank)
CALIB_LABEL=""
if [ "$MODE" = full ]; then
  case "${STAGE:-}" in
    *calib_ok*) CALIB_LABEL=" · calibration (PnLCalib, real)";;
    *)          CALIB_LABEL=" · calibration: PROXY";;
  esac
fi
printf '\n%s┌──────────────────────────────────────────────────────────────┐%s\n' "$CG" "$C0"
printf '%s│  DEMO COMPLETE%s\n' "$CG" "$C0"
printf '%s└──────────────────────────────────────────────────────────────┘%s\n' "$CG" "$C0"
cat <<EOF
  Artifacts under $OUT_LOCAL/:
    • 3D bodies (the headline):  $HERO
    • reprojection overlay:      $OUT_LOCAL/render/scene-1_preview/frame_*.png
    • tactical radar (top-down): $OUT_LOCAL/observations/scene-1_radar_*.png
    • SMPL-X export (per player):$EXPORT_DIR/subject_*.npz

  What ran for real this pass: detect (RF-DETR) · track (ByteTrack) · pose (SMPLest-X-H, 0.69B)
  · ball (WASB)$CALIB_LABEL
  Measured accuracy (separate benchmarks): calibration ≈ 0.236 m (SoccerNet) · pose ≈ 0.51 m
  Local MPJPE (3DPW). Overlay = honest reprojection markers, not a broadcast composite.

  Re-render the bodies without the pod:   scripts/demo.sh --local
EOF
