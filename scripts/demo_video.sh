#!/usr/bin/env bash
# scripts/demo_video.sh — one command: a broadcast clip → multi-angle 3D animation mp4s, on the pod.
#
# Unlike scripts/demo.sh (which renders a STILL of the bodies locally), this renders the whole
# ANIMATION — bodies + ball, several camera angles — ON THE GPU POD with Blender Cycles, and pulls
# the finished mp4s back to the machine. The local box needs no Blender/SMPL-X at all.
#
# Flow: bring up the pod → stage the chosen clip → ensure Blender+ffmpeg on the pod →
#       scripts/pod_make_video.sh (reconstruct → SMPL-X anim → render → mp4/angle) → pull *.mp4 → stop pod.
#
# Flags / env:
#   --clip PATH     broadcast clip to use (default samples/video/Colombia-1-0-Congo-DR1080p.mp4)
#   --frames N      frames to reconstruct & animate (default 60)
#   --cameras LIST  comma list of broadcast,sideline,top,goal (default all four)
#   --res WxH       render resolution (default 1280x720)
#   --samples N     Cycles samples/px (default 32)
#   --device gpu|cpu  pod render device (default gpu; falls back to CPU automatically)
#   --no-real-calib use the PROXY fake calibrator (default is REAL PnLCalib — #203). The proxy maps
#                   the whole frame into a 30 m top-down box (no perspective) → players collapse onto
#                   a thin band; only for a quick smoke test where geometry doesn't matter.
#   --real-calib    (default) calibrate with REAL PnLCalib on the pod: points the calib seam at the
#                   staged /workspace/repos/PnLCalib + weights unless .env set PNLCALIB_REPO.
#   --reuse-scene   reuse an existing pod-side scene.json (skip the ~3min reconstruction; cheap re-render)
#   --keep-pod      do NOT stop the pod at the end (debugging)
#   OUT_LOCAL       local output dir (default out/anim)
#   STITCH/COHERENCE  forward-continuity + temporal-coherence into reconstruction (default 1; =0 to disable)
#   PITCH3D_FADE_FRAMES  entry/exit mesh-opacity ramp length in frames (default 4; 0 = hard pop)
#
# Machine paths/keys come from the repo-root .env (see .env.example). The pod is ALWAYS stopped on
# exit (even on error) unless --keep-pod — GPU time is billed.
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
set -a; [ -f .env ] && . ./.env; set +a
# Shared knob defaults (FRAMES/STITCH/COHERENCE/ANIM_*) — single source of truth with pod_make_video.sh.
. scripts/video_defaults.sh

if [ -t 1 ]; then C0=$'\033[0m'; CH=$'\033[1;36m'; CG=$'\033[1;32m'; CY=$'\033[1;33m'; CR=$'\033[1;31m'; CD=$'\033[2m'
else C0=""; CH=""; CG=""; CY=""; CR=""; CD=""; fi
STEP=0
say(){ STEP=$((STEP+1)); printf '\n%s━━ [%d] %s%s\n' "$CH" "$STEP" "$*" "$C0"; }
info(){ printf '   %s%s%s\n' "$CD" "$*" "$C0"; }
ok(){   printf '   %s✓ %s%s\n' "$CG" "$*" "$C0"; }
warn(){ printf '   %s! %s%s\n' "$CY" "$*" "$C0"; }
die(){  printf '\n%sVIDEO DEMO FAILED: %s%s\n' "$CR" "$*" "$C0" >&2; exit 1; }

CLIP_LOCAL="${PITCH3D_CLIP_LOCAL:-samples/video/Colombia-1-0-Congo-DR1080p.mp4}"
FRAMES="${FRAMES:-$VIDEO_FRAMES_DEFAULT}"; CAMERAS="${ANIM_CAMERAS:-$VIDEO_CAMERAS_DEFAULT}"
RES="${ANIM_RES:-${VIDEO_RES_X_DEFAULT}x${VIDEO_RES_Y_DEFAULT}}"
SAMPLES="${ANIM_SAMPLES:-$VIDEO_SAMPLES_DEFAULT}"; DEVICE="${ANIM_DEVICE:-$VIDEO_DEVICE_DEFAULT}"
OUT_LOCAL="${OUT_LOCAL:-out/anim}"; KEEP_POD=0; REUSE_SCENE="${REUSE_SCENE:-0}"; REAL_CALIB="${REAL_CALIB:-1}"
# Direction A polish on by default: re-linked tracklets (--stitch) + gap-fill (--coherence) so animated
# bodies don't pop in/out, and a 4-frame mesh-opacity ramp at genuine entries/exits. Override with =0.
STITCH="${STITCH:-$VIDEO_STITCH_DEFAULT}"; COHERENCE="${COHERENCE:-$VIDEO_COHERENCE_DEFAULT}"
FADE_FRAMES="${PITCH3D_FADE_FRAMES:-4}"
while [ $# -gt 0 ]; do case "$1" in
  --clip)     CLIP_LOCAL="$2"; shift;;
  --frames)   FRAMES="$2"; shift;;
  --cameras)  CAMERAS="$2"; shift;;
  --res)      RES="$2"; shift;;
  --samples)  SAMPLES="$2"; shift;;
  --device)   DEVICE="$2"; shift;;
  --real-calib) REAL_CALIB=1;;
  --no-real-calib) REAL_CALIB=0;;
  --reuse-scene) REUSE_SCENE=1;;
  --keep-pod) KEEP_POD=1;;
  -h|--help)  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
  *)          die "unknown arg: $1 (try --clip PATH | --frames N | --cameras LIST | --res WxH | --samples N | --device gpu|cpu | --no-real-calib | --keep-pod)";;
esac; shift; done
RES_X="${RES%x*}"; RES_Y="${RES#*x}"
REPO_POD="${PITCH3D_REPO:-/workspace/fifa}"
OUT_POD="out/anim"
CLIP_POD="/workspace/$(basename "$CLIP_LOCAL")"

# Real PnLCalib (opt-in via --real-calib): point the calib seam at the pod's staged checkout +
# weights unless .env already set them. pod_real_e2e.sh swaps the proxy planar calibrator for the
# wired PnLCalib backend only when PNLCALIB_REPO is a live dir; empty (the default) keeps the proxy.
if [ "$REAL_CALIB" = 1 ]; then
  PNLCALIB_REPO="${PNLCALIB_REPO:-/workspace/repos/PnLCalib}"
  PNLCALIB_WEIGHTS_KP="${PNLCALIB_WEIGHTS_KP:-/workspace/weights/pnlcalib/SV_kp}"
  PNLCALIB_WEIGHTS_LINES="${PNLCALIB_WEIGHTS_LINES:-/workspace/weights/pnlcalib/SV_lines}"
fi

POD_UP=0
cleanup(){
  if [ "$KEEP_POD" = 0 ] && [ "$POD_UP" = 1 ]; then
    printf '\n%s━━ cleanup: stopping GPU pod%s\n' "$CH" "$C0"
    ./scripts/pod.sh down || warn "pod down failed — run scripts/pod.sh status and stop it manually (\$)"
  elif [ "$KEEP_POD" = 1 ] && [ "$POD_UP" = 1 ]; then
    warn "pod left RUNNING (--keep-pod) — stop it when done: scripts/pod.sh down"
  fi
}
trap cleanup EXIT

printf '%s┌──────────────────────────────────────────────────────────────┐%s\n' "$CH" "$C0"
printf '%s│  pitch3d — broadcast clip → multi-angle 3D animation (POD)    │%s\n' "$CH" "$C0"
printf '%s└──────────────────────────────────────────────────────────────┘%s\n' "$CH" "$C0"
info "clip=$CLIP_LOCAL frames=$FRAMES cams=$CAMERAS ${RES_X}x${RES_Y} samples=$SAMPLES device=$DEVICE out=$OUT_LOCAL"
info "polish: stitch=$STITCH coherence=$COHERENCE fade_frames=$FADE_FRAMES"
if [ -n "${PNLCALIB_REPO:-}" ]; then info "calibration: REAL PnLCalib ($PNLCALIB_REPO)"; else info "calibration: proxy planar"; fi

say "Preflight"
[ -f "$CLIP_LOCAL" ] && ok "clip present: $CLIP_LOCAL ($(du -h "$CLIP_LOCAL" | cut -f1))" \
  || die "clip not found: $CLIP_LOCAL (pass --clip PATH)"

say "Bring up the GPU pod"
info "reusing a live pod or resuming an EXITED one (auto host-failover) ..."
./scripts/pod.sh up || die "could not place a GPU on any pod (see scripts/pod.sh output)"
POD_UP=1
ok "pod is RUNNING"

say "Sync source → pod ($REPO_POD)"
tar czf - src scripts | ./scripts/pod.sh ssh "mkdir -p $REPO_POD && cd $REPO_POD && tar xzf - --no-same-owner" \
  || die "source sync failed"
ok "src/ + scripts/ synced"

say "Stage the clip on the pod ($CLIP_POD)"
LSIZE="$(stat -c%s "$CLIP_LOCAL")"
RSIZE="$(./scripts/pod.sh ssh "stat -c%s '$CLIP_POD' 2>/dev/null || echo 0" 2>/dev/null | tr -dc 0-9)"
if [ "${RSIZE:-0}" = "$LSIZE" ]; then
  ok "already staged (size match, $LSIZE bytes) — skipping upload"
else
  info "uploading $((LSIZE/1024/1024)) MB ..."
  ./scripts/pod.sh ssh "cat > '$CLIP_POD'" < "$CLIP_LOCAL" || die "clip upload failed"
  ok "clip uploaded → $CLIP_POD"
fi

say "Render the multi-angle animation on the pod"
info "reconstruct (RF-DETR · ByteTrack · SMPLest-X-H · WASB) → SMPL-X anim → Blender Cycles → mp4/angle"
info "Blender installs on first use (pip bpy, cached on the volume); Cycles render is the slow part ..."
./scripts/pod.sh ssh "
  cd $REPO_POD
  export PITCH3D_REPO='$REPO_POD' PITCH3D_PY='${PITCH3D_PY:-/workspace/.venv/bin/python}' PITCH3D_CLIP='$CLIP_POD'
  export PITCH3D_SMPLESTX_REPO='${PITCH3D_SMPLESTX_REPO:-/workspace/repos/SMPLest-X}'
  export PITCH3D_SMPLX_MODEL_PATH='${PITCH3D_SMPLX_MODEL_PATH:-/workspace/repos/SMPLest-X/human_models/human_model_files}'
  # anim_export.py's smplx-package models dir is the POD model path — derive it from the pod var,
  # NOT local PITCH3D_SMPLX_MODELS (which points at this machine's SMPL-X dir and would leak).
  export PITCH3D_SMPLX_MODELS='${PITCH3D_SMPLX_MODEL_PATH:-/workspace/repos/SMPLest-X/human_models/human_model_files}'
  export PITCH3D_WASB_REPO='${PITCH3D_WASB_REPO:-/workspace/repos/WASB-SBDT}' PITCH3D_WASB_CKPT='${PITCH3D_WASB_CKPT:-/workspace/weights/wasb/wasb_soccer_best.pth.tar}' PITCH3D_WASB_DATASET='${PITCH3D_WASB_DATASET:-soccer}'
  export PNLCALIB_REPO='${PNLCALIB_REPO:-}' PNLCALIB_WEIGHTS_KP='${PNLCALIB_WEIGHTS_KP:-}' PNLCALIB_WEIGHTS_LINES='${PNLCALIB_WEIGHTS_LINES:-}'
  export PITCH3D_BLENDER_TARBALL_URL='${PITCH3D_BLENDER_TARBALL_URL:-}'
  export PITCH3D_FADE_FRAMES='$FADE_FRAMES'
  # Measured appearance (stadium crowd / body texture / floodlight colour) samples the SAME staged
  # clip we reconstruct from. Always the POD path — a '\${VAR:-}' default here would expand to the
  # LOCAL machine's value and leak a path that does not exist on the pod (the SMPLX_MODELS gotcha).
  export PITCH3D_STADIUM_VIDEO='$CLIP_POD'
  FRAMES='$FRAMES' OUT='$OUT_POD' REUSE_SCENE='$REUSE_SCENE' \
  STITCH='$STITCH' COHERENCE='$COHERENCE' DEMO_EDITS='${DEMO_EDITS:-0}' \
  ANIM_DEVICE='$DEVICE' ANIM_RES_X='$RES_X' ANIM_RES_Y='$RES_Y' ANIM_SAMPLES='$SAMPLES' \
  ANIM_CAMERAS='$CAMERAS' bash scripts/pod_make_video.sh
" || die "pod video generation failed (see output above)"
ok "animation rendered on the pod"

say "Pull the mp4s → local ($OUT_LOCAL/video)"
mkdir -p "$OUT_LOCAL"
./scripts/pod.sh ssh "cd $REPO_POD/$OUT_POD && tar czf - video" | tar xzf - -C "$OUT_LOCAL" \
  || die "mp4 pull failed"
ok "pulled $(ls "$OUT_LOCAL"/video/*.mp4 2>/dev/null | wc -l | tr -d ' ') mp4(s)"

if [ "$KEEP_POD" = 0 ]; then
  say "Stop the GPU pod (done with it)"
  ./scripts/pod.sh down && POD_UP=0 && ok "pod stopped — billing back to volume-only"
fi

printf '\n%s┌──────────────────────────────────────────────────────────────┐%s\n' "$CG" "$C0"
printf '%s│  VIDEO DEMO COMPLETE%s\n' "$CG" "$C0"
printf '%s└──────────────────────────────────────────────────────────────┘%s\n' "$CG" "$C0"
cat <<EOF
  Multi-angle 3D animation (bodies + ball) under $OUT_LOCAL/video/:
$(for f in "$OUT_LOCAL"/video/*.mp4; do [ -f "$f" ] && echo "    • $f"; done)

  Rendered on the pod from $CLIP_LOCAL ($FRAMES frames, cams: $CAMERAS, ${RES_X}x${RES_Y}, ${SAMPLES}spp).
  What ran for real: detect (RF-DETR) · track (ByteTrack) · pose (SMPLest-X-H, 0.69B) · ball (WASB).
  Bodies = recovered SMPL-X meshes; ball = resolved 3D track. Re-run on another clip: --clip PATH.
EOF
