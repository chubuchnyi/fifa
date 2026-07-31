#!/usr/bin/env bash
# scripts/stage_smplestx_weights.sh — link the SMPLest-X checkpoint into the layout its loader reads.
#
# runpod-agent-setup.md §4 puts the ~8.2 GB Huge (ViT-H) checkpoint in $VOL/weights/smplest-x/, but
# SMPLest-X's own `Config.load_config` reads $PITCH3D_SMPLESTX_REPO/pretrained_models/smplest_x_h/
# (see smplestx_backend._load). Two symlinks bridge the two, and this script (re)creates them.
#
# They must be RELATIVE. The staged links were absolute — /workspace/weights/smplest-x/... — which
# is fine on a pod that mounts the network volume at /workspace and dangles on one that mounts the
# SAME volume at /runpod, because the mount point is a per-pod setting. That failure is quiet in a
# nasty way: `stat` still reports 43 bytes / mode 777 (the symlink itself), `ls` shows the entry,
# and only the actual open fails — the 2026-07-31 run died 4 stages in with a FileNotFoundError
# naming a path `ls` had just printed. A relative link survives any mount point.
#
# Idempotent: safe to re-run, and re-running is the repair when a pod comes back on a new mount.
#
#   bash scripts/stage_smplestx_weights.sh
#
# Env (defaults match pod_real_e2e.sh's resolver and runpod-agent-setup.md's $VOL layout):
#   PITCH3D_VOL=<auto>            network-volume mount; searched by content (/workspace, /runpod)
#   PITCH3D_SMPLESTX_REPO=$VOL/repos/SMPLest-X
#   SMPLESTX_WEIGHTS=$VOL/weights/smplest-x
set -euo pipefail

VOL="${PITCH3D_VOL:-}"
if [ -z "$VOL" ]; then
  for c in /workspace /runpod; do [ -d "$c/fifa" ] && { VOL="$c"; break; }; done
  VOL="${VOL:-/workspace}"
fi

REPO="${PITCH3D_SMPLESTX_REPO:-$VOL/repos/SMPLest-X}"
SRC="${SMPLESTX_WEIGHTS:-$VOL/weights/smplest-x}"
DST="$REPO/pretrained_models/smplest_x_h"

echo "== staging SMPLest-X weights :: vol=$VOL src=$SRC dst=$DST"
[ -d "$REPO" ] || { echo "!! SMPLest-X checkout missing: $REPO (see docs/runpod-agent-setup.md §3)"; exit 1; }
for f in config_base.py smplest_x_h.pth.tar; do
  [ -f "$SRC/$f" ] || { echo "!! weight missing: $SRC/$f (see docs/runpod-agent-setup.md §4)"; exit 1; }
done

mkdir -p "$DST"
# Relative depth from $DST back to $VOL: smplest_x_h → pretrained_models → SMPLest-X → repos → $VOL.
# Computed rather than hard-coded so an overridden PITCH3D_SMPLESTX_REPO still links correctly.
REL="$(python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$SRC" "$DST")"
for f in config_base.py smplest_x_h.pth.tar; do
  ln -sfn "$REL/$f" "$DST/$f"
done

# Prove the links OPEN, not merely that they exist — the whole point of this script is that
# existence and readability came apart.
for f in config_base.py smplest_x_h.pth.tar; do
  head -c 1 "$DST/$f" > /dev/null || { echo "!! link unreadable: $DST/$f -> $(readlink "$DST/$f")"; exit 1; }
  echo "   ok  $f -> $(readlink "$DST/$f")"
done
echo "== SMPLest-X weights staged"
