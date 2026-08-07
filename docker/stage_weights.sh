#!/usr/bin/env bash
# docker/stage_weights.sh — stage the real backends' weights on a Docker box.
#
# Run it INSIDE the pitch3d image, with the volume mounted at /workspace:
#
#   docker run -d --name stage -v /vol:/workspace -v /models:/models \
#     -w /workspace/fifa pitch3d:cu124 bash docker/stage_weights.sh
#
# Mount at /workspace, not somewhere prettier: the repo's scripts and the three model
# backends carry baked-in /workspace defaults, and stage_wasb_weight.sh ignores
# PITCH3D_VOL outright — under --rm its 6 MB checkpoint vanished with the container.
#
# Why curl and not huggingface_hub: on a link that resets every ~250 MB,
# snapshot_download restarts each transfer from zero and never finishes the 8.2 GB
# checkpoint (measured 2026-08-07: one hour looping between 0 and 335 MB).
# `curl -C -` resumes, and --speed-limit/--speed-time cuts a stalled socket in 45 s
# instead of hanging on it. The same fetch then completed across 34 resets.
set -uo pipefail
VOL="${PITCH3D_VOL:-/workspace}"
mkdir -p "$VOL"/repos "$VOL"/weights/smplest-x "$VOL"/weights/pnlcalib "$VOL"/weights/wasb

fetch() {  # url dest — resumable, verified against Content-Length
  local url="$1" dest="$2" want got tries=0
  want=$(curl -sSIL "$url" | awk 'BEGIN{IGNORECASE=1}/^content-length:/{v=$2}END{gsub(/\r/,"",v);print v}')
  [ -z "$want" ] && want=0
  echo "  target $(basename "$dest") : ${want} bytes"
  while [ "$tries" -lt 400 ]; do
    got=$(stat -c %s "$dest" 2>/dev/null || echo 0)
    if [ "$want" -gt 0 ] && [ "$got" -ge "$want" ]; then
      echo "  COMPLETE $(basename "$dest") ${got} bytes"; return 0; fi
    tries=$((tries+1))
    curl -sS -L -C - --retry 3 --retry-delay 3 \
         --speed-limit 20000 --speed-time 45 -o "$dest" "$url" 2>/dev/null || true
  done
  echo "  GAVE UP on $(basename "$dest") at ${got}/${want}"; return 1
}

echo "=== research repos (ADR-0006 backends are dotted-path imports, not pip packages) ==="
cd "$VOL/repos"
for r in MotrixLab/SMPLest-X mguti97/PnLCalib nttcom/WASB-SBDT; do
  n="${r##*/}"
  [ -d "$n/.git" ] && { echo "  have $n"; continue; }
  git clone --quiet --depth 1 "https://github.com/$r.git" "$n" && echo "  cloned $n"
done

echo ""
echo "=== SMPLest-X Huge (8.2 GB; HF ships only the full file) ==="
fetch "https://huggingface.co/waanqii/SMPLest-X/resolve/main/smplest_x_h.pth.tar" \
      "$VOL/weights/smplest-x/smplest_x_h.pth.tar"
[ -s "$VOL/weights/smplest-x/config_base.py" ] || curl -sSL \
  -o "$VOL/weights/smplest-x/config_base.py" \
  "https://huggingface.co/waanqii/SMPLest-X/resolve/main/config_base.py"

echo ""
echo "=== PnLCalib SV_kp + SV_lines (GitHub release v1.0.0) ==="
for a in SV_kp SV_lines; do
  fetch "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/$a" "$VOL/weights/pnlcalib/$a"
done

echo ""
echo "=== WASB + SMPLest-X link staging (the repo's own idempotent scripts) ==="
bash "$VOL/fifa/scripts/stage_wasb_weight.sh"      2>&1 | tail -4
bash "$VOL/fifa/scripts/stage_smplestx_weights.sh" 2>&1 | tail -4

# SMPLest-X's human_models.py:20-32 builds neutral+male+female and reads three aux
# files from smplx/ (not aux/). A neutral-only stage fails at POSE, four stages in.
echo ""
echo "=== SMPL-X (pushed by hand — MPI login, no token download) ==="
SMPLX_SRC="${PITCH3D_SMPLX_SRC:-/models/smplx}"
DST="$VOL/repos/SMPLest-X/human_models/human_model_files"
mkdir -p "$(dirname "$DST")"
[ -e "$DST" ] || ln -s "$SMPLX_SRC" "$DST"
for f in SMPLX_NEUTRAL.npz SMPLX_MALE.npz SMPLX_FEMALE.npz \
         SMPLX_to_J14.pkl MANO_SMPLX_vertex_ids.pkl SMPL-X__FLAME_vertex_ids.npy; do
  [ -f "$SMPLX_SRC/smplx/$f" ] && echo "  ok   $f" || echo "  MISSING $f — POSE will fail"
done

echo ""
echo "=== slim the checkpoint: inference wants the network, not the optimizer state ==="
python - <<'PY'
import os, torch
src, dst = "/workspace/weights/smplest-x/smplest_x_h.pth.tar", \
           "/workspace/weights/smplest-x/smplest_x_h_slim.pth.tar"
if os.path.exists(dst):
    print("  slim already present")
elif os.path.exists(src):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    keep = {k: ck[k] for k in ("network",) if k in ck}
    if keep:
        torch.save(keep, dst)
        print(f"  slim {os.path.getsize(src)/1e9:.2f} GB -> {os.path.getsize(dst)/1e9:.2f} GB")
    else:
        print("  no 'network' key; keys =", list(ck)[:10])
PY

echo ""
du -sh "$VOL"/weights/* 2>/dev/null
echo "STAGING_COMPLETE"
