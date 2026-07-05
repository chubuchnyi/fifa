#!/usr/bin/env bash
# scripts/pod_ensure_blender.sh — make a usable Blender available ON THE POD for headless render.
#
# Two ways to get Blender, tried in order:
#   1) the `bpy` PyPI module in our venv — works ONLY if a wheel matches the venv's CPython
#      (bpy ships wheels for specific versions, e.g. 3.11; our pod venv is 3.12 → no wheel, so this
#      path is usually skipped here, but kept for pods on a compatible Python).
#   2) a standalone Blender binary downloaded once to the persistent /workspace volume (default the
#      4.5 LTS build from the official release server; override with PITCH3D_BLENDER_TARBALL_URL).
# Also ensures ffmpeg + the X/GL shared libs Blender links even under --background.
#
# Emits EXACTLY ONE line on stdout for the caller to consume (all chatter goes to stderr):
#   BLENDER_MODE=module                → run: "$PY" scripts/blender_animate.py ...
#   BLENDER_MODE=binary:/path/blender  → run: /path/blender --background --python ... --
set -euo pipefail
PY="${PITCH3D_PY:-/workspace/.venv/bin/python}"
# 4.5 LTS: 4.2 predates Blackwell (sm_120) — on an RTX PRO 4500 pod BOTH OptiX and CUDA kernel
# loads hung indefinitely at "Loading render kernels" (2026-07-05); 4.5 ships sm_120 kernels.
TARBALL_URL="${PITCH3D_BLENDER_TARBALL_URL:-https://download.blender.org/release/Blender4.5/blender-4.5.11-linux-x64.tar.xz}"
BLENDER_DIR=/workspace/blender

log(){ echo "[ensure] $*" >&2; }
apt_quiet(){ DEBIAN_FRONTEND=noninteractive apt-get -qq "$@" >/dev/null 2>&1 || true; }

# ffmpeg (frame→mp4) + the runtime libs Blender needs to even load headless
need=""
command -v ffmpeg >/dev/null 2>&1 || need="$need ffmpeg"
for lib in libxrender1 libxxf86vm1 libxfixes3 libxi6 libxkbcommon0 libsm6 libgl1; do
  dpkg -s "$lib" >/dev/null 2>&1 || need="$need $lib"
done
if [ -n "$need" ]; then
  log "apt-get install:$need"
  apt_quiet update
  apt_quiet install -y $need
fi

# 1) bpy module (only if a wheel matches this Python)
if "$PY" -c 'import bpy' >/dev/null 2>&1; then
  log "bpy already importable"; echo "BLENDER_MODE=module"; exit 0
fi
log "bpy not importable; trying pip install bpy (no-op if no wheel for this Python) ..."
if "$PY" -m pip install --quiet bpy >/dev/null 2>&1 && "$PY" -c 'import bpy' >/dev/null 2>&1; then
  log "installed bpy"; echo "BLENDER_MODE=module"; exit 0
fi
log "no bpy wheel for this Python — using a standalone Blender binary"

# 2) standalone binary on the persistent volume (downloaded once)
BIN="$(find "$BLENDER_DIR" -maxdepth 2 -name blender -type f 2>/dev/null | head -1 || true)"
if [ -z "$BIN" ]; then
  log "downloading Blender: $TARBALL_URL"
  mkdir -p "$BLENDER_DIR" && cd "$BLENDER_DIR"
  curl -fsSL "$TARBALL_URL" -o blender.tar.xz || { echo "BLENDER_ENSURE_FAILED: download failed ($TARBALL_URL)" >&2; exit 1; }
  tar xf blender.tar.xz --strip-components=1
  rm -f blender.tar.xz
  BIN="$(find "$BLENDER_DIR" -maxdepth 2 -name blender -type f 2>/dev/null | head -1 || true)"
fi
if [ -n "$BIN" ] && "$BIN" --background --version >/dev/null 2>&1; then
  log "blender binary OK: $BIN"; echo "BLENDER_MODE=binary:$BIN"; exit 0
fi
echo "BLENDER_ENSURE_FAILED: bpy unavailable and Blender binary not runnable (libs? url: $TARBALL_URL)" >&2
exit 1
