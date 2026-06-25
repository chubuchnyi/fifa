#!/usr/bin/env bash
# On-pod RTSP MVP: MediaMTX (TCP-only) + ffmpeg test pattern. Run as root on a RunPod pod.
#
# Proves the user-facing question "can we stream off a RunPod pod?": YES over RTSP-over-TCP,
# which traverses RunPod's Direct-TCP port remap transparently (no UDP media ports needed).
# WebRTC does NOT work the same way — its media is UDP/ICE, which RunPod does not cleanly expose.
#
# Validated 2026-06-25 (pod rtsp-mvp / RTX 4090 / cuda11.8 image): on-box AND from a home laptop
# `ffprobe -rtsp_transport tcp rtsp://<publicIp>:<extPort>/test` -> h264 1280x720 25/1, "OK".
#
# How to use (see docs/rtsp-mvp.md for the full rent->expose->VLC recipe):
#   1. Rent a pod with `supportPublicIp:true` and 8554/tcp exposed as a Direct TCP Port.
#      Image must be CUDA <= host driver (cu128 crash-loops on Secure RTX 4090 / CUDA-12.4 hosts).
#   2. ssh -i ~/.ssh/id_ed25519_runpod -p <sshExtPort> root@<publicIp> 'bash -s' < scripts/rtsp_mvp_setup.sh
#   3. Read the *live* 8554 mapping from GraphQL runtime.ports (NOT the stale REST portMappings —
#      the public port re-rolls on every stop/start), then open in VLC, forcing TCP:
#        vlc --rtsp-tcp rtsp://<publicIp>:<extPort>/test
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "== [1/5] apt: ffmpeg + tools =="
apt-get update -qq
apt-get install -y -qq ffmpeg wget jq ca-certificates >/dev/null
ffmpeg -version | head -1

echo "== [2/5] download MediaMTX (latest linux amd64) =="
cd /root
MTX_URL="$(wget -qO- https://api.github.com/repos/bluenviron/mediamtx/releases/latest \
  | jq -r '.assets[].browser_download_url' | grep -E 'linux_amd64\.tar\.gz$' | head -1)"
echo "url: $MTX_URL"
wget -q "$MTX_URL" -O mediamtx.tar.gz
tar xzf mediamtx.tar.gz
./mediamtx --version 2>/dev/null || true

echo "== [3/5] start MediaMTX (RTSP TCP-only, binds 0.0.0.0:8554) =="
pkill -f '[m]ediamtx' 2>/dev/null || true
sleep 1
# Env override forces TCP-only transport regardless of the shipped config's key names.
MTX_RTSPTRANSPORTS=tcp nohup ./mediamtx > /root/mediamtx.log 2>&1 &
sleep 3
tail -n 6 /root/mediamtx.log

echo "== [4/5] publish test pattern -> rtsp://localhost:8554/test (TCP) =="
pkill -f '[f]fmpeg' 2>/dev/null || true
sleep 1
nohup ffmpeg -hide_banner -loglevel warning -re \
  -f lavfi -i "testsrc=size=1280x720:rate=25,drawtext=text='RunPod RTSP MVP %{localtime}':fontcolor=white:fontsize=28:x=20:y=20" \
  -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p -g 50 \
  -f rtsp -rtsp_transport tcp rtsp://localhost:8554/test > /root/ffmpeg.log 2>&1 &
sleep 5
tail -n 6 /root/ffmpeg.log

echo "== [5/5] verify on-box (ffprobe over TCP) =="
ffprobe -rtsp_transport tcp -v error \
  -show_entries stream=codec_name,width,height,avg_frame_rate \
  -of default=noprint_wrappers=1 rtsp://localhost:8554/test \
  && echo "ON-BOX RTSP OK" || echo "ON-BOX RTSP FAILED"
