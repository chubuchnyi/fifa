#!/usr/bin/env bash
# On-pod WebRTC MVP over ICE-TCP (RunPod exposes NO public UDP). Run as root on the pod.
#
# WebRTC media normally rides UDP/ICE; RunPod doesn't expose UDP, so we force ICE-TCP — all media
# over ONE TCP port — and bind it to a RunPod *symmetric* port (request a port >70000; RunPod maps
# external == internal). MediaMTX advertises its ICE candidate as webrtcAdditionalHosts + the LOCAL
# TCP listen port, so only a symmetric port makes the advertised port correct outside the pod.
#
# Validated 2026-06-25 (pod rtsp-mvp / RTX 4090 / cuda11.8): MediaMTX logged
#   [WebRTC] started with listeners on :8889 (TCP/HTTP), :10867 (TCP/ICE)
# and a WHEP POST from a laptop returned 201 with answer candidate
#   a=candidate:... 1 tcp ... 103.196.86.192 10867 typ host tcptype passive
# Final pixel-level playback needs a browser (Chrome/Firefox do ICE-TCP; aiortc does not).
#
# Usage (read the symmetric ICE port + public IP from GraphQL runtime.ports first):
#   ssh -i ~/.ssh/id_ed25519_runpod -p <sshExtPort> root@<publicIp> \
#     'PUBLIC_IP=<publicIp> ICE_TCP_PORT=<symmetric port, priv==pub> bash -s' < scripts/webrtc_mvp_setup.sh
# Then open in a browser:  http://<publicIp>:<signaling 8889 ext port>/test
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
: "${PUBLIC_IP:?set PUBLIC_IP}"
: "${ICE_TCP_PORT:?set ICE_TCP_PORT (the RunPod symmetric port, priv==pub)}"

echo "== [1/6] apt: ffmpeg + tools =="
apt-get update -qq
apt-get install -y -qq ffmpeg wget jq ca-certificates curl iproute2 >/dev/null
ffmpeg -version | head -1

echo "== [2/6] download MediaMTX =="
cd /root
MTX_URL="$(wget -qO- https://api.github.com/repos/bluenviron/mediamtx/releases/latest \
  | jq -r '.assets[].browser_download_url' | grep -E 'linux_amd64\.tar\.gz$' | head -1)"
wget -q "$MTX_URL" -O mediamtx.tar.gz
tar xzf mediamtx.tar.gz
./mediamtx --version || true

echo "== [3/6] write WebRTC ICE-TCP config (UDP off; single TCP port=$ICE_TCP_PORT; advertise $PUBLIC_IP) =="
cat > /root/mediamtx.yml <<EOF
# WebRTC over ICE-TCP only — RunPod exposes no public UDP.
# The advertised ICE candidate = webrtcAdditionalHosts + the TCP listener port,
# so the TCP port MUST be a RunPod *symmetric* port (external == internal) or the
# candidate's port would be wrong outside the pod.
webrtc: yes
webrtcAddress: :8889
webrtcEncryption: no
webrtcLocalUDPAddress: ''
webrtcLocalTCPAddress: :$ICE_TCP_PORT
webrtcIPsFromInterfaces: no
webrtcAdditionalHosts: ["$PUBLIC_IP"]
webrtcICEServers2: []
paths:
  test:
    source: publisher
EOF
cat /root/mediamtx.yml

echo "== [4/6] start MediaMTX =="
pkill -f '[m]ediamtx' 2>/dev/null || true
sleep 1
nohup ./mediamtx /root/mediamtx.yml > /root/mediamtx.log 2>&1 &
sleep 3
echo "--- mediamtx.log ---"; grep -Ei 'webrtc|rtsp|listener|error|warn' /root/mediamtx.log | head -25

echo "== [5/6] publish test pattern -> rtsp://localhost:8554/test (H264 baseline for WebRTC) =="
pkill -f '[f]fmpeg' 2>/dev/null || true
sleep 1
nohup ffmpeg -hide_banner -loglevel warning -re \
  -f lavfi -i "testsrc=size=1280x720:rate=25,drawtext=text='RunPod WebRTC MVP %{localtime}':fontcolor=white:fontsize=28:x=20:y=20" \
  -c:v libx264 -preset veryfast -tune zerolatency -profile:v baseline -pix_fmt yuv420p -g 50 \
  -f rtsp -rtsp_transport tcp rtsp://localhost:8554/test > /root/ffmpeg.log 2>&1 &
sleep 5
echo "--- ffmpeg.log ---"; tail -n 6 /root/ffmpeg.log

echo "== [6/6] verify on-box =="
ffprobe -rtsp_transport tcp -v error -show_entries stream=codec_name,width,height \
  -of default=noprint_wrappers=1 rtsp://localhost:8554/test && echo "INGEST OK" || echo "INGEST FAILED"
echo "-- WHEP endpoint http code (expect 2xx/204) --"
curl -s -o /dev/null -w "OPTIONS /test/whep -> %{http_code}\n" -X OPTIONS http://localhost:8889/test/whep || true
echo "-- listeners (expect TCP :8889 signaling AND TCP :$ICE_TCP_PORT ICE; the WebRTC line must show no UDP) --"
ss -ltnp 2>/dev/null | grep -E ":8889|:$ICE_TCP_PORT" || true
echo "DONE."
