# RTSP streaming off a RunPod pod — feasibility + MVP

**Question (2026-06-25):** can we bring up a live stream on a RunPod pod and connect to it
locally (VLC)? **Answer: yes over RTSP-over-TCP. No for WebRTC within a minimal MVP.**

| transport | works on RunPod? | why |
|-----------|------------------|-----|
| **RTSP-over-TCP** | **yes** | media is interleaved on the single RTSP TCP socket, which rides RunPod's Direct-TCP port remap transparently — no separate media ports needed |
| WebRTC | no (here) | media is UDP/ICE; RunPod does not cleanly expose UDP. Would need TURN-over-TCP — beyond a minimal MVP |
| RTSP-over-UDP | no | needs separate UDP media ports RunPod won't map |

## Validated result

Pod `rtsp-mvp` (RTX 4090, `cuda11.8` image), MediaMTX v1.19.1 + ffmpeg `testsrc`:

- **on-box** `ffprobe -rtsp_transport tcp rtsp://localhost:8554/test` → `h264 1280x720 25/1`, `ON-BOX RTSP OK`
- **from a home laptop** `ffprobe -rtsp_transport tcp rtsp://103.196.86.192:19777/test` → same, `EXTERNAL RTSP OK`

The external probe is exactly the path VLC takes, so a VLC client on the laptop plays the stream.

## Recipe

1. **Rent** a pod with `supportPublicIp:true` and **8554/tcp** exposed as a Direct TCP Port.
   The image must be **CUDA ≤ host driver** — a `cu128` image crash-loops on Secure RTX 4090 /
   CUDA-12.4 hosts (uptime stays 0, billing continues; see `docs/runpod-runbook.md`). We used
   `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`.
2. **Run the on-pod setup** (apt ffmpeg + MediaMTX TCP-only + ffmpeg testsrc + on-box verify):
   ```bash
   ssh -i ~/.ssh/id_ed25519_runpod -p <sshExtPort> root@<publicIp> \
     'bash -s' < scripts/rtsp_mvp_setup.sh
   ```
3. **Read the LIVE external 8554 mapping** from GraphQL `runtime.ports` — **not** the REST
   `portMappings` field, which is stale after a restart. The public port **re-rolls on every
   stop/start** (this session: 8554 → 13737 before restart, → **19777** after):
   ```bash
   key=$(sed -n "s/^apikey *= *'\([^']*\)'.*/\1/p" "$HOME/.runpod/config.toml")
   curl -s "https://api.runpod.io/graphql?api_key=$key" -H 'Content-Type: application/json' \
     -d '{"query":"query { pod(input:{podId:\"<podId>\"}) { runtime { ports { ip isIpPublic privatePort publicPort type } } } }"}' \
     | jq '.data.pod.runtime.ports'
   ```
4. **Connect from VLC**, forcing TCP (RunPod does not pass the UDP fallback):
   ```bash
   vlc --rtsp-tcp rtsp://<publicIp>:<extPort>/test
   ```

## Gotchas (each one bit us or would have)

- **Force TCP end-to-end.** Server: `MTX_RTSPTRANSPORTS=tcp`. Client: `vlc --rtsp-tcp`. UDP media
  never traverses the pod boundary.
- **Bind 0.0.0.0**, never 127.0.0.1 (MediaMTX defaults to 0.0.0.0 — fine).
- **Public port is not stable** across stop/start — always re-read `runtime.ports`.
- **Teardown:** the pod is named `rtsp-mvp`, deliberately **not** matched by `scripts/pod.sh`'s
  `pitch3d` glob, so `scripts/pod.sh down` will NOT stop it. Stop it explicitly to end billing
  (~$0.69/hr): `mcp__runpod__stop-pod` / `runpodctl pod stop <podId>`.
