# Streaming off a RunPod pod — feasibility + MVP (RTSP + WebRTC)

**Question (2026-06-25):** can we bring up a live stream on a RunPod pod and connect to it
from outside? **Answer: yes. RTSP-over-TCP works directly; WebRTC works too — but only over
ICE-TCP (RunPod exposes no public UDP), bound to a RunPod *symmetric* port. Both validated.**

| transport | works on RunPod? | why |
|-----------|------------------|-----|
| **RTSP-over-TCP** | **yes** | media is interleaved on the single RTSP TCP socket, which rides RunPod's Direct-TCP port remap transparently — no separate media ports needed |
| **WebRTC (ICE-TCP)** | **yes** | force WebRTC media onto one TCP ICE port (`webrtcLocalTCPAddress`), disable UDP, and bind it to a RunPod **symmetric** port so the advertised candidate's port is correct outside the pod. No TURN needed |
| WebRTC (native UDP) | no | media is UDP/ICE; RunPod exposes no public UDP, and the naive setup wants a UDP port *range* |
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

## WebRTC over ICE-TCP (validated 2026-06-25)

WebRTC media normally rides UDP/ICE, which RunPod does not expose. The trick: force **ICE-TCP**
(all WebRTC media over one TCP port) and bind that port to a RunPod **symmetric** port — request a
port number `>70000` and RunPod returns a mapping where **external == internal**. MediaMTX advertises
its ICE candidate as `webrtcAdditionalHosts` + the **local** TCP listen port, so only a symmetric
port makes the advertised port correct from outside.

`scripts/webrtc_mvp_setup.sh` installs MediaMTX + ffmpeg and writes this config (`$ICE_TCP_PORT` =
the symmetric port, `$PUBLIC_IP` = the pod IP):

```yaml
webrtc: yes
webrtcAddress: :8889            # WHEP signaling (HTTP)
webrtcEncryption: no
webrtcLocalUDPAddress: ''       # disable UDP ICE
webrtcLocalTCPAddress: :10867   # single TCP ICE port == a RunPod symmetric port
webrtcIPsFromInterfaces: no     # don't leak the container's private IPs
webrtcAdditionalHosts: ["103.196.86.192"]   # advertise the pod's public IP
webrtcICEServers2: []           # no STUN/TURN needed
```

Run it, then open in a **browser**: `http://<publicIp>:<signaling-ext-port>/<path>`
(this session: SSH 22→10864, signaling 8889→10866, **ICE-TCP 10867→10867 symmetric** →
`http://103.196.86.192:10866/test`):

```bash
ssh -i ~/.ssh/id_ed25519_runpod -p <sshExtPort> root@<publicIp> \
  'PUBLIC_IP=<publicIp> ICE_TCP_PORT=<symmetric port> bash -s' < scripts/webrtc_mvp_setup.sh
```

**What was empirically verified**
- MediaMTX log: `[WebRTC] started with listeners on :8889 (TCP/HTTP), :10867 (TCP/ICE)` — no UDP ICE listener.
- Both ports reachable from a home laptop; `OPTIONS /test/whep → 204`.
- A real WHEP `POST → 201`, and the SDP answer advertised exactly
  `a=candidate:… 1 tcp … 103.196.86.192 10867 typ host tcptype passive` — the correct public IP +
  symmetric port. **This is the proof the mechanism works.**

**Honest caveats (R-6)**
- The final pixel-level "it plays" needs an **ICE-TCP-capable client = a browser**. CLI clients like
  `aiortc`/`aioice` do **not** implement client-side ICE-TCP — our probe negotiated fine (`201` +
  correct candidate) but `ice=failed` at connect. Chrome/Firefox do support ICE-TCP.
- Forced-TCP drops WebRTC's low-latency edge (head-of-line blocking; MediaMTX warns of a
  "progressive delay when network is congested"). Fine for a demo; a compromise for hard real-time.
- Signaling here is plain HTTP (`webrtcEncryption: no`). For anything real, terminate TLS.

## Gotchas (each one bit us or would have)

- **Force TCP end-to-end.** Server: `MTX_RTSPTRANSPORTS=tcp`. Client: `vlc --rtsp-tcp`. UDP media
  never traverses the pod boundary.
- **Bind 0.0.0.0**, never 127.0.0.1 (MediaMTX defaults to 0.0.0.0 — fine).
- **Public port is not stable** across stop/start — always re-read `runtime.ports`. A **symmetric**
  port (the `>70000` request) stays symmetric (external == internal) but its actual number still
  re-rolls each restart, so re-read it and re-write the MediaMTX `webrtcLocalTCPAddress` every time.
- **Adding ports to an existing pod:** `update-pod` accepts a new `ports` list, but it only takes
  effect after a **stop → start** (same machine/volume/IP). The in-container `$RUNPOD_TCP_PORT_70000`
  env var is **not** visible in an SSH shell (it's PID-1 scoped) — read the symmetric port from
  `runtime.ports` instead.
- **Teardown:** the pod is named `rtsp-mvp`, deliberately **not** matched by `scripts/pod.sh`'s
  `pitch3d` glob, so `scripts/pod.sh down` will NOT stop it. Stop it explicitly to end billing
  (~$0.69/hr): `mcp__runpod__stop-pod` / `runpodctl pod stop <podId>`.
