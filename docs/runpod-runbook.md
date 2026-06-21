# RunPod GPU box — provisioning runbook

Reproducible steps to spin up a GPU box on RunPod for pitch3d (`--device cuda`), driven from
**Claude Code** via the RunPod **MCP server** and/or the **`runpodctl`** CLI. This is the
operational "do-it-again" checklist; [`cloud-dev.md`](cloud-dev.md) is the conceptual side (what a
GPU box buys, box spec, gotchas) and [`../scripts/cloud_setup.sh`](../scripts/cloud_setup.sh) is the
on-box installer. Tie-in with the north-star (ADR-0008): the same agent can drive **two** MCP layers
— RunPod MCP manages the *box*, pitch3d's own `serve()` manages the *scene* on it.

## 0. One-time local setup

```bash
# runpodctl CLI — download the release binary into ~/.local/bin (no sudo, on PATH):
curl -fsSL https://api.github.com/repos/runpod/runpodctl/releases/latest \
  | grep -o '"tag_name": *"[^"]*"' | cut -d'"' -f4                 # find latest TAG, e.g. v2.5.0
curl -fsSL -o ~/.local/bin/runpodctl \
  https://github.com/runpod/runpodctl/releases/download/<TAG>/runpodctl-linux-amd64
chmod +x ~/.local/bin/runpodctl
#   (alternative one-liner, needs root: wget -qO- cli.runpod.net | sudo bash)

# Auth the CLI (interactive — prompts for the key from runpod.io/console/user/settings):
runpodctl doctor                  # saves the key; or: export RUNPOD_API_KEY=rpa_xxx
runpodctl me                      # verify — prints balance/account

# Wire the RunPod MCP server into Claude Code (lets the agent create/stop pods):
claude mcp add runpod -e RUNPOD_API_KEY=rpa_xxx -- npx -y @runpod/mcp-server@latest
```

**Do this once and SSH stops being a chore:** add your **public** SSH key in the RunPod console
→ *Settings → SSH Public Keys*. It auto-injects into every pod on boot, so you skip the per-pod
`PUBLIC_KEY` + restart dance in step 3.

## 1. Pick the GPU + check stock

`runpodctl gpu`, or MCP `list-gpu-types` with `searchTerm: "RTX 4090"`. Price/quality pick for
this project is the **RTX 4090 (24 GB)** — only GVHMR (pose) really stresses the GPU; RF-DETR/
ByteTrack fit in far less. Rough rates: Community ~$0.34/hr, **Secure ~$0.69/hr**.

> **Caveat:** the catalog `stockStatus: "High"` / `available: true` is an *aggregate* — it does
> **not** guarantee a host can place your exact pod.

## 2. Create the pod

MCP `create-pod` (or `runpodctl pod create`) with the params that worked:

| Field | Value |
|---|---|
| `gpuTypeIds` | `["NVIDIA GeForce RTX 4090"]`, `gpuCount: 1` |
| `imageName` | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (CUDA 12.4 — **must be ≤ host driver**, see lesson) |
| `cloudType` | `SECURE` (see lesson) |
| `containerDiskInGb` | `40` |
| `ports` | `["22/tcp", "8888/http"]` (SSH + Jupyter) |

**Lesson learned (2026-06-21):** `COMMUNITY` RTX 4090 failed **three times** with
`"This machine does not have the resources to deploy your pod"` despite the catalog showing
High stock — and shrinking the disk / dropping the persistent volume did **not** help. `SECURE`
placed on the first try (US-NC-1, $0.69/hr). So: for a box you need *now*, go Secure; chase the
cheaper Community rate only by retrying later or pinning a datacenter (`dataCenterIds`).

**Lesson learned (2026-06-21) — the image's CUDA must be ≤ the host driver's CUDA.** A
`...-cu1281-...` (CUDA 12.8) image **crash-loops** on today's Secure RTX 4090 hosts: the
container never starts and the console logs cycle
`nvidia-container-cli: requirement error: unsatisfied condition: cuda>=12.8` /
`Current machine CUDA version is 12.4, but the required version for this image is 12.8`.
The host driver here was `550.127.05` = CUDA **12.4**. The base image bakes in
`NVIDIA_REQUIRE_CUDA`, which the `nvidia-container-cli` prestart hook enforces *before the
container even starts* — this is independent of whichever torch wheel you later install. So
pick a **`cuda12.4.1`** image (matches the driver **and** `cloud_setup.sh`'s default `cu124`
torch — verified `torch.cuda.is_available() → True`). Fixing an already-broken pod **without
re-renting the GPU**: `update-pod` the `imageName` to the `cuda12.4.1` tag, then `stop-pod`
+ `start-pod` (the image swap only applies on the next container start; same machine/volume
are kept). The crash-looping pod **bills the whole time**, so fix or stop it promptly.

Secure often grants a small persistent **volume** (e.g. 20 GB at `/workspace`) even if you didn't
ask; clone into it so work survives stop/start.

## 3. SSH access

- **Preferred:** the account-level key from step 0 — nothing else to do.
- **Per-pod fallback:** set `env.PUBLIC_KEY` to your `id_ed25519_runpod.pub` via MCP `update-pod`.
  ⚠️ It only lands in `~/.ssh/authorized_keys` on a container **(re)start**; via MCP that means
  `stop-pod` + `start-pod`, which **releases and re-acquires the GPU** (a small availability risk).
  To avoid the release, open the console **web terminal** and append the key by hand.
- **Connection string (via MCP `get-pod`):** once the container is *healthily* running,
  `get-pod` **does** return the direct SSH endpoint — `publicIp` plus `portMappings` (a Secure
  4090 has `supportPublicIp: true` and gets an IP without renting one). Build the command from
  those two fields:
  - direct: `ssh root@<publicIp> -p <portMappings."22"> -i ~/.ssh/id_ed25519_runpod`
  - proxy (always works): `ssh <podId>-<hash>@ssh.runpod.io -i ~/.ssh/id_ed25519_runpod`
  ⚠️ Two gotchas learned the hard way (2026-06-21): (1) the mapped port **changes on every
  stop/start** (we saw `56185 → 36540` across one cycle), so re-read `get-pod` after each start
  rather than caching it; (2) `publicIp`/`portMappings` are **empty while the container is
  crash-looping** — a *healthy* boot is what populates them, so a blank endpoint is a symptom,
  not an MCP limitation. First connect, pass `-o StrictHostKeyChecking=accept-new` to skip the
  host-key prompt.

## 4. On the box

```bash
git clone git@github.com:chubuchnyi/fifa.git && cd fifa     # SSH remote; repo is "fifa"
./scripts/cloud_setup.sh                                    # CUDA torch + reals + verify GPU
PYTHONPATH=src python -m pitch3d --clip clip.mp4 --frames 6 \
  --detector rfdetr --tracker bytetrack --device cuda \
  --render overlay --export gltf --format smplx_npz --out-dir out/cuda
# push work back (dedicated GitHub key):
GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519_gh_chubuchnyi" git push -u origin main
```

## 5. Cost — stop the box when idle

Billing runs the whole time the pod is `RUNNING` (~$0.69/hr for a Secure 4090).

```bash
runpodctl pod stop <id>     # or MCP stop-pod — GPU stops billing, /workspace volume persists
runpodctl pod start <id>    # or MCP start-pod — resume later
# MCP delete-pod <id>       # tear down completely (also loses the volume)
```
