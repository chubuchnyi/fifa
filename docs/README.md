# pitch3d — documentation map

One screen to find the right doc. **pitch3d** reconstructs an editable, photorealistic 3D
football episode from a single broadcast camera (see the root [`README.md`](../README.md) for the
pitch and quickstart).

Each doc has one job. If two docs seem to say the same thing, the table below says which one owns it.

## Start here

| Doc | What it is |
|---|---|
| [`../README.md`](../README.md) | Entry point: what the tool is, quickstart, layout. |
| [`../TZ_3D_football_reconstruction.md`](../TZ_3D_football_reconstruction.md) | The requirements spec (ТЗ, v0.3). The *what we must build* baseline. |
| [`architecture.md`](architecture.md) | The design: layering, port contracts, data/control flow, the edit↔render model. **Read this to understand how it fits together.** |

## Plan & current state

| Doc | What it is |
|---|---|
| [`roadmap.md`](roadmap.md) | The durable plan: milestones M0→M4, the M1 vertical slice, per-ticket tables. *Where we're going.* |
| [`m1-status-and-plan.md`](m1-status-and-plan.md) | The dated snapshot: what's wired, current blockers (B1–B4), the phase plan. *Where we are right now* — this is the volatile one. |
| [`risk-map.md`](risk-map.md) | Risk register (R-1…R-16) mapped to architectural mitigations. |
| [`competitive-landscape.md`](competitive-landscape.md) | Strategy: who else does this, the honest moat, multi-sport extension. |

## Reference

| Doc | What it is |
|---|---|
| [`pipeline.md`](pipeline.md) | **The current clip→final pipeline with diagrams:** reconstruction (perception → export → Cycles) + the structure-locked generative finishing chain, why each step exists, where each piece lives. |
| [`scene-schema.md`](scene-schema.md) | Field-by-field spec of the canonical scene model (the data types). |
| [`adr/`](adr/) | Architecture Decision Records (0001–0010) — each decision, its context, and why. Index: [`adr/README.md`](adr/README.md). |

## Visualize the output

| Doc | What it is |
|---|---|
| [`demo.md`](demo.md) | **The one-command demo:** `scripts/demo.sh` runs the whole pipeline (real GPU perception → render → export → bodies) and narrates each step. Start here to *show* the product. |
| [`video-demo.md`](video-demo.md) | **The multi-angle video demo:** `scripts/demo_video.sh` renders the whole *animation* (bodies + ball, several camera angles) **on the GPU pod** with Blender Cycles and pulls the `.mp4`s back. |
| [`blender-demo.md`](blender-demo.md) | Turn an `smplx_npz` export into a skinned **SMPL-X Blender render** (close-up + scene) — the real body mesh, not the `--observer blender` box proxies. |

## Running on a GPU box

| Doc | What it is |
|---|---|
| [`cloud-dev.md`](cloud-dev.md) | *Why* and *what* a GPU box buys us (concept + gotchas), above the runbooks. |
| [`runpod-runbook.md`](runpod-runbook.md) | **Step 1 — from your laptop:** provision a RunPod box (`runpodctl`/MCP, GPU pick, SSH). |
| [`runpod-agent-setup.md`](runpod-agent-setup.md) | **Step 2 — on the box:** install the stack, pull weights/datasets, verify CUDA. |
| [`pose-bakeoff-runbook.md`](pose-bakeoff-runbook.md) | The pose-model bake-off procedure (SMPLest-X vs fallback), scored in metres. |

## Archive

[`archive/`](archive/) — frozen historical docs (e.g. the executed kickoff brief). Not living docs;
kept for provenance only.
