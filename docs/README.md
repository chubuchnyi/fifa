# pitch3d — documentation map

One screen to find the right doc. **pitch3d** reconstructs an editable, photorealistic 3D
football episode from a single broadcast camera (see the root [`README.md`](../README.md) for the
pitch and quickstart).

Each doc has one job. If two docs seem to say the same thing, the table below says which one owns it.

## Start here

| Doc | What it is |
|---|---|
| [`../CLAUDE.md`](../CLAUDE.md) | **How to work in this repo:** commands, working rules, architecture, the gotchas that cost a session. Read first. |
| [`STATUS.md`](STATUS.md) | **What is true right now and what to do next.** The volatile one — read it in full, every session. |
| [`code-map.md`](code-map.md) | Which file owns which subsystem. |
| [`findings/`](findings/) | Why an open item is open, and what was measured. Cross-thread index: [`findings/research-ledger-2026-08-07.md`](findings/research-ledger-2026-08-07.md). |
| [`../TZ_3D_football_reconstruction.md`](../TZ_3D_football_reconstruction.md) | The requirements spec (ТЗ, v0.3). The *what we must build* baseline. |
| [`architecture.md`](architecture.md) | The design: layering, port contracts, data/control flow, the edit↔render model. |

## Reference

| Doc | What it is |
|---|---|
| [`pipeline.md`](pipeline.md) | **The current clip→final pipeline with diagrams:** reconstruction (perception → export → Cycles) + the structure-locked generative finishing chain. |
| [`scene-schema.md`](scene-schema.md) | Field-by-field spec of the canonical scene model. Verified against `core/scene/scene.py`. |
| [`pipeline-studio.md`](pipeline-studio.md) | Design + build plan for the Studio stage inspector. Phase 1 shipped; Phases 0, 2–6 are not built. |
| [`poseannot-architecture.md`](poseannot-architecture.md) | The annotator's shape. **Behind the code** — it predates Studio, multi-clip and body-pose editing, all of which shipped. |
| [`adr/`](adr/) | Architecture Decision Records (0001–0012). Index: [`adr/README.md`](adr/README.md). Rejected approaches with numbers: [`adr/0012-rejected-approaches-log.md`](adr/0012-rejected-approaches-log.md). |
| [`risk-map.md`](risk-map.md) | Risk register (R-1…R-16) mapped to architectural mitigations. Statuses were scored against the 2026-06 scaffold and never re-scored. |
| [`competitive-landscape.md`](competitive-landscape.md) | Strategy: who else does this, the honest moat, multi-sport extension. |
| [`roadmap.md`](roadmap.md) | **Not current state** (CLAUDE.md says so too) — but it is the per-ticket evidence registry (#94, #105, …) that ADR-0012 and the research ledger cite. Use it to look a ticket up, not to plan. |

## Running it

| Doc | What it is |
|---|---|
| [`local-gpu-box.md`](local-gpu-box.md) | **Start here for GPU work.** The local RTX 4080 runs the full reconstruction chain in Docker, free — 48 frames in 75 s at 24% VRAM. The pod is now only needed for the generative tail and Blender. |
| [`cloud-dev.md`](cloud-dev.md) | *Why* and *what* a GPU box buys us, above the runbooks. |
| [`runpod-runbook.md`](runpod-runbook.md) | **Pod step 1 — from your laptop:** provision a RunPod box (`runpodctl`/MCP, GPU pick, SSH). |
| [`runpod-agent-setup.md`](runpod-agent-setup.md) | **Pod step 2 — on the box:** install the stack, pull weights/datasets, verify CUDA. |
| [`models-dir.md`](models-dir.md) | Where every downloaded checkpoint lives (`models/`, gitignored, ~48 GB). |
| [`pose-bakeoff-runbook.md`](pose-bakeoff-runbook.md) | The pose-model bake-off procedure, scored in metres. Its "WorldPose frames are gated" blocker is **stale** — they are on disk. |
| [`rtsp-mvp.md`](rtsp-mvp.md) | The live-ingest MVP sketch. |

## Visualize the output

| Doc | What it is |
|---|---|
| [`demo.md`](demo.md) | **The one-command demo:** `scripts/demo.sh` runs the whole pipeline and narrates each step. Start here to *show* the product. |
| [`video-demo.md`](video-demo.md) | **The multi-angle video demo:** `scripts/demo_video.sh` renders the animation with Blender Cycles and pulls the `.mp4`s back. The render half is genuinely pod-only. |
| [`blender-demo.md`](blender-demo.md) | Turn an `smplx_npz` export into a skinned **SMPL-X Blender render**. Needs a local Blender binary — the path in its prerequisites table is **dead**. |

## Archive

[`archive/`](archive/) — frozen historical docs. Not living docs; kept for provenance only.
Includes the verbatim status log ([`archive/status-log-2026-07.md`](archive/status-log-2026-07.md),
grep it, don't read it) and, retired 2026-08-07: `m1-status-and-plan.md` (a 2026-06-27 snapshot),
`v0-geometry-defects.md` (all five defects closed), `pipeline-en.md` (superseded by `pipeline.md`;
its A/B section is a frozen run) and `poseannot-roadmap.md` (superseded by `pipeline-studio.md`).
