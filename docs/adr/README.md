# Architecture Decision Records

Each ADR records one decision: its context, the decision, consequences (positive and costs), and
the alternatives rejected. They are the *why* behind [`../architecture.md`](../architecture.md).
Format follows [MADR](https://adr.github.io/madr/)-style headers; all are **Accepted** (2026-06-19 onward).

| ADR | Decision | Key drivers |
|---|---|---|
| [0001](0001-hexagonal-core.md) | Hexagonal core, pure-`numpy`; everything else behind a port | C1, NFR-6, AC-7 |
| [0002](0002-dual-representation-source-of-truth.md) | Dual representation; `proposal ⊕ corrections → resolved` is the single source of truth | C2, FR-21..24, R-3 |
| [0003](0003-blender-editing-host-external-render.md) | Blender is the editing host; photoreal render is external and imported back | C2, C6, R-2, R-9 |
| [0004](0004-offline-queue-and-content-cache.md) | Offline job queue + content-addressable cache | C5, NFR-4, UX-8, R-10 |
| [0005](0005-canonical-json-scene-model.md) | Canonical model in native tagged JSON; export formats are separate targets | C3, FR-1, FR-25..28 |
| [0006](0006-swappable-model-providers.md) | Swappable model providers behind one `ModelProvider` family, with provenance | C4, NFR-6/7/8, UX-7 |
| [0007](0007-viewsynthesizer-two-seams.md) | One `ViewSynthesizer` port, two seams (A render / B amplify+inpaint) | C6, FR-29..32, R-14..16 |
| [0008](0008-llm-in-the-loop-mcp-and-visual-feedback.md) | LLM-in-the-loop editing via MCP + multi-view visual feedback | FR-21..23, UX-4 |
| [0009](0009-device-config-and-cpu-validation-profile.md) | Inference `device` is a runtime knob; CPU is the local validation profile, GPU the production target | NFR-6, M1 |
| [0010](0010-live-blender-edit-bridge.md) | Live Blender edits become `Correction`s over a socket; the host owns the scene, human ≡ LLM | C2, FR-14/21/22a, ADR-0008 |
| [0011](0011-deliverable-video-path-contract-and-virtual-operator.md) | Deliverable video path: package CLI + versioned export manifest + virtual-operator cameras | R-6, eye-verdict 2026-07-03, ADR-0003 |
| [0012](0012-rejected-approaches-log.md) | Rejected approaches, tiered by evidence; every entry records what would re-open it | R-6, roadmap R8, v2 §9 |
