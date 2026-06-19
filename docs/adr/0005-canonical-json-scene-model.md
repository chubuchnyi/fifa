# ADR-0005 — Canonical scene model in native tagged JSON; export formats are separate targets

- **Status:** Accepted
- **Date:** 2026-06-19
- **Deciders:** architecture
- **Related:** ADR-0002 (source of truth), ADR-0003 (Z-up/meters); C3, FR-1, FR-25, FR-26, FR-27, FR-28, NFR-7

## Context

The tool needs one **canonical, serializable** representation of a reconstructed episode that:
loses nothing on round-trip (it contains `numpy` arrays, dataclasses and enums); is human-diffable
for review and versioning (FR-25); has **no heavy dependency** so it round-trips in CI without a
DCC tool; leaves room for **multi-camera** later (C3); and maps cleanly onto USD. The interchange
formats (glTF/USD/FBX/Alembic) are each lossy, Y-up, or format-specific — none is a good *store*.

## Decision

1. **Native save format = self-describing tagged JSON** (`core/scene/serialization.py`), using
   reserved keys `__ndarray__`, `__enum__`, `__type__`, `__tuple__`, `__dict__`. It round-trips
   any registered dataclass/enum/array **losslessly with stdlib `json` only**. In production, very
   large arrays move to an `.npz` sidecar; the scaffold keeps everything inline.
2. **Containers are `Source → Episode → Scene → Project`** (`core/scene/scene.py`). A `Scene` is
   the canonical unit: world frame, field+homography, camera, subjects (proposal motion), ball,
   the correction stack, confidence, asset/synth-view refs, and the `RunLog`.
3. **World frame is Z-up / meters** (ADR-0003); exporters convert to Y-up where required.
4. **Multi-camera is a data-model seam, not a build.** The camera is a `CameraTrack` (intrinsics +
   per-frame world→camera pose) and the field homography is a degenerate calibration, so adding
   synchronised calibrated sources later is an adapter/orchestration change, not a core rewrite (C3).
5. **glTF/USD/FBX/Alembic/three.js are `Exporter` *targets*, never the store** (FR-26..28). The
   exported artifact is always the *resolved* scene (ADR-0002), with corrections baked.

## Consequences

**Positive**
- Lossless, dependency-free round-trip — the scene round-trips in unit tests with no DCC tool.
- JSON is diffable, so versioning/named snapshots/rollback (FR-25) are straightforward.
- USD-mappable structure and a list-friendly camera keep USD export and multi-camera open.

**Negative / costs**
- Inline arrays bloat the JSON until the `.npz` sidecar split lands (deferred past the scaffold).
- A bespoke codec is one more thing to maintain and keep in sync with the dataclasses (the
  `_CLASSES` registry must list every serializable type).
- The native file is not directly openable by a DCC tool without going through an exporter.

## Alternatives considered

- **USD as the native store** — rejected: heavyweight dependency, not `numpy`-native, overkill for
  the MVP; kept as an *export* target instead.
- **Pickle** — rejected: unsafe to load untrusted files and not human-diffable.
- **Protobuf/flatbuffers** — rejected: schema-churn cost during scaffolding outweighs the benefit;
  the model is still moving.
