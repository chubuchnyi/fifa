# M1 — Status & Forward Plan

*Snapshot: 2026-06-22. Companion to `roadmap.md` (which is the step-by-step M1 build log);
this doc is the higher-level "where are we / what's blocking / what's next" view.*

## TL;DR

The full pipeline runs end-to-end on fake adapters (`tests/e2e/test_dry_run.py`) and three
of five perception stages are real: **detector (RF-DETR)** and **tracker (ByteTrack)** are
fully wired built-ins, and the **calibrator** is live via the box-local injected PnLCalib
backend (validated on `messi_sample`: RANSAC + confidence-weighted DLT cut reprojection RMS
3.87 m → 0.36 m). The two remaining live backends — **pose (GVHMR)** and **ball (TrackNet)** —
have real, unit-tested *pure* halves but their heavy/GPU halves are `NotImplementedError`
stubs behind the ADR-0006 dotted-path injection seam.

The single biggest external blocker is **data**: we lack a landscape 16:9 broadcast clip (the
distribution PnLCalib's HRNet was trained on) to measure calibration honestly on independent
footage. Everything else on the near-term path is autonomous, pure-core work.

## Milestone status

| Milestone | Scope | Status |
|---|---|---|
| **M0** | Hexagonal skeleton, ports/adapters, fakes, e2e dry-run | ✅ Complete |
| **M1** | Editable vertical slice: real perception backends behind the seams | 🟡 In progress (calib live; pose+ball stubbed; geometry + typing work open) |
| **M2** | Photoreal render / observer (real Blender SCENE_3D) | ⬜ Not started |
| **M3** | Polish, LLM-over-MCP editing north-star (ADR-0008) | ⬜ Not started |

## Perception-backend matrix

"Pure half" = numpy-only, CPU, unit-tested, lives in public core. "Live half" = heavy/GPL
network, injected box-local via `--X-backend pkg.module:Factory` (ADR-0006) so it stays out
of the public repo.

| Stage | CLI | Pure half | Live half | Notes |
|---|---|---|---|---|
| **Detector** | `--detector rfdetr` | n/a | ✅ built-in (RF-DETR, Apache-2.0) | needs `cv` extra + weights + GPU; no injection seam needed (permissive, in-core) |
| **Tracker** | `--tracker bytetrack` | n/a | ✅ built-in (ByteTrack, MIT) + team clustering | adapter exposes a `TrackingBackend` injection seam, but CLI doesn't thread it → see **T3** |
| **Calibrator** | `--calibrator keypoints` | ✅ DLT + RANSAC + confidence-weighted solve (core) | ✅ injected PnLCalib HRNet (box-local) | validated messi 3.87 m → 0.36 m; `--calibrator-backend` ✓ |
| **Pose** | `--pose gvhmr` | ✅ root-grounding + constraint refit (core) | ❌ `NotImplementedError` stub | `--pose-backend` seam exists; no box-local backend written yet |
| **Ball** | `--ball tracknet` | ✅ threshold + gap-fill (core) | ❌ `NotImplementedError` stub | `--ball-backend` seam exists; no box-local backend written yet |
| Render | `--render overlay` | ✅ reprojection PNGs (dependency-free) | n/a | real, no GPU |
| Export | `--export gltf` | ✅ SMPL-X npz + JSON round-trip | (glTF binary TODO) | |
| Observer | `--observer blender` | — | ⬜ needs Blender + display | see **B4** |

## Blockers / Bugs / Open tickets

| ID | Type | Item | Autonomous? | Notes |
|---|---|---|---|---|
| **B1** | Blocker (data) | No landscape 16:9 broadcast clip (or WorldPose video) to evaluate calibration on independent footage | ❌ needs asset | every clip we have is OOD for PnLCalib's HRNet (portrait / drone / faint amateur markings). WorldPose-Light on the box has annotations but **no video** |
| **B2** | Blocker (decision) | Pose-model pick (GVHMR vs SMPLest-X+SMART vs SAM 3D Body) not finalized | ✅ research-only | benchmark against WorldPose / FIFA-Skeleton; may revise the memory's stub-backend pick |
| **B3** | Blocker (resource) | Pose + ball live backends unwired | ❌ needs GPU box + weights + GPL research repos cloned box-local | mirrors how PnLCalib was wired |
| **B4** | Blocker (env) | Blender observer can't render real SCENE_3D headless without a display/GPU profile | ❌ needs env | M2 concern |
| **Bug1** | Bug (typing) | `refit_port`/`clip` typed as bare `object` in correction engine + assemble | ✅ | latent: `object` has no `.refit`, so misuse is caught only at runtime. Fix = Protocol types |
| **Bug2** | Bug (typing debt) | ~35 mypy errors repo-wide (project doesn't gate on mypy) | ✅ partial | triage + fix cheap ones |
| **T2** | Ticket (quality) | Foot-plane anchoring in `_ground_root` (root Z is a fixed 0.92 m constant today) | ✅ pure geometry | memory: SMART's single biggest quality jump |
| **T3** | Ticket (consistency) | `--tracker-backend` flag absent though the adapter supports `TrackingBackend` injection | ✅ low value | default ByteTrack is real, so not blocking; pure symmetry with pose/ball/calibrator seams |

## Phased plan

### Phase A — Geometry & cheap wins (autonomous, no box) ← executing now
The pure-core, unit-testable work that needs no GPU and no external asset:
- **T2** foot-plane anchoring in `_ground_root` — highest *local* quality leverage.
- **Bug1** typing: `object` → `PoseEstimator | None` / `ClipRef | None` (TYPE_CHECKING guard).
- **T3** thread a `--tracker-backend` seam for symmetry (only if clean and cheap).
- **Bug2** mypy triage: fix the safe ones, document the rest.

### Phase B — Calibration data & honest evaluation (needs asset / box)
- **B1** acquire a landscape 16:9 broadcast clip *or* a WorldPose video.
- Measure PnLCalib end-to-end on it (independent accuracy, not in-sample RMS).
- Evaluate PnLCalib's *own* full camera-calibration module vs our bare DLT.

### Phase C — Pose decision → heavy wiring (research → box)
- **B2** finalize the pose model against WorldPose (research, autonomous).
- **B3** wire the chosen pose backend + ball TrackNet live on the box (box-local, like PnLCalib).
- Then: bundle-adjust pose with player keypoints, not field lines alone.

### Phase D — Polish & observer
- Finish **Bug2** mypy debt; tighten the seams.
- **B4** real Blender SCENE_3D observer (M2); progress toward the LLM-over-MCP north-star (ADR-0008).

## Autonomy boundaries for this run

**Executed autonomously (committed locally at each checkpoint):** Phase A items, B2 research,
and any pure-core code/test/doc work.

**Held for explicit user authorization:** starting the GPU box (~$0.69/hr, deliberately shut
down), pushing to public GitHub, and anything in Phases B/C/D that needs the box, an external
asset, or a display (B1, B3, B4, live pose/ball wiring).
