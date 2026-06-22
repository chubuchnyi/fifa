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
| **Tracker** | `--tracker bytetrack` | n/a | ✅ built-in (ByteTrack, MIT) + team clustering | `--tracker-backend` threads the `TrackingBackend` injection seam (T3 ✅), symmetric with pose/ball/calibrator |
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
| **Bug2** | Bug (lint/type debt) | mypy + ruff debt repo-wide (project gates on neither) | ✅ partial | mypy 33→22 via safe fixes; rest documented below |
| **T2** | Ticket (quality) | Foot-plane anchoring in `_ground_root` (root Z is a fixed 0.92 m constant today) | ❌ coupled to B3 | re-scoped 2026-06-22 — see design note below. The quality jump needs SMPL-X FK (foot→pelvis offset) from the heavy backend; not faithfully doable in the pure half |
| **T3** | Ticket (consistency) | ~~`--tracker-backend` flag absent though the adapter supports `TrackingBackend` injection~~ | ✅ **done** (09ef3f9) | flag now threads the seam; +1 injection +1 guard test (213→215). Symmetric with pose/ball/calibrator |

## Phased plan

### Phase A — Cheap wins & honest scoping (autonomous, no box) ← executing now
The pure-core, unit-testable work that needs no GPU and no external asset:
- **Bug1** typing: `object` → `PoseEstimator | None` / `ClipRef | None` (TYPE_CHECKING guard). ✅ done.
- **T2** investigated → re-scoped to Phase C (its quality core needs FK from B3; see design note).
- **T3** thread a `--tracker-backend` seam for symmetry (only if clean and cheap). ✅ done — the
  `TrackingBackend` protocol was already `@runtime_checkable` and `ByteTrackTracker` already took
  `backend=`, so it was a 4-line symmetric wiring + tests.
- **Bug2** mypy triage: fix the safe ones, document the rest.

### Phase B — Calibration data & honest evaluation (needs asset / box)
- **B1** acquire a landscape 16:9 broadcast clip *or* a WorldPose video.
- Measure PnLCalib end-to-end on it (independent accuracy, not in-sample RMS).
- Evaluate PnLCalib's *own* full camera-calibration module vs our bare DLT.

### Phase C — Pose decision → heavy wiring (research → box)
- **B2** finalize the pose model against WorldPose (research, autonomous).
- **B3** wire the chosen pose backend + ball TrackNet live on the box (box-local, like PnLCalib).
- **T2** foot-plane anchoring — implement alongside B3, once the backend yields foot/pelvis joints.
- Then: bundle-adjust pose with player keypoints, not field lines alone.

## Design note — T2 foot-plane anchoring (why it's coupled to B3)

`GVHMRPoseEstimator._ground_root` (pose.py) projects each tracklet's bbox foot point to world
XY via the homography (correct, anti-slide via path smoothing) and sets **root Z to a fixed
0.92 m** nominal pelvis height for every subject and every frame. The real "foot-plane anchor"
(SMART's biggest jump) makes the **feet** sit on the plane and derives the **per-frame pelvis
height** from the actual articulation — i.e. the vertical foot→pelvis offset, which varies with
crouch/run/jump/slide.

That offset is a forward-kinematics quantity: it needs SMPL-X joint *positions*, not just the
axis-angle articulation. The pure half doesn't have them — `RawBodyMotion` carries only
`global_orient` / `body_pose` / `betas`, and `FieldCalibration` exposes only a ground homography
(no full camera / vertical vanishing point, so single-view height metrology from the bbox isn't
available either). So the honest options are:

1. **Backend supplies it (preferred):** extend `RawBodyMotion` with a per-frame `pelvis_above_foot`
   (or the foot/pelvis joint Z) that the heavy backend computes from the SMPL-X mesh; `_ground_root`
   then anchors `root_z = plane_z + pelvis_above_foot` and falls back to the 0.92 m constant when
   absent. Byte-identical on the fake; the quality jump arrives with B3.
2. **Richer calibration:** keep PnLCalib's full camera (not just the homography) so heights can be
   measured — larger change, and still inferior to (1) for articulated subjects.

Decision: do **not** ship a pure-half heuristic now (it would be unvalidatable without B1 data and
would fake precision we don't have). Implement (1) together with B3.

## Lint/type debt (Bug2) — measured 2026-06-22

The project gates on neither mypy nor ruff, so debt has accumulated. Measured state:

- **mypy: 33 → 22** after safe fixes. Fixed: `ReconstructionResult.detections/tracks` were bare
  `object` (now `Detections`/`Tracks`); a `RenderResult` shadowing the `render: str` param in
  `cli.py`; widened a `look_at`/`camera_at` `up` default; explicit RGB 3-tuples in overlay/proxy.
  Remaining 22, deferred with reason:
  - **~14** — `Correction.payload` is `object`; the engine narrows it by `corr.mode` (which mypy
    can't follow), so a proper fix needs `isinstance`/`cast` at each dispatch site. Invasive on
    pure, well-tested code; low value. Deferred.
  - **3** `controller.py` + **3** `blender/runner.py` — real `None`-handling gaps (`BallTrack | None`,
    `SceneObserver | None`); each needs a behaviour decision + a covering test, not a blind guard.
  - **1** `detection.py` lazy `_model: object` — intentional, to avoid importing torch at module load.
  - **1** `tracking.py` list-comprehension element type.
- **ruff: 51 → 49** (removed an unused import + sorted a block in `cli.py`). The remaining 49 span
  23 files (e.g. `controller.py` has 14) and are mostly `E501` long lines. Pre-existing; not churned.

Stance: fix debt opportunistically in files we're already editing for real reasons; don't do a
repo-wide lint sweep as a standalone change (high churn, zero functional value, ungated).

### Phase D — Polish & observer
- Finish **Bug2** mypy debt; tighten the seams.
- **B4** real Blender SCENE_3D observer (M2); progress toward the LLM-over-MCP north-star (ADR-0008).

## Autonomy boundaries for this run

**Executed autonomously (committed locally at each checkpoint):** Phase A items, B2 research,
and any pure-core code/test/doc work.

**Held for explicit user authorization:** starting the GPU box (~$0.69/hr, deliberately shut
down), pushing to public GitHub, and anything in Phases B/C/D that needs the box, an external
asset, or a display (B1, B3, B4, live pose/ball wiring).
