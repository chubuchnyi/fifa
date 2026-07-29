# Football 3D Reconstruction — Implementation Brief

Target: agentic implementation via Claude Code. Companion to `football-3d-pipeline-v2.md` (which holds the *why*, the benchmark numbers, and the rejected-alternatives log). This document holds the *what* and *how*.

**Confidence convention.** `[meas.]` measured in a cited work · `[claimed]` stated by authors without absolute figures · `[est.]` derived from first principles, must be benchmarked before it enters a spec.

---

## 0. How to use this document

| Role | Read | Owns |
|---|---|---|
| Product Owner | §1, §11 | Milestone acceptance criteria; scope defence |
| Architect | §2, §3, §4, §6, §10 | Port contracts; frame conventions; no adapter leaks into core |
| Developer | §3–§7, §12, §13 | Adapters; factor graph; keeping invariants green |
| Tester | §8, §9 | Golden tests; property tests; regression gates |
| Tech Writer | §2, §14 | Frame conventions doc; open-questions log |

**Split before use.** This file is deliberately one document for review. Before handing to agents, split into `CLAUDE.md` (§1, §2, §8, §12 — the always-loaded context) and `docs/spec/*.md` (the rest, loaded on demand). §2 and §12 must be in the always-loaded set: they are the two sections whose violation produces silent, plausible, wrong output.

---

## 1. Scope and non-goals

### In scope

Reconstruct, from a single camera stream:
- Player positions on the pitch in metric world coordinates
- Player 3D articulated pose (SMPL-X parameters + derived joints)
- Ball position, with an explicit measured/estimated/unmeasured mode flag
- Derived coaching analytics: distance, speed, sprints, pitch control, body orientation

Two deployment contours with a shared core:
- **Static** — fixed 4K camera covering the pitch, calibration solved once. Training sessions.
- **Broadcast** — panning/zooming main camera with cuts and replays. Matches.

### Explicit non-goals

- Photorealistic rendering (separate project layer)
- Real-time 3D pose (see §13.2 — the GPU budget forbids it)
- Individual trajectory accuracy for off-screen players (imputation serves team metrics only)
- Any claim of measurement where the pipeline produces an estimate

### Accuracy envelope — build to this, not better

| Quantity | Broadcast | Static camera |
|---|---|---|
| Global MPJPE | 0.35–0.45 m | < 0.15 m (target) |
| Local MPJPE | < 0.06 m | < 0.06 m |
| Planar position | 0.3–0.5 m | < 0.15 m |
| Ball, on ground | 0.3–0.5 m | < 0.2 m |
| Ball, airborne | **unmeasured** unless audio TDOA available | — |
| Hidden player position | 10–17 m median | n/a |

Reference: best published single-broadcast-camera result is Global MPJPE 0.324 m / Local 0.054 m (FIFA Skeletal Tracking Challenge 2026, SMART, arXiv 2605.31551) `[meas.]`. Multi-camera industrial systems reach ~0.10 m `[meas.]`. Do not spec below the broadcast envelope.

---

## 2. Coordinate frames and units — READ FIRST

Frame confusion is the dominant bug class in this system and it fails *silently* with plausible output. Every function that crosses a frame boundary names both frames in its signature. No exceptions.

### 2.1 Frames

**Pitch frame `W` (world).** The single source of truth.
- Origin: centre mark
- `+X`: along pitch length, toward the goal that is on the right in the main-camera view at the start of the analysed half
- `+Y`: along pitch width, chosen so the frame is right-handed with `+Z` up
- `+Z`: up, normal to the playing surface
- Units: metres. Ground plane is exactly `z = 0`
- Extents: `X ∈ [-L/2, +L/2]`, `Y ∈ [-Wd/2, +Wd/2]`, defaults `L = 105.0`, `Wd = 68.0`, both configurable per venue

**Camera frame `C`.** OpenCV convention, no deviation.
- `+X` right in the image, `+Y` down, `+Z` forward along the optical axis
- Extrinsics stored as **world → camera**: `X_C = R · X_W + t`
- Camera centre in world is therefore `c_W = -Rᵀ · t`. Never store `c_W` as the primary representation; derive it
- `R` stored as a rotation matrix in the domain model, as an `so3` tangent vector inside the optimiser only

**Image frame `I`.** Pixels, origin at the top-left corner of the top-left pixel, `+x` right, `+y` down. Intrinsics `K = [[fx,0,cx],[0,fy,cy],[0,0,1]]`. Distortion: Brown–Conrady `(k1, k2, p1, p2)`, `k3` optional. Undistortion happens once, at ingest, and everything downstream assumes undistorted pixels — record this in the `Frame` metadata so it cannot be applied twice.

**SMPL frame `S`.** SMPL-X has its own canonical orientation. **Do not hardcode the `S → W` rotation from memory.** Define it as a module constant `SMPL_TO_WORLD: np.ndarray` and pin it with a golden test (§9.2): load a known rest-pose mesh, assert the head is at higher `z` than the feet and the body faces `+X` when `global_orient` is identity. If that test is absent, the constant is unverified.

### 2.2 Units and conventions

| Quantity | Unit | Notes |
|---|---|---|
| Length | m | Never cm, never px outside the image frame |
| Time | s | Frame index is `int`; timestamps are `float` seconds from clip start |
| Angle | rad | Degrees only at the presentation boundary |
| Rotation | rotation matrix in domain, `so3` in optimiser | Never Euler angles anywhere |
| Focal length | px | Convert to mm only for reporting |
| Confidence | `[0, 1]` | Never a raw logit or a distance |

### 2.3 Naming discipline

Suffix every spatial variable with its frame: `root_W`, `joints_C`, `bbox_I`. A function named `project(points_W, camera) -> points_I` is self-documenting; `project(points, cam)` is a future bug.

---

## 3. Domain model

Pure data, no I/O, no framework dependencies. `frozen=True` dataclasses or pydantic models — pick one and be consistent.

```python
# core/domain/geometry.py

@dataclass(frozen=True)
class PitchModel:
    length_m: float = 105.0
    width_m: float = 68.0
    line_width_m: float = 0.12          # IFAB max; used for two-edge line fitting
    # Named landmarks in W, generated from the two dimensions above.
    def landmarks_W(self) -> dict[str, np.ndarray]: ...

@dataclass(frozen=True)
class CameraParams:
    K: np.ndarray                        # 3x3
    dist: np.ndarray                     # (k1,k2,p1,p2[,k3])
    R_wc: np.ndarray                     # 3x3, world -> camera
    t_wc: np.ndarray                     # (3,), world -> camera
    frame_idx: int
    covariance: np.ndarray | None = None # 6x6 on se3, if the solver produced one

    @property
    def center_W(self) -> np.ndarray:    # derived, never stored
        return -self.R_wc.T @ self.t_wc
```

```python
# core/domain/tracking.py

class ShotType(StrEnum):
    MAIN_CAMERA = "main_camera"
    REPLAY      = "replay"
    CLOSE_UP    = "close_up"            # feeds beta estimation ONLY
    STUDIO      = "studio"

class Role(StrEnum):
    PLAYER = "player"; GOALKEEPER = "goalkeeper"
    REFEREE = "referee"; OTHER = "other"

@dataclass(frozen=True)
class Shot:
    start_frame: int
    end_frame: int                       # exclusive
    shot_type: ShotType
    confidence: float

@dataclass(frozen=True)
class Detection:
    frame_idx: int
    bbox_I: np.ndarray                   # (x1,y1,x2,y2)
    score: float
    class_id: int

@dataclass(frozen=True)
class Tracklet:
    track_id: int
    detections: list[Detection]
    team: Literal["left", "right"] | None
    role: Role
    jersey_number: int | None
    jersey_confidence: float             # expect ~0.45 coverage in practice
```

```python
# core/domain/state.py

class Provenance(StrEnum):
    MEASURED = "measured"
    IMPUTED  = "imputed"                 # off-screen; team metrics only
    INTERPOLATED = "interpolated"        # short visual gap, bridged

class BallMode(StrEnum):
    ON_GROUND  = "on_ground"             # z pinned to 0
    BALLISTIC  = "ballistic"             # z from segment fit
    UNMEASURED = "unmeasured"            # DO NOT feed to analytics

@dataclass(frozen=True)
class PlayerState:
    frame_idx: int
    track_id: int
    root_W: np.ndarray                   # (3,) pelvis
    com_W: np.ndarray | None             # (3,) centre of mass, see §7.4
    orient_W: np.ndarray                 # (3,) axis-angle, global orientation
    pose_body: np.ndarray                # (63,) or VPoser latent (32,)
    joints_W: np.ndarray                 # (J,3) derived, cached
    velocity_W: np.ndarray               # (3,) m/s
    contact: tuple[bool, bool]           # (left_foot, right_foot)
    provenance: Provenance
    confidence: float

@dataclass(frozen=True)
class BallState:
    frame_idx: int
    position_W: np.ndarray
    velocity_W: np.ndarray | None
    spin_W: np.ndarray | None
    mode: BallMode
    confidence: float

@dataclass(frozen=True)
class PlayerShape:
    """One per (match, track identity). Constant across the whole match."""
    track_identity: str                  # jersey-anchored where available
    betas: np.ndarray                    # (10,)
    height_m: float                      # derived from betas; the scale anchor
    n_closeup_frames: int                # provenance of the estimate

@dataclass(frozen=True)
class GameState:
    frame_idx: int
    camera: CameraParams
    players: list[PlayerState]
    ball: BallState
```

**Invariant carried by the type system where possible:** anything with `provenance != MEASURED` or `mode == UNMEASURED` must be visually distinct in every renderer and excluded from any per-player technical metric. Do not rely on discipline; make the analytics functions take `Sequence[PlayerState]` and filter explicitly, and unit-test the filter.

---

## 4. Ports

`typing.Protocol`, in `core/ports/`. The core imports nothing from `adapters/`.

```python
class ShotSegmenter(Protocol):
    def segment(self, video: VideoHandle) -> list[Shot]: ...

class Detector(Protocol):
    def detect(self, frames: Sequence[Frame]) -> list[list[Detection]]: ...

class Tracker(Protocol):
    def track(self, dets: Sequence[list[Detection]]) -> list[Tracklet]: ...

class TeamAssigner(Protocol):
    def assign(self, tracklets: Sequence[Tracklet], frames: Sequence[Frame]
               ) -> dict[int, Literal["left","right"]]: ...

class FieldRegistrar(Protocol):
    """Per-frame registration to the pitch model. PnLCalib-class."""
    def register(self, frame: Frame, pitch: PitchModel) -> CameraParams | None: ...

class CameraTracker(Protocol):
    """Temporal propagation between anchors. RAFT+MAD-class."""
    def propagate(self, frames: Sequence[Frame], anchor: CameraParams,
                  pitch_mask: Sequence[np.ndarray]) -> list[CameraParams]: ...

class MeshLifter(Protocol):
    """Per-frame SMPL-X regression from a crop. SMPLest-X-class."""
    def lift(self, crops: Sequence[Crop]) -> list[MeshEstimate]: ...
    # MeshEstimate carries betas, body_pose, global_orient, AND pelvis_depth_C.
    # pelvis_depth_C is load-bearing (see §12.3) — a lifter without it is not usable.

class BallDetector(Protocol):
    def detect(self, frames: Sequence[Frame]) -> list[BallObservation | None]: ...
    # BallObservation carries centre_I, radius_px, score.
    # radius_px is the monocular depth cue — never discard it.

class ContactSpotter(Protocol):
    """Ball-touch events with player attribution. FOOTPASS/DST-class."""
    def spot(self, frames, tracklets) -> list[TouchEvent]: ...

class Imputer(Protocol):
    def impute(self, visible: Sequence[PlayerState], roster_size: int,
               pitch: PitchModel) -> list[PlayerState]: ...

class Optimizer(Protocol):
    def solve(self, problem: FactorGraphProblem) -> FactorGraphSolution: ...

class Sink(Protocol):
    def write(self, states: Iterable[GameState]) -> None: ...
```

---

## 5. Adapters

| Port | Primary adapter | Fallback / notes |
|---|---|---|
| `ShotSegmenter` | Small CNN on histogram + motion features, trained on labelled cuts | Must exist before any 3D work (§12.1) |
| `Detector` | RF-DETR or YOLO26 fine-tuned on SoccerNet | + adaptive tiling for ball and distant players |
| `Tracker` | BoT-SORT + ReID (DINOv2 embeddings) | ByteTrack as a lighter baseline |
| `TeamAssigner` | Torso-colour clustering, per match | NOT open-vocabulary detection (§12.7) |
| `FieldRegistrar` | PnLCalib (arXiv 2404.08401) | No Bells Just Whistles |
| `CameraTracker` | RAFT-small + MAD 3σ + RANSAC homography | Params in §7.2 |
| `MeshLifter` | SMPLest-X ViT-H fine-tuned on WorldPose | ViT-S variant for the live contour (§13.2) |
| `BallDetector` | TrackNet-family heatmap over frame stack | **Must be fine-tuned on football** (§12.6) |
| `ContactSpotter` | FOOTPASS TAAD+DST | |
| `Imputer` | B2 formation anchor → B4 centroid voting | Reference impl: github.com/nowayfootball/offscreen-impute |
| `Optimizer` | PyTorch + Theseus, sparse LM | Ceres/g2o for the camera-only stage |
| `Sink` | Parquet + JSON | |

**Adapter rule:** every adapter is constructible from a config dict and has a `NullAdapter` / `StubAdapter` twin returning fixed synthetic data. The stubs are what let the pipeline be integration-tested without GPUs, and they are not optional.

---

## 6. Pipeline composition

```
ingest ─► shot segmentation ─┬─► MAIN_CAMERA ──────────────► pose pipeline
                             ├─► CLOSE_UP ──► beta estimation (§7.5)
                             └─► REPLAY / STUDIO ──────────► discard

pose pipeline (per shot):
  detect ─► track ─► team assign ─► jersey OCR
     │
     ├─► field registration (per frame, anchors)
     │        └─► camera tracking (dense, between anchors)
     │                 └─► camera bundle adjustment ────► stage A
     │
     ├─► crop ─► mesh lift ─► foot-plane anchoring ────► init for stage B
     │
     └─► ball detect ─► contact spot ─► segment split

  factor graph:  stage A (camera) ─► stage B (per player) ─► stage C (coupled)

  post: imputation ─► smoothing ─► analytics ─► sink
```

Orchestration lives in `core/pipeline/`, is pure, and takes all ports by constructor injection. It must be runnable end-to-end with every port stubbed, on CPU, in under 10 seconds. That run is the smoke test.

---

## 7. Factor graph — implementation detail

Everything in this section is `[est.]` unless marked. No published system does exactly this on football; the structure is derived from bundle adjustment and is sound, the timings are not measured.

### 7.1 Variables

Window = one shot, capped at 10 s with 1 s overlap. Broadcast main-camera shots run 3–15 s.

| Variable | Dim | Count (N=250, P=14) | Init | Bound |
|---|---|---|---|---|
| `R_t` | 3 (so3) | 250 | RAFT chain from PnLCalib anchor | \|ΔR\| < 60°/frame |
| `t_t` | 3 | 250 | same | height 5–50 m |
| `f_t` | 1 | 250 | PnLCalib per-frame, smoothed | monotone within a zoom segment |
| `k1,k2` | 2 | 1 per shot | PnLCalib | — |
| `betas_p` | 10 | 14, **per match** | median over close-ups | ±3σ PCA, tanh reparam |
| `z_p,t` (VPoser) | 32 | 3500 | encode SMPLest-X θ | penalty on ‖z‖ |
| `root_p,t` | 3 | 3500 | foot-plane anchoring | \|v\|<11 m/s, \|a\|<9 m/s² soft |
| `orient_p,t` | 3 | 3500 | SMPLest-X | — |
| `ball_t` | 3 | 250 | z=0 projection + radius cue | pitch + margin |
| `spin_seg` | 3 | ~20 | 0 | ‖ω‖ < 10 rev/s |
| `t_switch` | 1 | ~20 | from contact spotter | ±0.1 s from event |

Total ≈ 136k variables. All bounds are **soft one-sided hinges** except `betas` (reparameterised) — hard bounds destabilise LM.

### 7.2 Residuals

| Residual | Weight | Notes |
|---|---|---|
| 2D keypoint reprojection | 1.0 | primary data term |
| Silhouette reprojection (SAM mask) | 0.3 | contested frames only — expensive |
| Foot contact `z=0` | 0.5 | soft binary with hysteresis, jointly optimised |
| CoM parabola when `contact=false` | 0.5 | §7.4 |
| Angular momentum conservation in flight | 0.2 | §7.4 |
| Capsule non-penetration | 0.3 | active set, re-linearised per outer iteration |
| Joint limits | 0.2 | §7.3 — **not free from SMPL** |
| Root velocity / acceleration ceilings | 0.1 | |
| Acceleration prior (smoothness) | 0.1 | on accelerations, never on positions |
| Ball ballistics with drag + Magnus | 0.5 | between switching times |
| Ball at foot/head on touch event | 1.0 | strongest height constraint available |

Camera tracking parameters `[meas.]` from SMART: RAFT-small dense flow inside the convex hull of projected pitch landmarks at stride 4; MAD 3σ outlier rejection; RANSAC homography 2000 iterations, 1 px threshold; reject updates with |ΔR| > 60°. Rotation error 0.041° vs 0.043° for Lucas–Kanade; RAFT-large gives no gain at 2× cost; ECC homography degrades to 0.107° because players occupy too much of the frame.

### 7.3 Joint limits are not free

SMPL's pose space is per-joint axis-angle with **no hard limits**. Hyperextended knees are representable and reachable. Regressors avoid them because their training distribution does, not because the model forbids it. If this is assumed free, the constraint never gets written and never appears.

Two layers:
1. VPoser latent norm penalty — soft, implicit, covers the whole body
2. Explicit one-sided hinges for the one-DOF joints: reverse flexion of knee and elbow, and off-axis components of those joints

### 7.4 Airborne: CoM, not root

Foot-plane anchoring breaks when the player is airborne, and jumps are underrepresented in WorldPose `[meas.]`.

The pelvis is offset from the centre of mass by ~0.10–0.15 m vertically, and the offset **changes with trunk flexion**. A parabola applied to `root_W` produces a hanging-pelvis artefact on headers. Apply it to `com_W`.

CoM derivation: segment masses (de Leva or Dempster) assigned to mesh parts; `betas` already gives per-subject segment volumes, so the scaling is individual, not tabular. Validation source: **AddBiomechanics** (273 subjects, >70 h with force plates, >24 M frames, open). No published SMPL-CoM-vs-force-plate validation found — this is an open item (§14).

**Second, stronger constraint: angular momentum is conserved in free flight.** Total angular momentum about the CoM is constant while airborne. This couples the whole body configuration across frames and is independent of the CoM parabola. It is what catches the tuck-jump case: the CoM parabola holds regardless of configuration, but tucking changes the moment of inertia, so angular velocity must rise correspondingly. Violations indicate lifter error.

Contact classification is a residual, not a separate network: ankle height above the pitch plane, foot vertical velocity, contact area from the mask. Soft binary with hysteresis, jointly optimised. Hard switching chatters at the boundary.

### 7.5 Beta comes from the discarded footage

Body shape is best recovered where the player is large in the image — close-ups and replays, discarded by shot segmentation. Route `CLOSE_UP` shots into beta estimation only, keyed by jersey number, and apply the result to main-camera frames.

This removes the scale ambiguity for free: `betas` gives player height, height gives metric scale.

Outer loop over the match: fix `betas` → solve shots in parallel → refit `betas` across all shots → repeat, 2–3 iterations.

### 7.6 Sparsity and staging

The Hessian is exactly bundle-adjustment structured: "cameras" are frames, "landmarks" are per-frame player states. Reprojection is the only camera↔player coupling and it is per-frame. Within a player, temporal factors form a block-tridiagonal chain. `betas` is shared across the match — the arrowhead.

Solve by Schur complement on the player blocks: each chain eliminates in O(N) by a Thomas-like sweep, leaving a reduced 1752-variable camera system that solves directly.

Collisions break chain independence but are rare — pairs within 1.5 m are roughly 5–15% of pairs. Handle as an active set; the coupling is local in time and space, so marginalisation widens the band slightly rather than destroying the structure.

**Staged, not monolithic:**

```
Stage A — camera only.       1752 vars. Ceres/g2o. Milliseconds. Freeze.
Stage B — per player,        P independent ~9500-var problems.
          camera frozen.     Embarrassingly parallel, batched on GPU.
Stage C — joint refine       Only in temporal windows containing close pairs.
          with collisions.   Typically 10–20% of the shot.
```

~90% of the benefit at ~10% of the complexity, and each stage is separately testable. Monolithic joint solve is a later architecture revision, not v1.

Solver: PyTorch + Theseus, sparse LM with Cholesky. SMPL forward kinematics autodifferentiates for free and stays the same model as in training. **LM, not Dogleg** — Dogleg wins with poor initialisation, and you start at 0.37 m global error. 15–30 iterations `[est.]`. Tens of seconds per 10 s shot on one modern GPU `[est.]` — post-match contour. Benchmark before this number enters any spec.

---

## 8. Invariants

Assert these in `core/invariants.py`. Run in tests always; run in production behind a `--strict` flag. A violation is a bug, not a warning.

| # | Invariant | Where |
|---|---|---|
| I1 | `abs(min_ankle_z) < 0.05 m` whenever `contact` is true | per player-frame |
| I2 | `\|position_W\|` within pitch + 5 m margin | per player-frame |
| I3 | `\|velocity_W\| <= 11.0 m/s`, `\|accel\| <= 9.0 m/s²` | per player, after smoothing |
| I4 | `betas` byte-identical for a `track_identity` across the whole match | after outer loop |
| I5 | Bone lengths constant per player across frames (tolerance 1 mm) | derived check on I4 |
| I6 | `ball.position_W.z >= -0.02 m` | per frame |
| I7 | `ball.mode == UNMEASURED` ⟹ excluded from every analytics function | analytics boundary |
| I8 | `provenance == IMPUTED` ⟹ excluded from per-player technical metrics | analytics boundary |
| I9 | Measured + imputed players per team ≤ 11 | per frame |
| I10 | `\|ΔR\|` between consecutive camera frames < 60° | camera track |
| I11 | 95th percentile keypoint reprojection error < 8 px | after stage B |
| I12 | Angular momentum about CoM constant within 15% during a flight segment | airborne segments |
| I13 | No frame is undistorted twice | ingest metadata flag |
| I14 | Every cross-frame transform is applied exactly once | property test, §9.3 |

---

## 9. Test strategy

### 9.1 Levels

| Level | Scope | Runtime budget | Gate |
|---|---|---|---|
| Unit | Pure functions, geometry, domain | < 5 s total | Every commit |
| Property | Frame transforms, invariants | < 30 s | Every commit |
| Golden | Fixed inputs → committed expected outputs | < 2 min | Every commit |
| Integration | Full pipeline, all ports stubbed, CPU | < 10 s | Every commit |
| Regression | WorldPose subset, real models, GPU | < 20 min | Nightly / pre-merge |

### 9.2 Golden tests that must exist before anything else

1. **SMPL frame constant.** Load rest pose with identity `global_orient`; assert head `z` > feet `z` and facing direction along `+X` after `SMPL_TO_WORLD`. This pins §2.1's unverified constant.
2. **Round-trip projection.** For 1000 random points in the pitch volume and a random valid camera: `unproject(project(p_W)) ≈ p_W` to 1e-6 along the ray.
3. **Foot-plane anchoring.** Synthetic camera + known player height; assert recovered `root_W.z` matches ground truth to 1 cm.
4. **Homography sign convention.** A synthetic pan of known magnitude must recover that magnitude with the correct sign. This catches the world→camera vs camera→world inversion, which otherwise produces a mirrored but self-consistent reconstruction.

### 9.3 Property tests

- Applying a transform and its inverse is the identity (catches I14)
- Any permutation of player order yields identical `GameState` up to ordering
- Pitch dimensions scale linearly: doubling `L` doubles all `X_W`
- Imputation never places a player outside the pitch
- Contact hysteresis is monotone: a player that leaves contact cannot re-enter within one frame

### 9.4 Regression gates

Held-out WorldPose clips, stratified by camera height and viewing angle **and by clip boundary** — a chronological split leaks viewpoint between train and test and produces the specific failure documented in the fine-tuning ablation (`[meas.]`: naive chronological split improves Global to 0.425 but degrades Local from 0.065 to 0.079).

Metrics tracked per merge:

| Metric | Gate |
|---|---|
| Global MPJPE | no regression > 2% |
| Local MPJPE | no regression > 2% |
| **Local MPJPE, top decile by joint speed** | no regression > 5% |
| Camera rotation error | < 0.05°/frame |
| Pitch control MAE vs full tracking (Metrica) | < 10 pp |

The speed-stratified row is not optional. A temporal refiner that over-smooths fast actions improves the mean and destroys the tail — this is exactly what happened to the 27-frame refiner in the reference work (0.647 → 0.680 on the challenge despite a 22% local improvement on its own validation, `[meas.]`). Mean-only metrics cannot see it.

---

## 10. Repo layout

```
src/
  core/                      # no I/O, no torch, no cv2
    domain/                  # §3 dataclasses
    ports/                   # §4 Protocols
    pipeline/                # orchestration, pure
    invariants.py            # §8
    analytics/               # pitch control, sprints, orientation
  adapters/
    detection/               # rf_detr.py, yolo26.py, stub.py
    tracking/                # botsort.py, stub.py
    calibration/             # pnlcalib.py, raft_tracker.py, stub.py
    mesh/                    # smplest_x.py, stub.py
    ball/                    # tracknet.py, stub.py
    imputation/              # b2_anchor.py, b4_voting.py
    optim/                   # theseus_graph.py, ceres_camera.py
    sinks/                   # parquet.py, json.py
  config/                    # hydra or pydantic-settings
tests/
  unit/ property/ golden/ integration/ regression/
  fixtures/                  # committed synthetic scenes, small
docs/
  spec/                      # this file, split
  decisions/                 # ADRs, including the rejected-alternatives log
scripts/
  bench_latency.py           # fills in the [est.] numbers in §13.2
```

Rule enforced by a lint check in CI: nothing under `core/` imports `torch`, `cv2`, or anything from `adapters/`.

---

## 11. Milestones

### M0 — Skeleton (no models)

- Domain model, ports, all stub adapters
- Full pipeline runs end-to-end on CPU with stubs in < 10 s
- Golden tests §9.2 items 1–4 pass
- Invariants module wired, all green on stub data

**Accept when:** `pytest` is green and `python -m pipeline run --config stub` produces a valid Parquet file.

### M1 — Static camera contour (the MVP)

- Fixed 4K, one-time calibration
- Detection + tracking + pelvis and ground-projection regression (two-keypoint head, as in the SynLoc winners — **not** bbox centre, see §12.2)
- Ball on ground only, `z ≡ 0`, airborne segments marked `UNMEASURED`
- Positional analytics: distance, speed, sprints, pitch control
- Provenance flags in the schema from day one

**Accept when:** mAP-LocSim > 90 on a held-out static-camera set; every analytics function provably filters on provenance (test, not review).

Reference for what is achievable: SynLoc 2026 winner reached 97.67 mAP-LocSim, 81.91% frames with perfect prediction at 0.48 m tolerance `[meas.]`.

### M2 — Broadcast calibration

- Shot segmentation and frame-type classification
- PnLCalib anchors + RAFT/MAD camera tracking + stage-A bundle adjustment
- TCE watchdog for calibration drift
- Two-edge line fitting (§12.5)

**Accept when:** camera rotation error < 0.05°/frame on WorldPose clips with known GT camera poses; no shot boundary is bridged.

### M3 — Pose

- SMPLest-X fine-tuned on WorldPose: stratified clip split, pelvis-depth supervision, broadcast augmentation
- Foot-plane anchoring
- Beta estimation from close-ups, outer loop
- Factor graph stages A and B

**Accept when:** Global MPJPE < 0.45 m, Local < 0.06 m, speed-stratified Local within 1.5× of mean on held-out WorldPose.

### M4 — Off-screen and ball

- Imputation B2, then B4
- Ball: fine-tuned detector, radius cue, ballistic segments, touch-event constraints
- Own trajectory annotation set (no public football GT exists — §12.8)

**Accept when:** pitch control MAE < 10 pp vs full Metrica tracking; ball mode distribution reported and stable.

### M5 — Coupled solve and airborne

- Factor graph stage C with collision active set
- CoM + angular momentum for airborne segments
- Contact classifier as a joint residual

**Accept when:** invariants I1, I3, I12 green on a manually reviewed set of 50 airborne events.

---

## 12. Known traps

These are the specific things that will be got wrong. Each has been got wrong at least once in the design discussions that produced this document.

**12.1 No shot segmentation.** Broadcast is replays, close-ups and cuts. A pipeline without frame-type classification produces confident garbage on its first real match. Build it in M2, before any 3D.

**12.2 Bbox centre instead of the ground point.** The bbox centre sits at mid-torso, not on the pitch plane. Projecting it through the homography systematically pushes the player away from the camera. Use the bottom of the bbox, or better, regress the pelvis and its ground projection as two coupled keypoints.

**12.3 A lifter without depth output.** Pelvis depth supervision in camera space is the dominant factor for local accuracy — without it local MPJPE stagnates at 0.067 m regardless of augmentation `[meas.]`. A `MeshLifter` adapter that does not expose `pelvis_depth_C` cannot be used.

**12.4 Decoupling z from (x, y).** `z = root_height + depth_refinement` breaks foot-ground contact. Foot-plane anchoring is worth −44 mm global and −12 mm local `[meas.]`.

**12.5 Measuring line thickness.** Perspective changes apparent thickness without any zoom, so thickness as an independent constraint needs the distance to the line — circular. The usable version: IFAB regulates line width (≤ 0.12 m), so the two **edges** of each line are two parallel line correspondences at a known separation. Falls out of the same fitting, better conditioned in close views.

**12.6 Assuming ball trackers transfer.** TOTNet (CVIU 264, 2026) and TrackNetV6 (ICMR 2026) are validated on racket sports and table tennis. Football balls deform on impact, the scene scale is ~20× larger, and occlusion is by 22 legs rather than 2 rackets. Fine-tune and measure before this enters the critical path.

**12.7 Reaching for open-vocabulary detection for kit variation.** Bibs and kit changes are a *team assignment* problem, solved by torso-colour clustering within a match. YOLO-World/CLIP solve a problem you do not have and cost real latency.

**12.8 Assuming public ball trajectory GT exists.** SoccerNet-v3D is 4051 **images** with triangulated 3D ball positions; ISSIA-3D is a two-minute six-camera sequence. Both validate single-frame localisation. There is no public football trajectory GT — budget for annotation.

**12.9 Running physics after the data term.** A simulator or engine that resolves collisions without seeing pixels moves players arbitrarily, destroying the observations. All constraints go into the same functional as reprojection, with weights.

**12.10 Trusting mean MPJPE.** See §9.4. Stratify by joint speed or the fast phases — the ones a coach actually cares about — regress invisibly.

**12.11 Isaac Gym and G-API.** Isaac Gym is deprecated (use Isaac Lab); G-API moved to `opencv_contrib` in OpenCV 5. Both appear in older material and in model output.

**12.12 OpenCV 5's new DNN engine is CPU-only.** GPU support lands in later releases. For NVIDIA, either force the classic engine or build with ONNX Runtime and NVIDIA execution providers. Do not plan a GPU inference path around it.

---

## 13. Environment and infrastructure

### 13.1 Stack

- Python core, hexagonal / ports-and-adapters, no framework in `core/`
- PyTorch for models; Theseus for the factor graph; Ceres or g2o for camera-only stage A
- OpenCV 5 for geometry: USAC is now the default backend for homography, E/F, and PnP, with MAGSAC++, LO-RANSAC, GC-RANSAC, PROSAC and SPRT early termination. On grass contaminated by player motion this matters more than the feature detector. `calib3d` is split into `geometry` / `calib` / `stereo` / `ptcloud`
- Cloud GPU (RunPod / vast.ai) for training and the post-match contour; workstation for development

### 13.2 Latency budget

**Measured:** RAFT-small camera tracking 55 ms/frame; RAFT-large 110 ms/frame with no accuracy gain. ViT-S is ~4× faster than ViT-H `[claimed]`, absolutes not published.

**Estimated `[est.]`, arithmetic shown so it can be checked:** ViT-H/14 at 512×384 ≈ 1000 tokens, ~1.3 TFLOP per crop. An A10G at a realistic 40% of peak ≈ 50 TFLOPS → ~25 ms per crop unbatched, 10–15 ms batched. 14 players × 25 fps = 350 crops/s → ~4–5 GPU-seconds per second of video, i.e. **~5 A10G for real time at 25 fps, ~9 at 50 fps**.

This is why 3D is not in the live contour. It is also why RAFT's 55 ms/frame is not a live bottleneck: **RAFT is not in the live path.** Live uses per-frame PnLCalib homography; RAFT plus bundle adjustment is the offline accuracy path.

| Contour | Contents | Latency | Hardware |
|---|---|---|---|
| Live | 2D detection, tracking, per-frame homography, minimap, speeds, distances, sprints, pitch control, imputation | < 1 s | Edge, laptop/mini-PC |
| Half-time | 3D poses on selected episodes, batched | minutes | Local GPU |
| Post-match | Full factor graph, physics validation | hours | Cloud |

`scripts/bench_latency.py` exists to replace every `[est.]` in this section with a measured number. Until it has run, none of these figures may enter a customer-facing spec.

### 13.3 ViT-S: supervised, not distilled

Distilling from a teacher with 0.32 m global error sounds bad, but that error is dominated by **camera** (~55% of the composite score `[meas.]`), not by the lifter. The lifter's own error is 0.054 m local. For local pose the teacher is good.

Better than distillation: WorldPose ground truth exists. Train ViT-S directly with the same recipe. Distillation only buys something on *unlabelled* data — your own matches. So the right scheme is semi-supervised: WorldPose GT plus pseudo-labels on own footage.

Specialise on **capture conditions** (stadium, camera, lighting), not on team — squads change, conditions do not.

---

## 14. Open questions

Track these in `docs/decisions/`. None blocks M0–M2.

1. **SMPL CoM vs force plate.** No published validation found. AddBiomechanics is the right source. Needed before §7.4's angular-momentum residual can be weighted with confidence.
2. **Ball-strike audio detection in stadium noise.** No precision/recall figures found for non-laboratory conditions. Blocks the broadcast audio idea (already rejected for a different reason — no multichannel feed) but not the training contour, where TDOA multilateration across four synchronised pitch-corner microphones gives direct 3D impact position including height. This is the cheapest available fix for the weakest node in the pipeline and should be prototyped in the M1 timeframe.
3. **Ball tracker transfer from racket sports.** §12.6. Measure before committing.
4. **Turf restitution.** Coefficient ~0.6–0.8, varies with wetness, with spin–velocity coupling at bounce. The error that matters is not amplitude but the **switching times** of the piecewise-ballistic model. Needs per-venue, per-condition calibration from episodes with observed bounces.
5. **Temporal refiner gate.** Hypothesis: the 27-frame window (540 ms at 50 fps) is 2–3× the duration of a kick's swing phase (150–250 ms), so an unconditioned model regresses to the window mean. Test 9–13 frame windows, speed-percentile resampling, asymmetric peak-underestimation loss, and a motion-magnitude gate. Refiner enters the critical path only when it beats Gaussian smoothing on the speed-stratified top decile.
