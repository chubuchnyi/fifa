# M1 — Status & Forward Plan

*Snapshot: 2026-06-24. Companion to `roadmap.md` (which is the step-by-step M1 build log);
this doc is the higher-level "where are we / what's blocking / what's next" view.*

## TL;DR

The full pipeline runs end-to-end on fake adapters (`tests/e2e/test_dry_run.py`) and three
of five perception stages are real: **detector (RF-DETR)** and **tracker (ByteTrack)** are
fully wired built-ins, and the **calibrator** is live via the injected PnLCalib backend — now an
in-repo module (`pitch3d.adapters.models.pnlcalib_backend:make`) and **measured on SoccerNet
`calibration-2023` test** (200→400 frames: completeness 0.745, median reprojection 1.79 px / 0.236 m,
line_acc@5px 0.618; RANSAC + confidence-weighted DLT; a 2026-06-24 kp-threshold sweep showed the
heatmap gate trades completeness for accuracy at ~par — net line-acc stays ~0.61, so the real lever
is better landmarks). The two remaining live backends — **pose (GVHMR)** and **ball (TrackNet)** —
have real, unit-tested *pure* halves and now in-core, torch-free heavy adapters behind the
ADR-0006 dotted-path seam (SMPLest-X for pose, WASB for ball). Both are now **pod-verified on
CUDA** (2026-06-23): the ball adapter (`wasb_backend.py`, staged via `scripts/stage_wasb_weight.sh`)
runs standalone (`scripts/smoke_wasb_gpu.py`) and inside the full pipeline
(`scripts/pod_real_e2e.sh`) end to end — detect→track→calibrate→pose→ball→assemble→export green.

The calibration-data blocker is **resolved**: SoccerNet `calibration-2023` (open, no NDA) supplied
independent landscape-broadcast frames + pitch-line GT, and **B1 is now measured** (see B1 below).
The remaining external blocker is **WorldPose video** (FIFA-licence-gated), needed for the *pose*
bake-off (B2); everything else on the near-term path is autonomous, pure-core work.

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
| **Calibrator** | `--calibrator keypoints` | ✅ DLT + RANSAC + confidence-weighted solve (core) | ✅ injected PnLCalib HRNet (in-repo `pnlcalib_backend:make`) | SoccerNet test: completeness 0.745, median 1.79 px / 0.236 m; messi 3.87 → 0.36 m; `--calibrator-backend` ✓ |
| **Pose** | `--pose gvhmr` | ✅ root-grounding + constraint refit (core) | ✅ SMPLest-X-H (0.69B) **pod-verified on CUDA** — smoke + real-frame seam + full golden-path export (re-run 2026-06-24 on Blackwell via `scripts/pod.sh`) | `--pose-backend pitch3d.adapters.models.smplestx_backend:make`; runtime/wiring proven on real broadcast pixels (1920×1080 clip → 6 subjects → `subject_*.npz`) — **first MPJPE MEASURED 2026-06-24**: 3DPW `test`, off-the-shelf, condition A (GT camera), no-PA → **Local MPJPE ≈ 0.51 m** (mean of 3 single-subject seqs; see B2) |
| **Ball** | `--ball tracknet` | ✅ threshold + gap-fill (core) | ✅ WASB adapter **pod-verified on CUDA** (smoke + full E2E, 2026-06-23) | `--ball-backend pitch3d.adapters.models.wasb_backend:make`; stage via `scripts/stage_wasb_weight.sh`; smoke `scripts/smoke_wasb_gpu.py` |
| Render | `--render overlay` | ✅ reprojection PNGs (dependency-free) | n/a | real, no GPU |
| Export | `--export gltf` | ✅ SMPL-X npz + JSON round-trip | (glTF binary TODO) | |
| Observer | `--observer blender` | — | ⬜ needs Blender + display | see **B4** |

## Blockers / Bugs / Open tickets

| ID | Type | Item | Autonomous? | Notes |
|---|---|---|---|---|
| **B1** | Data-unblocked 2026-06-23; harness built (CPU) | Evaluate `FieldCalibrator` (PnLCalib) on *independent* real broadcast frames with GT pitch calibration | 🟢 MEASURED 2026-06-23 (200) + swept 2026-06-24 (400) | **Correction:** B1 was never truly asset-blocked. SoccerNet `calibration-2023` (real broadcast frames + per-image pitch-line GT) is **openly downloadable, NO NDA password** — only the broadcast *video* is NDA-gated. Built + unit-tested the full CPU half: re-derived pitch template, GT parser, reprojection metrics, CLI (`scripts/run_calib_eval.py`), open downloader (`scripts/get_soccernet_calibration.py`). Synthetic oracle reproj_rms ≈ 1.9e-14 m, perturb grows monotonically (6/6 tests, ruff+mypy clean). **Measured 2026-06-23 (GPU):** glue committed + pulled (730f23a), CLI run on SoccerNet `test` (first 200/3143 frames): **completeness 0.745**, **median reproj 1.79 px / 0.236 m**, **line_acc@5px 0.618 / @10px 0.691** — RMS/p95 (131.8 m / 83.9 m) are outlier-inflated by the ~25 % uncalibrated frames, so median is the real stat. PnLCalib is sub-2 px where it locks on; **completeness, not planar accuracy, is the limiter.** Honesty (R-6): this is a homography-plane proxy named `line_acc@Npx`, **not** the official SoccerNet Completeness×JaC@5; SoccerNet-trained weights → in-distribution upper bound. **Sweep 2026-06-24 (400 frames):** dialling the kp gate 0.3434→0.10 raises completeness 0.745→0.918 but degrades accuracy-where-locked-on in lock-step (on_completed line_acc@5px 0.748→0.640); net all-frames line_acc@5px stays flat ~0.61 → the gate is a **precision/recall dial, not a free win**, and the real lever is **better landmarks** (PnLCalib's own camera module / line-only fusion). Baseline 0.3434 reproduced 0.745 on 2× the frames. (WorldPose video still gated, but no longer the B1 path.) |
| **B2** | Decision **made** | Pose-model pick = **SMPLest-X + SMART recipe** (SAM 3D Body = alt fallback behind same seam) — **user-signed-off 2026-06-22** | ✅ decided | remaining: empirical bake-off (confirm vs fallback + quantify calibration cost) → [`pose-bakeoff-runbook.md`](pose-bakeoff-runbook.md). Bake-off still needs per-crop **frames**; WorldPose video stalled → verified alternatives (EMDB best for global MPJPE, 3DPW for local; runbook §0a). **Synthetic bake-off harness now built & unit-tested (`pitch3d.eval`, pure/no-box): FK seam + conditions A/B, real SMPL-X FK on CPU, a runnable driver (`scripts/run_bakeoff.py`), and camera-sweep + occlusion masking — oracle scores ~0 from every viewpoint, ready for real frames.** B3 wiring may proceed on this backbone now. **SMART now published (arXiv 2605.31551): Global 0.324 m / Local 0.054 m on WorldPose = baseline-to-beat; the FIFA Skeletal-Light 2026 leaderboard is live with our exact metric (refresh 2026-06-22).** **Box check 2026-06-24:** the SMPLest-X seam itself is *not* the blocker — re-verified E2E on CUDA (smoke + real-frame seam + full golden-path → `subject_*.npz`, Blackwell via `scripts/pod.sh`). **First MPJPE MEASURED 2026-06-24** (3DPW staged on the volume via `scripts/get_3dpw.sh`; needed a `file://` decode fix in `_iter_frames`): off-the-shelf **SMPLest-X-H** on 3DPW `test`, condition **A** (GT camera, no PnLCalib), **no Procrustes/PA**, all 16 canonical joints scored — mean over **3 single-subject** seqs (downstairs/stairs/weeklyMarket, 98 frames, stride 30): **Local MPJPE ≈ 0.51 m** (per-seq 0.500 / 0.520 / 0.513; `global == local` since condition A seats the prediction at the GT root; geometry clean: depth>0 = 1.0, in-frame ≥ 0.82). 3DPW is **condition-A-only** (moving cam, no pitch plane). Honesty (R-6): this is the harness's own root-relative metric on our `SMPLX_TO_CANONICAL` 16-joint set, **not** the official starter-kit evaluator, and 3DPW is out-of-domain (no soccer, close-range handheld). The 0.51 m sits between the ~0.6 m zero/T-pose floor and SMART's published **0.054 m** → **confirms the prior that the SMART *recipe*, not the backbone, buys the jump**; it does *not* overturn the SMPLest-X pick. Next: EMDB (Global-MPJPE) / WorldPose (in-domain) when access clears, + the SMART recipe (depth-FT + foot-anchor + 2-pass smoothing). Repro: `scripts/run_pose_eval.py --dataset 3dpw --pkl …/test/<seq>.pkl --images …/<seq> --joint-model smplx --backend …smplestx_backend:make --stride 30` (pose-bakeoff-runbook §0b). 3DPW © von Marcard et al., **ECCV'18** — cite per its licence. |
| **B3** | **Resolved** (pod-verified 2026-06-23) | Pose + ball live adapters **run on CUDA** (SMPLest-X, WASB) | ✅ done | both import torch-free behind the seam; staged box-local (`scripts/stage_wasb_weight.sh`). WASB pod-verified standalone (`scripts/smoke_wasb_gpu.py`) **and** in the full pipeline (`scripts/pod_real_e2e.sh` — all real backends → export). Needed two compat fixes, now in-tree: `torch.no_grad()` around inference (WASB's postprocessor calls `.numpy()` without `detach`); `_load` forces WASB's `src` ahead of SMPLest-X on `sys.path` to dodge a `utils` collision. NumPy-2 `np.Inf` patch folded into the staging script. |
| **B4** | Blocker (env) | Blender observer can't render real SCENE_3D headless without a display/GPU profile | ❌ needs env | M2 concern |
| **Bug1** | Bug (typing) | `refit_port`/`clip` typed as bare `object` in correction engine + assemble | ✅ | latent: `object` has no `.refit`, so misuse is caught only at runtime. Fix = Protocol types |
| **Bug2** | Bug (lint/type debt) | mypy + ruff debt repo-wide (project gates on neither) | ✅ partial | mypy 33→15 via safe fixes (incl. the 6 real None-handling latent bugs, now fixed + tested); rest documented below |
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
- **Bake-off harness (pure):** synthetic broadcast-soccer GT + MPJPE metrics + the FK seam
  (`JointModel`) + a backend-driven A/B pass built & tested in `pitch3d.eval` (numpy-only, no
  box/asset; unit suite now 250 passed + 7 env-gated skips). GT is generated *through* the FK seam
  over the same `HMRBackend` contract the product uses, so an oracle backend scores ~0 and a
  zero-pose backend gives the Local-MPJPE floor. Grown since the snapshot, all still pure/no-box:
  **real SMPL-X FK on CPU** (`SmplxJointModel` — oracle round-trips at machine precision, 4e-16);
  a **runnable driver** (`scripts/run_bakeoff.py` — candidate × condition MPJPE table);
  **condition B** (foot-point grounding via the GT-homography stand-in, so the A→B gap is the
  grounding floor, Global-only — ~0.05–0.08 m placeholder FK, ~0.15 m SMPL-X); and **camera-sweep
  + occlusion masking** (`CAMERA_VIEWS` presets + a mesh-free inter-person visibility proxy +
  opt-in `visible_only`) that hardens condition-A placement — the oracle still scores ~0 from
  every viewpoint. Only condition B's *PnLCalib* and the real pose nets remain box-gated. ✅ done.

### Phase B — Calibration data & honest evaluation (box only — data unblocked)
- **B1 data unblocked 2026-06-23.** SoccerNet `calibration-2023` (real landscape-16:9 broadcast
  frames + per-image pitch-line GT) is **openly downloadable, no NDA** — fetch with
  `scripts/get_soccernet_calibration.py` (only the broadcast *video* needs the NDA password).
- **CPU harness built & unit-tested** (`pitch3d.eval.datasets_soccernet` + `calib_metrics`, CLI
  `scripts/run_calib_eval.py`): re-derived pitch template, normalized-GT parser, point-to-segment
  reprojection metrics, plus a synthetic self-test (oracle ≈ 0, perturb grows). Pure/no-box.
- **MEASURED 2026-06-23 (box, GPU).** Glue committed as `pitch3d.adapters.models.pnlcalib_backend:make`
  (730f23a) and pulled on the pod; ran `scripts/run_calib_eval.py --dataset soccernet` over the
  **first 200 frames of the `test` split** (3143 available) with PnLCalib `SV_kp`/`SV_lines`
  (kp_th 0.3434, line_th 0.7867). Result: **completeness 0.745** (74.5 % of frames calibrated,
  confidence > 0), **median reprojection 1.79 px / 0.236 m** (robust central stat),
  **line_acc@5px 0.618 / @10px 0.691** (correct lines / 1344 total; uncalibrated frames count as
  misses). RMS/p95 (131.8 m / 83.9 m) are **inflated by the ~25 % uncalibrated frames**
  (identity/degenerate H) — median is the meaningful figure. Read: PnLCalib registers the pitch
  **sub-2 px on the ~¾ of broadcast frames where it locks on**; the limiter is **completeness
  (landmark coverage), not planar accuracy**.
- Honesty (R-6): our number is a homography-plane `line_acc@Npx` proxy, **not** the official
  SoccerNet Completeness×JaC@5 (full camera params, distortion, circles, L/R ambiguity); and these
  weights are in-distribution for SoccerNet, so this is an upper-ish bound, not OOD generalization.
- **Done (pure-core, `e6bbd09`):** accuracy-on-completed is now reported separately from
  completeness. `evaluate_calibration` adds an `on_completed` sub-grid + `n_completed` that pool the
  reprojection stats over **only** the confident frames, so the outlier-inflated all-frames RMS no
  longer masks the sub-2 px accuracy where PnLCalib actually locks on. Existing top-level fields are
  unchanged (the measured B1 numbers still reproduce); shared `_pool_summary` keeps both grids identical.
- **MEASURED 2026-06-24 (box, kp-threshold sweep over 400 frames).** Ran the new sweep harness over
  the first **400** `test` frames (2× the B1 baseline), dialling the PnLCalib keypoint heatmap gate
  down from its 0.3434 default. Exact command (recorded for reproducibility):

  ```
  PYTHONPATH=src python scripts/run_calib_eval.py --dataset soccernet \
      --frames-dir /workspace/SoccerNet/calibration-2023/test/test \
      --backend pitch3d.adapters.models.pnlcalib_backend:make --device cuda \
      --limit 400 --threshold-sweep "0.3434,0.25,0.15,0.10"
  ```

  | kp_th  | completeness | n_completed | on_compl median_m | on_compl acc@5px | all acc@5px |
  |--------|--------------|-------------|-------------------|------------------|-------------|
  | 0.3434 | 0.745        | 298         | 0.165             | 0.748            | 0.603       |
  | 0.25   | 0.793        | 317         | 0.173             | 0.724            | 0.614       |
  | 0.15   | 0.888        | 355         | 0.200             | 0.658            | 0.608       |
  | 0.10   | 0.918        | 367         | 0.210             | 0.640            | 0.605       |

  **Honest read (R-6):** the kp gate is a **precision/recall dial, not a free completeness win**.
  Dropping it lifts completeness 0.745 → 0.918, but the recovered frames are the *hard* ones, so
  accuracy-where-locked-on degrades in lock-step (`on_completed` line_acc@5px 0.748 → 0.640, median
  0.165 m → 0.210 m). Net `all_line_acc@5px` stays **flat at ~0.60–0.61** across the whole sweep —
  the extra calibrated frames are too noisy to add correct lines on balance. Baseline 0.3434
  **reproduces the prior B1 number (0.745)** on 2× the frames, confirming the measurement is stable.
- Next (box): the real lever for *better* calibration is **better landmarks, not the gate** —
  evaluate PnLCalib's *own* full camera-calibration module (lines + circles + L/R disambiguation)
  vs our bare planar DLT, and/or a line-only fusion that adds correct lines without the keypoint
  noise. Threshold-tuning is exhausted as a quality lever (the table above).

### Phase C — Pose decision → heavy wiring (research → box)
- **B2** finalize the pose model against WorldPose — bake-off procedure in
  [`pose-bakeoff-runbook.md`](pose-bakeoff-runbook.md) (needs the box **and** WorldPose frames).
  The runbook runs two grounding conditions (GT camera vs our PnLCalib) so the result also tells us
  whether the next effort goes to the pose net or to calibration.
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

- **mypy: 33 → 15** after safe fixes. Fixed: `ReconstructionResult.detections/tracks` were bare
  `object` (now `Detections`/`Tracks`); a `RenderResult` shadowing the `render: str` param in
  `cli.py`; widened a `look_at`/`camera_at` `up` default; explicit RGB 3-tuples in overlay/proxy;
  the 6 `controller.py`/`blender/runner.py` `None`-handling gaps (below); and the `tracking.py`
  comprehension (below).
  Remaining 15, deferred with reason:
  - **~14** — `Correction.payload` is `object` (13 in `correction/engine.py` + 1 `blender/live.py`,
    all `.delta`/keyframe attr access); the engine narrows it by `corr.mode` (which mypy can't
    follow), so a proper fix needs `isinstance`/`cast` at each dispatch site. Invasive on pure,
    well-tested code; low value. Deferred.
  - ~~**3** `controller.py` + **3** `blender/runner.py` — real `None`-handling gaps~~ **FIXED**:
    `preview()` on a ball-less scene now degrades to a no-op (`max_abs_change` 0); a non-ball target
    with no `subject_track_id` raises `ValueError`; `BlenderSceneObserver` routes its overlay/ui/radar
    delegators through a `_delegate` property narrowing the always-set fallback. +3 covering tests.
  - **1** `detection.py` lazy `_model: object` — intentional, to avoid importing torch at module load.
  - ~~**1** `tracking.py` list-comprehension element type~~ **FIXED**: a walrus guard narrows the
    appearance features (already filtered non-None by the `idx` selection above) — behavior-identical.
- **ruff: 48 remaining** (cleared 7 safe auto-fixes this pass — 2 unused imports, 2 unsorted import
  blocks, 3 redundant quoted annotations under `from __future__ import annotations`). The rest are
  29 `E501` long lines + 18 `UP042` (str-enum style) + 1 `B024` — repo-wide style, not churned.

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
