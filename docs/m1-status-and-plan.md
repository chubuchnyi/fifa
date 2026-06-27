# M1 — Status & Forward Plan

*Snapshot: 2026-06-27. Companion to `roadmap.md` (which is the step-by-step M1 build log);
this doc is the higher-level "where are we / what's blocking / what's next" view.*

## TL;DR

**M1 CLOSED 🟢 (2026-06-27).** The editable loop runs end to end on a real clip and all three exit
criteria are MET (AC-1 B1 reprojection median **0.236 m**; AC-2 four-mode propagation + preview +
non-destructive resolve, tested + pod corr-1; AC-3 F-curve edits + export). Suite **480 passed / 5
display-gated skips**. Documented ceiling (R-6): the photoreal-grade pose/ball *heavy* nets ship as
injection-seam stubs (real backends inject by dotted path, ADR-0006); the GUI live-Blender session is
display-gated; metric-XY is measured on lined footage. M2 (measured photoreal) is also 🟢, and **M3
(quality & polish) closed 🟢 (2026-06-27)** — every M3 seam (M3-1…M3-8) plus the **A-10** bounded
autonomy loop is wired end-to-end on fakes (suite **557 passed / 12 skipped**), with the
generative/learned heavy halves kept as honest gated stubs (R-8). Next: **M4** (optional, real
multi-camera) or wiring the gated reals on a GPU box.


The full pipeline runs end-to-end on fake adapters (`tests/e2e/test_dry_run.py`) and three
of five perception stages are real: **detector (RF-DETR)** and **tracker (ByteTrack)** are
fully wired built-ins, and the **calibrator** is live via the injected PnLCalib backend — now an
in-repo module (`pitch3d.adapters.models.pnlcalib_backend:make`) and **measured on SoccerNet
`calibration-2023` test** (200→400 frames: completeness 0.745, median reprojection 1.79 px / 0.236 m,
line_acc@5px 0.618; RANSAC + confidence-weighted DLT; a 2026-06-24 kp-threshold sweep showed the
heatmap gate trades completeness for accuracy at ~par — net line-acc stays ~0.61; a 2026-06-25
landmark-supply diagnostic then ruled out line-only fusion (#122: 0/51 dropped frames have ≥2 lines),
so the real lever is **better landmarks / detector recall**, not the solver or fusion). The two
remaining live backends — **pose (GVHMR)** and **ball (TrackNet)** —
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
| **M1** | Editable vertical slice: real perception backends behind the seams | 🟢 Complete (2026-06-27 — AC-1/2/3 met; pose/ball heavy nets are injection-seam stubs, GUI session display-gated) |
| **M2** | Photoreal render / observer (real Blender SCENE_3D) | 🟢 Complete (measured photoreal via Cycles; broadcast fidelity → M3; see `roadmap.md`) |
| **M3** | Polish, LLM-over-MCP editing north-star (ADR-0008) + broadcast-grade generative fill | 🟢 Complete (2026-06-27 — M3-1…M3-8 + A-10 all wired E2E on fakes; seam B AC-5b, three.js viewer AC-6, suite 557 passed AC-7; generative/learned heavy halves gated R-8; agent track A-1…A-10 ✅; see `roadmap.md`) |
| **M4** | (optional) Real multi-camera — synchronized calibrated sources | ⬜ Not started (optional; the data model already treats cameras as a list, so it's an adapter + orchestration change, not a core rewrite) |

## Perception-backend matrix

"Pure half" = numpy-only, CPU, unit-tested, lives in public core. "Live half" = heavy/GPL
network, injected box-local via `--X-backend pkg.module:Factory` (ADR-0006) so it stays out
of the public repo.

| Stage | CLI | Pure half | Live half | Notes |
|---|---|---|---|---|
| **Detector** | `--detector rfdetr` | n/a | ✅ built-in (RF-DETR, Apache-2.0) | needs `cv` extra + weights + GPU; no injection seam needed (permissive, in-core) |
| **Tracker** | `--tracker bytetrack` | n/a | ✅ built-in (ByteTrack, MIT) + team clustering | `--tracker-backend` threads the `TrackingBackend` injection seam (T3 ✅), symmetric with pose/ball/calibrator |
| **Calibrator** | `--calibrator keypoints` | ✅ DLT + RANSAC + confidence-weighted solve (core) | ✅ injected PnLCalib HRNet (in-repo `pnlcalib_backend:make`) | SoccerNet test: completeness 0.745, median 1.79 px / 0.236 m; messi 3.87 → 0.36 m; `--calibrator-backend` ✓ |
| **Pose** | `--pose gvhmr` | ✅ root-grounding (per-frame foot-plane anchor, T2) + constraint refit (core) | ✅ SMPLest-X-H (0.69B) **pod-verified on CUDA** — smoke + real-frame seam + full golden-path export (re-run 2026-06-24 on Blackwell via `scripts/pod.sh`) | `--pose-backend pitch3d.adapters.models.smplestx_backend:make`; runtime/wiring proven on real broadcast pixels (1920×1080 clip → 6 subjects → `subject_*.npz`) — **first MPJPE MEASURED 2026-06-24**: 3DPW `test`, off-the-shelf, condition A (GT camera), no-PA → **Local MPJPE 0.512 m** (mean of **11** single-subject 3DPW seqs, 767 frames, stride 15; see B2) |
| **Ball** | `--ball tracknet` | ✅ threshold + gap-fill (core) | ✅ WASB adapter **pod-verified on CUDA** (smoke + full E2E, 2026-06-23) | `--ball-backend pitch3d.adapters.models.wasb_backend:make`; stage via `scripts/stage_wasb_weight.sh`; smoke `scripts/smoke_wasb_gpu.py` |
| Render | `--render overlay` | ✅ reprojection PNGs (dependency-free) | n/a | real, no GPU |
| Export | `--export gltf` | ✅ SMPL-X npz + JSON round-trip + glTF/GLB assembly (Z-up→Y-up, unit-tested) | `pygltflib` serialization behind `export` extra | real; pod E2E exports a full scene |
| Observer | `--observer blender` | — | ⬜ needs Blender + display | see **B4** |

## Blockers / Bugs / Open tickets

| ID | Type | Item | Autonomous? | Notes |
|---|---|---|---|---|
| **B1** | Data-unblocked 2026-06-23; harness built (CPU) | Evaluate `FieldCalibrator` (PnLCalib) on *independent* real broadcast frames with GT pitch calibration | 🟢 MEASURED 2026-06-23 (200) + swept 2026-06-24 (400) | **Correction:** B1 was never truly asset-blocked. SoccerNet `calibration-2023` (real broadcast frames + per-image pitch-line GT) is **openly downloadable, NO NDA password** — only the broadcast *video* is NDA-gated. Built + unit-tested the full CPU half: re-derived pitch template, GT parser, reprojection metrics, CLI (`scripts/run_calib_eval.py`), open downloader (`scripts/get_soccernet_calibration.py`). Synthetic oracle reproj_rms ≈ 1.9e-14 m, perturb grows monotonically (6/6 tests, ruff+mypy clean). **Measured 2026-06-23 (GPU):** glue committed + pulled (730f23a), CLI run on SoccerNet `test` (first 200/3143 frames): **completeness 0.745**, **median reproj 1.79 px / 0.236 m**, **line_acc@5px 0.618 / @10px 0.691** — RMS/p95 (131.8 m / 83.9 m) are outlier-inflated by the ~25 % uncalibrated frames, so median is the real stat. PnLCalib is sub-2 px where it locks on; **completeness, not planar accuracy, is the limiter.** Honesty (R-6): this is a homography-plane proxy named `line_acc@Npx`, **not** the official SoccerNet Completeness×JaC@5; SoccerNet-trained weights → in-distribution upper bound. **Sweep 2026-06-24 (400 frames):** dialling the kp gate 0.3434→0.10 raises completeness 0.745→0.918 but degrades accuracy-where-locked-on in lock-step (on_completed line_acc@5px 0.748→0.640); net all-frames line_acc@5px stays flat ~0.61 → the gate is a **precision/recall dial, not a free win**, and the real lever is **better landmarks** (PnLCalib's own camera module / line-only fusion). Baseline 0.3434 reproduced 0.745 on 2× the frames. **#122 line-only fusion DIAGNOSED → won't-fix 2026-06-25:** of the 51/200 frames the DLT can't lock, **0** have ≥2 detected lines (histogram {0:48, 1:3}) — the line head is empty on exactly the frames the kp head drops, so fusion rescues nothing; the ~26 % drop is a detector/data ceiling (`scripts/diag_calib_landmarks.py`). (WorldPose video still gated, but no longer the B1 path.) |
| **B2** | Decision **made** | Pose-model pick = **SMPLest-X + SMART recipe** (SAM 3D Body = alt fallback behind same seam) — **user-signed-off 2026-06-22** | ✅ decided | remaining: empirical bake-off (confirm vs fallback + quantify calibration cost) → [`pose-bakeoff-runbook.md`](pose-bakeoff-runbook.md). Bake-off still needs per-crop **frames**; WorldPose video stalled → verified alternatives (EMDB best for global MPJPE, 3DPW for local; runbook §0a). **Synthetic bake-off harness now built & unit-tested (`pitch3d.eval`, pure/no-box): FK seam + conditions A/B, real SMPL-X FK on CPU, a runnable driver (`scripts/run_bakeoff.py`), and camera-sweep + occlusion masking — oracle scores ~0 from every viewpoint, ready for real frames.** B3 wiring may proceed on this backbone now. **SMART now published (arXiv 2605.31551): Global 0.324 m / Local 0.054 m on WorldPose = baseline-to-beat; the FIFA Skeletal-Light 2026 leaderboard is live with our exact metric (refresh 2026-06-22).** **Box check 2026-06-24:** the SMPLest-X seam itself is *not* the blocker — re-verified E2E on CUDA (smoke + real-frame seam + full golden-path → `subject_*.npz`, Blackwell via `scripts/pod.sh`). **First MPJPE MEASURED 2026-06-24** (3DPW staged on the volume via `scripts/get_3dpw.sh`; needed a `file://` decode fix in `_iter_frames`): off-the-shelf **SMPLest-X-H** on 3DPW `test`, condition **A** (GT camera, no PnLCalib), **no Procrustes/PA**, all 16 canonical joints scored — mean over **3 single-subject** seqs (downstairs/stairs/weeklyMarket, 98 frames, stride 30): **Local MPJPE ≈ 0.51 m** (per-seq 0.500 / 0.520 / 0.513; `global == local` since condition A seats the prediction at the GT root; geometry clean: depth>0 = 1.0, in-frame ≥ 0.82). **Deepened 2026-06-25:** re-ran over **all 11 single-subject** `test` seqs at **stride 15** (767 frames) → **mean Local MPJPE 0.512 m** (frame-weighted 0.515 m; per-seq range 0.442 `flat_guitar_01` … 0.569 `outdoors_fencing_01`), confirming and tightening the 3-seq number — the off-the-shelf backbone is a stable ~0.51 m on 3DPW, well above SMART's 0.054 m (recipe, not backbone). 3DPW is **condition-A-only** (moving cam, no pitch plane). Honesty (R-6): this is the harness's own root-relative metric on our `SMPLX_TO_CANONICAL` 16-joint set, **not** the official starter-kit evaluator, and 3DPW is out-of-domain (no soccer, close-range handheld). The 0.51 m sits between the ~0.6 m zero/T-pose floor and SMART's published **0.054 m** → **confirms the prior that the SMART *recipe*, not the backbone, buys the jump**; it does *not* overturn the SMPLest-X pick. Next: EMDB (Global-MPJPE) / WorldPose (in-domain) when access clears, + the SMART recipe (depth-FT + foot-anchor + 2-pass smoothing). Repro: `scripts/run_pose_eval.py --dataset 3dpw --pkl …/test/<seq>.pkl --images …/<seq> --joint-model smplx --backend …smplestx_backend:make --stride 30` (pose-bakeoff-runbook §0b). 3DPW © von Marcard et al., **ECCV'18** — cite per its licence. |
| **B3** | **Resolved** (pod-verified 2026-06-23; **T2 anchor 2026-06-25**) | Pose + ball live adapters **run on CUDA** (SMPLest-X, WASB) + per-frame foot-plane anchor | ✅ done | both import torch-free behind the seam; staged box-local (`scripts/stage_wasb_weight.sh`). **T2 (2026-06-25):** the pose backend now also emits `pelvis_above_foot` (SMPL-X FK at go=0) so the grounded root Z varies per frame — pod E2E confirms posture-tracking Z (subject 1.03→0.76 m crouch). WASB pod-verified standalone (`scripts/smoke_wasb_gpu.py`) **and** in the full pipeline (`scripts/pod_real_e2e.sh` — all real backends → export). Needed two compat fixes, now in-tree: `torch.no_grad()` around inference (WASB's postprocessor calls `.numpy()` without `detach`); `_load` forces WASB's `src` ahead of SMPLest-X on `sys.path` to dodge a `utils` collision. NumPy-2 `np.Inf` patch folded into the staging script. |
| **B4** | Blocker (env) | Blender observer can't render real SCENE_3D headless without a display/GPU profile | ❌ needs env | M2 concern |
| **Bug1** | Bug (typing) | `refit_port`/`clip` typed as bare `object` in correction engine + assemble | ✅ | latent: `object` has no `.refit`, so misuse is caught only at runtime. Fix = Protocol types |
| **Bug2** | Bug (lint/type debt) | mypy + ruff debt repo-wide (project gates on neither) | ✅ partial | mypy 33→15 via safe fixes (incl. the 6 real None-handling latent bugs, now fixed + tested); rest documented below |
| **T2** | Ticket (quality) | ~~Foot-plane anchoring in `_ground_root` (root Z a fixed 0.92 m constant)~~ | ✅ **done 2026-06-25** | per-frame foot-plane anchor shipped (design-note **option 1**). Backend computes `pelvis_above_foot` from SMPL-X FK at zero global-orient (`smplestx_backend._pelvis_above_foot`); pure half grounds `root_z` with it, falls back to 0.92 m when absent. FK validated locally (rest 0.935 m; crouch monotonically drops). Pod E2E (10f, 6 subjects): root Z now tracks posture — e.g. subject crouch **1.03→0.76 m** vs flat 0.92 m before |
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
- **SoccerNet original-video NDA APPROVED 2026-06-25.** The broadcast *video* bank (NDA-gated,
  distinct from the open `calibration-2023` frames above) is now accessible: download via the
  official `SoccerNet` pip package (`pip install SoccerNet --upgrade`) with
  `SoccerNetDownloader(...).password` — the password lives in `.env` as `SOCCERNET_PASSWORD`
  (gitignored; never committed to the repo or quoted in docs). Unlocks a large in-domain
  broadcast bank for: more/varied eval clips (esp. wide framings with visible pitch lines — the
  calib-confidence limiter is framing, not the backend), a failure-mode eval set (M2-0
  SAM-Body4D brief), and per-crop pose frames for the B2 bake-off (in-domain alternative to the
  stalled WorldPose video). Security (R-6): the approval email wrapped its links through a
  third-party tracker (`yatrack3.com`) — use the official soccer-net.org /
  github.com/SoccerNet channel + the pip package, not those links.
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
- The real lever for *better* calibration is **better landmarks, not the gate** — threshold-tuning is
  exhausted as a quality lever (the table above). Two candidates: (a) PnLCalib's *own* full
  camera-calibration module (lines + circles + L/R disambiguation) vs our bare planar DLT — **done**,
  A/B'd below (camera tightens lines + tail but leaves the median a wash); (b) a line-only fusion that
  adds correct lines without the keypoint noise — **diagnosed → won't-fix** (#122, below: the line
  head is empty on exactly the frames the kp head drops, so there is nothing to fuse). With the solver
  A/B'd and the fusion lever ruled out by data, the remaining lever is **completeness** (recovering the
  ~¼ of frames the detector drops) — now shown to be **detector/data-bound**, not a fusion gap.
  **Camera-module lever WIRED 2026-06-24, A/B measured 2026-06-25.** The same PnLCalib
  backend now implements a second path: `_PnLCalibBackend.calibrate_frames()` runs the full
  `FramebyFrameCalib` (points **and** lines, mode + RANSAC voting, optional PnL line refinement) and
  emits a per-frame image→world homography, converted from the solved `cam_params` in the *same*
  centre-origin metric frame as the DLT path (`image_to_world_from_cam_params`, verified against
  PnLCalib's own `projection_from_cam_params`). A new pure `CameraModuleFieldCalibrator` +
  `HomographyBackend` protocol score/smooth it (sibling to `KeypointFieldCalibrator`/`KeypointBackend`),
  and `run_calib_eval.py --solver dlt|camera` A/Bs them through the one dotted-path seam. Pure half
  is unit-tested with no GPU (conversion round-trip + scoring/last-good-carry); ruff+mypy clean.
  **Measured 2026-06-25 (box A/B, SoccerNet `test`, first 200 frames, same set, kp/line gate
  0.3434/0.7867, smooth_window=1).** Both solvers are driven by the *same* PnLCalib detector, so they
  share completeness — only the solve differs. `on_completed` = the frames that actually solve:

  | metric (on_completed) | DLT (bare planar) | Camera module (pts+lines) |
  |---|---|---|
  | completeness | 0.745 (149/200) | 0.740 (148/200) |
  | reproj median | **0.170 m** / 1.38 px | 0.179 m / 1.45 px |
  | line_acc@5px | 0.757 | **0.811** |
  | line_acc@10px | 0.846 | **0.905** |
  | reproj p95 | 6.41 m | **1.15 m** |

  Verdict (R-6 honest): **median accuracy is a wash** (camera ~1 cm worse, within noise) and
  **completeness is unchanged** (same detector/gate — the solve method cannot recover a frame the
  detector dropped). The camera module's real wins are **line registration** (+5–6 pp at both 5/10 px;
  all-frames line_acc@5px 0.618→0.661) and **tail robustness** — it all but eliminates the
  near-singular blow-ups the bare DLT occasionally emits (on_completed reproj p95 6.41→1.15 m;
  all-frames pixel RMS ~5e14→~1e3 px). So the camera path buys *reliability + better lines*, not a
  lower central error; **completeness, not the solver, stays the limiter.** `PNLCALIB_PNL_REFINE`
  toggles the PnL line refinement (default on).
- **Line-only fusion lever — DIAGNOSED → won't-fix 2026-06-25 (#122).** The camera A/B measured the
  *outcome* (completeness flat at ~0.74); a focused diagnostic (`scripts/diag_calib_landmarks.py`,
  torch-free factory, runs both HRNet heads on the box) measured the *cause*. For each of the same
  first 200 SoccerNet `test` frames it records `n_kp` (keypoints **after** `complete_keypoints`, i.e.
  line-intersections already folded in — the DLT input, needs ≥4) and `n_lines` (detected straight
  lines), then asks: among frames the DLT cannot lock (`n_kp < 4`), how many lines did the line head
  find? Result (box run 2026-06-25, kp/line gate 0.3434/0.7867):

  | bucket | frames | of which line-rich (`n_lines≥2`) | line-poor (`n_lines<2`) |
  |---|---|---|---|
  | locked (`n_kp≥4`) | 149 (0.745) | — | — |
  | **failed (`n_kp<4`)** | **51 (0.255)** | **0 (0.000)** | **51 (1.000)** |

  Failed-frame `n_lines` histogram: **{0 lines: 48, 1 line: 3}** — **not a single** dropped frame
  has ≥2 detected lines. The lock-rate 149/200 = **0.745 reproduces the B1 completeness exactly**, so
  the diagnostic samples the benchmark faithfully. **Verdict (R-6):** when the keypoint head fails,
  the line head fails on the *same* frames (extreme zoom / motion blur / occlusion — GT-annotated but
  visually marking-sparse in the actual pixels). A line-only fusion fallback would rescue **0** of the
  51, which is exactly *why* the camera module (which already uses lines) left completeness flat. The
  ~26 % drop is a **detector/data ceiling**, not a fusion gap — so #122 is closed **won't-fix**. The
  real completeness levers lie elsewhere: a stronger / fine-tuned landmark detector (better recall on
  hard broadcast views), or — for the *video* path, not these independent calibration stills —
  temporal propagation of a lock across neighbouring frames. Unsolved frames are already surfaced as
  zero-confidence drift (R-6), never crashes.

### Phase C — Pose decision → heavy wiring (research → box)
- **B2** finalize the pose model against WorldPose — bake-off procedure in
  [`pose-bakeoff-runbook.md`](pose-bakeoff-runbook.md) (needs the box **and** WorldPose frames).
  The runbook runs two grounding conditions (GT camera vs our PnLCalib) so the result also tells us
  whether the next effort goes to the pose net or to calibration.
- **B3** wire the chosen pose backend + ball TrackNet live on the box (box-local, like PnLCalib).
- **T2** foot-plane anchoring — ✅ **done 2026-06-25** (per-frame `pelvis_above_foot` from SMPL-X FK; see design note).
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

**RESOLVED 2026-06-25 — option 1 shipped.** `RawBodyMotion` now carries an optional per-frame
`pelvis_above_foot`; `GVHMRPoseEstimator._ground_root` uses it as the root Z (foot on plane z=0)
and falls back to the 0.92 m constant when absent — **byte-identical on the fake** (it leaves the
field `None`), so the pure-half port tests are unchanged and three new tests cover the supplied-
height, row-alignment, and length-validation paths. The heavy half (`SMPLestXBackend`) fills it via
`_pelvis_above_foot`: a quick SMPL-X forward pass at **zero global orient** (articulation kept,
world tilt dropped), measuring the native-+Y drop from the pelvis to the lowest foot/ankle joint.
Validation: (a) **local, no GPU** — rest pose → 0.935 m (≈ the old nominal), increasing crouch
drops it monotonically 0.913→0.838→0.720 m, ±2σ stature shifts it 0.82↔1.05 m; (b) **pod E2E**
(10 frames, 6 subjects, full real path) — the exported root Z now tracks posture (a crouching
subject 1.03→0.76 m; upright subjects sit tight at ~0.90–0.93 m) instead of a flat 0.92 m.
R-6 caveat (documented in code): zeroing global orient assumes body-up ≈ world-up, so a markedly
leaning torso biases the height, and the foot proxy is the ankle/toe joint (~ankle-height above the
sole) — good enough for the per-frame *variation* that was the point; absolute sub-cm grounding
would need the richer-calibration path (option 2).

## Lint/type debt (Bug2) — measured 2026-06-22

The project gates on neither mypy nor ruff, so debt has accumulated. Measured state:

- **mypy: 33 → 15** after safe fixes (M0/M1); **→ 20 today (2026-06-27)** — the M2-7..M2-10 render
  adapters regressed it by 5 (a new `int | None`/tuple cluster, see the render-cluster bullet below);
  the original 15 are unchanged. Fixed: `ReconstructionResult.detections/tracks` were bare
  `object` (now `Detections`/`Tracks`); a `RenderResult` shadowing the `render: str` param in
  `cli.py`; widened a `look_at`/`camera_at` `up` default; explicit RGB 3-tuples in overlay/proxy;
  the 6 `controller.py`/`blender/runner.py` `None`-handling gaps (below); and the `tracking.py`
  comprehension (below).
  Remaining 20, deferred with reason:
  - **~14** — `Correction.payload` is `object` (13 in `correction/engine.py` + 1 `blender/live.py`,
    all `.delta`/keyframe attr access); the engine narrows it by `corr.mode` (which mypy can't
    follow), so a proper fix needs `isinstance`/`cast` at each dispatch site. Invasive on pure,
    well-tested code; low value. Deferred.
  - ~~**3** `controller.py` + **3** `blender/runner.py` — real `None`-handling gaps~~ **FIXED**:
    `preview()` on a ball-less scene now degrades to a no-op (`max_abs_change` 0); a non-ball target
    with no `subject_track_id` raises `ValueError`; `BlenderSceneObserver` routes its overlay/ui/radar
    delegators through a `_delegate` property narrowing the always-set fallback. +3 covering tests.
  - **1** `detection.py` lazy `_model: object` — intentional, to avoid importing torch at module load.
  - **+5 (M2-7..M2-10 render cluster, new 2026-06-27)** — `int | None`/tuple arg-type guards on the
    photoreal render + avatar adapters: `render/avatar_splat.py` (3: lines 162/201/251 — frame index
    and an RGB tuple), `render/cycles.py` (1: line 189 — frame index), `models/avatar.py` (1: line
    185 — a numpy `signedinteger` assignment). Same low-value/invasive trade-off as the rest:
    `frame` is `int | None` on the port but always set on these paths; fixing means an `assert`/`cast`
    per site. Deferred; fix opportunistically when next editing these files.
  - ~~**1** `tracking.py` list-comprehension element type~~ **FIXED**: a walrus guard narrows the
    appearance features (already filtered non-None by the `idx` selection above) — behavior-identical.
- **ruff: 48 → 46 remaining today (2026-06-27, src+tests)** (cleared 7 safe auto-fixes the original
  pass — 2 unused imports, 2 unsorted import blocks, 3 redundant quoted annotations under
  `from __future__ import annotations`). The rest are 26 `E501` long lines + 18 `UP042` (str-enum
  style) + 1 `B024` + 1 `I001` (one auto-fixable unsorted-import block) — repo-wide style, not churned.

Stance: fix debt opportunistically in files we're already editing for real reasons; don't do a
repo-wide lint sweep as a standalone change (high churn, zero functional value, ungated).

### Phase D — Polish & observer
- **Continuity + temporal coherence ✅ done & pod-validated 2026-06-24** (the "players appear
  from nowhere" fix). Three separable, honest pieces, each at its correct seam:
  - **Continuity stitching is STRUCTURAL, before POSE** (`core/orchestration/continuity.py`,
    CLI `--stitch`): re-links fragmented tracklets so an occluded player keeps one identity
    instead of being re-detected as a new subject. Runs on the tracker output, not as a correction.
  - **Gap-fill is STRUCTURAL densification** (`core/correction/coherence.py:fill_pose_gaps`): only
    interior gaps `1 ≤ missing ≤ max_fill_gap` (=12) are bridged — *slerp* for every rotation,
    *lerp* for translation; measured rows are copied verbatim; longer occlusions are left as true
    gaps. It can't be a `Correction` because the engine never inserts frames. Bridged frames carry
    a low `subject_frame_conf` (0.3) so the attention list flags them **inferred, not measured** (R-6).
  - **Smoothing is a CORRECTION** (`coherence_corrections` → `TEMPORAL_SMOOTHING`, CLI `--coherence`):
    a normal zero-phase, inspectable, disableable correction layered over the (now dense) proposal —
    never baked in (ADR-0002).
  - **Pod validation (real models, real broadcast clip).** Ran the full real golden path
    (RF-DETR → ByteTrack → SMPLest-X-H → WASB) with both flags on a Colombia broadcast clip:
    ```
    cd /workspace/fifa && STITCH=1 COHERENCE=1 PITCH3D_CLIP=/workspace/colombia.mp4 \
      FRAMES=48 OUT=out/colombia_coh FORMAT=smplx_npz bash scripts/pod_real_e2e.sh
    ```
    → `ingested 1920×1080 @ 29.970 fps, 48 frame(s)` → `20 subject(s), ball=yes`;
    **continuity: 24→20 tracklets (2 merges, 2 blips dropped)** — real ByteTrack fragmentation
    re-linked; **coherence: bridged 23 gap frame(s) across 2/20 subjects, +20 auto-smoothing
    correction(s)** — real occlusion gaps filled; `done in 227s`, `scene.smplx_npz` exported
    (subject ids non-contiguous 1–18, 20, 31 — the fragmentation this fixes). Honesty (R-6): this
    proves the features **run end-to-end and do genuine non-trivial work on real fragmentation /
    occlusion** — it is a wiring/runtime proof, **not** a quantitative quality metric (no GT for
    these clips). Unit suite stays green (339 passed / 7 env-gated skips), no net-new lint/type debt.
  - Pod hygiene: a pre-validation `git stash@{0}` ("pod-local stale dupes (pre-coherence-validation)")
    holds the box-local edits that were already redundant with `origin/main` (the `file://` decode
    fix + cli lineup, both upstream); preserved (not dropped) so nothing is lost. Pod stopped after.
  - **#98 entry/exit fade — overlay half ✅ done** (local, pure numpy/stdlib). The reprojection
    overlay now ramps a marker's opacity toward the pitch background over `fade_frames` (default 4,
    `=0` disables) at *genuine* entries/exits **only**: a subject appearing after the clip start,
    leaving before its end, or either side of an interior gap too long for coherence to bridge —
    so a real substitution reads as an entrance, not a pop. A subject merely clipped by the window
    edge is **not** faded (no invented entrance/exit, R-6), and a subject present across the whole
    clip renders **byte-identical** with fade on/off (back-compat). Pure `appearance_alpha` (segment
    detection + ramp) and `fade_to_background` (blend) are unit-tested, plus a render-level test that
    the entry frame dims while a settled frame is untouched (overlay suite 18→27). On-by-default
    through the composition root (`wiring.py`), so `--render overlay` already fades.
  - **#100 entry/exit fade — mesh half ✅ done** (local Blender 5.1.2 validation, no pod needed for
    the render path). `anim_export.py` bakes the same `appearance_alpha` per-frame vector into each
    `anim_subject_*.npz` (`PITCH3D_FADE_FRAMES`, default 4), and `blender_animate.py` drives it into
    the Cycles **Principled BSDF `Alpha`** input per frame — each frame renders independently with
    `write_still`, so no material keyframes are needed. Validated locally: MSE-vs-opaque-reference
    decreases monotonically as alpha rises 0.25→1.0, and an alpha=1.0 frame is byte-identical to the
    opaque reference (full-clip bodies unchanged, back-compat). `demo_video.sh` now defaults
    `STITCH=1 COHERENCE=1` and forwards `PITCH3D_FADE_FRAMES` end-to-end (→ `pod_make_video.sh` →
    `pod_real_e2e.sh` / `anim_export.py`), so the pod video inherits continuity + gap-fill + mesh
    fade.
  - **#101 fresh pod video ✅ done & pod-validated 2026-06-24** (real golden path, mesh-fade live).
    One command — `OUT_LOCAL=out/anim_coh bash scripts/demo_video.sh --clip
    samples/video/Colombia-1-0-Congo-DR1080p.mp4 --frames 48` (STITCH/COHERENCE/fade now default) —
    brought up the pod, reconstructed, animated, rendered 4 angles, pulled the mp4s, and **stopped the
    pod**. Reconstruction: `ingested 1920×1080 @ 29.970fps, 48 frame(s)` → `20 subject(s), ball=yes`;
    **continuity 24→20 tracklets** (2 merges, 2 blips), **coherence bridged 23 gap frame(s) across
    2/20 subjects, +20 smoothing** (matches the prior validation), `done in 258s`. **Mesh fade is
    genuinely exercised:** 6/20 subjects are partial-range (frames 33/40/38/43/38 and a B-team
    substitute at **11/48**), and the rendered body count falls **20→16** by the last frame, so those
    exits ramp out via the baked `alpha` instead of popping; the 14 full-clip bodies stay opaque.
    Render: `BLENDER_ANIM_OK frames=48 cams=[broadcast,sideline,top,goal] 1280×720 32spp` on **OptiX /
    RTX PRO 4500 Blackwell** (low `nvidia-smi` GPU% is a sampling artifact — sub-second bursts between
    CPU-bound per-frame BVH re-sync; GPU memory ~1.8 GB confirms the device is live). Output: 4 mp4s
    under `out/anim_coh/video/` (48f @ 25fps, verified). **Honesty (R-6):** `pose=SMPLest-X-H` ran for
    real (checkpoint loaded, 687 M params / 0.69B); detect=RF-DETR, track=ByteTrack real; **calibration
    was the PROXY** (`PNLCALIB_REPO` unset), so the resolved ball (48f) sits at low height-confidence
    (~0.25) on the proxy plane — this is a runtime/feature proof of the fade on real fragmentation, not
    a calibrated-accuracy result. Two extracted broadcast frames render cleanly (bodies + motion +
    pitch, correct angle); a frame-by-frame eyeball of the opacity ramp was not done.
  - **#102 no-evaporation: edge extension ✅ done & pod-validated 2026-06-24.** The #101 run
    exposed the real issue behind the "20→16 by the last frame" drop: a
    player the tracker acquires late or loses early is **still physically on the pitch**, so fading
    or blinking it out is *less* honest than reconstructing it. New STRUCTURAL pass in
    `core/correction/coherence.py:extend_pose_to_span` extends every subject to the full clip span
    (union of all subject + ball frames, the same range `anim_export.py`/`blender_animate.py`
    iterate): **posture is held** (rotations clamp to the nearest measured pose — "standing stays
    standing") and **root translation coasts** with a geometric decaying edge velocity
    (`extrapolate_decay=0.9` → a runner keeps running, then eases to a bounded stop; a single-frame
    track holds position). Interior gaps are now bridged at **any** length when extending (both
    endpoints are real). R-6: extrapolated edge frames carry an even lower `subject_frame_conf`
    (`extrapolated_confidence=0.2` < interior `filled_confidence=0.3`); the fade machinery stays for
    genuine future exits but no longer fires on tracker loss, since every subject now spans the clip
    (so `appearance_alpha` sees no entries/exits). All on by default (`extend_to_span=True`),
    disableable. `CoherenceReport` now also reports `extended_frames`/`subjects_extended` (surfaced
    in the CLI `== coherence:` line). 11 new unit tests (coast-with-decay both edges, standing/
    single-frame hold, posture held, no-op when already full-span, non-mutation, optional hand/jaw,
    multi-subject span + low-conf flagging, disable path); full suite **357 passed**, ruff + mypy
    clean. **Pod-validated 2026-06-24** on the real clip (`demo_video.sh --frames 48`): coherence
    reported `extended 85 edge frame(s) across 6/20 subjects` (exactly the 6 partial-range tracks,
    incl. the 11/48 B-team sub now at 48/48), **all 20 subjects span frames=48**, and Blender's
    per-frame body count is **a flat 20 on every one of the 48 frames** (`global=0 … global=47`,
    20 bodies) — the prior 20→16 collapse is gone, nobody evaporates. 4 mp4s under
    `out/anim_extend/video/`, pod stopped on exit.
  - **#102b pod-render hygiene ✅ done & pod-validated 2026-06-24** (commit `d81b2a2`). The first
    #102 run surfaced a *separate* contamination bug: the pod's `out/anim/mesh` lives on the
    persistent volume and is reused across runs, so two **stale** `anim_subject_22/25.npz` from an
    earlier run lingered, got globbed by `blender_animate.py`, and rendered **phantom bodies** (21
    on 4 frames of a 20-subject scene). Fixed at the producers: `anim_export.py` purges
    `anim_subject_*.npz` + `ball.npz` before writing, `blender_animate.py` clears each camera's
    `frame_*.png` before rendering (same hazard for ffmpeg's glob when frame counts differ). A
    `--reuse-scene` re-render confirmed the purge: no `subject_22/25`, **20 bodies on all 48
    frames**. Honesty (R-6): both #102 runs reused the #101-class reconstruction, so the
    `calib=PROXY` / low-height-confidence-ball caveats carry over — this validates the *population
    stability* feature, not calibrated accuracy.
  - **#103 PnLCalib wired into the video path ✅ done & pod-validated 2026-06-24.** The `calib=PROXY`
    caveat that ran through #101/#102/#102b is now resolved: real field calibration is reachable in
    the video path via a new opt-in **`--real-calib`** flag on `scripts/demo_video.sh` that points the
    calib seam at the pod's staged `/workspace/repos/PnLCalib` + `/workspace/weights/pnlcalib/
    SV_{kp,lines}` unless `.env` already set them (a *committed* flag, because `.env` is git-ignored, so
    editing it would not be reproducible). **Preflight (box):** PnLCalib loads **in-process inside the
    pipeline venv `/workspace/.venv`** (torch **2.8.0+cu128**) — both HRNet heads build on CUDA,
    `torch.load` of the 265 MB `SV_kp`/`SV_lines` works under the 2.8 `weights_only=True` default (the
    checkpoints are pure `OrderedDict`s), and `shapely` is already installed → **no subprocess bridge
    and no `weights_only` patch needed.** (Weights actually live at `/workspace/weights/pnlcalib/`, not
    `…/repos/PnLCalib/weights/`; the `SV_WP_*` files there are 0-byte stubs.) **Clip framing, not the
    backend, is the limiter:** `pod_real_e2e.sh`'s default `clip.mp4` (tight framing on plain grass, no
    visible lines) yields **0 landmarks/frame** → identity H / confidence 0 (an honest R-6 fallback,
    no crash); the `demo_video.sh` default **Colombia 1-0 Congo** broadcast (wide, penalty box + arc +
    goal in view) yields **10–11 landmarks/frame**. **Validated** on an 8-frame Colombia reconstruction:
    `== calibration: REAL PnLCalib`, **field calibration confidence 0.61** (per-frame 0.55–0.64),
    **non-identity homography**, ball **height_confidence 0.42** (vs ~0.25–0.33 under the proxy). A full
    48-frame `demo_video.sh --real-calib` run confirmed the same in the production path
    (`== calibration: REAL PnLCalib` → `reconstructed scene-1: 20 subject(s), ball=yes`), rendered all
    4 cameras (`BLENDER_ANIM_OK frames=48 cams=[broadcast,sideline,top,goal] 1280x720 32spp`), and
    pulled **4 mp4s → `out/anim/video/`** before stopping the pod on exit
    (`✓ pod stopped — billing back to volume-only`). **Render stage is CPU-bound, not GPU:** with
    `--device gpu` the OptiX path is genuinely active (acceleration structures + denoising kernels +
    on-device sampling all in the log), but the scene is tiny (≈20 low-poly bodies + ball), so each
    render call is ~4.3 s of which only ~1.0 s is GPU sampling (32 spp @ 1280×720) — the other ~3.2 s
    is host-side scene re-sync (BVH/OptiX-AS rebuild + mesh upload, paid per camera because
    `blender_animate.py` dirties geometry every frame via `foreach_set`+`update`), plus a one-time
    ~6 min OptiX kernel compile on frame 0. Net: `nvidia-smi` reads GPU≈0 / CPU-pegged during render —
    the GPU's real value is in *perception/reconstruction* (calib + pose/ball nets), so the render
    stage could later be split onto a cheaper CPU box. **Honesty
    (R-6):** 0.61 is the calibrator's own
    inlier×inlier-fraction score (in-sample), **not** independent GT accuracy, and real calibration
    only fires on landscape-broadcast clips with visible markings.
  - **Full real-model E2E (post-M2-6) ✅ pod-validated 2026-06-25** (commit `004f305`). First run of
    the *whole* golden path with **every perception backend real and wired together** —
    `scripts/pod_real_e2e.sh` with `PNLCALIB_REPO=…/PnLCalib STITCH=1 COHERENCE=1 FRAMES=8` on the
    Blackwell pod (RTX PRO 4500, cu128 / torch 2.8). Log confirms `device: cuda` and **real adapters:
    detect=rfdetr, track=bytetrack, calibrate=keypoints (PnLCalib), pose=gvhmr (SMPLest-X),
    ball=tracknet (WASB), render=overlay, export=gltf**; only `env, avatar, observe` are fakes (the M2
    photoreal layer, honestly gated). **Pose is genuinely real:** SMPLest-X ViT-H checkpoint
    `smplest_x_h.pth.tar` loaded (`Total #parameters: 687223152 (0.69B)`) → `reconstructed scene-1:
    6 subject(s), ball=yes`, each 8f × 21 joints. `.npz` verification (`subject_1.npz`): `body_pose`
    (8,21,3) **per-frame std 0.053** (real articulation *moving* across frames, not a static fallback),
    non-zero `betas` (10,), varying `global_orient`, `body_model=SMPL-X`. **T2 foot-plane anchor fires
    end-to-end:** exported `transl` z-column varies **0.81–1.03 m** (pelvis height tracking
    crouch/stride), not the fixed nominal — the `pelvis_above_foot` path (commit `25ad2d3`) is live
    through real SMPLest-X FK. **Calib = identity/0.00 on `clip.mp4` is the known framing limit, not a
    regression:** same as #103 — tight grass framing → 0 landmarks → identity H, `field calibration
    confidence mean=0.00`, so world XY is pixel-passthrough (`transl` XY up to ~1440, off-pitch);
    articulation + pelvis-Z are unaffected (independent of H). For metric XY use a landscape-broadcast
    clip (Colombia → 0.61). **Ball:** WASB ran, `height_confidence mean=0.33`, **4 honest
    `low_ball_height` attention items** (frames 1,2,6,7 @ 0.00 — monocular depth, R-6 *marked* not
    hidden). **Continuity/coherence active:** `--stitch` 6→6 (no fragmentation on 8 clean frames),
    `--coherence` bridged 0 gaps + 6 auto-smoothing; edit→resolve committed `corr-1` (constant_offset,
    max_abs_change 0.1 m) → 7 corrections. **Seam A (M2-5) live:** orbit re-shoot `overlap=0.85
    editable=False`, cached+deduped (1 synth_view after 2 calls). **Export:** 6× `subject_N.npz`
    (6096 B) under `out/run/export/scene.smplx_npz/`; note `--export gltf` + `--format smplx_npz`
    writes **npz bodies, no standalone `.gltf` mesh** (FORMAT governs serialization — set FORMAT=gltf/
    json for a mesh/ball). **Timing:** **431 s** wall for 8 frames cold, dominated by model loads
    (RF-DETR + SMPLest-X 0.69B + WASB + PnLCalib). Pod **stopped on completion** (`zueopp6nzozxb7
    EXITED`, spend → $0.012/hr volume-only). **Honesty (R-6):** this validates the real single-cam →
    world-SMPL-X *articulation* path end-to-end on the real nets; metric world placement still needs a
    lined clip, and the photoreal avatar/env stay honest fakes (M2 gated).
  - **Full GPU E2E → multi-angle video (post-M2-10) ✅ pod-validated 2026-06-27.** First video run
    after pushing M2-10 (HEAD `74df699`): `OUT_LOCAL=out/anim_m210 bash scripts/demo_video.sh --clip
    samples/video/Colombia-1-0-Congo-DR1080p.mp4 --frames 48 --real-calib`. Pod `zueopp6nzozxb7`
    (RTX PRO 4500 Blackwell) resumed, ran **~21.5 min** (06:27→06:48 UTC ≈ **$0.27**), **auto-stopped**
    (`EXITED`). Pipeline E2E: `ingested 1920×1080 @ 29.970fps, 48 frame(s)` → `reconstructed scene-1:
    20 subject(s), ball=yes` → continuity `24→20 tracklets` → coherence `bridged 23 gap(s) / extended
    85 edge frame(s) across 6/20 / +20 smoothing` → `corr-1` committed → seam-A orbit `overlap=0.85
    editable=False` → export json; avatar/env are the honest M2 fakes. **Render genuinely on GPU:**
    `device: cuda` + `BLENDER_ANIM_GPU OPTIX` → `BLENDER_ANIM_OK frames=48
    cams=[broadcast,sideline,top,goal] 1280×720 32spp fps=25`, the one-time ~6 min OptiX kernel
    compile on frame 1, all `frame_0047` saved per camera → **4 mp4 pulled → `out/anim_m210/video/`**
    (broadcast verified 1280×720, 25fps, 48f, 1.92s). Eyeballed broadcast+top frames: ~20 colour-coded
    SMPL-X bodies spread in 3D + ball; **no pitch line markings** (the `blender_animate` video path
    uses a plain grass plane, *not* the M2-9 ribbon adapter). **Honesty (R-6):** reconstruction
    **`done in 6s` = a content-addressed cache hit (ADR-0004)** — the perception nets (RF-DETR/
    ByteTrack/SMPLest-X-H/WASB/PnLCalib) did **not** re-execute this run (continuity/coherence numbers
    are identical to the 2026-06-24/25 runs); cached outputs on the pod volume were reused, so what
    this run validates fresh is the **GPU OptiX video-render path**, not a new perception pass. `calib
    confidence mean=0.95` here is the 48-frame in-sample self-score (the `0.61` in #103 was an 8-frame
    subset — different sample, not GT accuracy); `ball height_confidence mean=0.25 min=0.00` with 8
    `low_ball_height` attention items (R-6 marked). The M2-10 photoreal **adapter** (observe / seam-A
    re-render / edit↔Cycles) is covered by unit+gated tests, not by this legacy `blender_animate` video.
  - **M1 CLOSED 🟢 — milestone flipped 🟡→🟢 (2026-06-27).** With the editable loop proven end to end
    on a real clip (above + the 2026-06-24/25 real-model E2E runs), all three TZ exit criteria are
    honestly MET, so M1 is closed in `roadmap.md` (M1 header + steps 6/7/12 + M0-6) and here (milestone
    table + TL;DR). **Evidence:** **AC-1** overlay-matches-source — quantified by **B1 SoccerNet
    `calibration-2023`** reprojection (median **0.236 m**, completeness 0.745), the measured
    overlay-to-source floor on real frames; **AC-2** fix-a-pose-propagate-three-ways — the four
    correction modes (offset / keyframe-interp / temporal-smoothing / re-fit) with `preview_subject_motion`
    and non-destructive resolve are unit-tested (`tests/unit/test_corrections.py`) and exercised on the
    pod (`corr-1`); **AC-3** curves-edit-show-export — root/ball/axis-angle F-curves in the editable
    `.blend`, mapped back to `Correction`s, resolved scene exports (glTF assembly tested + `.npz`/JSON;
    pod E2E exports a full scene). Suite **480 passed / 5 display-gated skips** (exit 0). **Ceiling kept
    visible (R-6, criteria not narrowed):** (1) the photoreal-grade pose/ball *heavy* nets ship as
    **injection-seam stubs** — the pure halves are real + CPU-validated, real backends inject by dotted
    path (ADR-0006, pod-validated on CUDA), the boxed heavy halves raise an actionable
    `NotImplementedError`; (2) the **GUI** live-Blender placement session is **display-gated** (host-side
    seam real + socket-tested, no display in CI); (3) the metric-XY floor is measured on **lined**
    footage — an unlined clip grounds Z + topology, not absolute pitch XY. These are delivery/environment
    seams, not missing capability. **Surveyed tails (open, non-blocking):** mypy 20 (15 deferred-payload
    + the 5 new M2-render-cluster guards, above), ruff 46 (style backlog, not churned) — both ungated;
    GUI live session + real Blender SCENE_3D observer track under **B4** / M2-M3.
- Finish **Bug2** mypy debt; tighten the seams.
- **B4** real Blender SCENE_3D observer (M2); progress toward the LLM-over-MCP north-star (ADR-0008).

## Autonomy boundaries for this run

**Executed autonomously (committed locally at each checkpoint):** Phase A items, B2 research,
and any pure-core code/test/doc work.

**Held for explicit user authorization:** starting the GPU box (~$0.69/hr, deliberately shut
down), pushing to public GitHub, and anything in Phases B/C/D that needs the box, an external
asset, or a display (B1, B3, B4, live pose/ball wiring).
