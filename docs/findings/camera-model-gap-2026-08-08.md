# The camera-model gap, measured — and what it costs in metres

**Question.** [`pipeline-io-proposed.md`](../pipeline-io-proposed.md) observes that the drawn pitch
does not sit on the painted pitch and skeletons do not sit on players, both worsening toward the
frame edges, and proposes three candidate causes — zoom, camera translation, lens distortion —
ordered by guess. This measures which of them is real.

**Answer.** Zoom, by a wide margin. Distortion second. Translation does not exist. The ordering in
the proposal is close to inverted.

Reproduce: `PYTHONPATH=src python scripts/bench_camera_model_gap.py` (CPU, seconds, no pipeline).
The reply sent to the documents' author is [`review-pipeline-io-2026-08-08.md`](review-pipeline-io-2026-08-08.md).

---

## 1. The evidence we already had on disk

89 WorldPose clips carry a **per-frame GT camera** — `K (T,3,3)`, `R`, `t`, and 5 distortion
coefficients — for real World Cup broadcast footage. All 89 also have GT SMPL poses
(`global_orient`, `body_pose`, `transl`, `betas` — our exact `PoseSequence` schema, 22 players)
**and** the video.

This corrects `pipeline-io-proposed.md:139-141`, which says the video "is an agreement with FIFA,
not annotation work". It is on disk: `models/worldpose/` (24 GB), cameras and poses in
`WorldPose/{cameras,poses}/`. Same stale claim that `pose-bakeoff-runbook.md` carried until
2026-08-07.

## 2. What a real broadcast camera does — 89 GT clips

| | median | p90 | max |
|---|---|---|---|
| focal `fx` (px) | 3821 | 5145 | 5605 |
| **focal drift within a clip** (% of mean) | **44.1 %** | 68.4 % | 100.2 % |
| principal-point wander (px) | 1.1 | 5.2 | 10.8 |
| \|k1\| max in clip | 0.509 | 0.725 | 0.845 |
| **camera translation** (m, max from mean) | **0.000** | 0.000 | 0.000 |

- **Focal moves >5 % in 89/89 clips.**
- **The camera translates in 0/89 clips** — exactly 0.000 m. Our fixed-centre assumption is right.
- Distortion at the frame corner, on WorldPose's own fitted model: **≈47 px** (median), 66 px (p90).

Verdict on the proposal's three hypotheses:

| hypothesis | proposal's rank | measured | |
|---|---|---|---|
| zoom | 3rd (step 2b, "only if the operator zoomed") | universal, median 44 % | **promote to 1st** |
| distortion | 1st (step 2a) | ~47 px at the corner — real | keep, 2nd |
| camera translation | 2nd | 0.000 m in every clip | **delete** |
| free principal point | step 2c, "sometimes the whole answer" | 1 px median wander | **delete** |

## 3. Scale matters — our fit is 60 frames, not 1000

A whole-clip figure overstates the problem for a 2-second fit. Drift inside a sliding window:

| window | median | p90 | max | windows >2 % |
|---|---|---|---|---|
| 30 frames | 0.8 % | 4.2 % | 21.8 % | 32 % |
| **60 frames — our fit** | **2.0 %** | 8.1 % | 31.5 % | **50 %** |
| 120 frames | 4.7 % | 13.6 % | 59.9 % | 69 % |
| 240 frames (8 s) | **9.2 %** | 21.8 % | 49.2 % | 87 % |
| 480 frames | 16.5 % | 30.7 % | 65.5 % | 97 % |

## 4. The number the goal is stated in

A ground point at distance `d` projects to `v = f·h/d` below the principal point. Perturb `f` and
invert. For our camera (f = 4169 px, height 17.22 m):

| ground distance | 1 % | 2 % | 5 % | 9 % |
|---|---|---|---|---|
| 20 m | 0.20 m | 0.39 m | 0.95 m | 1.65 m |
| 40 m | 0.40 m | 0.78 m | 1.90 m | 3.30 m |
| 60 m | 0.59 m | 1.18 m | 2.86 m | 4.95 m |
| 80 m | 0.79 m | 1.57 m | 3.81 m | 6.61 m |

**Read the two tables together.** At our current 60-frame scale one focal costs **~0.4–0.8 m** of
player position at pitch distances. At the 240-frame scale a real episode needs, it costs
**2–5 m**. The single-focal assumption does not merely blur the edges — it is a metre-scale
position error that grows with clip length, which is precisely the axis the goal wants to extend
along.

It also explains the observation that started this: `f236` is 8 seconds in, i.e. squarely in the
9 %-drift regime — **and outside the fitted span of `calib/Colombia-1-0-Congo-DR1080p.npz`, which
covers frames 0–59 only.**

## 5. What this does not answer

- **Whether *our* clip zooms.** These are 89 other clips. The cheap test on ours: fit the
  single-focal model over sliding windows and see whether the recovered focal ramps. If it is flat
  over 0–59, the golden test's pinned 4169.32 px is honest for that span and the problem starts
  where the span ends.
- **Camera vs player.** Every number here is about the camera. It cannot distinguish a camera
  error from per-player grounding or from an association failure. That needs the three-residual
  pass (pitch paint · player foot vs box bottom · common-mode vs per-player scatter).
- **Verticality.** #135 П5 measured the largest root-Z excursion in the whole scene at **0.082 m** —
  nobody ever leaves the ground. No camera fix touches that, and neither `pipeline-io.md` nor the
  proposal mentions it. For *poses* it is a first-order realism defect.

## 6. Consequences for the plan

1. **Per-frame focal is the first camera change, not the last.** It breaks
   `tests/e2e/test_golden_real_camera.py`, which pins one focal for 60 frames and is
   mutation-checked — that test must be **re-measured, not nudged**.
2. **Fit `k1`.** The four projectors listed in the proposal must stay in step or overlay and export
   diverge again.
3. **Drop free principal point and camera translation** from the plan.
4. **Replace synthetic calibration GT (proposal step 3) with WorldPose.** Real broadcast, GT camera
   *and* GT pose, in our schema, already local. It measures both halves of the goal; synthetic
   measures one and carries a domain gap.
