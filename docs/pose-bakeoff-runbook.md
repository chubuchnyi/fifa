# Pose bake-off runbook — measure candidates on WorldPose, in metres

**Audience:** an agent on the RunPod box (or the human) ready to *finalize the pose pick (B2)* with
numbers instead of vibes. Prereq: the heavy stack is installed per
[`runpod-agent-setup.md`](runpod-agent-setup.md). This doc is the **"then eval on WorldPose in
metres"** step that runbook defers (its Step 7) — the concrete procedure to turn *models + data on
box* into a **Global + Local MPJPE table** that decides SMPLest-X vs the SAM 3D Body fallback, and
that quantifies how much our **calibration** (not the pose net) costs us.

**Why this exists:** the project memory recommends **SMPLest-X + SMART** over the earlier SAM 3D Body
pick on *published* evidence (off-the-shelf global HMR on WorldPose = **metres** of error; SMART's
recipe = **0.32 m**). That recommendation is still **unconfirmed on our own stack**. This bake-off
confirms (or overturns) it before we sink the wiring effort of B3 into the wrong backbone.

---

## 0. Prerequisites — READ FIRST (two gates, one of them not yet met)

| gate | status | note |
|---|---|---|
| **GPU / box up** | ⏳ transient | host `r1zrm95x23ql` had no free RTX 4090 (2026-06-22); `scripts/_poll_start_pod.sh` auto-starts the pod the moment it frees. |
| **WorldPose FRAMES (RGB)** | ❌ **MISSING** | **blocking.** WorldPose-**Light** on the box is **annotations only** (boxes / `K_t` cameras / 3D skel / pitch landmarks) — **no video.** A per-crop HMR net (SMPLest-X, SAM 3D Body) needs the **pixels**. No frames → no predictions → no bake-off. |

> **The single most leveraged asset right now is the WorldPose video/frames.** It unblocks **both**
> this bake-off **and** B1 (honest calibration eval) — same data, two payoffs. The frames are the
> full **WorldPose** release (ETH AIT, `eth-ait.github.io/WorldPoseDataset`), *not* the Light
> annotations package. Pull them aligned to the Light `clip_id`/frame indices, onto `$WS/datasets/`,
> after sizing the volume ≥ 200 GB (footage is tens of GB; see runpod-agent-setup §Prereqs).

**Assets that ARE already on box** (verify before starting):

```bash
export WS=/workspace
ls "$WS/weights/smplest-x" "$WS/weights/sam-3d-body-dinov3" \
   "$WS/weights/smplx/models/smplx" \
   "$WS/datasets/worldpose-light" \
   "$WS/repos/FIFA-Skeletal-Tracking-Starter-Kit-2026"
ls "$WS"/SMPLest-X/human_models/human_model_files/smplx/SMPLX_to_J14.pkl   # joint-reg for the 14/15-kp GT
```

---

## 0a. Alternative eval data — verified 2026-06-22 (WorldPose video is gated)

WorldPose's **video is gated behind a FIFA content-licence form** (`worldpose.ait.ethz.ch`) —
registering for the challenge (Codabench comp. 11681 val / 11682 test) does **not** grant frames,
the test GT is held out, and the HF Light mirror is annotations-only. The block is real, not a
missing public mirror. There is **no single drop-in replacement** (WorldPose uniquely combines
real broadcast pixels + 3D-world GT + GT camera + soccer); assemble partial coverage instead:

| dataset | GT | real? | soccer? | access | use |
|---|---|---|---|---|---|
| **SoccerNet** (Calib / Field-Loc) | pitch-line → camera-calibration GT; **no 3D-pose GT** | ✅ broadcast | ✅ | `pip install SoccerNet`; video via NDA form | **B1** (calibration) — most accessible real soccer |
| **EMDB** | SMPL + **global body & GT camera trajectory** (world coords) | ✅ | ❌ | application form; non-commercial | **B2 global** — best for Global-MPJPE methodology |
| **3DPW** | SMPL params + per-frame camera, multi-person | ✅ moving cam | ❌ | homepage DL; MPI non-commercial | **B2 local** — standard articulation benchmark |
| **AGORA / BEDLAM** | SMPL-X params | ❌ synthetic | ❌ | register + license | pretrain/aux only (not metre-accurate real) |

**Our in-house substitute (no box, no asset):** `pitch3d.eval` — a deterministic synthetic
broadcast-soccer generator (`generate_scene`) with **perfect GT** (camera, world 3D joints,
2D, bboxes, GT homography) + the MPJPE metrics + condition-A placement. It lets the whole
harness (crop→place→correspond→eval geometry) be built and unit-tested *now*; swap the
placeholder skeleton for SMPL-X FK and the synthetic frames for EMDB/3DPW/WorldPose later.

---

## 1. The metric — defer to the challenge's OFFICIAL evaluator

The benchmark is **Global MPJPE** and **Local MPJPE**, both **in metres**, **no Procrustes / no PA**:

- **Global MPJPE** — error of joints in **world** coordinates (placement + articulation). This is what
  "looks great in a demo, garbage in reality" actually measures; world-grounding dominates it.
- **Local MPJPE** — error after subtracting each frame's **root** (pelvis) from pred and GT
  (root-relative; articulation only).

> **Do NOT hand-roll the metric for the headline numbers.** The exact joint correspondence (SMPL/
> SMPL-X joints → the GT 15-kp / J14 set), root choice, distortion handling, and visibility masking
> are baked into the **starter kit's evaluator** — reimplementing them is how you get numbers that
> look plausible and are wrong. Use the starter kit's `eval` against the Codabench format; the
> embedded snippet in §5 is a *sanity cross-check only*.

---

## 2. Procedure (per candidate)

For each candidate backbone, produce predictions in the challenge's submission format, then evaluate.

1. **Align inputs.** For each `clip_id`, pair the WorldPose **frame images** with the Light
   annotations (per-frame **boxes**, intrinsics **`K_t`** + distortion, initial extrinsics, GT 3D
   skel). The starter kit ships the loader — use it; don't re-derive the file layout.
2. **Per-crop pose.** Crop each box, run the candidate → camera-space **SMPL(-X)** params
   (`global_orient`, `body_pose`, `betas`). This is exactly what the `HMRBackend.estimate_bodies`
   seam (`src/pitch3d/adapters/models/pose.py:85`) will return once wired — so the bake-off harness
   and the product share one contract.
3. **FK → joints.** Run the body model forward to **joint positions** (not just axis-angle). Freeze
   hands/face (soccer GT is body-only).
4. **Place in world — TWO conditions** (this is the point; see §3):
   - **(A) GT camera:** use the challenge-provided `K_t` + extrinsics. Isolates **pose-net** quality.
   - **(B) our calibration:** use **PnLCalib** (+ foot-plane anchoring) to estimate the camera, then
     place. Measures what our **product** actually delivers.
5. **Correspond + evaluate.** Map predicted joints → the GT joint set (`SMPLX_to_J14.pkl`) and run the
   **official evaluator** → Global + Local MPJPE, per condition.

---

## 3. Conditions — separate pose-net error from calibration error

Run the grid below. The **A vs B gap** is the cost our calibration adds on top of a perfect camera —
i.e. it tells us whether to spend the next effort on the **pose net** or on **calibration**, the
project's #1 leverage item.

| condition | camera | grounding | isolates |
|---|---|---|---|
| **A** | GT `K_t`+extrinsics | GT | pure pose-net / articulation |
| **B** | PnLCalib (ours) | foot-plane anchor | full-pipeline (product reality) |

Candidates (same seam, swap the backend):

| candidate | weights on box | output | role |
|---|---|---|---|
| **SMPLest-X (Huge)** | `weights/smplest-x` | SMPL/SMPL-X native | **primary** (memory's pick) |
| **SAM 3D Body** | `weights/sam-3d-body-dinov3` | MHR → convert | fallback / cross-check |
| *(zero-pose baseline)* | — | T-pose @ grounded root | sanity floor for Local MPJPE |

---

## 4. Sanity anchors (published / memory — expect this order of magnitude)

- Off-the-shelf **global** HMR on WorldPose: **metres** (GLAMR ~18.9 m, SLAHMR ~8.3 m). If condition
  **B** lands in metres, that's *expected* for un-tuned global HMR — the recipe, not the backbone, is
  what buys the jump.
- **SMART** recipe (depth-supervised FT + foot-anchor + 2-pass smoothing): **~0.32 m** Global.
- Calibration ablation (Global MPJPE): field-markings-only **548 mm** vs +player-keypoint bundle-
  adjust **80 mm** (≈7×). Expect a large **A→B** gap until PnLCalib gets player-keypoint BA.

A result that is *too good* (sub-decimetre Global from an un-tuned net) almost always means the GT
camera leaked into condition B, or PA crept into the metric. Distrust it; re-check §1.

---

## 5. Embedded MPJPE — sanity cross-check ONLY (authoritative = starter kit)

Assumes joints are **already corresponded** (same order, metres) and shaped `(T, J, 3)`.

```python
import numpy as np

def mpjpe_global(pred, gt):                 # (T,J,3) world metres -> scalar metres
    return float(np.linalg.norm(pred - gt, axis=-1).mean())

def mpjpe_local(pred, gt, root=0):          # root-relative (no Procrustes, no PA)
    p = pred - pred[:, root:root+1, :]
    g = gt   - gt[:,   root:root+1, :]
    return float(np.linalg.norm(p - g, axis=-1).mean())

# Self-test: a pure 1 m world translation is invisible to LOCAL, full cost to GLOBAL.
if __name__ == "__main__":
    rng = np.random.default_rng(0); gt = rng.normal(size=(4, 15, 3))
    pred = gt + np.array([1.0, 0.0, 0.0])
    assert abs(mpjpe_global(pred, gt) - 1.0) < 1e-9
    assert mpjpe_local(pred, gt) < 1e-9
    print("sanity OK")
```

If these two numbers disagree wildly with the official evaluator on the same arrays, the **joint
correspondence or root choice is wrong** — fix that before trusting any headline number.

---

## 6. Record the result

Write the grid into [`m1-status-and-plan.md`](m1-status-and-plan.md) (B2 row) as a small table:
`candidate × {A,B} × {Global, Local} MPJPE (m)`. That table — not the published claim — is what
finalizes B2 and picks the backbone we wire in B3.

> Stop the box when the bake-off finishes (`runpodctl pod stop <id>`); `/workspace` persists.
```
