# Verticality — the headline number was mis-scaled, and there are two defects under it

The board has carried, for four plan revisions: *"largest root-Z excursion in a whole scene is
**0.082 m** against ~0.23 m for a real player."* Measured against real ground truth, that comparison
does not hold, and what it was hiding is more actionable than what it claimed.

Reproduce: `PYTHONPATH=src python scripts/bench_vertical_motion.py` (CPU, seconds).

---

## 1. The comparison was between two different statistics

Root-Z excursion is `max − min`, so it **grows with the window**. Measured over 1995 real
player-tracks (WorldPose, 89 clips × 22 players):

| window | p05 | median | p90 | p99 |
|---|---|---|---|---|
| 60 frames | 0.005 | **0.028** | 0.109 | 0.258 |
| 236 frames | 0.022 | **0.085** | 0.220 | 0.596 |
| 1032 frames | 0.081 | **0.204** | 0.357 | 0.904 |

The 0.23 m in the headline is the **whole-clip (1032-frame) median**. The 0.082 m is the **maximum
of a 60-frame scene**. A 7× window mismatch, and a max against a median. `0.082` is in fact at
roughly the p80 of GT's own 60-frame distribution.

Window-matched, on the same statistic:

| scene | window | ours (median) | GT | |
|---|---|---|---|---|
| `out/cue/scene_off.json` | 60 | 0.008 m | 0.028 m | **3.5× under** |
| `out/res_ab236/f236_res896.json` | 236 | 0.160 m | 0.085 m | **1.9× OVER** |

So there is no single "we have too little vertical motion". One scene has almost none and a newer
one has nearly twice too much.

## 2. Defect A — the root Z is sometimes a literal constant, unrecorded

`adapters/models/pose.py:334`:

```python
if pelvis_above_foot is None:
    z = np.full(tl.frames.shape[0], self.pelvis_height_m)   # constant
else:
    z = np.asarray(pelvis_above_foot, ...)                  # varies with crouch/run/stride
```

`pelvis_above_foot` is per-frame SMPL-X forward kinematics, computed by
`smplestx_backend.py:175`. When the backend does not supply it, the pelvis height becomes a fixed
nominal value and **nothing in the scene records that it did.**

Measured in `out/cue/scene_off.json` — the reference scene for the #135 eye labels:

> **6 of 24 subjects (25 %) have exactly constant Z**, all at **0.92 m**, the nominal.
> Their per-frame `|dZ|` median is **0.0000 m**.

In `out/res_ab236/f236_res896.json`: 0 of 38. So the FK path worked in the later run and not for a
quarter of the earlier one, and the only way to tell is to measure the variance of the output.

**This is the same shape as #140** — a capability degrades to a silent constant and the artifact
does not say so. It belongs to items A and B of the architecture plan, not to a physics fix.

## 3. Defect B — where it varies, it varies too abruptly, on measured frames

Real vertical motion is built from small steps. Per-frame `|dZ|` for real players:

> median **0.0005 m**, p90 **0.0025 m**, p99 0.0073 m

Ours, on `f236_res896`: median **0.0005 m** — identical to GT — but p90 **0.0097 m**, i.e.
**3.9× spikier in the tail**. The typical step is right and the outliers are not, which is why the
excursion overshoots while the motion still looks plausible frame to frame.

Split by provenance, counting steps above the GT p90:

| provenance | frames | above GT p90 |
|---|---|---|
| measured | 4031 | **60 %** |
| interpolated | 146 | 60 % |
| imputed | 4753 | 7 % |

**The spikes are on measured frames, not on gap-filled ones.** This is not coherence inventing
motion; the per-frame FK estimate itself is noisy. (Imputed frames are the *quietest*, which is
consistent with #135's finding that an imputed run is a frozen mannequin.)

## 4. The structural parallel

Defect B has the same shape as the camera thread's conclusion. The pelvis height is estimated
**independently per frame** from a single crop, with nothing tying consecutive frames together —
so each estimate is individually plausible and the sequence is collectively wrong. That is the
homography problem again: per-frame free fits, good singly, incoherent jointly.

## 5. What to do, and what not to

**Do not reach for a smoother first.** ADR-0012 Tier 1 records the yaw low-pass: an iterative
moving average removed 90 % of the jitter and flattened 100°-plus real turns. A blind filter on Z
would do the same to a real crouch.

What is new here is that we now have a **measured envelope from real football** rather than a
guessed one — GT per-frame `|dZ|` p90 = 0.0025 m, p99 = 0.0073 m, and excursion percentiles per
window. That is exactly the input a physics gate needs and the thing `config/physics.yaml`
constants have historically lacked. Order:

1. **Mark the constant-Z fallback** (`pose.py:334`) rather than substituting silently — the cheapest
   item and it is item B of the architecture plan.
2. **Gate on the measured envelope**, not on a chosen threshold: a per-frame vertical step above the
   GT p99 is not football. Reject or mark it; do not average it away.
3. **Only then consider temporal coupling in the FK estimate itself**, which is the real fix for
   defect B and the same remedy shape as constraining the camera solve.

## 6. What this does not answer

- **Why 25 % of subjects in one scene had no `pelvis_above_foot`.** Most likely a missing SMPL-X
  model dir (`PITCH3D_SMPLX_MODELS`) at the time of that run, but that run's provenance is not
  recorded — which is the point of §2.
- **Whether the eye can see any of this.** Every number here is against GT, not against the user's
  judgement. 0.008 m vs 0.028 m is 2 cm at 40 m; whether that reads as "nobody crouches" on screen
  is a separate question, and the user's eye decides it.
