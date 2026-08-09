# Verticality — the headline number was mis-scaled, and there are two defects under it

The board has carried, for four plan revisions: *"largest root-Z excursion in a whole scene is
**0.082 m** against ~0.23 m for a real player."* Measured against real ground truth, that comparison
does not hold, and what it was hiding is more actionable than what it claimed.

Reproduce: `PYTHONPATH=src python scripts/bench_vertical_motion.py` (CPU, seconds).

---

## 1. The comparison was between two different statistics

Root-Z excursion is `max − min`, so it **grows with the window**. Measured over 1995 real
player-tracks (WorldPose, 89 clips × 22 players):

> **Corrected 2026-08-09, same day.** The first version of this table compared windows in
> **frames**. WorldPose is **50 fps** (89/89 clips) and our clip is **29.97**, so 60 GT frames is
> 1.2 s against our 2.0 s — it understated GT by 1.67×. Every figure below is now matched in
> **seconds**, and per-frame steps are rescaled to our frame interval. Logged in
> [`landmines.md`](landmines.md).

| window | GT frames | p05 | median | p90 | p99 |
|---|---|---|---|---|---|
| 2.0 s | 100 | 0.010 | **0.044** | 0.150 | 0.360 |
| 7.87 s | 394 | 0.035 | **0.119** | 0.271 | 0.722 |
| 20.0 s | 1000 | 0.077 | **0.201** | 0.356 | 0.900 |

The 0.23 m in the headline is the **whole-clip median**. The 0.082 m is the **maximum of a
2-second scene**. Different window, different statistic.

Matched in seconds, on the same statistic:

| scene | duration | ours (median) | GT | |
|---|---|---|---|---|
| `out/cue/scene_off.json` | 2.0 s | 0.008 m | 0.044 m | **5.5× under** |
| `out/res_ab236/f236_res896.json` | 7.9 s | 0.160 m | 0.119 m | **1.3× over** |

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

> rescaled to 29.97 fps: median **0.0008 m**, p90 **0.0041 m**, p99 **0.0122 m**

Ours on `f236_res896`, **measured frames only**: median **0.0036 m** — **4.4× GT** — with p90
0.0147. (Over *all* frames the median reads 0.0005, because the imputed majority is frozen; that
average hides the defect, which is why the split below matters.)

**And that is out of proportion to the pose it is derived from.** On the same frames, our
per-joint `|Δbody_pose|` is **0.0197 rad against GT's rescaled 0.0144 — only 1.37×**. Root Z is
forward kinematics of that pose, so it should jitter *proportionally*. It jitters **three times
harder than its own input.**

Split by provenance, counting steps above the GT p90:

| provenance | frames | above GT p90 |
|---|---|---|
| measured | 4031 | **60 %** |
| interpolated | 146 | 60 % |
| imputed | 4753 | 7 % |

**The spikes are on measured frames, not on gap-filled ones.** This is not coherence inventing
motion; the per-frame FK estimate itself is noisy. (Imputed frames are the *quietest*, which is
consistent with #135's finding that an imputed run is a frozen mannequin.)

## 4. Where the amplification comes from

`smplestx_backend.py:_pelvis_above_foot` ends in:

```python
return float(j[0, 1] - j[list(_FOOT_JOINTS), 1].min())
```

Two amplifiers, one of them named in its own docstring:

- **`min()` over four foot joints** (both ankles, both toe bases). Under per-joint noise the
  minimum is biased downward, has larger variance than any single joint, and can **switch which
  joint wins between frames** — a step change from no real motion.
- **Zero global orient.** The docstring's own R-6 caveat: *"zeroing global orient assumes body-up ≈
  world-up, so a markedly leaning torso biases the height."* A running footballer leans, and the
  lean changes every frame.

That is the 1.37× → 4.4× gap, and it is a specific function rather than a research programme.

The structural parallel still holds: the estimate is made **independently per frame** with nothing
tying consecutive frames together, exactly like the per-frame homographies. But here the dominant
term is the derivation, not the estimator.

## 5. What to do, and what not to

**Do not reach for a smoother first.** ADR-0012 Tier 1 records the yaw low-pass: an iterative
moving average removed 90 % of the jitter and flattened 100°-plus real turns. A blind filter on Z
would do the same to a real crouch.

What is new here is that we now have a **measured envelope from real football** rather than a
guessed one — GT per-frame `|dZ|` p90 = 0.0041 m, p99 = 0.0122 m at our frame rate, and excursion
percentiles per second. That is exactly the input a physics gate needs and the thing
`config/physics.yaml` constants have historically lacked. Order:

1. **Mark the constant-Z fallback** (`pose.py:334`) rather than substituting silently — the cheapest
   item and it is item B of the architecture plan.
2. **Gate on the measured envelope**, not on a chosen threshold: a per-frame vertical step above the
   GT p99 is not football. Reject or mark it; do not average it away.
3. **Attack the two amplifiers in `_pelvis_above_foot` before touching the pose estimator.** The
   pose is 1.37× GT; the height derived from it is 4.4×. A softer foot statistic than `min()` over
   four joints, and honouring the torso lean instead of zeroing global orient, are local changes
   with a measured target to hit.

## 6. What this does not answer

- **Why 25 % of subjects in one scene had no `pelvis_above_foot`.** Most likely a missing SMPL-X
  model dir (`PITCH3D_SMPLX_MODELS`) at the time of that run, but that run's provenance is not
  recorded — which is the point of §2.
- **Whether the eye can see any of this.** Every number here is against GT, not against the user's
  judgement. 0.008 m vs 0.044 m is under 4 cm; whether that reads as "nobody crouches" on screen
  is a separate question, and the user's eye decides it.
