# #133 — human physics: filter motion, solid bodies, no popping, limbs in sync

**Stated by the user 2026-08-06**, and explicitly *re-*stated: it had been discussed before and
lost. That is why it is written down here rather than acted on and forgotten again.

> Нужно в фильтре поз и перемещения игроков заложить физику человека, чтобы фильтровались
> движения, тела были твёрдыми, не появлялись изниоткуда и не исчезали вникуда и не проницали
> сквозь друг друга. Чтобы человек перемещался синхронно с движениями ног, рук, наклоном тела и
> т.п. согласно реальной физике человека.

Four requirements: **(1)** implausible motion is filtered, **(2)** bodies are solid and do not
interpenetrate, **(3)** subjects do not pop into or out of existence, **(4)** translation is
consistent with the limbs — legs, arms, body lean — as a real human moves.

## The surprising part: most of this is already built, and switched off

`config/physics.yaml` ships nearly every one of these as a gate with `enabled: false`. The code
exists in `core/correction/`; with the flag off each gate takes a documented *measure-only* path.
This was discovered on 2026-08-06 while chasing pose accuracy: the joint and orientation ceilings
were being measured 118 and 11 times per scene and never enforced, under comments reading
"NOT YET BUILT (schema reserved)" that the code had outgrown.

| Requirement | Gate | State 2026-08-06 |
|---|---|---|
| 1 · filter motion | `kinematic` — root speed 10.5 m/s, accel 8 m/s², teleport marking | **on** (shipped) |
| 1 | `joint` — per-joint ω ≤ 600 °/s | **on** — turned on today via `safe_new` |
| 1 | `orientation` — root turn ≤ 720 °/s | **on** — today |
| 1 | `jerk_clamp` — peak jerk ≤ 200 m/s³ | built, **off** |
| 1 | `inertia_smooth` — angular accel ≤ 15 rad/s² | built, **off** |
| 1 | `joint_smooth` — per-joint MA, "kills HMR twitch" | built, **off** |
| 2 · solid bodies | `collision` — capsule repulsion, r = 0.35 m | built, **off** (known accel-spike tradeoff) |
| 3 · no popping | continuity stitch · kit-split (#132) · teleport marks (R-6) | **on** |
| 3 | `identity` — DBSCAN track split + cross-track merge | built, **off** |
| 4 · limbs in sync | `pose_motion_sync` — procedural stride when velocity and leg activity disagree | built, **off** |
| 4 | `contact_probe` — detect and measure foot slide during ground contact | built, **off** |
| 4 | `foot_plant` — recentre root Z so subjects stop hovering | built, **off** |
| 4 | `foot_floor` — no sinking below the pitch | **on** — today |
| 4 | `gravity_project` — airborne Z follows a ballistic parabola | built, **off** |
| 4 | `orient_verticality` — body-up to world-up, blocks lying flat | built, **off** |

`full_realism` turns on almost all of them; `full_realism_collide` adds collision. Neither has ever
been run on the real clip and judged.

## What is genuinely missing, stated honestly

Turning the flags on is not the same as having human physics, and the difference should not be
glossed:

* **These are kinematic clamps, not a simulation.** Each bounds a rate or resolves an overlap after
  the fact. Nothing enforces momentum, balance or ground reaction: a player can still be clamped
  into a pose no human could hold, as long as every individual rate is under its ceiling.
* **`pose_motion_sync` moves knees and hips only** (`knee_amplitude_rad`, `hip_amplitude_rad`).
  The user asked for arms too, and there is no arm-swing term.
* **`collision` is one capsule per player**, r = 0.35 m — it stops torsos merging, not limbs
  passing through each other.
* **No metric yet separates a wrong pose from unusual play.** A facing-vs-travel metric was tried
  on 2026-08-06 and withdrawn the same day: footballers legitimately run backwards and sideways,
  so it measured football rather than the reconstruction. The joint-rate ceiling is the counter-
  example worth copying — 2212 °/s beats a human limit no matter which way the player runs.

## How to proceed, and the rule that governs it

Each gate is turned on only with a measurement of **both halves**: what it fixes, and what it
costs in real motion. The precedent is `scripts/pose_gate_ab.py` for the joint/orientation pair —
violations 118 + 11 → 0, worst rates 2212 → 600 and 4514 → 720 °/s, while keeping 97.8 % of root
and 98.5 % of body-joint angular travel and moving players 0.0000 m
(`scripts/bench_subject_steadiness.py`).

This matters because the repo has been burnt: an iterative moving average on HMR yaw removed 90 %
of the jitter *and* flattened 100°+ real turns. A gate that buys plausibility by eating real motion
is worse than the jitter it removes, and only the paired measurement shows the difference.

Order to work through, cheapest evidence first:

1. `full_realism` vs `safe_new` on the real clip — most of requirement 1 and 4 in one step.
2. `collision` (`full_realism_collide`) separately, because its accel-spike tradeoff is documented
   and needs `momentum_smooth` to clean up.
3. `identity` for requirement 3, against the kit-split baseline already measured in #132.
4. Only then consider what is genuinely absent — arm swing, per-limb collision, and whether any of
   this needs to become a real dynamics pass rather than a stack of clamps.

## Requirement 3 measured: how often does a player pop in or out? (2026-08-06)

The user's phrasing — *"футболисты спекаются, потом рождаются новые из 2-х - 3"* — names two
symptoms. The second one is measurable without any model, and it is large.

A track that starts or ends **at the frame border** is a player walking into or out of shot, which
is legitimate. A track that starts or ends **mid-pitch** is not: nobody entered, the tracker simply
invented or dropped an identity. Splitting shot 1's events that way (53 tracks, raw tracker output):

| | total | of them mid-pitch |
|---|---|---|
| births after the clip start | 35 | **31** |
| deaths before the clip end | 37 | **29** |

**60 identity events in 8 seconds that no entry or exit explains** — about one spurious birth or
death *per track*. That is the "new ones are born out of two or three" the user sees, counted.

Two honesties about the number:

* It is the **raw tracker output, before stitching**. `continuity.py` re-links some of them — it
  takes 56 ids to 36 with 14 merges on the same shot — so the residual reaching the render is
  smaller, but still tens of events.
* It says nothing about *why*. It is consistent with the crossing hypothesis (94 % of these events
  fall within ±2 frames of a frame holding a contaminated crop, against a 76 % base rate) but does
  not prove the crossing caused them.

**Where this points.** Against it, the pose side is so far quiet: 11 stratified overlapping pairs
scored at the time of writing, every one **separate**, cross-contamination 0.01–0.24. If the tail
of that sweep stays quiet, the visible "fusing and multiplying" is an **identity** failure, not a
pose one — the player is not merging with his neighbour's body, he is losing his id and being
replaced by a new subject with a new avatar.

That class of failure has a direct candidate in the 2026-08-04 survey: **McByte** (CVPRW 2025, MIT,
training-free) is ByteTrack plus a temporally-propagated mask cue, reporting **HOTA 85.0 vs 72.1**
for plain ByteTrack on SoccerNet-tracking. Our tracker *is* ByteTrack, so it is an upgrade in
place rather than a new stack. Not adopted yet — the sweep has to finish first, because adopting a
model on partial evidence is exactly the mistake that produced the withdrawn A2 verdict.

## Splitting the identity failure: association or detection? (2026-08-06)

60 mid-pitch identity events is a symptom, not a cause, and the two possible causes have opposite
cures. `scripts/identity_failure_kind.py` separates them off the cached detections and the
tracker's own output, running no model: an **orphan** is a detection no live track claims, and an
event is an *association* miss if an orphan sits near where constant-velocity extrapolation puts
the dying track.

| | mid-pitch events | association miss | detection miss |
|---|---|---|---|
| `min_track_frames=4` | 60 | 44 (73 %) | 16 (27 %) |
| `min_track_frames=1` | 78 | 55 (70 %) | 23 (30 %) |

The second row exists because the first is confounded: tracks under 4 frames are filtered out of
the npz, so a track that *was* created could masquerade as an orphan. Dropping the filter raises
the absolute counts and leaves the proportion alone, so the confound is not the explanation.

**~70 % is therefore the ceiling for any association work** — masks, re-ID, McByte. The other 30 %
is detector recall and no tracker change reaches it. The orphans sit a median of **6–23 px** from
the extrapolated position, and **72 % of them score under 0.4** against a claimed-detection median
of 0.663.

### The knob I checked before the model — and got backwards

We construct `sv.ByteTrack(frame_rate=…)` and nothing else, so `minimum_matching_threshold` sits at
its default 0.8. Reading "matching threshold" as an IoU floor, I predicted it was too strict for
35-px-wide players and that lowering it would recover the orphans. **Wrong on the semantics:** it is
a threshold on ``1 - IoU``, so 0.8 already means "match anything above 0.2 IoU", and lowering it
*tightens* association. The sweep says so plainly:

| `minimum_matching_threshold` | player tracks | mid-pitch events |
|---|---|---|
| **0.80 (default, current)** | 56 | **66** |
| 0.70 | 67 | 88 |
| 0.60 | 97 | 142 |
| 0.50 | 104 | 162 |

A second flaw in that first sweep: the kit-change column read 0 at every setting, because it ran
with the #132 split **on** — which cuts swapped tracks apart by construction, so the sweep was blind
to its own downside. Both are fixed: the threshold sweep now runs upward from 0.8 and with the split
off, so a looser threshold that starts matching the *wrong* player shows up as kit changes.
