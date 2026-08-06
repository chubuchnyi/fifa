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
