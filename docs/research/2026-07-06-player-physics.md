# Player physics: what's already in place, what the complaints expose, what to build (2026-07-06)

A shared map for design discussion — not implementation. Structured so we can agree on scope before writing code.

**Trigger:** user report on 2026-07-06 — *"players float in the air, move unnaturally, change poses way too fast, cover the pitch too quickly, spatial orientation glitches, appear from nowhere and vanish, phase through objects, no endurance limit."*

**Format:** for every symptom — a measurable definition, what the code already covers, what remains uncovered, a proposed fix with a cost estimate and priority.

---

## 1. Decomposing the complaints into measurable limits

Grouped by the type of violation so we don't mix "already solved" with "not yet."

| # | Complaint | What it is in model terms | How to measure | Elite-football real limit |
|---|-----------|--------------------------|----------------|--------------------------|
| A | Hovering in the air | root Z above the pitch plane Z=0 | `transl[:, 2]` vs `pelvis_height_m ≈ 0.92 m` | Z should **oscillate 0.85–1.05 m** during running; a jump is a <0.5 s episode with a peak of +0.3 m; never a steady state above 1.10 m |
| B | Crossing the pitch too fast | root XY speed | `\|d(transl[:2])/dt\|` | Bolt sprint 12 m/s; elite footballer **peak 10 m/s (36 km/h)**, sustained <8 m/s |
| C | Poses change too fast | per-joint angular velocity | `\|Δ(body_pose)/dt\|` per joint | Shoulder up to ~1500°/s (baseball pitch); most joints in football **stay under 600°/s** |
| D | Spatial orientation jumps | root global_orient rate | `\|Δ(global_orient)/dt\|` | Elite body turn up to **720°/s** in short bursts; sustained turn ~360°/s |
| E | Appear/vanish | subject presence at span edges and interior gaps | `frame_conf` and span boundaries | A player is physically on the pitch from whistle to whistle — can only leave the frame, not the world |
| F | Phase through objects | player↔player and player↔ball collisions | pairwise capsule distances | Bodies don't interpenetrate (a plausible minimum of ~0.5 m between centres) |
| G | No endurance limit | long-horizon speed | fraction of time above a speed threshold | Elite players sustain a sprint **<10 s** and need ~30–60 s to recover; total high-intensity <3–5% of match time |

All seven are distinct physical quantities. Each needs its own gate/correction. A "one smart smoother for everything" (MA(5)) already exists and is structurally too weak to catch teleport-class errors (see §2).

---

## 2. What's already in place (don't reinvent)

### 2.1 M3-9 kinematic gate — `src/pitch3d/core/correction/kinematics.py`

**What it does** (surface only — you don't have to re-read the 331 lines):

* Thresholds **max_speed 10.5 m/s** and **max_accel 8.0 m/s²** (`HUMAN_MAX_SPEED`, `HUMAN_MAX_ACCEL`; env overrides `PITCH3D_KIN_MAX_SPEED` / `MAX_ACCEL`).
* Projection of the XY track onto the feasible set: velocity clamp → alternating forward/backward accel sweeps → one guaranteed-feasible final forward sweep. Both segment endpoints stay pinned to the measured positions.
* **Teleports MARKED, not erased** (R-6): a single interval whose speed >2× the limit becomes a `TeleportEvent`; the jump is preserved verbatim for identity/stitch review. One-off out-and-back spikes are demoted to clampable jitter by a velocity-reversal test.
* Consecutive teleport intervals collapse into ONE preserved region.
* Output — one dense `KEYFRAME_INTERP` correction per subject through the ADR-0002 seam (inspectable, disable-able, non-destructive).

**Real-scene result** (STATUS.md line 252): speed/accel violations 22/999 → **0/0**, 10 raw teleports → 1 marked region event (subj 1 f31, 8.7 m, n_intervals=8, conf 0.2 = coherence-extrapolated).

**Coverage scope:** root XY only. Z, joints, root orientation — untouched.

### 2.2 Coherence coast + gap-fill — `src/pitch3d/core/correction/coherence.py`

* **Gap-fill (`fill_pose_gaps`)**: interior gaps ≤ `max_fill_gap=12` frames bridged with slerp (rotations) + linear (translation), tagged `filled_confidence=0.3`.
* **Edge extension (`extend_pose_to_span`)**: subjects lost before/after the clip are extended to the full span. Posture is frozen; the root coasts with a decaying velocity, tagged `extrapolated_confidence=0.2`.
* **Coast velocity capped** at `coast_max_speed=HUMAN_MAX_SPEED=10.5 m/s` (#207 fix: without the cap, a dying track passed its 43 m/s edge velocity to the coast → 10.9 m ghost slide).
* **Auto smoothing correction** (MA(5) or gaussian): on root_translation by default; on root_orientation — off (can flatten fast turns).

### 2.3 Foot-ground anchor — `src/pitch3d/adapters/models/pose.py`

* `_ground_root` (line 256): XY from homography (bbox-bottom → world), Z from either the backend's per-frame `pelvis_above_foot` or a constant `pelvis_height_m ≈ 0.92 m`.
* `refit(constraints=…)` can apply `foot_floor` (clamp root Z ≥ floor) — but only when explicitly passed. Not on by default in the render pipeline.

### 2.4 Ball physics — DONE (#206)

Contact-anchoring: where the ball touches a foot, XY is pinned to the foot; between contacts, ballistic Z. Ball is clean (measured p95 = 16.2 m/s, 0 violations). Untouched.

### 2.5 Probe — `scripts/motion_stats.py`

Prints per-subject speed/accel/turn on raw proposal AND resolved (proposal ⊕ corrections). Run: `python scripts/motion_stats.py --scene out/anim_adr11/export/scene.json --fps 29.97`.

---

## 3. Unpacking the complaints — what's not covered

For each item in §1: **what likely happens in practice** + **why the current stack misses it**.

### A) "Hovering in the air"

**Likely cause:** the HMR backend's `pelvis_above_foot` is a near-constant ≈ 0.92 m; the gate enforces feasible XY but never checks that the foot actually touches Z=0. If the homography is slightly off (metric bias in U-V → world) or the SMPL-X shape is a bit taller than typical, the feet float 10–30 cm above the ground **continuously** — the eye reads it as "helicopter."

**Why it slips through:**
- `_ground_root` uses `bbox bottom` as the foot point, but HMR-inferred feet can sit inside the mask (boots recessed).
- `foot_floor` exists in `refit()` but isn't wired as an auto-default in `pipeline.py`.
- `motion_stats.py` never reports "fraction of frames with foot_z > ε."

**Quick self-check:** `python -c "import numpy as np; from pitch3d.core.scene.serialization import load_scene; s = load_scene(...); [print(sub.track_id, sub.proposal.pose.transl[:,2].min(), sub.proposal.pose.transl[:,2].max()) for sub in s.subjects]"` — if min/max Z hover around 0.92 without motion, the foot isn't grounded.

### B) "Moving too fast"

**Likely cause:** the M3-9 gate handles **root XY**, but:
- The threshold `HUMAN_MAX_SPEED=10.5` is a sprinter's PEAK, not a cruise. Sustained running is 6–8 m/s. Long ID-swap slides at 8–10 m/s can pass the gate and not trigger the spike-test.
- If the tracker delivers "camera drift" as apparent motion, the gate smooths it, but the eye still reads "everything shifted too fast."
- **FPS mismatch** (STATUS.md line 252): mux `FPS=25` vs source `29.97` — playback is 20% slower than real-time. If we compute at 29.97 but render at 25, motion looks **slower**, not faster. Still worth fixing for fidelity.

**Check:** `motion_stats.py --fps 29.97` before/after the gate + cross-check mux fps.

### C) "Poses change too fast"

**Likely cause:** per-joint angular velocity **is not bounded** anywhere. HMR emits per-joint axis-angle per frame; under partial occlusion, limbs jitter between frames (a classic neural-HMR failure mode).

**Why it slips through:** neither `kinematics.py` nor `coherence.py` looks at `body_pose`. The auto-smoothing correction is on `root_translation` only. Slerp is used purely for gap-fill.

**Needed:** a separate gate on per-joint angular velocity (or per-joint acceleration in the quaternion domain). First version — a **per-joint slerp clamp** capping the quaternion distance between frames (a rotation analogue of `max_speed`). Threshold ~600°/s × dt = 20° per frame at 30 fps.

### D) "Spatial orientation glitches"

**Likely cause:** `global_orient` can flip 180° between frames when HMR is uncertain about front/back. Or it resonates when player-camera geometry stays similar for a stretch of frames.

**Why it slips through:** `CoherenceConfig.smooth_root_orientation=False` by default (comment: "can over-flatten fast turns").

**Needed:** an M3-9 analogue for `global_orient` — a **max_turn_rate** gate with a two-tier threshold (720°/s for fast turns, above that = marked spike). And flip `smooth_root_orientation=True` as an auto default with a small window (3–5 frames), excluding marked spikes from the smoothing.

### E) "Appear from nowhere / vanish"

**Partially covered** by `extend_pose_to_span`: subjects are extended to the full clip span, holding posture and coasting the root. But:
- The renderer may not draw them if `subject_frame_conf < threshold` — need to audit `blender_animate.py` for any confidence-conditioned skip.
- Mid-clip appearances (new IDs) still "come from nowhere" if not stitched with pre-track continuity.
- The 1 marked teleport region reported today (subj 1 f31, 8.7 m) IS a "vanish + reappear" case. Marked but NOT filled in — R-6 (we don't want to fabricate a sprint that never happened).

**Needed:** confirm extrapolated frames actually render (no blink), plus a policy for marked teleport regions (blink vs plausible interpolation with an explicit R-6 tag).

### F) "Phase through objects"

**Not covered at all.** Players are independent SMPL-X meshes with no collision. The ball is a point (no mesh collision, only contact anchoring).

**Needed:** cheap capsule collision (one ~0.5 m radius × pelvis_height capsule per player), soft-repulsion between overlapping pairs. Not rigid-body sim — a Jacobi "pull them apart" iteration. Ball collision is harder (there are legitimate moments where a player must kick the ball); leave that to M3-2 territory.

### G) "No endurance limit"

**Not covered.** No fatigue model exists.

**Would need (low priority):** a running mean of speed on a sliding window (30 s?) — if >N% of the window is in the high-intensity zone, deprioritize the next sprint-class motion. But this is more of a *motion prior* than a *hard gate* — real players do sometimes over-exert.

**My recommendation:** defer. Of the seven complaints this one is the most cosmetic; you won't see the improvement until A–D are closed. If we do build it, make it a feature/prior, not a hard clamp.

---

## 4. Per-player and per-ball model with online-tuned characteristics

This is the piece the earlier revision didn't cover. Instead of one set of thresholds baked into `KinematicConfig`, we model each player and the ball as **stateful instances that carry their own characteristics** — starting from population defaults, then **auto-tuning in place** as we observe them.

### 4.1 Motivation

- **Per-player realism.** Cristiano's peak sprint is not the same as a goalkeeper's. A gate that uses `HUMAN_MAX_SPEED=10.5` for every subject silently over-permits keeper motion and under-permits a winger's counter-attack. Per-instance limits let the gate be tight on stationary players and loose on sprinters.
- **Continuity across clips.** A saved profile lets the same player carry his measured ceiling from one clip to the next (an M3-2 style measured-over-generative pattern).
- **Human/LLM inspection surface.** A player profile is a small human-readable object (team, jersey, position, personal peaks). It's the natural place for an operator to override a wrong estimate — the same "auto + manual override" pattern already committed in the memory rulebook.

### 4.2 Player instance — schema draft

```yaml
player_id: "COL#10"          # team code + jersey number (unique key)
team: COL
jersey: 10
position_hint: MID           # GK / DEF / MID / FWD — priors on speed/accel/turn
body:
  height_m: 1.75             # measured pelvis+trunk+head SMPL-X shape (or default)
  shape_betas: [...10]       # cached SMPL-X shape for identity persistence
kinematics:
  peak_speed_mps:      { value: 9.4, source: measured, ci: 0.6, n: 174 }
  cruise_speed_mps:    { value: 6.1, source: measured, ci: 0.4, n: 174 }
  peak_accel_mps2:     { value: 6.3, source: measured, ci: 0.8, n: 174 }
  peak_turn_rate_dps:  { value: 480, source: measured, ci: 60,  n: 174 }
  peak_joint_omega_dps:{ value: 540, source: default,  ci: null, n: 0 }
endurance:
  sprint_budget_s:     { value: 7.5, source: default, ci: null, n: 0 }
  recover_tau_s:       { value: 25,  source: default, ci: null, n: 0 }
appearance:
  shirt_hue_deg:       { value: 42.0, source: measured, ci: 3.0, n: 174 }
  shorts_hue_deg:      { value: 200,  source: measured, ci: 5.0, n: 174 }
provenance:
  first_seen_clip: "Colombia-1-0-Congo-DR1080p.mp4"
  clips_observed: 1
  last_updated: "2026-07-06T09:24:00Z"
```

Every field carries `source ∈ {default, measured, operator}` and a confidence interval so the gate — and every downstream consumer — can trust the number *and know how much*. That's the R-6 discipline: no field is "just a number."

### 4.3 Ball instance — schema draft

```yaml
ball_id: "match_ball_1"
kinematics:
  peak_speed_mps:   { value: 34.2, source: measured, ci: 1.5, n: 48 }
  typical_pass_mps: { value: 18.4, source: measured, ci: 2.1, n: 48 }
  peak_accel_mps2:  { value: 190,  source: measured, ci: 40,  n: 48 }  # a kick
physics:
  restitution:      { value: 0.85, source: default, ci: null, n: 0 }
  drag_coeff:       { value: 0.28, source: default, ci: null, n: 0 }
  spin_rps:         { value: null, source: unknown, ci: null, n: 0 }
appearance:
  radius_m:         { value: 0.11, source: default, ci: null, n: 0 }
  color_bgr:        { value: [242, 242, 242], source: measured, ci: 8, n: 48 }
provenance:
  first_seen_clip: "Colombia-1-0-Congo-DR1080p.mp4"
  clips_observed: 1
  last_updated: "2026-07-06T09:24:00Z"
```

### 4.4 Auto-tuning with filtering — the point

Naively updating characteristics from every frame breaks the gate: jitter spikes would raise the ceiling and eventually admit inhuman motion. We need **layered filtering**, in this order:

1. **Update only from resolved (post-gate) motion.** The M3-9 output is already feasible. Tuning from the *raw* proposal would train the ceiling on tracker noise.
2. **Reject outliers from the measured distribution.** Use robust estimators — an **exponentially-weighted percentile** (p95, not max) is resistant to a single bad frame. A single-frame max would inherit a residual jitter into the ceiling forever.
3. **Confidence-weight the update.** Frames with low `subject_frame_conf` (occluded / extrapolated) contribute proportionally less. An extrapolated coast frame contributes 0.2× a fully-observed frame.
4. **Require a minimum sample size before promoting a field.** Until n ≥ 30, `source` stays `default` even if the running estimate has moved — this prevents a five-frame sighting from overwriting a population prior.
5. **Guard against identity drift.** If the incoming stats jump beyond CI×3 of the running estimate, the sample is *quarantined* (probably an ID swap) — the auto-tuner logs it and doesn't update. The kinematic gate already catches these as teleports; the tuner just refuses to learn from them.
6. **Never lower a hard-limit ceiling below a floor.** Even if we observe a player who never exceeded 5 m/s, don't set the ceiling there — a substitute may sprint next minute. Ceiling floor = e.g. `population_p50 - 1σ`, keep some headroom.
7. **Operator override wins.** An `operator`-sourced field is never overwritten by the auto-tuner. Operators can override any field via the same control surface used by the correction seam (ADR-0002 pattern reused).

### 4.5 Update rule (concrete first cut)

For each numeric field with `source ∈ {default, measured}`:

```python
# once per frame (or once per resolved motion segment)
if not accept_sample(frame, subject):        # filters 3, 5 above
    return
observation = compute_peak_or_percentile(...)  # p95 EWMA, not max — filter 2
if profile.n < MIN_N:                          # filter 4
    profile.value  = (profile.value * profile.n + observation) / (profile.n + 1)
    profile.n     += 1
    profile.source = "default"                 # not promoted yet
else:
    alpha         = 1 / (1 + tau_frames)       # EWMA time constant (~500 frames)
    profile.value = (1 - alpha) * profile.value + alpha * observation
    profile.n    += 1
    profile.ci    = ewma_std(profile, observation)
    if profile.n >= MIN_N and profile.source == "default":
        profile.source = "measured"            # promoted after enough evidence
```

- `compute_peak_or_percentile` uses **p95 of the last 5 s window**, not the frame-instant. This is the ceiling that "resistant to a spike, responsive to a genuine faster episode."
- `accept_sample` gates on: subject_frame_conf ≥ 0.5 (skip extrapolated), the frame is not adjacent to a marked teleport, and the observation is within CI×3 of the running estimate (else quarantine — filter 5).
- Operator-sourced fields skip the update entirely.

### 4.6 Storage and lifecycle

- Local JSON per player: `profiles/players/<team>/<jersey>.json`; ball is `profiles/balls/<match_ball_id>.json`.
- Loaded lazily by the pipeline; the file's absence is fine — falls back to defaults.
- Written atomically after the pipeline resolves a clip (append-only within a session, snapshot at pipeline end).
- Deleting a file resets the player to defaults (useful when a jersey change reassigns a number to a new player).
- Reference to defaults: population priors table (see §4.7) shipped in the repo so first-clip runs get sensible values.

### 4.7 Population priors (defaults) — shipping today with the code

Rough numbers from published football biomechanics. Not authoritative — the priors are what a player starts with before we've seen them.

| Position | peak_speed_mps | peak_accel_mps2 | peak_turn_rate_dps | peak_joint_omega_dps | sprint_budget_s |
|----------|----------------|-----------------|--------------------|----------------------|-----------------|
| GK  | 8.0  | 5.5 | 360 | 400 | 5.0 |
| DEF | 9.0  | 6.5 | 480 | 500 | 6.5 |
| MID | 9.5  | 7.0 | 540 | 550 | 7.5 |
| FWD | 10.0 | 7.5 | 540 | 600 | 8.0 |

`HUMAN_MAX_SPEED / HUMAN_MAX_ACCEL` become the **safety ceiling** for the gate (a value we never let a player's personal ceiling exceed), not the per-subject limit. Per-subject limits come from the profile.

### 4.8 How this integrates with the gates

- `KinematicConfig` grows a `profile_provider` field — a callable `subject → PlayerProfile`. When present, per-subject limits override `max_speed` / `max_accel` etc.
- The gate emits its usual corrections and report; additionally, it emits `ProfileUpdateProposal`s. These land as *proposals* (ADR-0002 pattern): they're only applied after passing the filters in §4.4.
- Ball limits (`BALL_MAX_SPEED=36`) similarly become ball-instance-scoped.
- Auto-tuning is **off by default in tests** (deterministic runs), on by default in real pipelines.

### 4.9 Where this lives in the code

New module: `src/pitch3d/core/scene/player_profile.py` (pure, no adapters). Storage adapter: `src/pitch3d/adapters/profiles/local_json.py`. The gate imports the schema and never the adapter — the adapter is wired in `app/controller.py`.

Tests: `tests/unit/test_player_profile.py` (schema/serialization, filter policy, promotion after N samples, operator override immunity, ID-swap quarantine).

---

## 5. Proposed order of work (what to deliver first vs later)

Weighted by eye impact (does it move the video?), measurability and implementation cost.

### Tier 0 — measure before we fix anything (LANDED)

**T0** — DONE. ``scripts/motion_stats.py`` rewritten with per-category stats:

* ``foot_z_stats`` — z_min/z_max, hover_frac, below_floor_frac, plateau detection.
* ``orient_stats`` — root ``global_orient`` angular rate via the true group metric
  (``R_i+1 · R_i^-1``), not componentwise (pin test rejects the componentwise
  answer).
* ``joint_stats`` — per-joint ``body_pose`` angular rate + hottest-joint index.

All thresholds pulled from ``config/physics.yaml`` (added ``ball:`` and
``probe:`` sections, ``BallConfig`` / ``ProbeConfig``). CLI: ``--profile``,
``--config``, ``--json``. Every run prints the limits with their source so the
log is self-documenting.

**Real result on the dry-run scene** (``out/p2_3/export/scene.json``):

* Every subject Z = 0.92 constant across all frames (``z_max - z_min ≈ 0``).
* ``physics_compare.py`` reports ``plat=15`` — 15/15 subjects flagged as
  constant-Z plateaus. This IS the "hovering" complaint made measurable: the
  fake HMR path never emits ``pelvis_above_foot`` (line 42 of
  ``adapters/models/pose.py``), so ``_ground_root`` picks the nominal 0.92 m
  and stays there. A real HMR backend must emit per-frame ``pelvis_above_foot``
  for stride/crouch variation.

10 unit tests (``tests/unit/test_motion_stats.py``); 11 loader tests; 44 pre-existing
kinematic/coherence tests all still green.

### Tier 1 — cover the obvious untouched gaps (A, C, D)

**T1.a — foot-floor gate.** DONE. ``src/pitch3d/core/correction/foot_floor.py``:
sibling of the M3-9 gate. Reads ``FootFloorConfig`` from
``config/physics.yaml``; when enabled, emits ONE dense ``KEYFRAME_INTERP``
``ROOT_TRANSLATION`` correction per subject whose resolved Z sinks under
``floor_m + 0.92``. When disabled, still measures — plateau (``z_std < 0.02``),
below-floor and hover counts land in the report. Wired into
``scripts/physics_compare.py`` so profile sweeps show foot columns
(``blwFl / plat / hovr / ffFix``).

10 unit tests (``tests/unit/test_foot_floor.py``): disabled = measure-only,
enabled = clamps below-floor rows, plateau flagged only on constant Z,
idempotent, subject-isolated, empty-scene safe. Broke config→correction
circular import by moving pure config dataclasses to
``src/pitch3d/core/config/gates.py``.

*Cost: 1 iteration, $0. Verified on ``out/p2_3/export/scene.json``.*

**T1.b — per-joint angular gate.** DONE.
``src/pitch3d/core/correction/joint_kinematics.py``: sibling of the M3-9 gate on
``body_pose`` (T, K, 3). For each joint independently: forward slerp sweep in
rotation space — every ``t → t+1`` whose angular rate exceeds
``JointKinematicConfig.max_omega_dps`` (default 600°/s) becomes
``slerp(R_t, R_t+1, max·dt / actual_angle)``, keeping direction, capping speed.
Emits ONE ``KEYFRAME_INTERP`` ``POSE_BODY_JOINT`` correction per (subject, joint)
that actually needed clamping — untouched joints stay silent. Group-metric
``|angle(R_b, R_a)|`` via quaternion delta (test pins componentwise as WRONG).

11 unit tests: measure-only (disabled), clamps below limit, untouched joints
skipped, within-limits zero corrections, idempotent, per-track/per-joint
violations, empty/bad-fps safety.

**T1.c — root orientation gate.** DONE.
``src/pitch3d/core/correction/orientation.py``: same forward-sweep design but on
the scalar ``global_orient`` (T, 3). Emits one ``KEYFRAME_INTERP``
``ROOT_ORIENTATION`` correction per subject.
``OrientationConfig.max_turn_rate_dps`` default 720°/s (comes from
``config/physics.yaml``, never a hidden constant). ``smooth_root_orientation``
stays off — this gate replaces the naive smoother it always avoided.

7 unit tests mirror the joint gate.

Both gates wired into ``scripts/physics_compare.py``. Profile sweep now shows
``jOver`` / ``jFix`` / ``jMaxA`` and ``oOver`` / ``oFix`` / ``oMaxA`` columns
alongside XY + foot columns.

*Cost: 2 iterations combined, $0. Verified on ``out/p2_3/export/scene.json``:
harness runs clean; jOver/oOver are 0 because the dry-run pose is all zeros —
infrastructure waits for the real HMR run to fire.*

### Tier 2 — presence / persistence hardening (E)

**T2.a — rendering audit.** DONE. ``scripts/blender_animate.py`` hides a
subject only when the target frame has no row (line 612 ``hide_render=True``).
Coherence's ``extend_pose_to_span`` densifies every subject to the full clip
span, producing rows for every extrapolated frame — so coast-extended
subjects DO render (no blink). ``subject_frame_conf`` is not consulted by the
renderer at all; only by the overlay/attention debug views. The real
"appear/vanish" complaint traces to teleport regions: M3-9 preserves them
verbatim under the R-6 rule, so the resolved XY jumps between the pre- and
post-swap positions. That's the gap ``T2.b`` closes.

**T2.b — teleport policy.** DONE.
``KinematicConfig.teleport_policy: "hold" | "interpolate"`` (rejects other
values at construction). ``"hold"`` (default, R-6 strict) keeps the jump
exactly as measured. ``"interpolate"`` clamps the whole track as one anchored
segment — the velocity/accel sweeps produce a linear XY path across the
region, no invented sprint. In both modes the ``TeleportEvent`` is still
recorded on ``KinematicReport.teleports`` so the audit trail is preserved.
Interpolated rows get stamped in ``scene.confidence.subject_frame_conf`` at
``TELEPORT_INTERPOLATED_CONF = 0.15`` — below coherence's ``0.2`` extrapolated
confidence, so the attention list separates "smoothed across an ID swap" from
"coast-extended past the tracker."

Config: ``teleport_policy`` added to ``config/physics.yaml`` ``base:``; new
named profile ``humanize_teleports`` overrides to ``interpolate``. Lineage
records the source (``base`` vs ``profile:humanize_teleports``). CLI unchanged
— select via ``--physics-profile humanize_teleports``.

8 new unit tests (``tests/unit/test_teleport_policy.py``): invalid value
rejected, hold preserves the jump, interpolate smooths it and stamps
low-conf, TeleportEvent survives interpolation, YAML profile roundtrip,
default stays hold, no teleport → no interpolation regardless of policy.

Full suite: 768 passed, 12 skipped, 0 failures. **Tier 2 closed.**

*Cost: 1 iteration, $0. Verified on ``out/p2_3/export/scene.json``.*

### Tier 3 — collision (F) — DONE

**T3.a — capsule collision.** ``src/pitch3d/core/correction/collision.py``:
per-frame capsule soft-repulsion, ADR-0002 clean. Subjects modelled as
vertical capsules of radius ``CollisionConfig.capsule_radius_m=0.35`` on the
pitch plane; overlapping pairs get a Jacobi split push (each contributes
``strength · overlap / 2`` along the separation axis). ``n_passes=4``
iterations per frame converge stacks-of-three; ``max_push_per_frame_m=0.30``
caps each subject's per-frame net displacement so a tight blob never
launches anyone across the pitch.

Emits ONE ``KEYFRAME_INTERP`` ``ROOT_TRANSLATION`` correction per subject
whose max deviation exceeds ``min_correction_m=1e-4``. Z untouched — foot
floor is a separate gate. Report surfaces ``frames_with_overlap``,
``pairs_resolved``, ``max_overlap_before_m``, ``subjects_moved``,
``max_push_m``.

Config: ``collision:`` section added to ``config/physics.yaml`` base
(disabled by default); ``future_full`` profile activates it. Lineage
records the source.

13 unit tests (``tests/unit/test_collision.py``): Jacobi pass math on two /
none / coincident pairs, resolve_frame convergence, per-frame push cap,
disabled measure-only, enabled push, far-apart untouched, Z preserved,
disjoint frame ranges no false overlap, idempotent, empty scene, none-cfg.

Wired into ``scripts/physics_compare.py`` — profile sweep shows
``colFr / colPr / colMv / colOv`` columns. Verified on
``out/p2_3/export/scene.json``: 6 overlap frames, 29 pairs, max overlap
0.54 m across all profiles. ``future_full`` (collision enabled) moves 8
subjects and introduces +22 accel violations — the honest tradeoff (a
single-frame push is a step-function velocity change; a proper
compose-order or accel-aware push is future work).

Full suite: 781 passed, 12 skipped, 0 failures. **Tier 3 closed.**

*Cost: 1 iteration, $0.*

### Tier 4 — per-player + per-ball profile (from §4)

**T4.a — schema + local-JSON adapter + auto-tune policy.** DONE.

* ``config/player_priors.yaml`` — population priors per position (GK/DEF/MID/FWD/UNKNOWN)
  + shared body + auto-tune policy knobs + ball defaults. Parametric, versioned, editable.
* ``src/pitch3d/core/scene/player_profile.py`` — pure schema:
  ``PlayerProfile``, ``BallProfile``, ``ProfileField(value, source, ci, n)``,
  ``ProfileSource ∈ {DEFAULT, MEASURED, OPERATOR}``, ``Position`` enum,
  ``AutoTunePolicy``, ``PopulationPriors``.
  * ``default_player_profile(team, jersey, position, priors)`` — seed from
    priors, every field ``source=DEFAULT``, ``n=0``.
  * ``default_ball_profile(ball_id, priors)`` — same for the ball.
  * ``update_field(field, observation, confidence, policy, default_value)``
    — the seven-layer filter from §4.4 (operator lock, low-confidence skip,
    quarantine outside CI×mult, arithmetic mean up to ``min_promote_n`` then
    EWMA promotion, ceiling floor at ``ceiling_floor_mult × default``).
    Returns ``(new_field, UpdateOutcome)`` — the outcome enum is the audit
    trail (``APPLIED / QUARANTINED / LOW_CONFIDENCE / OPERATOR_LOCKED``).
  * ``set_operator_field(field, value)`` — human override, immutable
    from that call onward.
* ``src/pitch3d/adapters/profiles/local_json.py`` —
  ``LocalJsonPlayerStore(root)`` implements the ``ProfileStore`` protocol.
  Layout: ``root/players/<team>/<jersey>.json``, ``root/balls/<ball_id>.json``.
  Atomic write via ``.tmp + os.replace``. Missing file → ``load_player``
  returns ``None`` (caller falls back to ``default_player_profile``).
  ``delete_player`` returns ``True``/``False``; ``list_players`` walks the
  tree.

**20 unit tests**: priors load, position-driven defaults, roundtrip storage
preserves source+ci+n, operator lock immunity, low-confidence skip,
quarantine outside CI×mult, promotion at N, ceiling floor, EWMA reacts
slowly, confidence weighting, atomic write leaves no ``.tmp``, delete
idempotent, list across teams, path-separator sanitised. Full suite:
726 passed, 12 skipped, 0 failures.

**T4.b — wire into the M3-9 gate + CLI end-to-end.** DONE.

Core (previously shipped): ``kinematic_gate(...profile_provider=…)`` — per-subject
``peak_speed_mps`` / ``peak_accel_mps2`` from the ``PlayerProfile`` override
the shared ``KinematicConfig`` limits. Gate emits ``ProfileUpdateProposal`` on
``KinematicReport.profile_updates``. ``apply_profile_updates(...)`` applies each
through :func:`update_field` at the persistence seam.

**CLI end-to-end wiring (this iteration):**

* ``pitch3d.app.controller.run_reconstruction`` now accepts
  ``profile_provider`` + ``auto_tune_sink``. When the gate runs, the provider
  is forwarded; the sink is called ``sink(scene, report)`` after the gate so
  the caller decides how to persist.
* CLI flags: ``--player-profiles-dir DIR`` (activates the provider),
  ``--player-priors PATH`` (alternate YAML), ``--auto-tune`` (activates the
  sink), ``--ball-id ID`` (keys ``domain="ball"`` proposals).
* When active the CLI builds a ``LocalJsonPlayerStore``, a provider that falls
  back to ``default_player_profile`` from priors, and a sink that calls
  ``apply_profile_updates`` with the scene's per-subject
  ``(team_id, jersey_number, Position.UNKNOWN)`` lookup. The run prints
  ``== profiles: dir=…`` and ``== auto-tune: {applied, quarantined, …}
  (N proposal(s))`` lines for operator inspection.

**End-to-end verification.** Smoke-run:

```
pitch3d --out-dir /tmp/out --physics --physics-profile default \
        --player-profiles-dir /tmp/profiles --auto-tune \
        --pose fake --detector fake ...
```

Landed 4 subject profiles under ``/tmp/profiles/players/{UNK,A,B}/``, each
with ``kinematics.peak_speed_mps`` and ``peak_accel_mps2`` fields carrying
their audit trail (``source``, ``ci``, ``n``). Re-running with the same
profiles-dir warm-starts each subject from disk (measured/EWMA continues from
where it left off).

2 new end-to-end tests (``tests/e2e/test_cli_auto_tune.py``): auto-tune
persists profiles and prints the audit line; no-auto-tune does not write.

Full suite: 760 passed, 12 skipped, 0 failures.

Small robustness fix as a side-effect: the quarantine filter now requires
CI > 1e-6 so a run of identical observations (ci collapses to 0) doesn't
trip on 2e-15 floating-point drift. Documented and covered.

8 new unit tests (``tests/unit/test_kinematics_profile.py``): backwards
compat with no provider, per-subject ceiling swap, ``profile_updates``
emitted from clamped motion, ``apply_profile_updates`` promotes measured
speed after N samples, operator lock still immune, missing player creates
default from priors, unknown track/field skipped safely.

Full suite: 734 passed, 12 skipped, 0 failures.

**T4.c — ball profile wiring.** DONE.
The ball doesn't go through a M3-9-style clamp gate (contact-anchored per
#206), so proposals come directly from the resolved ball motion. Helper
``emit_ball_proposals(ball_track_id, frames, positions_3d, fps, …)`` returns
``ProfileUpdateProposal`` for ``peak_speed_mps`` and ``peak_accel_mps2`` from
p95 of the diffs. ``apply_profile_updates(...ball_id_lookup=…)`` routes
``domain="ball"`` proposals to ``store.load_ball`` / ``store.save_ball``, so
one batch can carry mixed player + ball updates.

8 new unit tests (``tests/unit/test_ball_profile.py``): p95 from linear
motion, short-track safe, roundtrip, default seeding on missing, operator
lock, unknown ball id skipped, unknown field skipped, mixed batch routing.

Full suite: 742 passed, 12 skipped, 0 failures. **T4 is done.**

### Tier 5 — fatigue (G)

Defer. Only revisit after A–F+profile deliver eye-visible gains.

---

## 6. Parametric-first: config file + comparison harness (LANDED)

Every physics threshold now lives in ``config/physics.yaml`` as parametric data,
not as a Python constant. Loader records field lineage (base → profile → env →
override) so we always know where a number came from. A named-profile system
plus a comparison harness let us sweep approaches on the same scene and pick
the winner by measurement, not opinion.

* ``config/physics.yaml`` — base defaults + named profiles
  (``default``, ``conservative``, ``strict``, ``no_smoothing``, ``future_full``).
  Schema keys reserved for future gates (``foot_floor``, ``joint``,
  ``orientation``) even though those gates aren't built yet — no churn later.
* ``src/pitch3d/core/config/physics.py`` — pure loader:
  ``load_physics_config(path?, profile="default", env?, overrides?) → PhysicsConfig``.
  Precedence base → profile → env → override, per-field lineage. 11 unit tests
  in ``tests/unit/test_physics_config.py``.
* ``scripts/physics_compare.py`` — research harness:

  ```
  python scripts/physics_compare.py --scene out/…/scene.json
  python scripts/physics_compare.py --scene <s> --profiles default,strict --show-lineage
  python scripts/physics_compare.py --scene <s> --json > exp.json
  ```

  Runs the M3-9 gate (and coherence, unless ``--no-coherence``) for each
  profile, prints a side-by-side table (speed/accel violations, teleports,
  max_dev, corrections added), optionally the lineage tree.
* CLI: ``pitch3d --physics --physics-profile <name>`` (or
  ``--physics-config <path>``); env vars still work as the ops override
  channel. Every dry-run prints a ``== physics config:`` line so the run log
  self-documents which numbers went in.

**Consequence for the rest of this doc:** any concrete number we discuss
(``max_speed=10.5``, ``max_omega_dps=600``, per-position priors) is a **YAML
proposal**, not a code change. Adding a new gate is: (a) reserve a section
in ``base:``, (b) ship it disabled, (c) enable it under a named profile.

---

## 7. References

- `src/pitch3d/core/correction/kinematics.py` — M3-9 gate (XY only).
- `src/pitch3d/core/correction/coherence.py` — gap-fill + edge extend + coast cap + auto smoothing.
- `src/pitch3d/core/correction/engine.py` — `resolve_subject_motion` (composes proposal ⊕ corrections).
- `src/pitch3d/adapters/models/pose.py::_ground_root` (line 256), `refit(foot_floor=…)` (line 231).
- `src/pitch3d/core/orchestration/ball_lift.py` — contact-anchored ball (#206 closed).
- `scripts/motion_stats.py` — probe.
- `docs/STATUS.md` line 252 (#207 table) — the fullest history of the topic.
- `docs/roadmap.md` line 500 — M3-9 roadmap card.

---

## 8. Mini-plan (if the priorities are acceptable)

1. **Now:** extend `motion_stats.py` (T0), run on the current scene, come back with a factual violation table.
2. **Then:** T1.a foot floor auto-default (fastest eye-visible win).
3. **Then:** T1.b joint angular gate (closes "poses change too fast").
4. **Then:** T1.c orientation gate (closes "spatial orientation").
5. **Then:** T2 presence audit + teleport policy.
6. **Then:** T4 per-player + per-ball profile (schema + storage + wire into the gate + auto-tune with §4.4 filters).
7. **After:** collision, fatigue (only if still perceptible).

Ready to start with T0 if the priorities are acceptable, or reshape them if you'd like.
