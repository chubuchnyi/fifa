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

### Tier 0 — measure before we fix anything

**T0.** Extend `scripts/motion_stats.py`: add `foot_z_stats` (min/max/fraction z>ε), `joint_omega_stats` (per-joint angular velocity), `turn_rate_stats` (`global_orient` rate).

*Cost: 1 iteration, local, $0.*

**Output:** an actual "violations per category" table on the current scene. Without this we're fixing by feel.

### Tier 1 — cover the obvious untouched gaps (A, C, D)

**T1.a — foot floor as an auto-default.** Wire `foot_floor=0.0` into the default constraint set for the render pipeline (currently only if explicitly passed). Add a sanity gate "foot_z > 0.30 → warn" (something's wrong with shape/homography).

*Cost: 1 iteration, $0. Verify on the existing scene.*

**T1.b — per-joint angular gate.** New module `core/correction/joint_kinematics.py`: `max_omega_deg_per_s` (first version = 600°/s per joint), quaternion-slerp clamp, marked spikes with R-6. One `KEYFRAME_INTERP` per joint per subject through ADR-0002. Tests follow the M3-9 pattern.

*Cost: 2–3 iterations, local $0, pod E2E $0.10 for eye-verify.*

**T1.c — root orientation gate.** Extend `KinematicConfig` with `max_turn_rate_deg_per_s=720`; flip `smooth_root_orientation=True`, excluding marked spikes.

*Cost: 1–2 iterations, $0.*

### Tier 2 — presence / persistence hardening (E)

**T2.a — rendering audit.** Confirm extrapolated frames actually render (no blink). Search `blender_animate.py` for any condition on `subject_frame_conf`.

**T2.b — marked-teleport interpolation policy.** Option `teleport_policy=hold|interpolate|flash` — default `hold` (no invented motion), with a flag for smooth `interpolate` (marked R-6).

*Cost: T2.a — 1 iteration; T2.b — 1–2 iterations. $0.*

### Tier 3 — collision (F)

**T3.a — capsule collision.** Post-process over resolved motion: overlapping capsules → soft push apart. Not a physics sim — one Jacobi iteration.

*Cost: 3–5 iterations (defaults need tuning), local $0, pod $0.10.*

### Tier 4 — per-player + per-ball profile (from §4)

**T4.a — schema + local-JSON adapter.** Pure schema (`PlayerProfile`, `BallProfile`), a local-JSON store, promotion/filter policy, unit tests.

*Cost: 2 iterations, $0.*

**T4.b — wire into the M3-9 gate.** `profile_provider` on `KinematicConfig`; per-subject ceilings; auto-tuning as `ProfileUpdateProposal`s applied after §4.4 filtering.

*Cost: 2 iterations, $0 unit; pod E2E $0.10 to check per-player realism (goalkeeper vs winger different ceilings).*

**T4.c — ball profile.** Same pattern for the ball; ball_id policy per match.

*Cost: 1–2 iterations, $0.*

### Tier 5 — fatigue (G)

Defer. Only revisit after A–F+profile deliver eye-visible gains.

---

## 6. Open questions to settle before coding

1. **Foot-floor policy:** hard clamp to Z=0 (never sink) vs soft attractor (blend) vs adaptive (foot_z<ε → free, foot_z>ε → clamp)? Preserving jumps matters.
2. **Per-joint gate — on raw HMR or on post-coherence?** Coherence has already smoothed via MA(5); a second smoother risks flattening a real fast motion (a kick, a header).
3. **Teleport policy:** given the complaint about "appearing from nowhere," should the default switch from `hold` to `interpolate` (marked R-6)? Or would that erase legitimate ID-swap signal for stitch review?
4. **FPS mismatch:** fix the mux fps at the source (29.97) as the primary perception fix, or keep 25 as a friendly compatible number?
5. **Collision player↔player vs player↔ball:** ball-touch is already used in `ball_lift.py` for anchoring. Extend to full collision, or leave ball-touch as-is and only add player↔player?
6. **Ceiling default:** 10.5 m/s is an elite peak. This is Colombia vs DR Congo — not Olympians. Drop the default to 9.0 m/s (env override already exists)? Note this becomes moot once §4 lands — the per-player profile takes over.
7. **Profile auto-tune ON by default?** Auto-tuning creates a state file that changes across runs. Convenient for realism, less convenient for reproducibility. Recommendation: on for delivery pipeline, off for tests / synthetic replay.
8. **Profile identity key.** `team + jersey` works within a match, but jerseys are reassigned across matches. Extend the key with `season + competition` for cross-clip persistence?
9. **Ball profile per match or per ball model?** A tournament uses one ball model; individual balls can be swapped. Do we key on the match, on the ball SKU, or one profile per active ball ID?

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
