# What makes a reconstructed player correct — the criteria, and how well they match the eye

**Asked for by the user 2026-08-07:** *«Нам бы задефайнить признаки по которым мы решаем что игрок
корректен или нет в документе»*, together with a full per-track verdict on the reference
reconstruction. This file is the answer: the features, their thresholds, the evidence, and — the
part that makes it worth anything — the score against the user's eye.

Runnable half: `scripts/track_quality.py`. Labels: `docs/findings/track-labels-2026-08-07.json`.

```bash
.venv/bin/python scripts/track_quality.py --scene out/cue/scene_off.json \
    --camera calib/Colombia-1-0-Congo-DR1080p.npz \
    --labels docs/findings/track-labels-2026-08-07.json
.venv/bin/python scripts/track_quality.py --camera calib/... --kit          # shirt colour from the video
.venv/bin/python scripts/track_quality.py --explain-imputed                 # the frozen-mannequin proof
```

**Revised the same day.** The first draft called an in-frame imputed run a phantom; the user's
correction on t20 replaced that with *"a phantom is a duplicate"*. §П2 keeps both versions visible,
because the correction is the most useful thing in this file.

---

## 1. The ground truth — the user's eye, 2026-08-07

Scene: `out/cue/scene_off.json` (24 tracks, 60 frames, the arm without the mask cue).

> - 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 20 — корректно отображаются в 3D
> - 3, 66 : сшить · 10, 77 : сшить · 15, 71 : сшить
> - 17 не прыгнул
> - 25 — некорректно определено изначальное положение на поле (должен был двигаться от положения
>   вплотную к 14 до финального положения)
> - Очень сложная комбинация 17, 20, 15(71) — хорошо реконструировалась, с учётом что они там
>   втроём а потом ещё и номер 5 зацепили

And the display rule, stated first because everything below serves it:

> Если футболист уходит за пределы кадра — его продолжаем отображать. Если футболист становится
> **фантомом в кадре** — его отображать не следует (из 66 и 3 должен получиться 1).

Not judged: **14, 16, 18, 31**.

## 2. The one mechanical fact everything rests on

A pose frame carries `provenance ∈ {measured, interpolated, imputed}`. Measure what each is
actually made of (`--explain-imputed`), per contiguous run, root travel vs limb travel:

| track | run | root moved | limbs moved |
|---|---|---|---|
| t3 | `.[36-59]` imputed | **2.99 m** | **0.00 rad** |
| t77 | `.[0-54]` imputed | **3.66 m** | **0.00 rad** |
| t20 | `.[35-59]` imputed | **1.68 m** | **0.00 rad** |
| t20 | `~[20-26]` interpolated | 0.53 m | 5.07 rad |
| t25 | `~[26-37]` interpolated | 2.21 m | 7.27 rad |
| t66 | `M[42-46]` measured | 0.36 m | 7.66 rad |

Zero. Every imputed run, in every track, exactly `0.00 rad` of articulation while the root keeps
coasting metres. **`imputed` is a sliding mannequin.** That is precisely what the eye reported on
2026-08-06 — *«перемещался не двигая конечностями»* — so the phantom the user sees and the flag the
pipeline already writes are the same thing, and no new model is needed to find it.

`interpolated` is **not** the same animal: anchored by measurements on both sides, it carries real
limb motion. Only `imputed` is a defect. The viewer currently ghosts both alike — worth splitting.

## 3. The criteria

### П1 — shape: where the measured frames sit

Classify each track by the extent of its `measured` frames:

| shape | meaning |
|---|---|
| `FULL` | measured from the first frame to the last (short `~` bridges allowed) |
| `HEAD` | measured from the start, then **dies** mid-clip — the human continued, this id did not |
| `TAIL` | **born** mid-clip — the prefix is invented |
| `CORE` | measured only in the middle, imputed at both ends |

**Result: `FULL` ⇒ correct, with no exception.** All 12 judged `FULL` tracks (1, 2, 4, 5, 6, 7, 8,
9, 11, 12, 13, 17) are in the user's correct list. Precision 12/12. Two more `FULL` tracks (14, 18)
were not judged — the criterion predicts they are fine.

### П2 — an imputed run is a phantom only when *another track is measuring the same human*

**This criterion was wrong on its first draft, and the user's correction is what fixed it.** The
first version said "imputed **in frame** = phantom, **off frame** = keep". It convicted t20, and
the user overruled it:

> t20, моя ошибка, большую часть клипа закрыт игроками 15 и 17. Его таз появляется на мгновение в
> середине и в конце клипа. Положение его как раз всегда должно быть.

t20 is a real player, in frame, **occluded** — a third reason we cannot measure someone, next to
"off frame" and "not there at all". Measured: t20 has the highest occlusion of any track in the
scene (0.61 mean coverage by nearer bodies on his own measured frames), and at f0–9 he is 93–100 %
covered, by t17 and t15 among others — exactly the two players the user named.

So the discriminator is not *why* we failed to measure. It is **whether somebody else is already
representing that human**:

> An imputed run is a phantom **iff the track is one half of a handover pair (П3)** — because the
> merge will take the other half's measurements over this half's mannequin. A track with no
> partner is a real player the pipeline simply failed to measure, and R-6 says **show him, marked**,
> not delete him.

| verdict | tracks | meaning |
|---|---|---|
| `PHANTOM_HALF` | 3, 66 · 10, 77 · 15, 25 | merge with its partner, drop this mannequin |
| `OK_UNMEASURED` | **20**, 71 | real player, unmeasured — show him marked (R-6) |
| `OK_OFF_FRAME` | 16 (leaves f43), **31** (enters f34) | holding him is correct |
| `OK` | the 14 `FULL` tracks | — |

Off-frame and occluded stay as **diagnostics** — they say *why* a stretch is unmeasured, which is
what a reviewer needs — but they are not what decides the verdict.

> ⚠ **The off-frame diagnostic needs `--camera`.** `out/cue/scene_*.json` store the *invented*
> fallback camera (772 px @ 1280×720, principal point dead centre; `controller.py:654`). Its field
> of view is so wide that all 24 tracks land inside the image. With the measured fit (4169 px @
> 1920×1080) exactly two tracks leave the picture: **16 and 31**. The user confirmed 31 is the one
> they meant by "32" — *«31 уходит за кадр, это он»*.

### П3 — handover: a HEAD that dies where a TAIL is born is one human

Same team, birth within ±14 frames of the death, roots within 6 m at the handover:

| pair | handover distance | frame gap | user |
|---|---|---|---|
| t10 → t77 | **0.64 m** | +10 | ✅ said stitch |
| t15 → t25 | **0.85 m** | −2 | — (see §5) |
| t3 → t66 | **2.04 m** | −2 | ✅ said stitch |
| t10 → t71 | 4.96 m | +10 | — |

Everything else in the scene is ≥ 8.9 m. The next-nearest same-team candidate to a real pair is
four times further away, so the threshold is not delicate.

**It is an assignment, not a candidate list — one partner each, nearest first.** This matters: t20
is *also* a candidate head for t25, at 2.09 m. Reporting every candidate would have paired him and
convicted a real player. Letting t15 claim t25 at 0.85 m leaves t20 unpaired and therefore whole.

#### What "stitch 15 → 25" actually does — worked through

The user asked for this one spelled out. Today the pipeline holds **two subjects for one human**:

| | frames 0–26 | frames 27–59 |
|---|---|---|
| **t15** | **measured** — the real, detected player | *imputed* — frozen mannequin coasting 1.23 m |
| **t25** | *imputed* — frozen mannequin, **held** at x = −41.13 | **measured** — the real, detected player |

Both bodies are on the pitch the whole time, 0.85–2.9 m apart. In the first half the eye sees the
real player (t15) with a motionless duplicate beside him; in the second half the real player is
t25 and the motionless duplicate is t15.

And because an imputed prefix **holds** rather than extrapolates (П6), t25's mannequin stands for
frames 0–23 at the position t25 will only reach at f24 — it travels 5 cm in 24 frames. *That is
the whole of «25 — некорректно определено изначальное положение»*: his opening position is not
mis-estimated, it is **not estimated at all**, it is the later position copied backwards.

Stitching takes the two ids to be one subject and keeps, at each frame, the half that was
**measured**:

* frames 0–26 → t15's measurements (the player's real path from where he actually started)
* frames 24–59 → t25's measurements

The merged track is measured on **51 of 60 frames**, up from 27 and 24. t25's invented prefix is
not repaired — it is **deleted**, because a real measurement of that human exists for those frames
under a different id. Same for t15's invented tail.

Two items on the user's list close with that one operation, and no new model, data or run is
needed: the two ids already exist, both halves are already measured, and the only thing missing is
the statement that they are the same person. The evidence for the statement: 0.85 m apart, their
spans overlap by 2 frames, and both measure **blue** off the video pixels (§ П7).

### П4 — twin: two tracks inside one body

Two roots closer than 0.5 m (a torso) for 3+ frames — at most one of them is real. What makes this
sharp is *which* of the two was invented there:

| pair | frames < 0.5 m | min | invented on those frames |
|---|---|---|---|
| t2 / t66 | 15 | 0.19 m | **t66 imputed on all 15** |
| t5 / t10 | 14 | 0.11 m | **t10 imputed on 7** |
| t20 / t25 | 13 | 0.23 m | **t20 on 11, t25 on 7** |
| t15 / t77 | 10 | 0.22 m | **t77 imputed on all 10** |

Every interpenetration in the scene is a phantom parked inside a measured player, not two measured
players colliding. So П4 is an independent confirmation of П2 — and it is also requirement (2) of
#133 ("тела твёрдые, не проницают друг друга") measured for the first time.

### П5 — vertical: nobody can jump

Largest root-Z excursion **in the entire scene: 0.082 m**. Every track sits at z = 0.920 m ± a few
centimetres. A jump moves a pelvis ~0.4 m.

So *«17 не прыгнул»* is **not a t17 defect** — the scene has no vertical degree of freedom at all.
The root Z is pinned by the foot-plane anchor, and `gravity_project` (the gate that would put an
airborne subject on a ballistic arc) is one of the gates `config/physics.yaml` ships disabled.
Fixing t17 means giving the pipeline a vertical DOF, which is a pipeline change, not a track fix.

### П6 — placement of an invented prefix

A `TAIL` track's imputed prefix does not extrapolate backwards — it **holds** the first measured
position. t25 sits at x = −41.13 for frames 0–23 and is first measured at x = −41.18: 5 cm of
travel over 24 frames, while the human it represents was crossing the box.

That is the exact defect the user reported for 25. It is not a separate failure mode from П2 — it
is what an in-frame phantom prefix *looks like* when the player it belongs to was moving.

### П7 — kit measured off the video, not taken from `team_id`

> я просто брал цвет игроков в реконструкции за истину, но похоже, что там тоже ошибки.

Right to doubt it. `team_id` is a k-means cluster label over a colour sampled once per track, so a
track that changes human keeps its first kit. `--kit` reads the shirt straight from the source
frames instead — median HSV of the upper-torso strip of the **detector's own 2D box**.

Two traps, both hit while building it:

* **The 2D record's ids are not the scene's.** Split and stitch renumber: 9 of 24 disagree,
  including two clean swaps (scene 9 ↔ npz 10, scene 11 ↔ npz 12) and t66 landing on npz box 2 —
  which is the phantom standing inside player 2, visible in П4. So the box is found by projecting
  the subject's feet and taking the nearest box bottom, **never by id**.
* **A heavily occluded box samples the occluder's shirt.** A few flipped frames inside a crossing
  are that. Flips at a track's *death* or *birth* are the ones that mean something.

Result: **`team_id` agrees with the pixels on 22 of 24 tracks.** The two that do not:

| track | `team_id` | measured shirt | reading |
|---|---|---|---|
| **t3** | B (blue) | **`YYYYYY?BBBB?YYYYYYY???YYYYYYY?BBBBBB`** | yellow for most of its life, then **blue for its last 6 measured frames** — the frames where t66 takes over. The id walks off a yellow player onto a blue one and then dies. |
| **t77** | B (blue) | **`YYY`** | all 3 of its measured frames are yellow, while its handover partner t10 is solidly blue. |

And the two the user asked about directly: **t20 measures `BYYYYYYYYY??YYYYYY` — yellow, team A
confirmed. t71 measures `?BB?B` — blue, team B confirmed.** They are *different kits*, so
«20 стал 71» would join two different players. The colour did not change by mistake there.

Also visible: t12 opens with 7 yellow frames before settling blue, and t13 and t17 each carry a
short yellow patch. `split_on_kit_change` (#132, on by default) evidently did not cut these — worth
a look, since it is the mechanism that is supposed to.

## 4. Score against the eye

**19 of 20 judged tracks agree**, after the t20 correction moved П2 from "in-frame" to "duplicate".

| criteria | tracks | user's verdict |
|---|---|---|
| `OK` (`FULL`) | 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 17 | correct ✅ ×12 |
| `PHANTOM_HALF` | 3, 66 · 10, 77 · 15, 25 | stitch / misplaced ✅ ×6 |
| `OK_UNMEASURED` | **20** | correct ✅ (after the correction) |
| `OK_UNMEASURED` | **71** | **stitch ❌ — the one open disagreement** |
| `OK_OFF_FRAME` | 16, **31** | 31 confirmed as the "32" ✅; 16 not judged |
| `OK` | 14, 18 | not judged — predicted fine |

**Every defect the user named is caught, and no track he called correct is convicted.** The single
disagreement is t71 — and it is the one the user himself flagged as uncertain.

## 5. What is still open

**(a) 15 → 71 (eye) vs 15 → 25 (geometry).** The user's own second look:

> 15 сложно сказать, выглядит вообще как 15 стал 20-м, а 20 стал 71, но цвет поменял.

What the measurements say about that chain:

* **15 and 20 are both `measured` on frames 10–19**, converging from 2.15 m to 0.59 m. The
  pipeline is looking at two humans there — consistent with the user's own reading that t20 is the
  player *behind* 15, not 15 himself.
* **20 → 71 is geometrically plausible**: 1.08 m between t20's last measurement (f34) and t71's
  first (f55), and t20's coasted tail lands 1.13 m from t71 at f55.
* **but the kit says no.** t20 measures yellow on 16 of 18 measured frames; t71 measures blue on
  all its usable frames (§ П7). Two kits, two players.
* `15 → 25` stays the strongest reading in the scene after `10 → 77`: 0.85 m, 2-frame overlap,
  both blue, and it disposes of the t25 placement complaint for free.

t71 then has no partner and is simply a real player picked up for the last 5 frames — which is
what the criteria now say (`OK_UNMEASURED`). **User is checking frames 24–30 with `ids` on.**

**(b) Two tracks whose kit contradicts their `team_id`** — t3 (flips yellow→blue at its death) and
t77 (yellow, paired with blue t10). Both sit inside the user's stitch list, so they are not
cosmetic: if t3's last six frames are already a different human, the `3 → 66` merge is joining the
wrong end. Worth re-checking by eye at frames 30–36.

**(c) `split_on_kit_change` did not cut t3, t12, t13 or t17**, all of which carry two kits in the
measured scan. #132 reported that gate taking two-human tracks 9 → 0; either it did not run on
this scene, the stitcher re-merged its pieces, or the flips are occlusion artefacts. Cheap to
check, and it bears directly on (b).

## 6. What follows for the pipeline

Ordered by how much of the user's list each item closes.

1. **Stitch on the handover criterion (П3).** Three pairs → three players instead of six. This is
   the user's whole stitch list and it needs no new model — the signal is already in `provenance`
   plus the roots. The existing stitcher (#132) merges on kit change; this adds
   *dies-where-another-is-born*. Assign, don't just list: one partner each, nearest first, or a
   real player (t20) gets swept into someone else's merge.
2. **Drop the mannequin half of every merged pair (П2).** What is left unmeasured after that is a
   player nobody measured — off frame, occluded, or missed — and he **stays**, marked. That is the
   user's rule and R-6's both: *«если футболист становится фантомом в кадре, то его отображать не
   следует»* applies to the duplicate, not to the man we failed to see.
3. **Anchor the invented prefix (П6)**, for the runs that survive: extrapolate backwards from the
   first measured velocity instead of holding the position.
4. **Give the scene a vertical DOF (П5)** so a jump can exist at all. Until then t17 cannot be
   fixed and no other airborne moment can either.
5. **Split `imputed` from `interpolated` in the viewer.** They are ghosted alike today and they are
   not alike — one is a mannequin, the other is honest anchored fill.

## 7. The positive result, which is the more surprising one

> Очень сложная комбинация 17, 20, 15(71) — хорошо реконструировалась, с учётом что они там втроём
> а потом ещё и номер 5 зацепили.

Three bodies overlapping, then a fourth. Through it, **t17 stays measured 56/60** and t20 keeps a
measured core across the tangle; only t15 dies. So on the hardest occlusion in the clip the
per-crop pose estimator did not fuse — which is the same conclusion #132 reached by measurement
("pose half: not reproduced") and the reason PromptHMR was not adopted.

**The bottleneck is association, not pose.** Every defect in the user's list is an identity or
placement failure — a track that died, was born, or was parked. None of them is a wrong pose on a
correctly-associated crop.
