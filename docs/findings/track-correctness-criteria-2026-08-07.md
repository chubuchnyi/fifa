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
```

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

### П2 — an in-frame imputed run is a phantom; an off-frame one is not

The user's own rule, made computable. Project the root through the **measured** camera
(`calib/Colombia-1-0-Congo-DR1080p.npz`) and ask, for each imputed run ≥ 6 frames, whether it lies
inside the 1920×1080 image.

* inside → nothing hid this player; he was simply not detected, so the mannequin is fiction →
  **PHANTOM**
* outside → he left the picture; holding him is correct behaviour and R-6 *mark, never hide* →
  **OK_OFF_FRAME**

| verdict | tracks |
|---|---|
| PHANTOM (in-frame imputed) | 3, 10, 15, **20**, 25, 66, 71, 77 |
| OK_OFF_FRAME | **16** (leaves at f43), **31** (enters at f34) |

Exactly two tracks in the whole scene leave the picture, and both are in the user's *not judged*
list. The user's off-frame example — *«для 32 отображение корректное»* — names a track id that does
not exist in this scene; **31 or 16 is meant**, and both are the case being described.

> ⚠ **This test needs `--camera`.** `out/cue/scene_*.json` store the *invented* fallback camera
> (772 px @ 1280×720, principal point dead centre; `controller.py:654`). Its field of view is so
> wide that all 24 tracks land inside the image and the criterion silently becomes a constant. With
> the measured fit (4169 px @ 1920×1080) tracks 16 and 31 separate out. The script prints a warning
> when it detects the fallback.

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

## 4. Score against the eye

**19 of 20 judged tracks agree.**

| criteria | tracks | user's verdict |
|---|---|---|
| `OK` (FULL, no in-frame imputed run) | 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 17 | correct ✅ ×12 |
| `PHANTOM` | 3, 10, 15, 25, 66, 71, 77 | stitch / misplaced ✅ ×7 |
| `PHANTOM` | **20** | **correct ❌** |
| `OK_OFF_FRAME` | 16, 31 | not judged (one of them is the user's "32") |
| `OK` | 14, 18 | not judged — predicted fine |

Recall on defects is 7/7: **no track the user called broken was missed.** The single false positive
is t20.

## 5. The three things this does not settle

**(a) t20 — the one disagreement.** Its shape is `CORE`: imputed f0–9, measured f10–19,
interpolated f20–26, measured f27–34, then imputed f35–59. That is 42 of 60 frames not measured,
25 of them a frozen mannequin sliding 1.68 m in frame, and 11 frames of it standing inside t25. By
every criterion it is a phantom at the ends; the eye called it correct. Either the ends were not
noticed (they are late in the clip, f35–59), or `CORE` deserves more trust than `TAIL`/`HEAD`
because its position is anchored by real measurements on *both* sides. **Not resolved by narrowing
the criterion to fit** — the user re-checks t20 at frames 0–9 and 35–59 in `/app`, pitch panel,
`mark imputed` on.

**(b) 15 → 71 (eye) vs 15 → 25 (geometry).** The user named `15, 71`. The handover test ranks
`15 → 25` second-strongest in the scene (0.85 m, 2-frame gap) and `15 → 71` needs a **29-frame
blind gap** with no measurement of that human at all. The two readings are mutually exclusive, and
they are not equally cheap to be wrong about: if `15 → 25` is right, that single stitch *also*
fixes the 25 placement complaint, because 25's invented prefix would be replaced by 15's measured
frames 0–26. Check: `/app`, frames 24–30, `ids` on — does t15 hand over to t25 or does another
player take that box?

**(c) "32".** No such track. Tracks 16 and 31 are the only off-frame ones and both match the
description. Which one did the user mean?

## 6. What follows for the pipeline

Ordered by how much of the user's list each item closes.

1. **Stitch on the handover criterion (П3).** Three pairs → three players instead of six. This is
   the user's whole stitch list and it needs no new model — the signal is already in `provenance`
   plus the roots. The existing stitcher (#132) merges on kit change; this adds
   *dies-where-another-is-born*.
2. **Do not export a long in-frame imputed run (П2).** After stitching, whatever imputed run is
   left in frame is a mannequin the user has said should not be displayed. Off-frame runs stay —
   that half is already correct.
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
