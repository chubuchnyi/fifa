# What our quality metrics measure, and what the eye measures — they are not the same thing

**2026-08-13.** Written because a BoT-SORT/ByteTrack A/B produced the cleanest disagreement we have
had between the numbers and the user's eye: **13 of 14 metrics prefer arm B, several by 30–50 %,
and the eye's verdict is «равносильно херовые» — equally bad.** That is not a close call to
adjudicate. It means the metrics are measuring something correlated with each other and not with
what makes a scene watchable.

Scenes: `out/pod_ab/scene_{a,b}.json`, pod run 2026-08-12, 236 frames, all five real backends
(RF-DETR · ByteTrack|BoT-SORT · PnLCalib · SMPLest-X-H · WASB), `--coherence`, calibration
**236/236 measured in both arms**, so the association step is the only difference.

---

## 1. Every metric we have, on both arms

| metric | A ByteTrack | B BoT-SORT+GMC | who it prefers |
|---|---|---|---|
| subjects | 39 | 39 | — |
| subject-frames | 9204 | 9204 | — |
| **measured** | 4059 (44.1 %) | 4161 (**45.2 %**) | B, by 1.1 pp |
| **imputed** | 4980 (54.1 %) | 4850 (**52.7 %**) | B, by 1.4 pp |
| interpolated | 165 (1.8 %) | 193 (2.1 %) | B |
| `PHANTOM_HALF` (#135) | 10 | **6** | B, −40 % |
| `OK` clean track (#135) | 6 | **9** | B, +50 % |
| `OK_OFF_FRAME` | 18 | 20 | B |
| `OK_UNMEASURED` | 5 | 4 | B |
| teleport intervals (`motion_stats`) | 13 | **6** | B, −54 % |
| speed violations | 70 | 66 | B |
| accel violations | 3741 | 3683 | B |
| joint violation samples | 652 | 640 | B |
| orientation violations | 53 | 61 | **A** |
| coherence: gap frames bridged | 165 | 193 | B |
| coherence: edge frames extended | 4980 / 33 subj | 4850 / 30 subj | B |
| raw tracklets (CPU, cached dets) | 47 | 43 | B |
| mid-pitch identity events (CPU) | 48 | **35** | B, −27 % |

Reproduce: `scripts/track_quality.py --scene … --camera calib/Colombia-1-0-Congo-DR1080p.npz`,
`scripts/motion_stats.py --scene …`, `scripts/bench_association.py`.

## 2. What the eye said

The user watched both scenes in `/world`, track by track. Verbatim verdict: **«в целом сложно
судить что лучше 8010 или 8011, они равносильно херовые, вопрос в том как с их дефектами
работать.»** The specific observations, which are the useful part:

| # | observation | arm |
|---|---|---|
| E1 | t5 correct to f39; after crossing the referee t3 it becomes t62, and **t62 is born at t3's position, not t5's**, then jumps left back onto t5's trajectory | A |
| E2 | same crossing, same frame: t5 → t184, and «184 лишний, родился в момент пересечения 5 и 3» | B |
| E3 | **the prescription:** «если 184 и 183 убрать, а их позы отдать 5 и 3, а траектории 5 и 3 оставить, то будет идеально» — drop the fragment, keep the parent's trajectory, take the fragment's *pose* | B |
| E4 | t16/t18/t19 are a crossing group from f0; correct to f26; then t18 is marked phantom at f27–28 **while its pose still changes about correctly** | A |
| E5 | t19 leaves the ground at f46 and heads the ball at f55; in 3D there is no jump — he is «переместило чуть вглубь сцены» instead | A |
| E6 | f38: t19 and t16 swap places; t18 after f35 moves far from its group and later swaps with t10 | B |

## 3. Three defects, decomposed by measurement

### 3.1 The crossing birth is misallocation, not missing evidence — E1/E2 confirmed exactly

In **both** arms the spurious track is born at **f33**:

| | first measured frame | distance to t5 | distance to t3 |
|---|---|---|---|
| A, t62 | f33 | 1.96 m | **0.08 m** |
| B, t184 | f33 | 1.93 m | **0.07 m** |

and it stays glued to t3 (0.05–0.11 m) for its whole measured life, f33–f45 — **while t3's own rows
on those frames are `interpolated`.** So t3 *is* being measured through the crossing. The
measurement is simply wearing a new id, and t3 coasts beside it.

This is W5's 2026-08-10 measurement seen from the outside: *"our mid-pitch births are contested,
not orphaned — the right track was available and the assignment gave it to a competitor, which puts
the defect in allocation, not in evidence."* The eye has now independently described the same
mechanism, in the same clip, at the same frame. **BoT-SORT does not touch it** — same frame, same
0.07 m, in both arms. That alone explains «равносильно».

E3 is the right fix stated precisely: the fragment is not a second human, it is the parent's
measurements under a new id.

### 3.2 The missing jump is NOT an identity failure — and this refutes the obvious reading

The tempting story was "t19 was lost during the jump, so the jump went to another id". Measured, it
is false. In arm B **t19 is `measured` on every frame f44–f57**, straight through the header:

```
  f44 z=0.749 → f49 z=0.832 → f52 z=0.837 (peak) → f57 z=0.828
```

**Peak root-Z is 0.837 m, about 9 cm above his own baseline. A real jump moves a pelvis ~40 cm.**
So the header was measured and the measurement contains no jump: the vertical excursion is ~4×
too small and the rest of the motion appears as depth travel — exactly what the eye reported as
"moved slightly deeper into the scene". (In arm A the same human is measured under t16, with the
same 0.83 m ceiling.)

This is a **pose/grounding** defect — monocular height/depth ambiguity — and no tracker change can
reach it. It is also a correction to how #135's П5 gets quoted: the scene's largest root-Z
excursion is 0.463 m, so the scene *can* represent vertical motion; this human's jump was measured
flat.

### 3.3 Over half the scene is not measured, and that is what the eye is reacting to

**54.1 % (A) and 52.7 % (B) of subject-frames are `imputed`** — frozen limbs on a coasting root.
B's whole improvement is **1.4 percentage points** of that. Every other metric in §1 is a
derivative of this one number, computed per track instead of per frame, which is why they all move
together and why they all overstate a 1.4 pp change as "−40 %" or "−54 %".

Stated plainly: **both scenes are about half invented, and the A/B moved that by one and a half
points.** «Равносильно херовые» is the correct reading of the data. The metrics were not wrong,
they were reported without their denominator.

## 4. What a criterion has to do differently

Not a proposal to build yet — the constraints the evidence imposes.

1. **Per human per frame, not per track.** Every metric in §1 counts tracks or violations. The eye
   follows one человек and asks two questions: is this still him, and is he doing what he did.
   A defect that is fatal to the eye (a fragment born on the wrong player) costs ~nothing in a
   per-track aggregate.
2. **Separate "not measured" from "measured wrong".** §3.1 and §3.2 are opposite failures that our
   vocabulary spells the same way. In 3.1 the evidence exists and is misfiled; in 3.2 the evidence
   exists and is flat. `imputed` covers neither honestly.
3. **`imputed` is too coarse to be a verdict.** E4: the eye saw a phantom flag on a track whose pose
   was still changing about right. #135 defines an imputed frame as a frozen mannequin; that
   equivalence needs re-measuring, not re-quoting.
4. **Report the denominator.** A metric that improves 40 % of a quantity that is 6 % of the scene
   must say so, or it will keep buying decisions it cannot support — as it did here, twice, in this
   session alone.
5. **Score the criterion against labelled frames.** §2 is the first labelled set we have: six named
   defects with frame numbers and ids, on two scenes, from the ground truth. A criterion that does
   not rank these two scenes as roughly equal is not yet a criterion.

## 5. Status of the A/B that produced this

**BoT-SORT is not adopted as the default.** It measurably reduces identity churn (§1) and
measurably does not change what the eye sees (§2). It stays available behind
`--tracker-backend pitch3d.adapters.models.botsort_backend:make`, and the honest summary of it is
"1.4 pp more of the scene is measured".
