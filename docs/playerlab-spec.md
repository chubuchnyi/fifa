# playerlab — founding spec

Lives here until the repository is cut, then moves with it and this file becomes a pointer, the way
[`camlab-spec.md`](camlab-spec.md) is one. Written against **ADR-0013**'s checklist for cutting a
lab — item 1 is "state the question in one sentence, and the check that falsifies it from the clip
alone", and nothing should be cut before that exists.

---

## 1. The question

**Which human is this, and where exactly does he stand — on every frame of the clip.**

One human gets one id from his first frame to his last, and the body drawn for him lands on his own
pixels. Not "how many tracks", not "how many phantoms" — those are the metrics that failed on
2026-08-13 (§4).

## 2. The check that falsifies it from the clip alone

camlab's bar is *project the pitch through the camera and the lines must land on the paint*: no
scene, no poses, no labels, and it can be wrong and find out by itself. playerlab needs the same
property. Three checks, ordered by how mechanical they are — the first two need **nothing but the
scene**, the third needs camlab's camera and the raw frames.

### C1 — no human may wear two ids (label-free, built)

Two ids that are never `measured` on the same frame, stay within **0.15 m**, and alternate carrying
the detection are one human: there is only ever one detection to give, and two standing people's
pelvises are not 15 cm apart. `scripts/find_duplicate_tracks.py`, moves to playerlab as-is.

The unit is the **(pair, interval)**, not the pair — t3 and t62 are one human for 14 frames and
genuinely two for the next hundred, and a verdict on the pair as a whole erases the defect.

### C2 — the timeline of one human is continuous and mostly measured

Per human, not per scene: how many ids carried him, how many frames of his own span were
`measured` rather than coasted, and the longest run he went without a measurement.

### C3 — the body lands on its own pixels (needs camlab, not built)

**This is the check that makes the other two mean something, and it is the direct analogue of
camlab's 2 px.** Reproject the posed body through camlab's camera and compare with the frame:

* foot/root point against the bottom-centre of the detection box that fed that frame;
* silhouette overlap against the person mask.

It is falsifiable from the clip alone — camera and detections are both artefacts, no labels — and
until camlab's 2 px camera existed it could not be computed at all, because camera error swamped
placement error (ADR-0013 §2).

**Nothing in playerlab is believed before C3 runs.** C1 and C2 are self-consistency: they can be
satisfied by a scene that is smooth, continuous, and entirely wrong.

## 3. Where we stand today — the baseline to beat

Measured 2026-08-12/13 on `out/pod_ab/scene_{a,b}.json`, 236 frames of the broadcast clip, all five
real backends, the two arms differing only in the association step
([`findings/quality-criterion-2026-08-13.md`](findings/quality-criterion-2026-08-13.md)):

| | A ByteTrack | B BoT-SORT+GMC |
|---|---|---|
| ids | 39 | 39 |
| **distinct humans (C1)** | **33** | **35** |
| humans smeared over several ids | 3 | 1 |
| **the referee t3 is split across** | **5 ids** | **5 ids** |
| **measured share of subject-frames (C2)** | **44.1 %** | **45.2 %** |
| imputed | 54.1 % | 52.7 % |
| C3 | **never computed** | **never computed** |

**The eye's verdict on both: «равносильно херовые».** Thirteen of fourteen existing metrics preferred
B; the eye saw no difference. That disagreement is why this lab exists.

## 4. Why the existing metrics did not work

Recorded so playerlab does not rebuild them.

1. **They were reported without their denominator.** B's entire advantage is **1.4 percentage
   points** of the imputed share. "Phantom halves −40 %", "teleports −54 %", "identity events −27 %"
   are that same 1.4 pp counted per track instead of per frame.
2. **They count tracks; the eye follows one human.** A fragment born on the wrong player is fatal to
   a viewer and costs almost nothing in a per-track aggregate.
3. **`imputed` is one word for two opposite failures.** In the crossing defect the evidence exists
   and is filed under the wrong id; in the flat jump the evidence exists and is wrong. Both print
   the same label.
4. **The unit was the track, not the interval.** Every criterion judged a track over its whole life
   and averaged transitions away — while the eye describes exactly transitions: "correct until
   frame 39, then it became t62".

## 5. Inherited claims — re-verify before use

camlab's rule, and the reason it is here: *nothing inherited is true until re-measured*. Six
inherited claims died in camlab's first sessions. This is playerlab's opening register.

| claim | source | status |
|---|---|---|
| **"Every defect the user listed is association or placement — none is a wrong pose on a correct crop"** | #135 | ⚠ **REFUTED 2026-08-13.** t19 is `measured` on every frame f44–57 straight through a header, on a crop that was fine, and the pose is flat: peak root-Z **0.837 m, ~9 cm above his own baseline against a real jump's ~40 cm**. A wrong pose on a correct crop. **ADR-0013 §2 rests on this claim** when it calls pose-on-a-good-crop "a black box we call". It is not a black box; it has a measured failure mode. See §7. |
| "an imputed frame is a frozen mannequin — exactly 0.00 rad of limb motion" | #135 | ⚠ **suspect.** The eye saw a phantom flag at f27–28 on a track whose pose was still changing about right. Re-measure limb motion across imputed spans before quoting it. |
| "mid-pitch births are contested, not orphaned — the defect is in allocation, not evidence" | W5, 2026-08-10 | ✅ **independently confirmed 2026-08-13** from the outside: the spurious track is born 0.07–0.08 m from the referee while the referee's own rows are `interpolated`. Same frame, same distance, in both arms. |
| "the largest root-Z excursion in the whole scene is 0.082 m, so a missing jump is a scene-wide missing DOF" | #135 П5 | ⚠ **superseded.** On these scenes the largest excursion is 0.463 m — the scene *can* represent vertical motion, so a flat jump is that human's defect, not the scene's. |
| detector resolution 896 beats 560 and 1512 | W1 | not re-checked; playerlab consumes cached detections, so it inherits whatever produced them. Record which. |
| "a real root ranges 0.23 m per clip" | W9, WorldPose GT | usable — it is GT, not ours. The vertical bound for C3 should come from here. |

## 6. Artefact and contract

Output, per ADR-0013 §4 — a file with a schema and provenance inside it, not a Python import:

```
tracks/<clip>.json
  schema: 1
  source: {detections: <hash+origin>, camera: <calib npz + its schema>, solver: <name+version>}
  humans: [ {id, frames: [...], boxes_xyxy: [...], provenance: [measured|inferred], merged_from: [ids]} ]
```

`merged_from` is not decoration: the whole defect class is one human wearing several ids, so the
artefact has to say which ids were folded and on which intervals, or the same information is lost
again.

**Consumes:** the clip, camlab's `calib/<clip>.npz`, cached detections (4362 already on disk for the
broadcast clip), and — for C3 only — cached poses. CPU, no GPU, no checkpoints (ADR-0013 §7).

## 7. The one open question before cutting

ADR-0013 puts identity and placement in **one** lab and pose-on-a-good-crop outside it, on the
grounds that pose is a black box that works. §5 row 1 refutes the grounds.

That does not automatically move pose inside — the flat jump may be a **grounding** defect
(monocular height/depth ambiguity, which is placement and therefore already playerlab's) rather
than a **pose** defect (SMPLest-X's body). Those are separable by measurement and have not been
separated: if the body's own joint angles show the jump while the root does not, it is grounding
and belongs here; if the body is flat too, it is the pose backend and belongs outside.

**Measure that before cutting the repo.** It decides where the boundary goes, it is CPU-only, and
the scenes to measure it on are already on disk.

## 8. Not in scope

* **Pose on a good crop** — pending §7.
* **Detection** — a cached input (ADR-0013: the boundary has no check of its own).
* **Kit, numbers, appearance** — blocked by resolution, not focus: the median player box is
  28 × 72 px and #109 measured 0 of 23 crops usable for OCR.
* **Deciding which id is "the real one" in a merge** — that is a policy the eye sets. The user's
  formulation on 2026-08-13: *drop the fragment, keep the parent's trajectory, take the fragment's
  pose.* Verified as correct for the crossing case; **not** verified for the referee's other four
  fragments, which sit at f57–62, f76–80 and f148–158 and have not been looked at.
