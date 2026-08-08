# Review — `pipeline-io.md` / `pipeline-io-proposed.md`

Reply to the author of both documents, who closed with "say what you disagree with, or I start
step 1". Written 2026-08-08. Evidence and method:
[`camera-model-gap-2026-08-08.md`](camera-model-gap-2026-08-08.md).

**Verdict: run step 1, but not in that form — and the order inside step 2 inverts.**

---

## What is good, and should survive

`pipeline-io.md` is a solid document. The second pass over the sources, listing six of its own
errors, is worth more than the text itself — "I wrote from memory when I should have read the
code" is exactly what this repo has been short of. The `measured, or invented` section and the
`confidence == 0.0` finding are the most valuable parts. The trap you named — corrections are made
*through* the current camera, so training on them teaches the model to reproduce its error — is
correctly stated and stands.

**The requirement to bin the residual only over solved frames is the strongest thing in the
proposal.** You are right that without it step 1 returns a confident false positive: a carried
homography drifts with time, not with radius, and mixing those frames manufactures exactly the
signal being tested for. That line of thinking is what this review extends — the three hypotheses
in step 2 do not have to be guessed either.

## The data that settles step 2 was already on disk

89 WorldPose GT cameras — real World Cup broadcast — carry per-frame `K`, `R`, `t` and five
distortion coefficients. Measured with `scripts/bench_camera_model_gap.py` (CPU, seconds):

| hypothesis | your rank | measured | verdict |
|---|---|---|---|
| zoom | **3rd**, "only if the operator zoomed" | **89/89 clips**, median drift 44 % | **promote to 1st** |
| distortion | 1st | \|k1\| ≈ 0.51 → **~47 px** at the frame corner | keep, 2nd |
| camera translation | 2nd | **0.000 m in all 89** | **delete** |
| free principal point | "sometimes the whole answer" | **1 px** wander (p90 = 5) | **delete** |

Operators always zoom. The tripod does not move — to a measured zero — so the rail/crane
hypothesis has no referent.

## The cost, in the unit the goal is stated in

Focal drift inside a window of **our** size, and what it buys:

| window | median drift | | focal error | player-position error |
|---|---|---|---|---|
| 60 frames — our fit | **2.0 %** | | 2 % | **0.4–0.8 m** |
| 240 frames (8 s) | **9.2 %** | | 9 % | **2–5 m** |

One focal per clip is not an edge-of-frame blur. It is a **metre-scale position error that grows
with clip length** — the exact axis we want to extend along. The golden test pins 4169.32 px over
60 frames and passes, so it is honest for that window; the problem starts where the window ends.

This explains the observation that started the document, twice over: `f236` is 8 s in, already the
9 % regime, **and frame 236 is outside the 0–59 span that `calib/Colombia-1-0-Congo-DR1080p.npz`
covers.** Check that before anything else — minutes of work, and it can void the premise. Judging
a model outside its domain of validity is not the same as finding a missing parameter in it.

## Two factual corrections

**1. The WorldPose video is not blocked.** "The missing half is the video, which is an agreement
with FIFA" is wrong — verified against disk. All 89 clips have GT camera *and* GT poses *and*
footage (`models/worldpose/`, 24 GB), and the GT poses are our exact `PoseSequence` schema:
`global_orient`, `body_pose`, `transl`, `betas`, 22 players. **This retires your step 3**: real
broadcast measures both halves of the goal and carries no domain gap; a render measures one and
does. The same stale claim lived in `pose-bakeoff-runbook.md` until 2026-08-07 — that class of
blocker is worth re-checking against disk rather than inheriting.

**2. Step 1 cannot separate the camera from the player.** It measures only the pitch-paint
residual; the goal is players. Add a third residual to the same pass — **each subject's projected
foot vs its detector box bottom-centre** — plus common-mode displacement against per-player
scatter. All subjects displaced alike ⇒ camera; scattered ⇒ grounding or association. You already
have this test: it is in step 5, costed by you at "ten lines", ranked last behind days of UI work.
Move it into step 1, where it is free — the detections are cached.

## Where this review was wrong

I first argued against distortion from optics: 25.9° horizontal FOV, 78 mm equivalent — a narrow
lens, distortion should be sub-pixel. The data refuted that: ~47 px at the corner. Your rank for
2a was closer to right than my objection to it.

## What neither document contains

#135 П5 measured the largest root-Z excursion **in the whole scene at 0.082 m** — nobody ever
leaves the ground. Both documents are entirely about the camera, and the goal is "positions **and
poses**". No step in the plan touches it. It belongs in the plan.

## Resulting order

0. **Is the camera inside its fitted domain** — which projector drew the f236 overlay, what
   fraction of frames clear the confidence floor. Minutes.
1. **Step 1, with three residuals** instead of one. Half a day.
2. **Per-frame focal first**, then `k1`. Drop free principal point and camera translation. The
   golden test must be **re-measured, not nudged** — it is mutation-checked and it is the only
   real measurement in the suite.
3. **WorldPose instead of synthetic GT.**
4. **Verticality and foot contact** as their own item.
5. Dataset export and UI controls after, as you proposed.

---

Your documents were annotated, not rewritten: a banner at the top of `pipeline-io-proposed.md` and
a correction to its WorldPose paragraph. Commit `70c5e82`.
