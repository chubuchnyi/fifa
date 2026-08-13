# The problem, stated for someone who has not seen this repo

This is playerlab's `PROBLEM.md`, written before the repository is cut. It lives here next to
[`playerlab-spec.md`](playerlab-spec.md) — the founding decision, per **ADR-0013**'s checklist —
and becomes `docs/PROBLEM.md` when the repo moves.

Read this, then `STATUS.md`, then `findings/landmines.md`. Everything below is measured; where it is
not, it says so.

---

## What playerlab is for

One clip of a football match goes in. For every frame it answers **which human is this, and where
exactly does he stand**: one human keeps one id from his first frame to his last, and the body drawn
for him lands on his own pixels.

The consumer is a reconstruction that re-renders the same episode from a new camera, judged by a
person's eye. So "exactly" is not a metric target either — it is *a viewer follows one player across
the clip and he stays the same player, in the right place, doing what he did*.

Two clips are on disk. `broadcast`: 236 frames, 1920×1080, professional camera, Colombia–Congo DR at
night; 4362 detections cached. `fan`: 1080×1920, a phone from the stands at a different floodlit
night match, panning and zooming; detections cached for frames 0–236 at four detector resolutions.

**No GPU.** playerlab consumes cached detections and cached poses; the association benches run in
~3 min on a laptop CPU.

## What works

**The camera is no longer the problem, and that is what makes this lab possible.** camlab solves it
to 1.82 px (`fan`) and 2.96 px (`broadcast`) against the paint. Until that existed, placement error
could not be measured because camera error swamped it — AVATAR shipped nine scenes out of nine
carrying an invented `fx = 772`.

**The evidence is not the problem either.** 96 % of mid-pitch births have an unclaimed detection a
median 6–23 px away. The boxes exist.

**C1 is built and label-free.** Two ids never measured on the same frame, staying within 0.15 m and
alternating who carries the detection are one human — there is only ever one detection to give, and
two standing pelvises are not 15 cm apart. `scripts/find_duplicate_tracks.py`. On the 2026-08-12
A/B: 39 ids collapse to **33 distinct humans** (ByteTrack) and **35** (BoT-SORT+GMC), and the
referee is smeared across **5 ids in both arms**. Scored against the only labels we have, it is
4 for 4 — it catches both pairs the eye named and rejects both pairs the eye called two players in
contact.

## What does not, and why it is not a matter of effort

### 1. Half the scene is not measured, and six interventions did not move it

**54.1 % (ByteTrack) and 52.7 % (BoT-SORT) of subject-frames are `imputed`** — frozen limbs on a
coasting root. Six association fixes have been measured against that:

| intervention | what it moved | the plateau |
|---|---|---|
| match threshold, both directions | null | — |
| detector threshold | null | — |
| `split_on_kit_change` | tracks carrying two humans 9 → 0 | post-stitch count stays 33–38 |
| Deep-EIoU expansion IoU | raw ids 56 → 43, worst seam 25.7 → 9.5 px/f | same 33–38 band |
| McByte mask cue (Cutie, 4.2 h/pass GPU) | mid-pitch events 28 → 24 | **14 % against a 96 % ceiling** |
| BoT-SORT + `sparseOptFlow` GMC | mid-pitch events −27 % / −30 % | **1.4 pp of the scene; the eye saw nothing** |

The last one is the sharpest statement of the problem. Thirteen of fourteen metrics preferred it,
several by 30–50 %, and the user's verdict on the two scenes was **«равносильно херовые»** — equally
bad. The metrics were not wrong; they were reported without their denominator, all of them being the
same 1.4 pp counted per track instead of per frame.

### 2. The defect is allocation, not evidence — which is harder than a better cue

At f33, in **both** arms, a spurious track is born **0.07–0.08 m from the referee** (and 1.9 m from
the player it was supposed to be) and stays glued to him for its whole measured life — **while the
referee's own rows on those frames are `interpolated`.** He is being measured; the measurement is
wearing a new id.

The rival explanation was tested and measures **0.0 %**: "fire when no track claims the detection"
never triggers, because a birth's best assignment cost is **0.176–0.221 against a 0.250 gate** for
every column. The right track was available and the assignment gave it to a competitor. So the
missing thing is not a stronger appearance cue feeding the same assignment — it is the assignment.

### 3. There is nothing to identify with

The median player box is **28 × 72 px** on the phone clip and **41 × 86** on broadcast — about
**573 px of shirt, and the same 573 px for eleven teammates**. This is measured against the
literature rather than assumed:

- KPR's keypoint prompt is worth **+7.0 R-1** where multi-person ambiguity exists and **+0.2** where
  it does not (Market-1501 93.0 → 93.2) — it disambiguates, it does not identify.
- BMPv2's authors state mask refinement **"is not suitable"** below 100 px, and that looping it
  *"pronounces the error"*.
- Sapiens filtered its training data to boxes **>300 px** (v1) / **≥384 px short side** (v2) — 11×
  above our subjects.
- **No method publishes AP_S.** OCHuman selects by occlusion and never by size.

Appearance-based re-identification is not an unexplored option here. It is measured absent.

### 4. Height and depth trade against each other in a monocular view

t19 leaves the ground at f46 and heads the ball at f55. He is `measured` on **every frame f44–57**,
straight through the header, and the measurement contains no jump: peak root-Z **0.837 m, about 9 cm
above his own baseline, against a real jump's ~40 cm**, with the rest of the motion appearing as
travel deeper into the scene. That is exactly what the eye reported.

No tracker change reaches this. Whether it is a **grounding** defect (monocular height/depth
ambiguity — placement, therefore playerlab's) or a **pose backend** defect (SMPLest-X's body) is
separable by measurement and **has not been separated**: if the joint angles show the jump while the
root does not, it is grounding; if the body is flat too, it is not playerlab's. This is the open
boundary question — [`playerlab-spec.md`](playerlab-spec.md) §7 — and it should be measured before
the repo is cut, because it decides what is inside it.

### And there are no labels

`WorldPose/` holds 89 clips of FIFA 2022 ground truth — per-player poses, per-frame `K`, `R`, `t`.
**The video is gated by a separate agreement**, so it cannot score our estimator on their frames.
What it gives is outside bounds: real max speed **9.74 m/s**, acceleration p99 **8.35 m/s²** on a
100 ms average, and a real root ranging **0.23 m** per clip.

The only labelled set that exists is **six observations the user made on 2026-08-13**, with frame
numbers and ids, in `findings/quality-criterion-2026-08-13.md` §2, plus the 24-track verdict in
`findings/track-labels-2026-08-07.json`. A criterion that does not rank those two A/B scenes as
roughly equal is not yet a criterion.

## Where the code is

What moves out of AVATAR when the repo is cut:

| what | file |
|---|---|
| **C1 — the label-free duplicate detector** | `scripts/find_duplicate_tracks.py` |
| the per-track criteria and the provenance timeline (#135) | `scripts/track_quality.py` |
| association A/B on cached detections | `scripts/bench_association.py` |
| the assignment margin — why a birth is contested, not orphaned | `scripts/bench_assignment_margin.py` |
| speed / acceleration plausibility | `scripts/motion_stats.py` |
| continuity, failure kind, id budget | `scripts/track_continuity.py`, `identity_failure_kind.py`, `identity_budget.py` |
| the tracker, the kit split and the 2D stitcher | `src/pitch3d/adapters/models/tracking.py` |
| BoT-SORT + GMC, injected by dotted path | `src/pitch3d/adapters/models/botsort_backend.py` |
| the mask cue (Cutie propagation) | `src/pitch3d/adapters/models/mask_propagation.py` |
| the handover merge, continuity and the identity gate | `src/pitch3d/core/orchestration/{handover,continuity,identity}.py` |
| **where `imputed` is written** — coasting and gap bridging | `src/pitch3d/core/correction/coherence.py` |
| the labels | `docs/findings/track-labels-2026-08-07.json`, `quality-criterion-2026-08-13.md` §2 |

Data: `out/pod_ab/scene_{a,b}.json` (the A/B, 236 frames, five real backends, association the only
difference), `out/dets_fan/dets_r{560,896,1064,1288}_0_236.npz`, `calib/*.npz` from camlab.

## The check that is not the scene judging itself

C1 and C2 read **only the scene**. A scene that is smooth, continuous, single-id and entirely wrong
passes both. They are self-consistency, and nothing here is believed on their evidence alone.

**C3 is the analogue of camlab's paint residual**: reproject the posed body through camlab's camera
and compare with the frame — foot/root against the bottom-centre of the detection box that fed that
frame, silhouette against the person mask. It is falsifiable from the clip alone, it needs no
labels, and it **has never been computed**, because until camlab existed the camera error was larger
than the thing being measured.

**Where playerlab is weaker than camlab, stated rather than glossed.** camlab has a channel that
shares nothing with its solver: mowing stripes are evenly spaced in metres, so through a right
camera their period holds under zoom — 11.00 m ± 2.3 % across a 1.61× zoom on `fan`. C3 is not that.
Its box and its mask both come from our own detector, so C3 can be self-consistent and wrong in
exactly the way the detector is wrong. Two candidates for a genuinely independent channel exist and
neither is built: a segmentation from a model that is not the detector, and WorldPose's GT bounds
above. Do not quote C3 as if it were camlab's residual until one of them backs it.

## Remarks on the founding spec

Nine objections to [`playerlab-spec.md`](playerlab-spec.md), written the same day it was. They are
here rather than in the spec because the spec is the decision and this is the review of it. Each
says what to do, not only what is wrong.

**1. The baseline has no denominator, which is the spec's own §4 rule applied to the spec.**
"33 distinct humans (A) / 35 (B)" is quoted as the number to beat, and **nobody has counted how
many humans are actually in the clip.** That is one frame's work and it changes the reading
completely: if the answer is ~22, then 33 and 35 are both roughly 50 % over and the A/B moved
nothing, which is what the eye said. It also makes "more distinct humans" a **gameable target** —
a tracker that alternates less produces a higher C1 count without fixing anything. Count the
humans; state the target as the eye's six defects being gone, with C1 as a monitor rather than a
score.

**2. C1's 0.15 m is a fitted number wearing a physical justification, and its units are not
validated.** The spec calls it "a bound on human bodies, not a fitted number"; the derivation in
the findings is *0.20 m let one pair the eye had called two-players-in-contact leak in* — a fit
against a single counter-example. Worse, the quantity is **reconstructed pelvis position in
metres**, and the error bar on that is exactly what C3 has not measured yet. C1 is currently
calibrated against a ruler of unknown length. Re-derive it after C3, or express the gate in image
pixels, where the error is known.

**3. C1's premise is supported; its converse is not excluded.** "There is only ever one detection
to give" has real backing — the NMS follow-up in #132 measured ≤1 id. But the rule fires on ids
that are *never co-measured*, and two genuinely different humans produce exactly that pattern when
one is heavily occluded: #135's t20 was 93–100 % covered by two other players at f0–9 and is a
real, separate person. Add the occlusion case to the false-positive controls alongside the contact
case.

**4. C2 has no bar, and it repeats the unit the spec just condemned.** C1 has a rule and a number,
C3 has a comparison, C2 names three quantities and no threshold — which is how §4's failure returns
("a quantity without a denominator gets quoted as a verdict"). It is also stated per human over his
whole span, i.e. the per-track unit §4.4 says averaged the defects away. Make C2 per-interval and
give it a bar.

**5. C3 is the whole epistemic foundation and it is the one unbuilt thing — so build it first.**
"Nothing in playerlab is believed before C3 runs" means every hour spent on C1/C2 or on a solver
before C3 exists produces unfalsifiable work. It also needs a number stated up front: camlab's
bar is 20 px because a person with a ruler can see that on the overlay. What counts as "lands on
his own pixels" here is unstated.

**6. §7's decisive measurement may be answerable by reading the code instead of running an
experiment — do that first, it is minutes.** Root Z has **at least three possible providers**: the
pose backend's own `pelvis_above_foot` (SMPLest-X reports it, which is why #142's constant never
fired on the fan clip), our FK provider (the net for backends that do not), and the constant. If
the rows at f44–57 were grounded by *us* through the homography, the flat jump is placement and the
boundary question is already settled inside playerlab. Read which provider wrote those frames
before designing anything.

**7. The artefact schema loses the distinction §4.2 exists to preserve, and omits four things a
consumer cannot guess.** `provenance: [measured|inferred]` cannot separate "not measured" from
"measured wrong" — the crossing birth and the flat jump both print the same word today, and the
schema keeps it that way. Missing, each with a run behind it:

- **which detection fed this frame** (id or none) — without it a misallocation is not auditable and
  C3 is not computable;
- **whether the ground plane was solved on this frame** — grounding that did not read calibration
  confidence produced a 3079.7 m root spread (#136);
- **the crop rect** the boxes are in — the fan clip is 4 segments, not one, and a box without its
  rect is unusable;
- **the frame-index convention and the detector resolution** — `bench_association.py` refuses a
  width/height mismatch because scoring the portrait clip as landscape excuses every birth, and
  detector resolution 896 vs 560 moves identity events 89 → 61 (W1). §5 says "record which" and §6
  has no field for it.

Also state the 180° roll convention in the file. camlab's schema-2 rule is the one to copy: every
key present on every frame, and refuse by name rather than guess.

**8. §8 declares the merge policy out of scope while §6 requires `merged_from`.** Those cannot both
hold: writing `merged_from` *is* applying the policy. Make the artefact record **candidate** merges
with their intervals and not apply them, leaving the decision to AVATAR or to a person in the UI.
The policy itself — *drop the fragment, keep the parent's trajectory, take the fragment's pose* — is
verified on **one** case; the referee's other four fragments (f57–62, f76–80, f148–158) have not
been looked at. #139 already refused a heuristic tuned on n=1 in this exact area.

**9. Two gaps in scope.** The spec does not mention **the UI at all**, and ADR-0013 §3 makes it part
of a lab rather than an afterthought — here its first job is not viewing but **producing labels**,
because six observations is the entire labelled set and it is the scarcest input this lab has.
And **the ball is unassigned**: it is the same question (which object, where), and the heading
defect at f55 cannot be judged without it. Decide, do not leave it implicit.

## How to work here

- **Per human per frame, not per track.** Every metric that failed on 2026-08-13 counted tracks or
  violations. A fragment born on the wrong player is fatal to a viewer and costs ~nothing in a
  per-track aggregate.
- **The unit is the (pair, interval), not the pair.** t3 and t62 are one human for 14 frames and
  genuinely two for the next hundred; averaged over the pair, the defect disappears. The eye said
  the same thing first — *"correct until frame 39, then it became t62"* — a transition, not a
  property.
- **Report the denominator.** A metric that improves 40 % of a quantity that is 6 % of the scene has
  to say so. Not doing this bought two wrong decisions in one session.
- **Separate "not measured" from "measured wrong".** The crossing birth (evidence exists, misfiled)
  and the flat jump (evidence exists, wrong) both print `imputed` today.
- **Nothing inherited is true until re-measured.** The register is `playerlab-spec.md` §5, and one
  entry is already refuted — the #135 claim that "none is a wrong pose on a correct crop", which is
  the claim ADR-0013 used to put pose outside this lab.
- **A person's eye outranks the metric.** It has been right against it here already: thirteen of
  fourteen numbers preferred one arm and the eye was right that nothing had changed.
- `findings/landmines.md` is where a new trap goes, in the session it is hit.
