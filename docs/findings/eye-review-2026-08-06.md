# Eye review of the 3D scrubber, 2026-08-06 — the user's frame-level verdict

The first review of `/world` against the source video. Frame-level, specific, and it found things
no metric here had surfaced. Recorded verbatim in substance because several items are *new
defect classes*, not instances of known ones.

## a) Team colours were swapped — **fixed the same day**

> жёлтые и синие игроки перепутаны цветами. В клипе синие в жёлтой форме а жёлтые в синей.

My bug, not the pipeline's. The viewer hard-coded `A → light blue, B → yellow`, but team ids are
**k-means cluster labels** — "A" is whichever cluster seeded first, not a side. The scene has
carried the measured colour all along (`Team.color_rgb`: A = RGB 0.71/0.70/0.29 → yellow,
B = 0.36/0.51/0.62 → blue), i.e. exactly the opposite of what I drew.

Fixed by serving `color_rgb` from `/api/world/{n}/skeletons` and painting from it. A palette
indexed by team id cannot be right by construction; a measured colour cannot be wrong.

## b) Limb activity separates the real player from the phantom — **the most actionable item here**

> номер 3 после пересечения судьи 2 превращается в 66(64). До пересечения скелет 3 фактически
> бежал, а 66(64) перемещался не двигая конечностями. После пересечения наоборот… Позиция на поле
> верная у того, который двигает конечностями. И вообще если нет движения конечностями совсем, то
> это признак того, что человек — фантом.

Two claims, both testable, and together they are a **new signal this repo does not use**:

1. **A track that translates with no limb motion is a phantom.** Per-crop HMR on a box that is not
   really a player returns a near-static pose while the root is dragged along by the tracker.
2. **At a crossing the limb-active one holds the correct position.** So when track X goes still at
   the same frame Y comes alive, they are the same human and the active one is authoritative.

That is a *stitching cue and a phantom detector from the same measurement*, and unlike the mask cue
it needs no new model — the joint angles are already in the scene. `config/physics.yaml` even
carries a `pose_motion_sync.joint_activity_threshold`, so the quantity is half-named already.

Next step is to measure it before building: per track, per frame, total absolute joint-angle change;
then (i) how many tracks are near-zero throughout, (ii) whether the 78 mid-pitch identity events
line up with an activity handover between two tracks.

## c) A referee is missing entirely

> Судьи на задней бровке за 18-м номером вообще нет. Он чётко просматривается на видео.

A visible person with no track at all. This is the **detection-recall** half measured earlier
(~30 % of identity events had no box within 4 frames at threshold 0.3, dropping to 4 % at 0.10) —
but this one is a whole subject missing for the entire clip, not a gap. Worth checking whether the
far-touchline referee is below the score threshold, filtered by the `plausible()` box bound, or
dropped as the `referee` class somewhere downstream.

## d) The penalty area — where the reconstruction most clearly disagrees with the video

Four distinct failures, ordered as the user listed them:

1. **Frame 0: adjacency lost.** Player 14 overlaps an opponent standing *right against him* in the
   frame; by pose and orientation that opponent is 20 and 25 in 3D — but 20 and 25 are placed at a
   distance from 14, not adjacent. The 2D contact is real and the 3D says otherwise.
2. **Frame 43: one player, two ids, and a jump.** The opponent separates from 14 in the frame and
   moves a short distance; in 3D he *splits into 25 and 20* and both snap toward 14.
3. **Top of the box: 10 splits into 10 and 77** — the same one-player-two-ids failure.
4. **Centre: 17 jumps for the ball, and 3D keeps him on the plane.** Vertical motion is not
   reconstructed at all. `gravity_project` (airborne Z → ballistic parabola) is one of the gates
   `config/physics.yaml` ships **disabled**; whether it would fire here is untested, and the root Z
   may simply be pinned by the foot-plane anchor regardless.

Items 1–3 are the same underlying thing seen three ways: a player adjacent to another gets his
identity split, and the pieces are then placed by whichever box each piece inherited. That is the
crossing failure, but the *placement* consequence is new information — it says the split does not
merely churn ids, it moves bodies metres out of position.

## The mask cue, judged by eye

> по красным скелетам, субъективно хуже, объективно сложно сказать, т.к. там где они с ошибкой то
> и референсный результат тоже с ошибкой.

Honest and damning in a specific way: the comparison is uninformative because **both arms are wrong
in the same places**. The cue moved 28 → 24 mid-pitch events, but it does not fix the failures the
eye actually notices — which are the ones above. That is consistent with the measured weakness
(14 % against a 96 % ceiling) and argues against polishing the cue further before items (b) and (d)
are addressed.
