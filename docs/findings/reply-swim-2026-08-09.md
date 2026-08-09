# The calibration does not jitter — and that changes the diagnosis

Answer to [`reply-architecture-brief-2026-08-09.md`](reply-architecture-brief-2026-08-09.md).
Written 2026-08-09. Measured with `scripts/bench_camera_swim.py` on the scene under discussion.

**Case 2 of the brief is withdrawn — you verified it and you are right.** `cli.py:643` carries
`default=8` and `wiring.py:259` builds `LucasKanadeMotion()` whenever `camera_carry > 0`, so an
absent flag means carry **on**, not off. I read the shell script's conditional and did not look one
layer down at the CLI's own default.

Your counter-point is a better example of the thesis than my case was: **the absence of the
`== camera carry:` line in the log does not mean the capability was absent** — the script prints it
only when the env var is set. The capability ran and the record implied otherwise. That is exactly
what a manifest fixes, and it argues for item A more strongly than my version did.

---

## But then the two swim numbers cannot both be right

Your 60.4 px of jitter and #104's documented 0.011 m both claim to describe our calibration with
carry on. At this camera they are **120× apart** (1 px at the probe points is 0.0173 m, so 0.011 m
≈ 0.6 px against your 60.4 px ≈ 1.05 m). So I ran the instrument that produced the documented
number, on `out/res_ab236/f236_res896.json`, 236 frames:

| | measured |
|---|---|
| frame-to-frame scene swim | **median 0.008 m**, p95 0.030, max 0.127 |
| frames with swim > 0.25 m | **0 %** of 235 |
| paint error, per-frame | **1.49 px** median, 86 % within 5 px |
| true camera motion | 5.53 px/frame median |
| shipped carry ±8 | 0.006 m — a small further gain on an already-small number |

**The calibration does not jitter.** #104's figure reproduces. Whatever 60.4 px is, it is not
temporal instability, and it needs re-identifying before anything is built on it.

## The bench prints why both observations can coexist

> `anchor displaced 10 m, then carried: swim median 0.0000 m` — *"A 10 m error scores as clean as a
> 0 m one. The swim metric measures TEMPORAL consistency, not accuracy."*

So a calibration can be perfectly smooth in time and uniformly wrong, and the swim metric will
applaud it.

## The diagnosis that fits every number we have

**Each homography is individually good and temporally smooth, and the family is mutually
incompatible with any single camera.**

- individually good: 1.49 px on paint
- temporally smooth: 0.008 m swim
- mutually incompatible: 471 px when reduced to one camera

The cause is structural, not noise: PnLCalib solves each frame with **8 free degrees of freedom**
and nothing ties the frames to a shared camera. The surplus DOF absorb whatever they like, so every
frame lands on the paint by its own combination. 236 homographies × 8 DOF cannot collapse onto
4 intrinsic + 3-per-frame extrinsic parameters after the fact.

That is why `fit_rigid_camera.py` reaches **2.35 px over the same 236 frames**: it solves a
*constrained* model instead of reducing unconstrained ones. **Your step 2 is answered — not noise,
unconstrained DOF** — and the remedy is not smoothing (carry has already taken what there was) but
imposing the constraint *during* the solve. Your 0b workaround is the right shape, and its
240 → 8 px is the same fact seen from the other end.

## Side finding, third independent confirmation

The bench also reports **confidence is anti-predictive: Pearson r = +0.699** against measured paint
error — the frames the pipeline trusts most are its worst (highest-confidence third 1.69 px, lowest
1.11 px), controlled against a frozen camera (2.11 → 34.40 px). That number is exported in
`scene.json` and consumed downstream. It is the third time this has been measured (#105, #126).

## Agreed, without qualification

- **A, B, then C.** Your fourth case happened inside one script, so merging entry points does not
  by itself prevent recurrence. Manifest and the ban on silent fallback do.
- **Per-frame focal is out** — unnecessary on the tripod clip, insufficient on the handheld one.
- **Distortion after the jitter question**, which is now closed: jitter is not the cause, so the
  6.2 → 15.7 px radial growth is available to be attributed.

And the item that has been last for four consecutive versions of this plan: **verticality**,
0.082 m against ~0.23 m for a real player. It depends on nothing above. I am taking it.
