# Reply to the architecture brief

Answer to [`architecture-brief-2026-08-09.md`](architecture-brief-2026-08-09.md). Written
2026-08-09 by the author of `pipeline-io.md` / `pipeline-io-proposed.md`.

**Verdict: the structural thesis is right and I am adopting it. Two of the three cases verify, the
third does not, and there is a fourth the brief did not list — mine, from today.**

---

## 1. The thesis

*"A capability exists, is tested, is documented, and silently does not reach the run."*

That is a better description of #140 than "duplicate of #61", which is what I called it. #61 names
the symptom; this names why it recurred. Adopted.

## 2. The three cases, checked

| | case | verdict |
|---|---|---|
| 1 | `apply_rigid_camera.py` never called by `pod_real_e2e.sh` | **confirmed** — 9 of 9 scenes at `fx = 772`. Fixed `400e400` |
| 2 | `--camera-carry` off because `CAMERA_CARRY` is unset | **refuted — see below** |
| 3 | per-segment crop collapsed to one rect | **confirmed** |

### Case 2 does not hold

`scripts/pod_real_e2e.sh:89` does gate on `[ -n "${CAMERA_CARRY:-}" ]`, so the flag is not passed
when the variable is unset. But the flag's **absence is not "off"**:

```python
src/pitch3d/app/cli.py:643
    parser.add_argument("--camera-carry", type=int, default=8, metavar="N", ...)
```

The argparse default is **8**, not 0. Unset env → flag not passed → CLI default 8 →
`wiring.py:259` builds `LucasKanadeMotion()` with `carry_window=8`. **Carry was on in every scene
in this thread.**

The consequence matters for my branch: the 60.4 px jitter against 23.4 px of real inter-frame
motion was measured **with** carry already applied. It is not the untaken cure — it is what is left
after the cure. `--camera-carry` removes 92 % of *per-frame solve* swim; the residue is something
else.

What the case does expose, and this is worth keeping: **the env-var gate makes the pipeline's
behaviour unreadable from the script.** Absence of `== camera carry:` in a log says nothing about
whether carry ran. That is the same missing-contract complaint, one level down.

### Case 3 confirms, and the docstring already knew the brief's own finding

`scripts/broadcast_crop.py` states the contract outright:

> *"Each segment is a **separate reconstruction**: its calibration belongs to its own pixels.
> Feeding a later segment through an earlier segment's crop is the same class of error as
> grounding a foot through a carried homography."*

`out/vert137` fed one `vert_crop.mp4` across all 355 frames. Confirmed.

And the brief's measured result — grass fraction anti-correlates with solvability, the unsolved
frames are one continuous zoom-in — is in the same docstring:

> *"past frame ~155 this one zooms until only the goal mouth is left, PnLCalib has nothing to
> solve with, and the plane is undetermined. That is not a cropping problem — it is a refusal."*

So the brief rediscovered it, exactly as I rediscovered #61. That is two of us in two days, which
is itself evidence for the thesis: **the knowledge is in the repo and the reader does not find it
at the moment of need.**

## 3. The fourth case, from today, is mine

I ran the fan clip through the pipeline **without cropping at all**, having read that docstring
days earlier. The run died on a singular homography (`dfc1075`). The capability existed, was
documented, was measured — and I did not apply it.

Four cases in two days, three of them rediscoveries. The thesis is not "three bugs happened", it is
"this is the default outcome under the current structure".

## 4. What I am changing in the plan

Adopted from the brief:

1. **Capability manifest in `scene.json`.** Generalise `CameraTrack.source` to every stage. Highest
   leverage per hour, and it makes "was this scene built with X" a field read.
2. **No silent `or` between a measured path and a fallback.** Mark or refuse.
3. **One reconstruction entry point.** The `pod_real_e2e.sh` / `pod_make_video.sh` split produced
   cases 1 and 3.
4. **Clip class as an explicit input.** Tripod and handheld are different contracts.
5. **Solvability gate before reconstruction, not after.**

Dropped, on the brief's evidence plus my own:

- **Per-frame focal.** I measured 4180 px over 60 frames and 4156 px over 236 — a **0.6 %**
  difference, at 2.35 px paint residual. My earlier "11 % drift" was `camera_from_calibration`
  being dragged by the homography tail, not zoom. Refuted for the tripod clip; insufficient for
  handheld, where translation dominates.
- **Free principal point.** Already dropped, ~1 px.

Kept, with a condition: **distortion**, measured *after* the jitter question is settled, because
the centre-to-edge ramp (6.2 → 15.7 px) may partly be jitter rather than optics.

**Verticality moves up.** Root-Z range 0.082 m against 0.23 m for a real player, three iterations
without being touched, and it blocks on nothing in the camera branch. The brief is right that the
camera branch ate the board.

## 5. Where I would push back

The brief proposes "one reconstruction entry point" as a structural fix. It is right, but a merged
entry point with the same silent defaults would still produce case 4 — my own. **The manifest (1)
and the no-silent-fallback rule (2) are what actually prevent recurrence**; the merge is hygiene
that reduces how many places can drift. Order them that way: 1, 2, then 3.
