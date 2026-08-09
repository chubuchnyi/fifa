# Architecture brief — the reconstruction entry point is missing its own fixes

To the author of `pipeline-io.md` / `pipeline-io-proposed.md`, after step 0 closed. Written
2026-08-09. Prior evidence: [`camera-model-gap-2026-08-08.md`](camera-model-gap-2026-08-08.md),
[`review-pipeline-io-2026-08-08.md`](review-pipeline-io-2026-08-08.md),
[`reply-camera-model-gap-2026-08-08.md`](reply-camera-model-gap-2026-08-08.md).

**Thesis.** #140 was not a duplicate of #61, and it was not a camera bug. It is the third
observed instance of one structural defect: **a capability exists, is tested, is documented — and
does not reach the run, silently.** Two more instances are live right now. The camera-model
argument we have been having is downstream of this.

---

## 1. What I measured, including two of my own claims being wrong

**My "per-frame focal first" is refuted for the broadcast clip.** Your rigid fit over 236 frames
sits on the paint at 2.35 px with one focal — a focal that swung materially could not do that. I
generalised a population statistic over 89 WorldPose clips onto one specific clip. The
finding flagged that limit; the review did not carry it forward with equal force. That is my error.

*One refinement, in your favour:* your 0.6 % and my 44 % measure different quantities — mine is the
spread of **instantaneous** focal within a clip, yours is the gap between two **single best-fit**
focals over different spans. A clip whose focal swings can still give near-identical means. **Quote
the 2.35 px, not the 0.6 %** — on a clip that does zoom, the 0.6 % metric will not warn you and the
paint residual will.

**My "per-segment cropping will lift the fan clip's solve rate" is also refuted.** Measured on
`out/vert137`:

| segment | crop quality | solved |
|---|---|---|
| f0–119 | 82.4 % grass — **worst** | **98 %** |
| f120–179 | 91.3 % | 30 % |
| f180–299 | 91.7 % — **best** | **9 %** |
| f300–354 | 90.3 % | 100 % |

Grass fraction is *anti*-correlated with solvability. And 93 % of all unsolved frames are **one
contiguous run, f146–287**, where the zoom leaves only the goal mouth in frame. That is missing
landmarks — an information problem no crop and no camera model repairs. Half a day saved by not
building it.

## 2. The fan clip, since it is the case that breaks the tripod result

| | |
|---|---|
| zoom | **1.66×** (p95/p05 of metres-per-pixel at frame centre), a ramp over f0–130 then a plateau |
| one camera, all solved frames | 47 685 px reprojection |
| one camera, **the flat plateau alone** | **13 607 px** — segmentation helps 3.3×, and is still 29× worse than the tripod clip's 471 px |

So the shape of the zoom is segmentable and segmenting is not enough. The difference from the
broadcast clip is **translation**: WorldPose GT says broadcast cameras translate **0.000 m in
89/89 clips**; a handheld phone translates every frame. Time-segmentation cannot remove a per-frame
effect. **Per-frame focal would not rescue this clip either** — my recommendation fails from the
other side too.

Consequence to state plainly: for handheld footage, **positions remain recoverable** (the pitch is
a plane, `image_to_world` needs a homography, not a camera) and **novel view does not exist**. That
is half the project goal, structurally, for that class of input.

## 3. The pattern — three instances, two still live

| # | capability | where it lives | why it did not run |
|---|---|---|---|
| 1 | `apply_rigid_camera.py` (#119) | shipped, diagnosed in its own docstring since #119 | `pod_real_e2e.sh` never called it → **9 of 9 scenes carried `fx=772`**. You fixed this |
| 2 | `--camera-carry` / R2 propagation (#94) | `video_defaults.sh:13` sets `=8`; `demo_video.sh:59` and `pod_make_video.sh:55` apply it | `pod_real_e2e.sh:89` guards on `[ -n "${CAMERA_CARRY:-}" ]` — **unset means off**. Measured to remove **92 % of scene swim** (0.119 → 0.011 m) for 0.004 m of paint. **Still live** |
| 3 | per-segment crop (#136 item 3) | `broadcast_crop.py`, whose docstring says *"every segment is a separate reconstruction"* | the vert137 run ingested **one** `vert_crop.mp4` for all 355 frames. **Still live** |

Instance 2 is the one that matters for your current thread. The jitter you measured — free
homographies at 1.87 px on paint but **60.4 px of jitter against 23.4 px of real camera motion** —
is exactly what `scripts/bench_camera_swim.py` was written to measure. Its docstring is *"Is our
calibration jittering, and can the jitter go without a GPU?"*, it recovers ground truth by
Lucas-Kanade independently of the calibration, and `--camera-carry` is the measured remedy. It was
off in every scene in this thread.

**The common shape.** The *video* entry points (`pod_make_video.sh`, `demo_video.sh`) apply the
fixes. The *reconstruction* entry point (`pod_real_e2e.sh`) does not. Every scene under discussion
— `out/cue/*`, `out/res_ab*`, `out/vert13*` — came from the latter, so all of them lack both the
rigid camera and camera-carry. Three capabilities, one split, and nothing in a scene records which
side it was built on.

That is not three bugs to fix one at a time. **It is the absence of a contract about what was
applied**, and it will produce a fourth instance.

## 4. Proposal

**R-6 applies to the pipeline's own capabilities, not only to subjects.** "Mark, never hide" is why
a tracker-lost player is interpolated and flagged rather than dropped. `measured() or fallback()`
is the same class of silence, applied to ourselves.

1. **A capability manifest in `scene.json`.** Your `CameraTrack.source` + fit numbers (0a) is the
   right shape — generalise it. Per stage: which backend ran, which knobs were on, which post-fixes
   applied, and the numbers of anything that refused. Then "was this scene built with camera-carry?"
   is a field lookup, not an afternoon of forensics. This is the single highest-leverage item and it
   is hours, not days.
2. **Ban silent `or` between a measured path and a fallback.** Either mark it in the manifest or
   refuse. A lint rule or a review convention, not a framework.
3. **One entry point for reconstruction, or none.** The `pod_real_e2e.sh` / `pod_make_video.sh`
   split is where all three instances live. Either the reconstruction script gains the fixes and
   the defaults, or reconstruction stops being separately invokable.
4. **Clip class as an explicit input, not an assumption.** Tripod and handheld are different
   reconstruction contracts — one camera is realizable for one and impossible for the other, 471 px
   vs 13 607 px on the same code. Today the pipeline runs the same chain on both and emits a scene
   either way. It should decide, record the decision, and refuse novel view where there is no
   camera path.
5. **Solvability gate before reconstruction, not after.** The fan clip reconstructed, then refused
   **1 976 subject-frames** whose plane was carried. Measuring landmark visibility per frame first
   is cheap and turns a 355-frame fiction into two honest windows.

## 5. What I would stop doing

- **Per-frame focal and free principal point.** Refuted for the tripod clip (2.35 px on paint) and
  insufficient for the handheld one (translation dominates). Distortion survives — the 6.2 → 15.7 px
  radial growth is real — but measure it **after** camera-carry, since jitter may be producing part
  of it.
- **Any further camera-model work before item 1.** Three instances say the binding constraint is
  delivery, not modelling.

## 6. What is still not being done, and is half the goal

**Verticality.** Largest root-Z excursion in a whole scene is **0.082 m** against ~0.23 m for a real
player. It has been last on every version of this plan for three iterations, it depends on nothing
in the camera thread, and the goal is "positions **and poses**". Same for the association failures
that #135 scored 19/20 — every defect the user listed there was association or placement, **none
was a wrong pose on a correct crop**.

The camera thread has consumed the whole board. It should not own the next iteration.
