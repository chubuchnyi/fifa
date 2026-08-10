# M-1 — one camera centre DOES hold for the handheld clip

Measured 2026-08-10. Probe: `scripts/probe_handheld_centre.py`. Re-run it rather than trust this.

```bash
PYTHONPATH=src .venv/bin/python scripts/probe_handheld_centre.py --frames 14 --last 108
PYTHONPATH=src .venv/bin/python scripts/probe_handheld_centre.py --control --frames 8
```

## The question

[`architecture-brief-2026-08-09.md:57`](architecture-brief-2026-08-09.md) concluded that for
handheld footage **"novel view does not exist"**, because a phone translates every frame while a
broadcast rig does not. The broadcast half is measured — WorldPose GT, 0.000 m in 89/89 clips. The
phone half is not: WorldPose contains no phone clips. It was an analogy, and the whole
`camlab` proposal ([`../camlab-spec.md`](../camlab-spec.md)) stands or falls on it.

**The confound.** The fan clip zooms **1.66×** over f0–130. Fitting #119's model — one focal, one
centre — fails from the zoom alone, and that failure looks exactly like translation. So three
models, not one:

| | model | DOF | isolates |
|---|---|---|---|
| **seed** | the stored per-frame homographies | 8 F | how good the paint evidence is at all |
| **B** | per-frame focal + rotation, **one centre** | 3 + 4 F | the cost of fixing the *position* |
| **A** | one focal, one centre, per-frame rotation | 4 + 3 F | the extra cost of fixing the *focal* |

B is the diagnostic: it grants everything except a moving centre.

## Result

Median distance from the projected pitch markings to the painted lines, with the sample count.

| clip | frames | seed | **B — centre fixed** | A — centre + focal fixed |
|---|---|---|---|---|
| **tripod control** | 8 over f0–59 | 1.70 px · n=2536 | **1.30 px · 0.76× · n=2535** | 1.33 px · 0.79× · n=2541 |
| fan | 6 over f0–119 | 8.09 px · n=1074 | **9.97 px · 1.23× · n=2090** | 17.13 px · 2.10× · n=1691 |
| fan | 8 over f0–119 | 7.98 px · n=1449 | **8.88 px · 1.11× · n=2679** | 15.66 px · 1.96× · n=1801 |
| fan | 16 over f0–119 | 11.44 px · n=2755 | **10.30 px · 0.90× · n=4519** | 16.96 px · 1.48× · n=3693 |
| fan | 14 over f0–108 | 8.00 px · n=2479 | **8.45 px · 1.06× · n=3395** | 14.90 px · 1.86× · n=3166 |

**Fixing the camera position costs 0.90–1.23× against 8 free DOF per frame, on every run, while
placing 37–64 % MORE of the pitch inside the frame.** One centre holds. The 2026-08-09 conclusion
does not survive this.

**Fixing the focal as well costs 1.5–2.1× on the fan clip and 1.02× on the tripod.** That is the
zoom, and it is why the spec's per-frame `f_t` is not optional.

B's recovered focal curve over f0–108 — a smooth ramp, ×1.8, matching the independently measured
1.66×:

```
2994 3093 3353 2995 3519 4744 4100 4537 4807 5036 4726 5752 4975 | 300
                                                                   ^ degenerate, see below
```

## Why the control matters

The same harness on the tripod clip recovers **focal 4175 px, centre (−2.18, −70.25, 17.28) m,
paint 1.33 px**. `tests/e2e/test_golden_real_camera.py` pins **4169.32 px** and
**(−2.29, −70.13, 17.22) m**, and #119 reports **1.4 px**. Agreement to 0.14 % on the focal and
11 cm on the position, from an independent implementation.

Two further checks the control passes: the seed lands at 1.70 px against #114's documented
1.4–1.7 px, and variant B's free focal comes out at ×1.02 across the clip — it does **not** invent
a zoom where there is none.

## Three things this measurement forced out, each of which had to be fixed to get a real number

### 1. The homographies live in the CROP rect, and the pipeline tells the camera fit otherwise

`--crop auto` measured `1080×608+0+1294` (grass 0.285 → 0.816) and set `ClipRef.crop`. Every
adapter decodes through it (`adapters/io/frames.py:iter_clip_frames`), so PnLCalib only ever saw
**1080×608**. But `ClipRef.width/height` stay at the source size — `cli.py` does
`replace(_c, crop=_fr.rect)` and nothing else — and `controller.py:709` passes
`width=clip.width, height=clip.height` to `camera_from_calibration`. **The camera fit is handed
1080×1920 for homographies that live in 1080×608.** The principal point is placed 656 px outside an
image 608 px tall.

Measured, by projecting the markings and scoring against the paint:

| space | rot180 | paint median | n |
|---|---|---|---|
| **crop 1080×608+0+1294** | **False** | **9.47 px** | **1271** |
| crop 1080×608+0+1294 | True | 16.95 px | 1148 |
| full 1080×1920 | True | 17.32 px | 1168 |
| full 1080×1920 | False | 30.59 px | 36 |

Consequence: **the 12 382 px this project has been quoting all week is measured in the wrong
space.** At the correct 1080×608 the same function returns **18 313 px** with focal 2099 px. Both
refuse (`REALIZABLE_PX = 1.0`), so no verdict changes — but the number, and the `fit_focal_px`
stored beside it, are wrong. Not fixed here: it changes pipeline output and belongs in the
`camlab` copy.

Also settled: **this clip has no 180° roll.** The roll in `landmines.md` is a property of the
solved `CameraTrack`, not of every calibration.

### 2. Four frames carry near-degenerate homographies that every existing guard passes

Frames 115 and 117 measure *mirrored* while the other 118 do not — which
`fit_rigid_camera.load_world_to_image` asserts cannot happen. They are not a mid-clip frame change:
`|det|` is **1.0e-6** and **5.3e-8** against a clip median of **3.4e-3**, four to five orders down.
The plane has collapsed toward a line, and the handedness test reads the wrong sign off it.

Both pass `plane_camera._SINGULAR_DET = 1e-12` by six orders of magnitude. Both carry ordinary
confidence — **0.475 and 0.394**, against a clip median near 0.45. Neither guard sees them.

The probe excludes by a *relative* test, `|det| < 1e-3 × median(|det|)`, which catches 115–118. The
focal curve shows the tail is bad from about f109 onward regardless, so f0–108 is the honest span.

### 3. A paint median is meaningless without its sample count

`paint_error` scores only markings that land inside the image **and** on the playing surface. A fit
that has run away projects almost everything off-surface, where it goes unscored, and posts a
*flattering* median on the survivors. The first version of this probe did exactly that: it reported
a confident "ONE CENTRE HOLDS" off a variant-B fit whose focal was **87 px on a 1080 px wide
image**, with the sample count never printed.

Fixed three ways, all still in the script: every median carries `n`; the verdict is declared
INVALID if a variant scores on under 60 % of the seed's samples or if a fitted focal is unphysical
for the image width; and variant B's focal is bounded to `FOCAL_BOUNDS` with an optional (default
off) smoothness term, so the answer cannot come from the regulariser.

The right reading of "B places 37–64 % more samples than the seed" is that this comparison is
**conservative**: B scores over a larger and therefore harder set and still matches.

## What this does and does not license

**Does:** build `camlab`. A fixed camera position is defensible for this clip, so the 0.899 m/frame
ground swim is surplus DOF rather than physics, and constraining the solve can remove it.

**Does not:** claim a shippable camera. This scores paint only — no SIFT/MAGSAC pan term, no jitter
budget. #119's own docstring records that a paint-only fit jitters *worse* than the free
homographies it replaces. And the fan clip's evidence floor is **8 px against the tripod's 1.7 px**:
even with 8 free DOF per frame the calibration does not sit on the paint, which caps what any
camera model can reach here and is a separate defect.
