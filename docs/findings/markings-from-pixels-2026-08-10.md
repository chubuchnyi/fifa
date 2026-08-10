# Reading the pitch markings off the pixels — what works, what a model would add, what it would cost

**2026-08-10**, from the request: *«хочу отдельный алгоритм, который на всех фрэймах нарисует
оверлэем разметку при помощи компьютерного зрения. opencv5 или модель»*.

Answer up front: **OpenCV, and it already works.** A model buys exactly one thing that classical
CV cannot give — *names* for the lines — and every model that gives it fails a licence gate. The
detector is committed and measured; the naming step is the open question and it has a cheap route
that does not need a new model at all.

---

## 1. Why this is worth having at all

Everything else in the repo that draws markings projects the pitch model through a solved camera.
That means it can only ever agree with itself: if the calibration is wrong, the overlay is wrong
in the same way and still looks confident. `scripts/detect_markings.py` never sees the camera, the
calibration or the pitch model. It reads pixels.

So it is the independent side of every overlay check — and it is the only thing here that runs on
a clip with no calibration at all.

The inventory that preceded this found two gaps worth naming: **no script in the repo writes a
video with the markings drawn on it**, and `--render overlay` — the mode whose name promises it —
draws only subject and ball markers, no pitch lines at all.

## 2. What the pixels already give (committed, `8c9c859`)

Three stages, none of them new science:

1. **paint** — `poseannot.pitch_evidence._masks`, the repo's canonical ridge filter with the
   turf-on-both-sides test. Deliberately reused rather than re-implemented: there are already
   four separate copies of the paint idea in the tree and a fifth would be debt.
2. **extent** — LSD (present and working in the installed OpenCV 5.0.0) over the ridge band, then
   a length floor. Extent is the discriminator, and `bench_camera_swim.py:364` had already
   measured why: brightness alone marks 2.5 % of the pitch, a bare ridge filter is *worse* at
   3–8 % because floodlit grass texture is full of short ridges, and length takes it to 0.8 %.
3. **carry** — frames with no evidence keep the last measurement, warped by pixel motion, drawn
   dimmed and labelled on the frame. R-6: marked, never hidden.

**Coverage over every frame of both clips, 224 ms/frame on CPU, no GPU, no weights:**

| clip | frames | measured | carried | blind |
|---|---|---|---|---|
| fan portrait 1080×1920 | 355 | **100 %** | 0 | 0 |
| broadcast 1920×1080 | 334 | 88 % | 12 % | 0 |

The broadcast clip's 12 % is not a detector failure — it is a **cut**. One contiguous 2 s block
(f272–330) plus three short ones, all of them the goalmouth close-up after the goal, where the
playing surface drops from 51 % of the frame to 31 % and no marking is long enough to see. A
detector that reported markings there would be inventing them.

Note the ordering: the *fan phone clip does better than the broadcast*, 100 % against 88 %. The
paint is closer, higher-contrast and less often cut away from.

### Two constants, swept rather than guessed

`MIN_LEN_PX` was set with the inside-the-net frame as the false-positive control — it contains no
real marking at all, so all 1080 of its LSD segments are noise:

| MIN_LEN | fan f0 | fan f50 | bcast f67 | **NET f333 (all false)** |
|---|---|---|---|---|
| 30 | 16 | 14 | 16 | **136** |
| 40 | 10 | 11 | 10 | **52** |
| **60** | 11 | 11 | 8 | **3** |
| 80 | 9 | 6 | 7 | **0** |

80 is where the net finally reaches zero and it is not worth it: it also deletes 25–31 % of the
real markings on every ordinary frame to clean up three false segments on one frame that has no
markings to get wrong.

The merge step needed a gap limit, not just collinearity. Without one it joined the goal-area line
to an unrelated fragment 500 px away and drew a diagonal straight across the penalty area that no
marking follows.

## 3. The detector and the camera now check each other

`scripts/bench_markings_vs_camera.py` — the comparison `bench_overlay_residual.py`'s header has
promised since it was written without ever implementing. Over **all 60 calibrated frames**:

| | |
|---|---|
| **recall** — model markings the detector finds | **97.8 %** (p10 94.7) |
| **precision** — detections the camera explains | **100.0 %** (zero unexplained) |

Precision is the load-bearing one. The detector cannot invent a straight 60 px line on the playing
surface out of nothing, so a confident detection with no model marking under it would have been
evidence against the camera. There are none.

> The first version of that bench read **precision 2.3 % against recall 92 %**. That pair is
> impossible, which is the only reason the bug was caught: it scored detections against
> `pitch_polylines`' 0.5 m samples, tens of pixels apart in the near field. Same family as
> "radial binning invents slopes" — recorded in `landmines.md`.

## 4. What a model would add, and what it costs

The one thing classical CV does not give is **which line is which**. Researched across every
released candidate; the table that matters is licence and weight availability, not accuracy.

| model | output | code licence | weights | dense per-line-class? |
|---|---|---|---|---|
| **PnLCalib** (already wired here) | 57 kp + 23 line-*extremity* heatmaps | **GPL-2.0, v2 only** | yes | **no** — endpoint blobs |
| SoccerNet baseline | dense **29-class** line seg | **no licence file** | yes | **yes** |
| TVCalib | dense 29-class line seg | MIT wrapper, **unlicensed submodule** | yes | **yes** |
| Spiideo soccersegcal | 6 **region** masks | **MIT** | yes | no — regions |
| Roboflow sports | 32 keypoints | MIT code, **AGPL-3.0 weights** | yes | no |
| RF-DETR Keypoint | arbitrary keypoints | **Apache-2.0 code *and* weights** | yes | no — train your own |

**You can have any two of the three.** The two genuinely dense models are the same lineage
(`deeplabv3_resnet*(num_classes=29)`) and neither carries a usable licence. The one with a clean
MIT licence outputs regions, not line classes.

A second gate catches all of them anyway: every weight above is SoccerNet-trained, and SoccerNet
is research-only — *"Can I use the data from SoccerNet for commercial purposes? A: No."*

**Speed, measured, not quoted:** PnLCalib is 131.9 M params across two HRNet heads and runs
**~4.2 s/frame on CPU** (2.0 s + 2.1 s at 8 threads; 16 threads is ~3× worse, not better). On GPU
the shipped default path is 439 ms/frame. Against 224 ms/frame for the whole classical stack.

### Three things found in our own code while checking this

1. **`pnlcalib_backend.py:124` throws away a dense line map we already pay for.**
   `get_line(heatmaps_l[:, :-1, :, :])` drops channel 24 — the only dense pixelwise line map in
   the output. The 23 channels it keeps are *not* dense lines; each is two Gaussian blobs at one
   segment's endpoints. The dropped channel has no class labels (all 23 are summed into it), but
   we have a 17-class world table in `core/scene/pitch.py` and a keypoint homography, so it can
   be labelled by reprojection. **One array slice, zero extra compute.**
2. **`pnlcalib_backend.py:323` resizes to a fixed 540×960 regardless of aspect ratio.** The
   current probe clip is 1080×1920 portrait, so it reaches the net anamorphically squashed —
   0.5× across, 0.28× down. Same shape as the 560×560 pose-crop defect.
3. **PnLCalib is GPL-2.0 and this repo is public.** It is imported by dotted path from
   `$PNLCALIB_REPO` and never vendored, which is the mitigation. Keep it that way.

## 5. What this does not claim

- **That the markings are labelled.** They are not. The overlay says "there is a painted line
  here", not "this is the goal line". Everything in §4 is about closing that, and none of it is
  done.
- **That 100 % on the fan clip generalises.** Two clips is two clips. The broadcast number (88 %)
  is the one that shows the failure mode, and it is a shot cut, not a lighting or pitch change.
- **That precision 100 % validates the camera everywhere.** It validates it on the 60 frames the
  calibration covers, against the markings that are visible there. The close-up block has neither.
- **Anything about distortion.** The detector is 2-D and never forms a camera.
