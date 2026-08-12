# Landmines — the single register

**The rule: hit one, add a line here, same session, before moving on.** One file, so the next
reader finds it by looking in one place instead of grepping fourteen findings docs.

This exists because of #141: a capability or a fact is written down, and is not found at the
moment of need. Six instances in two days, four of them by the same reader who had already read
the docstring that warned about it. Scattering the warnings is what makes that the default outcome.

**What belongs here:** anything that made a run wrong, a number wrong, or a session expensive, and
that a reasonable person would not have predicted from the code in front of them. Not bugs with a
fix in flight — those live in `STATUS.md`. Not design decisions — those live in `adr/`.

**Format:** what it looks like when it bites · the check · status. Keep each to three lines. If it
needs more, write a findings doc and link it.

---

## Camera and calibration

**A scene can carry an invented camera and nothing on disk says so.**
`fx = 772.02 @ 1280×720` is `_static_camera`'s synthetic broadcast view, substituted whenever
`camera_from_calibration` refuses. Nine of nine scenes on disk carried it, including the one the
#135 eye labels were judged on — so every scene-to-source comparison in that thread was void.
· **Check:** `fx`, or since 2026-08-08 `CameraTrack.source` / `is_measured`.
· Marked (`794fd46`); `RIGID_CAMERA` now wired into `pod_real_e2e.sh` (`400e400`).

**`calib/<clip>.npz` covers only the frames it was fitted over.**
The shipped one is frames 0–59. `track_quality.py --camera` silently returns `None` outside that
range, and a subject with no projection reads as *off-frame* — an in-frame test that quietly
inverts. `apply_rigid_camera.py` refuses loudly instead; that is the pattern to copy.
· **Check:** `np.load(npz)['frames'].min(), .max()` against the scene's span.
· **Live** for `track_quality.py`.

**A hand-fitted camera npz is per clip and does not generalise.**
`apply_rigid_camera.py` took the overlay residual 240 px → 8 px on Colombia and does nothing for
any clip without an npz. A per-clip artefact is not a fix.
· **Check:** does the clip have `calib/<name>.npz`? If not, expect the synthetic camera.
· **Live** — no automatic path exists.

**A singular homography used to kill a whole run.**
PnLCalib can emit one; `np.linalg.inv` raises, and it killed a 236-frame GPU run twice, from two
different call sites. Such a frame is now marked `confidence = 0`, which `solved_mask` already
understood.
· **Check:** `FieldCalibration.degenerate_frames`.
· Fixed (`dfc1075`, `976fcf9`).

**`confidence` is anti-predictive, measured three times.**
Pearson **r = +0.699** against real paint error — the frames the pipeline trusts most are its
worst. It is exported in `scene.json` and consumed downstream. #105, #126, and again 2026-08-09.
· **Check:** do not rank frames by it. `confidence == 0.0` is still meaningful: never solved.
· **Live** as a defect, understood as a fact.

**Nothing records the image size the homographies live in — and the filename lies.**
`FieldCalibration` has no `width`/`height`; `camera_from_calibration` makes the caller supply them.
`14604731_1080_1920_30fps.mp4` is ingested as **1920×1080** (`vert_crop.mp4`), so reading the
dimensions off the source filename gives a plausible-looking wrong answer — 24 176 px of
reprojection instead of 47 685.
· **Check:** the run log's `== ingested` line, never the filename.
· **Live.**

**Handheld footage: one camera is not realizable, and segmenting does not rescue it.**
The fan clip zooms 1.66×, in a clean ramp-then-plateau, so segmentation looks like the answer. It
is not: the flat plateau alone still reduces at **13 607 px** against the tripod clip's 471 px on
the same code. The difference is **translation** — WorldPose GT says broadcast cameras translate
0.000 m in 89/89 clips, a phone translates every frame, and time-segmentation cannot remove a
per-frame effect. Positions survive (the pitch is a plane); a novel view does not exist.
· [`architecture-brief-2026-08-09.md`](architecture-brief-2026-08-09.md)

**A crop moves the principal point, and nothing here accounts for it.**
The optical axis stays where the lens put it; the crop moves the image around it. Cutting
`1080x608+0+1294` out of a 1080x1920 clip puts the axis at `cy = 1920/2 - 1294 = -334` — 638 px
above the crop, further than the crop is tall — while `camera_from_calibration` takes `cx, cy` as
the centre of whatever size it is handed. **Measured, not reasoned:** sweeping `cy` through
image-to-image SIFT homographies, which know nothing about any crop, puts the minimum at
**-334.0**, the arithmetic value to the decimal, against **2.4x worse** at the crop centre, in an
instrument precise to 0.05. Every `--crop auto` run carries this.
· **Check:** `source_height/2 - crop_y`, and compare with the `height/2` the fit was given.
· **Live here.** Fixed in camlab (`ClipInfo.principal_point`); found while scoping M2 there.

**`--crop auto` moves the homographies into the crop rect; the camera fit is still told the full
frame size.** `cli.py` does `replace(_c, crop=_fr.rect)` and leaves `ClipRef.width/height` at the
source size. Every adapter decodes through the crop (`adapters/io/frames.py:iter_clip_frames`), so
PnLCalib saw **1080×608**, while `controller.py:709` hands `camera_from_calibration`
`width=clip.width, height=clip.height` = **1080×1920** — a principal point placed 656 px outside an
image 608 px tall. **The 12 382 px quoted for the fan clip all week is measured in the wrong
space**; at the correct 1080×608 the same call returns 18 313 px, focal 2099 px. Both refuse, so no
verdict flips, but the number and the stored `fit_focal_px` are wrong. Settled at the same time:
this clip has **no** 180° roll — that roll is a property of the solved `CameraTrack`, not of every
calibration.
· **Check:** score the projected markings against the paint in each candidate space and compare
the **sample count** as well as the median — a wrong space projects most markings off-surface where
they go unscored, and posts a flattering median on the survivors (full 1080×1920 scored 30.59 px on
n=36 against the crop's 9.47 px on n=1271).
· **Live.** [`m1-handheld-centre-2026-08-10.md`](m1-handheld-centre-2026-08-10.md)

**Near-degenerate homographies pass every guard we have, including confidence.**
Fan clip frames 115 and 117 measure *mirrored* while the other 118 do not — which
`fit_rigid_camera.load_world_to_image` asserts is impossible. Not a mid-clip frame change:
`|det|` is **1.0e-6** and **5.3e-8** against a clip median of **3.4e-3**. The plane has collapsed
toward a line and the handedness test reads the wrong sign off it. Both clear
`plane_camera._SINGULAR_DET = 1e-12` by six orders of magnitude, and both carry ordinary confidence
— **0.475 and 0.394** against a clip median near 0.45.
· **Check:** relative, not absolute — `|det| < 1e-3 * median(|det|)` over the clip. An absolute
threshold cannot work; the scale of `|det|` depends on image size and world units.
· **Live.** No pipeline guard does this today.

**PnLCalib already computed a full camera per frame, and we throw it away in three lines.**
`_PnLCalibBackend.calibrate_frames` (`pnlcalib_backend.py:237`) returns `cam_params` with
`x_focal_length`, `principal_point`, `position_meters`, `rotation_matrix` — a complete pinhole.
`image_to_world_from_cam_params` (`calibration.py:623-657`) builds `K·[R|t]`, **drops the Z column,
inverts, and keeps only the 3×3 plane homography.** Downstream, 231 lines of `plane_camera.py` and
544 of `fit_rigid_camera.py` try to guess those same numbers back. Worse, the only consumer of that
path, `CameraModuleFieldCalibrator` (`calibration.py:661`), is **unreachable from `wiring.py`**,
which accepts `"fake"` and `"keypoints"` only — it is reachable solely via
`scripts/run_calib_eval.py --solver camera`. Caveat before over-reading this: a per-frame PnLCalib
camera is still free per frame, so it does not fix the swim by itself — but it is a far better
initialisation than a homography, and it gives a direct per-frame focal, which is what a zooming
clip needs.
· **Check:** `grep -n "\[0, 1, 3\]" src/pitch3d/adapters/models/calibration.py`
· **Live.** Found 2026-08-10 while scoping [`camlab-spec.md`](../camlab-spec.md).

**The phone clip's storm: 79 % of measured steps are physically impossible, and 83 % of it is the
ground.** Measured 2026-08-09 on `out/fan_auto`, consecutive frame pairs where *both* ends are
`measured`, split into a per-frame common-mode median vector across all subjects versus each
subject's residual. Fan clip: 0.832 m/frame median step against a 0.35 m/frame human limit, p95
4.177 m, worst 11.568 m (347 m/s); common mode 0.899 m/frame median. Broadcast control: 0.072
median, 1.1 % over the limit, common mode 0.042. **The camera-carry remedy was already on** —
`--camera-carry` defaults to 8 at all three layers (`cli.py:690`, `cli.py:129`, `wiring.py:114`),
so `pod_real_e2e.sh` not setting it leaves the default, not off. It removes 92 % of swim on the
tripod clip and does not touch this.
· **Check:** decompose into common mode before attributing swim to per-player error — 83 % of it is
not the players. Do not ask for an eye verdict on a scene until the common mode is under ~0.1 m.
· **Live.** This is what [`camlab-spec.md`](../camlab-spec.md) exists to attack.

**PnLCalib solves 8 free DOF per frame with nothing tying frames to one camera.**
Each homography is individually good (1.49 px on paint) and temporally smooth (0.008 m swim), and
the family is mutually incompatible with any single camera (471 px). Not noise — surplus DOF.
Smoothing cannot fix it; constraining the solve can.
· [`reply-swim-2026-08-09.md`](reply-swim-2026-08-09.md)

**PnLCalib squashes any clip that is not 16:9, and the resize is unconditional.**
`pnlcalib_backend.py:117` resizes whenever the width is not 960, and `:323` is
`Resize((540, 960))` — a *fixed* size, not a scale. The current probe clip is 1080×1920
portrait, so it reaches the net at 0.5× across and 0.28× down, an anamorphic squash it was
never trained on. This is the same shape as the 560×560 pose-crop defect.
· **Check:** letterbox to 16:9 before the resize, or measure on a 16:9 clip and say so.

**PnLCalib computes a dense pitch-line map every frame and throws it away.**
`pnlcalib_backend.py:124` is `get_line(heatmaps_l[:, :-1, :, :])`. The 23 kept channels are
*not* dense lines — each is two Gaussian blobs at one segment's endpoints. Channel 24, the one
being dropped, is the only dense pixelwise line map, already paid for. It carries no class
labels (all 23 classes are summed into it), so it needs the world table to label it.

**PnLCalib is GPL-2.0, version 2 only, and this repo is public.**
Verified from the licence text, not the GitHub badge. It is imported by dotted path from
`$PNLCALIB_REPO` and never vendored, which is the mitigation — **keep it that way.** Do not
vendor it, do not copy code out of it. Separately, every calibration weight in this stack is
SoccerNet-trained, and SoccerNet is research-only: *"Can I use the data from SoccerNet for
commercial purposes? A: No."*

**SoccerNet's class `'Goal left post left '` has a trailing space.**
In the code, the README and every derived dict. A `.strip()`-normalising parser deletes the
class silently. Our `pitch_plane_line_segments()` sidesteps it by carrying only the 17 straight
plane lines; anything keying all 28 will hit it.

## Metrics quoted outside their window

**Distance to a *sampled* model is not distance to the model.**
`bench_markings_vs_camera.py` first scored detected lines against `pitch_polylines`' 0.5 m
samples — tens of pixels apart in the near field — and charged a line up to half a sample
spacing for lying exactly on the model. It read **precision 2.3 % against recall 92 %**, an
impossible pair, which is the only reason it was caught. Rasterise and distance-transform the
model side. Same family as "radial binning invents slopes".

**`jitter` from `smooth_residual` is honest to about 2 seconds.**
It fits a cubic across the whole span. The same clip reads **6.42 px over 60 frames** and
**60.42 px over 236** — the second is unmodelled camera motion, not noise, and it sent a day
after temporal instability that does not exist.
· **Check:** the printed value now carries `OUT OF DOMAIN` past 90 frames (`f94a32e`).

**WorldPose GT is 50 fps and our broadcast clip is 29.97 — nothing warns you.**
Any per-frame delta and any fixed-length window mean different things at different rates. 60 GT
frames is 1.2 s against our 2.0 s, so comparing "60 frames" to "60 frames" understates GT by
**1.67×**. It inverted a verticality conclusion for several hours: pose jitter read as 2.3× GT and
is 1.37×, and the 2-second excursion deficit read as 3.5× and is 5.5×.
· **Check:** `ffprobe -show_entries stream=r_frame_rate`. Compare in **seconds**, and rescale
per-frame steps by `gt_fps / our_fps`. `bench_vertical_motion.py` now does both.
· Fixed in that bench; **live** for any new GT comparison.

**Excursion statistics are window-dependent, and the board compared two of them.**
Root-Z `max−min` grows with the window: GT medians are **0.028 / 0.085 / 0.204 m** at 60 / 236 /
1032 frames. "0.082 m in a whole scene against 0.23 m for a real player" put a 60-frame **maximum**
against a 1032-frame **median** — a 7× window mismatch that survived four plan revisions.
· **Check:** same window, same statistic, both sides.
· Reproduced 2026-08-09 on 25 GT clips: median root-Z range **0.028 / 0.084 / 0.210 m** at
60 / 236 / 1032 frames. **And the conclusion reverses.** At the matched window our current scene
`f236_res896` (236 frames, no constant-Z subjects) has median **0.160 m** against the GT's
**0.084 m** — nearly double, not flat. The "we have no vertical DOF" item rested on a 60-frame
maximum from a scene where a quarter of the subjects had the FK fallback firing.
· [`vertical-motion-2026-08-09.md`](vertical-motion-2026-08-09.md)

**The kit reader called the grass yellow.**
Yellow was `18 ≤ H ≤ 48, S > 90`; the floodlit pitch is `H 39–40, S ≈ 150`. **64.9 % of every
frame** classified as "yellow kit". It produced four false findings, one of which had already
reached a committed file.
· **Check:** point any colour threshold at a whole frame before trusting it.
· Fixed, 15 tests (`classify_kit`, grass rejected before the median).

**The principal point is assumed, not measured — and it is written out as if it were.**
`fit_rigid_camera.kmat()` hardcodes `cx, cy = W/2, H/2`, and `apply_rigid_camera.py:138` writes
that into every scene's `CameraIntrinsics` beside a genuinely fitted focal. PnLCalib *returns* a
`principal_point`; the fit that consumes its output drops it.
· **Measured 2026-08-10 and it is not identifiable here, so do not free it:** `cy` is flat over
±900 px (1.415–1.443 px) and `cx`'s minimum sits at **+600 px, 81 % across the frame**, with the
focal walking monotonically alongside it — a valley in *(cx, focal)*, not a measurement.
· [`camera-parameters-dropped-2026-08-10.md`](camera-parameters-dropped-2026-08-10.md)

**Radial binning invents a slope out of a localised band.**
The paint residual read 0.75 → 2.97 px centre-to-edge and looked like a lens. Binned by each axis
instead, `|v−cy|` goes 1.12 / **2.54** / 1.26 / **1.08** — it peaks near the middle and falls
toward the edge, which no lens can do. Before attributing a centre-to-edge rise to optics, bin
`|u−cx|` and `|v−cy|` separately; it costs one extra loop.
· And when sweeping a parameter that the radius is measured *from*, the slope is not comparable
between candidates — moving the centre re-bins the samples and flattens a profile for free.

**The render camera cannot roll, and has no `cx`/`cy` or `fov_y`.**
`core/scene/cameras.py` builds its basis as `right = cross(fwd, world_up)`, so the horizon is
always level and there is no field a roll could live in. `blender_animate.py` sets
`sensor_fit = "HORIZONTAL"`, so the vertical angle is a consequence of the render aspect. camlab's
panel offers `roll` — it would be silently dropped downstream.

**Swim measures temporal consistency, not accuracy.**
The bench says so itself: an anchor displaced 10 m scores `swim 0.0000 m`. A calibration can be
perfectly smooth and uniformly wrong.

**Quote a findings doc by commit hash — ours get corrected within the hour.**
`occlusion-stack-review-2026-08-07.md` was revised 69 minutes after it was first committed
(`06d3f00` 19:21 → `1250585` 20:30, +48 lines). A reply written against the first copy spent two
of its four objections on text the doc already contained. #141 in its purest form: written down,
did not reach the reader.
· **Check:** cite `path@hash` when a doc leaves this repo in a message.

**A findings doc does not know it has been superseded — `STATUS.md` is the current state.**
`occlusion-stack-review-2026-08-07.md` §8a says the detector-resolution knob was measured null and
*"the default stays 560"*. It was re-measured on **2026-08-08** against identity instead of
detection count, the conclusion **reversed** (896 beats 560 by 31–36 % of mid-pitch events), and
the default moved into `config/detector_resolution.yaml` — while the findings doc kept its
original sentence. I quoted that sentence to an outside reviewer two days after it stopped being
true.
· **Check:** before quoting a number out of `docs/findings/`, grep `STATUS.md` for the same topic.
Findings are dated evidence; STATUS is the verdict.

**Scoring a tracking change with a detection metric will retract a real win.**
The same episode: "players found per frame" moved +5 % and the knob was dropped. Mid-pitch identity
events moved −31 %. More boxes says nothing about whether the boxes already there stay on the same
person.

**Before writing "this has not been measured", grep `STATUS.md` and `scripts/`.**
An hour after adding the landmine above, I wrote that selective mask propagation was *"the cheapest
thing on the list that has not already been measured null"* — while `scripts/bench_assignment_margin.py`
sat in the tree and `STATUS.md` **W5** carried its verdict (per-frame lift 0.86–1.10×, per-track
5.32×, shelved as *cost, not quality*). Seventh instance of #141. The doc-hash rule covers documents
leaving the repo; it does not cover **not reading our own board before a claim about what is
unknown**.
· **Check:** `grep -n "<topic>" docs/STATUS.md && ls scripts/ | grep -i <topic>` — two commands.

## Pipeline wiring

**Two reconstruction entry points apply different fixes.**
`pod_real_e2e.sh` and `pod_make_video.sh` both drive the controller and do not agree. That
produced #140 (rigid camera never applied) and the vert137 crop collapse.
· **Check:** which script produced the scene? Nothing in `scene.json` records it.
· **Live** — item C of the plan.

**An absent flag is not an off flag.**
`pod_real_e2e.sh` passes `--camera-carry` only when `CAMERA_CARRY` is set, but the argparse default
is **8**, so carry runs regardless. A missing `== camera carry:` log line says nothing about
behaviour.
· **Check:** the CLI default, not the shell conditional.

**`broadcast_crop.py` emits one rect per framing, not one per clip.**
"Each segment is a **separate reconstruction**: its calibration belongs to its own pixels."
vert137 fed one crop across 355 frames. The same docstring also warns that past frame ~155 the
clip zooms until only the goal mouth is left and the plane is undetermined — rediscovered by
measurement two days later.
· **Live.**

**Root Z silently becomes a constant when SMPL-X FK is absent.**
`pose.py:334` substitutes the nominal `pelvis_height_m` whenever the backend returns no
`pelvis_above_foot`, and nothing records it. In `out/cue/scene_off.json` — the scene the #135 eye
labels were judged on — **6 of 24 subjects have exactly constant Z at 0.92 m**, per-frame `|dZ|`
median 0.0000. Same shape as #140.
· **Check:** `np.std(transl[:,2]) < 1e-9` per subject, or `bench_vertical_motion.py`.
· **Fixed 2026-08-09.** Marked (`root_z_source`, `nominal_root_z`, a CLI line) **and measured**:
`make_smplx_pose_height_provider` now runs SMPL-X FK at the pose stage when the backend reports no
height. The measurement existed the whole time — `smplx_foot_z.py` — wired only into `foot_plant`
and `body_scale_probe`, gates that run *after* the scene is assembled, never into the one place
root Z is decided. **I then asserted in writing that "a backend without SMPL-X FK cannot know the
offset", which a two-minute grep disproves.** #141 inside an explanation of #141.
· Independently reproduced 2026-08-09: `scene_off.json` 6 of 24 constant at exactly
0.92 m; `f236_res896.json` **0 of 38**, so the current path does not hit it.

**Every subject spans the whole clip, however little was measured.**
`extend_to_span` (on by default, #102) extends each subject to the union of all present frames, and
turning it on also raises the interior cap from `max_fill_gap` to the full span — so neither edge
nor middle is bounded. `f236_res896`: 38 subjects, all 236 frames, **median 37 % measured, worst
2 %** — 5 real frames and 231 held. **47.9 %** of its subject-frames sit further than 12 frames from
ANY measurement, 11.9 % further than 120; worst 228. `fan_auto`: 41.4 % beyond 12. Decay bounds the
coasted distance and `coast_max_speed` the velocity; nothing bounded the **time**, which is why the
eye keeps finding phantoms standing around.
· **Check:** per subject, `measured/total` from `track_quality.py`; per frame, distance to the
nearest `measured` row. Or the new `== coherence: … edge reach:` log line.
· **Bounded 2026-08-09** (`cec52ae`): `CoherenceConfig.max_extend_frames`, clamped per subject.
**Default is still `None` = unbounded** — the cap has to be asked for until the eye has judged an
A/B, so every scene written before that date still has this shape.

**Grass fraction does not predict solvability.**
On the fan clip the *worst*-cropped segment (82.4 % grass) solved **98 %** of its frames and the
*best* (91.7 %) solved **9 %**. 93 % of all unsolved frames are one contiguous run where the zoom
leaves no landmarks. Cropping is not the lever it looks like.
· **Live.**

**A guard that demands a quorum rejects working input.**
`_MIN_FIT_FRAMES = 4` broke 7 shipped tests, which fit a 3-frame synthetic scene. A guard's job is
to reject *nothing*, not to require a majority.

## Identity and tracks

**`require_same_team` treats `None` as a wildcard.**
An unlabelled subject is stitchable to anyone. The identity gate used to blank `team_id` on every
track it cleaned, so 23 of 27 subjects became universally stitchable.
· Fixed (`6f4c270`).

**`clip_hash` hashes only the first frame, the last frame and the count.**
Two different frame subsets with the same endpoints and length share a cache key.
· **Live.**

**ultralytics track ids come off a CLASS-level counter, so a second run continues the first.**
`BaseTrack._count` is shared by every tracker instance in the process. Constructing a fresh
`BOTSORT` does not reset it: an A/B in one process gives the two arms disjoint id spaces, and any
metric that keys on id compares two different numberings. `BotSortBackend.associate` calls
`BaseTrack.reset_id()` before each run for exactly this reason — do the same in any script that
drives an ultralytics tracker directly.
· **Check:** the second arm's lowest track id should be the same as the first arm's.

**The tracker→stitcher path has zero run noise, so a "band" of results is signal, not scatter.**
Three independent runs of `bench_expansion_iou.py` at `scale=1.0` give **56 / 36 / 14** every time.
A sweep that wanders 33–38 across parameter values is a deterministic **non-monotonic** response —
a small change to the cost matrix reshuffles which merges the stitcher then makes. Calling it
noise (I did, in the §5 correction) excuses a result that actually needs explaining.
· **Check:** run the same config twice before attributing spread to noise.
· [`reply-occlusion-stack-2026-08-10.md`](reply-occlusion-stack-2026-08-10.md)

## Fakes and tests

**A fakes run writes a file of exactly the same shape as a real one.**
`FakeFieldCalibrator` is a plain affine scale of the image — **no perspective at all**.
`FakePoseEstimator` writes `body_pose = zeros`, a T-pose on every frame, which makes any
articulation number vacuous.
· **Check:** `track_quality.py` prints a banner when it sees either.

**A green suite is a smoke signal, not evidence.**
`tests/conftest.py` is fakes-backed by design. The one test backed by a real measurement is
`tests/e2e/test_golden_real_camera.py`.

**`pyproject` already sets `-q`.**
Adding another makes `-qq`, which swallows the "N passed" summary line. That is how a ">5 min"
claim about the suite survived unchallenged.

## Front end

**Alpine's `:style` as a template string clobbers `x-show`.**
A string binding rewrites the whole style attribute, wiping the `display:none` that `x-show` wrote.
The overlay reappeared on every frame change while its flag still read "hidden". Object bindings
set properties individually and are safe.
· Fixed 2026-08-08.

**Editing `poseannot/static/style.css`? Bump the `?v=` token in `index.html`.**
Currently `?v=layout8`. Without it the browser verifies the old stylesheet and a fix is judged
against the file it replaced.

## Boxes and infrastructure

**ruff is pinned in four places.** `[dev]`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`,
`requirements-dev.txt`. The fourth was missed once and CI checked a different rule set.

**`lint_changed.py` lies in two directions, and neither looks like an error.**
The hook is `entry: python scripts/lint_changed.py`, `language: system` — a shell with only
`python3` on PATH aborts the commit with *"Executable `python` not found"*, which reads as a lint
failure and is not. And run by hand **before** `git add`, the script prints *"no Python files
changed"* — it reads the index, so an unstaged edit reads as clean.
· **Check:** `PATH="$PWD/.venv/bin:$PATH" git commit …`, and stage before you trust the verdict.

**`pgrep -f batch.sh` over ssh matches the watcher itself.** Use `[b]atch.sh`, or grep the log for
`BATCH_FINISH_OK`.

**The pod repo `/workspace/fifa` is a stale mirror.** It reports already-pushed work as
uncommitted. Byte-compare against pushed HEAD before discarding anything.

**WSL kills background jobs when the launching `wsl.exe` exits.** Long work must run as a detached
container, which dockerd owns.
