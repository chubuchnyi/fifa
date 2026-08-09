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

**PnLCalib solves 8 free DOF per frame with nothing tying frames to one camera.**
Each homography is individually good (1.49 px on paint) and temporally smooth (0.008 m swim), and
the family is mutually incompatible with any single camera (471 px). Not noise — surplus DOF.
Smoothing cannot fix it; constraining the solve can.
· [`reply-swim-2026-08-09.md`](reply-swim-2026-08-09.md)

## Metrics quoted outside their window

**`jitter` from `smooth_residual` is honest to about 2 seconds.**
It fits a cubic across the whole span. The same clip reads **6.42 px over 60 frames** and
**60.42 px over 236** — the second is unmodelled camera motion, not noise, and it sent a day
after temporal instability that does not exist.
· **Check:** the printed value now carries `OUT OF DOMAIN` past 90 frames (`f94a32e`).

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

**Swim measures temporal consistency, not accuracy.**
The bench says so itself: an anchor displaced 10 m scores `swim 0.0000 m`. A calibration can be
perfectly smooth and uniformly wrong.

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
· **Live.** Independently reproduced 2026-08-09: `scene_off.json` 6 of 24 constant at exactly
0.92 m; `f236_res896.json` **0 of 38**, so the current path does not hit it.

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

**`pgrep -f batch.sh` over ssh matches the watcher itself.** Use `[b]atch.sh`, or grep the log for
`BATCH_FINISH_OK`.

**The pod repo `/workspace/fifa` is a stale mirror.** It reports already-pushed work as
uncommitted. Byte-compare against pushed HEAD before discarding anything.

**WSL kills background jobs when the launching `wsl.exe` exits.** Long work must run as a detached
container, which dockerd owns.
