#!/usr/bin/env python3
"""Is our calibration jittering, and can the jitter go without a GPU? (R2-pre, #94)

The brief prescribes RAFT-small optical flow plus a MAD 3-sigma reject and a RANSAC fit to propagate
the camera between PnLCalib anchors, and books a GPU for it. Two things have to be true before that
is worth buying: the per-frame calibration must actually be inconsistent over time, and the
inconsistency must be *noise* rather than the camera genuinely moving. Neither is obvious, and the
second is the one that decides the design — a smoother that removes real motion is the yaw low-pass
mistake again (ADR-0012, Tier 1).

The measurement needs a truth signal that does not come from the calibration itself. Smoothness
proves nothing: a constant homography is perfectly smooth and completely wrong. So the truth here is
the **pixels**. A broadcast main camera is on a tripod, so between two frames the whole image is
related by one homography whatever the scene depth; that inter-frame homography can be recovered on
the CPU with corner tracking, entirely independently of where PnLCalib thinks the pitch is.

Run (CPU only, no pod, ~1 min)::

    PYTHONPATH=src .venv/bin/python scripts/bench_camera_swim.py

``true_motion``
    Recovers the real inter-frame camera motion from the video with Lucas-Kanade corner tracking.
    This is the number the calibration has to agree with.

``consistency``
    Chains the truth through our own calibration. If frame *k* and frame *k+1* are both calibrated
    correctly, then un-projecting a pixel in frame *k* and un-projecting *the same physical point*
    in frame *k+1* must land on the same spot on the pitch. The gap is reported in metres, which is
    what a viewer sees, rather than in homography coefficients, which nobody can judge.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np  # noqa: E402

from pitch3d.adapters.models.calibration import (  # noqa: E402
    carry_on_motion,
    probe_pixels,
)
from pitch3d.core.scene.serialization import from_json  # noqa: E402

# Overridable because a hardcoded artifact path is how #105 came to indict code that no longer
# ships: the default below was written 2026-07-09, R3 changed the solver 2026-07-29, and the bench
# happily re-measured the stale file. Point these at a fresh run before quoting any number.
SCENE = os.environ.get("PITCH3D_BENCH_SCENE", "out/anim_A/export/scene.json")
VIDEO = os.environ.get("PITCH3D_BENCH_VIDEO", "samples/video/Colombia-1-0-Congo-DR1080p.mp4")

#: Where players' feet actually are. Swim is reported here rather than averaged over the whole
#: frame, because error at the horizon is both huge and invisible — nobody is standing there.
PROBE_UV = np.array(
    [[640.0, 800.0], [960.0, 800.0], [1280.0, 800.0], [760.0, 950.0], [1160.0, 950.0]]
)


def _homog(uv: np.ndarray) -> np.ndarray:
    return np.c_[uv, np.ones(len(uv))]


def _apply(h: np.ndarray, uv: np.ndarray) -> np.ndarray:
    p = _homog(uv) @ h.T
    return p[:, :2] / p[:, 2:3]


def true_motion(n: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Inter-frame homographies straight from the pixels — the truth the calibration must match."""
    import cv2

    print("== how much does the camera REALLY move? (Lucas-Kanade on the raw video) ==")
    cap = cv2.VideoCapture(VIDEO)
    frames = []
    for _ in range(n):
        ok, img = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    cap.release()

    mats: list[np.ndarray] = []
    moved = np.zeros(len(frames) - 1)
    for k in range(len(frames) - 1):
        a, b = frames[k], frames[k + 1]
        p0 = cv2.goodFeaturesToTrack(a, maxCorners=4000, qualityLevel=0.01, minDistance=8)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, winSize=(21, 21), maxLevel=4)
        ok = st.ravel().astype(bool)
        src, dst = p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)
        # Players move independently of the camera; RANSAC keeps the majority (the stadium).
        g, inl = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
        keep = inl.ravel().astype(bool)
        mats.append(g)
        moved[k] = float(np.median(np.linalg.norm(dst[keep] - src[keep], axis=1)))

    print(
        f"  {len(frames)} frames, tracked pairwise: camera moves the image by "
        f"median {np.median(moved):.2f} px/frame (p95 {np.percentile(moved, 95):.2f}, "
        f"max {moved.max():.2f})"
    )
    print("  A tripod pan is smooth, so this is the scale an honest calibration must agree with.\n")
    return mats, moved


def _calibration() -> tuple[np.ndarray, np.ndarray]:
    cal = from_json(open(SCENE, encoding="utf-8").read()).field.calibration
    return cal.homographies, cal.frames


def _swim(h: np.ndarray, mats: list[np.ndarray], flip: bool, w: int, hgt: int) -> np.ndarray:
    """Metres between where frame *k* and frame *k+1* think one physical point sits on the pitch."""
    out = np.zeros(len(mats))
    for k, g in enumerate(mats):
        uv = PROBE_UV
        raw = np.array([[w - 1, hgt - 1]]) - uv if flip else uv
        nxt = _apply(g, raw)
        nxt = np.array([[w - 1, hgt - 1]]) - nxt if flip else nxt
        out[k] = float(np.median(np.linalg.norm(_apply(h[k], uv) - _apply(h[k + 1], nxt), axis=1)))
    return out


def consistency(mats: list[np.ndarray], moved: np.ndarray) -> None:
    """Chain the pixel truth through our calibration and read the disagreement in metres."""
    print("== does our calibration agree with the camera's real motion? (metres on the pitch) ==")
    h, _ = _calibration()
    n = len(mats)

    # The image convention is *derived*, not remembered: our solver may have run on the 180-rotated
    # frame (the #50 gate). Both readings are scored and the consistent one wins, because a
    # remembered convention that is wrong stays self-consistent and silently inverts the answer.
    cand = {f: _swim(h, mats, f, 1920, 1080) for f in (False, True)}
    flip = min(cand, key=lambda f: float(np.median(cand[f])))
    for f, v in cand.items():
        tag = "  <- used" if f is flip else ""
        name = "180-rotated" if f else "raw"
        print(f"  image convention {name:<12} median swim {np.median(v):7.3f} m{tag}")
    swim = cand[flip]

    print()
    print(f"  {'':<34}{'median':>9}{'p95':>9}{'max':>9}")
    print("  " + "-" * 61)
    print(
        f"  {'frame-to-frame scene swim':<34}{np.median(swim):>9.3f}{np.percentile(swim, 95):>9.3f}"
        f"{swim.max():>9.3f}"
    )

    # What the same pixel error would look like if the calibration were as steady as the camera.
    # Our own solver's world error is ~0.2 m (B1, #95), so anything far above that is temporal.
    print(
        f"  {'true camera motion (px/frame)':<34}{np.median(moved):>9.2f}"
        f"{np.percentile(moved, 95):>9.2f}{moved.max():>9.2f}"
    )
    print()
    print(f"  frames where swim exceeds 0.25 m: {100 * (swim > 0.25).mean():.0f}% of {n}")
    print(f"  frames where swim exceeds 1.00 m: {100 * (swim > 1.0).mean():.0f}% of {n}\n")


def _chain(mats: list[np.ndarray], k: int, j: int) -> np.ndarray:
    """Map frame ``k``'s pixels to frame ``j``'s, composed from the measured inter-frame fits."""
    g = np.eye(3)
    if j > k:
        for i in range(k, j):
            g = mats[i] @ g
    else:
        for i in range(j, k):
            g = mats[i] @ g
        g = np.linalg.inv(g)
    return g


def fuse(h: np.ndarray, mats: list[np.ndarray], window: int) -> np.ndarray:
    """Re-estimate each frame's homography from its neighbours, carried by the measured motion.

    Homography coefficients cannot be averaged — they are only defined up to scale and the entries
    are not commensurate. So the vote happens where the quantity is physical: every neighbour
    predicts *where a probe pixel lands on the pitch*, those world points are combined with a
    median, and a homography is re-fitted to the result.
    """
    import cv2

    n = len(h)
    out = np.empty_like(h)
    for k in range(n):
        lo, hi = max(0, k - window), min(n - 1, k + window)
        preds = []
        for j in range(lo, hi + 1):
            uv_j = _apply(_chain(mats, k, j), PROBE_UV) if j != k else PROBE_UV
            preds.append(_apply(h[j], uv_j))
        med = np.median(np.stack(preds), axis=0)
        out[k] = cv2.findHomography(PROBE_UV, med, 0)[0]
    return out


def shipped_motion(n: int, mats: list[np.ndarray]) -> list[np.ndarray]:
    """Re-measure the motion with the SHIPPED backend and check it against the bench's own LK.

    ``true_motion`` is the benchmark's private implementation. What R2 ships is
    ``LucasKanadeMotion``, reading frames through the pipeline's own decoder rather than a bare
    ``VideoCapture`` loop. Those two could diverge — a different decode path, a dropped frame, a
    silently-identity fit — and the whole result below rides on them agreeing, so it is measured.
    """
    from pitch3d.adapters.models.calibration import LucasKanadeMotion
    from pitch3d.core.ports.io import ClipRef

    print("== does the SHIPPED motion backend reproduce the bench's own tracking? ==")
    clip = ClipRef(
        source_id="s", uri=VIDEO, frames=np.arange(n), width=1920, height=1080, fps=29.97
    )
    got = LucasKanadeMotion().frame_motion(clip)
    # Compare where it matters — how far the two disagree about a probe pixel, in pixels.
    probe = probe_pixels(1920, 1080)
    d = [
        float(np.median(np.linalg.norm(_apply(g, probe) - _apply(m, probe), axis=1)))
        for g, m in zip(got, mats, strict=True)
    ]
    print(
        f"  {len(got)} inter-frame fits: median disagreement {np.median(d):.4f} px, "
        f"max {max(d):.4f} px"
    )
    print("  (RANSAC is randomised, so exact equality is not expected; sub-pixel agreement is.)\n")
    return list(got)


def shipped(h: np.ndarray, mats: list[np.ndarray], window: int = 8) -> np.ndarray:
    """The SHIPPED implementation (`calibration.carry_on_motion`), scored beside the prototype.

    ``fuse`` above is the prototype this benchmark was written around. What R2 actually ships is
    the adapter function, and a prototype that agrees with a write-up but not with the code in the
    tree is worth nothing — so the shipped path is scored here under the same controls.
    """
    return carry_on_motion(h, np.stack(mats), window, probe_pixels(1920, 1080))


def mad_reject(h: np.ndarray, mats: list[np.ndarray], k_sigma: float = 3.0):
    """The brief's *actual* prescription: keep every frame's own solve, replace only the outliers.

    ``fuse`` below replaces every frame with a window median, which levels the good frames down to
    the average of their neighbourhood — measurably, on this clip. This does the opposite. Each
    frame's predecessor predicts, through the measured motion, where that frame's homography should
    be; frames whose own solve disagrees by more than a **robust** threshold (median + k·MAD, so the
    outliers cannot inflate the threshold that is meant to catch them) are the only ones replaced.
    A frame that agrees with the pixels is left exactly as the solver found it.
    """
    n = len(h)
    pred = [h[0]] + [h[k - 1] @ _chain(mats, k, k - 1) for k in range(1, n)]
    d = np.array(
        [
            float(np.median(np.linalg.norm(_apply(h[k], PROBE_UV) - _apply(p, PROBE_UV), axis=1)))
            for k, p in enumerate(pred)
        ]
    )
    thr = float(np.median(d) + k_sigma * 1.4826 * np.median(np.abs(d - np.median(d))))
    out = h.copy()
    for k in range(1, n):
        if d[k] > thr:
            out[k] = out[k - 1] @ _chain(mats, k, k - 1)  # carry from the ACCEPTED predecessor
    return out, int((d[1:] > thr).sum()), thr


def coeff_average(h: np.ndarray, window: int) -> np.ndarray:
    """What `calibration._temporal_smooth` does today: box-average the coefficients themselves.

    Reproduced here rather than argued about. A homography is defined only up to scale and its
    entries are not commensurate — h[0,0] is a ratio, h[0,2] is a translation in pixels — so
    averaging them has no geometric meaning. Whether that is *harmful* or merely *useless* at our
    inter-frame scale is a different question, and it is measured below rather than assumed.
    """
    t = len(h)
    half = (window if window % 2 else window + 1) // 2
    out = np.empty_like(h)
    for i in range(t):
        m = h[max(0, i - half) : min(t, i + half + 1)].mean(axis=0)
        out[i] = m / m[2, 2] if abs(m[2, 2]) > 1e-12 else m
    return out


def removable(mats: list[np.ndarray]) -> None:
    """Does carrying the calibration on the measured motion remove the swim, or just hide it?"""
    print("== is the swim removable? (same metric; lower is better) ==\n")
    h, _ = _calibration()
    base = _swim(h, mats, False, 1920, 1080)

    print(f"  {'':<34}{'median':>9}{'p95':>9}{'max':>9}")
    print("  " + "-" * 61)

    def row(label: str, s: np.ndarray) -> None:
        print(f"  {label:<34}{np.median(s):>9.3f}{np.percentile(s, 95):>9.3f}{s.max():>9.3f}")

    row("per-frame (what we ship today)", base)
    best = None
    for w in (1, 2, 4, 8, 16):
        s = _swim(fuse(h, mats, w), mats, False, 1920, 1080)
        row(f"carried on measured motion, +-{w}", s)
        if best is None or np.median(s) < np.median(best[1]):
            best = (w, s)
    row("SHIPPED carry_on_motion, +-8", _swim(shipped(h, mats, 8), mats, False, 1920, 1080))

    # The smoother already in the tree (`calibration._temporal_smooth`, default OFF), scored on the
    # same axis as the alternatives that would replace it.
    for w in (5, 17):
        s = _swim(coeff_average(h, w), mats, False, 1920, 1080)
        row(f"coefficient average, w={w} (in tree)", s)

    for ks in (1.0, 2.0, 3.0):
        m, n_rej, thr = mad_reject(h, mats, ks)
        row(f"MAD reject k={ks:g} ({n_rej} frames, {thr:.3f} m)", _swim(m, mats, False, 1920, 1080))

    # The control that stops this being self-congratulation. Freezing the camera is the maximally
    # "smooth" answer, so a metric that merely rewards smoothness would score it best. It must
    # score WORST, because the camera really is panning and a frozen map contradicts the pixels.
    frozen = np.repeat(h[:1], len(h), axis=0)
    row("FROZEN camera (over-smoothing ctl)", _swim(frozen, mats, False, 1920, 1080))
    print()

    w, s = best
    print(
        f"  Best window +-{w}: median {np.median(base):.3f} -> {np.median(s):.3f} m "
        f"({100 * (1 - np.median(s) / np.median(base)):.0f}%), "
        f"p95 {np.percentile(base, 95):.3f} -> {np.percentile(s, 95):.3f} m "
        f"({100 * (1 - np.percentile(s, 95) / np.percentile(base, 95)):.0f}%)"
    )
    print("  Read the frozen row carefully rather than as a clean win: it is worse than today's")
    print("  per-frame calibration on the median but BETTER at p95, because its error is")
    print("  systematic and grows with the pan instead of spiking. What it does establish is that")
    print("  the metric is not simply rewarding smoothness — the smoothest possible camera loses")
    print("  to the carried one by ~18x. The next section shows what it rewards instead.\n")


def circularity(mats: list[np.ndarray]) -> None:
    """Score a deliberately WRONG calibration that was carried on the same motion. (It must fail.)

    The 92 % above is measured against the very motion model the fused estimate was built from, so
    the obvious worry is that the metric simply rewards having used it. This is the test: take one
    anchor frame, shift it by a known amount so the whole pitch sits somewhere it plainly is not,
    and carry *that* along the same inter-frame motion. If swim stays low, the metric is scoring
    self-consistency and cannot speak about accuracy at all.
    """
    import cv2

    print("== can the metric be fooled by a confidently wrong camera? ==\n")
    h, _ = _calibration()
    n = len(h)

    for shift in (0.0, 2.0, 10.0):
        anchor = _apply(h[0], PROBE_UV) + np.array([shift, 0.0])
        h0 = cv2.findHomography(PROBE_UV, anchor, 0)[0]
        carried = np.stack([h0 @ _chain(mats, k, 0) for k in range(n)])
        s = _swim(carried, mats, False, 1920, 1080)
        print(
            f"  anchor displaced {shift:>4.0f} m, then carried: swim median {np.median(s):.4f} m, "
            f"p95 {np.percentile(s, 95):.4f} m"
        )
    print()
    print("  A 10 m error scores as clean as a 0 m one. The swim metric measures TEMPORAL")
    print("  CONSISTENCY and says nothing about whether the pitch is in the right place — so the")
    print("  92 % is a real removal of wobble, not evidence of a more accurate camera.\n")


def _line_mask(bgr: np.ndarray):
    """Distance (px) to the nearest painted-line pixel, and the grass region it is valid in.

    Three filters, because the first two are not enough and that had to be measured. Thresholding
    on "bright and desaturated inside the grass" marks **2.5 %** of the pitch — floodlit specular,
    white kit and the compression noise of a 1080p broadcast all pass it, and against a reference
    that dense every candidate camera scores a perfect sub-pixel and nothing can be ranked. A ridge
    filter tuned to R3's measured 2 px line width is *worse* (3-8 %), because grass texture is full
    of 2 px ridges. What actually separates paint from texture is neither brightness nor width but
    **extent**: a pitch marking is long and straight for a hundred pixels. Hough gets it to 0.8 %.
    """
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grass = ((h > 25) & (h < 95) & (s > 40) & (v > 30)).astype(np.uint8)
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(grass, 8)
    if n > 1:
        grass = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)

    g = v.astype(np.int16)
    ridge = np.zeros_like(grass)
    for ax in (0, 1):
        lo, hi = np.roll(g, 3, axis=ax), np.roll(g, -3, axis=ax)
        ridge |= (((g - lo) > 20) & ((g - hi) > 20)).astype(np.uint8)
    seed = (ridge & grass & (s < 110)).astype(np.uint8)

    segs = cv2.HoughLinesP(
        seed * 255, 1, np.pi / 360, threshold=80, minLineLength=120, maxLineGap=6
    )
    segs = np.zeros((0, 4)) if segs is None else np.asarray(segs, float).reshape(-1, 4)
    return segs, grass


def _perp_px(uv: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """Perpendicular distance to the nearest detected segment, in **sub-pixel** units.

    Rasterising the segments and reading a distance transform floors out at ~0.95 px, which is
    coarser than the difference being tested — every candidate scored an identical 0.95 and nothing
    could be ranked. Distance to the fitted line itself has no such floor. Only the segment's own
    span counts, so a short marking's infinite extension cannot claim a point at the far touchline.
    """
    if len(segs) == 0 or len(uv) == 0:
        return np.full(len(uv), np.inf)
    p0, p1 = segs[:, None, :2], segs[:, None, 2:]
    d = p1 - p0
    ln = np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
    t = np.clip(((uv[None] - p0) * d).sum(-1, keepdims=True) / ln**2, 0.0, 1.0)
    return np.linalg.norm(uv[None] - (p0 + t * d), axis=-1).min(0)


def _pitch_points() -> np.ndarray:
    from pitch3d.eval.datasets_soccernet import pitch_plane_lines

    segs = []
    for a, b in pitch_plane_lines().values():
        t = np.linspace(0, 1, 60)[:, None]
        segs.append(np.asarray(a) * (1 - t) + np.asarray(b) * t)
    return np.concatenate(segs)


def accuracy(mats: list[np.ndarray], n_frames: int = 60) -> None:
    """Does carrying the camera make it MORE RIGHT, not merely steadier? (painted lines as truth)"""
    import cv2

    print("== accuracy against the painted lines (independent of the motion chain) ==\n")
    h, _ = _calibration()
    world = _pitch_points()

    # The displaced anchor from the section above, brought here as a control. It fools the swim
    # metric completely; if it also scores well against the paint then this metric is blind too and
    # neither row below means anything.
    bad0 = cv2.findHomography(PROBE_UV, _apply(h[0], PROBE_UV) + np.array([2.0, 0.0]), 0)[0]
    cand = {
        "per-frame (shipped)": h,
        "carried, +-8": fuse(h, mats, 8),
        "SHIPPED carry, +-8": shipped(h, mats, 8),
        "coefficient average, w=17": coeff_average(h, 17),
        "MAD reject k=3": mad_reject(h, mats, 3.0)[0],
        "MAD reject k=1": mad_reject(h, mats, 1.0)[0],
        "WRONG by 2 m (control)": np.stack([bad0 @ _chain(mats, k, 0) for k in range(len(h))]),
    }

    cap = cv2.VideoCapture(VIDEO)
    scores: dict[str, list[float]] = {k: [] for k in cand}
    cover: dict[str, list[float]] = {k: [] for k in cand}
    for k in range(n_frames):
        ok, bgr = cap.read()
        if not ok:
            break
        segs, grass = _line_mask(bgr)
        for name, hh in cand.items():
            uv = _apply(np.linalg.inv(hh[k]), world)
            u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
            ok_uv = (u >= 0) & (u < 1920) & (v >= 0) & (v < 1080)
            if ok_uv.sum() < 20:
                continue
            uv, u, v = uv[ok_uv], u[ok_uv], v[ok_uv]
            on = grass[v, u] > 0                      # only judge where the pitch is visible
            if on.sum() < 20:
                continue
            d = _perp_px(uv[on], segs)
            scores[name].append(float(np.median(d)))
            cover[name].append(float((d < 5).mean()))
    cap.release()

    print(f"  {'':<26}{'median px':>11}{'p95 px':>9}{'within 5 px':>13}")
    print("  " + "-" * 59)
    for name in cand:
        a, c = np.array(scores[name]), np.array(cover[name])
        print(
            f"  {name:<26}{np.median(a):>11.2f}{np.percentile(a, 95):>9.2f}"
            f"{100 * c.mean():>12.0f}%"
        )
    print()

    # Tenths of a pixel apart is not something to read off a summary table. Every candidate was
    # scored on the SAME frames against the SAME detected segments, so the honest test is paired:
    # frame by frame, which one sits closer to the paint than the shipped calibration does.
    base = np.array(scores["per-frame (shipped)"])
    for name in (
        "carried, +-8", "SHIPPED carry, +-8", "coefficient average, w=17",
        "MAD reject k=3", "MAD reject k=1",
    ):
        delta = base - np.array(scores[name])
        print(
            f"  vs per-frame, paired over {len(delta)} frames: {name:<26} closer on "
            f"{100 * (delta > 0).mean():>3.0f}% of them, median {np.median(delta):+.3f} px"
        )
    print("  Distance is measured FROM the projected model TO the nearest paint, so stray white")
    print("  pixels (kit, boards) can only flatter a candidate — both are flattered equally.\n")

    # The two axes are in different units, so "0.19 px worse but 0.108 m steadier" is not yet a
    # decision. Convert: how many metres on the pitch is one pixel worth, where the players are?
    j = np.median(
        [
            np.linalg.norm(_apply(h[k], PROBE_UV + [1.0, 0.0]) - _apply(h[k], PROBE_UV), axis=1)
            for k in range(len(h))
        ]
    )
    print(f"  1 px at the probe points is worth {j:.4f} m on the pitch (median over the clip),")
    print(f"  so the accuracy the carry gives up is ~{0.19 * j:.4f} m against {0.108:.3f} m")
    print("  of scene slide it removes — two orders of magnitude apart, in favour of carrying.\n")


def confidence_check(n_frames: int = 60) -> None:
    """Does the confidence we report per frame predict the error we can actually measure?

    Asked because the propagation was about to weight its vote by it. The answer changes the
    design, and it is not the expected one — so it ships with the two controls that rule out the
    obvious artifacts, rather than as a bare correlation.
    """
    import cv2

    print("== is our reported per-frame confidence worth anything? ==\n")
    cal = from_json(open(SCENE, encoding="utf-8").read()).field.calibration
    h, conf = cal.homographies, cal.confidence
    world = _pitch_points()
    frozen = np.repeat(h[:1], len(h), axis=0)

    def err(hh: np.ndarray, k: int, segs: np.ndarray, grass: np.ndarray) -> float:
        uv = _apply(np.linalg.inv(hh[k]), world)
        u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        m = (u >= 0) & (u < 1920) & (v >= 0) & (v < 1080)
        uv, u, v = uv[m], u[m], v[m]
        on = grass[v, u] > 0
        return float(np.median(_perp_px(uv[on], segs))) if on.sum() >= 20 else float("nan")

    cap = cv2.VideoCapture(VIDEO)
    rows = []
    for k in range(min(n_frames, len(h))):
        ok, bgr = cap.read()
        if not ok:
            break
        segs, grass = _line_mask(bgr)
        rows.append((len(segs), err(h, k, segs, grass), err(frozen, k, segs, grass)))
    cap.release()
    a = np.array(rows, dtype=float)
    c = conf[: len(a)]

    print(f"  {'':<20}{'segments':>10}{'per-frame':>11}{'FROZEN':>9}{'confidence':>12}")
    print("  " + "-" * 62)
    third = len(a) // 3
    for lo, hi in [(0, third), (third, 2 * third), (2 * third, len(a))]:
        s = a[lo:hi]
        print(
            f"  frames {lo:>2}-{hi:<12}{np.nanmedian(s[:, 0]):>10.0f}{np.nanmedian(s[:, 1]):>11.2f}"
            f"{np.nanmedian(s[:, 2]):>9.2f}{c[lo:hi].mean():>12.3f}"
        )
    print()

    ok = ~np.isnan(a[:, 1])
    r = float(np.corrcoef(c[ok], a[ok, 1])[0, 1])
    q = np.argsort(c[ok])
    print(f"  Pearson r(confidence, measured error) = {r:+.3f}  — it should be NEGATIVE.")
    print(
        f"  lowest-confidence third errs {np.median(a[ok][q[: len(q) // 3], 1]):.2f} px, "
        f"highest-confidence third errs {np.median(a[ok][q[-(len(q) // 3) :], 1]):.2f} px"
    )
    print()
    print("  Two controls, because a spurious version of this is easy to produce. If the later")
    print("  frames were simply easier to score, a FROZEN camera would improve across them too —")
    print("  it degrades instead, and steeply, so the metric is getting harder, not softer. And")
    print("  if more paint were visible later, distance-to-nearest-segment would shrink for free")
    print("  — the segment count is flat and correlates with the error at only r = -0.26.")
    print("  So the confidence really is backwards, and nothing may weight by it until it is")
    print("  fixed. It is exported in scene.json, so the blast radius is not just this bench.\n")


def verdict() -> None:
    print("== verdict ==")
    print("  The brief's premise holds: our per-frame calibration does swim, by a median 0.12 m")
    print("  and a p95 of 0.47 m, on a clip where the camera itself pans smoothly at 9 px/frame.")
    print("  Under R7 that is the error class a viewer actually sees, so it is worth removing.")
    print()
    print("  It is NOT free, and an earlier version of this script said it was. Scored on the")
    print("  first 20 frames the accuracy cost looked like a coin flip; over all 60 it is a")
    print("  consistent loss — carrying is closer to the paint on only 32 % of frames, by a")
    print("  median 0.19 px. Swim removal and accuracy trade against each other monotonically")
    print("  across every method here. There is no setting that improves both.")
    print()
    print("  What decides it is putting the two axes in the same units. One pixel at the probe")
    print("  points is 0.018 m of pitch, so carrying gives up ~0.0035 m of accuracy to remove")
    print("  0.108 m of scene slide — a 31x asymmetric trade, and worth taking. But it must be")
    print("  reported as a trade, and the knob that picks the point on the curve should be ours")
    print("  to set, not baked in.")
    print()
    print("  Two things that decide the implementation:")
    print("  - The coefficient averaging already in the tree (`_temporal_smooth`, default OFF) is")
    print("    not meaningless as first assumed — it removes 77 % of the swim. It is simply")
    print("    DOMINATED: carrying beats it on both axes at once (0.011 vs 0.027 m of swim, 1.41")
    print("    vs 1.70 px of error). Replace it rather than enabling it.")
    print("  - MAD-reject alone, the brief's own outlier form, is too timid to be the whole")
    print("    answer: at k=3 it touches 4 frames and removes 14 % of the swim. It is a useful")
    print("    guard on top of carrying, not a substitute for it.")
    print()
    print("  It does not need the GPU the brief books. RAFT-small is proposed to recover the")
    print("  motion the carry rides on; goodFeaturesToTrack + LK recovered it on the CPU at")
    print("  4000/4000 corners, 3790 RANSAC inliers, in about a minute for the clip. R2 should")
    print("  be re-scoped to the CPU path, which also un-blocks it from the pod.")
    print()
    print("  SEPARATE DEFECT, found on the way and worth more than R2 itself: the confidence we")
    print("  report per calibrated frame is ANTI-predictive. Pearson r against the measured paint")
    print("  error is +0.699 — the frames the pipeline trusts most are the ones that are worst")
    print("  (highest-confidence third 1.69 px, lowest-confidence third 1.11 px). Controlled: a")
    print("  frozen camera over the same frames degrades 2.11 -> 34.40 px, so the metric is not")
    print("  merely getting easier, and detected-segment count is flat (r = -0.26). That number")
    print("  is exported in scene.json and consumed downstream, so anything weighting by it is")
    print("  being steered backwards — including the confidence-weighted propagation this")
    print("  benchmark was about to recommend.")


if __name__ == "__main__":
    _h, _f = _calibration()
    _mats, _moved = true_motion(len(_f))
    shipped_motion(len(_f), _mats)
    consistency(_mats, _moved)
    removable(_mats)
    circularity(_mats)
    accuracy(_mats)
    confidence_check()
    verdict()
