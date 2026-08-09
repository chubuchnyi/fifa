"""#117 — what would frame preprocessing actually buy the calibration?

The ask was to preprocess frames so auto-calibration has more to bite on: painted markings,
mowing stripes, other fixed field objects. Before building any of it, this measures whether
there is anything to win, because the obvious answer is already suspect — #114 measured the
solved overlay at 1.4 px against real paint, so the ground plane is not where the error is.

Four questions, answered on the target clip's own 60 solved homographies and its pixels:

A  **How much of the per-frame wobble is noise?**  A broadcast camera pans smoothly, so anything
   a projected world point does that a smooth curve cannot follow is estimation noise. That
   number, not the 1.4 px, is the ceiling on what more evidence per frame can remove.

B  **Is the solve even a camera?**  Where it puts the camera centre, and whether that camera is
   standing on the right side of the pitch — the mirror in §B is the reason nothing 3D has ever
   been drawable from this calibration.

C  **What focal does the clip have?**  Three instruments disagree by 45%, and this pins down which
   assumption each one rests on, with synthetic controls that prove the disagreement is real.

D  **Are mowing stripes really there, and where?**  Measured on the frames: how much of the
   playing surface lies far from any painted line, since evidence that only exists where paint
   already is buys nothing.

Run: ``PYTHONPATH=src .venv/bin/python scripts/bench_frame_preprocessing.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCENE = ROOT / "out/carry_off/export/scene.json"
VIDEO = ROOT / "samples/video/Colombia-1-0-Congo-DR1080p.mp4"
WIDTH, HEIGHT = 1920, 1080

#: The exported calibration is mirrored in world Y (see ``main``, section B). Everything that reads
#: a *camera* out of a homography must undo it first; everything that only draws on the lawn need
#: not, which is exactly why it survived this long.
Y_MIRROR = np.diag([1.0, -1.0, 1.0])


def load_homographies(path: Path) -> np.ndarray:
    """The exported world→image maps, ``(F, 3, 3)``."""
    blob = json.loads(path.read_text())
    cal = blob["fields"]["field"]["fields"]["calibration"]["fields"]
    return np.asarray(cal["homographies"]["__ndarray__"]["data"], dtype=float)


def world_to_image(h_img2world: np.ndarray) -> np.ndarray:
    """The export stores image→world; everything here works the other way round."""
    return np.linalg.inv(h_img2world)


def project(w2i: np.ndarray, xy: np.ndarray) -> np.ndarray:
    p = np.column_stack([xy, np.ones(len(xy))]) @ w2i.T
    return p[:, :2] / p[:, 2, None]


def probe_points(w2i: np.ndarray) -> np.ndarray:
    """Pitch points that stay inside the frame for the whole clip.

    Measuring the wobble anywhere else would measure extrapolation instead: a point 40 m outside
    the observed region swings hundreds of px for a homography that is locally perfect, which
    says nothing about the solve. On this clip 34 of a 15x11 grid survive, spanning the left
    half — x -52.5..-15 m, y -34..13.6 m — which is simply where the camera is pointed.
    """
    gx, gy = np.meshgrid(np.linspace(-52.5, 52.5, 15), np.linspace(-34.0, 34.0, 11))
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    uv = np.stack([project(w, xy) for w in w2i])
    inside = (
        (uv[..., 0] > 0) & (uv[..., 0] < WIDTH) & (uv[..., 1] > 0) & (uv[..., 1] < HEIGHT)
    ).all(axis=0)
    return xy[inside]


#: Frames over which a cubic honestly describes a broadcast pan, at ~30 fps. Past this the
#: residual is dominated by the polynomial being too low-order, not by noise — see
#: :func:`smooth_residual_domain`.
SMOOTH_RESIDUAL_MAX_FRAMES = 90


def smooth_residual_domain(n_frames: int, order: int = 3) -> str:
    """The label this residual must be quoted with, or ``""`` when it is inside its domain.

    Exists because the number escaped its domain and cost a day. `fit_rigid_camera.py` printed a
    bare ``jitter`` from this function over a 236-frame span; read as noise it said 60.4 px, which
    is 120x the measured swim (#104: 0.011 m) and sent a whole investigation after temporal
    instability that does not exist. The same metric over 60 frames says 6.42 px. Nothing was
    wrong with either number — the second was quoted where the first applies.

    So the span now travels with the value. Modelled on `apply_rigid_camera.py`, which refuses a
    scene outside its fit's frame range and says which range that is.
    """
    if n_frames <= SMOOTH_RESIDUAL_MAX_FRAMES:
        return ""
    return (f"OUT OF DOMAIN: a degree-{order} fit over {n_frames} frames "
            f"(>{SMOOTH_RESIDUAL_MAX_FRAMES}) measures unmodelled camera motion, not noise")


def smooth_residual(tracks: np.ndarray, order: int = 3) -> np.ndarray:
    """Distance from each sample to a low-order polynomial through its own track.

    A broadcast pan is smooth over 2 s. Fitting a cubic in time and taking what is left is a
    deliberately *generous* estimate of noise — a real camera move is smoother than a cubic,
    so anything this reports is a lower bound on the jitter, not an inflated one.

    **Only over about 2 s.** Past ~90 frames the cubic cannot follow the pan and the residual is
    model error, not jitter: the same clip reads 6.42 px over 60 frames and 60.42 px over 236.
    Callers must label the value with :func:`smooth_residual_domain`.
    """
    n_frames = tracks.shape[0]
    t = np.linspace(-1.0, 1.0, n_frames)
    basis = np.vander(t, order + 1)
    fit = basis @ np.linalg.lstsq(basis, tracks.reshape(n_frames, -1), rcond=None)[0]
    return np.linalg.norm(tracks - fit.reshape(tracks.shape), axis=2)


def decompose(h: np.ndarray, focal: float) -> tuple[np.ndarray, np.ndarray]:
    """``(R, C)`` of the closest real pinhole to one ground homography, camera above the pitch.

    ``H`` and ``-H`` are the same projective map but *not* the same camera: negating the first two
    columns leaves ``r3 = r1 x r2`` alone, so the two branches differ by a 180° turn about world Z
    and put the centre at ``(x, y, +z)`` and ``(x, y, -z)``. Pick the one standing above the pitch.
    """
    k = np.array([[focal, 0, WIDTH / 2], [0, focal, HEIGHT / 2], [0, 0, 1.0]])
    m = np.linalg.inv(k) @ h
    m = m / ((np.linalg.norm(m[:, 0]) + np.linalg.norm(m[:, 1])) / 2.0)
    best: tuple[np.ndarray, np.ndarray] | None = None
    for sign in (1.0, -1.0):
        r1, r2, t = sign * m[:, 0], sign * m[:, 1], sign * m[:, 2]
        u, _s, vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
        rot = u @ np.diag([1.0, 1.0, float(np.linalg.det(u @ vt))]) @ vt
        centre = -rot.T @ t
        if best is None or centre[2] > best[1][2]:
            best = (rot, centre)
    assert best is not None
    return best


def rotation_cost(h: np.ndarray, focal: float) -> float:
    """How far ``K⁻¹ H K`` is from a rotation — zero iff its three singular values agree.

    A camera turning about its own centre maps image to image by ``H = K R K⁻¹`` whatever the
    scene is, so the plane — the thing #116 proved cannot see the focal — drops out entirely.
    Scoring the singular-value spread beats the textbook linear solve on the dual conic here: the
    linear form needs a large rotation to be conditioned at all and returns nonsense below ~30
    frames of baseline, while this is stable from 5 frames up (measured, section C).
    """
    k = np.array([[focal, 0, WIDTH / 2], [0, focal, HEIGHT / 2], [0, 0, 1.0]])
    s = np.linalg.svd(np.linalg.inv(k) @ h @ k, compute_uv=False)
    return float((s[0] - s[2]) / s[1])


def focal_from_pan(pairs: list[np.ndarray], grid: np.ndarray) -> float:
    """The focal that best explains a set of image→image homographies as pure rotation."""
    cost = np.array([[rotation_cost(h, f) for f in grid] for h in pairs]).mean(axis=0)
    return float(grid[int(np.argmin(cost))])


def pixel_homographies(frames: list[tuple[int, int]]) -> list[np.ndarray]:
    """Image→image homographies straight from the pixels — SIFT + MAGSAC, no calibration."""
    from poseannot.video import read_frame

    sift = cv2.SIFT_create(nfeatures=6000)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def feats(i: int) -> tuple[np.ndarray, np.ndarray]:
        if i not in cache:
            kp, desc = sift.detectAndCompute(read_frame(str(VIDEO), i), None)
            cache[i] = (np.float32([k.pt for k in kp]), desc)
        return cache[i]

    out = []
    for i, j in frames:
        (pi, di), (pj, dj) = feats(i), feats(j)
        good = [m for m, n in bf.knnMatch(di, dj, k=2) if m.distance < 0.75 * n.distance]
        a = np.float32([pi[m.queryIdx] for m in good])
        b = np.float32([pj[m.trainIdx] for m in good])
        h, _ = cv2.findHomography(a, b, cv2.USAC_MAGSAC, 2.0, maxIters=20000, confidence=0.9999)
        out.append(h)
    return out


def synthetic_pan(focal: float, total_deg: float, n: int, noise_px: float) -> list[np.ndarray]:
    """Image→image homographies of an exact pure pan, optionally refitted through noisy points."""
    rng = np.random.default_rng(0)
    k = np.array([[focal, 0, WIDTH / 2], [0, focal, HEIGHT / 2], [0, 0, 1.0]])
    rots = [
        cv2.Rodrigues(np.array([0.0, 1.0, 0.15]) / np.linalg.norm([0, 1, 0.15]) * np.deg2rad(d))[0]
        for d in np.linspace(0.0, total_deg, n)
    ]
    exact = [k @ r @ rots[0].T @ np.linalg.inv(k) for r in rots[1:]]
    if noise_px <= 0.0:
        return exact
    pts = np.column_stack([rng.uniform(0, WIDTH, 800), rng.uniform(0, HEIGHT, 800), np.ones(800)])
    out = []
    for h in exact:
        q = pts @ h.T
        uv = q[:, :2] / q[:, 2, None] + rng.normal(0.0, noise_px, (len(pts), 2))
        out.append(cv2.findHomography(pts[:, :2], uv, 0)[0])
    return out


def plane_pan(focal: float, drift_m: float) -> np.ndarray:
    """One image→image homography of a camera that pans *and* drifts, seeing one plane.

    A plane still induces an exact homography under translation — but ``K(R + t nᵀ/d)K⁻¹``, not
    ``K R K⁻¹``. So drift does not make :func:`rotation_cost` fail loudly; it makes it quietly
    return a different focal. Worth knowing which way, before blaming a reading on a moving rig.
    """
    def look_at(c: np.ndarray, target: np.ndarray) -> np.ndarray:
        z = target - c
        z /= np.linalg.norm(z)
        x = np.cross(z, [0.0, 0.0, 1.0])
        x /= np.linalg.norm(x)
        return np.array([x, np.cross(z, x), z])

    k = np.array([[focal, 0, WIDTH / 2], [0, focal, HEIGHT / 2], [0, 0, 1.0]])
    c_a = np.array([0.0, -73.2, 17.8])
    c_b = c_a + np.array([drift_m, 0.0, 0.0])
    r_a = look_at(c_a, np.array([-30.0, 0.0, 0.0]))
    r_b = look_at(c_b, np.array([-10.0, 0.0, 0.0]))
    normal = np.array([0.0, 0.0, 1.0])
    d = float(normal @ (np.zeros(3) - c_a))
    return k @ (r_b @ r_a.T + np.outer(r_b @ (c_a - c_b), r_a @ normal) / d) @ np.linalg.inv(k)


def stripe_evidence(frame_index: int) -> tuple[float, float, float]:
    """``(% of surface far from paint, turf tone range, fine texture)`` on one frame."""
    from poseannot.pitch_evidence import _masks
    from poseannot.video import read_frame

    bgr = read_frame(str(VIDEO), frame_index)
    dist, surf = _masks(bgr)
    surf = surf > 0
    far = surf & (dist > 30)
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    # A stripe is a broad, smooth luminance band: blur hard enough to erase players, lines and
    # sensor noise, and whatever modulation survives is turf tone.
    low = cv2.GaussianBlur(v, (0, 0), 21)
    detail = cv2.GaussianBlur(v, (0, 0), 3) - low
    return (
        far.sum() / surf.sum() * 100.0,
        float(np.percentile(low[far], 95) - np.percentile(low[far], 5)),
        float(np.percentile(np.abs(detail[far]), 90)),
    )


def main() -> None:
    h_i2w = load_homographies(SCENE)
    w2i = np.array([world_to_image(h) for h in h_i2w])
    n = len(w2i)
    print(f"clip: {n} solved frames, {WIDTH}x{HEIGHT}\n")

    probes = probe_points(w2i)
    tracks = np.stack([project(w, probes) for w in w2i])
    print(f"A. per-frame wobble ({len(probes)} pitch points that stay in shot all clip)")
    res = smooth_residual(tracks)
    step = np.linalg.norm(np.diff(tracks, axis=0), axis=2)
    print(f"   residual vs cubic-in-time: median {np.median(res):.2f} px, "
          f"p90 {np.percentile(res, 90):.2f} px, max {res.max():.1f} px")
    print(f"   real inter-frame motion:   median {np.median(step):.2f} px")
    print(f"   => jitter is {np.median(res) / np.median(step) * 100:.0f}% of the frame-to-frame "
          "move")

    print("\nB. is the solve a camera at all?")
    print("   the exported homographies are MIRRORED IN WORLD Y. As shipped, every frame")
    print("   decomposes to a camera whose optical axis points UP and whose centre is under the")
    print("   pitch, at every focal. The pitch is symmetric about Y=0, so the markings can never")
    print("   catch it and only something with height can — this is why goalposts never worked.")
    for tag, mirror in (("as exported", np.eye(3)), ("Y mirrored ", Y_MIRROR)):
        rot, centre = decompose(w2i[0] @ mirror, 3903.0)
        print(f"   {tag}: optical axis z {rot[2, 2]:+.3f} "
              f"({'looking down, correct' if rot[2, 2] < 0 else 'looking UP, impossible'}), "
              f"centre ({centre[0]:6.1f},{centre[1]:6.1f},{centre[2]:5.1f}) m")
    fixed = np.array([w @ Y_MIRROR for w in w2i])
    for f in (2700.0, 3903.0):
        cc = np.array([decompose(w, f)[1] for w in fixed])
        rng = cc.max(axis=0) - cc.min(axis=0)
        med = np.median(cc, axis=0)
        print(f"   f={f:6.0f} px: centre median ({med[0]:6.1f},{med[1]:6.1f},{med[2]:5.1f}) m, "
              f"spread over the clip ({rng[0]:.1f}, {rng[1]:.1f}, {rng[2]:.1f}) m")
    print("   a camera bolted to a gantry does not move 10 m in 2 s — that spread is the solve's")
    print("   own error, in units anyone can judge.")

    grid = np.arange(1200.0, 8001.0, 25.0)
    print("\nC. what focal does the clip have? three instruments, 45% apart")
    from poseannot.camera import focal_from_homography

    fs = np.array([focal_from_homography(w, WIDTH, HEIGHT) or np.nan for w in fixed])
    print(f"   1. per-frame plane decomposition: median {np.nanmedian(fs):.0f} px "
          f"(p10 {np.nanpercentile(fs, 10):.0f}, p90 {np.nanpercentile(fs, 90):.0f})")
    print("   2. one rigid pinhole fitted to all 60 homographies: 4277 px "
          "(scripts/bench_rigid_camera.py)")

    wide = [(0, 20), (0, 40), (0, 59), (10, 35), (10, 59), (20, 45), (25, 59), (5, 30)]
    print("   3. the pan itself, from pixels only (SIFT + MAGSAC, no calibration, no world model)")
    pix = pixel_homographies(wide)
    print(f"      {len(pix)} wide-baseline pairs -> f = {focal_from_pan(pix, grid):.0f} px")
    for gap in (5, 15, 30, 45):
        pairs = [w2i[i + gap] @ np.linalg.inv(w2i[i]) for i in range(n - gap)]
        print(f"      the solved homographies' own inter-frame motion, baseline {gap:2d}: "
              f"f = {focal_from_pan(pairs, grid):.0f} px")

    print("\n   the disagreement is not noise and not a moving rig — two controls:")
    for noise in (0.0, 0.5, 1.0):
        f = focal_from_pan(synthetic_pan(4277.0, 8.0, 13, noise), grid)
        print(f"      synthetic pure pan, true focal 4277, {noise:.1f} px homography noise "
              f"-> reads {f:.0f} px")
    for drift in (0.0, 0.5, 2.0):
        f = focal_from_pan([plane_pan(4277.0, drift)], grid)
        print(f"      synthetic pan + {drift:.1f} m of rig drift, true focal 4277 "
              f"-> reads {f:.0f} px")
    print("   the criterion is unbiased under noise, and drift pushes it UP. Nothing makes a")
    print("   4277 camera read 2700. The pan is blind to any fixed world-side transform (it")
    print("   cancels in H_j H_i⁻¹) and the decomposition is not, so the inconsistency is in")
    print("   how the homographies attach to the world — i.e. they are not one camera at all.")

    print("\nD. mowing stripes — is there evidence where the paint is not?")
    rows = [stripe_evidence(i) for i in (0, 20, 40, 59)]
    for (i, (far, tone, tex)) in zip((0, 20, 40, 59), rows, strict=True):
        print(f"   frame {i:2d}: {far:.0f}% of the playing surface is >30 px from any painted "
              f"line; turf tone spans {tone:.0f}/255 there, fine texture {tex:.1f}/255")
    print(f"   => {np.mean([r[0] for r in rows]):.0f}% of the surface carries no painted evidence")
    print("      at all, and it is not featureless: the stripe modulation is 6-8% of full scale,")
    print("      an order above the texture noise. But a stripe has no known world coordinate —")
    print("      it constrains a DIRECTION (its vanishing point), never a position.")


if __name__ == "__main__":
    main()
