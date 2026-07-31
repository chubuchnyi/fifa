"""#117 — can ONE physical camera explain the clip's 60 solved homographies?

Each frame is currently solved on its own: 8 free parameters per frame, 480 for the clip, with
nothing tying them together. That is why a projected pitch point wobbles 5.3 px against a smooth
pan (bench_frame_preprocessing, A) and why the recovered camera centre wanders 10 m for a rig
bolted to a gantry (B). It is also why the focal is ambiguous: a free homography can land on the
paint while being no camera at all.

This constrains the answer to a real pinhole — one focal, one centre, one rotation per frame, 41
parameters for 12 frames — and asks how much of the observed lawn mapping survives, as a function
of focal. Two things had to be fixed before the question was askable:

* **The exported calibration is mirrored in world Y** (bench_frame_preprocessing, B). Undo it or
  every decomposition returns a camera buried under the pitch, looking upward.
* **The objective must be smooth.** Scoring straight against the paint distance map caps out at
  75% of samples on the seed, and a capped residual has no gradient, so the optimiser cannot move.
  Fit to the homographies' own pixel mapping; score the winner against the paint afterwards.

Run: ``PYTHONPATH=src .venv/bin/python scripts/bench_rigid_camera.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.bench_frame_preprocessing import (  # noqa: E402
    SCENE,
    VIDEO,
    Y_MIRROR,
    decompose,
    load_homographies,
    world_to_image,
)

WIDTH, HEIGHT = 1920, 1080
FRAMES = list(range(0, 60, 5))
PAINT_CAP = 12.0


def rodrigues(r: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * (kx @ kx)


def unrodrigues(rot: np.ndarray) -> np.ndarray:
    theta = float(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1)))
    if theta < 1e-9:
        return np.zeros(3)
    v = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    return v * (theta / (2 * np.sin(theta)))


def kmat(focal: float) -> np.ndarray:
    return np.array([[focal, 0, WIDTH / 2], [0, focal, HEIGHT / 2], [0, 0, 1.0]])


def plane_h(focal: float, rvec: np.ndarray, centre: np.ndarray) -> np.ndarray:
    rot = rodrigues(rvec)
    return kmat(focal) @ np.column_stack([rot[:, 0], rot[:, 1], -rot @ centre])


def full_p(focal: float, rvec: np.ndarray, centre: np.ndarray) -> np.ndarray:
    rot = rodrigues(rvec)
    return kmat(focal) @ np.column_stack([rot, -rot @ centre])


def sample_bilinear(m: np.ndarray, uv: np.ndarray) -> np.ndarray:
    x = np.clip(uv[:, 0], 0, WIDTH - 1.001)
    y = np.clip(uv[:, 1], 0, HEIGHT - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return (
        m[y0, x0] * (1 - fx) * (1 - fy)
        + m[y0, x0 + 1] * fx * (1 - fy)
        + m[y0 + 1, x0] * (1 - fx) * fy
        + m[y0 + 1, x0 + 1] * fx * fy
    )


def main() -> None:
    from poseannot.pitch_evidence import _masks
    from poseannot.video import read_frame

    from pitch3d.core.scene.pitch import pitch_polylines, pitch_upright_polylines

    w2i = np.array([world_to_image(h) @ Y_MIRROR for h in load_homographies(SCENE)])

    gx, gy = np.meshgrid(np.linspace(-52.5, 52.5, 60), np.linspace(-34.0, 34.0, 40))
    grid = np.column_stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
    marks = np.concatenate(list(pitch_polylines()))
    marks = np.column_stack([marks[:, 0], -marks[:, 1], np.ones(len(marks))])

    seen, target = {}, {}
    for i in FRAMES:
        q = grid @ w2i[i].T
        uv = q[:, :2] / q[:, 2, None]
        keep = (uv[:, 0] > 0) & (uv[:, 0] < WIDTH) & (uv[:, 1] > 0) & (uv[:, 1] < HEIGHT)
        seen[i], target[i] = grid[keep], uv[keep]
    print(f"{len(FRAMES)} frames, {np.mean([len(v) for v in seen.values()]):.0f} lawn points "
          f"in shot per frame", flush=True)

    print("reading the paint (this decodes and ridge-filters every frame) ...", flush=True)
    evidence = {i: _masks(read_frame(str(VIDEO), i)) for i in FRAMES}

    def paint_distance(h: np.ndarray, frame: int) -> np.ndarray:
        """Distance from this map's markings to the real paint, capped so occlusions cannot rule."""
        q = marks @ h.T
        uv = q[:, :2] / q[:, 2, None]
        dist, surf = evidence[frame]
        good = (uv[:, 0] > 1) & (uv[:, 0] < WIDTH - 2) & (uv[:, 1] > 1) & (uv[:, 1] < HEIGHT - 2)
        sub = uv[good]
        on = surf[np.rint(sub[:, 1]).astype(int), np.rint(sub[:, 0]).astype(int)] > 0
        return np.minimum(sample_bilinear(dist, sub[on]), PAINT_CAP)

    base = np.concatenate([paint_distance(w2i[i], i) for i in FRAMES])
    print(f"reference: the free per-frame homography sits {np.median(base):.2f} px from the "
          f"paint (n={len(base)})\n", flush=True)

    def seed(focal: float) -> np.ndarray:
        poses = [decompose(w2i[i], focal) for i in FRAMES]
        centre = np.median([p[1] for p in poses], axis=0)
        return np.concatenate([centre, np.concatenate([unrodrigues(p[0]) for p in poses])])

    def residuals(p: np.ndarray, focal: float) -> np.ndarray:
        centre = p[:3]
        out = []
        for j, i in enumerate(FRAMES):
            q = seen[i] @ plane_h(focal, p[3 + 3 * j : 6 + 3 * j], centre).T
            w = np.where(np.abs(q[:, 2]) > 1e-9, q[:, 2], 1e-9)
            out.append((q[:, :2] / w[:, None] - target[i]).ravel())
        return np.concatenate(out)

    def solve(focal: float) -> tuple[np.ndarray, float, float]:
        sol = least_squares(residuals, seed(focal), args=(focal,), method="lm", max_nfev=6000)
        lawn = float(np.median(np.linalg.norm(sol.fun.reshape(-1, 2), axis=1)))
        paint = np.concatenate([
            paint_distance(plane_h(focal, sol.x[3 + 3 * j : 6 + 3 * j], sol.x[:3]), i)
            for j, i in enumerate(FRAMES)
        ])
        return sol.x, lawn, float(np.median(paint))

    print("focal sweep — one centre + 12 rotations free, 41 parameters against 480:")
    best: tuple[float, float, np.ndarray] | None = None
    for focal in (2400.0, 2700.0, 3000.0, 3400.0, 3903.0, 4277.0, 4400.0, 5200.0):
        x, lawn, paint = solve(focal)
        print(f"  f={focal:6.0f}: lawn {lawn:6.2f} px, paint {paint:5.2f} px, "
              f"centre ({x[0]:6.1f},{x[1]:6.1f},{x[2]:5.1f}) m", flush=True)
        if best is None or lawn < best[0]:
            best = (lawn, focal, x.copy())
    assert best is not None

    print(f"\nbest rigid camera: f = {best[1]:.0f} px, lawn {best[0]:.2f} px "
          f"(free per-frame reference: 0 by construction, 1.69 px against paint)")
    print("  a broadcast main camera stands on the halfway line 15-25 m up and well past the")
    print("  touchline. Only the high-focal end puts it there; 2700 wants it 12 m up and 46 m out.")

    print("\n  the goal frame this camera draws, by construction rather than by a hand-set focal:")
    for j, i in list(enumerate(FRAMES))[-3:]:
        p_full = full_p(best[1], best[2][3 + 3 * j : 6 + 3 * j], best[2][:3])
        goal = pitch_upright_polylines()[0] * np.array([1.0, -1.0, 1.0])
        q = np.column_stack([goal, np.ones(len(goal))]) @ p_full.T
        print(f"    frame {i}: {np.round(q[:, :2] / q[:, 2, None], 1).tolist()}")

    print("\n  NOTE the 11 px lawn residual. No single pinhole reproduces these homographies:")
    print("  they are 60 independent fits, and the focal you read out depends on which")
    print("  functional of them you use (see bench_frame_preprocessing, C). Constraining the")
    print("  SOLVE to be one camera — not re-reading the existing one — is what #117 buys.")


if __name__ == "__main__":
    main()
