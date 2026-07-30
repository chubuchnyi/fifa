#!/usr/bin/env python3
"""Compare two exports by SUBJECT PLACEMENT — the only channel calibration has to the picture.

Why this exists beside `bench_camera_swim.py`. That bench scores
`field.calibration.homographies`, which is the right thing to score if you want to know how good
the *solve* is. But #107 established that the render never sees that solve: `AppController`
replaces the solved `CameraTrack` with a synthetic tiled `standard_viewpoints(BROADCAST)` pose
before export, so the exported camera is bit-identical on every frame of every run. Calibration
reaches the image only by moving *where players stand*. So a calibration change that cannot be
seen here cannot be seen at all, no matter what the swim number says.

It was written to settle #94/R2 and immediately earned its keep: carrying removes 92 % of the
homography swim and left subject steadiness unchanged (0.0516 -> 0.0532 m median step, i.e. 3 %
*worse*), which the swim bench had no way to reveal.

Two numbers, and they answer different questions:

  displacement  ||A - B|| per subject-frame — how far the change MOVES players. Large means the
                two runs really differ; near zero means you rendered the same thing twice.
  step          ||x[t+1] - x[t]|| within one run — the frame-to-frame slide an eye reads as
                jitter. This is the one that says whether a change made the result steadier.
                Compare the two runs' steps: real motion is common to both, so a genuine
                stabilisation shows up as a lower step distribution.

Usage:
  PYTHONPATH=src python scripts/bench_subject_steadiness.py A.json B.json [--label-a X --label-b Y]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pitch3d.core.correction.engine import resolve_subject_motion  # noqa: E402
from pitch3d.core.scene.serialization import load_scene  # noqa: E402


def _tracks(path: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """track_id -> (frames, world root translation), corrections applied as the render sees them."""
    scene = load_scene(path)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for subj in scene.subjects:
        motion = resolve_subject_motion(subj.proposal, scene.corrections_for(subj.track_id))
        out[int(subj.track_id)] = (
            np.asarray(motion.pose.frames, dtype=int),
            np.asarray(motion.pose.transl, dtype=float),
        )
    return out


def _q(x: np.ndarray) -> str:
    return f"median {np.median(x):.4f}  p95 {np.percentile(x, 95):.4f}  max {x.max():.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_a")
    ap.add_argument("scene_b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()

    a, b = _tracks(args.scene_a), _tracks(args.scene_b)
    common = sorted(set(a) & set(b))
    if not common:
        print("no shared track ids — these exports are not comparable", file=sys.stderr)
        return 1

    disp, step_a, step_b = [], [], []
    for t in common:
        fa, xa = a[t]
        fb, xb = b[t]
        fs = np.intersect1d(fa, fb)
        if fs.size == 0:
            continue
        # Ground-plane only: vertical is the pose net's business, not the calibration's.
        pa = xa[np.searchsorted(fa, fs)][:, :2]
        pb = xb[np.searchsorted(fb, fs)][:, :2]
        disp.append(np.linalg.norm(pa - pb, axis=1))
        if fs.size > 2:
            step_a.append(np.linalg.norm(np.diff(pa, axis=0), axis=1))
            step_b.append(np.linalg.norm(np.diff(pb, axis=0), axis=1))

    d = np.concatenate(disp)
    sa, sb = np.concatenate(step_a), np.concatenate(step_b)
    la, lb = args.label_a, args.label_b

    print(f"tracks compared {len(common)}   subject-frame samples {d.size}\n")
    print(f"displacement {la} -> {lb} (m):  {_q(d)}")
    print("  how far the change moves players. ~0 means you rendered the same thing twice.\n")
    print(f"step within {la} (m):  {_q(sa)}")
    print(f"step within {lb} (m):  {_q(sb)}")
    med = 100.0 * (1.0 - np.median(sb) / np.median(sa))
    p95 = 100.0 * (1.0 - np.percentile(sb, 95) / np.percentile(sa, 95))
    print(f"  {lb} vs {la}: {med:+.1f}% at the median, {p95:+.1f}% at p95 (positive = steadier)")
    print("  Real motion is common to both runs, so only a genuine stabilisation moves this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
