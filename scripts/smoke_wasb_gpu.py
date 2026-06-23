"""Smoke-test the WASB ball backend on the GPU box — first real run of the adapter's heavy half.

The pure seam of ``pitch3d.adapters.models.wasb_backend`` (window tiling, per-frame assembly, the
env factory, dotted-path resolution) is unit-tested headlessly. The GPU inference half — ``_load``'s
Hydra build, ``_preprocess_window``'s warpAffine, the detector ``run_tensor``, the ``OnlineTracker``
linking — is ``# pragma: no cover``: it needs a WASB checkout + soccer weight + CUDA. This script is
that missing on-hardware exercise: it builds a real :class:`ClipRef` over the first ``FRAMES`` of a
clip, calls ``make().detect_ball(clip)``, and prints the per-frame ball track. A green run (prints
``WASB_SMOKE_OK``) is the validation the adapter docstring asks for.

Run on the pod (``PYTHONPATH=src`` so ``pitch3d`` imports; the backend reads ``PITCH3D_WASB_*`` /
``PITCH3D_DEVICE`` from the environment — the defaults already match the staged pod layout, so no
overrides are needed)::

    PYTHONPATH=src /workspace/.venv/bin/python scripts/smoke_wasb_gpu.py

Overridable via env:
  PITCH3D_CLIP   input video        (default /workspace/clip.mp4)
  FRAMES         how many frames    (default 9 = three non-overlapping WASB windows)
  STRIDE         gap between frames (default 1 = consecutive, how the pipeline feeds WASB)
  PITCH3D_WASB_REPO / PITCH3D_WASB_CKPT / PITCH3D_WASB_DATASET / PITCH3D_DEVICE — see make().
"""

from __future__ import annotations

import os
import time

import numpy as np

from pitch3d.adapters.models.wasb_backend import make
from pitch3d.core.ports.io import ClipRef

CLIP = os.environ.get("PITCH3D_CLIP", "/workspace/clip.mp4")
FRAMES = int(os.environ.get("FRAMES", "9"))
STRIDE = int(os.environ.get("STRIDE", "1"))


def main() -> None:
    frame_idx = np.arange(0, FRAMES * STRIDE, STRIDE, dtype=int)
    clip = ClipRef(
        source_id="wasb-smoke",
        uri=CLIP,
        frames=frame_idx,
        width=1920,
        height=1080,
        fps=59.94,
    )
    backend = make()
    print(f"backend: repo={backend.repo_dir} ckpt={backend.weights} device={backend.device}")
    print(f"clip: {CLIP}  frames={frame_idx.tolist()}")

    t0 = time.time()
    raw = backend.detect_ball(clip)
    dt = time.time() - t0

    n = len(frame_idx)
    visible = int((raw.scores > 0).sum())
    print(f"detect_ball: {n} frames in {dt:.1f}s  ->  visible {visible}/{n}")
    for i in range(len(raw.frames)):
        vis = bool(raw.scores[i] > 0)
        x, y = float(raw.points_xy[i, 0]), float(raw.points_xy[i, 1])
        print(f"  frame {int(raw.frames[i]):4d}: visible={vis!s:5} xy=({x:8.1f}, {y:8.1f})")

    assert raw.frames.shape == (FRAMES,), raw.frames.shape
    assert raw.points_xy.shape == (FRAMES, 2), raw.points_xy.shape
    assert raw.scores.shape == (FRAMES,), raw.scores.shape
    print("WASB_SMOKE_OK")


if __name__ == "__main__":
    main()
