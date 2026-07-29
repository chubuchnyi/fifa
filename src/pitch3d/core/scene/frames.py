"""SMPL-X → world frame conventions.

The world frame is ``x`` right, ``y`` forward, ``z`` up, with the pitch plane at ``z = 0``.

SMPL-X output reaches us in **two different source frames**, and they need *different* remaps.
Collapsing them into a single constant silently turns half the pipeline upside down, so both
live here side by side with the test that pins them:

``CANONICAL``
    A plain SMPL-X forward pass with ``global_orient = 0``: native axes, ``+y`` up, ``+z`` front.
    Produced by rest-pose evaluation and by fake/degenerate exports.

``CAMERA``
    Real SMPLest-X output, whose ``global_orient`` is a *camera-frame* rotation. Image axes put
    ``+y`` DOWN, so a standing body has its head at ``-y`` — the opposite of canonical.

``CAMERA`` treats the broadcast camera as level; residual tilt is left to the camera solve.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

SourceFrame = Literal["canonical", "camera"]

#: SMPL-X canonical (``+y`` up, ``+z`` front) → world. ``world = (x, -z, y)``, ``det = +1``.
R_SMPLX_CANONICAL_TO_WORLD: np.ndarray = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
)

#: SMPL-X camera frame (``+y`` down) → world. ``world = (x, z, -y)``, ``det = +1``.
R_SMPLX_CAMERA_TO_WORLD: np.ndarray = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
)

_ROTATIONS: dict[str, np.ndarray] = {
    "canonical": R_SMPLX_CANONICAL_TO_WORLD,
    "camera": R_SMPLX_CAMERA_TO_WORLD,
}


def rotation_for(frame: SourceFrame) -> np.ndarray:
    """Return the SMPL-X → world rotation for ``frame``."""
    try:
        return _ROTATIONS[frame]
    except KeyError as err:
        raise ValueError(
            f"unknown source frame {frame!r}; expected one of {sorted(_ROTATIONS)}"
        ) from err


def detect_source_frame(verts: np.ndarray, pelvis: np.ndarray) -> SourceFrame:
    """Infer the source frame from where the body's far end sits along native ``y``.

    The pelvis splits a body ~0.6 m to the head and ~0.95 m to the toes, so the extreme that is
    *farther* from it is always the feet. Feet at ``+y`` means the vertical axis points down.
    """
    y = np.asarray(verts, dtype=float)[:, 1] - float(np.asarray(pelvis, dtype=float)[1])
    return "camera" if abs(y.max()) > abs(y.min()) else "canonical"


def smplx_to_world(
    verts: np.ndarray,
    *,
    pelvis: np.ndarray | None = None,
    frame: SourceFrame | None = None,
    transl: np.ndarray | None = None,
) -> np.ndarray:
    """Map SMPL-X vertices/joints into the world frame.

    ``frame`` overrides the auto-detection. ``pelvis`` re-origins the body before rotating, which
    is required whenever ``transl`` is the world position of the *pelvis* rather than of the model
    origin — SMPL-X puts the pelvis ~0.35 m off its own origin, a bias that otherwise rides along.
    """
    v = np.asarray(verts, dtype=float)
    if pelvis is not None:
        p = np.asarray(pelvis, dtype=float).reshape(3)
        frame = frame or detect_source_frame(v, p)
        v = v - p
    elif frame is None:
        raise ValueError("pass `pelvis` for auto-detection, or name the `frame` explicitly")
    out = v @ rotation_for(frame).T
    if transl is not None:
        out = out + np.asarray(transl, dtype=float).reshape(3)
    return out
