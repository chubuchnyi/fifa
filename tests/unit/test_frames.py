"""Golden tests for the SMPL-X → world frame remaps.

These pin the failure class the research brief calls "silent, plausible, wrong": an orientation
constant that type-checks, renders something body-shaped, and is upside down or mirrored.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.scene.frames import (
    R_SMPLX_CAMERA_TO_WORLD,
    R_SMPLX_CANONICAL_TO_WORLD,
    detect_source_frame,
    rotation_for,
    smplx_to_world,
)

PELVIS = np.array([0.0, -0.35, 0.0])


def _body(*, y_down: bool) -> np.ndarray:
    """Minimal body: head, pelvis, toe — laid out along native ``y``, plus a front marker."""
    s = 1.0 if y_down else -1.0
    return PELVIS + np.array(
        [
            [0.0, -0.60 * s, 0.0],   # head
            [0.0, 0.00, 0.0],        # pelvis
            [0.0, 0.95 * s, 0.0],    # toe
            [0.0, 0.00, 0.20],       # front marker (+z is front in both frames)
        ]
    )


@pytest.mark.parametrize("frame", ["canonical", "camera"])
def test_remap_is_a_proper_rotation(frame: str) -> None:
    r = rotation_for(frame)
    assert np.isclose(np.linalg.det(r), 1.0), "det must be +1 — det -1 mirrors the body"
    assert np.allclose(r @ r.T, np.eye(3))


@pytest.mark.parametrize("frame, y_down", [("canonical", False), ("camera", True)])
def test_head_ends_up_above_the_feet(frame: str, y_down: bool) -> None:
    w = smplx_to_world(_body(y_down=y_down), pelvis=PELVIS, frame=frame)
    head, _pelvis, toe, _front = w
    assert head[2] > toe[2], f"{frame}: body is upside down"
    assert np.isclose(head[2], 0.60) and np.isclose(toe[2], -0.95)


def test_the_two_remaps_are_not_interchangeable() -> None:
    """Guard against 'unifying' them: each is upside down when fed the other's data."""
    assert not np.allclose(R_SMPLX_CANONICAL_TO_WORLD, R_SMPLX_CAMERA_TO_WORLD)
    for frame, wrong, y_down in [("canonical", "camera", False), ("camera", "canonical", True)]:
        w = smplx_to_world(_body(y_down=y_down), pelvis=PELVIS, frame=wrong)
        assert w[0][2] < w[2][2], f"{wrong} on {frame} data should invert the body"


@pytest.mark.parametrize("y_down, expected", [(False, "canonical"), (True, "camera")])
def test_source_frame_is_detected_from_the_far_end(y_down: bool, expected: str) -> None:
    assert detect_source_frame(_body(y_down=y_down), PELVIS) == expected


def test_pelvis_reorigin_removes_the_model_origin_bias() -> None:
    """SMPL-X puts the pelvis ~0.35 m off its own origin; `transl` is the world pelvis."""
    transl = np.array([12.0, -3.0, 0.90])
    w = smplx_to_world(_body(y_down=True), pelvis=PELVIS, transl=transl)
    assert np.allclose(w[1], transl), "pelvis must land exactly on transl"
    assert np.isclose(w[2][2], transl[2] - 0.95), "toe sits a leg-length below the pelvis"


def test_front_marker_keeps_a_consistent_handedness() -> None:
    """Native +z is the body's front; a proper rotation may not mirror it away."""
    for frame in ("canonical", "camera"):
        w = smplx_to_world(_body(y_down=frame == "camera"), pelvis=PELVIS, frame=frame)
        assert not np.isclose(w[3][1], 0.0), f"{frame}: front marker collapsed onto the up axis"


def test_unknown_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown source frame"):
        rotation_for("world")  # type: ignore[arg-type]
