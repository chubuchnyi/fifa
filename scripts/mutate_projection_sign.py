#!/usr/bin/env python3
"""Prove the R6 golden tests actually catch sign errors, by injecting them.

``tests/unit/test_golden_projection_sign.py`` guards a bug class that produces a *mirrored but
perfectly self-consistent* scene: nothing crashes, every mutual-inverse test still passes, and
the only detector is a human eye on a finished render. A test suite for that class is worth
exactly what it catches — so this script breaks the production code five ways and reports which
test notices. A golden test that survives its own mutation is decorative, and this prints that
verdict in as many words.

Run::

    PYTHONPATH=src .venv/bin/python scripts/mutate_projection_sign.py

Mutations 4 and 5 are not hypothetical. Both are states this repo has actually shipped: the roll
gate has been inverted, and the roll correction was once an X-only mirror that was validated by
eye on 2026-07-07 and falsified by an objective harness a day later (it left every body
vertically inverted — invisible at ~22 px tall). Those two are regression tests for our own
history, not imagined faults.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                    # `tests` and `poseannot` live at the repo root
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402
import tests.unit.test_golden_projection_sign as G  # noqa: E402

from pitch3d.core.scene import field, projection  # noqa: E402

POSEANNOT_CAMERA = "poseannot/camera.py"
TESTS = {n: getattr(G, n) for n in dir(G) if n.startswith("test_")}


def run() -> dict[str, bool]:
    """Run every golden test; True means it passed."""
    passed = {}
    for name, fn in TESTS.items():
        try:
            fn()
            passed[name] = True
        except BaseException:  # a mutation may raise anything; all of it counts as "caught"
            passed[name] = False
    return passed


def report(name: str, passed: dict[str, bool]) -> bool:
    caught = sorted(k.removeprefix("test_") for k, ok in passed.items() if not ok)
    print(name)
    if not caught:
        print("    NOTHING CAUGHT IT — the golden tests are decorative for this defect.")
        return False
    for c in caught:
        print(f"    caught by {c}")
    print()
    return True


def _patched_frame_projector(old: str, new: str):
    """Recompile ``poseannot.camera`` with one line swapped, without touching the file."""
    import poseannot.camera as pcam

    with open(POSEANNOT_CAMERA, encoding="utf-8") as fh:
        src = fh.read()
    if old not in src:
        raise SystemExit(f"{POSEANNOT_CAMERA} no longer contains {old!r} — update this script")
    namespace = dict(pcam.__dict__)
    exec(compile(src.replace(old, new), POSEANNOT_CAMERA, "exec"), namespace)  # noqa: S102
    return namespace["frame_projector"]


def homography_direction_reversed() -> dict[str, bool]:
    """Store world→image where image→world belongs — invisible to any round-trip test."""
    original = field.FieldCalibration.image_to_world

    def mutated(self, frame_index, uv):
        inverted = field.FieldCalibration(
            np.linalg.inv(self.homographies[0])[None], self.frames, self.confidence
        )
        return original(inverted, frame_index, uv)

    field.FieldCalibration.image_to_world = mutated
    try:
        return run()
    finally:
        field.FieldCalibration.image_to_world = original


def image_v_axis_negated() -> dict[str, bool]:
    """The classic y-up/y-down slip: mirror pixels about the principal point's row."""
    original = projection.project_world_points_with_depth

    def mutated(camera, frame_index, world_xyz):
        uv, depth, visible = original(camera, frame_index, world_xyz)
        uv = uv.copy()
        uv[:, 1] = 2 * camera.intrinsics.cy - uv[:, 1]
        return uv, depth, visible

    projection.project_world_points_with_depth = mutated
    G.project_world_points = lambda c, f, w: mutated(c, f, w)[::2]
    try:
        return run()
    finally:
        projection.project_world_points_with_depth = original
        G.project_world_points = projection.project_world_points


def quaternion_order_swapped() -> dict[str, bool]:
    """Read our (w, x, y, z) as scipy's (x, y, z, w): a wrong camera that still projects."""
    original = projection.quat_to_rotation_matrix
    mutated = lambda q: original(np.roll(np.asarray(q, float).reshape(4), -1))  # noqa: E731
    projection.quat_to_rotation_matrix = G.quat_to_rotation_matrix = mutated
    try:
        return run()
    finally:
        projection.quat_to_rotation_matrix = G.quat_to_rotation_matrix = original


def roll_gate_inverted() -> dict[str, bool]:
    """Flip the ``-R[1,2] < 0`` comparison: un-flips good cameras, flips bad ones."""
    import poseannot.camera as pcam

    original = pcam.frame_projector
    sys.modules["poseannot.camera"].frame_projector = _patched_frame_projector(
        "flipped = (not aligned) and bool(-R[1, 2] < 0)",
        "flipped = (not aligned) and bool(-R[1, 2] > 0)",
    )
    try:
        return run()
    finally:
        sys.modules["poseannot.camera"].frame_projector = original


def roll_fix_regressed_to_x_only_mirror() -> dict[str, bool]:
    """The real 2026-07-07 wrong fix: diag(-1,1,1) leaves every body vertically inverted."""
    import poseannot.camera as pcam

    original = pcam.frame_projector
    sys.modules["poseannot.camera"].frame_projector = _patched_frame_projector(
        "D = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)",
        "D = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)",
    )
    try:
        return run()
    finally:
        sys.modules["poseannot.camera"].frame_projector = original


MUTATIONS = [
    ("MUT-1  homography stored world->image, not image->world", homography_direction_reversed),
    ("MUT-2  image v axis negated in the pinhole projector", image_v_axis_negated),
    ("MUT-3  quaternion read as (x,y,z,w) instead of (w,x,y,z)", quaternion_order_swapped),
    ("MUT-4  180-roll gate comparison inverted [shipped once]", roll_gate_inverted),
    ("MUT-5  roll fix regressed to the X-only mirror [shipped once]", roll_fix_regressed_to_x_only_mirror),  # noqa: E501
]


def main() -> int:
    baseline = run()
    unexpected = sorted(k for k, ok in baseline.items() if not ok)
    if unexpected:
        print(f"baseline is not green — fix these first: {unexpected}")
        return 1
    print(f"baseline: all {len(baseline)} golden tests pass\n")

    survivors = [name for name, mutate in MUTATIONS if not report(name, mutate())]
    if survivors:
        print(f"\n{len(survivors)} mutation(s) went undetected:")
        for s in survivors:
            print(f"  {s}")
        return 1
    print(f"all {len(MUTATIONS)} mutations were caught")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
