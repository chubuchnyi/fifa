"""Reading a hand-registered pitch plane back out of a scene (#112 read path, #128).

``PlaneTransformPayload`` is the one correction no residual can compute for us: it is where the
operator's eye says the pitch model sits, and a residual scored against the same lines that
placed the model cannot tell you it is wrong. The write side has existed since #112 — the
annotator drags, and stores a ``FIELD_CALIBRATION`` correction.

This is the read side, and it lives in ``core`` rather than in ``poseannot`` because the export
needs it too. That was #128: the payload was declared in :mod:`pitch3d.core.scene.layers`,
applied in ``poseannot/camera.py``, and consumed by nothing under ``src/pitch3d/`` — so an
operator could spend a session registering the pitch and the exported scene would still carry the
raw solve. The functions moved here unchanged; ``poseannot.camera`` re-exports them, so the
annotator and the exporter cannot drift into two answers.

**What the adjustment moves is the camera, not the bodies.** Subjects are stored in world metres,
and the drag re-registers where the pitch model sits under a camera that is only approximately
right. Both halves of the camera move together (``camera`` and ``field.calibration``) because
#107 exists precisely because those two were once allowed to disagree.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pitch3d.core.correction.rotations import matrix_to_quat
from pitch3d.core.scene.layers import TargetKind
from pitch3d.core.scene.projection import quat_to_rotation_matrix


def plane_adjustment(corrections, frame: int) -> np.ndarray:
    """The composed ``B`` for ``frame`` from every enabled FIELD_CALIBRATION correction.

    Applied in insertion order on the right, because each drag was measured against the layout
    as it stood *after* the previous ones — the same order the user made them in.
    """
    b = np.eye(3)
    for c in corrections:
        if c.target.kind is not TargetKind.FIELD_CALIBRATION or not c.enabled:
            continue
        if frame not in c.frame_range:
            continue
        b = b @ np.asarray(c.payload.matrix, dtype=float)
    return b


def adjusted_camera(camera, corrections):
    """``camera`` re-expressed after the user re-registered the pitch plane (#112).

    A scene holds two descriptions of one camera, and #107 exists because they were allowed to
    drift apart. Moving the pitch under a fixed camera would split them again — measured live at
    2500 px on the first drag — so the camera moves too, and it moves *exactly*: for the plane
    ``Z = 0`` the world→image map is ``K[r₁ r₂ t]``, i.e. two rotation columns and the
    translation, which is precisely what a plane transform acts on. ``K`` never enters, so this
    is the same right-multiply as :func:`adjusted_calibration` and cannot disagree with it.

    The SVD snap only removes float noise here: for a similarity the columns come out orthogonal
    already, scaled by ``σ``, which is what dividing by ``‖m₀‖`` takes out.
    """
    if camera is None:
        return None
    frames = np.asarray(camera.frames, dtype=int)
    per_frame = [plane_adjustment(corrections, int(f)) for f in frames]
    if all(np.array_equal(b, np.eye(3)) for b in per_frame):
        return camera

    quat = np.asarray(camera.rotation_quat, dtype=float)
    transl = np.asarray(camera.translation, dtype=float)
    rots = np.zeros((len(per_frame), 3, 3))
    out_t = np.zeros_like(transl)
    for i, b in enumerate(per_frame):
        r = quat_to_rotation_matrix(quat[i])
        m = np.column_stack([r[:, 0], r[:, 1], transl[i]]) @ b
        scale = np.linalg.norm(m[:, 0])
        r1, r2, out_t[i] = m[:, 0] / scale, m[:, 1] / scale, m[:, 2] / scale
        u, _, vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
        rots[i] = u @ vt
        if np.linalg.det(rots[i]) < 0:
            rots[i] = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    return replace(camera, rotation_quat=matrix_to_quat(rots), translation=out_t)


def adjusted_calibration(calibration, corrections):
    """``calibration`` with the user's layout drags folded in — or itself, if there are none.

    Returns a copy: the stored solve stays untouched, so disabling the corrections restores it
    exactly. ``H_i2w`` is the inverse direction, hence ``B⁻¹`` on the left.
    """
    frames = np.asarray(calibration.frames, dtype=int)
    per_frame = [plane_adjustment(corrections, int(f)) for f in frames]
    if all(np.array_equal(b, np.eye(3)) for b in per_frame):
        return calibration
    h = np.asarray(calibration.homographies, dtype=float)
    moved = np.stack([np.linalg.inv(b) @ h[i] for i, b in enumerate(per_frame)])
    return replace(calibration, homographies=moved)


def has_plane_corrections(corrections) -> bool:
    """Is there any enabled FIELD_CALIBRATION correction to apply at all?

    Used to decide whether an export is carrying hand registration, so it can say so rather than
    apply it silently — the #125 lesson was that a run which quietly did the wrong thing reads
    exactly like one that did the right thing.
    """
    return any(
        c.target.kind is TargetKind.FIELD_CALIBRATION and c.enabled for c in corrections
    )


def apply_plane_corrections(scene):
    """``scene`` with the operator's pitch registration folded into both halves of its camera.

    Returns the scene unchanged when there is nothing to apply, so callers can use it
    unconditionally. Subject and ball motion are untouched by design — see the module docstring.
    """
    corrections = list(getattr(scene, "corrections", ()) or ())
    if not has_plane_corrections(corrections):
        return scene

    changed: dict = {}
    cam = getattr(scene, "camera", None)
    if cam is not None:
        changed["camera"] = adjusted_camera(cam, corrections)
    field = getattr(scene, "field", None)
    cal = getattr(field, "calibration", None) if field is not None else None
    if cal is not None:
        changed["field"] = replace(field, calibration=adjusted_calibration(cal, corrections))
    return replace(scene, **changed) if changed else scene
