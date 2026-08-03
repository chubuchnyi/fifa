"""User-edit persistence — appends Correction rows to edits.json.

Design:
- Every human edit is one Correction row (same class the pipeline uses),
  wrapped as ``{"corrections": [Correction, ...]}`` for forward-
  compatibility with additional layers of metadata.
- The whole file is (re)written atomically on each append so a mid-edit
  crash leaves the file valid.
- On backend start we load edits.json, feed the corrections into the
  scene, then subsequent scene_state.build_scene_state resolves them
  through the standard corrections engine — no special-case code.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from pitch3d.core.correction.engine import make_keyframes, make_offset

# The reader lives in core because the EXPORT needs this file too (#128) — until then the
# operator's edits stopped at the annotator. Re-exported under its old name, not reimplemented.
from pitch3d.core.correction.sidecar import load_corrections as load_edits
from pitch3d.core.scene.layers import (
    Correction,
    CorrectionMode,
    CorrectionTarget,
    FrameRange,
    PlaneTransformPayload,
    TargetKind,
)

#: string edit-kind (wire/API) → the scene TargetKind it addresses. The root
#: kinds edit the SMPL-X root (global_orient / transl) rather than a body joint.
_ROOT_KINDS: dict[str, TargetKind] = {
    "root_orientation": TargetKind.ROOT_ORIENTATION,
    "root_translation": TargetKind.ROOT_TRANSLATION,
}
from pitch3d.core.scene.serialization import to_json

_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _uid() -> str:
    """The unique part of a correction id.

    The timestamp alone is not one. Every edit here comes from a nudge gesture, which is repeated
    by nature, and the ids were `...-{ts}` at one-second resolution — so two drags in the same
    second minted the same id. Undo pops by position and kept working, which is why it went
    unnoticed until a live undo returned the same id twice.
    """
    return uuid4().hex[:8]


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_edits(path: Path, corrections: list[Correction]) -> None:
    corr_json = to_json(list(corrections), indent=2)
    # to_json returns a JSON-of-list; embed it under {"corrections": ...}
    corr_list = json.loads(corr_json)
    _atomic_write(path, json.dumps({"corrections": corr_list}, indent=2))


def build_body_pose_edit(
    *,
    track_id: int,
    frame: int,
    joint_index: int,
    axis_angle: list[float] | np.ndarray,
    user: str,
) -> Correction:
    """Build a single-frame POSE_BODY_JOINT correction for one joint's axis-angle."""
    aa = np.asarray(axis_angle, dtype=float).reshape(3)
    ts = _now_iso()
    return make_keyframes(
        f"manual-{user}-t{track_id}-f{frame}-j{joint_index}-{ts}-{_uid()}",
        CorrectionTarget(
            kind=TargetKind.POSE_BODY_JOINT,
            subject_track_id=int(track_id),
            joint_index=int(joint_index),
        ),
        (int(frame), int(frame)),
        key_frames=np.array([float(frame)]),
        key_values=aa.reshape(1, 3),
        interp="slerp",
        note=f"manual-{user}-{ts}",
    )


def build_root_edit(
    *,
    track_id: int,
    frame: int,
    kind: str,
    delta: list[float] | np.ndarray,
    user: str,
    frame_end: int | None = None,
) -> Correction:
    """Build a CONSTANT_OFFSET correction for the subject root.

    ``kind`` is a wire string in ``_ROOT_KINDS``. ``delta`` is a pure nudge the
    engine composes onto the resolved root: an axis-angle offset for
    ROOT_ORIENTATION (left-composed) or an xyz metre offset for
    ROOT_TRANSLATION (added). The client sends only the delta — no current-value
    round-trip, no client-side quaternion math.

    ``frame_end`` widens the range past the single frame the user was looking at.
    Placement error inherited from the calibration is systematic along a track, so
    correcting it on one frame alone would trade a steady offset for a visible pop.
    """
    tk = _ROOT_KINDS[kind]
    d = np.asarray(delta, dtype=float).reshape(3)
    ts = _now_iso()
    end = int(frame if frame_end is None else frame_end)
    return make_offset(
        f"manual-{user}-t{track_id}-f{frame}-{kind}-{ts}-{_uid()}",
        CorrectionTarget(kind=tk, subject_track_id=int(track_id)),
        (int(frame), end),
        d,
        note=f"manual-{user}-{ts}",
    )


def build_calibration_edit(
    *,
    frame: int,
    frame_end: int,
    matrix: np.ndarray,
    user: str,
    note: str | None = None,
) -> Correction:
    """Build a FIELD_CALIBRATION correction re-registering the pitch on its own plane (#112).

    Whole-clip by default (``frame_end`` at the last frame): a layout that is a metre out on one
    frame is a metre out on all of them, since the offset lives in the pitch model's placement
    and not in the pan. Narrowing the range is possible but is almost always the wrong answer —
    it buys alignment on one frame and a jump on the next.
    """
    ts = _now_iso()
    return Correction(
        id=f"pitch-{user}-f{frame}-{ts}-{_uid()}",
        target=CorrectionTarget(kind=TargetKind.FIELD_CALIBRATION),
        frame_range=FrameRange(int(frame), int(frame_end)),
        mode=CorrectionMode.CONSTANT_OFFSET,
        payload=PlaneTransformPayload(matrix=matrix),
        note=note or f"manual-{user}-{ts}",
        created_at=ts,
    )


#: Marks the one correction the typed layout panel owns. A drag appends a correction per
#: gesture, because each gesture was aimed at the layout as the previous one left it. The panel
#: is not a gesture — its sliders state where the layout should *end up* — so it rewrites a
#: single correction in place, and its sliders can therefore hold a value instead of springing
#: back to zero after every commit.
PANEL_NOTE = "layout-panel"


def upsert_panel_calibration_edit(path: Path, correction: Correction) -> list[Correction]:
    """Replace the panel's correction **in place**, or append it if this is the first one.

    In place matters: ``plane_adjustment`` composes right-to-left in insertion order, so moving
    the panel's entry to the end would silently reinterpret every drag made after it.
    """
    with _LOCK:
        current = load_edits(path)
        for i, c in enumerate(current):
            if c.target.kind is TargetKind.FIELD_CALIBRATION and c.note == PANEL_NOTE:
                current[i] = correction
                break
        else:
            current.append(correction)
        save_edits(path, current)
        return current


def remove_panel_calibration_edit(path: Path) -> bool:
    """Drop the panel's correction — what "back to neutral" means on disk."""
    with _LOCK:
        current = load_edits(path)
        kept = [
            c for c in current
            if not (c.target.kind is TargetKind.FIELD_CALIBRATION and c.note == PANEL_NOTE)
        ]
        if len(kept) == len(current):
            return False
        save_edits(path, kept)
        return True


def pop_last_calibration_edit(path: Path) -> Correction | None:
    """Pop the most recent pitch-layout edit — undo for the one correction with no track_id."""
    with _LOCK:
        current = load_edits(path)
        for i in range(len(current) - 1, -1, -1):
            if current[i].target.kind is TargetKind.FIELD_CALIBRATION:
                popped = current.pop(i)
                save_edits(path, current)
                return popped
        return None


def append_edit(path: Path, correction: Correction) -> list[Correction]:
    """Append ``correction`` to ``path`` under lock and return the full list."""
    with _LOCK:
        current = load_edits(path)
        current.append(correction)
        save_edits(path, current)
        return current


def pop_last_matching(
    path: Path,
    *,
    track_id: int,
    frame: int,
    joint_index: int | None = None,
    kind: TargetKind | None = None,
) -> Correction | None:
    """Pop the most recent edit matching (track_id, frame, joint_index?, kind?) — undo primitive."""
    with _LOCK:
        current = load_edits(path)
        for i in range(len(current) - 1, -1, -1):
            c = current[i]
            if c.target.subject_track_id != track_id:
                continue
            if not (c.frame_range.start <= frame <= c.frame_range.end):
                continue
            if joint_index is not None and c.target.joint_index != joint_index:
                continue
            if kind is not None and c.target.kind != kind:
                continue
            popped = current.pop(i)
            save_edits(path, current)
            return popped
        return None
