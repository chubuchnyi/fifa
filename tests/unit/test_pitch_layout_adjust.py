"""Hand-registering the pitch may not cost us the camera (#112).

The layout drag exists because a calibration can be a perfectly good camera and still place the
pitch model a metre off along its own plane — no residual we compute can see that, since the
residual is scored against the same lines that placed it. The operator's eye is the instrument.

That freedom is exactly what makes it dangerous. #107 had just established that a scene carries
ONE camera and proved it by refusing to invent one; a manual nudge that quietly re-broke that
would undo the more valuable fix. Hence the choice of a **world-plane similarity** composed as
``H @ B`` rather than the obvious image-space ``A @ H``: it is the one form that provably cannot
turn a camera-realizable calibration into an unrealizable one. These tests pin that claim, and
pin that the gesture does what the user thinks it does.
"""

from __future__ import annotations

import numpy as np
import pytest
from poseannot.camera import (
    adjusted_calibration,
    adjusted_camera,
    image_to_ground,
    plane_adjustment,
    plane_similarity,
    project_ground,
    world_to_image,
)
from poseannot.edits import append_edit, build_calibration_edit, pop_last_calibration_edit

from pitch3d.core.correction.plane import apply_plane_corrections, has_plane_corrections
from pitch3d.core.correction.sidecar import merge_sidecar, sidecar_for
from pitch3d.core.scene.layers import (
    Correction,
    CorrectionMode,
    CorrectionTarget,
    FrameRange,
    PlaneTransformPayload,
    TargetKind,
)
from pitch3d.core.scene.plane_camera import camera_from_calibration
from pitch3d.core.scene.projection import quat_to_rotation_matrix
from pitch3d.eval.synthetic import CAMERA_VIEWS, generate_scene

PROBE = np.array([[0.0, 0.0], [30.0, 20.0], [-40.0, -25.0], [10.0, -30.0]])


def _truth():
    scene = generate_scene(n_subjects=1, n_frames=3, seed=0, camera=CAMERA_VIEWS["main_sideline"])
    return scene, scene.field_calibration()


def _edit(matrix, frames=(0, 2)):
    return Correction(
        id="t",
        target=CorrectionTarget(kind=TargetKind.FIELD_CALIBRATION),
        frame_range=FrameRange(*frames),
        mode=CorrectionMode.CONSTANT_OFFSET,
        payload=PlaneTransformPayload(matrix=matrix),
    )


def test_dragging_the_layout_puts_the_grabbed_point_under_the_cursor():
    # The whole contract of the gesture, stated in pixels: whatever the user grabbed ends up
    # where they dropped it. Everything below is about not wrecking anything else while doing it.
    scene, cal = _truth()
    w, h = int(scene.intrinsics.width), int(scene.intrinsics.height)
    grab = np.array([[w * 0.5, h * 0.6]])
    drop = np.array([[w * 0.5 + 90.0, h * 0.6 - 40.0]])
    src, dst = image_to_ground(grab, cal, 0)[0], image_to_ground(drop, cal, 0)[0]

    b = plane_similarity(anchor=src, src=src, dst=dst, turn=False)
    moved = adjusted_calibration(cal, [_edit(b)])
    np.testing.assert_allclose(project_ground(src.reshape(1, 2), world_to_image(moved, 0)),
                               drop, atol=1e-6)


def test_the_turn_handle_spins_the_layout_about_the_move_handle():
    # Two gestures span the 4-DOF similarity only if `turn` really leaves the anchor alone —
    # otherwise the user chases a layout that slides away as they rotate it.
    scene, cal = _truth()
    w, h = int(scene.intrinsics.width), int(scene.intrinsics.height)
    pts = image_to_ground(np.array([[w * 0.5, h * 0.6], [w * 0.8, h * 0.6]]), cal, 0)
    anchor, src = pts[0], pts[1]
    dst = anchor + np.array([[0.0, -1.0], [1.0, 0.0]]) @ (src - anchor) * 1.5  # 90 deg, 1.5x

    b = plane_similarity(anchor=anchor, src=src, dst=dst, turn=True)
    np.testing.assert_allclose(b[:2, :2] @ src + b[:2, 2], dst, atol=1e-9)
    np.testing.assert_allclose(b[:2, :2] @ anchor + b[:2, 2], anchor, atol=1e-9)
    assert np.hypot(b[0, 0], b[1, 0]) == pytest.approx(1.5)


def test_a_hand_registered_pitch_is_still_the_same_one_camera():
    # The #107 property, carried through the edit. A world-plane similarity keeps K[r1 r2 t] @ B
    # of the form K[r1' r2' t'], so the calibration stays decomposable — and the focal it reports
    # is unchanged, because B never touched K.
    scene, cal = _truth()
    w, h = int(scene.intrinsics.width), int(scene.intrinsics.height)
    before = camera_from_calibration(cal, width=w, height=h)
    assert before.realizable

    b = plane_similarity(
        anchor=np.zeros(2), src=np.array([20.0, 0.0]), dst=np.array([4.0, 18.0]), turn=True
    )
    b[:2, 2] += np.array([2.5, -1.25])
    after = camera_from_calibration(adjusted_calibration(cal, [_edit(b)]), width=w, height=h)

    assert after.realizable, "a layout drag must not cost us the measured camera"
    assert after.focal_px == pytest.approx(before.focal_px, rel=1e-6)
    assert after.reprojection_px < 1e-6


def test_the_layout_drag_moves_scene_camera_with_the_calibration():
    # Found by the #107 badge on the first live drag, not by reasoning: moving the pitch under a
    # fixed `scene.camera` split the two descriptions again — 0 px before, 2500 px after — which
    # is #61 walking back in through the editor. Realizable-in-principle was never the invariant;
    # "the scene's camera IS the scene's calibration" is.
    scene, cal = _truth()
    fit = camera_from_calibration(
        cal, width=int(scene.intrinsics.width), height=int(scene.intrinsics.height)
    )
    b = plane_similarity(anchor=np.zeros(2), src=np.array([12.0, 3.0]),
                         dst=np.array([-2.0, 9.0]), turn=True)
    edits = [_edit(b)]

    moved_cam = adjusted_camera(fit.camera, edits)
    moved_w2i = world_to_image(adjusted_calibration(cal, edits), 0)
    rot = quat_to_rotation_matrix(moved_cam.rotation_quat[0])
    ground = np.column_stack([PROBE, np.zeros(len(PROBE))])
    p = (ground @ rot.T + moved_cam.translation[0]) @ moved_cam.intrinsics.matrix().T
    np.testing.assert_allclose(p[:, :2] / p[:, 2, None], project_ground(PROBE, moved_w2i),
                               atol=1e-6)
    # ...and the intrinsics are untouched, because a plane transform never reaches K.
    assert moved_cam.intrinsics.fx == fit.camera.intrinsics.fx


def test_an_image_space_nudge_would_have_cost_us_the_camera():
    # Guards the choice above rather than the code: without this, `test_..._same_one_camera`
    # could be passing because the check is blind. The rejected design — shifting the drawing on
    # screen, A @ H — is not decomposable at any focal, which is why it was rejected.
    scene, cal = _truth()
    w, h = int(scene.intrinsics.width), int(scene.intrinsics.height)
    a = np.array([[1.0, 0.0, 60.0], [0.0, 1.0, -35.0], [0.0, 0.0, 1.0]])
    h_i2w = np.stack([m @ np.linalg.inv(a) for m in np.asarray(cal.homographies, dtype=float)])
    cal.homographies = h_i2w
    assert not camera_from_calibration(cal, width=w, height=h).realizable


def test_two_drags_compose_in_the_order_they_were_made():
    # Each drag is measured against the layout as it stood after the previous one, so they
    # right-multiply. Composing them the other way silently mis-places every drag after the first.
    _, cal = _truth()
    b1 = plane_similarity(anchor=np.zeros(2), src=np.zeros(2), dst=np.array([3.0, 0.0]),
                          turn=False)
    b2 = plane_similarity(anchor=np.zeros(2), src=np.array([1.0, 0.0]),
                          dst=np.array([0.0, 2.0]), turn=True)
    np.testing.assert_allclose(plane_adjustment([_edit(b1), _edit(b2)], 0), b1 @ b2, atol=1e-12)

    both = adjusted_calibration(cal, [_edit(b1), _edit(b2)])
    stepwise = adjusted_calibration(adjusted_calibration(cal, [_edit(b1)]), [_edit(b2)])
    np.testing.assert_allclose(both.homographies, stepwise.homographies, atol=1e-9)


def test_the_stored_solve_is_untouched_and_disabling_restores_it():
    # Non-destructive is the whole edit model: the user must be able to take the drag back and
    # get the solve they were given, bit for bit.
    _, cal = _truth()
    original = np.array(cal.homographies, copy=True)
    b = plane_similarity(anchor=np.zeros(2), src=np.zeros(2), dst=np.array([5.0, 5.0]),
                         turn=False)
    off = _edit(b)
    off.enabled = False

    assert adjusted_calibration(cal, [off]) is cal
    adjusted_calibration(cal, [_edit(b)])
    np.testing.assert_array_equal(cal.homographies, original)


def test_back_to_back_drags_are_two_separately_undoable_edits(tmp_path):
    # Found on the live undo: the first two pops returned the SAME id. The layout id was
    # `pitch-{user}-f{frame}-{ts}` with `frame` pinned at 0 by the whole-clip range and `ts` at
    # one-second resolution, so a nudge gesture — which is repeated by nature — minted duplicates.
    # Undo pops by position and so still worked, but two edits sharing an identity is the kind of
    # thing that reads as one edit to anything that looks them up.
    path = tmp_path / "edits.json"
    b = plane_similarity(anchor=np.zeros(2), src=np.zeros(2), dst=np.array([1.0, 0.0]), turn=False)
    made = [build_calibration_edit(frame=0, frame_end=59, matrix=b, user="admin") for _ in range(4)]
    for c in made:
        append_edit(path, c)

    assert len({c.id for c in made}) == 4
    for expected in reversed(made):
        assert pop_last_calibration_edit(path).id == expected.id
    assert pop_last_calibration_edit(path) is None


def test_a_drag_outside_its_frame_range_does_nothing():
    _, cal = _truth()
    b = plane_similarity(anchor=np.zeros(2), src=np.zeros(2), dst=np.array([5.0, 0.0]),
                         turn=False)
    np.testing.assert_array_equal(plane_adjustment([_edit(b, frames=(1, 2))], 0), np.eye(3))


# --- the read path: does anything downstream of poseannot actually apply this? (#128) ---------


def _scene_with(cal, cam, corrections):
    """A :class:`Scene` carrying one subject, so "the bodies did not move" is checkable."""
    from pitch3d.core.scene.field import FieldModel
    from pitch3d.core.scene.scene import Scene

    return Scene(
        id="s", episode_id="e", source_id="src",
        field=FieldModel(calibration=cal), camera=cam, corrections=list(corrections),
    )


def test_the_exporter_and_the_annotator_place_the_pitch_in_the_same_pixels():
    # #128 in one assertion. The annotator previewed the drag through `adjusted_calibration`
    # and the user approved THOSE pixels; the exporter now reads the same correction through
    # `apply_plane_corrections`. If the two ever disagree, the operator judged one picture and
    # the render shipped another — which is exactly the failure that made the edit worthless.
    scene_t, cal = _truth()
    w, h = int(scene_t.intrinsics.width), int(scene_t.intrinsics.height)
    fit = camera_from_calibration(cal, width=w, height=h)
    b = plane_similarity(anchor=np.zeros(2), src=np.array([12.0, 3.0]),
                         dst=np.array([-2.0, 9.0]), turn=True)
    b[:2, 2] += np.array([1.75, -0.5])

    exported = apply_plane_corrections(_scene_with(cal, fit.camera, [_edit(b)]))

    annotator = project_ground(PROBE, world_to_image(adjusted_calibration(cal, [_edit(b)]), 0))
    rot = quat_to_rotation_matrix(exported.camera.rotation_quat[0])
    ground = np.column_stack([PROBE, np.zeros(len(PROBE))])
    p = (ground @ rot.T + exported.camera.translation[0]) @ exported.camera.intrinsics.matrix().T
    np.testing.assert_allclose(p[:, :2] / p[:, 2, None], annotator, atol=1e-6)
    # and the exported calibration is moved too, not just the camera — they are one thing (#107)
    np.testing.assert_allclose(
        project_ground(PROBE, world_to_image(exported.field.calibration, 0)), annotator, atol=1e-6
    )


def test_the_layout_edit_moves_the_camera_and_leaves_the_bodies_alone():
    # The documented semantics, pinned so it cannot be "fixed" by accident. Subjects are stored
    # in world metres; the drag says where the pitch model sits under an approximate camera, so
    # it re-registers the camera. Moving the bodies too would double-count the same correction.
    scene_t, cal = _truth()
    fit = camera_from_calibration(cal, width=int(scene_t.intrinsics.width),
                                  height=int(scene_t.intrinsics.height))
    b = np.eye(3)
    b[:2, 2] = [3.0, -2.0]
    scene = _scene_with(cal, fit.camera, [_edit(b)])
    out = apply_plane_corrections(scene)

    assert out.subjects == scene.subjects
    assert not np.allclose(out.camera.translation, scene.camera.translation)


def test_a_scene_with_no_layout_edits_comes_back_untouched():
    # Cheap, but it is the branch every ordinary export takes: no corrections must mean no copy
    # and no numerical drift through the SVD snap.
    scene_t, cal = _truth()
    fit = camera_from_calibration(cal, width=int(scene_t.intrinsics.width),
                                  height=int(scene_t.intrinsics.height))
    scene = _scene_with(cal, fit.camera, [])
    assert apply_plane_corrections(scene) is scene
    assert not has_plane_corrections(scene.corrections)


def test_a_disabled_layout_edit_is_not_exported():
    # Toggling a correction off in the editor has to mean the render stops using it, or "disable"
    # is a lie in exactly the place a user would check it.
    scene_t, cal = _truth()
    fit = camera_from_calibration(cal, width=int(scene_t.intrinsics.width),
                                  height=int(scene_t.intrinsics.height))
    b = np.eye(3)
    b[:2, 2] = [5.0, 5.0]
    off = _edit(b)
    off.enabled = False
    assert not has_plane_corrections([off])
    assert apply_plane_corrections(_scene_with(cal, fit.camera, [off])).camera is fit.camera


def test_anim_export_main_actually_calls_the_read_path():
    # The semantics above were all true before #128 too — what was missing was the CALL. Parsing
    # for it is the same trick `test_gate_chain_parity.py` uses: cheaper than rendering, and it
    # fails if someone deletes the wiring while leaving the helper importable.
    import ast
    import inspect

    from pitch3d.app import anim_export

    src = inspect.getsource(anim_export.main)
    called = {
        n.func.id for n in ast.walk(ast.parse(src.lstrip()))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "apply_plane_corrections" in called, "the export dropped the hand registration again"
    assert "has_plane_corrections" in called
    assert "merge_sidecar" in called, "the export stopped reading the annotator's edits.json"
    assert "sidecar_for" in called


# --- and does the export ever SEE the edit? the annotator writes a sidecar, not the scene (#128)


def test_the_export_finds_the_sidecar_the_annotator_wrote(tmp_path):
    # The gap that made the read path above worthless on real data: `poseannot` never rewrites
    # scene.json, so every operator edit lives in a sibling file that nothing outside the
    # annotator opened. Both names are minted by poseannot/clips.py.
    (tmp_path / "scene_edits.json").write_text('{"corrections": []}')
    assert sidecar_for(tmp_path / "scene.json") == tmp_path / "scene_edits.json"
    (tmp_path / "scene_edits.json").unlink()
    (tmp_path / "edits.json").write_text('{"corrections": []}')
    assert sidecar_for(tmp_path / "scene.json") == tmp_path / "edits.json"
    assert sidecar_for(tmp_path / "nowhere" / "scene.json") is None


def test_a_sidecar_drag_reaches_the_render(tmp_path):
    # End to end in miniature: the annotator appends a FIELD_CALIBRATION row to its sidecar, and
    # a scene loaded WITHOUT that file must come back with the pitch moved once it is merged.
    scene_t, cal = _truth()
    fit = camera_from_calibration(cal, width=int(scene_t.intrinsics.width),
                                  height=int(scene_t.intrinsics.height))
    b = np.eye(3)
    b[:2, 2] = [2.5, -1.25]
    path = tmp_path / "edits.json"
    append_edit(path, build_calibration_edit(matrix=b, frame=0, frame_end=2, user="t"))

    scene = _scene_with(cal, fit.camera, [])
    assert not has_plane_corrections(scene.corrections)  # the render's view before the merge
    merged = merge_sidecar(scene, path)
    assert has_plane_corrections(merged.corrections)
    assert not np.allclose(apply_plane_corrections(merged).camera.translation,
                           scene.camera.translation)


def test_the_sidecar_stacks_after_the_scenes_own_corrections(tmp_path):
    # Order is the meaning here — each drag was measured against the layout the previous ones
    # left. The annotator appends the operator's rows after the pipeline's, so the export must
    # too, or the same file renders as a different edit.
    scene_t, cal = _truth()
    fit = camera_from_calibration(cal, width=int(scene_t.intrinsics.width),
                                  height=int(scene_t.intrinsics.height))
    first, second = np.eye(3), np.eye(3)
    first[:2, 2] = [4.0, 0.0]
    second[0, 0] = second[1, 1] = 1.5
    path = tmp_path / "edits.json"
    append_edit(path, build_calibration_edit(matrix=second, frame=0, frame_end=2, user="t"))

    merged = merge_sidecar(_scene_with(cal, fit.camera, [_edit(first)]), path)
    np.testing.assert_allclose(plane_adjustment(merged.corrections, 0), first @ second)


def test_a_missing_or_empty_sidecar_changes_nothing(tmp_path):
    # An ordinary pipeline export has no annotator beside it; that must not be an error, and it
    # must not be reported as "merged 0 edits" either — the caller checks for None.
    scene_t, cal = _truth()
    fit = camera_from_calibration(cal, width=int(scene_t.intrinsics.width),
                                  height=int(scene_t.intrinsics.height))
    scene = _scene_with(cal, fit.camera, [])
    assert merge_sidecar(scene, tmp_path / "absent.json") is scene
    (tmp_path / "blank.json").write_text("   ")
    assert merge_sidecar(scene, tmp_path / "blank.json") is scene
