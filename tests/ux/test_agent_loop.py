"""UX / agent-loop invariants — preview non-mutation, reset-to-model, observe, seams (ADR-0008)."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.correction.engine import make_offset
from pitch3d.core.ports.observation import ObservationKind
from pitch3d.core.scene.camera import CameraIntrinsics, CameraTrack
from pitch3d.core.scene.layers import CorrectionTarget, TargetKind


def test_preview_is_nondestructive_fr23(reconstructed):
    app, scene_id = reconstructed
    track_id = app.get_scene(scene_id).subjects[0].track_id
    candidate = make_offset(
        "cand", CorrectionTarget(kind=TargetKind.ROOT_TRANSLATION, subject_track_id=track_id),
        (0, 3), np.array([0, 0, 0.2]),
    )
    n_before = len(app.get_scene(scene_id).corrections)
    result = app.preview(scene_id, candidate)
    assert result["committed"] is False
    assert result["max_abs_change"] == pytest.approx(0.2, abs=1e-6)
    assert len(app.get_scene(scene_id).corrections) == n_before  # nothing stored


def test_preview_ball_target_without_a_ball_is_zero_change(app, make_scene):
    # A BALL_POSITION preview on a scene with no ball must degrade to a no-op (max change 0),
    # mirroring the resolve-time None guard — never pass None into resolve_ball and crash.
    scene = make_scene(ball=None)
    app._scenes[scene.id] = scene
    candidate = make_offset(
        "cand", CorrectionTarget(kind=TargetKind.BALL_POSITION), (0, 3), np.array([0, 0, 0.5])
    )
    result = app.preview(scene.id, candidate)
    assert result["committed"] is False
    assert result["max_abs_change"] == 0.0


def test_preview_non_ball_target_requires_a_subject_id(reconstructed):
    # A non-ball target carrying no subject_track_id is malformed; preview must reject it
    # explicitly rather than passing None into scene.subject(...).
    app, scene_id = reconstructed
    candidate = make_offset(
        "cand", CorrectionTarget(kind=TargetKind.ROOT_TRANSLATION, subject_track_id=None),
        (0, 3), np.array([0, 0, 0.2]),
    )
    with pytest.raises(ValueError, match="subject_track_id"):
        app.preview(scene_id, candidate)


def test_toggle_correction_resets_to_model_ux5(reconstructed):
    app, scene_id = reconstructed
    track_id = app.get_scene(scene_id).subjects[0].track_id
    baseline = app.resolved(scene_id).subject(track_id).proposal.pose.transl[:, 2].copy()

    corr = app.apply_offset(
        scene_id, {"kind": "root_translation", "subject_track_id": track_id}, (0, 3), [0, 0, 0.5]
    )
    lifted = app.resolved(scene_id).subject(track_id).proposal.pose.transl[:, 2]
    assert not np.allclose(lifted, baseline)

    app.set_correction_enabled(scene_id, corr.id, False)  # compare/reset without deleting
    reset = app.resolved(scene_id).subject(track_id).proposal.pose.transl[:, 2]
    np.testing.assert_allclose(reset, baseline)


def test_observe_returns_multiview_overlay_and_ui(reconstructed):
    app, scene_id = reconstructed
    obs = app.observe(scene_id, frame=0, include_ui=True, n_orbit=2)
    kinds = {img.kind for img in obs.images}
    assert {ObservationKind.SCENE_3D, ObservationKind.FRAME_OVERLAY, ObservationKind.UI} <= kinds
    assert len(obs.images) == 8  # 4 default views + 2 orbit + overlay + UI
    assert obs.summary  # textual digest accompanies the pixels


def test_attention_flags_airborne_ball_first(app, clip):
    episode = app.register_clip(clip)
    on_ground = np.ones(clip.n_frames, dtype=bool)
    on_ground[1:-1] = False  # bracketed arc → apex confidence dips below threshold
    scene_id = app.run_reconstruction(episode.id, on_ground=on_ground)
    items = app.get_attention(scene_id)
    assert items, "an airborne ball must surface in the attention list (UX-4)"
    assert items[0].reason == "low_ball_height"
    assert [it.score for it in items] == sorted([it.score for it in items], reverse=True)


def test_seam_a_video_is_not_editable(app, clip):
    intr = CameraIntrinsics(fx=100, fy=100, cx=320, cy=180, width=640, height=360)
    seam_a = app.ports.viewsynth.render_orbit(clip, CameraTrack.identity(intr, clip.n_frames))
    assert seam_a.editable is False  # ADR-0007: orbit video is feedback, never edited

    seam_b = app.ports.viewsynth.amplify(clip, n_views=2, deviation=0.4)
    assert all(r.frustum_overlap == pytest.approx(0.6) for r in seam_b)  # 1 - deviation
