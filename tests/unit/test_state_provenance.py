"""R4 (#96): per-frame state provenance and the ball's 3-state mode.

The point of these types is that "we measured this and were unsure" and "we invented this"
stop being the same number. Before R4 both were a low ``subject_frame_conf``, so a photoreal
renderer had no way to avoid presenting a coasted body as an observed one.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pitch3d.core.correction.coherence import (
    CoherenceConfig,
    add_temporal_coherence,
    extend_pose_to_span,
    fill_pose_gaps,
)
from pitch3d.core.correction.engine import resolve_subject_motion
from pitch3d.core.orchestration.ball_lift import lift_ball_to_3d
from pitch3d.core.scene.field import FieldCalibration
from pitch3d.core.scene.motion import (
    Ball2DTrack,
    BallMode,
    BallTrack,
    PoseSequence,
    Provenance,
    SmplxShape,
    SubjectMotion,
)
from pitch3d.core.scene.scene import Scene
from pitch3d.core.scene.serialization import encode, from_json, to_json
from pitch3d.core.scene.subject import Subject


def _pose(frames, tz=None) -> PoseSequence:
    frames = np.asarray(frames, dtype=int).reshape(-1)
    t = frames.shape[0]
    tr = np.zeros((t, 3))
    if tz is not None:
        tr[:, 2] = tz
    return PoseSequence(
        frames=frames, global_orient=np.zeros((t, 3)), body_pose=np.zeros((t, 3, 3)), transl=tr
    )


def _scene(*poses) -> Scene:
    subjects = [
        Subject(
            track_id=i,
            proposal=SubjectMotion(shape=SmplxShape(betas=np.zeros(10)), pose=p),
        )
        for i, p in enumerate(poses)
    ]
    return Scene(id="s", episode_id="e", source_id="src", subjects=subjects)


def _identity_calib(frames) -> FieldCalibration:
    n = len(frames)
    return FieldCalibration(
        homographies=np.tile(np.eye(3), (n, 1, 1)),
        frames=np.asarray(frames, dtype=int),
        confidence=np.ones(n),
    )


def _ball2d(frames) -> Ball2DTrack:
    n = len(frames)
    return Ball2DTrack(
        frames=np.asarray(frames),
        positions_2d=np.column_stack([np.arange(n), np.zeros(n)]),
        confidence=np.ones(n),
    )


# --- the types themselves ------------------------------------------------------------


def test_a_fresh_pose_claims_measurement_and_a_fresh_ball_does_not():
    # Asymmetric on purpose: a PoseSequence is built from detector output, so MEASURED is the
    # truth. A BallTrack's height needs a contact segmentation that a bare constructor has not
    # run, so claiming ground contact there would be the fabrication R-6 forbids.
    pose = _pose([0, 1, 2])
    assert list(pose.provenance) == [Provenance.MEASURED.value] * 3
    assert pose.measured_mask.all()

    ball = BallTrack(frames=np.arange(3), positions_3d=np.zeros((3, 3)),
                     height_confidence=np.ones(3))
    assert list(ball.mode) == [BallMode.UNMEASURED.value] * 3
    assert not ball.on_ground.any()


def test_on_ground_is_derived_from_mode_not_stored_beside_it():
    ball = BallTrack(
        frames=np.arange(3), positions_3d=np.zeros((3, 3)), height_confidence=np.ones(3),
        mode=[BallMode.ON_GROUND.value, BallMode.BALLISTIC.value, BallMode.UNMEASURED.value],
    )
    np.testing.assert_array_equal(ball.on_ground, [True, False, False])
    # ...and the two "False" frames are NOT the same fact, which is the whole point.
    assert ball.mode[1] != ball.mode[2]


def test_an_unknown_label_is_rejected_rather_than_silently_stored():
    with pytest.raises(ValueError, match="not valid Provenance values"):
        PoseSequence(frames=[0], global_orient=np.zeros((1, 3)), body_pose=np.zeros((1, 1, 3)),
                     transl=np.zeros((1, 3)), provenance=["probably"])
    with pytest.raises(ValueError, match="not valid BallMode values"):
        BallTrack(frames=[0], positions_3d=np.zeros((1, 3)), height_confidence=np.ones(1),
                  mode=["airborne"])


def test_copy_carries_provenance_and_does_not_alias_it():
    pose = _pose([0, 1, 2])
    pose.mark([1], Provenance.IMPUTED)
    clone = pose.copy()
    assert clone.provenance[1] == Provenance.IMPUTED.value
    clone.mark([0], Provenance.INTERPOLATED)
    assert pose.provenance[0] == Provenance.MEASURED.value


# --- the gates that fabricate rows must say so ---------------------------------------


def test_bridged_rows_are_interpolated_and_the_anchors_stay_measured():
    out, filled = fill_pose_gaps(_pose([0, 4], tz=[0.0, 4.0]), max_gap=12)
    np.testing.assert_array_equal(filled, [1, 2, 3])
    np.testing.assert_array_equal(
        out.provenance,
        [Provenance.MEASURED.value] + [Provenance.INTERPOLATED.value] * 3
        + [Provenance.MEASURED.value],
    )


def test_coasted_edges_are_imputed_not_interpolated():
    # There is no measurement on the far side of a clip edge, so the coast is inference all the
    # way down — a weaker claim than bridging between two observations.
    out, added = extend_pose_to_span(_pose([2, 3], tz=[0.0, 1.0]), 0, 5)
    np.testing.assert_array_equal(added, [0, 1, 4, 5])
    np.testing.assert_array_equal(
        out.provenance,
        [Provenance.IMPUTED.value] * 2 + [Provenance.MEASURED.value] * 2
        + [Provenance.IMPUTED.value] * 2,
    )


def test_coherence_report_counts_agree_with_the_stamped_rows():
    # The report and the provenance channel are computed independently; if a future edit stamps
    # one without the other, this catches it.
    scene = _scene(_pose([2, 6], tz=[0.0, 4.0]), _pose([0, 1, 2, 3, 4, 5, 6]))
    out, report = add_temporal_coherence(scene, CoherenceConfig(max_fill_gap=12))
    prov = np.concatenate([s.proposal.pose.provenance for s in out.subjects])
    assert int((prov == Provenance.INTERPOLATED.value).sum()) == report.filled_frames
    assert int((prov == Provenance.IMPUTED.value).sum()) == report.extended_frames
    assert report.filled_frames and report.extended_frames  # the fixture exercises both


def test_resolving_corrections_preserves_provenance():
    # Corrections change values, not whether a frame was observed.
    scene = _scene(_pose([0, 4], tz=[0.0, 4.0]))
    out, _ = add_temporal_coherence(scene, CoherenceConfig(max_fill_gap=12))
    subj = out.subjects[0]
    resolved = resolve_subject_motion(subj.proposal, out.corrections_for(subj.track_id))
    np.testing.assert_array_equal(resolved.pose.provenance, subj.proposal.pose.provenance)


# --- the ball's three states -----------------------------------------------------------


def test_lift_separates_a_fitted_arc_from_frames_it_could_not_fit():
    # Contacts at 1 and 5; frames 0 and 6 have no bracketing contact, so their Z is a hold.
    frames = np.arange(7)
    og = np.zeros(7, bool)
    og[[1, 5]] = True
    bt = lift_ball_to_3d(_ball2d(frames), _identity_calib(frames), on_ground=og, fps=25.0)
    np.testing.assert_array_equal(
        bt.mode,
        [BallMode.UNMEASURED.value, BallMode.ON_GROUND.value] + [BallMode.BALLISTIC.value] * 3
        + [BallMode.ON_GROUND.value, BallMode.UNMEASURED.value],
    )
    # the old bool collapsed frames 0/6 and 2-4 into one "not on ground" bucket
    assert set(np.asarray(bt.mode)[~bt.on_ground]) == {
        BallMode.BALLISTIC.value, BallMode.UNMEASURED.value
    }


def test_no_contact_anywhere_is_unmeasured_throughout():
    frames = np.arange(4)
    bt = lift_ball_to_3d(_ball2d(frames), _identity_calib(frames), on_ground=np.zeros(4, bool))
    assert set(bt.mode) == {BallMode.UNMEASURED.value}


# --- persistence -----------------------------------------------------------------------


def test_provenance_and_mode_survive_a_json_round_trip():
    pose = _pose([0, 1, 2])
    pose.mark([2], Provenance.IMPUTED)
    scene = _scene(pose)
    scene.ball = BallTrack(
        frames=np.arange(3), positions_3d=np.zeros((3, 3)), height_confidence=np.ones(3),
        mode=[BallMode.ON_GROUND.value, BallMode.BALLISTIC.value, BallMode.UNMEASURED.value],
    )
    back = from_json(to_json(scene))
    np.testing.assert_array_equal(
        back.subjects[0].proposal.pose.provenance, pose.provenance
    )
    np.testing.assert_array_equal(back.ball.mode, scene.ball.mode)


def test_a_pre_r4_save_migrates_its_bare_bool_without_inventing_ballistics():
    # A legacy ``on_ground=False`` meant "airborne OR unknown" and the two are not separable
    # after the fact, so the migration takes the weaker reading. Re-running the lift recovers
    # the real BALLISTIC frames.
    legacy = {
        "__type__": "BallTrack",
        "fields": {
            "frames": encode(np.arange(3)),
            "positions_3d": encode(np.zeros((3, 3))),
            "height_confidence": encode(np.ones(3)),
            "track_2d": None,
            "on_ground": encode(np.array([True, False, True])),
        },
    }
    back = from_json(json.dumps(legacy))
    np.testing.assert_array_equal(back.on_ground, [True, False, True])
    np.testing.assert_array_equal(
        back.mode,
        [BallMode.ON_GROUND.value, BallMode.UNMEASURED.value, BallMode.ON_GROUND.value],
    )
