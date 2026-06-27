"""A-10: the bounded, attention-driven autonomy loop — the eval harness + its guarantees.

The headline (:func:`test_eval_harness_seeded_wrong_pose_clears_attention`) is the eval: seed
one subject's root off its measured anchor, let :func:`auto_correct` run, and prove the
"needs attention" list clears — a measured fact (the resolved root is back on the anchor),
no GPU/LLM. The rest pin the properties that keep the loop honest: the edit is bounded
(clipped, not a teleport), targeting picks the worst first, the proposal is never mutated
(ADR-0002), and the run is deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.agent import (
    EditBudget,
    auto_correct,
    propose_anchor_offset,
    rescore_from_anchors,
    residual_to_confidence,
)
from pitch3d.core.orchestration.assemble import resolve_scene
from pitch3d.core.scene.motion import N_SMPLX_BODY_JOINTS, PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.review import attention_list
from pitch3d.core.scene.subject import Subject

_N = 8


def _subject(track_id: int, xy, *, z: float = 1.0) -> Subject:
    """A subject whose root sits at constant ground ``xy`` (``(2,)``) over ``_N`` frames."""
    transl = np.zeros((_N, 3))
    transl[:, :2] = np.asarray(xy, dtype=float)
    transl[:, 2] = z
    pose = PoseSequence(
        frames=np.arange(_N),
        global_orient=np.zeros((_N, 3)),
        body_pose=np.zeros((_N, N_SMPLX_BODY_JOINTS, 3)),
        transl=transl,
    )
    return Subject(
        track_id=track_id,
        proposal=SubjectMotion(shape=SmplxShape(betas=np.zeros(10)), pose=pose),
    )


def _resolved_xy(scene, track_id: int) -> np.ndarray:
    return resolve_scene(scene).subject(track_id).proposal.pose.transl[:, :2]


# --- the confidence bridge (residual → attention signal) ----------------------------
def test_residual_to_confidence_crosses_threshold_at_tolerance():
    m = 0.5
    assert residual_to_confidence(np.array([0.0]), m)[0] == pytest.approx(1.0)
    assert residual_to_confidence(np.array([m]), m)[0] == pytest.approx(0.5)   # the attention edge
    assert residual_to_confidence(np.array([2 * m]), m)[0] == pytest.approx(1 / 3)
    seq = residual_to_confidence(np.array([0.0, 0.5, 1.0, 2.0]), m)
    assert np.all(np.diff(seq) < 0)                                            # monotone decreasing


def test_rescore_flags_off_anchor_and_passes_on_anchor(make_scene):
    scene = make_scene(subjects=[_subject(1, [0.0, 0.0]), _subject(2, [3.0, 0.0])])
    anchors = {1: np.array([0.0, 0.0]), 2: np.array([0.0, 0.0])}
    scored = rescore_from_anchors(scene, anchors, max_residual_m=0.5)
    items = attention_list(scored)
    flagged = {it.track_id for it in items}
    assert flagged == {2}  # only the off-anchor one


# --- one bounded edit ---------------------------------------------------------------
def test_propose_offset_is_clipped_to_budget(make_scene):
    scene = make_scene(subjects=[_subject(1, [4.0, 0.0])])
    anchor = np.array([0.0, 0.0])
    out = propose_anchor_offset(
        scene, 1, anchor, budget=EditBudget(max_abs_change_m=1.0, blend=1.0), edit_id="e"
    )
    assert out is not None
    _corr, step = out
    assert step.clipped
    assert np.linalg.norm(step.delta_m) == pytest.approx(1.0)  # capped, not the full 4 m


def test_propose_offset_none_when_on_anchor(make_scene):
    scene = make_scene(subjects=[_subject(1, [0.0, 0.0])])
    out = propose_anchor_offset(scene, 1, np.array([0.0, 0.0]), budget=EditBudget(), edit_id="e")
    assert out is None


# --- the eval harness ---------------------------------------------------------------
def test_eval_harness_seeded_wrong_pose_clears_attention(make_scene):
    # two subjects on their measured tracks, one seeded 2 m off (the wrong reconstruction)
    scene = make_scene(
        subjects=[_subject(1, [0.0, 0.0]), _subject(2, [2.0, 0.0]), _subject(3, [5.0, 5.0])]
    )
    anchors = {1: np.array([0.0, 0.0]), 2: np.array([0.0, 0.0]), 3: np.array([5.0, 5.0])}

    work, report = auto_correct(scene, anchors, budget=EditBudget(max_abs_change_m=2.5))

    assert report.attention_before > 0 and report.attention_after == 0
    assert report.cleared and report.edits_applied == 1
    np.testing.assert_allclose(
        _resolved_xy(work, 2), np.broadcast_to(anchors[2], (_N, 2)), atol=1e-9
    )                                                                          # root back on anchor
    assert scene.corrections == [] and not work.corrections == []  # input untouched, work edited
    np.testing.assert_allclose(scene.subject(2).proposal.pose.transl[:, 0], 2.0)  # proposal intact


def test_targeting_fixes_the_worst_subject_first(make_scene):
    scene = make_scene(subjects=[_subject(1, [1.0, 0.0]), _subject(2, [3.0, 0.0])])
    anchors = {1: np.array([0.0, 0.0]), 2: np.array([0.0, 0.0])}
    _work, report = auto_correct(
        scene, anchors, budget=EditBudget(max_edits=1, max_abs_change_m=5.0)
    )
    assert report.steps[0].track_id == 2  # the 3 m error, not the 1 m


def test_budget_bites_then_converges_over_bounded_steps(make_scene):
    anchors = {1: np.array([0.0, 0.0])}

    # max_edits=1 with a 0.5 m cap cannot fix a 2 m error in one step
    s = make_scene(subjects=[_subject(1, [2.0, 0.0])])
    _w, tight = auto_correct(s, anchors, budget=EditBudget(max_edits=1, max_abs_change_m=0.5))
    assert not tight.cleared and tight.edits_applied == 1 and tight.steps[0].clipped

    # the same cap with room to iterate converges: 2 m → ≤0.5 m tolerance in 3 bounded pulls
    s2 = make_scene(subjects=[_subject(1, [2.0, 0.0])])
    work, loose = auto_correct(s2, anchors, budget=EditBudget(max_edits=10, max_abs_change_m=0.5))
    assert loose.cleared and loose.edits_applied == 3
    assert np.linalg.norm(_resolved_xy(work, 1)[0] - anchors[1]) <= 0.5 + 1e-9


def test_on_anchor_scene_needs_no_edits(make_scene):
    scene = make_scene(subjects=[_subject(1, [0.0, 0.0]), _subject(2, [1.0, 1.0])])
    anchors = {1: np.array([0.0, 0.0]), 2: np.array([1.0, 1.0])}
    _work, report = auto_correct(scene, anchors)
    assert report.attention_before == 0 and report.edits_applied == 0 and report.cleared


def test_auto_correct_is_deterministic(make_scene):
    def run():
        scene = make_scene(subjects=[_subject(1, [3.0, 1.0]), _subject(2, [0.0, 0.0])])
        anchors = {1: np.array([0.0, 0.0]), 2: np.array([0.0, 0.0])}
        return auto_correct(
            scene, anchors, budget=EditBudget(max_abs_change_m=1.0, max_edits=10)
        )[1]

    a, b = run(), run()
    assert (a.edits_applied, a.cleared) == (b.edits_applied, b.cleared)
    assert [s.delta_m for s in a.steps] == [s.delta_m for s in b.steps]
