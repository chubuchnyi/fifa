"""Two ids, one human: the handover merge (П3 + П2, #135).

Every fixture here is the shape of a case the user actually judged on 2026-08-07, not an
invented one. The reference scene `out/cue/scene_off.json` is the source:

* **t3 + t66, t10 + t77, t15 + t25** — a subject measured to f35 and another measured from f33,
  2.04 m apart. The eye called each pair one player.
* **t20** — a real player the pipeline lost, whom the criteria first read as a phantom. He is a
  candidate head for t25 at 2.09 m, but t15 claims t25 at 0.85 m, so the *assignment* must leave
  t20 alone. The user's correction is what created this rule: «моя ошибка, большую часть клипа
  закрыт игроками 15 и 17… Положение его как раз всегда должно быть.»
* **t10 within 0.05 m of t5 after merging** — the reason `suspect` exists. WorldPose says two
  real players do come that close (39 pairs in 20 clips inside 0.5 m, one for 3.0 s), so this is
  reported and never rejected.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.orchestration.handover import HandoverConfig, merge_handovers
from pitch3d.core.scene.motion import PoseSequence, Provenance, SmplxShape, SubjectMotion
from pitch3d.core.scene.subject import Subject

SPAN = 60
MEAS, IMP = Provenance.MEASURED.value, Provenance.IMPUTED.value


def _subject(tid: int, measured: range, *, x: float, y: float = 0.0,
             team: str | None = "B", betas: float = 0.0) -> Subject:
    """A subject dense over the whole clip, measured only on ``measured`` — the post-coherence
    shape: the rest is an imputed mannequin whose root coasts and whose limbs are frozen."""
    frames = np.arange(SPAN)
    prov = np.array([MEAS if f in measured else IMP for f in frames])
    transl = np.zeros((SPAN, 3))
    transl[:, 0] = x + 0.01 * frames           # drifts slowly, so endpoints differ
    transl[:, 1] = y
    body = np.zeros((SPAN, 3, 3))
    body[list(measured), 0, 2] = 0.3           # measured frames articulate; imputed do not
    return Subject(
        track_id=tid,
        proposal=SubjectMotion(
            shape=SmplxShape(betas=np.full(10, betas)),
            pose=PoseSequence(frames=frames, global_orient=np.zeros((SPAN, 3)),
                              body_pose=body, transl=transl, provenance=prov),
        ),
        team_id=team,
    )


def _cfg(**kw) -> HandoverConfig:
    return HandoverConfig(enabled=True, **kw)


def test_a_handover_pair_becomes_one_subject(make_scene):
    """The regression the user named: a head that dies where a tail is born is one player."""
    scene = make_scene(subjects=[
        _subject(3, range(0, 36), x=10.0),
        _subject(66, range(33, SPAN), x=10.3),
        _subject(4, range(0, SPAN), x=40.0),       # measured throughout: never a candidate
    ])
    out, rep = merge_handovers(scene, _cfg())

    assert rep.merges == [(3, 66)]
    assert [s.track_id for s in out.subjects] == [3, 4]
    assert rep.n_in == 3 and rep.n_out == 2


def test_the_merged_subject_carries_no_mannequin_frames(make_scene):
    """П2: the duplicate half is a frozen mannequin, and merging is what removes it."""
    scene = make_scene(subjects=[
        _subject(3, range(0, 36), x=10.0),
        _subject(66, range(33, SPAN), x=10.3),
    ])
    out, rep = merge_handovers(scene, _cfg())

    prov = np.asarray(out.subject(3).proposal.pose.provenance)
    assert not (prov == IMP).any(), (
        f'{int((prov == IMP).sum())} mannequin frames survived the merge')
    assert rep.mannequin_frames_dropped > 0
    # and the survivor really is measured over both halves' spans, not just one
    measured = np.flatnonzero(prov == MEAS)
    assert measured.min() == 0 and measured.max() == SPAN - 1


def test_two_humans_measured_at_once_are_never_merged(make_scene):
    """The load-bearing gate: simultaneity beats proximity, however close the endpoints are.

    The overlap is 11 frames but the *gap* is only −10, so this reaches `max_both` rather than
    being thrown out by `max_gap` first — the first version of this fixture did not, and the
    mutation that deletes the simultaneity gate passed it.
    """
    scene = make_scene(subjects=[
        _subject(3, range(0, 36), x=10.0),
        _subject(66, range(25, SPAN), x=10.05),   # 11 overlapping measured frames, gap -10
    ])
    out, rep = merge_handovers(scene, _cfg())

    assert rep.merges == []
    assert len(out.subjects) == 2
    # …and the same pair DOES merge once the overlap alone is relaxed, proving that gate is why.
    _out2, rep2 = merge_handovers(scene, _cfg(max_both=20))
    assert rep2.merges == [(3, 66)]


def test_a_different_team_is_never_merged(make_scene):
    scene = make_scene(subjects=[
        _subject(3, range(0, 36), x=10.0, team="B"),
        _subject(66, range(33, SPAN), x=10.3, team="A"),
    ])
    _out, rep = merge_handovers(scene, _cfg())
    assert rep.merges == []


def test_the_assignment_leaves_the_loser_whole(make_scene):
    """t20's case. Two heads want the same tail; the nearer wins and the other stays a player.

    Reporting every candidate instead of assigning would convict t20 — whom the eye called
    correct — of being half of someone else.
    """
    scene = make_scene(subjects=[
        _subject(15, range(0, 27), x=10.0),        # ends near t25 -> the close claim
        _subject(20, range(10, 35), x=12.0),       # ends further away -> the far claim
        _subject(25, range(38, SPAN), x=10.4),
    ])
    out, rep = merge_handovers(scene, _cfg())

    assert rep.merges == [(15, 25)]
    assert 20 in [s.track_id for s in out.subjects], 't20 must survive as his own player'
    assert any(head == 20 for head, _tail, _d in rep.runners_up)


def test_disabled_is_the_default_and_returns_the_scene_untouched(make_scene):
    scene = make_scene(subjects=[
        _subject(3, range(0, 36), x=10.0),
        _subject(66, range(33, SPAN), x=10.3),
    ])
    out, rep = merge_handovers(scene, HandoverConfig())
    assert out is scene and rep.merges == [] and rep.n_out == 2


def test_a_merge_that_lands_inside_a_third_subject_is_flagged_not_rejected(make_scene):
    """R-6: mark, never hide. WorldPose says real players do get this close, so it is a flag."""
    scene = make_scene(subjects=[
        _subject(10, range(0, 46), x=10.0),
        _subject(77, range(55, SPAN), x=10.2),
        _subject(5, range(0, SPAN), x=10.1),       # stands right where the merge will land
    ])
    out, rep = merge_handovers(scene, _cfg())

    assert rep.merges == [(10, 77)], 'the merge must still happen'
    assert any(tid == 10 and other == 5 for tid, other, *_ in rep.suspect)
    assert len(out.subjects) == 2


def test_the_survivor_keeps_the_lower_id_and_the_better_measured_shape(make_scene):
    """betas fitted on 36 frames beat betas fitted on 5, whichever id happens to be lower."""
    scene = make_scene(subjects=[
        _subject(66, range(0, 36), x=10.0, betas=0.7),   # long half, HIGHER id
        _subject(3, range(38, SPAN), x=10.3, betas=0.1),  # short half, lower id
    ])
    out, _rep = merge_handovers(scene, _cfg())

    surv = out.subject(3)
    assert surv is not None, 'the lower id must survive, as in the 2D stitcher'
    assert surv.proposal.shape.betas[0] == pytest.approx(0.7)


def test_the_seam_articulates_instead_of_freezing(make_scene):
    """A merged pair's gap is bridged by the coherence interpolator, not held like an imputed run.

    This is the whole difference between a stitch and a mannequin: an imputed run carries
    exactly 0.00 rad of limb travel, an interpolated one does not.
    """
    scene = make_scene(subjects=[
        _subject(3, range(0, 30), x=10.0),
        _subject(66, range(40, SPAN), x=10.3),
    ])
    out, rep = merge_handovers(scene, _cfg())

    pose = out.subject(3).proposal.pose
    prov = np.asarray(pose.provenance)
    seam = np.flatnonzero(prov == Provenance.INTERPOLATED.value)
    assert seam.size == rep.seam_frames_filled > 0
    body = np.asarray(pose.body_pose)
    travel = np.abs(np.diff(body[seam.min() - 1:seam.max() + 2], axis=0)).sum()
    assert travel > 0.0, 'the bridged frames must move, or the merge just built a longer mannequin'
