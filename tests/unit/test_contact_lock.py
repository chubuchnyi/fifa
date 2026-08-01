"""Step 3b — contact-lock gate zeros foot slide during stance runs."""

from __future__ import annotations

import numpy as np

from pitch3d.core.config.gates import ContactProbeConfig
from pitch3d.core.correction.contact_lock import contact_lock_gate
from pitch3d.core.correction.engine import resolve_subject_motion
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.scene import Scene, Subject


def _subject(track_id: int, transl: np.ndarray) -> Subject:
    T = transl.shape[0]
    frames = np.arange(T, dtype=int)
    motion = SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(
            frames=frames, global_orient=np.zeros((T, 3)),
            body_pose=np.zeros((T, 21, 3)), transl=transl,
        ),
    )
    return Subject(track_id=track_id, proposal=motion)


def _scene(*subjects: Subject) -> Scene:
    return Scene(
        id="s", episode_id="e", source_id="c",
        subjects=list(subjects), corrections=[],
    )


def _provider(feet_by_id: dict[int, np.ndarray]):
    def p(subject):
        return feet_by_id.get(int(subject.track_id))
    return p


def test_disabled_gate_passthrough():
    """When cfg.enabled=False the gate emits nothing."""
    transl = np.zeros((5, 3))
    transl[:, 0] = np.linspace(0, 1.0, 5)  # subject moving
    s = _subject(1, transl)
    feats = np.zeros((5, 3))
    feats[:, 0] = np.linspace(0, 1.0, 5)  # foot moving with root = sliding
    scene, report = contact_lock_gate(
        _scene(s), ContactProbeConfig(enabled=False),
        foot_position_provider=_provider({1: feats}),
    )
    assert report.corrections_added == 0
    assert scene.corrections == []


def test_no_provider_passthrough():
    s = _subject(1, np.zeros((5, 3)))
    scene, report = contact_lock_gate(
        _scene(s), ContactProbeConfig(enabled=True),
        foot_position_provider=None,
    )
    assert report.corrections_added == 0


def test_stance_slide_zeroed_after_correction():
    """Root moves +1m in X, foot planted (Z=0) → gate rolls root back to anchor."""
    T = 5
    transl = np.zeros((T, 3))
    transl[:, 0] = np.linspace(0.0, 1.0, T)
    s = _subject(1, transl)
    feats = np.zeros((T, 3))
    feats[:, 0] = np.linspace(0.0, 1.0, T)  # foot slides with root
    scene, report = contact_lock_gate(
        _scene(s), ContactProbeConfig(enabled=True, slide_threshold_m=0.05,
                                     min_contact_run_frames=2),
        foot_position_provider=_provider({1: feats}),
    )
    assert report.corrections_added == 1
    assert report.runs_locked == 1
    resolved = resolve_subject_motion(s.proposal, scene.corrections_for(1))
    got = np.asarray(resolved.pose.transl)
    # root should have been pulled back to the anchor: after correction, root
    # XY at every stance frame equals the anchor's root XY
    assert np.allclose(got[:, 0], 0.0, atol=1e-9)


def test_swing_frames_untouched():
    """Stance run at frames 0-4 (foot slides with root), then swing 5-9.
    The gate zeroes root drift over the stance region; swing frames stay as
    measured (subject legitimately airborne)."""
    T = 10
    transl = np.zeros((T, 3))
    transl[:, 0] = np.linspace(0.0, 2.0, T)
    s = _subject(1, transl)
    # foot XY == root XY on stance (foot vertex at pelvis-XY, plausible for
    # a "planted" stance); airborne after.
    feats = np.zeros((T, 3))
    feats[:5, 0] = transl[:5, 0]
    feats[5:, 2] = 0.5
    feats[5:, 0] = transl[5:, 0]
    scene, report = contact_lock_gate(
        _scene(s), ContactProbeConfig(enabled=True, slide_threshold_m=0.05,
                                     min_contact_run_frames=2),
        foot_position_provider=_provider({1: feats}),
    )
    resolved = resolve_subject_motion(s.proposal, scene.corrections_for(1))
    got = np.asarray(resolved.pose.transl)
    # stance frames locked at anchor (transl[0]=0)
    assert np.allclose(got[:5, 0], 0.0, atol=1e-9)
    # swing frames still at their measured positions
    assert np.allclose(got[5:, 0], transl[5:, 0])


def test_below_threshold_slide_skipped():
    """A run with only 2cm slide is left alone (below the 5cm threshold)."""
    T = 5
    transl = np.zeros((T, 3))
    transl[:, 0] = np.linspace(0.0, 0.02, T)
    s = _subject(1, transl)
    feats = np.zeros((T, 3))
    feats[:, 0] = np.linspace(0.0, 0.02, T)
    _, report = contact_lock_gate(
        _scene(s), ContactProbeConfig(enabled=True, slide_threshold_m=0.05,
                                     min_contact_run_frames=2),
        foot_position_provider=_provider({1: feats}),
    )
    assert report.corrections_added == 0
    assert report.runs_locked == 0


def test_multi_subject_independent():
    T = 5
    transl_a = np.zeros((T, 3)); transl_a[:, 0] = np.linspace(0, 1.0, T)
    transl_b = np.zeros((T, 3))
    s1 = _subject(1, transl_a)
    s2 = _subject(2, transl_b)
    feats_a = np.zeros((T, 3)); feats_a[:, 0] = np.linspace(0, 1.0, T)
    feats_b = np.zeros((T, 3))
    scene, report = contact_lock_gate(
        _scene(s1, s2), ContactProbeConfig(enabled=True, slide_threshold_m=0.05,
                                          min_contact_run_frames=2),
        foot_position_provider=_provider({1: feats_a, 2: feats_b}),
    )
    assert report.corrections_added == 1   # only subject 1 slides
    assert report.subjects_corrected == 1


def test_empty_scene():
    scene, report = contact_lock_gate(
        _scene(), ContactProbeConfig(enabled=True),
        foot_position_provider=lambda _: None,
    )
    assert report.corrections_added == 0


def test_idempotent():
    T = 5
    transl = np.zeros((T, 3)); transl[:, 0] = np.linspace(0, 1.0, T)
    s = _subject(1, transl)
    feats = np.zeros((T, 3)); feats[:, 0] = np.linspace(0, 1.0, T)
    cfg = ContactProbeConfig(enabled=True, slide_threshold_m=0.05,
                            min_contact_run_frames=2)
    once, _ = contact_lock_gate(
        _scene(s), cfg, foot_position_provider=_provider({1: feats}),
    )
    # after resolving, the foot should now match anchor; if we re-probed
    # foot pos would still be at initial anchor → no additional slide
    feats_after = feats.copy()
    feats_after[:, 0] = 0.0   # zeroed
    _, twice_report = contact_lock_gate(
        once, cfg, foot_position_provider=_provider({1: feats_after}),
    )
    assert twice_report.corrections_added == 0
