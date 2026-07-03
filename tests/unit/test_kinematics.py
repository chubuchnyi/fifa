"""M3-9 kinematic gate: clamp impossible motion, mark (never erase) teleports (#207, R-6)."""

from __future__ import annotations

import numpy as np

from pitch3d.core.correction.engine import resolve_subject_motion
from pitch3d.core.correction.kinematics import (
    KinematicConfig,
    clamp_track_xy,
    gate_subject_xy,
    kinematic_gate,
)
from pitch3d.core.scene.layers import CorrectionMode, TargetKind
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.serialization import load_scene, save_scene
from pitch3d.core.scene.subject import Subject

FPS = 25.0
CFG = KinematicConfig()


def _motion(frames, xy, z=0.9) -> SubjectMotion:
    frames = np.asarray(frames, dtype=int).reshape(-1)
    t = frames.shape[0]
    transl = np.column_stack([np.asarray(xy, dtype=float), np.full(t, float(z))])
    return SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(
            frames=frames,
            global_orient=np.zeros((t, 3)),
            body_pose=np.zeros((t, 3, 3)),
            transl=transl,
        ),
    )


def _viols(frames, xy, cfg=CFG, fps=FPS):
    dt = np.diff(np.asarray(frames, float)) / fps
    vel = np.diff(np.asarray(xy, float), axis=0) / dt[:, None]
    speed = np.linalg.norm(vel, axis=1)
    accel = np.linalg.norm(np.diff(vel, axis=0), axis=1) / dt[1:]
    return int((speed > cfg.max_speed).sum()), int((accel > cfg.max_accel).sum())


def _walk(n, v=3.0, fps=FPS):
    """A clean constant-velocity walk along X."""
    t = np.arange(n) / fps
    return np.column_stack([v * t, np.zeros(n)])


# --- clamp_track_xy -------------------------------------------------------------------


def test_clamp_clean_track_is_identity():
    frames = np.arange(30)
    xy = _walk(30)
    out = clamp_track_xy(frames, xy, FPS, CFG)
    np.testing.assert_allclose(out, xy, atol=1e-9)


def test_clamp_removes_jitter_violations():
    rng = np.random.default_rng(7)
    frames = np.arange(50)
    xy = _walk(50) + rng.normal(0, 0.25, (50, 2))  # ~heavy tracker jitter
    sp_b, ac_b = _viols(frames, xy)
    assert ac_b > 10  # the synthetic jitter really is implausible
    out = clamp_track_xy(frames, xy, FPS, CFG)
    sp_a, ac_a = _viols(frames, out)
    assert sp_a == 0
    assert ac_a <= 1  # endpoint anchoring may leave a marginal residual, never a blow-up
    # both endpoints stay measured
    np.testing.assert_allclose(out[0], xy[0], atol=1e-9)
    np.testing.assert_allclose(out[-1], xy[-1], atol=1e-9)


def test_clamp_preserves_endpoints_and_stays_near_true_path():
    rng = np.random.default_rng(3)
    frames = np.arange(40)
    true_xy = _walk(40)
    xy = true_xy + rng.normal(0, 0.2, (40, 2))
    out = clamp_track_xy(frames, xy, FPS, CFG)
    np.testing.assert_allclose(out[0], xy[0], atol=1e-9)
    np.testing.assert_allclose(out[-1], xy[-1], atol=1e-9)
    # the feasible projection acts as a denoiser: it should stay near the TRUE underlying
    # walk (endpoints are anchored to the noisy measurements, so ~3σ leaks in there)
    assert float(np.linalg.norm(out - true_xy, axis=1).max()) < 1.5


def test_clamp_handles_nonuniform_frames():
    frames = np.array([0, 1, 2, 5, 6, 7, 8])  # interior gap: dt jumps 1→3 frames
    xy = _walk(7)
    xy[3] += [0.9, 0.0]  # a kick right at the gap
    out = clamp_track_xy(frames, xy, FPS, CFG)
    sp_a, ac_a = _viols(frames, out)
    assert (sp_a, ac_a) == (0, 0)


# --- gate_subject_xy (teleports) ------------------------------------------------------


def test_gate_marks_and_preserves_teleport():
    frames = np.arange(30)
    xy = _walk(30, v=2.0)
    xy[15:] += [1.8, 0.0]  # ID-swap teleport: 1.8m in one frame = 45 m/s
    out, teleports = gate_subject_xy(frames, xy, FPS, CFG)
    assert teleports == [(14, 14)]
    # the jump displacement is preserved verbatim — marked, not erased (R-6)
    np.testing.assert_allclose(out[15] - out[14], xy[15] - xy[14], atol=1e-9)


def test_gate_clamps_segments_around_teleport_independently():
    rng = np.random.default_rng(11)
    frames = np.arange(40)
    xy = _walk(40, v=2.0) + rng.normal(0, 0.2, (40, 2))
    xy[20:] += [2.5, 0.0]
    out, teleports = gate_subject_xy(frames, xy, FPS, CFG)
    assert teleports == [(19, 19)]
    for seg in (slice(0, 20), slice(20, 40)):
        sp, ac = _viols(frames[seg], out[seg])
        assert sp == 0 and ac <= 1


def test_gate_collapses_consecutive_run_into_one_region():
    """A multi-interval slide (tracker sliding off, #207 subj-1 style) is ONE region."""
    rng = np.random.default_rng(21)
    frames = np.arange(40)
    xy = _walk(40, v=2.0) + rng.normal(0, 0.1, (40, 2))
    # 5 consecutive impossible intervals, decaying 42→24 m/s, all same direction
    for i, d in enumerate([1.7, 1.4, 1.2, 1.0, 0.95]):
        xy[16 + i :] += [d, 0.0]
    out, regions = gate_subject_xy(frames, xy, FPS, CFG)
    assert regions == [(15, 19)]
    # the whole slide region stays measured verbatim (R-6: no invented path)
    np.testing.assert_allclose(out[15:21], xy[15:21], atol=1e-9)
    for seg in (slice(0, 16), slice(20, 40)):
        sp, ac = _viols(frames[seg], out[seg])
        assert sp == 0 and ac <= 1


def test_gate_clamps_out_and_back_spike_as_noise():
    """A one-frame outlier row (out + straight back) is jitter, NOT a marked teleport."""
    frames = np.arange(30)
    xy = _walk(30, v=2.0)
    xy[10] += [0.0, 1.2]  # single-row spike: both adjacent intervals ~30 m/s, reversed
    out, teleports = gate_subject_xy(frames, xy, FPS, CFG)
    assert teleports == []
    sp, ac = _viols(frames, out)
    assert sp == 0 and ac <= 1


# --- kinematic_gate (scene level) -----------------------------------------------------


def test_gate_clean_scene_adds_nothing(make_scene):
    sub = Subject(track_id=1, proposal=_motion(np.arange(30), _walk(30)))
    scene = make_scene(subjects=[sub])
    out, report = kinematic_gate(scene, CFG, fps=FPS)
    assert report.corrections_added == 0
    assert report.accel_viol_before == 0
    assert out.corrections == []


def test_gate_emits_keyframe_correction_and_resolve_applies_it(make_scene):
    rng = np.random.default_rng(5)
    xy = _walk(50) + rng.normal(0, 0.25, (50, 2))
    sub = Subject(track_id=3, proposal=_motion(np.arange(50), xy))
    scene = make_scene(subjects=[sub])
    out, report = kinematic_gate(scene, CFG, fps=FPS)
    assert report.accel_viol_before > 0
    assert report.corrections_added == 1
    assert report.subjects_corrected == 1
    assert report.accel_viol_after <= 1 and report.speed_viol_after == 0
    (corr,) = out.corrections
    assert corr.mode == CorrectionMode.KEYFRAME_INTERP
    assert corr.target.kind == TargetKind.ROOT_TRANSLATION
    assert corr.target.subject_track_id == 3

    resolved = resolve_subject_motion(sub.proposal, out.corrections_for(3))
    sp, ac = _viols(resolved.pose.frames, resolved.pose.transl[:, :2])
    assert sp == 0 and ac <= 1
    # Z (body height) passes through untouched
    np.testing.assert_allclose(resolved.pose.transl[:, 2], 0.9, atol=1e-9)


def test_gate_never_mutates_input_scene(make_scene):
    rng = np.random.default_rng(5)
    xy = _walk(50) + rng.normal(0, 0.25, (50, 2))
    sub = Subject(track_id=3, proposal=_motion(np.arange(50), xy))
    scene = make_scene(subjects=[sub])
    before = sub.proposal.pose.transl.copy()
    kinematic_gate(scene, CFG, fps=FPS)
    np.testing.assert_array_equal(sub.proposal.pose.transl, before)
    assert scene.corrections == []


def test_gate_layers_on_existing_corrections_basis(make_scene):
    """The gate resolves through prior corrections (e.g. coherence smoothing) first."""
    from pitch3d.core.correction.engine import make_smoothing
    from pitch3d.core.scene.layers import CorrectionTarget

    rng = np.random.default_rng(9)
    xy = _walk(50) + rng.normal(0, 0.25, (50, 2))
    sub = Subject(track_id=4, proposal=_motion(np.arange(50), xy))
    smooth = make_smoothing(
        "auto-coh-transl-4",
        CorrectionTarget(TargetKind.ROOT_TRANSLATION, subject_track_id=4),
        (0, 49),
        window=5,
    )
    scene = make_scene(subjects=[sub], corrections=[smooth])
    out, report = kinematic_gate(scene, CFG, fps=FPS)
    resolved = resolve_subject_motion(sub.proposal, out.corrections_for(4))
    sp, ac = _viols(resolved.pose.frames, resolved.pose.transl[:, :2])
    assert sp == 0 and ac <= 1


def test_gate_reports_teleport_event(make_scene):
    xy = _walk(30, v=2.0)
    xy[15:] += [1.8, 0.0]
    sub = Subject(track_id=7, proposal=_motion(np.arange(30), xy))
    out, report = kinematic_gate(scene := make_scene(subjects=[sub]), CFG, fps=FPS)
    assert len(report.teleports) == 1
    tp = report.teleports[0]
    assert (tp.track_id, tp.frame) == (7, 15)
    assert tp.jump_m > 1.7 and tp.speed_mps > CFG.teleport_factor * CFG.max_speed
    assert tp.n_intervals == 1


def test_gate_reports_consecutive_run_as_one_event(make_scene):
    """#207 measured artifact: a coasting slide must be ONE event, not 5 teleports."""
    xy = _walk(40, v=2.0)
    for i, d in enumerate([1.7, 1.4, 1.2, 1.0, 0.95]):
        xy[16 + i :] += [d, 0.0]
    sub = Subject(track_id=1, proposal=_motion(np.arange(40), xy))
    _, report = kinematic_gate(make_scene(subjects=[sub]), CFG, fps=FPS)
    assert len(report.teleports) == 1
    tp = report.teleports[0]
    assert (tp.track_id, tp.frame, tp.n_intervals) == (1, 16, 5)
    assert tp.jump_m > 6.0
    # preserved region is excluded from "after" counts — marked, not counted as unfixed
    assert report.speed_viol_after == 0 and report.accel_viol_after == 0


def test_gate_corrections_survive_save_load(tmp_path, make_scene):
    """Persistence: the gated trajectory survives serialize → load → resolve."""
    rng = np.random.default_rng(13)
    xy = _walk(50) + rng.normal(0, 0.25, (50, 2))
    sub = Subject(track_id=5, proposal=_motion(np.arange(50), xy))
    out, _ = kinematic_gate(make_scene(subjects=[sub]), CFG, fps=FPS)
    path = tmp_path / "scene.json"
    save_scene(out, path)
    loaded = load_scene(path)
    resolved = resolve_subject_motion(
        loaded.subjects[0].proposal, loaded.corrections_for(5)
    )
    sp, ac = _viols(resolved.pose.frames, resolved.pose.transl[:, :2])
    assert sp == 0 and ac <= 1
