"""M3-9 gate ↔ player profile wiring (T4.b): per-subject ceilings + auto-tune proposals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pitch3d.adapters.profiles import LocalJsonPlayerStore
from pitch3d.core.correction.engine import resolve_subject_motion
from pitch3d.core.correction.kinematics import KinematicConfig, kinematic_gate
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.player_profile import (
    Position,
    ProfileField,
    ProfileSource,
    ProfileUpdateProposal,
    UpdateOutcome,
    apply_profile_updates,
    default_player_profile,
    load_priors,
    set_operator_field,
)
from pitch3d.core.scene.scene import Scene, Subject

REPO_ROOT = Path(__file__).resolve().parents[2]


def _subject(track_id: int, xy: np.ndarray, team_id: int = 1,
             jersey: int = 10) -> Subject:
    T = xy.shape[0]
    frames = np.arange(T, dtype=int)
    transl = np.zeros((T, 3))
    transl[:, :2] = xy
    motion = SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(
            frames=frames, global_orient=np.zeros((T, 3)),
            body_pose=np.zeros((T, 21, 3)), transl=transl,
        ),
    )
    return Subject(
        track_id=track_id, proposal=motion,
        team_id=team_id, jersey_number=jersey,
    )


def _scene(*subjects: Subject) -> Scene:
    return Scene(
        id="s", episode_id="e", source_id="c",
        subjects=list(subjects), corrections=[],
    )


def _first_correction_note(scene: Scene) -> str:
    for c in scene.corrections:
        if c.note:
            return c.note
    return ""


def _scene_with_report(s: Subject, provider, fps: float) -> Scene:
    new_scene, _ = kinematic_gate(
        _scene(s), KinematicConfig(), fps=fps, profile_provider=provider,
    )
    return new_scene


def test_gate_without_provider_matches_prior_behaviour():
    """Backwards compat: no profile_provider → gate runs exactly as before."""
    fps = 30.0
    T = 10
    xy = np.zeros((T, 2))
    # step every OTHER frame to trigger accel violations without teleport-scale speed
    xy[:, 0] = [0.0, 0.5, 0.4, 0.9, 0.8, 1.3, 1.2, 1.7, 1.6, 2.1]
    s = _subject(1, xy)
    _, report = kinematic_gate(_scene(s), KinematicConfig(), fps=fps)
    assert report.subjects_using_profile == []
    assert report.profile_updates == []
    assert report.accel_viol_before > 0
    assert report.subjects_corrected >= 1


def test_provider_swaps_max_speed_per_subject():
    """When the profile is present its peak_speed_mps overrides the shared cfg."""
    fps = 30.0
    T = 10
    xy = np.zeros((T, 2))
    # non-linear motion with a spike that fires the accel clamp under the
    # tight profile (5 m/s²) but not under the shared default (8 m/s²).
    xy[:, 0] = [0.0, 0.2, 0.4, 0.6, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1]
    s = _subject(1, xy)
    priors = load_priors()
    tight = default_player_profile("A", 10, Position.GK, priors=priors)
    tight_kin = dict(tight.kinematics)
    tight_kin["peak_speed_mps"] = ProfileField(
        value=8.0, source=ProfileSource.MEASURED, ci=0.5, n=200,
    )
    tight_kin["peak_accel_mps2"] = ProfileField(
        value=5.0, source=ProfileSource.MEASURED, ci=0.4, n=200,
    )
    from dataclasses import replace as _replace
    tight = _replace(tight, kinematics=tight_kin)

    def provider(subject):
        return tight if subject.track_id == 1 else None

    # Gate with the tight profile: provider called, subjects_using_profile
    # populated, profile_updates emitted (regardless of whether more clamping
    # happens — the wiring is what we're pinning here).
    _, prof_report = kinematic_gate(
        _scene(s), KinematicConfig(), fps=fps, profile_provider=provider,
    )
    assert prof_report.subjects_using_profile == [1]
    assert prof_report.profile_updates            # observations for the auto-tuner
    keys = {u.field_key for u in prof_report.profile_updates}
    assert keys == {"peak_speed_mps", "peak_accel_mps2"}
    # The correction note reflects the per-subject ceiling (8 m/s, not 10.5)
    # — check the emitted correction records the profile limit.
    if prof_report.subjects_corrected:
        assert "profile" in _first_correction_note(_scene_with_report(s, provider, fps))


def test_gate_emits_profile_update_proposals_from_clamped_motion():
    """Auto-tune observation is a p95 of the post-clamp speed/accel (§4.4 layer 1).

    Note: when the endpoints demand a fast average speed the M3-9 endpoint
    anchor restores it, so "post-clamp" is not necessarily below max_speed.
    That's the honest behavior — the tuner learns from what the resolved motion
    actually is, not from a fabricated ceiling.
    """
    fps = 30.0
    T = 10
    xy = np.zeros((T, 2))
    # small back-and-forth: real accel violations, no teleport
    xy[:, 0] = [0.0, 0.15, 0.10, 0.25, 0.20, 0.35, 0.30, 0.45, 0.40, 0.55]
    s = _subject(1, xy)
    priors = load_priors()
    profile = default_player_profile("A", 10, Position.MID, priors=priors)

    def provider(subject):
        return profile

    _, report = kinematic_gate(
        _scene(s), KinematicConfig(), fps=fps, profile_provider=provider,
    )
    # gate proposed updates for BOTH kinematic fields
    keys = sorted({u.field_key for u in report.profile_updates})
    assert keys == ["peak_accel_mps2", "peak_speed_mps"]
    for u in report.profile_updates:
        assert u.track_id == 1
        assert u.domain == "player"
        assert u.observation > 0.0
        assert u.default_value is not None
        assert u.confidence == 1.0
    # p95 speed on this back-and-forth motion is well below the clamp
    speed_prop = next(u for u in report.profile_updates
                      if u.field_key == "peak_speed_mps")
    assert speed_prop.observation < 10.5


def test_apply_profile_updates_persists_measured_speed(tmp_path: Path):
    """Consumer feeds proposals through update_field; profile is saved."""
    store = LocalJsonPlayerStore(tmp_path)
    priors = load_priors()
    lookup = {1: ("A", 10, Position.MID)}

    # Feed enough clean samples to promote the field
    updates = [
        ProfileUpdateProposal(
            track_id=1, domain="player", field_key="peak_speed_mps",
            observation=9.4, confidence=1.0, default_value=9.5,
        )
    ] * priors.policy.min_promote_n
    outcomes = apply_profile_updates(store, priors, lookup, updates)
    assert outcomes["applied"] == priors.policy.min_promote_n
    saved = store.load_player("A", 10)
    assert saved is not None
    field = saved.kinematics["peak_speed_mps"]
    assert field.source is ProfileSource.MEASURED
    # value settled near the observation and above the floor (0.90 * 9.5 = 8.55)
    assert 8.55 <= field.value <= 9.5 + 1e-6


def test_apply_profile_updates_respects_operator_lock(tmp_path: Path):
    """A field the operator set stays unchanged no matter how many updates arrive."""
    store = LocalJsonPlayerStore(tmp_path)
    priors = load_priors()
    profile = default_player_profile("A", 10, Position.MID, priors=priors)
    from dataclasses import replace as _replace
    locked_kin = dict(profile.kinematics)
    locked_kin["peak_speed_mps"] = set_operator_field(
        locked_kin["peak_speed_mps"], value=8.5,
    )
    profile = _replace(profile, kinematics=locked_kin)
    store.save_player(profile)

    updates = [
        ProfileUpdateProposal(
            track_id=1, domain="player", field_key="peak_speed_mps",
            observation=11.0, confidence=1.0, default_value=9.5,
        )
    ] * 40
    outcomes = apply_profile_updates(
        store, priors, {1: ("A", 10, Position.MID)}, updates,
    )
    assert outcomes["operator_locked"] == 40
    saved = store.load_player("A", 10)
    assert saved.kinematics["peak_speed_mps"].source is ProfileSource.OPERATOR
    assert saved.kinematics["peak_speed_mps"].value == 8.5


def test_apply_profile_updates_creates_default_when_missing(tmp_path: Path):
    """First observation for an unseen player seeds a fresh profile from priors."""
    store = LocalJsonPlayerStore(tmp_path)
    priors = load_priors()
    assert store.load_player("A", 10) is None
    updates = [
        ProfileUpdateProposal(
            track_id=1, domain="player", field_key="peak_speed_mps",
            observation=9.0, confidence=1.0, default_value=9.5,
        ),
    ]
    outcomes = apply_profile_updates(
        store, priors, {1: ("A", 10, Position.FWD)}, updates,
    )
    assert outcomes["applied"] == 1
    saved = store.load_player("A", 10)
    assert saved is not None
    assert saved.position is Position.FWD


def test_apply_profile_updates_skips_unknown_track(tmp_path: Path):
    """Proposals for unknown track_ids are counted as skipped, never crash."""
    store = LocalJsonPlayerStore(tmp_path)
    priors = load_priors()
    updates = [ProfileUpdateProposal(
        track_id=99, domain="player", field_key="peak_speed_mps",
        observation=9.0, confidence=1.0,
    )]
    outcomes = apply_profile_updates(store, priors, {}, updates)
    assert outcomes["skipped"] == 1


def test_apply_profile_updates_skips_unknown_field_key(tmp_path: Path):
    """A proposal for a field not in the profile is skipped, never invented."""
    store = LocalJsonPlayerStore(tmp_path)
    priors = load_priors()
    updates = [ProfileUpdateProposal(
        track_id=1, domain="player", field_key="mystery_knob",
        observation=1.0, confidence=1.0,
    )]
    outcomes = apply_profile_updates(
        store, priors, {1: ("A", 10, Position.MID)}, updates,
    )
    assert outcomes["skipped"] == 1
