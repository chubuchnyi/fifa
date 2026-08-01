"""Per-player profile schema + auto-tune update policy (T4.a).

These tests pin the seven-layer filter from ``docs/research/2026-07-06-...`` §4.4.
If a silent regression makes auto-tune accept jitter, the ceiling drifts up and
the whole M3-9 gate is defanged.
"""

from __future__ import annotations

from pathlib import Path

from pitch3d.core.scene.player_profile import (
    AutoTunePolicy,
    Position,
    ProfileField,
    ProfileSource,
    UpdateOutcome,
    default_ball_profile,
    default_player_profile,
    load_priors,
    set_operator_field,
    update_field,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ─── priors + defaults ───────────────────────────────────────────────────────

def test_priors_load_from_repo_yaml():
    priors = load_priors()
    assert Position.GK in priors.positions
    assert Position.FWD in priors.positions
    assert priors.policy.min_promote_n >= 1
    assert priors.body_height_m > 1.0


def test_default_player_uses_position_priors():
    priors = load_priors()
    gk = default_player_profile("COL", 1, Position.GK, priors=priors)
    fwd = default_player_profile("COL", 9, Position.FWD, priors=priors)
    assert gk.position is Position.GK
    assert fwd.position is Position.FWD
    # FWD has a higher peak sprint than GK by construction
    assert (fwd.kinematics["peak_speed_mps"].value
            > gk.kinematics["peak_speed_mps"].value)
    # every seeded field is DEFAULT source with n=0
    for f in gk.kinematics.values():
        assert f.source is ProfileSource.DEFAULT
        assert f.n == 0
        assert f.ci is None


def test_default_ball_profile_has_kinematics_and_physics():
    ball = default_ball_profile("match_ball_1")
    assert "peak_speed_mps" in ball.kinematics
    assert "restitution" in ball.physics
    assert "radius_m" in ball.appearance


# ─── auto-tune 7-layer filter ────────────────────────────────────────────────

def _policy(**over) -> AutoTunePolicy:
    """Small-N policy so tests can promote after a handful of samples."""
    base = dict(min_promote_n=5, ewma_time_constant=50.0,
                quarantine_ci_mult=3.0, ceiling_floor_mult=0.90,
                min_confidence=0.5)
    base.update(over)
    return AutoTunePolicy(**base)


def test_operator_locked_field_is_immutable():
    """Layer 7 — operator override wins over every subsequent observation."""
    f0 = ProfileField(value=9.0, source=ProfileSource.OPERATOR, ci=0.0, n=100)
    f1, outcome = update_field(f0, observation=15.0, policy=_policy())
    assert outcome is UpdateOutcome.OPERATOR_LOCKED
    assert f1 == f0


def test_low_confidence_frame_is_ignored():
    """Layer 3 — subject_frame_conf below the threshold contributes nothing."""
    f0 = ProfileField(value=9.0, source=ProfileSource.DEFAULT, ci=None, n=0)
    f1, outcome = update_field(f0, observation=12.0, confidence=0.1, policy=_policy())
    assert outcome is UpdateOutcome.LOW_CONFIDENCE
    assert f1 == f0


def test_quarantine_rejects_wild_outlier():
    """Layer 5 — an observation outside CI×mult is dropped (probable ID swap)."""
    # existing running mean of 9.0 with a small ci=0.5, plenty of samples
    f0 = ProfileField(value=9.0, source=ProfileSource.MEASURED, ci=0.5, n=100)
    f1, outcome = update_field(f0, observation=50.0, policy=_policy())
    assert outcome is UpdateOutcome.QUARANTINED
    assert f1 == f0


def test_default_field_promotes_to_measured_after_min_n():
    """Layer 4 — the first ``min_promote_n`` samples average in, then promote."""
    policy = _policy(min_promote_n=5)
    f = ProfileField(value=10.0, source=ProfileSource.DEFAULT, ci=None, n=0)
    outcomes = []
    for obs in (9.0, 9.5, 10.0, 9.8, 10.2):
        f, outcome = update_field(f, obs, policy=policy)
        outcomes.append(outcome)
    assert all(o is UpdateOutcome.APPLIED for o in outcomes)
    assert f.source is ProfileSource.MEASURED  # promoted on the 5th sample
    assert f.n == 5


def test_ceiling_floor_prevents_collapse():
    """Layer 6 — a player who never sprinted must not lock the ceiling to 5 m/s."""
    policy = _policy(min_promote_n=3, ceiling_floor_mult=0.90)
    default_v = 10.0
    f = ProfileField(value=default_v, source=ProfileSource.DEFAULT, ci=None, n=0)
    for obs in (5.0, 5.0, 5.0, 5.0):
        f, _ = update_field(f, obs, policy=policy, default_value=default_v)
    # floor at 0.90 × 10.0 = 9.0; value must never drop below
    assert f.value >= 0.9 * default_v


def test_ewma_reacts_slowly_after_promotion():
    """After promotion, a new outlier (still inside CI) shifts the mean SLIGHTLY."""
    policy = _policy(min_promote_n=3, ewma_time_constant=100.0, quarantine_ci_mult=5.0)
    f = ProfileField(value=9.0, source=ProfileSource.MEASURED, ci=0.5, n=100)
    f1, outcome = update_field(f, 11.0, policy=policy)
    assert outcome is UpdateOutcome.APPLIED
    assert 9.0 < f1.value < 9.05, f1.value  # ~ alpha·(11-9) = 0.02


def test_set_operator_field_marks_immutable():
    f = ProfileField(value=9.0, source=ProfileSource.MEASURED, ci=0.5, n=100)
    f2 = set_operator_field(f, 8.0)
    assert f2.source is ProfileSource.OPERATOR
    assert f2.value == 8.0
    f3, outcome = update_field(f2, 12.0, policy=_policy())
    assert outcome is UpdateOutcome.OPERATOR_LOCKED
    assert f3 == f2


def test_ci_grows_from_none_to_a_number_after_second_sample():
    policy = _policy(min_promote_n=5)
    f = ProfileField(value=9.0, source=ProfileSource.DEFAULT, ci=None, n=0)
    f1, _ = update_field(f, 9.4, policy=policy)
    assert f1.ci is not None
    assert f1.ci > 0.0


def test_confidence_weight_shifts_less_than_full_weight():
    """Confidence=0.5 must update the mean less aggressively than confidence=1.0."""
    policy = _policy(min_promote_n=3, ewma_time_constant=100.0, quarantine_ci_mult=5.0)
    f = ProfileField(value=9.0, source=ProfileSource.MEASURED, ci=0.5, n=100)
    hi, _ = update_field(f, 11.0, confidence=1.0, policy=policy)
    lo, _ = update_field(f, 11.0, confidence=0.6, policy=policy)
    assert (hi.value - f.value) > (lo.value - f.value)
