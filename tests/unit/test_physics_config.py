"""Physics config loader: precedence, lineage, dataclass wiring, error paths.

Every physics threshold must live in the YAML and only the YAML — otherwise the
"parametric stays parametric" rule breaks. These tests pin the loader semantics
so a silent regression (a stray Python default overriding the YAML) fails CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pitch3d.core.config import PhysicsConfig, load_physics_config
from pitch3d.core.correction.coherence import CoherenceConfig
from pitch3d.core.correction.kinematics import KinematicConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "physics.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return p


def _minimal(**profiles) -> dict:
    """A minimal YAML tree using the current defaults with the given profiles added."""
    return {
        "version": 1,
        "base": {
            "kinematic": {
                "max_speed": 10.5, "max_accel": 8.0, "teleport_factor": 2.0,
                "spike_neighbor_frac": 0.5, "spike_reversal_cos": -0.5,
                "max_passes": 50, "min_correction_m": 1e-6,
                "teleport_policy": "hold",
            },
            "coherence": {
                "max_fill_gap": 12, "smooth_window": 5, "smooth_method": "moving_average",
                "smooth_sigma": 1.0, "smooth_root_translation": True,
                "smooth_root_orientation": False, "filled_confidence": 0.3,
                "real_confidence": 1.0, "extend_to_span": True,
                "extrapolate_decay": 0.9, "extrapolated_confidence": 0.2,
                "extrapolate_velocity_window": 3, "coast_max_speed": 10.5,
            },
            "foot_floor": {"enabled": False, "floor_m": 0.0, "warn_hover_m": 0.30},
            "foot_plant": {"enabled": False, "mode": "median_lock",
                           "target_pelvis_m": 0.92, "bias_threshold_m": 0.05,
                           "min_correction_m": 1e-4},
            "joint":      {"enabled": False, "max_omega_dps": 600.0},
            "orientation": {"enabled": False, "max_turn_rate_dps": 720.0},
            "ball":       {"max_speed": 36.0, "max_accel": 200.0},
            "probe":      {"turn_min_speed": 2.0, "joint_min_omega_dps": 600.0,
                           "orient_min_dps": 720.0, "foot_hover_m": 0.30},
        },
        "profiles": {"default": {"description": "d", "overrides": {}}, **profiles},
    }


def test_default_profile_returns_current_kinematic_config_ceilings(tmp_path):
    cfg = load_physics_config(_write(tmp_path, _minimal()), env={})
    assert isinstance(cfg, PhysicsConfig)
    assert isinstance(cfg.kinematic, KinematicConfig)
    assert isinstance(cfg.coherence, CoherenceConfig)
    assert cfg.kinematic.max_speed == 10.5
    assert cfg.kinematic.max_accel == 8.0
    assert cfg.coherence.coast_max_speed == 10.5
    assert cfg.profile_name == "default"
    # Every leaf field records its provenance
    assert cfg.lineage["kinematic.max_speed"] == "base"
    assert cfg.lineage["coherence.smooth_window"] == "base"


def test_profile_override_wins_over_base(tmp_path):
    y = _minimal(
        conservative={
            "description": "c",
            "overrides": {"kinematic": {"max_speed": 9.0, "max_accel": 6.0}},
        }
    )
    cfg = load_physics_config(_write(tmp_path, y), profile="conservative", env={})
    assert cfg.kinematic.max_speed == 9.0
    assert cfg.kinematic.max_accel == 6.0
    # non-overridden fields still come from base
    assert cfg.kinematic.teleport_factor == 2.0
    assert cfg.lineage["kinematic.max_speed"] == "profile:conservative"
    assert cfg.lineage["kinematic.teleport_factor"] == "base"


def test_env_var_wins_over_profile(tmp_path):
    y = _minimal(
        conservative={"description": "c", "overrides": {"kinematic": {"max_speed": 9.0}}}
    )
    cfg = load_physics_config(
        _write(tmp_path, y), profile="conservative",
        env={"PITCH3D_KIN_MAX_SPEED": "9.5"},
    )
    assert cfg.kinematic.max_speed == 9.5
    assert cfg.lineage["kinematic.max_speed"] == "env:PITCH3D_KIN_MAX_SPEED"


def test_python_override_wins_over_env(tmp_path):
    cfg = load_physics_config(
        _write(tmp_path, _minimal()),
        env={"PITCH3D_KIN_MAX_SPEED": "9.5"},
        overrides={"kinematic": {"max_speed": 7.0}},
    )
    assert cfg.kinematic.max_speed == 7.0
    assert cfg.lineage["kinematic.max_speed"] == "override:kinematic.max_speed"


def test_unknown_profile_raises(tmp_path):
    with pytest.raises(KeyError, match="nope"):
        load_physics_config(_write(tmp_path, _minimal()), profile="nope", env={})


def test_unknown_field_in_section_raises_clearly(tmp_path):
    y = _minimal()
    y["base"]["kinematic"]["mystery_knob"] = 42
    with pytest.raises(ValueError, match="mystery_knob"):
        load_physics_config(_write(tmp_path, y), env={})


def test_env_var_cast_error_is_actionable(tmp_path):
    with pytest.raises(ValueError, match="not_a_number"):
        load_physics_config(
            _write(tmp_path, _minimal()),
            env={"PITCH3D_KIN_MAX_SPEED": "not_a_number"},
        )


def test_env_bool_recognizes_yes_true_1_on(tmp_path):
    for truthy in ("1", "true", "TRUE", "yes", "on"):
        cfg = load_physics_config(
            _write(tmp_path, _minimal()),
            env={"PITCH3D_FOOT_FLOOR_ON": truthy},
        )
        assert cfg.foot_floor.enabled is True, truthy
    cfg = load_physics_config(_write(tmp_path, _minimal()), env={"PITCH3D_FOOT_FLOOR_ON": "0"})
    assert cfg.foot_floor.enabled is False


def test_repo_config_loads_and_all_named_profiles_build(tmp_path):
    """The shipped config/physics.yaml must load and expose every named profile."""
    from pitch3d.core.config.physics import DEFAULT_CONFIG_PATH
    assert DEFAULT_CONFIG_PATH.exists(), DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    for profile in raw["profiles"]:
        cfg = load_physics_config(profile=profile, env={})
        # sanity: gate constructors accept it
        assert cfg.kinematic.max_speed > 0
        assert cfg.coherence.smooth_window >= 1
        assert cfg.profile_name == profile


def test_lineage_covers_every_leaf(tmp_path):
    cfg = load_physics_config(_write(tmp_path, _minimal()), env={})
    # For every field of every sub-config, lineage has an entry
    for section, dc in (
        ("kinematic", cfg.kinematic), ("coherence", cfg.coherence),
        ("foot_floor", cfg.foot_floor), ("foot_plant", cfg.foot_plant),
        ("joint", cfg.joint), ("orientation", cfg.orientation),
        ("ball", cfg.ball), ("probe", cfg.probe),
    ):
        for f in dc.__dataclass_fields__:
            key = f"{section}.{f}"
            assert key in cfg.lineage, f"lineage missing {key}"


def test_summary_is_deterministic_and_names_profile(tmp_path):
    cfg = load_physics_config(_write(tmp_path, _minimal()), env={})
    s = cfg.summary()
    assert "profile='default'" in s
    assert "max_speed=10.5" in s
